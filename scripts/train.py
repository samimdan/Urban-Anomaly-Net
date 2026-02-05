"""
Training script for Urban Anomaly Detection Network
Trains the spatio-temporal autoencoder on normal traffic videos
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import SpatioTemporalAutoencoder
from utils.data_utils import create_data_loaders
from utils.visualization import plot_training_curves, visualize_reconstruction


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Urban Anomaly Detection Network'
    )
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/videos',
                       help='Directory containing training videos')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data for testing')
    parser.add_argument('--sequence-length', type=int, default=16,
                       help='Number of frames per sequence')
    parser.add_argument('--frame-size', type=int, default=128,
                       help='Frame size (will be frame-size x frame-size)')
    
    # Model parameters
    parser.add_argument('--hidden-channels', type=int, nargs='+',
                       default=[64, 128, 256],
                       help='Hidden channels for encoder/decoder layers')
    parser.add_argument('--kernel-size', type=int, default=3,
                       help='Convolutional kernel size')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--save-interval', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, videos in enumerate(progress_bar):
        videos = videos.to(device)
        
        # Forward pass
        reconstructed = model(videos)
        
        # Compute loss
        loss = criterion(reconstructed, videos)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for videos in val_loader:
            videos = videos.to(device)
            
            # Forward pass
            reconstructed = model(videos)
            
            # Compute loss
            loss = criterion(reconstructed, videos)
            
            total_loss += loss.item()
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    frame_size = (args.frame_size, args.frame_size)
    
    train_loader, val_loader = create_data_loaders(
        video_dir=args.data_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        frame_size=frame_size,
        num_workers=args.num_workers,
        use_synthetic=args.use_synthetic
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = SpatioTemporalAutoencoder(
        input_channels=3,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Tensorboard writer
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Visualize reconstruction
        if epoch % args.save_interval == 0:
            model.eval()
            with torch.no_grad():
                sample_videos = next(iter(val_loader)).to(device)
                sample_reconstructed = model(sample_videos)
                
                # Save visualization
                vis_path = os.path.join(args.log_dir, f'reconstruction_epoch_{epoch}.png')
                visualize_reconstruction(
                    sample_videos[0], sample_reconstructed[0],
                    num_frames=8, save_path=vis_path
                )
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, args.epochs, val_losses[-1], final_path)
    
    # Plot and save training curves
    curves_path = os.path.join(args.log_dir, 'training_curves.png')
    plot_training_curves(train_losses, val_losses, save_path=curves_path)
    
    writer.close()
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
