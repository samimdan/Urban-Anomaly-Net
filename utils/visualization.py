"""
Visualization utilities for results and analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2


def visualize_reconstruction(original, reconstructed, num_frames=8, save_path=None):
    """
    Visualize original vs reconstructed frames
    
    Args:
        original: Original video tensor (T, C, H, W) or (B, T, C, H, W)
        reconstructed: Reconstructed video tensor
        num_frames: Number of frames to display
        save_path: Path to save figure (optional)
    """
    # Handle batch dimension
    if len(original.shape) == 5:
        original = original[0]
        reconstructed = reconstructed[0]
    
    # Convert to numpy and transpose to (T, H, W, C)
    if isinstance(original, torch.Tensor):
        original = original.cpu().detach().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().detach().numpy()
    
    original = np.transpose(original, (0, 2, 3, 1))
    reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))
    
    # Clip values to [0, 1]
    original = np.clip(original, 0, 1)
    reconstructed = np.clip(reconstructed, 0, 1)
    
    # Select frames to display
    T = original.shape[0]
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 2, 4))
    
    for i, frame_idx in enumerate(frame_indices):
        # Original
        axes[0, i].imshow(original[frame_idx])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[frame_idx])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def visualize_anomaly_detection(frames, anomaly_map, is_anomaly, save_path=None):
    """
    Visualize anomaly detection results with heatmap
    
    Args:
        frames: Video frames (T, C, H, W)
        anomaly_map: Spatial anomaly map (H, W) or (T, H, W)
        is_anomaly: Boolean indicating if sequence is anomalous
        save_path: Path to save figure (optional)
    """
    # Convert to numpy
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().detach().numpy()
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.cpu().detach().numpy()
    
    # Transpose frames to (T, H, W, C)
    frames = np.transpose(frames, (0, 2, 3, 1))
    frames = np.clip(frames, 0, 1)
    
    # Handle different anomaly map shapes
    if len(anomaly_map.shape) == 2:
        # Single spatial map
        anomaly_map = np.expand_dims(anomaly_map, 0)
    
    # Select frames to display
    num_frames = min(4, frames.shape[0])
    frame_indices = np.linspace(0, frames.shape[0] - 1, num_frames, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    
    if num_frames == 1:
        axes = axes.reshape(2, 1)
    
    for i, frame_idx in enumerate(frame_indices):
        # Original frame
        axes[0, i].imshow(frames[frame_idx])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Frame', fontsize=10)
        
        # Anomaly heatmap overlay
        map_idx = min(frame_idx, anomaly_map.shape[0] - 1)
        
        # Create overlay
        heatmap = cv2.applyColorMap(
            (anomaly_map[map_idx] * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Blend with original
        overlay = 0.6 * frames[frame_idx] + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        axes[1, i].imshow(overlay)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Anomaly Heatmap', fontsize=10)
    
    # Add overall title
    status = "ANOMALY DETECTED" if is_anomaly else "NORMAL"
    color = "red" if is_anomaly else "green"
    fig.suptitle(status, fontsize=14, fontweight='bold', color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def plot_training_curves(train_losses, val_losses=None, save_path=None):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def create_video_from_frames(frames, output_path, fps=30):
    """
    Create video file from frames
    
    Args:
        frames: Array of frames (T, H, W, C)
        output_path: Path to save video
        fps: Frames per second
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().detach().numpy()
    
    # Ensure frames are in correct format
    if frames.shape[-1] != 3:
        frames = np.transpose(frames, (0, 2, 3, 1))
    
    # Convert to uint8
    frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    
    # Get dimensions
    height, width = frames.shape[1:3]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"Saved video to {output_path}")
