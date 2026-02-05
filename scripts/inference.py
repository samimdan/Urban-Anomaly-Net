"""
Inference script for Urban Anomaly Detection
Detects anomalies in traffic videos using trained model
"""

import torch
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import SpatioTemporalAutoencoder
from models.anomaly_detector import AnomalyDetector
from utils.data_utils import create_data_loaders, extract_frames, normalize_frames
from utils.visualization import visualize_anomaly_detection, visualize_reconstruction


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Detect anomalies in traffic videos'
    )
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--hidden-channels', type=int, nargs='+',
                       default=[64, 128, 256],
                       help='Hidden channels (must match training)')
    parser.add_argument('--kernel-size', type=int, default=3,
                       help='Kernel size (must match training)')
    
    # Data parameters
    parser.add_argument('--video-dir', type=str,
                       help='Directory containing test videos')
    parser.add_argument('--video-file', type=str,
                       help='Single video file to process')
    parser.add_argument('--normal-data-dir', type=str,
                       help='Directory with normal videos for threshold fitting')
    parser.add_argument('--use-synthetic', action='store_true',
                       help='Use synthetic data')
    parser.add_argument('--sequence-length', type=int, default=16,
                       help='Number of frames per sequence')
    parser.add_argument('--frame-size', type=int, default=128,
                       help='Frame size')
    
    # Detection parameters
    parser.add_argument('--threshold-percentile', type=float, default=95,
                       help='Percentile for anomaly threshold')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for inference')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Directory to save results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers')
    
    return parser.parse_args()


def load_model(checkpoint_path, hidden_channels, kernel_size, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = SpatioTemporalAutoencoder(
        input_channels=3,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    return model


def fit_threshold(detector, normal_data_dir, args, device):
    """Fit anomaly threshold on normal data"""
    print("Fitting anomaly threshold on normal data...")
    
    frame_size = (args.frame_size, args.frame_size)
    
    # Create data loader for normal data
    if args.use_synthetic:
        from utils.data_utils import SyntheticTrafficDataset
        from torch.utils.data import DataLoader
        
        normal_dataset = SyntheticTrafficDataset(
            num_samples=50,
            sequence_length=args.sequence_length,
            frame_size=frame_size
        )
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    else:
        from utils.data_utils import VideoDataset
        from torch.utils.data import DataLoader
        
        normal_dataset = VideoDataset(
            video_dir=normal_data_dir,
            sequence_length=args.sequence_length,
            frame_size=frame_size,
            stride=args.sequence_length
        )
        normal_loader = DataLoader(
            normal_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Fit threshold
    threshold = detector.fit_threshold(normal_loader, device=device)
    
    return threshold


def process_video_file(video_path, model, detector, args, device):
    """Process a single video file"""
    print(f"\nProcessing video: {video_path}")
    
    # Extract frames
    frame_size = (args.frame_size, args.frame_size)
    frames = extract_frames(video_path, frame_size=frame_size)
    
    if len(frames) < args.sequence_length:
        print(f"Warning: Video has only {len(frames)} frames, need at least {args.sequence_length}")
        return None
    
    # Create sequences
    sequences = []
    for i in range(0, len(frames) - args.sequence_length + 1, args.sequence_length // 2):
        seq = frames[i:i + args.sequence_length]
        seq = normalize_frames(seq)
        seq = torch.from_numpy(seq).permute(0, 3, 1, 2).float()
        sequences.append(seq)
    
    # Detect anomalies
    results = []
    for seq_idx, seq in enumerate(sequences):
        seq_batch = seq.unsqueeze(0)  # Add batch dimension
        
        is_anomaly, error = detector.detect_anomalies(
            seq_batch, device=device, return_errors=True
        )
        
        is_anomaly = is_anomaly[0].item()
        error = error[0].item()
        
        results.append({
            'sequence_idx': seq_idx,
            'is_anomaly': is_anomaly,
            'error': error,
            'frames': seq
        })
    
    # Summary
    num_anomalies = sum(1 for r in results if r['is_anomaly'])
    print(f"  Processed {len(results)} sequences")
    print(f"  Anomalies detected: {num_anomalies}/{len(results)} "
          f"({100 * num_anomalies / len(results):.1f}%)")
    
    return results


def main():
    """Main inference function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.checkpoint,
        args.hidden_channels,
        args.kernel_size,
        device
    )
    
    # Create anomaly detector
    detector = AnomalyDetector(
        model,
        threshold_percentile=args.threshold_percentile
    )
    
    # Fit threshold on normal data
    if args.normal_data_dir or args.use_synthetic:
        normal_dir = args.normal_data_dir if args.normal_data_dir else None
        fit_threshold(detector, normal_dir, args, device)
    else:
        print("Warning: No normal data provided. Using default threshold.")
        detector.threshold = 0.01  # Default threshold
    
    # Process videos
    if args.video_file:
        # Process single video
        results = process_video_file(
            args.video_file, model, detector, args, device
        )
        
        # Visualize results
        if args.visualize and results:
            for i, result in enumerate(results[:5]):  # Visualize first 5 sequences
                if result['is_anomaly']:
                    seq = result['frames'].unsqueeze(0).to(device)
                    anomaly_map = detector.get_spatial_anomaly_map(seq, device=device)
                    
                    vis_path = os.path.join(
                        args.output_dir,
                        f'anomaly_{Path(args.video_file).stem}_seq{i}.png'
                    )
                    visualize_anomaly_detection(
                        result['frames'],
                        anomaly_map[0],
                        True,
                        save_path=vis_path
                    )
    
    elif args.video_dir:
        # Process directory of videos
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(Path(args.video_dir).glob(f'**/*{ext}'))
        
        print(f"Found {len(video_files)} videos to process")
        
        all_results = []
        for video_file in tqdm(video_files, desc='Processing videos'):
            results = process_video_file(
                str(video_file), model, detector, args, device
            )
            if results:
                all_results.append({
                    'video_path': str(video_file),
                    'results': results
                })
        
        # Summary
        total_sequences = sum(len(r['results']) for r in all_results)
        total_anomalies = sum(
            sum(1 for seq in r['results'] if seq['is_anomaly'])
            for r in all_results
        )
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total videos processed: {len(all_results)}")
        print(f"Total sequences: {total_sequences}")
        print(f"Total anomalies: {total_anomalies} "
              f"({100 * total_anomalies / total_sequences:.1f}%)")
        print(f"{'='*60}")
    
    elif args.use_synthetic:
        # Process synthetic data
        print("Processing synthetic test data...")
        frame_size = (args.frame_size, args.frame_size)
        
        from utils.data_utils import SyntheticTrafficDataset
        from torch.utils.data import DataLoader
        
        test_dataset = SyntheticTrafficDataset(
            num_samples=20,
            sequence_length=args.sequence_length,
            frame_size=frame_size
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        anomalies = []
        errors = []
        
        for batch_idx, videos in enumerate(test_loader):
            is_anomaly, error = detector.detect_anomalies(
                videos, device=device, return_errors=True
            )
            
            anomalies.extend(is_anomaly.numpy())
            errors.extend(error.numpy())
        
        num_anomalies = sum(anomalies)
        print(f"\nProcessed {len(anomalies)} sequences")
        print(f"Anomalies detected: {num_anomalies}/{len(anomalies)} "
              f"({100 * num_anomalies / len(anomalies):.1f}%)")
    
    else:
        print("Error: Must provide --video-file, --video-dir, or --use-synthetic")
        return
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
