"""
Video preprocessing utilities for extracting and processing frames
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os


def extract_frames(video_path, max_frames=None, frame_size=(224, 224)):
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None for all)
        frame_size: Target size for frames (width, height)
    
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, frame_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frames.append(frame)
        frame_count += 1
        
        if max_frames and frame_count >= max_frames:
            break
    
    cap.release()
    
    return np.array(frames)


def compute_optical_flow(frames, method='farneback'):
    """
    Compute optical flow between consecutive frames
    
    Args:
        frames: Array of frames (T, H, W, C)
        method: Optical flow method ('farneback' or 'lucaskanade')
    
    Returns:
        Optical flow fields (T-1, H, W, 2)
    """
    flows = []
    
    for i in range(len(frames) - 1):
        # Convert to grayscale
        prev_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        
        if method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
        elif method == 'lucaskanade':
            # For Lucas-Kanade, we'll use a dense grid
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unknown optical flow method: {method}")
        
        flows.append(flow)
    
    return np.array(flows)


def normalize_frames(frames, mean=None, std=None):
    """
    Normalize frames to [0, 1] range or using mean/std
    
    Args:
        frames: Array of frames
        mean: Mean for normalization (optional)
        std: Standard deviation for normalization (optional)
    
    Returns:
        Normalized frames
    """
    frames = frames.astype(np.float32) / 255.0
    
    if mean is not None and std is not None:
        frames = (frames - mean) / std
    
    return frames


class VideoDataset(Dataset):
    """
    Dataset for loading video sequences
    """
    
    def __init__(self, video_dir, sequence_length=16, frame_size=(224, 224),
                 stride=8, transform=None, max_videos=None):
        """
        Args:
            video_dir: Directory containing video files
            sequence_length: Number of frames per sequence
            frame_size: Target frame size (width, height)
            stride: Stride between sequences
            transform: Optional transforms to apply
            max_videos: Maximum number of videos to load
        """
        self.video_dir = Path(video_dir)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.stride = stride
        self.transform = transform
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        self.video_files = []
        
        if self.video_dir.exists():
            for ext in video_extensions:
                self.video_files.extend(list(self.video_dir.glob(f'**/*{ext}')))
        
        if max_videos:
            self.video_files = self.video_files[:max_videos]
        
        # Build index of all sequences
        self.sequences = []
        self._build_sequence_index()
    
    def _build_sequence_index(self):
        """Build index of all valid sequences"""
        for video_idx, video_path in enumerate(self.video_files):
            try:
                cap = cv2.VideoCapture(str(video_path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Create sequences with stride
                for start_idx in range(0, frame_count - self.sequence_length + 1, self.stride):
                    self.sequences.append({
                        'video_idx': video_idx,
                        'start_frame': start_idx,
                        'video_path': video_path
                    })
            except Exception as e:
                print(f"Warning: Could not process {video_path}: {e}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a video sequence
        
        Returns:
            Tensor of shape (T, C, H, W) where T is sequence_length
        """
        seq_info = self.sequences[idx]
        video_path = seq_info['video_path']
        start_frame = seq_info['start_frame']
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                # If we run out of frames, repeat the last one
                if frames:
                    frame = frames[-1]
                else:
                    break
            else:
                # Resize and convert
                frame = cv2.resize(frame, self.frame_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            frames.append(frame)
        
        cap.release()
        
        # Convert to numpy array
        frames = np.array(frames)
        
        # Normalize
        frames = normalize_frames(frames)
        
        # Convert to tensor (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        # Apply transforms
        if self.transform:
            frames = self.transform(frames)
        
        return frames


class SyntheticTrafficDataset(Dataset):
    """
    Synthetic dataset for testing (generates random traffic-like patterns)
    """
    
    def __init__(self, num_samples=100, sequence_length=16, frame_size=(224, 224)):
        """
        Args:
            num_samples: Number of synthetic samples
            sequence_length: Number of frames per sequence
            frame_size: Frame dimensions
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.frame_size = frame_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate synthetic traffic video sequence
        
        Returns:
            Tensor of shape (T, C, H, W)
        """
        # Generate random motion patterns
        h, w = self.frame_size
        
        # Create base background
        background = np.random.rand(h, w, 3) * 0.3 + 0.5
        
        # Generate moving objects (vehicles)
        num_vehicles = np.random.randint(2, 6)
        frames = []
        
        for t in range(self.sequence_length):
            frame = background.copy()
            
            for v in range(num_vehicles):
                # Vehicle properties
                x = int((idx * 7 + v * 13 + t * 5) % w)
                y = int((idx * 11 + v * 17 + t * 3) % h)
                size = 20 + (idx * v) % 10
                
                # Draw vehicle as rectangle
                x1, y1 = max(0, x - size), max(0, y - size)
                x2, y2 = min(w, x + size), min(h, y + size)
                
                color = np.random.rand(3) * 0.5 + 0.3
                frame[y1:y2, x1:x2] = color
            
            frames.append(frame)
        
        frames = np.array(frames, dtype=np.float32)
        
        # Convert to tensor (T, H, W, C) -> (T, C, H, W)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        return frames


def create_data_loaders(video_dir, batch_size=4, sequence_length=16,
                       frame_size=(224, 224), num_workers=4,
                       train_split=0.8, use_synthetic=False):
    """
    Create train and validation data loaders
    
    Args:
        video_dir: Directory containing videos (ignored if use_synthetic=True)
        batch_size: Batch size
        sequence_length: Number of frames per sequence
        frame_size: Frame dimensions
        num_workers: Number of data loading workers
        train_split: Fraction of data for training
        use_synthetic: Whether to use synthetic data
    
    Returns:
        train_loader, val_loader
    """
    if use_synthetic:
        # Use synthetic dataset
        num_train = int(100 * train_split)
        num_val = 100 - num_train
        
        train_dataset = SyntheticTrafficDataset(
            num_samples=num_train,
            sequence_length=sequence_length,
            frame_size=frame_size
        )
        
        val_dataset = SyntheticTrafficDataset(
            num_samples=num_val,
            sequence_length=sequence_length,
            frame_size=frame_size
        )
    else:
        # Use real video dataset
        full_dataset = VideoDataset(
            video_dir=video_dir,
            sequence_length=sequence_length,
            frame_size=frame_size,
            stride=sequence_length // 2
        )
        
        # Split into train and validation
        train_size = int(len(full_dataset) * train_split)
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
