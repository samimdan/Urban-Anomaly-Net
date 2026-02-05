"""
Anomaly Detector using reconstruction error from autoencoder
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.mixture import GaussianMixture


class AnomalyDetector:
    """
    Detects anomalies in traffic videos by analyzing reconstruction error
    from the spatio-temporal autoencoder
    """
    
    def __init__(self, autoencoder, threshold_percentile=95):
        """
        Args:
            autoencoder: Trained SpatioTemporalAutoencoder model
            threshold_percentile: Percentile for anomaly threshold (default: 95)
        """
        self.autoencoder = autoencoder
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.mean_errors = None
        self.std_errors = None
    
    def compute_reconstruction_error(self, original, reconstructed, reduction='mean'):
        """
        Compute reconstruction error between original and reconstructed frames
        
        Args:
            original: Original video tensor
            reconstructed: Reconstructed video tensor
            reduction: How to reduce error ('mean', 'sum', or 'none')
        
        Returns:
            Reconstruction error
        """
        # Compute mean squared error
        mse = torch.nn.functional.mse_loss(
            reconstructed, original, reduction='none'
        )
        
        if reduction == 'mean':
            # Average across all dimensions except batch
            error = mse.mean(dim=(1, 2, 3, 4))
        elif reduction == 'sum':
            error = mse.sum(dim=(1, 2, 3, 4))
        elif reduction == 'spatial':
            # Average across spatial dimensions only
            error = mse.mean(dim=(3, 4))
        else:
            error = mse
        
        return error
    
    def fit_threshold(self, data_loader, device='cuda'):
        """
        Fit anomaly threshold on normal traffic data
        
        Args:
            data_loader: DataLoader with normal traffic videos
            device: Device to run computation on
        """
        self.autoencoder.eval()
        errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    videos = batch[0]
                else:
                    videos = batch
                
                videos = videos.to(device)
                
                # Get reconstruction
                reconstructed = self.autoencoder(videos)
                
                # Compute errors
                batch_errors = self.compute_reconstruction_error(
                    videos, reconstructed, reduction='mean'
                )
                errors.extend(batch_errors.cpu().numpy())
        
        errors = np.array(errors)
        
        # Compute statistics
        self.mean_errors = np.mean(errors)
        self.std_errors = np.std(errors)
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, self.threshold_percentile)
        
        print(f"Anomaly threshold set at {self.threshold:.6f}")
        print(f"Mean error: {self.mean_errors:.6f}, Std: {self.std_errors:.6f}")
        
        return self.threshold
    
    def detect_anomalies(self, videos, device='cuda', return_errors=False):
        """
        Detect anomalies in video sequences
        
        Args:
            videos: Input video tensor (batch, time_steps, channels, height, width)
            device: Device to run computation on
            return_errors: Whether to return reconstruction errors
        
        Returns:
            anomaly_scores: Boolean tensor indicating anomalies
            errors: Reconstruction errors (if return_errors=True)
        """
        self.autoencoder.eval()
        
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")
        
        with torch.no_grad():
            videos = videos.to(device)
            
            # Get reconstruction
            reconstructed = self.autoencoder(videos)
            
            # Compute errors
            errors = self.compute_reconstruction_error(
                videos, reconstructed, reduction='mean'
            )
            
            # Detect anomalies
            anomaly_scores = errors > self.threshold
        
        if return_errors:
            return anomaly_scores.cpu(), errors.cpu()
        else:
            return anomaly_scores.cpu()
    
    def get_spatial_anomaly_map(self, videos, device='cuda'):
        """
        Generate spatial anomaly heatmap for visualization
        
        Args:
            videos: Input video tensor
            device: Device to run computation on
        
        Returns:
            Spatial anomaly maps
        """
        self.autoencoder.eval()
        
        with torch.no_grad():
            videos = videos.to(device)
            
            # Get reconstruction
            reconstructed = self.autoencoder(videos)
            
            # Compute spatial errors
            spatial_errors = self.compute_reconstruction_error(
                videos, reconstructed, reduction='spatial'
            )
            
            # Average across time and channels
            anomaly_maps = spatial_errors.mean(dim=(1, 2))
        
        return anomaly_maps.cpu()
    
    def save(self, path):
        """Save detector state"""
        state = {
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'mean_errors': self.mean_errors,
            'std_errors': self.std_errors,
        }
        torch.save(state, path)
    
    def load(self, path):
        """Load detector state"""
        state = torch.load(path)
        self.threshold = state['threshold']
        self.threshold_percentile = state['threshold_percentile']
        self.mean_errors = state['mean_errors']
        self.std_errors = state['std_errors']
