"""
Models package initialization
"""

from .autoencoder import SpatioTemporalAutoencoder
from .anomaly_detector import AnomalyDetector

__all__ = ['SpatioTemporalAutoencoder', 'AnomalyDetector']
