"""
Urban Anomaly Detection Network
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Urban-Anomaly-Net"

from .models.autoencoder import SpatioTemporalAutoencoder
from .models.anomaly_detector import AnomalyDetector

__all__ = [
    'SpatioTemporalAutoencoder',
    'AnomalyDetector',
]
