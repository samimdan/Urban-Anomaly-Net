"""
Configuration file for Urban Anomaly Detection Network
"""

# Model configuration
MODEL_CONFIG = {
    'input_channels': 3,
    'hidden_channels': [64, 128, 256],
    'kernel_size': 3,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'sequence_length': 16,
    'frame_size': (224, 224),
    'save_interval': 5,
}

# Data configuration
DATA_CONFIG = {
    'train_split': 0.8,
    'num_workers': 4,
    'stride': 8,  # Stride for sequence extraction
}

# Anomaly detection configuration
ANOMALY_CONFIG = {
    'threshold_percentile': 95,
    'use_spatial_map': True,
}

# System configuration
SYSTEM_CONFIG = {
    'device': 'cuda',  # 'cuda' or 'cpu'
    'seed': 42,
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'output_dir': 'outputs',
}
