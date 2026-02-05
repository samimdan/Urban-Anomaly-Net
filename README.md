# Urban Anomaly Detection Network

> **Unsupervised Learning of Normal Urban Traffic Dynamics for Automatic Anomaly Detection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Urban Anomaly Detection Network is a deep learning system that learns normal urban traffic dynamics from large-scale surveillance videos to automatically detect anomalous events such as accidents and abnormal vehicle behaviors. By modeling spatio-temporal motion patterns without manual annotations, the system identifies deviations from normal traffic flow, enabling scalable traffic monitoring and urban safety analysis in real-world city environments.

### Key Features

- 🚗 **Unsupervised Learning**: No manual annotations required - learns from normal traffic patterns
- 🎯 **Spatio-Temporal Modeling**: ConvLSTM-based autoencoder captures both spatial and temporal dynamics
- 🔍 **Anomaly Detection**: Identifies deviations using reconstruction error analysis
- 📊 **Visualization**: Spatial heatmaps highlight anomalous regions
- ⚡ **Scalable**: Batch processing for real-world deployment
- 🔧 **Flexible**: Works with any surveillance video format

## Architecture

The system uses a **Spatio-Temporal Autoencoder** architecture:

```
Input Video → ConvLSTM Encoder → Latent Representation → ConvLSTM Decoder → Reconstructed Video
                                        ↓
                              Reconstruction Error → Anomaly Score
```

**Key Components:**

1. **ConvLSTM Encoder**: Extracts spatio-temporal features from video sequences
2. **ConvLSTM Decoder**: Reconstructs the original video from learned features
3. **Anomaly Detector**: Uses reconstruction error to identify anomalies

Normal traffic patterns are reconstructed with low error, while anomalies (accidents, unusual behaviors) produce high reconstruction errors.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/samimdan/Urban-Anomaly-Net.git
cd Urban-Anomaly-Net

# Install required packages
pip install -r requirements.txt
```

### Dependencies

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- NumPy >= 1.21.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.4.0
- tqdm >= 4.62.0

## Quick Start

### 1. Training on Synthetic Data (Demo)

```bash
# Train on synthetic data for quick testing
python scripts/train.py \
    --use-synthetic \
    --epochs 20 \
    --batch-size 4 \
    --sequence-length 16 \
    --frame-size 128
```

### 2. Training on Real Videos

```bash
# Prepare your video dataset
# Place normal traffic videos in data/videos/

# Train the model
python scripts/train.py \
    --data-dir data/videos \
    --epochs 50 \
    --batch-size 4 \
    --sequence-length 16 \
    --frame-size 224 \
    --hidden-channels 64 128 256
```

### 3. Anomaly Detection

```bash
# Detect anomalies in test videos
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video-dir data/test_videos \
    --normal-data-dir data/videos \
    --visualize \
    --output-dir outputs
```

### 4. Interactive Demo

See the [demo notebook](examples/demo.ipynb) for an interactive example:

```bash
cd examples
jupyter notebook demo.ipynb
```

## Usage Guide

### Training

The training script supports various configurations:

```bash
python scripts/train.py \
    --data-dir <path_to_videos>     # Directory with training videos
    --sequence-length 16             # Frames per sequence
    --frame-size 224                 # Frame dimensions (224x224)
    --hidden-channels 64 128 256     # Encoder/decoder channels
    --batch-size 4                   # Batch size
    --epochs 50                      # Training epochs
    --lr 1e-3                        # Learning rate
    --checkpoint-dir checkpoints     # Save checkpoints here
    --log-dir logs                   # Tensorboard logs
```

**Key Parameters:**
- `--sequence-length`: Number of consecutive frames (longer captures more temporal info)
- `--frame-size`: Spatial resolution (higher = more detail, slower training)
- `--hidden-channels`: Network capacity (larger = more expressive, more memory)

### Inference

Detect anomalies in new videos:

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video-file path/to/test_video.mp4 \
    --normal-data-dir data/videos \
    --threshold-percentile 95 \
    --visualize
```

**Options:**
- `--video-file`: Process a single video
- `--video-dir`: Process all videos in a directory
- `--threshold-percentile`: Sensitivity (higher = fewer false positives)
- `--visualize`: Generate anomaly heatmaps

### Using as a Library

```python
import torch
from models.autoencoder import SpatioTemporalAutoencoder
from models.anomaly_detector import AnomalyDetector
from utils.data_utils import create_data_loaders

# Create model
model = SpatioTemporalAutoencoder(
    input_channels=3,
    hidden_channels=[64, 128, 256],
    kernel_size=3
)

# Load trained weights
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create anomaly detector
detector = AnomalyDetector(model, threshold_percentile=95)

# Fit threshold on normal data
detector.fit_threshold(normal_data_loader, device='cuda')

# Detect anomalies
is_anomaly, errors = detector.detect_anomalies(test_videos)
```

## Project Structure

```
Urban-Anomaly-Net/
├── models/
│   ├── __init__.py
│   ├── autoencoder.py         # Spatio-temporal autoencoder
│   └── anomaly_detector.py    # Anomaly detection logic
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading and preprocessing
│   └── visualization.py       # Visualization utilities
├── scripts/
│   ├── train.py               # Training script
│   └── inference.py           # Inference script
├── examples/
│   └── demo.ipynb             # Interactive demo notebook
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Model Details

### Spatio-Temporal Autoencoder

**Encoder:**
- Spatial feature extraction with 2D convolutions
- Temporal modeling with ConvLSTM cells
- Progressive downsampling (2x per layer)

**Decoder:**
- Temporal processing with ConvLSTM cells
- Spatial reconstruction with transpose convolutions
- Progressive upsampling (2x per layer)

**Loss Function:**
- Mean Squared Error (MSE) between original and reconstructed frames
- Encourages faithful reconstruction of normal patterns

### Anomaly Detection

**Method:**
1. Train autoencoder on normal traffic videos
2. Compute reconstruction errors on validation set
3. Set threshold at specified percentile (e.g., 95th)
4. Flag sequences with errors above threshold as anomalies

**Spatial Anomaly Maps:**
- Per-pixel reconstruction error heatmaps
- Highlight specific anomalous regions
- Useful for understanding what triggered the alert

## Examples

### Training Results

After training, you'll see reconstruction quality improve:

- **Epoch 1**: Blurry reconstructions, high error
- **Epoch 20**: Sharp reconstructions, low error on normal traffic
- **Final Model**: Accurate reconstructions enable reliable anomaly detection

### Anomaly Detection Results

The system can detect various anomalies:

- 🚨 **Accidents**: Sudden vehicle stops, unusual patterns
- 🚗 **Wrong-way driving**: Vehicles moving against traffic flow
- 🚶 **Pedestrians in roadway**: Unexpected objects in road
- ⚠️ **Congestion**: Abnormal traffic density patterns

## Performance Tips

### Training
- Start with smaller frame sizes (128x128) for faster iteration
- Use synthetic data to verify pipeline before training on real data
- Monitor validation loss to avoid overfitting
- Save checkpoints regularly

### Inference
- Batch process multiple videos for efficiency
- Adjust threshold based on desired sensitivity
- Use GPU for faster processing
- Cache threshold after fitting on normal data

## Limitations

- Requires sufficient normal traffic data for training
- May produce false positives on rare but normal events
- Performance depends on video quality and viewing angle
- Computational cost scales with video resolution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{urban_anomaly_net,
  title = {Urban Anomaly Detection Network},
  author = {Urban-Anomaly-Net Contributors},
  year = {2024},
  url = {https://github.com/samimdan/Urban-Anomaly-Net}
}
```

## Acknowledgments

- Inspired by research in video anomaly detection and unsupervised learning
- Built with PyTorch and OpenCV
- ConvLSTM architecture based on spatio-temporal modeling literature

## Support

For questions and issues:
- Open an issue on GitHub
- Check the [demo notebook](examples/demo.ipynb) for examples
- Review the code documentation

## Roadmap

Future enhancements:
- [ ] Real-time processing pipeline
- [ ] Pre-trained models for common scenarios
- [ ] Multi-camera support
- [ ] Alert notification system
- [ ] Web-based visualization dashboard
- [ ] Additional anomaly detection methods

---

**Built with ❤️ for safer cities**
