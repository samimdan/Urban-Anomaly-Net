"""
Quick validation test for Urban Anomaly Detection Network
Tests core functionality with synthetic data
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import SpatioTemporalAutoencoder
from models.anomaly_detector import AnomalyDetector
from utils.data_utils import SyntheticTrafficDataset
from torch.utils.data import DataLoader


def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    model = SpatioTemporalAutoencoder(
        input_channels=3,
        hidden_channels=[32, 64],  # Smaller for quick test
        kernel_size=3
    )
    print("✓ Model created successfully")
    return model


def test_forward_pass(model):
    """Test forward pass"""
    print("\nTesting forward pass...")
    batch_size = 2
    seq_length = 8
    channels = 3
    height, width = 64, 64
    
    # Create dummy input
    x = torch.randn(batch_size, seq_length, channels, height, width)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    print(f"✓ Forward pass successful: {x.shape} -> {output.shape}")
    return output


def test_synthetic_dataset():
    """Test synthetic dataset"""
    print("\nTesting synthetic dataset...")
    dataset = SyntheticTrafficDataset(
        num_samples=10,
        sequence_length=8,
        frame_size=(64, 64)
    )
    
    # Get sample
    sample = dataset[0]
    print(f"✓ Dataset created: {len(dataset)} samples, shape: {sample.shape}")
    return dataset


def test_data_loader(dataset):
    """Test data loader"""
    print("\nTesting data loader...")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    batch = next(iter(loader))
    print(f"✓ DataLoader works: batch shape {batch.shape}")
    return loader


def test_training_step(model, loader):
    """Test a single training step"""
    print("\nTesting training step...")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    batch = next(iter(loader))
    
    # Forward pass
    output = model(batch)
    loss = criterion(output, batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful: loss = {loss.item():.6f}")
    return loss.item()


def test_anomaly_detector(model, loader):
    """Test anomaly detector"""
    print("\nTesting anomaly detector...")
    
    detector = AnomalyDetector(model, threshold_percentile=95)
    
    # Fit threshold
    detector.fit_threshold(loader, device='cpu')
    
    # Test detection
    batch = next(iter(loader))
    is_anomaly, errors = detector.detect_anomalies(batch, device='cpu', return_errors=True)
    
    print(f"✓ Anomaly detector works: threshold = {detector.threshold:.6f}")
    print(f"  Detected {is_anomaly.sum().item()}/{len(is_anomaly)} anomalies")
    return detector


def test_spatial_anomaly_map(detector, loader):
    """Test spatial anomaly map generation"""
    print("\nTesting spatial anomaly map...")
    
    batch = next(iter(loader))
    anomaly_map = detector.get_spatial_anomaly_map(batch[:1], device='cpu')
    
    print(f"✓ Spatial anomaly map generated: shape {anomaly_map.shape}")
    return anomaly_map


def run_all_tests():
    """Run all validation tests"""
    print("="*60)
    print("Urban Anomaly Detection Network - Validation Tests")
    print("="*60)
    
    try:
        # Test model
        model = test_model_creation()
        test_forward_pass(model)
        
        # Test data
        dataset = test_synthetic_dataset()
        loader = test_data_loader(dataset)
        
        # Test training
        test_training_step(model, loader)
        
        # Test anomaly detection
        detector = test_anomaly_detector(model, loader)
        test_spatial_anomaly_map(detector, loader)
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Test failed: {str(e)}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
