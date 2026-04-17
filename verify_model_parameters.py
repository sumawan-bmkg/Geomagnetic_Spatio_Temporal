#!/usr/bin/env python3
"""
Model Parameter Verification Script
Verifies that EfficientNet-B0 is properly loaded and compares with LightweightModel

Author: Kiro AI Assistant
Date: April 16, 2026
"""
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_b0

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.spatio_temporal_model import SpatioTemporalPrecursorModel

def count_parameters(model):
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def create_lightweight_model():
    """Create the problematic LightweightModel for comparison."""
    class LightweightModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Simple CNN backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            
            # Simple heads
            self.binary_head = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            self.magnitude_head = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 5)  # 5 magnitude classes
            )
    
    return LightweightModel()

def verify_efficientnet_backbone():
    """Verify EfficientNet-B0 backbone parameters."""
    print("=== VERIFYING EFFICIENTNET-B0 BACKBONE ===")
    
    # Create standalone EfficientNet-B0
    efficientnet = efficientnet_b0(pretrained=False)
    efficientnet.classifier = nn.Identity()  # Remove classifier
    
    total_params, trainable_params = count_parameters(efficientnet)
    
    print(f"Standalone EfficientNet-B0 (backbone only):")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return total_params

def verify_spatio_temporal_model():
    """Verify full SpatioTemporalPrecursorModel parameters."""
    print("\n=== VERIFYING SPATIO-TEMPORAL MODEL ===")
    
    # Create station coordinates for 8 stations (dummy coordinates)
    station_coordinates = np.array([
        [-6.2, 106.8],  # Jakarta
        [-7.8, 110.4],  # Yogyakarta
        [-6.9, 107.6],  # Bandung
        [-7.3, 112.7],  # Surabaya
        [-8.7, 115.2],  # Denpasar
        [-0.9, 100.4],  # Padang
        [3.6, 98.7],    # Medan
        [-5.1, 119.4]   # Makassar
    ])
    
    # Create full model
    model = SpatioTemporalPrecursorModel(
        n_stations=8,
        n_components=3,
        station_coordinates=station_coordinates,
        efficientnet_pretrained=False,  # Don't download weights for verification
        gnn_hidden_dim=256,
        gnn_num_layers=3,
        dropout_rate=0.3,
        device='cpu'
    )
    
    total_params, trainable_params = count_parameters(model)
    
    print(f"Full SpatioTemporalPrecursorModel:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Verify components
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    gnn_params = sum(p.numel() for p in model.gnn_fusion.parameters())
    heads_params = sum(p.numel() for p in model.hierarchical_heads.parameters())
    
    print(f"  Backbone (EfficientNet-B0): {backbone_params:,}")
    print(f"  GNN Fusion: {gnn_params:,}")
    print(f"  Hierarchical Heads: {heads_params:,}")
    
    return total_params

def verify_lightweight_model():
    """Verify problematic LightweightModel parameters."""
    print("\n=== VERIFYING LIGHTWEIGHT MODEL (PROBLEMATIC) ===")
    
    model = create_lightweight_model()
    total_params, trainable_params = count_parameters(model)
    
    print(f"LightweightModel (problematic):")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Verify components
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    binary_head_params = sum(p.numel() for p in model.binary_head.parameters())
    magnitude_head_params = sum(p.numel() for p in model.magnitude_head.parameters())
    
    print(f"  Backbone (Custom CNN): {backbone_params:,}")
    print(f"  Binary Head: {binary_head_params:,}")
    print(f"  Magnitude Head: {magnitude_head_params:,}")
    
    return total_params

def main():
    """Main verification function."""
    print("MODEL PARAMETER VERIFICATION")
    print("=" * 50)
    
    # Verify EfficientNet-B0 backbone
    efficientnet_params = verify_efficientnet_backbone()
    
    # Verify full SpatioTemporal model
    spatio_temporal_params = verify_spatio_temporal_model()
    
    # Verify problematic LightweightModel
    lightweight_params = verify_lightweight_model()
    
    # Summary comparison
    print("\n=== PARAMETER COMPARISON SUMMARY ===")
    print(f"EfficientNet-B0 backbone:     {efficientnet_params:,} parameters")
    print(f"Full SpatioTemporal model:    {spatio_temporal_params:,} parameters")
    print(f"LightweightModel (problem):   {lightweight_params:,} parameters")
    
    print(f"\nParameter reduction: {spatio_temporal_params / lightweight_params:.1f}x smaller")
    
    # Verification results
    print("\n=== VERIFICATION RESULTS ===")
    
    if spatio_temporal_params >= 5_000_000:
        print("✅ SpatioTemporal model has 5M+ parameters (CORRECT)")
    else:
        print("❌ SpatioTemporal model has <5M parameters (INCORRECT)")
    
    if lightweight_params < 500_000:
        print("❌ LightweightModel has <500K parameters (PROBLEMATIC)")
    else:
        print("✅ LightweightModel has sufficient parameters")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    print("1. Use SpatioTemporalPrecursorModel (5M+ parameters)")
    print("2. Avoid LightweightModel (286K parameters)")
    print("3. Use corrected_production_training.py script")
    print("4. Expect 2-4 hours training time for full dataset")
    
    # Test forward pass
    print("\n=== TESTING FORWARD PASS ===")
    
    # Create test input
    batch_size = 2
    num_stations = 8
    num_components = 3
    freq_bins = 64
    time_steps = 128
    
    test_input = torch.randn(batch_size, num_stations, num_components, freq_bins, time_steps)
    
    print(f"Test input shape: {test_input.shape}")
    
    # Test SpatioTemporal model
    station_coordinates = np.array([
        [-6.2, 106.8],  # Jakarta
        [-7.8, 110.4],  # Yogyakarta
        [-6.9, 107.6],  # Bandung
        [-7.3, 112.7],  # Surabaya
        [-8.7, 115.2],  # Denpasar
        [-0.9, 100.4],  # Padang
        [3.6, 98.7],    # Medan
        [-5.1, 119.4]   # Makassar
    ])
    
    model = SpatioTemporalPrecursorModel(
        n_stations=8,
        n_components=3,
        station_coordinates=station_coordinates,
        efficientnet_pretrained=False,
        gnn_hidden_dim=256,
        gnn_num_layers=3,
        dropout_rate=0.3,
        device='cpu'
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_input)
    
    print("SpatioTemporal model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✅ Forward pass successful!")

if __name__ == "__main__":
    main()