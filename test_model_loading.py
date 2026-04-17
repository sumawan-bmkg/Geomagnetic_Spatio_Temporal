#!/usr/bin/env python3
"""
Test Model Loading Script
========================

Quick test to verify that the trained model can be loaded correctly
and is ready for inference validation.
"""

import os
import sys
import torch
import glob
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_model_loading():
    """Test loading of the trained model."""
    
    print("🔍 TESTING MODEL LOADING")
    print("=" * 40)
    
    # Find model checkpoint
    search_patterns = [
        "outputs/production_training/*/ground_truth_run/best_stage_3.pth",
        "outputs/production_training/*/best_stage_3.pth"
    ]
    
    model_path = None
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            model_path = matches[0]
            break
    
    if not model_path:
        print("❌ No model checkpoint found!")
        return False
    
    print(f"✓ Found model: {model_path}")
    
    try:
        # Load checkpoint
        print("📥 Loading checkpoint...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        print(f"✓ Checkpoint loaded on {device}")
        
        # Check checkpoint contents
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✓ Found model_state_dict in checkpoint")
        else:
            state_dict = checkpoint
            print("✓ Checkpoint is direct state_dict")
        
        # Print some key information
        print(f"✓ State dict contains {len(state_dict)} parameters")
        
        # Check for key model components
        key_components = [
            'backbone',
            'binary_head',
            'magnitude_head', 
            'localization_head'
        ]
        
        found_components = []
        for component in key_components:
            component_keys = [k for k in state_dict.keys() if component in k]
            if component_keys:
                found_components.append(component)
        
        print(f"✓ Found model components: {found_components}")
        
        # Try to load the actual model
        from models.spatio_temporal_model import SpatioTemporalPrecursorModel
        
        model_config = {
            'n_stations': 8,
            'n_components': 3,
            'station_coordinates': None,
            'efficientnet_pretrained': False,  # Already trained
            'gnn_hidden_dim': 256,
            'gnn_num_layers': 3,
            'dropout_rate': 0.2,
            'magnitude_classes': 5,
            'device': device
        }
        
        print("🏗️ Initializing model architecture...")
        model = SpatioTemporalPrecursorModel(**model_config)
        
        print("📋 Loading state dict...")
        model.load_state_dict(state_dict)
        
        print("🎯 Setting to evaluation mode...")
        model.eval()
        model.to(device)
        
        # Test forward pass with dummy data
        print("🧪 Testing forward pass...")
        dummy_input = torch.randn(1, 8, 3, 64, 128).to(device)  # [batch, stations, components, freq, time]
        
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✓ Forward pass successful!")
        print(f"✓ Output keys: {list(outputs.keys())}")
        
        for key, value in outputs.items():
            print(f"  - {key}: {value.shape}")
        
        print("\n" + "=" * 40)
        print("✅ MODEL LOADING TEST PASSED!")
        print("🚀 Ready for inference validation!")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_availability():
    """Test if dataset is available."""
    
    print("\n🗃️ TESTING DATASET AVAILABILITY")
    print("=" * 40)
    
    dataset_paths = [
        "real_earthquake_dataset.h5",
        "data/real_earthquake_dataset.h5"
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✓ Found dataset: {path}")
            
            # Check file size
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Dataset size: {size_mb:.1f} MB")
            
            return True
    
    print("❌ No dataset found!")
    print("Expected locations:")
    for path in dataset_paths:
        print(f"  - {path}")
    
    return False

def main():
    """Main test function."""
    
    print("🧪 PRODUCTION INFERENCE READINESS TEST")
    print("=" * 50)
    
    # Test model loading
    model_ok = test_model_loading()
    
    # Test dataset availability  
    dataset_ok = test_dataset_availability()
    
    print("\n" + "=" * 50)
    if model_ok and dataset_ok:
        print("✅ ALL TESTS PASSED!")
        print("🚀 System ready for production inference!")
        print("\nNext steps:")
        print("  python launch_inference_validation.py")
    else:
        print("❌ SOME TESTS FAILED!")
        if not model_ok:
            print("  - Model loading issues detected")
        if not dataset_ok:
            print("  - Dataset not found")
    print("=" * 50)

if __name__ == "__main__":
    main()