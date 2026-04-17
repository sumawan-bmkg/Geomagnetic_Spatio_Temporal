#!/usr/bin/env python3
"""
Quick Dataset Check - Verify Production Dataset

Script cepat untuk memeriksa struktur dataset production yang baru dibuat.
"""
import h5py
import numpy as np
from pathlib import Path

def check_dataset():
    """Check dataset structure and content."""
    dataset_path = Path('real_earthquake_dataset.h5')
    
    if not dataset_path.exists():
        print("❌ Dataset not found!")
        return
    
    print("🔍 QUICK DATASET CHECK")
    print("=" * 50)
    
    with h5py.File(dataset_path, 'r') as f:
        print(f"📁 File size: {dataset_path.stat().st_size / (1024**2):.2f} MB")
        
        # Check main tensor
        if 'scalogram_tensor' in f:
            tensor = f['scalogram_tensor']
            print(f"📊 Tensor shape: {tensor.shape}")
            print(f"📊 Tensor dtype: {tensor.dtype}")
            
            # Sample data
            sample = tensor[0, 0, 0, :5, :5]
            print(f"📊 Sample data:\n{sample}")
        
        # Check metadata
        if 'metadata' in f:
            metadata = f['metadata']
            print(f"\n📋 Metadata groups: {list(metadata.keys())}")
            
            if 'event_id' in metadata:
                event_ids = metadata['event_id'][:10]
                print(f"📋 Sample event IDs: {event_ids}")
            
            if 'magnitude' in metadata:
                magnitudes = metadata['magnitude'][:10]
                print(f"📋 Sample magnitudes: {magnitudes}")
        
        # Check integration info
        if 'integration_info' in f:
            integration = f['integration_info']
            print(f"\n🔧 Integration info:")
            for key in integration.attrs:
                print(f"   {key}: {integration.attrs[key]}")
        
        # Check file attributes
        print(f"\n📝 File attributes:")
        for key in f.attrs:
            print(f"   {key}: {f.attrs[key]}")

if __name__ == '__main__':
    check_dataset()