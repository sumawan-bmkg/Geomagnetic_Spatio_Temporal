#!/usr/bin/env python3
"""
Inspect Certified Dataset Structure
"""

import h5py
import pandas as pd

def inspect_dataset():
    dataset_path = 'outputs/corrective_actions/certified_spatio_dataset.h5'
    
    print("🔍 INSPECTING CERTIFIED DATASET")
    print("=" * 50)
    
    try:
        with h5py.File(dataset_path, 'r') as f:
            print(f"Dataset: {dataset_path}")
            print(f"Top-level keys: {list(f.keys())}")
            
            for key in f.keys():
                print(f"\n📁 {key}:")
                obj = f[key]
                
                if hasattr(obj, 'keys'):
                    subkeys = list(obj.keys())
                    print(f"  Subkeys: {subkeys}")
                    
                    for subkey in subkeys:
                        subobj = obj[subkey]
                        if hasattr(subobj, 'shape'):
                            print(f"    {subkey}: shape {subobj.shape}, dtype {subobj.dtype}")
                        elif hasattr(subobj, 'keys'):
                            print(f"    {subkey}: {list(subobj.keys())}")
                        else:
                            print(f"    {subkey}: {type(subobj)}")
                else:
                    if hasattr(obj, 'shape'):
                        print(f"  Shape: {obj.shape}, dtype: {obj.dtype}")
                    else:
                        print(f"  Type: {type(obj)}")
            
            # Check for test data
            print("\n🎯 LOOKING FOR TEST DATA:")
            
            # Check if there's a test split
            if 'test' in f:
                print("✓ Found 'test' split")
                test_obj = f['test']
                if hasattr(test_obj, 'keys'):
                    print(f"  Test keys: {list(test_obj.keys())}")
            
            # Check train/val splits for metadata
            for split in ['train', 'val']:
                if split in f:
                    split_obj = f[split]
                    if hasattr(split_obj, 'keys'):
                        split_keys = list(split_obj.keys())
                        print(f"  {split} keys: {split_keys}")
                        
                        if 'metadata' in split_keys:
                            metadata_obj = split_obj['metadata']
                            if hasattr(metadata_obj, 'keys'):
                                print(f"    {split} metadata fields: {list(metadata_obj.keys())}")
                                # Sample size
                                first_field = list(metadata_obj.keys())[0]
                                sample_count = len(metadata_obj[first_field])
                                print(f"    {split} samples: {sample_count}")
            
            # Check certification metadata
            if 'certification_metadata' in f:
                cert_obj = f['certification_metadata']
                print(f"\n📋 Certification metadata: {type(cert_obj)}")
                if hasattr(cert_obj, 'keys'):
                    print(f"  Certification keys: {list(cert_obj.keys())}")
    
    except Exception as e:
        print(f"❌ Error inspecting dataset: {str(e)}")

if __name__ == "__main__":
    inspect_dataset()