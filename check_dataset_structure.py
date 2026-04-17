#!/usr/bin/env python3
"""
Check HDF5 dataset structure
"""
import h5py
import numpy as np

def check_dataset(filepath):
    print(f"Checking dataset: {filepath}")
    
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"Top-level keys: {list(f.keys())}")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
            
            f.visititems(print_structure)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_dataset("real_earthquake_dataset.h5")