import h5py
import numpy as np

dataset_path = 'd:/multi/spatio/Spatio_Precursor_Project/outputs/corrective_actions/certified_spatio_dataset.h5'

def print_structure(name, obj):
    print(f"{name}: {type(obj)}")
    if isinstance(obj, h5py.Dataset):
        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  Keys: {list(obj.keys())}")

with h5py.File(dataset_path, 'r') as f:
    print(f"Dataset keys: {list(f.keys())}")
    
    if 'train' in f:
        group = f['train']
        print(f"Number of items in train: {len(group.keys())}")
        first_key = list(group.keys())[0]
        obj = group[first_key]
        print(f"First item in train: {first_key}, Type: {type(obj)}")
        
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}")
        elif isinstance(obj, h5py.Group):
            print(f"  Sub-keys: {list(obj.keys())}")
            if 'data' in obj:
                print(f"  Data shape: {obj['data'].shape}")
            if 'raw_signal' in obj:
                 print(f"  Raw signal shape: {obj['raw_signal'].shape}")

    if 'certification_metadata' in f:
        print("\nCertification Metadata Group:")
        meta = f['certification_metadata']
        for k in meta.attrs.keys():
            print(f"  Attr {k}: {meta.attrs[k]}")
