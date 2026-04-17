import h5py

dataset_path = 'd:/multi/spatio/Spatio_Precursor_Project/outputs/corrective_actions/certified_spatio_dataset.h5'

with h5py.File(dataset_path, 'r') as f:
    if 'train' in f:
        print(f"Keys in 'train': {list(f['train'].keys())}")
        for key in f['train'].keys():
            item = f['train'][key]
            if isinstance(item, h5py.Dataset):
                print(f"  {key}: {item.shape}, {item.dtype}")
            else:
                print(f"  {key}: Group, {len(item.keys())} items")
