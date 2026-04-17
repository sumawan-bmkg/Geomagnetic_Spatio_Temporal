import h5py
import numpy as np

dataset_path = 'd:/multi/spatio/Spatio_Precursor_Project/outputs/corrective_actions/certified_spatio_dataset.h5'

with h5py.File(dataset_path, 'r') as f:
    for split in ['train', 'val']:
        print(f"\nChecking split: {split}")
        if split in f:
            tensors = f[split]['tensors']
            print(f"  Tensors shape: {tensors.shape}")
            # Tensors shape is (N, 3, 128, 1440)
            # This confirms it is (N, Components, Freq, Time)
            # THERE IS NO STATION DIMENSION IN THE RAW TENSORS!
            
            # Let's check 'meta' to see if station info is there
            if 'meta' in f[split]:
                meta = f[split]['meta'][:]
                unique_stations = set()
                for m in meta[:100]:
                    try:
                        # meta items are likely strings or dicts
                        m_str = m.decode('utf-8') if isinstance(m, bytes) else str(m)
                        if 'station' in m_str.lower():
                            unique_stations.add(m_str)
                    except:
                        pass
                print(f"  Sample meta snippets: {meta[:3]}")
