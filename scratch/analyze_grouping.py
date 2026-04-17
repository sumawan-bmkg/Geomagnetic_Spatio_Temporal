import h5py
import pandas as pd
from collections import defaultdict

dataset_path = 'd:/multi/spatio/Spatio_Precursor_Project/outputs/corrective_actions/certified_spatio_dataset.h5'

with h5py.File(dataset_path, 'r') as f:
    for split in ['train', 'val']:
        print(f"\nAnalyzing split: {split}")
        if 'meta' in f[split]:
            meta = f[split]['meta'][:]
            days = defaultdict(list)
            for i, m in enumerate(meta):
                m_str = m.decode('utf-8') if isinstance(m, bytes) else str(m)
                # Format: event_STN_YYYYMMDD.npy
                parts = m_str.split('_')
                if len(parts) >= 3:
                    stn = parts[1]
                    date = parts[2].split('.')[0]
                    days[date].append(stn)
            
            # Print stats
            counts = [len(s) for s in days.values()]
            print(f"  Total unique days: {len(days)}")
            print(f"  Stations per day (mean): {sum(counts)/len(days):.2f}")
            print(f"  Max stations per day: {max(counts)}")
            
            # Show a few examples
            example_days = list(days.keys())[:5]
            for d in example_days:
                print(f"    {d}: {days[d]}")
