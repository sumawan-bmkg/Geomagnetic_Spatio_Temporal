import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def parse_kp_index(csv_path: str):
    """Parse Kp-index from CSV file."""
    if not Path(csv_path).exists():
        logger.warning(f"Kp-index file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['Date_Time_UTC'])
    # Normalize to naive UTC to avoid merge conflicts
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
    return df[['datetime', 'Kp_Index']]

def parse_dst_index(txt_path: str):
    """Parse Dst index from custom text format."""
    if not Path(txt_path).exists():
        logger.warning(f"Dst index file not found: {txt_path}")
        return None
    
    records = []
    with open(txt_path, 'r') as f:
        # Skip header
        lines = f.readlines()
        for line in lines:
            if '-' not in line:  # Skip headers or empty lines
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[0].startswith('20'):
                try:
                    dt_str = f"{parts[0]} {parts[1]}"
                    dst_val = float(parts[3])
                    records.append({'dt_str': dt_str, 'Dst': dst_val})
                except (ValueError, IndexError):
                    continue
    
    df = pd.DataFrame(records)
    if not df.empty:
        df['datetime'] = pd.to_datetime(df['dt_str'], format='ISO8601')
        # Normalize to naive UTC
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
        return df[['datetime', 'Dst']]
    return None

def align_indices(event_datetimes: pd.Series, kp_df: pd.DataFrame, dst_df: pd.DataFrame):
    """Align Kp and Dst indices with event datetimes."""
    # Use format='mixed' to handle various date formats
    dts = pd.to_datetime(event_datetimes, format='mixed')
    # Normalize to naive UTC
    if dts.dt.tz is not None:
        dts = dts.dt.tz_convert('UTC').dt.tz_localize(None)
    
    results = pd.DataFrame({'event_datetime': dts})
    
    if kp_df is not None:
        kp_df = kp_df.sort_values('datetime')
        results = pd.merge_asof(results.sort_values('event_datetime'), 
                                kp_df, 
                                left_on='event_datetime', 
                                right_on='datetime', 
                                direction='nearest')
    else:
        results['Kp_Index'] = 0.0
        
    if dst_df is not None:
        dst_df = dst_df.sort_values('datetime')
        results = pd.merge_asof(results.sort_values('event_datetime'), 
                                dst_df, 
                                left_on='event_datetime', 
                                right_on='datetime', 
                                direction='nearest')
    else:
        results['Dst'] = 0.0
        
    # Handle cases where merge_asof didn't find matches
    results['Kp_Index'] = results['Kp_Index'].fillna(0.0)
    results['Dst'] = results['Dst'].fillna(0.0)
    
    return results[['Kp_Index', 'Dst']].values


