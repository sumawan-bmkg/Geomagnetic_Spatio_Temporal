#!/usr/bin/env python3
"""
Production Dataset Adapter for Final Training
Adapts real_earthquake_dataset.h5 structure to training requirements

Author: Kiro AI Assistant
Date: April 16, 2026
"""
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from src.preprocessing.geophysical_parser import parse_kp_index, parse_dst_index, align_indices

class ProductionDatasetAdapter(Dataset):
    """
    Dataset adapter for real_earthquake_dataset.h5 structure
    """
    
    def __init__(self, hdf5_path: str, split: str = 'train', 
                 kp_path: str = '../awal/kp_index_2018_2026.csv',
                 dst_path: str = '../awal/dst.txt'):
        self.hdf5_path = hdf5_path
        self.split = split
        
        # Load dataset structure
        with h5py.File(hdf5_path, 'r') as f:
            # Get event IDs for split
            if split == 'train':
                self.event_ids = f['config']['train_event_ids'][:]
            else:  # val/test
                self.event_ids = f['config']['test_event_ids'][:]
            
            # Get tensor data shape
            self.tensor_shape = f['scalogram_tensor'].shape
            
            # Load metadata
            self.metadata = {
                'event_id': f['metadata']['event_id'][:],
                'magnitude': f['metadata']['magnitude'][:],
                'latitude': f['metadata']['latitude'][:],
                'longitude': f['metadata']['longitude'][:],
                'depth': f['metadata']['depth'][:],
                'datetime': f['metadata']['datetime'][:]
            }
        
        # Create event ID to index mapping
        self.event_id_to_tensor_idx = {}
        
        # Map event IDs to tensor indices (0-9155)
        for tensor_idx in range(self.tensor_shape[0]):  # 0 to 9155
            # Find corresponding event in metadata
            if tensor_idx < len(self.metadata['event_id']):
                event_id = self.metadata['event_id'][tensor_idx]
                event_id_str = event_id.decode('utf-8') if isinstance(event_id, bytes) else str(event_id)
                self.event_id_to_tensor_idx[event_id_str] = tensor_idx
        
        # Filter tensor indices for this split
        self.tensor_indices = []
        for event_id in self.event_ids:
            event_id_str = event_id.decode('utf-8') if isinstance(event_id, bytes) else str(event_id)
            if event_id_str in self.event_id_to_tensor_idx:
                tensor_idx = self.event_id_to_tensor_idx[event_id_str]
                if tensor_idx < self.tensor_shape[0]:  # Ensure within bounds
                    self.tensor_indices.append(tensor_idx)
        
        logger.info(f"Dataset {split}: {len(self.tensor_indices)} samples")
        
        # Load and align Kp/Dst indices
        self._load_geophysical_indices(kp_path, dst_path)
        
        # Prepare targets
        self._prepare_targets()
    
    def _load_geophysical_indices(self, kp_path, dst_path):
        """Load and align Kp and Dst indices for the active split."""
        logger.info("Loading and aligning Kp/Dst indices...")
        kp_df = parse_kp_index(kp_path)
        dst_df = parse_dst_index(dst_path)
        
        event_dts = []
        for tensor_idx in self.tensor_indices:
            dt = self.metadata['datetime'][tensor_idx]
            dt_str = dt.decode('utf-8') if isinstance(dt, bytes) else str(dt)
            event_dts.append(dt_str)
        
        self.geophysical_features = align_indices(pd.Series(event_dts), kp_df, dst_df)
        logger.info(f"Geophysical features aligned: {self.geophysical_features.shape}")

    def _prepare_targets(self):
        """Prepare targets for multi-task learning"""
        self.targets = {}
        
        # Get magnitudes for this split
        magnitudes = []
        for tensor_idx in self.tensor_indices:
            if tensor_idx < len(self.metadata['magnitude']):
                magnitudes.append(self.metadata['magnitude'][tensor_idx])
            else:
                magnitudes.append(5.0)  # Default magnitude
        
        magnitudes = np.array(magnitudes)
        
        # Binary classification: all earthquake events are precursors (1)
        self.targets['binary'] = np.ones(len(self.tensor_indices), dtype=np.float32)
        
        # Magnitude classification (5 classes)
        magnitude_bins = [4.0, 4.5, 5.0, 5.5, 6.0]
        magnitude_classes = np.digitize(magnitudes, magnitude_bins)
        magnitude_classes = np.clip(magnitude_classes, 0, 4)  # 5 classes: 0-4
        self.targets['magnitude_class'] = magnitude_classes.astype(np.int64)
        
        # Distance targets (dummy for now)
        self.targets['distance'] = np.ones(len(self.tensor_indices), dtype=np.float32) * 100.0
        
        logger.info(f"Targets prepared for {len(self.tensor_indices)} samples")
        if len(magnitudes) == 0:
            logger.warning(f"No samples found for split {self.split}")
            magnitudes = np.array([5.0])  # Default
        
        logger.info(f"Magnitude range: {magnitudes.min():.1f} - {magnitudes.max():.1f}")
    
    def __len__(self):
        return len(self.tensor_indices)
    
    def __getitem__(self, idx):
        # Get the actual tensor index
        tensor_idx = self.tensor_indices[idx]
        
        # Load tensor data
        with h5py.File(self.hdf5_path, 'r') as f:
            tensor = f['scalogram_tensor'][tensor_idx]
        
        # Convert to torch tensor
        tensor = torch.from_numpy(tensor.astype(np.float32))
        
        # Get geophysical features
        kp_dst = torch.from_numpy(self.geophysical_features[idx].astype(np.float32))
        
        # Prepare targets
        targets = {
            'binary': torch.tensor(self.targets['binary'][idx], dtype=torch.float32),
            'magnitude_class': torch.tensor(self.targets['magnitude_class'][idx], dtype=torch.long),
            'distance': torch.tensor(self.targets['distance'][idx], dtype=torch.float32),
            'geophysical': kp_dst  # [Kp, Dst]
        }
        
        return tensor, targets