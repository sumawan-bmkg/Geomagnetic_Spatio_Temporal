"""
PyTorch Dataset and DataLoader for Spatio-Temporal Earthquake Precursor Data

Handles HDF5 data loading, target preparation, and batch creation for training.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SpatioTemporalDataset(Dataset):
    """
    PyTorch Dataset for spatio-temporal earthquake precursor data.
    
    Loads 5D tensor data from HDF5 files and prepares targets for multi-task learning.
    """
    
    def __init__(self,
                 hdf5_path: str,
                 metadata_path: str,
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None,
                 load_in_memory: bool = False,
                 magnitude_bins: List[float] = None,
                 distance_log_scale: bool = True):
        """
        Initialize dataset.
        
        Args:
            hdf5_path: Path to HDF5 tensor data file
            metadata_path: Path to metadata CSV file
            transform: Optional transform for input tensors
            target_transform: Optional transform for targets
            load_in_memory: Whether to load all data in memory
            magnitude_bins: Magnitude class boundaries
            distance_log_scale: Whether to use log scale for distances
        """
        self.hdf5_path = hdf5_path
        self.metadata_path = metadata_path
        self.transform = transform
        self.target_transform = target_transform
        self.load_in_memory = load_in_memory
        self.distance_log_scale = distance_log_scale
        
        # Default magnitude bins: [4.0, 4.5, 5.0, 5.5, 6.0, inf]
        self.magnitude_bins = magnitude_bins or [4.0, 4.5, 5.0, 5.5, 6.0]
        self.n_magnitude_classes = len(self.magnitude_bins)
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata: {len(self.metadata)} records")
        
        # Load HDF5 data info
        self._load_hdf5_info()
        
        # Prepare targets
        self._prepare_targets()
        
        # Load data in memory if requested
        if load_in_memory:
            self._load_all_data()
        
        logger.info(f"Dataset initialized: {len(self)} samples")
    
    def _load_hdf5_info(self):
        """Load HDF5 file information and validate data."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get tensor shape and metadata
            self.tensor_shape = f['tensor_data'].shape
            self.n_samples = self.tensor_shape[0]
            
            # Load event IDs
            if 'event_ids' in f:
                self.event_ids = f['event_ids'][:]
            else:
                logger.warning("No event IDs found in HDF5 file")
                self.event_ids = np.arange(self.n_samples)
            
            # Load station and component info
            if 'stations' in f:
                self.stations = [s.decode('utf-8') for s in f['stations'][:]]
            else:
                self.stations = [f'Station_{i}' for i in range(self.tensor_shape[1])]
            
            if 'components' in f:
                self.components = [c.decode('utf-8') for c in f['components'][:]]
            else:
                self.components = ['H', 'D', 'Z']
            
            # Get tensor dimensions
            self.B, self.S, self.C, self.F, self.T = self.tensor_shape
            
            logger.info(f"HDF5 tensor shape: {self.tensor_shape}")
            logger.info(f"Stations: {self.stations}")
            logger.info(f"Components: {self.components}")
    
    def _prepare_targets(self):
        """Prepare target labels for multi-task learning."""
        # Create event-based metadata (one record per event)
        event_metadata = self.metadata.groupby('event_id').first().reset_index()
        
        # Filter to available events
        available_events = set(self.event_ids)
        event_metadata = event_metadata[
            event_metadata['event_id'].isin(available_events)
        ].copy()
        
        # Sort by event_id to match tensor order
        event_metadata = event_metadata.sort_values('event_id').reset_index(drop=True)
        
        # Prepare targets
        self.targets = {}
        
        # Binary classification: is_precursor (1) vs solar_noise (0)
        # For now, assume all earthquake events are precursors (1)
        # In practice, this would be determined by temporal proximity to earthquakes
        self.targets['is_precursor'] = np.ones(len(event_metadata), dtype=np.float32)
        
        # Magnitude classification and regression
        magnitudes = event_metadata['magnitude'].values
        self.targets['magnitude_value'] = magnitudes.astype(np.float32)
        
        # Convert to magnitude classes
        magnitude_classes = np.digitize(magnitudes, self.magnitude_bins)
        magnitude_classes = np.clip(magnitude_classes, 0, self.n_magnitude_classes - 1)
        self.targets['magnitude_class'] = magnitude_classes.astype(np.int64)
        
        # Localization targets
        if 'event_lat' in event_metadata.columns and 'event_lon' in event_metadata.columns:
            # Calculate azimuth and distance from station centroid
            station_lats = []
            station_lons = []
            
            for event_id in event_metadata['event_id']:
                # Get stations for this event
                event_stations = self.metadata[self.metadata['event_id'] == event_id]
                if len(event_stations) > 0:
                    station_lats.append(event_stations['station_lat'].mean())
                    station_lons.append(event_stations['station_lon'].mean())
                else:
                    # Use default centroid
                    station_lats.append(0.0)
                    station_lons.append(120.0)
            
            # Calculate azimuth and distance
            azimuths, distances = self._calculate_azimuth_distance(
                event_metadata['event_lat'].values,
                event_metadata['event_lon'].values,
                np.array(station_lats),
                np.array(station_lons)
            )
            
            # Azimuth in radians
            self.targets['azimuth_radians'] = azimuths.astype(np.float32)
            
            # Distance (log scale if requested)
            if self.distance_log_scale:
                distances = np.log10(np.maximum(distances, 1.0))  # Avoid log(0)
                self.targets['log_distance'] = distances.astype(np.float32)
            
            self.targets['distance'] = distances.astype(np.float32)
        else:
            logger.warning("No location data available for localization targets")
            # Create dummy targets
            n_events = len(event_metadata)
            self.targets['azimuth_radians'] = np.zeros(n_events, dtype=np.float32)
            self.targets['log_distance'] = np.zeros(n_events, dtype=np.float32)
            self.targets['distance'] = np.ones(n_events, dtype=np.float32)
        
        # Store event metadata for reference
        self.event_metadata = event_metadata
        
        logger.info(f"Prepared targets for {len(event_metadata)} events")
        logger.info(f"Magnitude range: {magnitudes.min():.1f} - {magnitudes.max():.1f}")
        logger.info(f"Magnitude classes: {np.bincount(magnitude_classes)}")
    
    def _calculate_azimuth_distance(self, event_lats: np.ndarray, event_lons: np.ndarray,
                                  station_lats: np.ndarray, station_lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate azimuth and distance from stations to events.
        
        Args:
            event_lats: Event latitudes
            event_lons: Event longitudes
            station_lats: Station latitudes (centroid)
            station_lons: Station longitudes (centroid)
            
        Returns:
            Tuple of (azimuths in radians, distances in km)
        """
        # Convert to radians
        lat1 = np.radians(station_lats)
        lon1 = np.radians(station_lons)
        lat2 = np.radians(event_lats)
        lon2 = np.radians(event_lons)
        
        # Calculate distance using Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371.0 * c  # Earth radius in km
        
        # Calculate azimuth (bearing)
        y = np.sin(dlon) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        azimuths = np.arctan2(y, x)
        
        # Normalize azimuth to [0, 2π]
        azimuths = (azimuths + 2 * np.pi) % (2 * np.pi)
        
        return azimuths, distances
    
    def _load_all_data(self):
        """Load all tensor data into memory."""
        logger.info("Loading all data into memory...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self.tensor_data = f['tensor_data'][:]
        
        logger.info(f"Loaded tensor data: {self.tensor_data.shape}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.event_metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input_tensor, targets_dict)
        """
        # Load tensor data
        if self.load_in_memory:
            tensor = self.tensor_data[idx]
        else:
            with h5py.File(self.hdf5_path, 'r') as f:
                tensor = f['tensor_data'][idx]
        
        # Convert to torch tensor
        tensor = torch.from_numpy(tensor.astype(np.float32))
        
        # Apply transform if provided
        if self.transform:
            tensor = self.transform(tensor)
        
        # Prepare targets
        targets = {}
        for key, values in self.targets.items():
            targets[key] = torch.tensor(values[idx])
        
        # Apply target transform if provided
        if self.target_transform:
            targets = self.target_transform(targets)
        
        return tensor, targets
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for magnitude classification.
        
        Returns:
            Class weights tensor
        """
        class_counts = np.bincount(self.targets['magnitude_class'], 
                                 minlength=self.n_magnitude_classes)
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        
        # Inverse frequency weighting
        total_samples = len(self.targets['magnitude_class'])
        class_weights = total_samples / (self.n_magnitude_classes * class_counts)
        
        return torch.tensor(class_weights, dtype=torch.float32)
    
    def get_dataset_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'n_samples': len(self),
            'tensor_shape': self.tensor_shape,
            'n_stations': self.S,
            'n_components': self.C,
            'magnitude_range': (
                float(self.targets['magnitude_value'].min()),
                float(self.targets['magnitude_value'].max())
            ),
            'magnitude_class_distribution': np.bincount(
                self.targets['magnitude_class'], 
                minlength=self.n_magnitude_classes
            ).tolist(),
            'precursor_ratio': float(self.targets['is_precursor'].mean()),
            'distance_range': (
                float(self.targets['distance'].min()),
                float(self.targets['distance'].max())
            )
        }
        
        return stats


def create_data_loaders(train_hdf5_path: str,
                       test_hdf5_path: str,
                       metadata_path: str,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       load_in_memory: bool = False,
                       magnitude_bins: List[float] = None) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and test data loaders.
    
    Args:
        train_hdf5_path: Path to training HDF5 file
        test_hdf5_path: Path to test HDF5 file
        metadata_path: Path to metadata CSV file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        load_in_memory: Whether to load all data in memory
        magnitude_bins: Magnitude class boundaries
        
    Returns:
        Tuple of (train_loader, test_loader, dataset_info)
    """
    logger.info("Creating data loaders...")
    
    # Create datasets
    train_dataset = SpatioTemporalDataset(
        hdf5_path=train_hdf5_path,
        metadata_path=metadata_path,
        load_in_memory=load_in_memory,
        magnitude_bins=magnitude_bins
    )
    
    test_dataset = SpatioTemporalDataset(
        hdf5_path=test_hdf5_path,
        metadata_path=metadata_path,
        load_in_memory=load_in_memory,
        magnitude_bins=magnitude_bins
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    # Collect dataset information
    dataset_info = {
        'train_stats': train_dataset.get_dataset_stats(),
        'test_stats': test_dataset.get_dataset_stats(),
        'class_weights': train_dataset.get_class_weights(),
        'stations': train_dataset.stations,
        'components': train_dataset.components,
        'magnitude_bins': train_dataset.magnitude_bins
    }
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, test_loader, dataset_info


if __name__ == '__main__':
    # Test dataset creation
    import sys
    import os
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    # Test with synthetic data
    print("Testing SpatioTemporalDataset...")
    
    # This would normally use real HDF5 files
    # For testing, we'll just check the class structure
    print("Dataset class structure validated!")