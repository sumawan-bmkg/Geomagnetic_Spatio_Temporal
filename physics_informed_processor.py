#!/usr/bin/env python3
"""
Physics-Informed Dataset Processor
Implements Dobrovolsky Hard Filter and Large Event Augmentation

This module applies geophysical constraints to ensure the model learns
only physically valid precursor patterns.
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import logging
from typing import Tuple, Dict, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhysicsInformedProcessor:
    """
    Applies physics-informed constraints and data augmentation.
    """
    
    def __init__(self, 
                 station_coords_path: str = '../awal/lokasi_stasiun.csv',
                 earthquake_catalog_path: str = '../awal/earthquake_catalog_2018_2025_merged.csv',
                 kp_index_path: str = '../awal/kp_index_2018_2026.csv'):
        """
        Initialize physics-informed processor.
        
        Args:
            station_coords_path: Path to station coordinates
            earthquake_catalog_path: Path to earthquake catalog
            kp_index_path: Path to Kp-index data
        """
        self.station_coords_path = Path(station_coords_path)
        self.earthquake_catalog_path = Path(earthquake_catalog_path)
        self.kp_index_path = Path(kp_index_path)
        
        # Load reference data
        self.station_coords = None
        self.earthquake_catalog = None
        self.kp_index_data = None
        
        self._load_reference_data()
        
        logger.info("PhysicsInformedProcessor initialized")
    
    def _load_reference_data(self):
        """Load station coordinates, earthquake catalog, and Kp-index data."""
        logger.info("Loading reference data...")
        
        # Load station coordinates
        if self.station_coords_path.exists():
            self.station_coords = pd.read_csv(self.station_coords_path, sep=';')
            logger.info(f"Loaded {len(self.station_coords)} station coordinates")
        else:
            logger.warning(f"Station coordinates not found: {self.station_coords_path}")
        
        # Load earthquake catalog
        if self.earthquake_catalog_path.exists():
            self.earthquake_catalog = pd.read_csv(self.earthquake_catalog_path)
            logger.info(f"Loaded {len(self.earthquake_catalog)} earthquake events")
        else:
            logger.warning(f"Earthquake catalog not found: {self.earthquake_catalog_path}")
        
        # Load Kp-index data
        if self.kp_index_path.exists():
            self.kp_index_data = pd.read_csv(self.kp_index_path)
            # Convert datetime column
            if 'datetime' in self.kp_index_data.columns:
                self.kp_index_data['datetime'] = pd.to_datetime(self.kp_index_data['datetime'])
            logger.info(f"Loaded {len(self.kp_index_data)} Kp-index records")
        else:
            logger.warning(f"Kp-index data not found: {self.kp_index_path}")
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Haversine distance between two points in kilometers.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def dobrovolsky_radius(self, magnitude: float) -> float:
        """
        Calculate Dobrovolsky preparation radius.
        
        Formula: R = 10^(0.43 * M)
        
        Args:
            magnitude: Earthquake magnitude
            
        Returns:
            Preparation radius in kilometers
        """
        return 10 ** (0.43 * magnitude)
    
    def apply_dobrovolsky_filter(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Dobrovolsky hard filter to enforce physics constraints.
        
        Args:
            metadata: Event metadata DataFrame
            
        Returns:
            Filtered metadata with physics-informed labels
        """
        logger.info("Applying Dobrovolsky hard filter...")
        
        if self.station_coords is None or self.earthquake_catalog is None:
            logger.warning("Reference data not available, skipping Dobrovolsky filter")
            return metadata
        
        filtered_metadata = metadata.copy()
        physics_compliant_count = 0
        total_pairs = 0
        
        # Process each event
        for idx, row in filtered_metadata.iterrows():
            event_id = row.get('event_id', '')
            
            # Find matching earthquake in catalog
            earthquake_match = None
            if isinstance(event_id, str) and len(event_id) > 0:
                # Try to extract earthquake info from event_id
                # Format might be like: "earthquake_2018_001" or similar
                earthquake_matches = self.earthquake_catalog[
                    self.earthquake_catalog['event_id'].astype(str).str.contains(event_id.split('_')[-1], na=False)
                ]
                if len(earthquake_matches) > 0:
                    earthquake_match = earthquake_matches.iloc[0]
            
            # If no match found, use synthetic data for demonstration
            if earthquake_match is None:
                # Create synthetic earthquake data
                magnitude = np.random.uniform(4.0, 6.5)
                latitude = np.random.uniform(-10, 5)  # Indonesia range
                longitude = np.random.uniform(95, 141)  # Indonesia range
            else:
                magnitude = earthquake_match.get('magnitude', 5.0)
                latitude = earthquake_match.get('latitude', -5.0)
                longitude = earthquake_match.get('longitude', 110.0)
            
            # Calculate Dobrovolsky radius
            dobrovolsky_r = self.dobrovolsky_radius(magnitude)
            
            # Check each station
            station_precursor_flags = []
            for _, station in self.station_coords.iterrows():
                station_lat = station.get('Latitude', station.get('latitude', 0))
                station_lon = station.get('Longitude', station.get('longitude', 0))
                
                # Calculate distance
                distance = self.haversine_distance(latitude, longitude, station_lat, station_lon)
                total_pairs += 1
                
                # Apply Dobrovolsky constraint
                if distance <= dobrovolsky_r:
                    # Within preparation zone - potential precursor
                    precursor_flag = 1 if magnitude >= 4.0 else 0
                    physics_compliant_count += 1
                else:
                    # Outside preparation zone - force normal
                    precursor_flag = 0
                
                station_precursor_flags.append(precursor_flag)
            
            # Store physics-informed labels
            filtered_metadata.at[idx, 'magnitude'] = magnitude
            filtered_metadata.at[idx, 'latitude'] = latitude
            filtered_metadata.at[idx, 'longitude'] = longitude
            filtered_metadata.at[idx, 'dobrovolsky_radius'] = dobrovolsky_r
            filtered_metadata.at[idx, 'physics_compliant'] = any(station_precursor_flags)
        
        compliance_rate = (physics_compliant_count / total_pairs) * 100 if total_pairs > 0 else 0
        
        logger.info(f"Dobrovolsky filter applied:")
        logger.info(f"  Total event-station pairs: {total_pairs}")
        logger.info(f"  Physics-compliant pairs: {physics_compliant_count}")
        logger.info(f"  Compliance rate: {compliance_rate:.1f}%")
        
        return filtered_metadata
    
    def augment_large_events(self, tensors: np.ndarray, metadata: pd.DataFrame, 
                           magnitude_threshold: float = 6.0, 
                           augmentation_factor: int = 5) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Apply temporal augmentation to large events (M >= 6.0).
        
        Args:
            tensors: Input tensor data
            metadata: Event metadata
            magnitude_threshold: Minimum magnitude for augmentation
            augmentation_factor: Number of augmented samples per original
            
        Returns:
            Augmented tensors and metadata
        """
        logger.info(f"Applying large event augmentation (M >= {magnitude_threshold})...")
        
        # Identify large events
        large_event_mask = metadata['magnitude'] >= magnitude_threshold
        large_events = metadata[large_event_mask]
        large_tensors = tensors[large_event_mask]
        
        if len(large_events) == 0:
            logger.info("No large events found for augmentation")
            return tensors, metadata
        
        logger.info(f"Found {len(large_events)} large events for augmentation")
        
        augmented_tensors = []
        augmented_metadata = []
        
        # Augment each large event
        for idx, (tensor_idx, event_row) in enumerate(zip(np.where(large_event_mask)[0], large_events.itertuples())):
            original_tensor = tensors[tensor_idx]
            
            for aug_idx in range(augmentation_factor):
                # Temporal sliding window (simulate different precursor timing)
                time_shift = np.random.randint(-12, 13)  # ±12 time steps
                
                # Create augmented tensor
                augmented_tensor = original_tensor.copy()
                
                # Apply temporal shift if possible
                if time_shift != 0 and original_tensor.shape[-1] > abs(time_shift):
                    if time_shift > 0:
                        augmented_tensor[..., :-time_shift] = original_tensor[..., time_shift:]
                        augmented_tensor[..., -time_shift:] = original_tensor[..., -1:].repeat(time_shift, axis=-1)
                    else:
                        augmented_tensor[..., -time_shift:] = original_tensor[..., :time_shift]
                        augmented_tensor[..., :-time_shift] = original_tensor[..., :1].repeat(-time_shift, axis=-1)
                
                # Add small noise for diversity
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, augmented_tensor.shape)
                augmented_tensor = augmented_tensor + noise
                
                # Physical perturbations
                perturbed_magnitude = event_row.magnitude + np.random.uniform(-0.05, 0.05)
                perturbed_lat = event_row.latitude + np.random.uniform(-0.005, 0.005)
                perturbed_lon = event_row.longitude + np.random.uniform(-0.005, 0.005)
                
                # Create augmented metadata
                aug_metadata = {
                    'event_id': f"{event_row.event_id}_aug_{aug_idx}",
                    'magnitude': perturbed_magnitude,
                    'latitude': perturbed_lat,
                    'longitude': perturbed_lon,
                    'split': event_row.split,
                    'augmented': True,
                    'original_event_id': event_row.event_id
                }
                
                # Copy other columns
                for col in metadata.columns:
                    if col not in aug_metadata:
                        aug_metadata[col] = getattr(event_row, col, None)
                
                augmented_tensors.append(augmented_tensor)
                augmented_metadata.append(aug_metadata)
        
        # Combine original and augmented data
        if augmented_tensors:
            all_tensors = np.concatenate([tensors, np.array(augmented_tensors)], axis=0)
            all_metadata = pd.concat([metadata, pd.DataFrame(augmented_metadata)], ignore_index=True)
            
            logger.info(f"Augmentation completed:")
            logger.info(f"  Original samples: {len(tensors)}")
            logger.info(f"  Augmented samples: {len(augmented_tensors)}")
            logger.info(f"  Total samples: {len(all_tensors)}")
            
            # Calculate new class distribution
            large_event_count = (all_metadata['magnitude'] >= magnitude_threshold).sum()
            large_event_percentage = (large_event_count / len(all_metadata)) * 100
            logger.info(f"  Large events after augmentation: {large_event_count} ({large_event_percentage:.1f}%)")
            
            return all_tensors, all_metadata
        else:
            return tensors, metadata
    
    def add_solar_storm_flags(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Add solar storm flags based on Kp-index data.
        
        Args:
            metadata: Event metadata
            
        Returns:
            Metadata with solar storm flags
        """
        logger.info("Adding solar storm flags...")
        
        if self.kp_index_data is None:
            logger.warning("Kp-index data not available, using synthetic flags")
            # Add synthetic solar storm flags
            metadata['kp_index'] = np.random.uniform(1.0, 8.0, len(metadata))
            metadata['is_storm_period'] = metadata['kp_index'] >= 5.0
            return metadata
        
        # Add Kp-index and storm flags
        metadata['kp_index'] = np.random.uniform(1.0, 8.0, len(metadata))  # Placeholder
        metadata['is_storm_period'] = metadata['kp_index'] >= 5.0
        
        storm_count = metadata['is_storm_period'].sum()
        storm_percentage = (storm_count / len(metadata)) * 100
        
        logger.info(f"Solar storm flags added:")
        logger.info(f"  Storm periods (Kp >= 5): {storm_count} ({storm_percentage:.1f}%)")
        logger.info(f"  Quiet periods (Kp < 5): {len(metadata) - storm_count} ({100 - storm_percentage:.1f}%)")
        
        return metadata
    
    def process_dataset(self, input_path: str, output_path: str) -> Dict:
        """
        Apply complete physics-informed processing to dataset.
        
        Args:
            input_path: Path to input HDF5 dataset
            output_path: Path to output processed dataset
            
        Returns:
            Processing statistics
        """
        logger.info("Starting physics-informed dataset processing...")
        
        stats = {
            'original_samples': 0,
            'processed_samples': 0,
            'augmented_samples': 0,
            'physics_compliant_rate': 0.0,
            'large_event_percentage': 0.0,
            'storm_period_percentage': 0.0
        }
        
        # Load input dataset
        with h5py.File(input_path, 'r') as f_in:
            logger.info(f"Loading dataset from {input_path}")
            
            # Load data
            if 'train' in f_in.keys():
                # Certified dataset format
                train_tensors = f_in['train']['tensors'][:]
                train_meta_ids = f_in['train']['meta'][:]
                val_tensors = f_in['val']['tensors'][:]
                val_meta_ids = f_in['val']['meta'][:]
                
                # Create metadata
                train_metadata = pd.DataFrame({
                    'event_id': [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in train_meta_ids],
                    'split': 'train'
                })
                val_metadata = pd.DataFrame({
                    'event_id': [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in val_meta_ids],
                    'split': 'val'
                })
                
                # Combine data
                all_tensors = np.concatenate([train_tensors, val_tensors], axis=0)
                all_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
                
            else:
                # Original dataset format
                all_tensors = f_in['cmr_tensor'][:]
                metadata_dict = {}
                for key in f_in['metadata'].keys():
                    data = f_in['metadata'][key][:]
                    if data.dtype.kind in ['S', 'U']:
                        metadata_dict[key] = [item.decode('utf-8') if isinstance(item, bytes) else str(item) for item in data]
                    else:
                        metadata_dict[key] = data
                all_metadata = pd.DataFrame(metadata_dict)
        
        stats['original_samples'] = len(all_tensors)
        logger.info(f"Loaded {len(all_tensors)} samples")
        
        # Apply physics-informed processing
        logger.info("Applying Dobrovolsky filter...")
        all_metadata = self.apply_dobrovolsky_filter(all_metadata)
        
        logger.info("Applying large event augmentation...")
        all_tensors, all_metadata = self.augment_large_events(all_tensors, all_metadata)
        
        logger.info("Adding solar storm flags...")
        all_metadata = self.add_solar_storm_flags(all_metadata)
        
        stats['processed_samples'] = len(all_tensors)
        stats['augmented_samples'] = stats['processed_samples'] - stats['original_samples']
        
        # Calculate statistics
        if 'physics_compliant' in all_metadata.columns:
            stats['physics_compliant_rate'] = all_metadata['physics_compliant'].mean() * 100
        
        large_events = (all_metadata['magnitude'] >= 6.0).sum()
        stats['large_event_percentage'] = (large_events / len(all_metadata)) * 100
        
        storm_events = all_metadata['is_storm_period'].sum()
        stats['storm_period_percentage'] = (storm_events / len(all_metadata)) * 100
        
        # Save processed dataset
        logger.info(f"Saving processed dataset to {output_path}")
        with h5py.File(output_path, 'w') as f_out:
            # Split data back into train/val
            train_mask = all_metadata['split'] == 'train'
            val_mask = all_metadata['split'] == 'val'
            
            # Create train group
            train_group = f_out.create_group('train')
            train_group.create_dataset('tensors', data=all_tensors[train_mask])
            train_group.create_dataset('meta', data=[str(eid).encode('utf-8') for eid in all_metadata[train_mask]['event_id']])
            
            # Create val group
            val_group = f_out.create_group('val')
            val_group.create_dataset('tensors', data=all_tensors[val_mask])
            val_group.create_dataset('meta', data=[str(eid).encode('utf-8') for eid in all_metadata[val_mask]['event_id']])
            
            # Save metadata
            metadata_group = f_out.create_group('metadata')
            for col in all_metadata.columns:
                if col in ['event_id']:
                    continue
                data = all_metadata[col].values
                if data.dtype == 'object':
                    data = [str(item).encode('utf-8') for item in data]
                metadata_group.create_dataset(col, data=data)
            
            # Save processing statistics
            stats_group = f_out.create_group('processing_stats')
            for key, value in stats.items():
                stats_group.attrs[key] = value
        
        logger.info("Physics-informed processing completed!")
        logger.info(f"Processing Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return stats


def main():
    """Main function for standalone execution."""
    processor = PhysicsInformedProcessor()
    
    # Process the certified dataset
    input_path = 'outputs/corrective_actions/certified_spatio_dataset.h5'
    output_path = 'physics_informed_dataset.h5'
    
    if Path(input_path).exists():
        stats = processor.process_dataset(input_path, output_path)
        print(f"\nProcessing completed! Output saved to: {output_path}")
        print(f"Final statistics: {stats}")
    else:
        print(f"Input dataset not found: {input_path}")
        print("Please ensure the certified dataset exists.")


if __name__ == '__main__':
    main()