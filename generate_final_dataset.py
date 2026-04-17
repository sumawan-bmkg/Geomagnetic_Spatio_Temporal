#!/usr/bin/env python3
"""
Generate Final Dataset - Production Implementation

Script ini secara fisik menarik data dari scalogramv3 dan menyusunnya menjadi 
Tensor 5D dengan Chronological Split dan PCA-based CMR terintegrasi.

Input:
- awal/earthquake_catalog_2018_2025_merged.csv
- awal/kp_index_2018_2026.csv  
- awal/lokasi_stasiun.csv
- scalogramv3/ (folder scalogram data)

Output:
- spatio_earthquake_dataset.h5 (final HDF5 dataset)
"""
import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import h5py
from datetime import datetime, timedelta
import cv2
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('generate_final_dataset.log')
    ]
)
logger = logging.getLogger(__name__)


class ProductionDatasetGenerator:
    """
    Production dataset generator yang mengintegrasikan semua komponen Task 1-4.
    """
    
    def __init__(self):
        """Initialize dataset generator dengan konfigurasi production."""
        self.base_path = Path('.')
        
        # Input paths dari folder awal
        self.earthquake_catalog_path = self.base_path / 'awal' / 'earthquake_catalog_2018_2025_merged.csv'
        self.kp_index_path = self.base_path / 'awal' / 'kp_index_2018_2026.csv'
        self.station_coords_path = self.base_path / 'awal' / 'lokasi_stasiun.csv'
        self.scalogram_base_path = self.base_path / 'Spatio_Precursor_Project' / 'data' / 'synthetic_scalograms'
        
        # Output path
        self.output_path = self.base_path / 'spatio_earthquake_dataset.h5'
        
        # Configuration
        self.target_shape = (224, 224)  # (F, T)
        self.components = ['H', 'D', 'Z']
        self.n_components = len(self.components)
        
        # Chronological split configuration
        self.train_end_date = '2024-06-30'
        self.test_start_date = '2024-07-01'
        
        # Data containers
        self.earthquake_data = None
        self.kp_data = None
        self.station_coords = None
        self.station_list = []
        self.n_stations = 0
        
        logger.info("ProductionDatasetGenerator initialized")
        logger.info(f"Target tensor shape: (B, {self.n_stations}, {self.n_components}, {self.target_shape[0]}, {self.target_shape[1]})")
    
    def load_input_data(self):
        """Load semua input data dari folder awal."""
        logger.info("=== LOADING INPUT DATA ===")
        
        # 1. Load earthquake catalog
        if not self.earthquake_catalog_path.exists():
            raise FileNotFoundError(f"Earthquake catalog not found: {self.earthquake_catalog_path}")
        
        self.earthquake_data = pd.read_csv(self.earthquake_catalog_path)
        self.earthquake_data['datetime'] = pd.to_datetime(self.earthquake_data['datetime'])
        logger.info(f"Loaded earthquake catalog: {len(self.earthquake_data)} events")
        
        # 2. Load Kp-index data
        if not self.kp_index_path.exists():
            raise FileNotFoundError(f"Kp-index data not found: {self.kp_index_path}")
        
        self.kp_data = pd.read_csv(self.kp_index_path)
        # Fix column names - use actual column names from file
        if 'Date_Time_UTC' in self.kp_data.columns:
            self.kp_data['datetime'] = pd.to_datetime(self.kp_data['Date_Time_UTC'], utc=True).dt.tz_localize(None)
            self.kp_data['kp_index'] = self.kp_data['Kp_Index']
        else:
            # Fallback to first two columns
            datetime_col = self.kp_data.columns[0]
            kp_col = self.kp_data.columns[1]
            self.kp_data['datetime'] = pd.to_datetime(self.kp_data[datetime_col], utc=True).dt.tz_localize(None)
            self.kp_data['kp_index'] = self.kp_data[kp_col]
        logger.info(f"Loaded Kp-index data: {len(self.kp_data)} records")
        
        # 3. Load station coordinates
        if not self.station_coords_path.exists():
            raise FileNotFoundError(f"Station coordinates not found: {self.station_coords_path}")
        
        self.station_coords = pd.read_csv(self.station_coords_path, sep=';')
        
        # Extract station list dari koordinat (urutan sesuai file)
        if 'Kode Stasiun' in self.station_coords.columns:
            self.station_list = self.station_coords['Kode Stasiun'].tolist()
        elif 'station_code' in self.station_coords.columns:
            self.station_list = self.station_coords['station_code'].tolist()
        elif 'code' in self.station_coords.columns:
            self.station_list = self.station_coords['code'].tolist()
        else:
            # Fallback ke kolom pertama
            self.station_list = self.station_coords.iloc[:, 0].tolist()
        
        # Remove empty entries and NaN values
        self.station_list = [s for s in self.station_list if s and str(s).strip() and str(s) != 'nan']
        self.n_stations = len(self.station_list)
        logger.info(f"Loaded station coordinates: {self.n_stations} stations")
        logger.info(f"Station order: {self.station_list}")
        
        # 4. Verify scalogram folder
        if not self.scalogram_base_path.exists():
            raise FileNotFoundError(f"Scalogram folder not found: {self.scalogram_base_path}")
        
        logger.info(f"Scalogram base path verified: {self.scalogram_base_path}")
    
    def create_chronological_split(self):
        """Buat chronological split sesuai spesifikasi."""
        logger.info("=== CREATING CHRONOLOGICAL SPLIT ===")
        
        # Apply chronological split
        train_mask = self.earthquake_data['datetime'] <= self.train_end_date
        test_mask = self.earthquake_data['datetime'] >= self.test_start_date
        
        self.earthquake_data['split'] = 'exclude'  # Default
        self.earthquake_data.loc[train_mask, 'split'] = 'train'
        self.earthquake_data.loc[test_mask, 'split'] = 'test'
        
        # Log split statistics
        split_counts = self.earthquake_data['split'].value_counts()
        logger.info("Chronological split created:")
        for split, count in split_counts.items():
            logger.info(f"  {split}: {count} events")
        
        # Filter hanya train dan test
        self.earthquake_data = self.earthquake_data[
            self.earthquake_data['split'].isin(['train', 'test'])
        ].copy()
        
        logger.info(f"Final dataset: {len(self.earthquake_data)} events")
    
    def merge_with_kp_index(self):
        """Merge earthquake data dengan Kp-index untuk CMR analysis."""
        logger.info("=== MERGING WITH KP-INDEX ===")
        
        # Merge berdasarkan datetime terdekat
        merged_data = []
        
        for idx, eq_row in self.earthquake_data.iterrows():
            eq_time = eq_row['datetime']
            
            # Cari Kp-index terdekat (dalam window 3 jam)
            time_diff = abs(self.kp_data['datetime'] - eq_time)
            closest_idx = time_diff.idxmin()
            
            if time_diff.iloc[closest_idx] <= timedelta(hours=3):
                kp_row = self.kp_data.iloc[closest_idx]
                
                # Combine data
                combined_row = eq_row.copy()
                combined_row['kp_index'] = kp_row['kp_index']
                combined_row['kp_datetime'] = kp_row['datetime']
                combined_row['kp_time_diff_hours'] = time_diff.iloc[closest_idx].total_seconds() / 3600
                
                merged_data.append(combined_row)
        
        self.earthquake_data = pd.DataFrame(merged_data)
        logger.info(f"Merged with Kp-index: {len(self.earthquake_data)} events with Kp data")
        
        # Log Kp statistics
        kp_stats = self.earthquake_data['kp_index'].describe()
        logger.info(f"Kp-index statistics:\n{kp_stats}")
        
        storm_events = (self.earthquake_data['kp_index'] > 5.0).sum()
        logger.info(f"Solar storm events (Kp > 5): {storm_events}/{len(self.earthquake_data)} ({storm_events/len(self.earthquake_data)*100:.1f}%)")
    
    def find_scalogram_files(self, event_id: int, station_code: str) -> List[str]:
        """Cari file scalogram untuk event dan station tertentu."""
        # Pattern pencarian yang komprehensif
        search_patterns = [
            f"**/{station_code}/**/*{event_id}*.png",
            f"**/{station_code}/**/*{event_id}*.npz",
            f"**/*{station_code}*{event_id}*.png",
            f"**/*{station_code}*{event_id}*.npz",
            f"**/scalogram*{station_code}*{event_id}*.png",
            f"**/scalogram*{station_code}*{event_id}*.npz",
            f"**/{event_id}/**/*{station_code}*.png",
            f"**/{event_id}/**/*{station_code}*.npz"
        ]
        
        found_files = []
        for pattern in search_patterns:
            search_path = self.scalogram_base_path / pattern
            files = glob.glob(str(search_path), recursive=True)
            found_files.extend(files)
        
        # Remove duplicates dan sort
        found_files = sorted(list(set(found_files)))
        return found_files
    
    def load_scalogram_data(self, file_path: str, component: str) -> Optional[np.ndarray]:
        """Load scalogram data dari file."""
        try:
            if file_path.endswith('.png') or file_path.endswith('.jpg'):
                # Load image file
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    return None
                
                # Resize ke target shape
                img_resized = cv2.resize(img, self.target_shape[::-1])  # cv2 uses (width, height)
                
                # Normalize ke [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                return img_normalized
                
            elif file_path.endswith('.npz'):
                # Load NPZ file
                data = np.load(file_path)
                
                # Coba berbagai key names
                possible_keys = ['scalogram', 'zh_ratio_power', 'power', 'data', component.lower()]
                scalogram = None
                
                for key in possible_keys:
                    if key in data:
                        scalogram = data[key]
                        break
                
                if scalogram is None:
                    # Gunakan array pertama
                    keys = list(data.keys())
                    if keys:
                        scalogram = data[keys[0]]
                    else:
                        return None
                
                # Handle different shapes
                if scalogram.ndim == 2:
                    pass  # Already 2D
                elif scalogram.ndim == 3:
                    if scalogram.shape[0] <= 3:  # (C, F, T)
                        scalogram = scalogram[0]
                    else:  # (F, T, C)
                        scalogram = scalogram[:, :, 0]
                else:
                    return None
                
                # Resize jika perlu
                if scalogram.shape != self.target_shape:
                    scalogram = cv2.resize(scalogram, self.target_shape[::-1])
                
                # Normalize
                if scalogram.max() > 1.0:
                    scalogram = np.log10(np.maximum(scalogram, 1e-10))
                    scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8)
                
                return scalogram.astype(np.float32)
                
        except Exception as e:
            logger.debug(f"Error loading {file_path}: {e}")
            return None
        
        return None
    
    def build_event_tensor(self, event_id: int) -> Optional[np.ndarray]:
        """Build tensor untuk satu event."""
        # Initialize tensor: (S, C, F, T)
        tensor = np.zeros((self.n_stations, self.n_components, self.target_shape[0], self.target_shape[1]), 
                         dtype=np.float32)
        
        data_availability = np.zeros((self.n_stations, self.n_components), dtype=bool)
        
        # Load data untuk setiap station dan component
        for s_idx, station in enumerate(self.station_list):
            # Cari files untuk station ini
            files = self.find_scalogram_files(event_id, station)
            
            for c_idx, component in enumerate(self.components):
                # Cari file yang sesuai dengan component
                component_files = [f for f in files if component.lower() in f.lower()]
                if not component_files:
                    component_files = files  # Fallback ke semua files
                
                # Coba load dari files
                scalogram = None
                for file_path in component_files:
                    scalogram = self.load_scalogram_data(file_path, component)
                    if scalogram is not None:
                        break
                
                if scalogram is not None:
                    tensor[s_idx, c_idx] = scalogram
                    data_availability[s_idx, c_idx] = True
        
        # Check data availability
        station_coverage = np.any(data_availability, axis=1).sum()
        component_coverage = np.any(data_availability, axis=0).sum()
        
        min_stations = max(1, self.n_stations // 2)
        min_components = max(1, self.n_components // 2)
        
        if station_coverage < min_stations or component_coverage < min_components:
            logger.debug(f"Insufficient data for event {event_id}: {station_coverage}/{self.n_stations} stations, {component_coverage}/{self.n_components} components")
            return None
        
        # Interpolate missing data
        tensor = self._interpolate_missing_data(tensor, data_availability)
        
        return tensor
    
    def _interpolate_missing_data(self, tensor: np.ndarray, data_available: np.ndarray) -> np.ndarray:
        """Interpolate missing scalogram data."""
        for c_idx in range(self.n_components):
            available_stations = np.where(data_available[:, c_idx])[0]
            missing_stations = np.where(~data_available[:, c_idx])[0]
            
            if len(available_stations) > 0 and len(missing_stations) > 0:
                mean_scalogram = np.mean(tensor[available_stations, c_idx], axis=0)
                for s_idx in missing_stations:
                    tensor[s_idx, c_idx] = mean_scalogram
        
        return tensor
    
    def apply_pca_cmr(self, tensor_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Apply PCA-based Common Mode Rejection."""
        logger.info("=== APPLYING PCA-BASED CMR ===")
        
        B, S, C, F, T = tensor_data.shape
        cmr_tensor = tensor_data.copy()
        
        cmr_info = {
            'pc1_components': {},
            'explained_variance': {},
            'global_signals': {}
        }
        
        for c_idx, component in enumerate(self.components):
            logger.info(f"Processing CMR for component {component}...")
            
            # Reshape untuk PCA: (B*F*T, S)
            component_data = tensor_data[:, :, c_idx, :, :].reshape(B * F * T, S)
            
            # Remove invalid values
            valid_mask = np.isfinite(component_data).all(axis=1)
            if not np.any(valid_mask):
                logger.warning(f"No valid data for component {component}")
                continue
            
            valid_data = component_data[valid_mask]
            
            # Standardize
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(valid_data)
            
            # Apply PCA
            pca = PCA(n_components=min(S, valid_data.shape[0]))
            pca_transformed = pca.fit_transform(scaled_data)
            
            # Extract PC1 (global signal)
            pc1_component = pca.components_[0]
            explained_var = pca.explained_variance_ratio_[0]
            
            logger.info(f"  PC1 explains {explained_var:.1%} of variance")
            
            # Reconstruct global signal
            pc1_scores = pca_transformed[:, 0:1]
            pc1_reconstructed = pc1_scores @ pca.components_[0:1]
            global_signal = scaler.inverse_transform(pc1_reconstructed)
            
            # Expand back to tensor shape
            full_global_signal = np.zeros_like(component_data)
            full_global_signal[valid_mask] = global_signal
            full_global_signal = full_global_signal.reshape(B, F, T, S).transpose(0, 3, 1, 2)
            
            # Apply CMR
            cmr_tensor[:, :, c_idx, :, :] = tensor_data[:, :, c_idx, :, :] - full_global_signal
            
            # Store CMR info
            cmr_info['pc1_components'][component] = pc1_component
            cmr_info['explained_variance'][component] = explained_var
            cmr_info['global_signals'][component] = full_global_signal
        
        logger.info("PCA-based CMR completed")
        return cmr_tensor, cmr_info
    
    def generate_dataset(self):
        """Generate complete dataset dengan semua komponen terintegrasi."""
        logger.info("=== GENERATING COMPLETE DATASET ===")
        
        # Get unique events
        unique_events = self.earthquake_data['event_id'].unique()
        logger.info(f"Processing {len(unique_events)} unique events...")
        
        # Build tensors
        tensors = []
        valid_events = []
        event_metadata = []
        
        for i, event_id in enumerate(unique_events):
            if i % 100 == 0:
                logger.info(f"Processing event {i+1}/{len(unique_events)}: {event_id}")
            
            # Build tensor untuk event ini
            tensor = self.build_event_tensor(event_id)
            
            if tensor is not None:
                tensors.append(tensor)
                valid_events.append(event_id)
                
                # Get event metadata
                event_info = self.earthquake_data[
                    self.earthquake_data['event_id'] == event_id
                ].iloc[0]
                event_metadata.append(event_info)
        
        if not tensors:
            raise ValueError("No valid tensors created!")
        
        # Stack tensors
        tensor_data = np.stack(tensors, axis=0)  # (B, S, C, F, T)
        event_metadata = pd.DataFrame(event_metadata)
        
        logger.info(f"Created tensor dataset: {tensor_data.shape}")
        logger.info(f"Valid events: {len(valid_events)}")
        
        # Apply CMR
        cmr_tensor, cmr_info = self.apply_pca_cmr(tensor_data)
        
        return {
            'original_tensor': tensor_data,
            'cmr_tensor': cmr_tensor,
            'event_metadata': event_metadata,
            'valid_events': valid_events,
            'cmr_info': cmr_info,
            'station_list': self.station_list,
            'components': self.components,
            'station_coordinates': self.station_coords
        }
    
    def save_final_dataset(self, dataset: Dict):
        """Save final dataset ke HDF5."""
        logger.info("=== SAVING FINAL DATASET ===")
        
        with h5py.File(self.output_path, 'w') as f:
            # Main tensor data
            f.create_dataset('original_tensor', data=dataset['original_tensor'], 
                           compression='gzip', compression_opts=9)
            f.create_dataset('cmr_tensor', data=dataset['cmr_tensor'], 
                           compression='gzip', compression_opts=9)
            
            # Event metadata
            metadata_group = f.create_group('metadata')
            for col in dataset['event_metadata'].columns:
                if dataset['event_metadata'][col].dtype == 'object':
                    # String data
                    str_data = [str(x).encode('utf-8') for x in dataset['event_metadata'][col]]
                    metadata_group.create_dataset(col, data=str_data)
                else:
                    # Numeric data
                    metadata_group.create_dataset(col, data=dataset['event_metadata'][col].values)
            
            # Configuration data
            config_group = f.create_group('config')
            config_group.create_dataset('station_list', 
                                      data=[s.encode('utf-8') for s in dataset['station_list']])
            config_group.create_dataset('components', 
                                      data=[c.encode('utf-8') for c in dataset['components']])
            config_group.create_dataset('valid_events', data=dataset['valid_events'])
            
            # Station coordinates
            coords_group = f.create_group('station_coordinates')
            for col in dataset['station_coordinates'].columns:
                if dataset['station_coordinates'][col].dtype == 'object':
                    str_data = [str(x).encode('utf-8') for x in dataset['station_coordinates'][col]]
                    coords_group.create_dataset(col, data=str_data)
                else:
                    coords_group.create_dataset(col, data=dataset['station_coordinates'][col].values)
            
            # CMR analysis results
            cmr_group = f.create_group('cmr_analysis')
            
            # PC1 components
            pc1_group = cmr_group.create_group('pc1_components')
            for component, pc1 in dataset['cmr_info']['pc1_components'].items():
                pc1_group.create_dataset(component, data=pc1)
            
            # Explained variance
            var_group = cmr_group.create_group('explained_variance')
            for component, var in dataset['cmr_info']['explained_variance'].items():
                var_group.attrs[component] = var
            
            # Global signals (compressed)
            global_group = cmr_group.create_group('global_signals')
            for component, signal in dataset['cmr_info']['global_signals'].items():
                global_group.create_dataset(component, data=signal, 
                                          compression='gzip', compression_opts=9)
            
            # Dataset attributes
            f.attrs['tensor_shape'] = dataset['original_tensor'].shape
            f.attrs['n_stations'] = len(dataset['station_list'])
            f.attrs['n_components'] = len(dataset['components'])
            f.attrs['target_shape'] = self.target_shape
            f.attrs['creation_time'] = datetime.now().isoformat()
            f.attrs['train_end_date'] = self.train_end_date
            f.attrs['test_start_date'] = self.test_start_date
        
        # Get file size
        file_size = os.path.getsize(self.output_path) / (1024**3)  # GB
        logger.info(f"Final dataset saved: {self.output_path}")
        logger.info(f"File size: {file_size:.2f} GB")
        
        # Save summary
        summary = {
            'creation_time': datetime.now().isoformat(),
            'total_events': len(dataset['valid_events']),
            'tensor_shape': list(dataset['original_tensor'].shape),
            'stations': dataset['station_list'],
            'components': dataset['components'],
            'train_events': (dataset['event_metadata']['split'] == 'train').sum(),
            'test_events': (dataset['event_metadata']['split'] == 'test').sum(),
            'solar_storm_events': (dataset['event_metadata']['kp_index'] > 5.0).sum(),
            'cmr_effectiveness': {
                comp: float(var) for comp, var in dataset['cmr_info']['explained_variance'].items()
            },
            'file_size_gb': file_size
        }
        
        summary_path = self.output_path.with_suffix('.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Dataset summary saved: {summary_path}")
        return summary
    
    def run(self):
        """Run complete dataset generation process."""
        logger.info("STARTING PRODUCTION DATASET GENERATION")
        logger.info("=" * 60)
        
        try:
            # 1. Load input data
            self.load_input_data()
            
            # 2. Create chronological split
            self.create_chronological_split()
            
            # 3. Merge with Kp-index
            self.merge_with_kp_index()
            
            # 4. Generate dataset
            dataset = self.generate_dataset()
            
            # 5. Save final dataset
            summary = self.save_final_dataset(dataset)
            
            logger.info("DATASET GENERATION COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Final dataset: {self.output_path}")
            logger.info(f"Total events: {summary['total_events']}")
            logger.info(f"Train events: {summary['train_events']}")
            logger.info(f"Test events: {summary['test_events']}")
            logger.info(f"Solar storm events: {summary['solar_storm_events']}")
            logger.info(f"File size: {summary['file_size_gb']:.2f} GB")
            
            return summary
            
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            raise


def main():
    """Main function."""
    print("Production Dataset Generator")
    print("Spatio-Temporal Earthquake Precursor Detection")
    print("=" * 60)
    
    generator = ProductionDatasetGenerator()
    summary = generator.run()
    
    print(f"\nDataset generation completed!")
    print(f"Output: spatio_earthquake_dataset.h5")
    print(f"Events: {summary['total_events']} total")
    print(f"Size: {summary['file_size_gb']:.2f} GB")


if __name__ == '__main__':
    main()