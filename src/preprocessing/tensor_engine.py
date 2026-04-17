"""
Tensor Engine Module for Spatio-Temporal Earthquake Precursor Analysis

This module converts scalogram data into 5D tensors with PCA-based Common Mode Rejection (CMR).
Tensor format: (B, S=8, C=3, F=224, T=224)
- B: Batch size
- S: Stations (8 primary stations)
- C: Components (H, D, Z or X, Y, Z)
- F: Frequency bins (224)
- T: Time bins (224)

Features:
- Scalogram data loading and preprocessing
- 5D tensor construction
- PCA-based Common Mode Rejection for noise reduction
- HDF5 export for efficient training
"""
import numpy as np
import pandas as pd
import h5py
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob

logger = logging.getLogger(__name__)


class TensorEngine:
    """
    Tensor Engine for converting scalogram data to 5D tensors with CMR.
    
    Handles scalogram loading, tensor construction, PCA-based noise reduction,
    and HDF5 export for machine learning pipelines.
    """
    
    def __init__(self, 
                 scalogram_base_path: str,
                 metadata_path: str,
                 target_shape: Tuple[int, int] = (224, 224),
                 primary_stations: List[str] = None,
                 components: List[str] = None):
        """
        Initialize TensorEngine.
        
        Args:
            scalogram_base_path: Base path to scalogram data
            metadata_path: Path to master metadata CSV
            target_shape: Target (F, T) shape for scalograms
            primary_stations: List of primary station codes
            components: List of components ['H', 'D', 'Z'] or ['X', 'Y', 'Z']
        """
        self.scalogram_base_path = Path(scalogram_base_path)
        self.metadata_path = metadata_path
        self.target_shape = target_shape  # (F=224, T=224)
        
        # Default 8 primary stations (can be customized)
        self.primary_stations = primary_stations or [
            'ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI'
        ]
        
        # Default components (H, D, Z)
        self.components = components or ['H', 'D', 'Z']
        
        # Tensor dimensions
        self.n_stations = len(self.primary_stations)  # S = 8
        self.n_components = len(self.components)      # C = 3
        self.n_freq = target_shape[0]                # F = 224
        self.n_time = target_shape[1]                # T = 224
        
        # Data containers
        self.metadata = None
        self.scalogram_cache = {}
        self.tensor_data = None
        self.cmr_components = None
        
        # PCA for CMR
        self.pca_models = {}  # One PCA per component
        self.scalers = {}     # One scaler per component
        
        logger.info(f"TensorEngine initialized:")
        logger.info(f"  Target shape: {self.target_shape}")
        logger.info(f"  Stations ({self.n_stations}): {self.primary_stations}")
        logger.info(f"  Components ({self.n_components}): {self.components}")
        logger.info(f"  Tensor shape: (B, {self.n_stations}, {self.n_components}, {self.n_freq}, {self.n_time})")
    
    def load_metadata(self) -> pd.DataFrame:
        """Load master metadata from CSV file."""
        logger.info(f"Loading metadata from: {self.metadata_path}")
        
        self.metadata = pd.read_csv(self.metadata_path)
        
        # Filter for primary stations only
        self.metadata = self.metadata[
            self.metadata['station_code'].isin(self.primary_stations)
        ].copy()
        
        logger.info(f"Loaded {len(self.metadata)} metadata records for primary stations")
        
        # Display station distribution
        station_counts = self.metadata['station_code'].value_counts()
        logger.info("Station distribution:")
        for station, count in station_counts.items():
            logger.info(f"  {station}: {count} records")
        
        return self.metadata
    
    def find_scalogram_files(self, event_id: int, station_code: str) -> List[str]:
        """
        Find scalogram files for specific event and station.
        
        Args:
            event_id: Event ID
            station_code: Station code
            
        Returns:
            List of scalogram file paths
        """
        # Search patterns for scalogram files
        patterns = [
            f"**/{station_code}/**/*{event_id}*scalogram*.png",
            f"**/{station_code}/**/*{event_id}*scalogram*.npz",
            f"**/*{station_code}*{event_id}*scalogram*.png",
            f"**/*{station_code}*{event_id}*scalogram*.npz",
            f"**/scalogram*{station_code}*{event_id}*.png",
            f"**/scalogram*{station_code}*{event_id}*.npz",
            f"**/{event_id}/**/*{station_code}*scalogram*.png",
            f"**/{event_id}/**/*{station_code}*scalogram*.npz"
        ]
        
        found_files = []
        for pattern in patterns:
            search_path = self.scalogram_base_path / pattern
            files = glob.glob(str(search_path), recursive=True)
            found_files.extend(files)
        
        # Remove duplicates and sort
        found_files = sorted(list(set(found_files)))
        
        return found_files
    
    def load_scalogram_image(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load scalogram from image file.
        
        Args:
            file_path: Path to scalogram image
            
        Returns:
            Scalogram array or None if loading fails
        """
        try:
            # Load image
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not load image: {file_path}")
                return None
            
            # Resize to target shape
            img_resized = cv2.resize(img, self.target_shape[::-1])  # cv2 uses (width, height)
            
            # Normalize to [0, 1]
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            return img_normalized
            
        except Exception as e:
            logger.error(f"Error loading scalogram image {file_path}: {e}")
            return None
    
    def load_scalogram_npz(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load scalogram from NPZ file.
        
        Args:
            file_path: Path to scalogram NPZ file
            
        Returns:
            Scalogram array or None if loading fails
        """
        try:
            # Load NPZ file
            data = np.load(file_path)
            
            # Try common key names for scalogram data
            scalogram_keys = ['scalogram', 'zh_ratio_power', 'power', 'data']
            scalogram = None
            
            for key in scalogram_keys:
                if key in data:
                    scalogram = data[key]
                    break
            
            if scalogram is None:
                # Use first array if no standard key found
                keys = list(data.keys())
                if keys:
                    scalogram = data[keys[0]]
                    logger.warning(f"Using key '{keys[0]}' from NPZ file: {file_path}")
                else:
                    logger.warning(f"No data found in NPZ file: {file_path}")
                    return None
            
            # Handle different scalogram shapes
            if scalogram.ndim == 2:
                # Already 2D (F, T)
                pass
            elif scalogram.ndim == 3:
                # 3D - take first channel or average
                if scalogram.shape[0] <= 3:  # Likely (C, F, T)
                    scalogram = scalogram[0]  # Take first component
                else:  # Likely (F, T, C)
                    scalogram = scalogram[:, :, 0]  # Take first component
            else:
                logger.warning(f"Unexpected scalogram shape {scalogram.shape} in {file_path}")
                return None
            
            # Resize to target shape
            if scalogram.shape != self.target_shape:
                scalogram = cv2.resize(scalogram, self.target_shape[::-1])
            
            # Normalize (handle different value ranges)
            if scalogram.max() > 1.0:
                # Assume log scale or large values
                scalogram = np.log10(np.maximum(scalogram, 1e-10))
                scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
            
            return scalogram.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error loading scalogram NPZ {file_path}: {e}")
            return None
    
    def load_scalogram_for_event_station(self, event_id: int, station_code: str, 
                                       component: str = 'Z') -> Optional[np.ndarray]:
        """
        Load scalogram for specific event, station, and component.
        
        Args:
            event_id: Event ID
            station_code: Station code
            component: Component ('H', 'D', 'Z', 'X', 'Y')
            
        Returns:
            Scalogram array (F, T) or None if not found
        """
        # Check cache first
        cache_key = f"{event_id}_{station_code}_{component}"
        if cache_key in self.scalogram_cache:
            return self.scalogram_cache[cache_key]
        
        # Find scalogram files
        files = self.find_scalogram_files(event_id, station_code)
        
        if not files:
            logger.debug(f"No scalogram files found for event {event_id}, station {station_code}")
            return None
        
        # Try to load from files (prefer component-specific files)
        scalogram = None
        
        # First, try to find component-specific files
        component_files = [f for f in files if component.lower() in f.lower()]
        if component_files:
            files = component_files
        
        # Try loading from each file
        for file_path in files:
            if file_path.endswith('.npz'):
                scalogram = self.load_scalogram_npz(file_path)
            elif file_path.endswith(('.png', '.jpg', '.jpeg')):
                scalogram = self.load_scalogram_image(file_path)
            
            if scalogram is not None:
                break
        
        # Cache the result
        if scalogram is not None:
            self.scalogram_cache[cache_key] = scalogram
        
        return scalogram
    
    def build_tensor_for_event(self, event_id: int) -> Optional[np.ndarray]:
        """
        Build 5D tensor for a single event.
        
        Args:
            event_id: Event ID
            
        Returns:
            Tensor of shape (S, C, F, T) or None if insufficient data
        """
        # Initialize tensor
        tensor = np.zeros((self.n_stations, self.n_components, self.n_freq, self.n_time), 
                         dtype=np.float32)
        
        # Track data availability
        data_available = np.zeros((self.n_stations, self.n_components), dtype=bool)
        
        # Load scalograms for each station and component
        for s_idx, station in enumerate(self.primary_stations):
            for c_idx, component in enumerate(self.components):
                scalogram = self.load_scalogram_for_event_station(event_id, station, component)
                
                if scalogram is not None:
                    tensor[s_idx, c_idx] = scalogram
                    data_available[s_idx, c_idx] = True
                else:
                    # Fill with zeros or interpolated data
                    logger.debug(f"Missing scalogram for event {event_id}, station {station}, component {component}")
        
        # Check if we have sufficient data
        station_coverage = np.any(data_available, axis=1).sum()  # Stations with any data
        component_coverage = np.any(data_available, axis=0).sum()  # Components with any data
        
        min_stations = max(1, self.n_stations // 2)  # At least half the stations
        min_components = max(1, self.n_components // 2)  # At least half the components
        
        if station_coverage < min_stations or component_coverage < min_components:
            logger.warning(f"Insufficient data for event {event_id}: "
                         f"{station_coverage}/{self.n_stations} stations, "
                         f"{component_coverage}/{self.n_components} components")
            return None
        
        # Apply data interpolation for missing values
        tensor = self._interpolate_missing_data(tensor, data_available)
        
        return tensor
    
    def _interpolate_missing_data(self, tensor: np.ndarray, 
                                data_available: np.ndarray) -> np.ndarray:
        """
        Interpolate missing scalogram data.
        
        Args:
            tensor: Input tensor (S, C, F, T)
            data_available: Boolean mask (S, C)
            
        Returns:
            Tensor with interpolated missing data
        """
        # For each component, interpolate missing stations
        for c_idx in range(self.n_components):
            available_stations = np.where(data_available[:, c_idx])[0]
            missing_stations = np.where(~data_available[:, c_idx])[0]
            
            if len(available_stations) > 0 and len(missing_stations) > 0:
                # Use mean of available stations for missing ones
                mean_scalogram = np.mean(tensor[available_stations, c_idx], axis=0)
                
                for s_idx in missing_stations:
                    tensor[s_idx, c_idx] = mean_scalogram
        
        return tensor
    
    def build_tensor_dataset(self, event_ids: List[int] = None, 
                           max_events: int = None) -> np.ndarray:
        """
        Build tensor dataset for multiple events.
        
        Args:
            event_ids: List of event IDs (if None, use all from metadata)
            max_events: Maximum number of events to process
            
        Returns:
            Tensor dataset of shape (B, S, C, F, T)
        """
        if self.metadata is None:
            self.load_metadata()
        
        # Get unique event IDs
        if event_ids is None:
            event_ids = self.metadata['event_id'].unique()
        
        if max_events is not None:
            event_ids = event_ids[:max_events]
        
        logger.info(f"Building tensor dataset for {len(event_ids)} events...")
        
        # Build tensors
        tensors = []
        valid_event_ids = []
        
        for i, event_id in enumerate(event_ids):
            if i % 100 == 0:
                logger.info(f"Processing event {i+1}/{len(event_ids)}: {event_id}")
            
            tensor = self.build_tensor_for_event(event_id)
            
            if tensor is not None:
                tensors.append(tensor)
                valid_event_ids.append(event_id)
        
        if not tensors:
            logger.error("No valid tensors created!")
            return None
        
        # Stack into batch dimension
        self.tensor_data = np.stack(tensors, axis=0)  # (B, S, C, F, T)
        self.valid_event_ids = valid_event_ids
        
        logger.info(f"Created tensor dataset: {self.tensor_data.shape}")
        logger.info(f"Valid events: {len(valid_event_ids)}/{len(event_ids)}")
        
        return self.tensor_data
    
    def apply_pca_cmr(self, tensor_data: np.ndarray = None) -> np.ndarray:
        """
        Apply PCA-based Common Mode Rejection (CMR).
        
        Removes global signals (PC1) from each component across stations
        to reduce solar noise and enhance local precursor signals.
        
        Args:
            tensor_data: Input tensor (B, S, C, F, T) or None to use self.tensor_data
            
        Returns:
            CMR-processed tensor (B, S, C, F, T)
        """
        if tensor_data is None:
            tensor_data = self.tensor_data
        
        if tensor_data is None:
            raise ValueError("No tensor data available. Build dataset first.")
        
        logger.info("Applying PCA-based Common Mode Rejection...")
        
        B, S, C, F, T = tensor_data.shape
        cmr_tensor = tensor_data.copy()
        
        # Store CMR components for analysis
        self.cmr_components = {
            'pc1_components': {},
            'explained_variance': {},
            'global_signals': {}
        }
        
        # Apply CMR for each component separately
        for c_idx, component in enumerate(self.components):
            logger.info(f"Processing component {component} ({c_idx+1}/{C})...")
            
            # Reshape for PCA: (B*F*T, S)
            component_data = tensor_data[:, :, c_idx, :, :].reshape(B * F * T, S)
            
            # Remove NaN and infinite values
            valid_mask = np.isfinite(component_data).all(axis=1)
            if not np.any(valid_mask):
                logger.warning(f"No valid data for component {component}")
                continue
            
            valid_data = component_data[valid_mask]
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(valid_data)
            
            # Apply PCA
            pca = PCA(n_components=min(S, valid_data.shape[0]))
            pca_transformed = pca.fit_transform(scaled_data)
            
            # Store PCA model and scaler
            self.pca_models[component] = pca
            self.scalers[component] = scaler
            
            # Extract PC1 (global signal)
            pc1_component = pca.components_[0]  # First principal component
            explained_var = pca.explained_variance_ratio_[0]
            
            logger.info(f"  PC1 explains {explained_var:.1%} of variance in {component}")
            
            # Reconstruct global signal (PC1 only)
            pc1_scores = pca_transformed[:, 0:1]  # Keep 2D shape
            pc1_reconstructed = pc1_scores @ pca.components_[0:1]
            
            # Transform back to original scale
            global_signal = scaler.inverse_transform(pc1_reconstructed)
            
            # Expand global signal back to full tensor shape
            full_global_signal = np.zeros_like(component_data)
            full_global_signal[valid_mask] = global_signal
            full_global_signal = full_global_signal.reshape(B, F, T, S).transpose(0, 3, 1, 2)
            
            # Remove global signal (CMR)
            cmr_tensor[:, :, c_idx, :, :] = (
                tensor_data[:, :, c_idx, :, :] - full_global_signal
            )
            
            # Store CMR analysis results
            self.cmr_components['pc1_components'][component] = pc1_component
            self.cmr_components['explained_variance'][component] = explained_var
            self.cmr_components['global_signals'][component] = full_global_signal
        
        logger.info("PCA-based CMR completed")
        
        return cmr_tensor
    
    def save_to_hdf5(self, output_path: str, tensor_data: np.ndarray = None,
                    metadata: Dict = None, compression: str = 'gzip') -> str:
        """
        Save tensor dataset to HDF5 format for efficient training.
        
        Args:
            output_path: Output HDF5 file path
            tensor_data: Tensor data to save (B, S, C, F, T)
            metadata: Additional metadata to save
            compression: HDF5 compression method
            
        Returns:
            Path to saved HDF5 file
        """
        if tensor_data is None:
            tensor_data = self.tensor_data
        
        if tensor_data is None:
            raise ValueError("No tensor data to save")
        
        logger.info(f"Saving tensor dataset to HDF5: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Save main tensor data
            f.create_dataset('tensor_data', data=tensor_data, 
                           compression=compression, compression_opts=9)
            
            # Save tensor metadata
            f.attrs['tensor_shape'] = tensor_data.shape
            f.attrs['n_stations'] = self.n_stations
            f.attrs['n_components'] = self.n_components
            f.attrs['n_freq'] = self.n_freq
            f.attrs['n_time'] = self.n_time
            f.attrs['target_shape'] = self.target_shape
            f.attrs['creation_time'] = datetime.now().isoformat()
            
            # Save station and component information
            f.create_dataset('stations', data=[s.encode('utf-8') for s in self.primary_stations])
            f.create_dataset('components', data=[c.encode('utf-8') for c in self.components])
            
            # Save event IDs
            if hasattr(self, 'valid_event_ids'):
                f.create_dataset('event_ids', data=self.valid_event_ids)
            
            # Save CMR analysis results
            if self.cmr_components is not None:
                cmr_group = f.create_group('cmr_analysis')
                
                # Save PC1 components
                pc1_group = cmr_group.create_group('pc1_components')
                for component, pc1 in self.cmr_components['pc1_components'].items():
                    pc1_group.create_dataset(component, data=pc1)
                
                # Save explained variance
                var_group = cmr_group.create_group('explained_variance')
                for component, var in self.cmr_components['explained_variance'].items():
                    var_group.attrs[component] = var
                
                # Save global signals (compressed)
                global_group = cmr_group.create_group('global_signals')
                for component, signal in self.cmr_components['global_signals'].items():
                    global_group.create_dataset(component, data=signal, 
                                              compression=compression, compression_opts=9)
            
            # Save additional metadata
            if metadata is not None:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        meta_group.attrs[key] = value
                    elif isinstance(value, np.ndarray):
                        meta_group.create_dataset(key, data=value)
                    elif isinstance(value, (list, tuple)):
                        meta_group.create_dataset(key, data=np.array(value))
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024**2)  # MB
        logger.info(f"HDF5 file saved: {output_path} ({file_size:.1f} MB)")
        
        return output_path
    
    def load_from_hdf5(self, hdf5_path: str) -> Dict:
        """
        Load tensor dataset from HDF5 file.
        
        Args:
            hdf5_path: Path to HDF5 file
            
        Returns:
            Dictionary containing loaded data
        """
        logger.info(f"Loading tensor dataset from HDF5: {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            # Load main tensor data
            tensor_data = f['tensor_data'][:]
            
            # Load metadata
            metadata = dict(f.attrs)
            
            # Load station and component information
            stations = [s.decode('utf-8') for s in f['stations'][:]]
            components = [c.decode('utf-8') for c in f['components'][:]]
            
            # Load event IDs if available
            event_ids = f['event_ids'][:] if 'event_ids' in f else None
            
            # Load CMR analysis if available
            cmr_data = None
            if 'cmr_analysis' in f:
                cmr_group = f['cmr_analysis']
                cmr_data = {
                    'pc1_components': {},
                    'explained_variance': {},
                    'global_signals': {}
                }
                
                # Load PC1 components
                if 'pc1_components' in cmr_group:
                    pc1_group = cmr_group['pc1_components']
                    for component in pc1_group.keys():
                        cmr_data['pc1_components'][component] = pc1_group[component][:]
                
                # Load explained variance
                if 'explained_variance' in cmr_group:
                    var_group = cmr_group['explained_variance']
                    for component in var_group.attrs.keys():
                        cmr_data['explained_variance'][component] = var_group.attrs[component]
                
                # Load global signals
                if 'global_signals' in cmr_group:
                    global_group = cmr_group['global_signals']
                    for component in global_group.keys():
                        cmr_data['global_signals'][component] = global_group[component][:]
        
        result = {
            'tensor_data': tensor_data,
            'metadata': metadata,
            'stations': stations,
            'components': components,
            'event_ids': event_ids,
            'cmr_analysis': cmr_data
        }
        
        logger.info(f"Loaded tensor dataset: {tensor_data.shape}")
        
        return result
    
    def analyze_cmr_effectiveness(self, original_tensor: np.ndarray, 
                                cmr_tensor: np.ndarray) -> Dict:
        """
        Analyze the effectiveness of CMR processing.
        
        Args:
            original_tensor: Original tensor before CMR
            cmr_tensor: Tensor after CMR processing
            
        Returns:
            Dictionary with CMR analysis results
        """
        logger.info("Analyzing CMR effectiveness...")
        
        analysis = {
            'noise_reduction': {},
            'signal_preservation': {},
            'spatial_correlation': {}
        }
        
        B, S, C, F, T = original_tensor.shape
        
        for c_idx, component in enumerate(self.components):
            # Calculate noise reduction (variance reduction)
            orig_var = np.var(original_tensor[:, :, c_idx, :, :])
            cmr_var = np.var(cmr_tensor[:, :, c_idx, :, :])
            noise_reduction = (orig_var - cmr_var) / orig_var
            
            # Calculate spatial correlation reduction
            orig_corr = self._calculate_spatial_correlation(original_tensor[:, :, c_idx, :, :])
            cmr_corr = self._calculate_spatial_correlation(cmr_tensor[:, :, c_idx, :, :])
            
            analysis['noise_reduction'][component] = noise_reduction
            analysis['spatial_correlation'][component] = {
                'original_mean_correlation': orig_corr,
                'cmr_mean_correlation': cmr_corr,
                'correlation_reduction': orig_corr - cmr_corr
            }
        
        return analysis
    
    def _calculate_spatial_correlation(self, data: np.ndarray) -> float:
        """
        Calculate mean spatial correlation between stations.
        
        Args:
            data: Data array (B, S, F, T)
            
        Returns:
            Mean correlation coefficient
        """
        B, S, F, T = data.shape
        
        # Flatten spatial-temporal dimensions
        flattened = data.reshape(B, S, F * T)
        
        # Calculate correlation matrix between stations
        correlations = []
        for b in range(B):
            corr_matrix = np.corrcoef(flattened[b])
            # Extract upper triangle (excluding diagonal)
            upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            correlations.extend(upper_tri[np.isfinite(upper_tri)])
        
        return np.mean(correlations) if correlations else 0.0
    
    def visualize_cmr_results(self, original_tensor: np.ndarray, cmr_tensor: np.ndarray,
                            event_idx: int = 0, save_path: str = None) -> None:
        """
        Visualize CMR results for analysis.
        
        Args:
            original_tensor: Original tensor before CMR
            cmr_tensor: Tensor after CMR processing
            event_idx: Event index to visualize
            save_path: Path to save visualization
        """
        if self.cmr_components is None:
            logger.warning("No CMR analysis results available")
            return
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        for c_idx, component in enumerate(self.components):
            # Original scalogram (first station)
            axes[c_idx, 0].imshow(original_tensor[event_idx, 0, c_idx], 
                                 aspect='auto', cmap='jet')
            axes[c_idx, 0].set_title(f'{component} - Original')
            
            # Global signal (PC1)
            if component in self.cmr_components['global_signals']:
                global_signal = self.cmr_components['global_signals'][component]
                axes[c_idx, 1].imshow(global_signal[event_idx, 0], 
                                     aspect='auto', cmap='jet')
                axes[c_idx, 1].set_title(f'{component} - Global Signal (PC1)')
            
            # CMR result
            axes[c_idx, 2].imshow(cmr_tensor[event_idx, 0, c_idx], 
                                 aspect='auto', cmap='jet')
            axes[c_idx, 2].set_title(f'{component} - After CMR')
            
            # PC1 component weights
            if component in self.cmr_components['pc1_components']:
                pc1_weights = self.cmr_components['pc1_components'][component]
                axes[c_idx, 3].bar(range(len(pc1_weights)), pc1_weights)
                axes[c_idx, 3].set_title(f'{component} - PC1 Weights')
                axes[c_idx, 3].set_xlabel('Station Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"CMR visualization saved: {save_path}")
        
        plt.show()
    
    def process_complete_dataset(self, output_dir: str, 
                               train_test_split: bool = True,
                               apply_cmr: bool = True,
                               max_events_per_split: int = None) -> Dict[str, str]:
        """
        Process complete dataset with train/test splitting and CMR.
        
        Args:
            output_dir: Output directory for HDF5 files
            train_test_split: Whether to create separate train/test files
            apply_cmr: Whether to apply CMR processing
            max_events_per_split: Maximum events per split (for testing)
            
        Returns:
            Dictionary of saved file paths
        """
        logger.info("Starting complete dataset processing...")
        
        # Load metadata
        self.load_metadata()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        if train_test_split and 'split' in self.metadata.columns:
            # Process train and test sets separately
            for split in ['train', 'test']:
                logger.info(f"Processing {split} set...")
                
                # Get events for this split
                split_metadata = self.metadata[self.metadata['split'] == split]
                split_event_ids = split_metadata['event_id'].unique()
                
                if max_events_per_split:
                    split_event_ids = split_event_ids[:max_events_per_split]
                
                logger.info(f"{split} set: {len(split_event_ids)} events")
                
                # Build tensor dataset
                tensor_data = self.build_tensor_dataset(split_event_ids)
                
                if tensor_data is None:
                    logger.warning(f"No valid tensors for {split} set")
                    continue
                
                # Apply CMR if requested
                if apply_cmr:
                    logger.info(f"Applying CMR to {split} set...")
                    cmr_tensor = self.apply_pca_cmr(tensor_data)
                    
                    # Save both original and CMR versions
                    original_path = os.path.join(output_dir, f'{split}_original.h5')
                    cmr_path = os.path.join(output_dir, f'{split}_cmr.h5')
                    
                    # Prepare metadata
                    metadata = {
                        'split': split,
                        'n_events': len(self.valid_event_ids),
                        'processing_applied': 'original'
                    }
                    
                    self.save_to_hdf5(original_path, tensor_data, metadata)
                    saved_files[f'{split}_original'] = original_path
                    
                    # CMR metadata
                    cmr_metadata = metadata.copy()
                    cmr_metadata['processing_applied'] = 'cmr'
                    
                    self.save_to_hdf5(cmr_path, cmr_tensor, cmr_metadata)
                    saved_files[f'{split}_cmr'] = cmr_path
                    
                    # Analyze CMR effectiveness
                    cmr_analysis = self.analyze_cmr_effectiveness(tensor_data, cmr_tensor)
                    
                    # Save CMR analysis
                    analysis_path = os.path.join(output_dir, f'{split}_cmr_analysis.json')
                    with open(analysis_path, 'w') as f:
                        json.dump(cmr_analysis, f, indent=2, default=str)
                    saved_files[f'{split}_cmr_analysis'] = analysis_path
                    
                else:
                    # Save original only
                    output_path = os.path.join(output_dir, f'{split}.h5')
                    metadata = {
                        'split': split,
                        'n_events': len(self.valid_event_ids),
                        'processing_applied': 'original'
                    }
                    
                    self.save_to_hdf5(output_path, tensor_data, metadata)
                    saved_files[split] = output_path
        
        else:
            # Process all data together
            logger.info("Processing complete dataset...")
            
            # Get all events
            all_event_ids = self.metadata['event_id'].unique()
            
            if max_events_per_split:
                all_event_ids = all_event_ids[:max_events_per_split]
            
            # Build tensor dataset
            tensor_data = self.build_tensor_dataset(all_event_ids)
            
            if tensor_data is None:
                logger.error("No valid tensors created")
                return saved_files
            
            # Apply CMR if requested
            if apply_cmr:
                logger.info("Applying CMR to complete dataset...")
                cmr_tensor = self.apply_pca_cmr(tensor_data)
                
                # Save both versions
                original_path = os.path.join(output_dir, 'complete_original.h5')
                cmr_path = os.path.join(output_dir, 'complete_cmr.h5')
                
                metadata = {
                    'split': 'complete',
                    'n_events': len(self.valid_event_ids),
                    'processing_applied': 'original'
                }
                
                self.save_to_hdf5(original_path, tensor_data, metadata)
                saved_files['complete_original'] = original_path
                
                cmr_metadata = metadata.copy()
                cmr_metadata['processing_applied'] = 'cmr'
                
                self.save_to_hdf5(cmr_path, cmr_tensor, cmr_metadata)
                saved_files['complete_cmr'] = cmr_path
                
                # Analyze CMR effectiveness
                cmr_analysis = self.analyze_cmr_effectiveness(tensor_data, cmr_tensor)
                
                analysis_path = os.path.join(output_dir, 'complete_cmr_analysis.json')
                with open(analysis_path, 'w') as f:
                    json.dump(cmr_analysis, f, indent=2, default=str)
                saved_files['complete_cmr_analysis'] = analysis_path
                
            else:
                # Save original only
                output_path = os.path.join(output_dir, 'complete.h5')
                metadata = {
                    'split': 'complete',
                    'n_events': len(self.valid_event_ids),
                    'processing_applied': 'original'
                }
                
                self.save_to_hdf5(output_path, tensor_data, metadata)
                saved_files['complete'] = output_path
        
        logger.info("Complete dataset processing finished!")
        
        return saved_files


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tensor Engine for Scalogram Processing')
    parser.add_argument('--scalogram-path', required=True, help='Path to scalogram data')
    parser.add_argument('--metadata-path', required=True, help='Path to master metadata CSV')
    parser.add_argument('--output-dir', required=True, help='Output directory for HDF5 files')
    parser.add_argument('--stations', nargs='+', help='Primary station codes')
    parser.add_argument('--components', nargs='+', default=['H', 'D', 'Z'], help='Components')
    parser.add_argument('--target-shape', nargs=2, type=int, default=[224, 224], help='Target F T shape')
    parser.add_argument('--no-cmr', action='store_true', help='Disable CMR processing')
    parser.add_argument('--max-events', type=int, help='Maximum events per split (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create tensor engine
    engine = TensorEngine(
        scalogram_base_path=args.scalogram_path,
        metadata_path=args.metadata_path,
        target_shape=tuple(args.target_shape),
        primary_stations=args.stations,
        components=args.components
    )
    
    # Process complete dataset
    saved_files = engine.process_complete_dataset(
        output_dir=args.output_dir,
        train_test_split=True,
        apply_cmr=not args.no_cmr,
        max_events_per_split=args.max_events
    )
    
    print("\nProcessing completed!")
    print("Saved files:")
    for key, path in saved_files.items():
        file_size = os.path.getsize(path) / (1024**2)  # MB
        print(f"  {key}: {path} ({file_size:.1f} MB)")


if __name__ == '__main__':
    main()