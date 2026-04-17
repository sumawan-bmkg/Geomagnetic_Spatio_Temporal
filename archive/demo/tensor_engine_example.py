"""
Example usage of TensorEngine for scalogram to tensor conversion.
Demonstrates complete workflow from scalogram data to 5D tensors with CMR.
"""
import os
import sys
import logging
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.tensor_engine import TensorEngine


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tensor_engine_example.log')
        ]
    )


def create_synthetic_scalogram_data():
    """Create synthetic scalogram data for demonstration."""
    print("\n📊 CREATING SYNTHETIC SCALOGRAM DATA...")
    
    # Create synthetic scalogram directory structure
    scalogram_dir = '../data/synthetic_scalograms'
    os.makedirs(scalogram_dir, exist_ok=True)
    
    # Primary stations
    stations = ['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI']
    components = ['H', 'D', 'Z']
    
    # Create some synthetic events
    event_ids = [35402, 47722, 9180, 35403, 35406]
    
    created_files = []
    
    for event_id in event_ids:
        for station in stations:
            # Create station directory
            station_dir = os.path.join(scalogram_dir, station)
            os.makedirs(station_dir, exist_ok=True)
            
            for component in components:
                # Generate synthetic scalogram (224x224)
                freq_bins = np.linspace(0.01, 0.1, 224)  # ULF frequencies
                time_bins = np.linspace(0, 24, 224)       # 24 hours
                
                # Create synthetic scalogram with some patterns
                F, T = np.meshgrid(freq_bins, time_bins, indexing='ij')
                
                # Base pattern
                scalogram = np.exp(-((F - 0.05)**2 / 0.01 + (T - 12)**2 / 36))
                
                # Add some noise and variations
                noise = 0.1 * np.random.randn(224, 224)
                scalogram += noise
                
                # Add station-specific and component-specific variations
                station_factor = 1.0 + 0.2 * np.sin(2 * np.pi * stations.index(station) / len(stations))
                component_factor = 1.0 + 0.1 * components.index(component)
                
                scalogram *= station_factor * component_factor
                
                # Normalize to [0, 1]
                scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min())
                
                # Save as NPZ file
                filename = f"scalogram_{station}_{component}_{event_id}.npz"
                filepath = os.path.join(station_dir, filename)
                
                np.savez_compressed(filepath, scalogram=scalogram)
                created_files.append(filepath)
    
    print(f"   Created {len(created_files)} synthetic scalogram files")
    print(f"   Directory: {scalogram_dir}")
    
    return scalogram_dir


def create_synthetic_metadata():
    """Create synthetic metadata CSV for demonstration."""
    print("\n📋 CREATING SYNTHETIC METADATA...")
    
    import pandas as pd
    
    # Create synthetic metadata matching the scalogram data
    stations = ['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI']
    event_ids = [35402, 47722, 9180, 35403, 35406]
    
    metadata_records = []
    
    for event_id in event_ids:
        for station in stations:
            record = {
                'event_id': event_id,
                'station_code': station,
                'datetime': f'2018-01-{event_ids.index(event_id)+1:02d} 12:00:00',
                'magnitude': 4.5 + np.random.rand() * 2.0,  # M4.5-6.5
                'event_lat': -5.0 + np.random.rand() * 10.0,  # Random coordinates
                'event_lon': 100.0 + np.random.rand() * 30.0,
                'station_lat': -5.0 + np.random.rand() * 10.0,
                'station_lon': 100.0 + np.random.rand() * 30.0,
                'distance_km': 50.0 + np.random.rand() * 200.0,
                'dobrovolsky_radius_km': 100.0 + np.random.rand() * 300.0,
                'split': 'train' if event_ids.index(event_id) < 3 else 'test',
                'has_scalogram_data': True,
                'data_synchronized': True
            }
            metadata_records.append(record)
    
    metadata_df = pd.DataFrame(metadata_records)
    
    # Save metadata
    metadata_path = '../data/synthetic_metadata.csv'
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"   Created metadata with {len(metadata_df)} records")
    print(f"   File: {metadata_path}")
    
    return metadata_path


def example_tensor_engine_workflow():
    """Example of complete tensor engine workflow."""
    print("=" * 80)
    print("TENSOR ENGINE WORKFLOW EXAMPLE")
    print("Scalogram to 5D Tensor Conversion with PCA-based CMR")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting tensor engine example")
    
    try:
        # Create synthetic data for demonstration
        scalogram_path = create_synthetic_scalogram_data()
        metadata_path = create_synthetic_metadata()
        
        # Initialize TensorEngine
        print(f"\n🔧 INITIALIZING TENSOR ENGINE...")
        
        engine = TensorEngine(
            scalogram_base_path=scalogram_path,
            metadata_path=metadata_path,
            target_shape=(224, 224),
            primary_stations=['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI'],
            components=['H', 'D', 'Z']
        )
        
        print(f"   ✅ TensorEngine initialized")
        print(f"   📊 Target tensor shape: (B, 8, 3, 224, 224)")
        
        # Load metadata
        print(f"\n📋 LOADING METADATA...")
        metadata = engine.load_metadata()
        print(f"   ✅ Loaded {len(metadata)} metadata records")
        
        # Build tensor dataset
        print(f"\n🏗️  BUILDING TENSOR DATASET...")
        
        # Get unique event IDs (limit for demo)
        event_ids = metadata['event_id'].unique()[:3]  # First 3 events for demo
        print(f"   Processing {len(event_ids)} events: {event_ids}")
        
        tensor_data = engine.build_tensor_dataset(event_ids)
        
        if tensor_data is not None:
            print(f"   ✅ Created tensor dataset: {tensor_data.shape}")
            print(f"   📈 Data statistics:")
            print(f"      Mean: {tensor_data.mean():.4f}")
            print(f"      Std:  {tensor_data.std():.4f}")
            print(f"      Min:  {tensor_data.min():.4f}")
            print(f"      Max:  {tensor_data.max():.4f}")
        else:
            print(f"   ❌ Failed to create tensor dataset")
            return
        
        # Apply PCA-based CMR
        print(f"\n🔬 APPLYING PCA-BASED COMMON MODE REJECTION...")
        
        cmr_tensor = engine.apply_pca_cmr(tensor_data)
        
        print(f"   ✅ CMR processing completed")
        print(f"   📊 CMR tensor shape: {cmr_tensor.shape}")
        
        # Display CMR analysis
        if engine.cmr_components:
            print(f"\n📈 CMR ANALYSIS RESULTS:")
            for component, variance in engine.cmr_components['explained_variance'].items():
                print(f"   {component} component - PC1 explains {variance:.1%} of variance")
        
        # Analyze CMR effectiveness
        print(f"\n🔍 ANALYZING CMR EFFECTIVENESS...")
        
        cmr_analysis = engine.analyze_cmr_effectiveness(tensor_data, cmr_tensor)
        
        print(f"   📊 Noise Reduction Analysis:")
        for component, reduction in cmr_analysis['noise_reduction'].items():
            print(f"      {component}: {reduction:.1%} variance reduction")
        
        print(f"   🗺️  Spatial Correlation Analysis:")
        for component, corr_data in cmr_analysis['spatial_correlation'].items():
            orig_corr = corr_data['original_mean_correlation']
            cmr_corr = corr_data['cmr_mean_correlation']
            reduction = corr_data['correlation_reduction']
            print(f"      {component}: {orig_corr:.3f} → {cmr_corr:.3f} (Δ{reduction:+.3f})")
        
        # Save to HDF5
        print(f"\n💾 SAVING TO HDF5 FORMAT...")
        
        output_dir = '../outputs/tensor_datasets'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save original tensor
        original_path = os.path.join(output_dir, 'example_original.h5')
        engine.save_to_hdf5(original_path, tensor_data, {
            'description': 'Example original tensor dataset',
            'n_events': len(event_ids)
        })
        
        # Save CMR tensor
        cmr_path = os.path.join(output_dir, 'example_cmr.h5')
        engine.save_to_hdf5(cmr_path, cmr_tensor, {
            'description': 'Example CMR-processed tensor dataset',
            'n_events': len(event_ids)
        })
        
        # Get file sizes
        orig_size = os.path.getsize(original_path) / (1024**2)
        cmr_size = os.path.getsize(cmr_path) / (1024**2)
        
        print(f"   ✅ Original tensor saved: {original_path} ({orig_size:.1f} MB)")
        print(f"   ✅ CMR tensor saved: {cmr_path} ({cmr_size:.1f} MB)")
        
        # Test loading from HDF5
        print(f"\n🔄 TESTING HDF5 LOADING...")
        
        loaded_data = engine.load_from_hdf5(cmr_path)
        
        print(f"   ✅ Successfully loaded tensor: {loaded_data['tensor_data'].shape}")
        print(f"   📋 Metadata keys: {list(loaded_data['metadata'].keys())}")
        print(f"   🏷️  Stations: {loaded_data['stations']}")
        print(f"   🏷️  Components: {loaded_data['components']}")
        
        # Verify data integrity
        if np.allclose(loaded_data['tensor_data'], cmr_tensor):
            print(f"   ✅ Data integrity verified")
        else:
            print(f"   ⚠️  Data integrity check failed")
        
        print(f"\n" + "="*80)
        print("✅ TENSOR ENGINE WORKFLOW COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📊 SUMMARY:")
        print(f"   📥 Input: {len(event_ids)} events × 8 stations × 3 components")
        print(f"   📤 Output: 5D tensor ({tensor_data.shape})")
        print(f"   🔬 Processing: PCA-based Common Mode Rejection applied")
        print(f"   💾 Storage: HDF5 format for efficient ML training")
        print(f"   📁 Files: {output_dir}")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Use HDF5 files for machine learning training")
        print(f"   2. Implement deep learning models for precursor detection")
        print(f"   3. Evaluate CMR effectiveness on real scalogram data")
        print(f"   4. Scale up processing for complete dataset")
        
        logger.info("Tensor engine example completed successfully")
        
        return {
            'original_tensor': tensor_data,
            'cmr_tensor': cmr_tensor,
            'cmr_analysis': cmr_analysis,
            'saved_files': {
                'original': original_path,
                'cmr': cmr_path
            }
        }
        
    except Exception as e:
        logger.error(f"Tensor engine example failed: {e}")
        print(f"\n❌ Example failed with error: {e}")
        raise


def demonstrate_tensor_operations():
    """Demonstrate tensor operations and data structure."""
    print(f"\n" + "="*60)
    print("TENSOR OPERATIONS DEMONSTRATION")
    print("="*60)
    
    # Create example tensor
    B, S, C, F, T = 2, 8, 3, 224, 224
    tensor = np.random.randn(B, S, C, F, T).astype(np.float32)
    
    print(f"\n📊 5D TENSOR STRUCTURE:")
    print(f"   Shape: {tensor.shape}")
    print(f"   B (Batch): {B} events")
    print(f"   S (Stations): {S} stations")
    print(f"   C (Components): {C} components (H, D, Z)")
    print(f"   F (Frequency): {F} frequency bins")
    print(f"   T (Time): {T} time bins")
    print(f"   Memory: {tensor.nbytes / (1024**2):.1f} MB")
    
    print(f"\n🔍 TENSOR OPERATIONS:")
    
    # Station-wise operations
    station_mean = tensor.mean(axis=(0, 2, 3, 4))  # Mean across batch, components, freq, time
    print(f"   Station means: {station_mean}")
    
    # Component-wise operations
    component_mean = tensor.mean(axis=(0, 1, 3, 4))  # Mean across batch, stations, freq, time
    print(f"   Component means: {component_mean}")
    
    # Frequency-wise operations
    freq_profile = tensor.mean(axis=(0, 1, 2, 4))  # Mean across batch, stations, components, time
    print(f"   Frequency profile shape: {freq_profile.shape}")
    
    # Time-wise operations
    time_profile = tensor.mean(axis=(0, 1, 2, 3))  # Mean across batch, stations, components, freq
    print(f"   Time profile shape: {time_profile.shape}")
    
    print(f"\n🎯 INDEXING EXAMPLES:")
    print(f"   Single event: tensor[0] → {tensor[0].shape}")
    print(f"   Single station: tensor[:, 0] → {tensor[:, 0].shape}")
    print(f"   Single component: tensor[:, :, 0] → {tensor[:, :, 0].shape}")
    print(f"   Specific scalogram: tensor[0, 0, 0] → {tensor[0, 0, 0].shape}")


if __name__ == '__main__':
    # Run demonstrations
    demonstrate_tensor_operations()
    
    # Run main workflow example
    try:
        results = example_tensor_engine_workflow()
    except Exception as e:
        print(f"Example failed: {e}")
        sys.exit(1)