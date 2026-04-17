#!/usr/bin/env python3
"""
Tensor Engine Runner Script
Converts scalogram data to 5D tensors with PCA-based Common Mode Rejection.
"""
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.tensor_engine import TensorEngine


def main():
    """Main function to run tensor engine with real data."""
    print("=" * 80)
    print("SPATIO-TEMPORAL EARTHQUAKE PRECURSOR TENSOR ENGINE")
    print("Scalogram to 5D Tensor Conversion with PCA-based CMR")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'tensor_engine_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting tensor engine processing")
    
    # Define data paths
    data_paths = {
        'scalogram_base': '../scalogramv3',  # Adjust path as needed
        'metadata': 'outputs/data_audit/master_metadata.csv',
        'output_dir': 'outputs/tensor_datasets'
    }
    
    # Verify data files
    print("\n📋 VERIFYING DATA PATHS...")
    missing_paths = []
    existing_paths = []
    
    for key, path in data_paths.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                file_size = os.path.getsize(path) / 1024  # KB
                existing_paths.append(f"✅ {key}: {path} ({file_size:.1f} KB)")
            else:
                existing_paths.append(f"✅ {key}: {path} (directory)")
        else:
            missing_paths.append(f"❌ {key}: {path}")
    
    for path_info in existing_paths:
        print(f"   {path_info}")
    
    if missing_paths:
        print(f"\n⚠️  Missing paths:")
        for path_info in missing_paths:
            print(f"   {path_info}")
        
        # Check if critical files are missing
        if not os.path.exists(data_paths['metadata']):
            print(f"\n❌ Critical file missing: metadata")
            print("Please run data auditor first: python run_data_audit.py")
            return False
        
        if not os.path.exists(data_paths['scalogram_base']):
            print(f"\n⚠️  Scalogram directory not found")
            print("Will create synthetic data for demonstration...")
            return run_with_synthetic_data()
    
    try:
        # Initialize TensorEngine
        print(f"\n🔧 INITIALIZING TENSOR ENGINE...")
        
        # Primary 8 stations (adjust based on your data)
        primary_stations = ['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI']
        
        engine = TensorEngine(
            scalogram_base_path=data_paths['scalogram_base'],
            metadata_path=data_paths['metadata'],
            target_shape=(224, 224),
            primary_stations=primary_stations,
            components=['H', 'D', 'Z']
        )
        
        print(f"   ✅ TensorEngine initialized")
        print(f"   🎯 Target tensor shape: (B, 8, 3, 224, 224)")
        print(f"   🏷️  Primary stations: {primary_stations}")
        
        # Create output directory
        os.makedirs(data_paths['output_dir'], exist_ok=True)
        print(f"   📁 Output directory: {data_paths['output_dir']}")
        
        # Process complete dataset
        print(f"\n🚀 PROCESSING COMPLETE DATASET...")
        
        saved_files = engine.process_complete_dataset(
            output_dir=data_paths['output_dir'],
            train_test_split=True,
            apply_cmr=True,
            max_events_per_split=50  # Limit for initial testing
        )
        
        # Display results
        print_results_summary(saved_files)
        
        logger.info("Tensor engine processing completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Tensor engine processing failed: {e}")
        print(f"\n❌ PROCESSING FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_with_synthetic_data():
    """Run tensor engine with synthetic data for demonstration."""
    print(f"\n🎭 RUNNING WITH SYNTHETIC DATA...")
    
    try:
        # Run the example script
        example_script = os.path.join('examples', 'tensor_engine_example.py')
        if os.path.exists(example_script):
            print(f"   Executing: {example_script}")
            os.system(f'python {example_script}')
            return True
        else:
            print(f"   ❌ Example script not found: {example_script}")
            return False
            
    except Exception as e:
        print(f"   ❌ Synthetic data processing failed: {e}")
        return False


def print_results_summary(saved_files):
    """Print summary of processing results."""
    print(f"\n" + "="*80)
    print("🎉 TENSOR ENGINE PROCESSING COMPLETED!")
    print("="*80)
    
    if not saved_files:
        print(f"\n⚠️  No files were saved")
        return
    
    # Categorize files
    train_files = {k: v for k, v in saved_files.items() if 'train' in k}
    test_files = {k: v for k, v in saved_files.items() if 'test' in k}
    analysis_files = {k: v for k, v in saved_files.items() if 'analysis' in k}
    
    # Display file summary
    print(f"\n📁 OUTPUT FILES:")
    
    total_size = 0
    
    if train_files:
        print(f"\n   🚂 TRAINING SET:")
        for key, path in train_files.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024**2)  # MB
                total_size += file_size
                print(f"      {key}: {os.path.basename(path)} ({file_size:.1f} MB)")
    
    if test_files:
        print(f"\n   🧪 TEST SET:")
        for key, path in test_files.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / (1024**2)  # MB
                total_size += file_size
                print(f"      {key}: {os.path.basename(path)} ({file_size:.1f} MB)")
    
    if analysis_files:
        print(f"\n   📊 ANALYSIS FILES:")
        for key, path in analysis_files.items():
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"      {key}: {os.path.basename(path)} ({file_size:.1f} KB)")
    
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"   📦 Total files: {len(saved_files)}")
    print(f"   💾 Total size: {total_size:.1f} MB")
    print(f"   📁 Output directory: outputs/tensor_datasets/")
    
    # Display tensor information
    print(f"\n🎯 TENSOR SPECIFICATIONS:")
    print(f"   📊 Shape: (B, S=8, C=3, F=224, T=224)")
    print(f"   🏷️  Dimensions:")
    print(f"      B: Batch size (number of events)")
    print(f"      S: Stations (8 primary stations)")
    print(f"      C: Components (H, D, Z)")
    print(f"      F: Frequency bins (224, ULF range)")
    print(f"      T: Time bins (224, temporal resolution)")
    
    print(f"\n🔬 PCA-BASED CMR PROCESSING:")
    print(f"   ✅ Global signal (PC1) extraction")
    print(f"   ✅ Common mode rejection applied")
    print(f"   ✅ Local precursor signal enhancement")
    print(f"   ✅ Solar noise reduction")
    
    print(f"\n💾 HDF5 FORMAT BENEFITS:")
    print(f"   ⚡ Fast loading for ML training")
    print(f"   🗜️  Efficient compression")
    print(f"   📋 Embedded metadata")
    print(f"   🔄 Cross-platform compatibility")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Load HDF5 files in ML training pipeline")
    print(f"   2. Implement spatio-temporal deep learning models")
    print(f"   3. Evaluate precursor detection performance")
    print(f"   4. Compare original vs CMR-processed results")
    print(f"   5. Scale up processing for complete dataset")
    
    # Show example loading code
    print(f"\n💡 EXAMPLE LOADING CODE:")
    print(f"   ```python")
    print(f"   import h5py")
    print(f"   import numpy as np")
    print(f"   ")
    print(f"   # Load tensor data")
    print(f"   with h5py.File('train_cmr.h5', 'r') as f:")
    print(f"       tensor_data = f['tensor_data'][:]  # Shape: (B, 8, 3, 224, 224)")
    print(f"       stations = [s.decode() for s in f['stations'][:]]")
    print(f"       components = [c.decode() for c in f['components'][:]]")
    print(f"   ```")


def show_tensor_structure_info():
    """Show detailed information about tensor structure."""
    print(f"\n" + "="*60)
    print("📊 5D TENSOR STRUCTURE DETAILS")
    print("="*60)
    
    print(f"\n🎯 TENSOR DIMENSIONS: (B, S=8, C=3, F=224, T=224)")
    
    print(f"\n📦 DIMENSION BREAKDOWN:")
    print(f"   B (Batch): Variable number of earthquake events")
    print(f"   S (Stations): 8 primary geomagnetic stations")
    print(f"      - ALR: Alor")
    print(f"      - TND: Ternate")
    print(f"      - PLU: Palu")
    print(f"      - GTO: Gorontalo")
    print(f"      - LWK: Luwuk")
    print(f"      - GSI: Gunung Sitoli")
    print(f"      - LWA: Liwa")
    print(f"      - SMI: Sorong")
    
    print(f"\n   C (Components): 3 geomagnetic field components")
    print(f"      - H: Horizontal intensity (nT)")
    print(f"      - D: Declination (degrees)")
    print(f"      - Z: Vertical intensity (nT)")
    
    print(f"\n   F (Frequency): 224 frequency bins")
    print(f"      - Range: 0.01 - 0.1 Hz (ULF band)")
    print(f"      - Resolution: ~0.0004 Hz per bin")
    print(f"      - Focus: Earthquake precursor frequencies")
    
    print(f"\n   T (Time): 224 time bins")
    print(f"      - Range: 24 hours")
    print(f"      - Resolution: ~6.4 minutes per bin")
    print(f"      - Coverage: Complete daily cycle")
    
    print(f"\n🔬 PCA-BASED CMR PROCESS:")
    print(f"   1. Extract PC1 from 8 stations per component")
    print(f"   2. PC1 represents global signal (solar noise)")
    print(f"   3. Subtract PC1 from original signals")
    print(f"   4. Result: Enhanced local precursor signals")
    
    print(f"\n💾 MEMORY REQUIREMENTS:")
    # Calculate memory for different batch sizes
    element_size = 4  # float32
    base_size = 8 * 3 * 224 * 224 * element_size  # bytes per event
    
    batch_sizes = [1, 10, 100, 1000]
    for batch_size in batch_sizes:
        total_size = batch_size * base_size
        mb_size = total_size / (1024**2)
        gb_size = total_size / (1024**3)
        
        if gb_size >= 1:
            print(f"   B={batch_size:4d}: {gb_size:.2f} GB")
        else:
            print(f"   B={batch_size:4d}: {mb_size:.1f} MB")


if __name__ == '__main__':
    # Show tensor structure information
    show_tensor_structure_info()
    
    # Run main processing
    success = main()
    sys.exit(0 if success else 1)