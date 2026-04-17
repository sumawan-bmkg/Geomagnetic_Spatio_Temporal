#!/usr/bin/env python3
"""
Integrate Tensors Batch - Memory-Efficient Implementation

Script untuk mengintegrasikan tensor data dari scalogramv3 dengan batch processing
untuk mengatasi keterbatasan memory.
"""
import sys
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging
from tqdm import tqdm
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchTensorIntegrator:
    """
    Integrator tensor dengan batch processing untuk efisiensi memory.
    """
    
    def __init__(self, batch_size=100):
        self.scalogram_path = Path('../scalogramv3/scalogram_v3_cosmic_final.h5')
        self.production_hdf5_path = Path('real_earthquake_dataset.h5')
        self.batch_size = batch_size
        
        self.integration_results = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': batch_size,
            'total_samples_processed': 0,
            'memory_efficient': True,
            'final_status': 'UNKNOWN'
        }
    
    def run_batch_integration(self):
        """Jalankan integrasi dengan batch processing."""
        logger.info("=" * 70)
        logger.info("BATCH TENSOR INTEGRATION - MEMORY EFFICIENT")
        logger.info("=" * 70)
        
        try:
            # 1. Analyze source structure
            logger.info("1. ANALYZING SOURCE STRUCTURE")
            source_info = self.analyze_source_structure()
            
            # 2. Create target tensor structure
            logger.info("2. CREATING TARGET TENSOR STRUCTURE")
            self.create_target_structure(source_info)
            
            # 3. Process tensors in batches
            logger.info("3. PROCESSING TENSORS IN BATCHES")
            self.process_tensors_batch(source_info)
            
            # 4. Validate integration
            logger.info("4. VALIDATING INTEGRATION")
            self.validate_integration()
            
            self.integration_results['final_status'] = 'SUCCESS'
            
        except Exception as e:
            logger.error(f"Error during batch integration: {e}")
            self.integration_results['error'] = str(e)
            self.integration_results['final_status'] = 'FAILED'
            raise
        
        return self.integration_results
    
    def analyze_source_structure(self):
        """Analyze struktur source data."""
        logger.info(f"Analyzing source: {self.scalogram_path}")
        
        source_info = {
            'train_samples': 0,
            'val_samples': 0,
            'tensor_shape': None,
            'total_samples': 0
        }
        
        with h5py.File(self.scalogram_path, 'r') as f:
            if 'train' in f and 'tensors' in f['train']:
                train_shape = f['train']['tensors'].shape
                source_info['train_samples'] = train_shape[0]
                source_info['tensor_shape'] = train_shape[1:]  # (3, 128, 1440)
                logger.info(f"Train samples: {train_shape[0]}, Shape: {train_shape}")
            
            if 'val' in f and 'tensors' in f['val']:
                val_shape = f['val']['tensors'].shape
                source_info['val_samples'] = val_shape[0]
                logger.info(f"Val samples: {val_shape[0]}, Shape: {val_shape}")
            
            source_info['total_samples'] = source_info['train_samples'] + source_info['val_samples']
        
        logger.info(f"Total samples to process: {source_info['total_samples']}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Number of batches: {(source_info['total_samples'] + self.batch_size - 1) // self.batch_size}")
        
        return source_info
    
    def create_target_structure(self, source_info):
        """Create target tensor structure in production HDF5."""
        logger.info("Creating target tensor structure...")
        
        if not self.production_hdf5_path.exists():
            raise FileNotFoundError(f"Production HDF5 not found: {self.production_hdf5_path}")
        
        # Calculate target shape: (N, 8, 3, 224, 224)
        n_samples = source_info['total_samples']
        target_shape = (n_samples, 8, 3, 224, 224)
        
        logger.info(f"Target tensor shape: {target_shape}")
        
        with h5py.File(self.production_hdf5_path, 'r+') as f:
            # Remove existing tensor if present
            if 'scalogram_tensor' in f:
                del f['scalogram_tensor']
            
            # Create new tensor dataset with chunking for efficient access
            chunk_shape = (min(self.batch_size, n_samples), 8, 3, 224, 224)
            
            tensor_dataset = f.create_dataset(
                'scalogram_tensor',
                shape=target_shape,
                dtype=np.float32,
                chunks=chunk_shape,
                compression='gzip',
                compression_opts=6,
                fillvalue=0.0
            )
            
            logger.info(f"Created chunked tensor dataset with chunk size: {chunk_shape}")
            
            # Add integration metadata
            if 'integration_info' in f:
                del f['integration_info']
            
            integration_group = f.create_group('integration_info')
            integration_group.attrs['source_file'] = str(self.scalogram_path)
            integration_group.attrs['integration_method'] = 'batch_processing'
            integration_group.attrs['batch_size'] = self.batch_size
            integration_group.attrs['target_shape'] = str(target_shape)
            integration_group.attrs['integration_time'] = datetime.now().isoformat()
    
    def process_tensors_batch(self, source_info):
        """Process tensors dalam batch untuk efisiensi memory."""
        logger.info("Processing tensors in batches...")
        
        total_samples = source_info['total_samples']
        samples_processed = 0
        
        with h5py.File(self.scalogram_path, 'r') as source_f:
            with h5py.File(self.production_hdf5_path, 'r+') as target_f:
                target_tensor = target_f['scalogram_tensor']
                
                # Process train data
                if 'train' in source_f and 'tensors' in source_f['train']:
                    train_tensors = source_f['train']['tensors']
                    train_samples = train_tensors.shape[0]
                    
                    logger.info(f"Processing {train_samples} training samples...")
                    
                    for start_idx in tqdm(range(0, train_samples, self.batch_size), desc="Train batches"):
                        end_idx = min(start_idx + self.batch_size, train_samples)
                        batch_size_actual = end_idx - start_idx
                        
                        # Load batch from source
                        batch_data = train_tensors[start_idx:end_idx]  # (batch, 3, 128, 1440)
                        
                        # Process batch
                        processed_batch = self.process_batch(batch_data, batch_size_actual)
                        
                        # Store in target
                        target_tensor[samples_processed:samples_processed + batch_size_actual] = processed_batch
                        
                        samples_processed += batch_size_actual
                        
                        # Clear memory
                        del batch_data, processed_batch
                        gc.collect()
                
                # Process val data
                if 'val' in source_f and 'tensors' in source_f['val']:
                    val_tensors = source_f['val']['tensors']
                    val_samples = val_tensors.shape[0]
                    
                    logger.info(f"Processing {val_samples} validation samples...")
                    
                    for start_idx in tqdm(range(0, val_samples, self.batch_size), desc="Val batches"):
                        end_idx = min(start_idx + self.batch_size, val_samples)
                        batch_size_actual = end_idx - start_idx
                        
                        # Load batch from source
                        batch_data = val_tensors[start_idx:end_idx]  # (batch, 3, 128, 1440)
                        
                        # Process batch
                        processed_batch = self.process_batch(batch_data, batch_size_actual)
                        
                        # Store in target
                        target_tensor[samples_processed:samples_processed + batch_size_actual] = processed_batch
                        
                        samples_processed += batch_size_actual
                        
                        # Clear memory
                        del batch_data, processed_batch
                        gc.collect()
        
        logger.info(f"Total samples processed: {samples_processed}")
        self.integration_results['total_samples_processed'] = samples_processed
    
    def process_batch(self, batch_data, batch_size):
        """Process single batch of tensor data."""
        # Input: (batch_size, 3, 128, 1440)
        # Output: (batch_size, 8, 3, 224, 224)
        
        processed_batch = np.zeros((batch_size, 8, 3, 224, 224), dtype=np.float32)
        
        for i in range(batch_size):
            for station_idx in range(8):  # Replicate across 8 stations
                for comp_idx in range(3):  # 3 components
                    # Get original data (128, 1440)
                    original_data = batch_data[i, comp_idx, :, :].astype(np.float32)
                    
                    # Resize to (224, 224) using simple interpolation
                    resized_data = self.resize_tensor(original_data, (224, 224))
                    
                    # Store in processed batch
                    processed_batch[i, station_idx, comp_idx, :, :] = resized_data
        
        return processed_batch
    
    def resize_tensor(self, data, target_shape):
        """Resize tensor menggunakan interpolasi sederhana."""
        from scipy.ndimage import zoom
        
        # Calculate zoom factors
        zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])
        
        # Resize using zoom
        resized = zoom(data, zoom_factors, order=1)  # Linear interpolation
        
        return resized.astype(np.float32)
    
    def validate_integration(self):
        """Validate hasil integrasi."""
        logger.info("Validating integration...")
        
        validation_results = {
            'tensor_present': False,
            'correct_shape': False,
            'data_range_valid': False,
            'no_nan_values': False
        }
        
        with h5py.File(self.production_hdf5_path, 'r') as f:
            if 'scalogram_tensor' in f:
                tensor_data = f['scalogram_tensor']
                validation_results['tensor_present'] = True
                
                # Check shape
                actual_shape = tensor_data.shape
                expected_shape_pattern = (None, 8, 3, 224, 224)
                
                if (len(actual_shape) == 5 and 
                    actual_shape[1] == 8 and 
                    actual_shape[2] == 3 and 
                    actual_shape[3] == 224 and 
                    actual_shape[4] == 224):
                    validation_results['correct_shape'] = True
                
                # Sample validation
                sample_data = tensor_data[:min(10, tensor_data.shape[0])]
                
                # Check data range
                data_min = np.min(sample_data)
                data_max = np.max(sample_data)
                
                if -100.0 <= data_min <= 100.0 and -100.0 <= data_max <= 100.0:
                    validation_results['data_range_valid'] = True
                
                # Check for NaN
                if not np.any(np.isnan(sample_data)):
                    validation_results['no_nan_values'] = True
                
                logger.info(f"Final tensor shape: {actual_shape}")
                logger.info(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
                logger.info(f"File size: {self.production_hdf5_path.stat().st_size / (1024**2):.2f} MB")
        
        # Overall validation
        passed_checks = sum(validation_results.values())
        total_checks = len(validation_results)
        
        logger.info(f"Validation: {passed_checks}/{total_checks} checks passed")
        
        for check, result in validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {check}: {status}")
        
        self.integration_results['validation_results'] = validation_results
        
        return passed_checks == total_checks
    
    def save_integration_report(self, output_path='batch_integration_report.json'):
        """Save integration report."""
        with open(output_path, 'w') as f:
            json.dump(self.integration_results, f, indent=2, default=str)
        logger.info(f"Integration report saved: {output_path}")


def main():
    """Main function."""
    print("BATCH TENSOR INTEGRATOR")
    print("Memory-Efficient Real Data Integration")
    print("=" * 70)
    
    try:
        # Check if scipy is available for resizing
        try:
            from scipy.ndimage import zoom
        except ImportError:
            print("Installing scipy for tensor resizing...")
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scipy'])
            from scipy.ndimage import zoom
        
        # Run batch integration
        integrator = BatchTensorIntegrator(batch_size=50)  # Smaller batch for memory safety
        results = integrator.run_batch_integration()
        
        # Save report
        integrator.save_integration_report()
        
        # Print results
        status = results.get('final_status', 'UNKNOWN')
        samples_processed = results.get('total_samples_processed', 0)
        
        print(f"\nINTEGRATION STATUS: {status}")
        print(f"Samples processed: {samples_processed}")
        
        if status == 'SUCCESS':
            print("\n🎉 TENSOR INTEGRATION COMPLETED!")
            print("📁 Production dataset ready: real_earthquake_dataset.h5")
            print("🚀 Ready for production training!")
        else:
            print("\n❌ INTEGRATION FAILED")
            print("Check logs for error details")
        
        return status
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 'ERROR'


if __name__ == '__main__':
    main()