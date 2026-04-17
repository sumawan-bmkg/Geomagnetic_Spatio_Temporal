#!/usr/bin/env python3
"""
Validate Production Dataset - Final Validation

Script validasi final untuk memastikan dataset production siap untuk training.
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionDatasetValidator:
    """
    Validator untuk dataset production final.
    """
    
    def __init__(self):
        self.dataset_path = Path('real_earthquake_dataset.h5')
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'validation_checks': {},
            'final_status': 'UNKNOWN'
        }
    
    def run_comprehensive_validation(self):
        """Jalankan validasi komprehensif."""
        logger.info("=" * 70)
        logger.info("PRODUCTION DATASET VALIDATION")
        logger.info("=" * 70)
        
        try:
            # 1. File Structure Validation
            logger.info("1. FILE STRUCTURE VALIDATION")
            logger.info("-" * 40)
            self.validate_file_structure()
            
            # 2. Tensor Data Validation
            logger.info("2. TENSOR DATA VALIDATION")
            logger.info("-" * 40)
            self.validate_tensor_data()
            
            # 3. Metadata Validation
            logger.info("3. METADATA VALIDATION")
            logger.info("-" * 40)
            self.validate_metadata()
            
            # 4. Integration Validation
            logger.info("4. INTEGRATION VALIDATION")
            logger.info("-" * 40)
            self.validate_integration()
            
            # 5. Training Readiness Check
            logger.info("5. TRAINING READINESS CHECK")
            logger.info("-" * 40)
            self.check_training_readiness()
            
            # 6. Final Assessment
            logger.info("6. FINAL ASSESSMENT")
            logger.info("-" * 40)
            self.assess_final_status()
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            self.validation_results['error'] = str(e)
            self.validation_results['final_status'] = 'ERROR'
        
        return self.validation_results
    
    def validate_file_structure(self):
        """Validate struktur file HDF5."""
        logger.info("Validating file structure...")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        file_size_mb = self.dataset_path.stat().st_size / (1024**2)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        required_groups = ['metadata', 'config', 'station_coordinates', 'kp_index']
        required_datasets = ['scalogram_tensor']
        
        structure_check = {
            'file_exists': True,
            'file_size_mb': file_size_mb,
            'required_groups_present': [],
            'required_datasets_present': [],
            'structure_valid': False
        }
        
        with h5py.File(self.dataset_path, 'r') as f:
            # Check groups
            for group in required_groups:
                if group in f:
                    structure_check['required_groups_present'].append(group)
                    logger.info(f"  ✅ Group '{group}' found")
                else:
                    logger.warning(f"  ❌ Group '{group}' missing")
            
            # Check datasets
            for dataset in required_datasets:
                if dataset in f:
                    structure_check['required_datasets_present'].append(dataset)
                    logger.info(f"  ✅ Dataset '{dataset}' found")
                else:
                    logger.warning(f"  ❌ Dataset '{dataset}' missing")
            
            # Overall structure validation
            groups_ok = len(structure_check['required_groups_present']) == len(required_groups)
            datasets_ok = len(structure_check['required_datasets_present']) == len(required_datasets)
            structure_check['structure_valid'] = groups_ok and datasets_ok
        
        self.validation_results['validation_checks']['file_structure'] = structure_check
        logger.info(f"File structure validation: {'✅ PASS' if structure_check['structure_valid'] else '❌ FAIL'}")
    
    def validate_tensor_data(self):
        """Validate tensor data quality."""
        logger.info("Validating tensor data...")
        
        tensor_check = {
            'tensor_present': False,
            'correct_shape': False,
            'correct_dtype': False,
            'data_range_valid': False,
            'no_nan_inf': False,
            'non_zero_data': False
        }
        
        with h5py.File(self.dataset_path, 'r') as f:
            if 'scalogram_tensor' in f:
                tensor_check['tensor_present'] = True
                tensor = f['scalogram_tensor']
                
                # Check shape
                shape = tensor.shape
                logger.info(f"Tensor shape: {shape}")
                
                if (len(shape) == 5 and 
                    shape[1] == 8 and 
                    shape[2] == 3 and 
                    shape[3] == 224 and 
                    shape[4] == 224):
                    tensor_check['correct_shape'] = True
                
                # Check dtype
                if tensor.dtype == np.float32:
                    tensor_check['correct_dtype'] = True
                
                # Sample data for quality checks
                sample_size = min(100, shape[0])
                sample_data = tensor[:sample_size]
                
                # Data range check
                data_min = np.min(sample_data)
                data_max = np.max(sample_data)
                logger.info(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
                
                if -1000.0 <= data_min <= 1000.0 and -1000.0 <= data_max <= 1000.0:
                    tensor_check['data_range_valid'] = True
                
                # NaN/Inf check
                if not (np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data))):
                    tensor_check['no_nan_inf'] = True
                
                # Non-zero data check
                non_zero_percentage = np.count_nonzero(sample_data) / sample_data.size * 100
                logger.info(f"Non-zero data: {non_zero_percentage:.1f}%")
                
                if non_zero_percentage > 10.0:  # At least 10% non-zero
                    tensor_check['non_zero_data'] = True
        
        self.validation_results['validation_checks']['tensor_data'] = tensor_check
        
        # Overall tensor validation
        tensor_valid = all(tensor_check.values())
        logger.info(f"Tensor data validation: {'✅ PASS' if tensor_valid else '❌ FAIL'}")
        
        for check, result in tensor_check.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check}")
    
    def validate_metadata(self):
        """Validate metadata completeness."""
        logger.info("Validating metadata...")
        
        metadata_check = {
            'metadata_group_present': False,
            'event_ids_present': False,
            'magnitudes_present': False,
            'coordinates_present': False,
            'datetime_present': False,
            'data_consistency': False
        }
        
        with h5py.File(self.dataset_path, 'r') as f:
            if 'metadata' in f:
                metadata_check['metadata_group_present'] = True
                metadata = f['metadata']
                
                # Check required metadata fields
                required_fields = ['event_id', 'magnitude', 'latitude', 'longitude', 'datetime']
                
                for field in required_fields:
                    if field in metadata:
                        field_key = f"{field}_present"
                        if field_key in metadata_check:
                            metadata_check[field_key] = True
                        logger.info(f"  ✅ {field} found ({len(metadata[field])} records)")
                    else:
                        logger.warning(f"  ❌ {field} missing")
                
                # Check data consistency
                if all([field in metadata for field in required_fields]):
                    lengths = [len(metadata[field]) for field in required_fields]
                    if len(set(lengths)) == 1:  # All same length
                        metadata_check['data_consistency'] = True
                        logger.info(f"  ✅ Data consistency: {lengths[0]} records")
                    else:
                        logger.warning(f"  ❌ Inconsistent lengths: {lengths}")
        
        self.validation_results['validation_checks']['metadata'] = metadata_check
        
        # Overall metadata validation
        metadata_valid = all(metadata_check.values())
        logger.info(f"Metadata validation: {'✅ PASS' if metadata_valid else '❌ FAIL'}")
    
    def validate_integration(self):
        """Validate integration info."""
        logger.info("Validating integration info...")
        
        integration_check = {
            'integration_info_present': False,
            'source_file_documented': False,
            'batch_processing_used': False,
            'target_shape_documented': False
        }
        
        with h5py.File(self.dataset_path, 'r') as f:
            if 'integration_info' in f:
                integration_check['integration_info_present'] = True
                integration = f['integration_info']
                
                # Check integration attributes
                required_attrs = ['source_file', 'integration_method', 'target_shape']
                
                for attr in required_attrs:
                    if attr in integration.attrs:
                        logger.info(f"  ✅ {attr}: {integration.attrs[attr]}")
                        
                        if attr == 'source_file':
                            integration_check['source_file_documented'] = True
                        elif attr == 'integration_method' and 'batch' in str(integration.attrs[attr]):
                            integration_check['batch_processing_used'] = True
                        elif attr == 'target_shape':
                            integration_check['target_shape_documented'] = True
                    else:
                        logger.warning(f"  ❌ {attr} missing")
        
        self.validation_results['validation_checks']['integration'] = integration_check
        
        # Overall integration validation
        integration_valid = all(integration_check.values())
        logger.info(f"Integration validation: {'✅ PASS' if integration_valid else '❌ FAIL'}")
    
    def check_training_readiness(self):
        """Check readiness untuk training."""
        logger.info("Checking training readiness...")
        
        training_check = {
            'tensor_shape_correct': False,
            'sufficient_samples': False,
            'memory_requirements_reasonable': False,
            'data_quality_acceptable': False
        }
        
        with h5py.File(self.dataset_path, 'r') as f:
            if 'scalogram_tensor' in f:
                tensor = f['scalogram_tensor']
                shape = tensor.shape
                
                # Shape check for training
                if len(shape) == 5 and shape[1:] == (8, 3, 224, 224):
                    training_check['tensor_shape_correct'] = True
                
                # Sample count check
                n_samples = shape[0]
                if n_samples >= 1000:  # Minimum for meaningful training
                    training_check['sufficient_samples'] = True
                
                logger.info(f"Training samples available: {n_samples}")
                
                # Memory requirements (rough estimate)
                memory_per_batch_gb = (64 * 8 * 3 * 224 * 224 * 4) / (1024**3)  # 64 batch size, float32
                if memory_per_batch_gb < 8.0:  # Reasonable for 8GB GPU
                    training_check['memory_requirements_reasonable'] = True
                
                logger.info(f"Estimated memory per batch (64): {memory_per_batch_gb:.2f} GB")
                
                # Data quality check
                sample_data = tensor[:min(10, n_samples)]
                if not (np.any(np.isnan(sample_data)) or np.any(np.isinf(sample_data))):
                    training_check['data_quality_acceptable'] = True
        
        self.validation_results['validation_checks']['training_readiness'] = training_check
        
        # Overall training readiness
        training_ready = all(training_check.values())
        logger.info(f"Training readiness: {'✅ READY' if training_ready else '❌ NOT READY'}")
        
        for check, result in training_check.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check}")
    
    def assess_final_status(self):
        """Assess final validation status."""
        logger.info("Assessing final validation status...")
        
        # Count passed validations
        all_checks = self.validation_results['validation_checks']
        total_categories = len(all_checks)
        passed_categories = 0
        
        for category, checks in all_checks.items():
            if isinstance(checks, dict):
                category_passed = all(checks.values())
                if category_passed:
                    passed_categories += 1
                
                logger.info(f"{category}: {'✅ PASS' if category_passed else '❌ FAIL'}")
        
        # Final status determination
        if passed_categories == total_categories:
            final_status = 'PRODUCTION_READY'
        elif passed_categories >= total_categories * 0.8:
            final_status = 'MOSTLY_READY'
        else:
            final_status = 'NOT_READY'
        
        self.validation_results['final_status'] = final_status
        self.validation_results['validation_summary'] = {
            'total_categories': total_categories,
            'passed_categories': passed_categories,
            'success_rate': passed_categories / total_categories * 100
        }
        
        logger.info("=" * 60)
        logger.info(f"FINAL VALIDATION STATUS: {final_status}")
        logger.info("=" * 60)
        logger.info(f"Validation success rate: {passed_categories}/{total_categories} ({passed_categories/total_categories*100:.1f}%)")
        
        if final_status == 'PRODUCTION_READY':
            logger.info("🎉 Dataset is READY for production training!")
        elif final_status == 'MOSTLY_READY':
            logger.info("⚠️  Dataset is mostly ready with minor issues")
        else:
            logger.info("❌ Dataset requires fixes before training")
        
        return final_status
    
    def save_validation_report(self, output_path='production_dataset_validation_report.json'):
        """Save validation report."""
        with open(output_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
        logger.info(f"Validation report saved: {output_path}")


def main():
    """Main function."""
    print("PRODUCTION DATASET VALIDATOR")
    print("Final Validation for Training Readiness")
    print("=" * 70)
    
    try:
        # Run validation
        validator = ProductionDatasetValidator()
        results = validator.run_comprehensive_validation()
        
        # Save report
        validator.save_validation_report()
        
        # Print final status
        status = results.get('final_status', 'UNKNOWN')
        summary = results.get('validation_summary', {})
        
        print(f"\nFINAL STATUS: {status}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
        
        if status == 'PRODUCTION_READY':
            print("\n🎉 DATASET VALIDATION COMPLETE!")
            print("🚀 Ready to proceed with production training!")
        else:
            print("\n⚠️  Validation completed with issues")
            print("📋 Check validation report for details")
        
        return status
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 'ERROR'


if __name__ == '__main__':
    main()