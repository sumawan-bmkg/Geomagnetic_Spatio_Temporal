#!/usr/bin/env python3
"""
Physics-Informed Production Training Recovery Script

This script implements the complete recovery pipeline:
1. Bug fix (subset data removal)
2. Physics-informed processing (Dobrovolsky filter)
3. Large event augmentation
4. Solar storm validation setup
5. Full production training execution

Usage:
    python run_physics_informed_training.py --config configs/physics_informed_config.yaml --experiment final_certified_run
"""

import sys
import os
from pathlib import Path
import logging
import argparse
import time
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from physics_informed_processor import PhysicsInformedProcessor
from run_production_train import ProductionTrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('physics_informed_recovery.log')
    ]
)
logger = logging.getLogger(__name__)


class PhysicsInformedRecovery:
    """
    Complete recovery pipeline for physics-informed production training.
    """
    
    def __init__(self):
        """Initialize recovery pipeline."""
        self.start_time = time.time()
        self.processor = PhysicsInformedProcessor()
        self.stats = {}
        
        logger.info("Physics-Informed Recovery Pipeline initialized")
    
    def verify_bug_fix(self):
        """Verify that the subset data bug has been fixed."""
        logger.info("=== PHASE 1: VERIFYING BUG FIX ===")
        
        # Check if the production training script has been fixed
        script_path = Path('run_production_train.py')
        if not script_path.exists():
            raise FileNotFoundError("Production training script not found")
        
        # Read the script and check for subset limitations
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for problematic patterns
        problematic_patterns = [
            "[:100]",  # Train subset
            "[:50]",   # Val subset
            "for dry run"
        ]
        
        issues_found = []
        for pattern in problematic_patterns:
            if pattern in content:
                issues_found.append(pattern)
        
        if issues_found:
            logger.error(f"BUG FIX VERIFICATION FAILED!")
            logger.error(f"Found problematic patterns: {issues_found}")
            logger.error("Please remove all subset limitations from run_production_train.py")
            return False
        
        logger.info("✅ Bug fix verification PASSED - No subset limitations found")
        return True
    
    def process_physics_informed_dataset(self):
        """Apply physics-informed processing to the dataset."""
        logger.info("=== PHASE 2: PHYSICS-INFORMED PROCESSING ===")
        
        # Define paths
        input_path = 'outputs/corrective_actions/certified_spatio_dataset.h5'
        output_path = 'physics_informed_dataset.h5'
        
        # Check if input exists
        if not Path(input_path).exists():
            logger.error(f"Input dataset not found: {input_path}")
            logger.info("Attempting to use alternative dataset...")
            
            # Try alternative paths
            alternative_paths = [
                'real_earthquake_dataset.h5',
                'certified_spatio_dataset.h5',
                'outputs/production_training/real_earthquake_dataset.h5'
            ]
            
            input_path = None
            for alt_path in alternative_paths:
                if Path(alt_path).exists():
                    input_path = alt_path
                    logger.info(f"Using alternative dataset: {input_path}")
                    break
            
            if input_path is None:
                raise FileNotFoundError("No suitable input dataset found")
        
        # Process the dataset
        logger.info(f"Processing dataset: {input_path} -> {output_path}")
        processing_stats = self.processor.process_dataset(input_path, output_path)
        
        self.stats['processing'] = processing_stats
        
        logger.info("✅ Physics-informed processing completed")
        logger.info(f"Processing statistics: {processing_stats}")
        
        return output_path
    
    def validate_dataset_integrity(self, dataset_path: str):
        """Validate the processed dataset integrity."""
        logger.info("=== PHASE 3: DATASET INTEGRITY VALIDATION ===")
        
        import h5py
        
        try:
            with h5py.File(dataset_path, 'r') as f:
                # Check structure
                required_groups = ['train', 'val']
                for group in required_groups:
                    if group not in f:
                        raise ValueError(f"Missing required group: {group}")
                
                # Check data sizes
                train_size = f['train']['tensors'].shape[0]
                val_size = f['val']['tensors'].shape[0]
                total_size = train_size + val_size
                
                logger.info(f"Dataset validation results:")
                logger.info(f"  Train samples: {train_size}")
                logger.info(f"  Val samples: {val_size}")
                logger.info(f"  Total samples: {total_size}")
                
                # Calculate expected batch counts
                batch_size = 8
                train_batches = train_size // batch_size
                val_batches = val_size // batch_size
                total_batches = train_batches + val_batches
                
                logger.info(f"Expected batch counts (batch_size={batch_size}):")
                logger.info(f"  Train batches per epoch: {train_batches}")
                logger.info(f"  Val batches per epoch: {val_batches}")
                logger.info(f"  Total batches per epoch: {total_batches}")
                
                # Verify we have sufficient data (should be >> 12 batches)
                if total_batches < 100:
                    logger.warning(f"Dataset seems small: {total_batches} batches per epoch")
                    logger.warning("Expected > 1000 batches for full dataset")
                else:
                    logger.info("✅ Dataset size validation PASSED")
                
                # Check processing statistics if available
                if 'processing_stats' in f:
                    stats_group = f['processing_stats']
                    logger.info("Processing statistics:")
                    for key in stats_group.attrs:
                        value = stats_group.attrs[key]
                        logger.info(f"  {key}: {value}")
                
                self.stats['dataset_validation'] = {
                    'train_samples': train_size,
                    'val_samples': val_size,
                    'total_samples': total_size,
                    'train_batches_per_epoch': train_batches,
                    'val_batches_per_epoch': val_batches,
                    'total_batches_per_epoch': total_batches
                }
                
                return True
                
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def estimate_training_time(self, total_samples: int):
        """Estimate training time based on dataset size."""
        logger.info("=== PHASE 4: TRAINING TIME ESTIMATION ===")
        
        # Training parameters
        batch_size = 8
        total_epochs = 25 + 30 + 35  # Stage 1 + Stage 2 + Stage 3
        batches_per_epoch = total_samples // batch_size
        total_batches = batches_per_epoch * total_epochs
        
        # Time estimates (CPU-based)
        seconds_per_batch_cpu = 3.0  # Conservative estimate for CPU
        estimated_seconds = total_batches * seconds_per_batch_cpu
        estimated_hours = estimated_seconds / 3600
        
        logger.info(f"Training time estimation:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Batches per epoch: {batches_per_epoch}")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Total batches: {total_batches}")
        logger.info(f"  Estimated time: {estimated_hours:.1f} hours")
        
        if estimated_hours < 2:
            logger.warning("⚠️  Estimated time seems too short - possible data issue")
        elif estimated_hours > 20:
            logger.warning("⚠️  Estimated time is very long - consider GPU acceleration")
        else:
            logger.info("✅ Training time estimate looks reasonable")
        
        self.stats['time_estimation'] = {
            'total_samples': total_samples,
            'batches_per_epoch': batches_per_epoch,
            'total_epochs': total_epochs,
            'total_batches': total_batches,
            'estimated_hours': estimated_hours
        }
        
        return estimated_hours
    
    def run_production_training(self, dataset_path: str, config_path: str, experiment_name: str):
        """Execute the full production training."""
        logger.info("=== PHASE 5: PRODUCTION TRAINING EXECUTION ===")
        
        # Create training pipeline
        pipeline = ProductionTrainingPipeline(
            dataset_path=dataset_path,
            station_coords_path='../awal/lokasi_stasiun.csv',
            output_dir='./outputs/physics_informed_training',
            experiment_name=experiment_name,
            config_path=config_path
        )
        
        # Run training
        logger.info("Starting production training...")
        training_start_time = time.time()
        
        try:
            results = pipeline.run_complete_pipeline()
            
            training_end_time = time.time()
            training_duration = training_end_time - training_start_time
            training_hours = training_duration / 3600
            
            logger.info(f"✅ Production training completed successfully!")
            logger.info(f"Training duration: {training_hours:.2f} hours")
            
            self.stats['training'] = {
                'duration_hours': training_hours,
                'results': results
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Production training failed: {e}")
            raise
    
    def generate_recovery_report(self):
        """Generate comprehensive recovery report."""
        logger.info("=== GENERATING RECOVERY REPORT ===")
        
        total_duration = time.time() - self.start_time
        total_hours = total_duration / 3600
        
        report = {
            'recovery_summary': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_duration_hours': total_hours,
                'status': 'COMPLETED'
            },
            'phases_completed': [
                'Bug Fix Verification',
                'Physics-Informed Processing',
                'Dataset Integrity Validation',
                'Training Time Estimation',
                'Production Training Execution'
            ],
            'statistics': self.stats
        }
        
        # Save report
        report_path = f'physics_informed_recovery_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Recovery report saved: {report_path}")
        
        # Print summary
        logger.info("🎉 PHYSICS-INFORMED RECOVERY COMPLETED!")
        logger.info(f"Total recovery time: {total_hours:.2f} hours")
        
        if 'dataset_validation' in self.stats:
            total_samples = self.stats['dataset_validation']['total_samples']
            logger.info(f"Dataset processed: {total_samples} samples")
        
        if 'training' in self.stats:
            training_hours = self.stats['training']['duration_hours']
            logger.info(f"Training completed in: {training_hours:.2f} hours")
        
        return report
    
    def run_complete_recovery(self, config_path: str = 'configs/physics_informed_config.yaml', 
                            experiment_name: str = 'final_certified_run'):
        """Run the complete recovery pipeline."""
        logger.info("🚀 STARTING PHYSICS-INFORMED RECOVERY PIPELINE")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Verify bug fix
            if not self.verify_bug_fix():
                raise RuntimeError("Bug fix verification failed")
            
            # Phase 2: Process physics-informed dataset
            dataset_path = self.process_physics_informed_dataset()
            
            # Phase 3: Validate dataset integrity
            if not self.validate_dataset_integrity(dataset_path):
                raise RuntimeError("Dataset validation failed")
            
            # Phase 4: Estimate training time
            total_samples = self.stats['dataset_validation']['total_samples']
            estimated_hours = self.estimate_training_time(total_samples)
            
            # Phase 5: Run production training
            results = self.run_production_training(dataset_path, config_path, experiment_name)
            
            # Generate final report
            report = self.generate_recovery_report()
            
            return report
            
        except Exception as e:
            logger.error(f"Recovery pipeline failed: {e}")
            raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Physics-Informed Production Training Recovery')
    parser.add_argument('--config', type=str, default='configs/physics_informed_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--experiment', type=str, default='final_certified_run',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    print("🔬 Physics-Informed Production Training Recovery")
    print("Spatio-Temporal Earthquake Precursor Detection")
    print("=" * 80)
    
    # Create and run recovery pipeline
    recovery = PhysicsInformedRecovery()
    report = recovery.run_complete_recovery(
        config_path=args.config,
        experiment_name=args.experiment
    )
    
    print(f"\n✅ Recovery completed successfully!")
    print(f"Report: {report}")


if __name__ == '__main__':
    main()