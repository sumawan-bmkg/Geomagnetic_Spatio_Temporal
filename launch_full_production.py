#!/usr/bin/env python3
"""
Full Production Training Launch

This script launches the complete production training with:
- Full certified dataset (9,156 samples)
- Physics-informed constraints
- Progressive 3-stage training
- Memory-efficient processing
- Comprehensive monitoring
"""

import sys
import os
from pathlib import Path
import logging
import time
import psutil
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('full_production_training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def log_system_info():
    """Log system information."""
    logger.info("=== SYSTEM INFORMATION ===")
    
    # Memory info
    memory = psutil.virtual_memory()
    logger.info(f"Total Memory: {memory.total / (1024**3):.1f} GB")
    logger.info(f"Available Memory: {memory.available / (1024**3):.1f} GB")
    logger.info(f"Memory Usage: {memory.percent:.1f}%")
    
    # CPU info
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    # Disk info
    disk = psutil.disk_usage('.')
    logger.info(f"Disk Space: {disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB")


def verify_prerequisites():
    """Verify all prerequisites are met."""
    logger.info("=== VERIFYING PREREQUISITES ===")
    
    # Check dataset
    dataset_path = 'outputs/corrective_actions/certified_spatio_dataset.h5'
    if not Path(dataset_path).exists():
        logger.error(f"Certified dataset not found: {dataset_path}")
        return False
    
    dataset_size = Path(dataset_path).stat().st_size / (1024**3)
    logger.info(f"Dataset found: {dataset_path} ({dataset_size:.2f} GB)")
    
    # Check station coordinates
    coords_path = '../awal/lokasi_stasiun.csv'
    if not Path(coords_path).exists():
        logger.error(f"Station coordinates not found: {coords_path}")
        return False
    
    logger.info(f"Station coordinates found: {coords_path}")
    
    # Check config
    config_path = 'configs/physics_informed_config.yaml'
    if not Path(config_path).exists():
        logger.error(f"Config not found: {config_path}")
        return False
    
    logger.info(f"Config found: {config_path}")
    
    logger.info("All prerequisites verified!")
    return True


def main():
    """Main training function."""
    print("🚀 FULL PRODUCTION TRAINING LAUNCH")
    print("=" * 60)
    print("Physics-Informed Earthquake Precursor Detection")
    print("Dataset: 9,156 samples (7,456 train + 1,700 val)")
    print("Expected Duration: 8-12 hours")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Log system info
        log_system_info()
        
        # Verify prerequisites
        if not verify_prerequisites():
            logger.error("Prerequisites check failed")
            return 1
        
        # Import training pipeline
        logger.info("Importing training components...")
        from run_production_train import ProductionTrainingPipeline
        
        # Create experiment name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'full_production_{timestamp}'
        
        logger.info(f"Experiment name: {experiment_name}")
        
        # Create pipeline
        logger.info("Creating production training pipeline...")
        pipeline = ProductionTrainingPipeline(
            dataset_path='outputs/corrective_actions/certified_spatio_dataset.h5',
            station_coords_path='../awal/lokasi_stasiun.csv',
            output_dir='./outputs/full_production',
            experiment_name=experiment_name,
            config_path='configs/physics_informed_config.yaml'
        )
        
        logger.info("Starting full production training...")
        logger.info("This will take 8-12 hours - please be patient")
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Calculate duration
        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600
        
        logger.info("=== FULL PRODUCTION TRAINING COMPLETED ===")
        logger.info(f"Total Duration: {duration_hours:.2f} hours")
        logger.info(f"Experiment: {experiment_name}")
        
        # Log final results
        if 'test_metrics' in results:
            test_metrics = results['test_metrics']
            logger.info("Final Test Results:")
            logger.info(f"  Binary Accuracy: {test_metrics.get('binary_accuracy', 0):.3f}")
            logger.info(f"  Binary F1-Score: {test_metrics.get('binary_f1', 0):.3f}")
            logger.info(f"  Magnitude Accuracy: {test_metrics.get('magnitude_class_accuracy', 0):.3f}")
            logger.info(f"  Magnitude MAE: {test_metrics.get('magnitude_mae', 0):.3f}")
            logger.info(f"  Distance MAE: {test_metrics.get('distance_mae', 0):.1f} km")
        
        # Save final report
        final_report = {
            'experiment_name': experiment_name,
            'start_time': datetime.fromtimestamp(start_time).isoformat(),
            'end_time': datetime.fromtimestamp(end_time).isoformat(),
            'duration_hours': duration_hours,
            'dataset_samples': 9156,
            'training_stages': 3,
            'results': results,
            'status': 'COMPLETED'
        }
        
        report_path = f'full_production_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Final report saved: {report_path}")
        
        print("\n" + "=" * 60)
        print("✅ FULL PRODUCTION TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Duration: {duration_hours:.2f} hours")
        print(f"Experiment: {experiment_name}")
        print(f"Report: {report_path}")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Full production training failed: {e}")
        print(f"\n❌ TRAINING FAILED: {e}")
        return 1


if __name__ == '__main__':
    exit(main())