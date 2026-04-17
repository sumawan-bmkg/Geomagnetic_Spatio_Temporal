#!/usr/bin/env python3
"""
Final Production Training Recovery - Post Bug Fix

This script runs the corrected production training with full dataset.
"""

import sys
import os
from pathlib import Path
import logging
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('final_recovery.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run final recovery training."""
    print("🚀 FINAL PRODUCTION TRAINING RECOVERY")
    print("=" * 60)
    print("Post-Subset Bug Fix - Full Dataset Training")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Import training pipeline
        from run_production_train import ProductionTrainingPipeline
        
        # Use certified dataset
        dataset_path = 'outputs/corrective_actions/certified_spatio_dataset.h5'
        
        # Verify dataset exists
        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return 1
        
        logger.info(f"Using dataset: {dataset_path}")
        
        # Create experiment name with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'final_recovery_{timestamp}'
        
        # Create pipeline
        pipeline = ProductionTrainingPipeline(
            dataset_path=dataset_path,
            station_coords_path='../awal/lokasi_stasiun.csv',
            output_dir='./outputs/final_recovery',
            experiment_name=experiment_name,
            config_path='configs/physics_informed_config.yaml'
        )
        
        logger.info("Starting production training...")
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        # Calculate duration
        end_time = time.time()
        duration_hours = (end_time - start_time) / 3600
        
        logger.info("=== TRAINING COMPLETED ===")
        logger.info(f"Duration: {duration_hours:.1f} hours")
        logger.info(f"Results: {results}")
        
        print("✅ RECOVERY TRAINING COMPLETED SUCCESSFULLY")
        print(f"Duration: {duration_hours:.1f} hours")
        print(f"Experiment: {experiment_name}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ TRAINING FAILED: {e}")
        return 1


if __name__ == '__main__':
    exit(main())