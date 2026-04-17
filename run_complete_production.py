#!/usr/bin/env python3
"""
Complete Production Pipeline - One-Liner Execution

Script ini menjalankan seluruh pipeline production:
1. Generate final dataset dari folder awal
2. Train model dengan progressive learning
3. Evaluate dan save results

Usage:
    python run_complete_production.py
"""
import sys
import os
from pathlib import Path
import subprocess
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run command dengan logging."""
    logger.info(f"Run command dengan logging.")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        logger.info(f"Completed: {description}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    """Main execution function."""
    print("Complete Production Pipeline")
    print("Spatio-Temporal Earthquake Precursor Detection")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Step 1: Generate final dataset
    logger.info("STEP 1: Generating final dataset...")
    success = run_command(
        "python generate_final_dataset.py",
        "Dataset generation"
    )
    
    if not success:
        logger.error("Dataset generation failed. Stopping pipeline.")
        return False
    
    # Check if dataset was created
    dataset_path = Path("spatio_earthquake_dataset.h5")
    if not dataset_path.exists():
        logger.error("Dataset file not found. Generation may have failed.")
        return False
    
    logger.info(f"Dataset created: {dataset_path}")
    
    # Step 2: Run production training
    logger.info("STEP 2: Running production training...")
    success = run_command(
        "python run_production_train.py --config configs/production_config.yaml",
        "Production training"
    )
    
    if not success:
        logger.error("Production training failed.")
        return False
    
    # Calculate total time
    end_time = datetime.now()
    total_time = end_time - start_time
    
    logger.info("COMPLETE PRODUCTION PIPELINE FINISHED")
    logger.info("=" * 60)
    logger.info(f"Total execution time: {total_time}")
    logger.info(f"Dataset: spatio_earthquake_dataset.h5")
    logger.info(f"Training results: outputs/production_training/")
    
    print(f"\nProduction pipeline completed successfully!")
    print(f"Total time: {total_time}")
    print(f"Check outputs/production_training/ for results")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)