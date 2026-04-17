#!/usr/bin/env python3
"""
Main Training Script for Spatio-Temporal Earthquake Precursor Model

This script demonstrates the complete training workflow:
1. Load HDF5 tensor data from tensor engine
2. Create PyTorch datasets and data loaders
3. Initialize model with station coordinates
4. Train progressively through 3 stages
5. Evaluate on test set and save results

Usage:
    python train_model.py --config configs/training_config.yaml
    python train_model.py --train-data outputs/tensors/train_cmr.h5 --test-data outputs/tensors/test_cmr.h5
"""
import argparse
import sys
import os
from pathlib import Path
import logging
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import project modules
from src.models.spatio_temporal_model import create_model
from src.training.dataset import create_data_loaders
from src.training.trainer import SpatioTemporalTrainer
from src.training.utils import (
    setup_training, save_training_config, create_experiment_directory,
    backup_code, create_training_summary
)

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Spatio-Temporal Earthquake Precursor Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training HDF5 file')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test HDF5 file')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Path to metadata CSV file')
    parser.add_argument('--station-coords', type=str,
                       help='Path to station coordinates CSV')
    
    # Training arguments
    parser.add_argument('--config', type=str,
                       help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='./outputs/training',
                       help='Output directory for training results')
    parser.add_argument('--experiment-name', type=str,
                       help='Name for this experiment')
    
    # Model arguments
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--load-in-memory', action='store_true',
                       help='Load all data in memory')
    
    # System arguments
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # Training control
    parser.add_argument('--stages', nargs='+', type=int, default=[1, 2, 3],
                       choices=[1, 2, 3], help='Training stages to run')
    parser.add_argument('--resume-from', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate model (no training)')
    
    return parser.parse_args()


def load_station_coordinates(station_coords_path: str):
    """Load station coordinates from CSV file."""
    if not station_coords_path or not os.path.exists(station_coords_path):
        logger.warning("Station coordinates file not found, using default coordinates")
        return None
    
    import pandas as pd
    
    try:
        coords_df = pd.read_csv(station_coords_path)
        
        # Expected columns: station_code, latitude, longitude
        if not all(col in coords_df.columns for col in ['station_code', 'latitude', 'longitude']):
            logger.warning("Station coordinates CSV missing required columns")
            return None
        
        # Convert to numpy array (lat, lon pairs)
        coords_array = coords_df[['latitude', 'longitude']].values
        
        logger.info(f"Loaded coordinates for {len(coords_array)} stations")
        return coords_array
        
    except Exception as e:
        logger.error(f"Error loading station coordinates: {e}")
        return None


def create_model_with_config(config: dict, station_coordinates=None):
    """Create model with configuration."""
    model_config = config.get('model', {})
    
    # Add station coordinates if available
    if station_coordinates is not None:
        model_config['station_coordinates'] = station_coordinates
    
    # Create model
    model = create_model(config=model_config)
    
    logger.info("Model created successfully")
    logger.info(f"Model summary: {model.get_model_summary()}")
    
    return model


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup training environment
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = setup_training(
        config_path=args.config,
        seed=args.seed,
        device=device
    )
    
    # Override config with command line arguments
    if args.batch_size:
        config.setdefault('data', {})['batch_size'] = args.batch_size
    if args.num_workers:
        config.setdefault('data', {})['num_workers'] = args.num_workers
    if args.load_in_memory:
        config.setdefault('data', {})['load_in_memory'] = args.load_in_memory
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(
        base_dir=args.output_dir,
        experiment_name=args.experiment_name
    )
    
    # Setup logging
    from src.training.utils import setup_logging
    setup_logging(
        log_file=experiment_dir / 'training.log',
        level=args.log_level
    )
    
    logger.info("Starting Spatio-Temporal Earthquake Precursor Training")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Training stages: {args.stages}")
    
    # Save configuration
    save_training_config(config, experiment_dir)
    
    # Backup source code
    try:
        backup_code(
            source_dir=project_root / 'src',
            backup_dir=experiment_dir / 'code_backup'
        )
    except Exception as e:
        logger.warning(f"Could not backup code: {e}")
    
    # Load station coordinates
    station_coordinates = load_station_coordinates(args.station_coords)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    data_config = config.get('data', {})
    train_loader, test_loader, dataset_info = create_data_loaders(
        train_hdf5_path=args.train_data,
        test_hdf5_path=args.test_data,
        metadata_path=args.metadata,
        batch_size=data_config.get('batch_size', 16),
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        load_in_memory=data_config.get('load_in_memory', False),
        magnitude_bins=data_config.get('magnitude_bins')
    )
    
    logger.info("Data loaders created successfully")
    logger.info(f"Train samples: {dataset_info['train_stats']['n_samples']}")
    logger.info(f"Test samples: {dataset_info['test_stats']['n_samples']}")
    
    # Save dataset info
    with open(experiment_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2, default=str)
    
    # Create model
    logger.info("Creating model...")
    model = create_model_with_config(config, station_coordinates)
    
    # Create training summary
    training_summary = create_training_summary(
        model=model,
        train_loader=train_loader,
        config=config,
        output_dir=experiment_dir
    )
    
    # Create trainer
    trainer = SpatioTemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for now
        device=device,
        output_dir=experiment_dir,
        experiment_name=args.experiment_name,
        log_level=args.log_level
    )
    
    try:
        if args.evaluate_only:
            # Evaluation only
            logger.info("Running evaluation only...")
            
            checkpoint_path = args.resume_from
            if not checkpoint_path:
                # Look for best model from stage 3
                checkpoint_path = experiment_dir / 'best_stage_3.pth'
                if not checkpoint_path.exists():
                    raise FileNotFoundError("No checkpoint found for evaluation")
            
            test_metrics = trainer.evaluate_model(
                test_loader=test_loader,
                checkpoint_path=checkpoint_path
            )
            
            logger.info("Evaluation completed")
            logger.info("Test metrics:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        else:
            # Training
            logger.info("Starting progressive training...")
            
            # Resume from checkpoint if specified
            if args.resume_from:
                logger.info(f"Resuming from checkpoint: {args.resume_from}")
                from src.training.utils import load_checkpoint
                load_checkpoint(model, args.resume_from, device=device)
            
            # Filter training config for requested stages
            training_config = {}
            for stage in args.stages:
                stage_key = f'stage_{stage}'
                if stage_key in config:
                    training_config[stage_key] = config[stage_key]
            
            # Add loss weights
            if 'loss_weights' in config:
                training_config['loss_weights'] = config['loss_weights']
            
            # Train model
            training_history = trainer.train_progressive(training_config)
            
            logger.info("Training completed successfully")
            
            # Final evaluation on test set
            logger.info("Running final evaluation...")
            
            # Use best model from last trained stage
            last_stage = max(args.stages)
            best_checkpoint = experiment_dir / f'best_stage_{last_stage}.pth'
            
            if best_checkpoint.exists():
                test_metrics = trainer.evaluate_model(
                    test_loader=test_loader,
                    checkpoint_path=best_checkpoint
                )
                
                logger.info("Final test metrics:")
                for key, value in test_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.warning("No best checkpoint found for final evaluation")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        trainer.close()
        logger.info("Training script completed")


if __name__ == '__main__':
    main()