#!/usr/bin/env python3
"""
Production Training Launcher
Senior Deep Learning Engineer Protocol

Integrates:
1. Dry run validation
2. Checkpoint system
3. Monitoring system
4. Production training pipeline

Author: Senior Deep Learning Engineer
Date: April 15, 2026
"""

import sys
import os
from pathlib import Path
import logging
import torch
import argparse
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from run_production_train import ProductionTrainingPipeline
from production_checkpoint_system import create_production_checkpoint_system
from production_monitoring_system import create_production_monitoring_system

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('production_training_launch.log')
    ]
)
logger = logging.getLogger(__name__)

class ProductionTrainingLauncher:
    """
    Production training launcher with full senior engineer protocol.
    """
    
    def __init__(self, 
                 dataset_path: str = 'outputs/corrective_actions/certified_spatio_dataset.h5',
                 config_path: str = 'configs/production_config.yaml',
                 experiment_name: str = None,
                 resume: bool = True):
        """
        Initialize production training launcher.
        
        Args:
            dataset_path: Path to certified dataset
            config_path: Path to training configuration
            experiment_name: Experiment name
            resume: Whether to enable resume capability
        """
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.resume = resume
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f'ground_truth_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.experiment_name = experiment_name
        
        # Output directory
        self.output_dir = Path('./outputs/production_training')
        self.experiment_dir = self.output_dir / experiment_name
        
        logger.info(f"ProductionTrainingLauncher initialized")
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Resume enabled: {resume}")
    
    def run_mandatory_dry_run(self) -> bool:
        """Run mandatory dry run before production training."""
        logger.info("=== MANDATORY DRY RUN ===")
        
        try:
            # Import and run simple dry run
            from simple_dry_run import simple_dry_run
            
            logger.info("Running system validation...")
            success = simple_dry_run()
            
            if success:
                logger.info("✅ Dry run PASSED - System ready for production")
                return True
            else:
                logger.error("❌ Dry run FAILED - System NOT ready")
                return False
                
        except Exception as e:
            logger.error(f"Dry run failed with exception: {str(e)}")
            return False
    
    def setup_production_systems(self):
        """Setup checkpoint and monitoring systems."""
        logger.info("=== SETTING UP PRODUCTION SYSTEMS ===")
        
        # Create training pipeline
        self.pipeline = ProductionTrainingPipeline(
            dataset_path=self.dataset_path,
            station_coords_path='../awal/lokasi_stasiun.csv',
            output_dir=str(self.output_dir),
            experiment_name=self.experiment_name,
            config_path=self.config_path
        )
        
        # Load components
        logger.info("Loading training components...")
        self.pipeline.load_station_coordinates()
        self.pipeline.load_dataset()
        self.pipeline.create_data_loaders()
        self.pipeline.create_model()
        
        # Setup checkpoint system
        logger.info("Setting up checkpoint system...")
        self.checkpoint_manager, self.auto_resume_trainer = create_production_checkpoint_system(
            output_dir=str(self.output_dir),
            experiment_name=self.experiment_name,
            trainer=None  # Will be set later
        )
        
        # Setup monitoring system
        logger.info("Setting up monitoring system...")
        self.monitoring_system = create_production_monitoring_system(
            log_dir=str(self.experiment_dir),
            experiment_name=self.experiment_name,
            early_stopping_patience=10
        )
        
        logger.info("Production systems setup complete")
    
    def execute_production_training(self):
        """Execute production training with full monitoring."""
        logger.info("=== EXECUTING PRODUCTION TRAINING ===")
        
        try:
            # Check for resume
            should_resume = False
            if self.resume:
                should_resume = self.checkpoint_manager.resume_training_prompt()
            
            # Load checkpoint if resuming
            checkpoint_data = None
            if should_resume:
                checkpoint_data = self.checkpoint_manager.load_checkpoint()
                if checkpoint_data:
                    logger.info(f"Resuming from epoch {checkpoint_data['epoch']}, stage {checkpoint_data['stage']}")
            
            # Create enhanced trainer with monitoring
            from src.training.trainer import SpatioTemporalTrainer
            
            trainer = SpatioTemporalTrainer(
                model=self.pipeline.model,
                train_loader=self.pipeline.train_loader,
                val_loader=self.pipeline.val_loader,
                device=self.pipeline.device,
                output_dir=str(self.experiment_dir),
                experiment_name=self.experiment_name,
                log_level='INFO'
            )
            
            # Integrate monitoring system
            original_train_epoch = trainer.train_epoch
            original_validate_epoch = trainer.validate_epoch
            
            def enhanced_train_epoch(optimizer):
                # Start epoch monitoring
                self.monitoring_system.start_epoch(trainer.epoch, trainer.current_stage)
                
                # Run training
                train_metrics = original_train_epoch(optimizer)
                
                # Log training metrics
                self.monitoring_system.log_training_metrics(
                    train_metrics, trainer.model, optimizer
                )
                
                return train_metrics
            
            def enhanced_validate_epoch():
                # Run validation
                val_metrics = original_validate_epoch()
                
                # Log validation metrics and check early stopping
                should_stop = self.monitoring_system.log_validation_metrics(val_metrics)
                
                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    model=trainer.model,
                    optimizer=trainer.optimizer if hasattr(trainer, 'optimizer') else None,
                    scheduler=trainer.scheduler if hasattr(trainer, 'scheduler') else None,
                    epoch=trainer.epoch,
                    stage=trainer.current_stage,
                    metrics=val_metrics
                )
                
                # Save best model if improved
                self.checkpoint_manager.save_best_model(
                    model=trainer.model,
                    stage=trainer.current_stage,
                    metric_value=val_metrics.get('avg_total_loss', float('inf')),
                    epoch=trainer.epoch,
                    metric_name='val_loss'
                )
                
                return val_metrics, should_stop
            
            # Replace trainer methods
            trainer.train_epoch = enhanced_train_epoch
            trainer.validate_epoch_original = enhanced_validate_epoch
            
            # Run progressive training
            logger.info("Starting progressive training...")
            training_history = trainer.train_progressive(self.pipeline.config)
            
            # Evaluate on test set if available
            if len(self.pipeline.test_dataset) > 0:
                logger.info("Evaluating on test set...")
                test_metrics = trainer.evaluate_model(
                    self.pipeline.test_loader,
                    str(self.experiment_dir / 'best_stage_3.pth')
                )
            else:
                logger.info("No test set available, skipping test evaluation")
                test_metrics = {}
            
            # Complete training
            self.monitoring_system.close()
            trainer.close()
            
            # Save final results
            final_results = {
                'experiment_name': self.experiment_name,
                'training_history': training_history,
                'test_metrics': test_metrics,
                'config': self.pipeline.config,
                'completion_time': datetime.now().isoformat(),
                'checkpoint_summary': self.checkpoint_manager.get_checkpoint_summary(),
                'training_summary': self.monitoring_system.get_training_summary()
            }
            
            results_path = self.experiment_dir / 'final_production_results.json'
            with open(results_path, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info("=== PRODUCTION TRAINING COMPLETED ===")
            logger.info(f"Results saved to: {results_path}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Production training failed: {str(e)}")
            raise
    
    def run_complete_pipeline(self):
        """Run complete production training pipeline."""
        logger.info("🚀 STARTING PRODUCTION TRAINING PIPELINE")
        logger.info("Senior Deep Learning Engineer Protocol")
        logger.info("=" * 60)
        
        try:
            # Step 1: Mandatory dry run
            if not self.run_mandatory_dry_run():
                logger.error("❌ Dry run failed - ABORTING production training")
                return None
            
            # Step 2: Setup production systems
            self.setup_production_systems()
            
            # Step 3: Execute production training
            results = self.execute_production_training()
            
            # Step 4: Final summary
            logger.info("🏆 PRODUCTION TRAINING PIPELINE COMPLETED")
            logger.info("=" * 60)
            
            if 'test_metrics' in results and results['test_metrics']:
                test_metrics = results['test_metrics']
                logger.info("Final Test Results:")
                logger.info(f"  Binary Accuracy: {test_metrics.get('binary_accuracy', 0):.3f}")
                logger.info(f"  Binary F1-Score: {test_metrics.get('binary_f1', 0):.3f}")
                logger.info(f"  Magnitude Accuracy: {test_metrics.get('magnitude_class_accuracy', 0):.3f}")
                logger.info(f"  Magnitude MAE: {test_metrics.get('magnitude_mae', 0):.3f}")
            
            logger.info(f"Experiment directory: {self.experiment_dir}")
            logger.info(f"TensorBoard: tensorboard --logdir {self.experiment_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Production training pipeline failed: {str(e)}")
            raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Production Training Launcher')
    parser.add_argument('--dataset', type=str, 
                       default='outputs/corrective_actions/certified_spatio_dataset.h5',
                       help='Path to certified dataset')
    parser.add_argument('--config', type=str, 
                       default='configs/production_config.yaml',
                       help='Path to training configuration')
    parser.add_argument('--experiment', type=str, 
                       default='ground_truth_run',
                       help='Experiment name')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Enable resume capability')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Disable resume capability')
    
    args = parser.parse_args()
    
    print("🚀 Production Training Launcher")
    print("Senior Deep Learning Engineer Protocol")
    print("Spatio-Temporal Earthquake Precursor Detection")
    print("=" * 60)
    
    # Create and run launcher
    launcher = ProductionTrainingLauncher(
        dataset_path=args.dataset,
        config_path=args.config,
        experiment_name=args.experiment,
        resume=args.resume
    )
    
    results = launcher.run_complete_pipeline()
    
    if results:
        print(f"\n🏆 Training completed successfully!")
        print(f"Experiment: {launcher.experiment_name}")
        print(f"Results: {launcher.experiment_dir}")
    else:
        print(f"\n❌ Training failed!")


if __name__ == '__main__':
    main()