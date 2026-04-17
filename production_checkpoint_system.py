#!/usr/bin/env python3
"""
Production Checkpointing & Resume System
Senior Deep Learning Engineer Protocol

Sistem checkpointing otomatis untuk:
1. Auto-save setiap epoch
2. Best model tracking per stage
3. Resume capability
4. Crash recovery

Author: Senior Deep Learning Engineer
Date: April 15, 2026
"""

import os
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ProductionCheckpointManager:
    """
    Advanced checkpoint management system untuk production training.
    """
    
    def __init__(self, 
                 checkpoint_dir: str,
                 experiment_name: str,
                 save_top_k: int = 3,
                 save_every_n_epochs: int = 1):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory untuk menyimpan checkpoints
            experiment_name: Nama experiment
            save_top_k: Jumlah best models yang disimpan
            save_every_n_epochs: Interval penyimpanan checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.save_top_k = save_top_k
        self.save_every_n_epochs = save_every_n_epochs
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint files
        self.latest_checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"
        self.best_models_dir = self.checkpoint_dir / "best_models"
        self.best_models_dir.mkdir(exist_ok=True)
        
        # Tracking
        self.best_metrics = {
            'stage_1': {'metric': float('inf'), 'epoch': -1, 'path': None},
            'stage_2': {'metric': float('inf'), 'epoch': -1, 'path': None},
            'stage_3': {'metric': float('inf'), 'epoch': -1, 'path': None}
        }
        
        # Load existing best metrics if available
        self._load_best_metrics()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       epoch: int,
                       stage: int,
                       metrics: Dict[str, float],
                       additional_info: Optional[Dict] = None) -> str:
        """
        Save checkpoint dengan semua informasi training.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            stage: Current training stage (1, 2, or 3)
            metrics: Current metrics
            additional_info: Additional information to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_data = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_metrics': self.best_metrics,
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name
        }
        
        # Add additional info
        if additional_info:
            checkpoint_data.update(additional_info)
        
        # Save latest checkpoint
        torch.save(checkpoint_data, self.latest_checkpoint_path)
        logger.info(f"Saved latest checkpoint: epoch {epoch}, stage {stage}")
        
        # Save periodic checkpoint
        if epoch % self.save_every_n_epochs == 0:
            periodic_path = self.checkpoint_dir / f"checkpoint_stage_{stage}_epoch_{epoch:03d}.pth"
            torch.save(checkpoint_data, periodic_path)
            logger.info(f"Saved periodic checkpoint: {periodic_path}")
        
        return str(self.latest_checkpoint_path)
    
    def save_best_model(self,
                       model: torch.nn.Module,
                       stage: int,
                       metric_value: float,
                       epoch: int,
                       metric_name: str = 'loss',
                       is_better_func: callable = None) -> bool:
        """
        Save model jika merupakan best model untuk stage tertentu.
        
        Args:
            model: PyTorch model
            stage: Training stage (1, 2, or 3)
            metric_value: Metric value untuk comparison
            epoch: Current epoch
            metric_name: Nama metric
            is_better_func: Function untuk menentukan apakah metric lebih baik
        
        Returns:
            True jika model disimpan sebagai best model
        """
        if is_better_func is None:
            # Default: lower is better (untuk loss)
            is_better_func = lambda new, old: new < old
        
        stage_key = f'stage_{stage}'
        current_best = self.best_metrics[stage_key]['metric']
        
        if is_better_func(metric_value, current_best):
            # Update best metrics
            self.best_metrics[stage_key] = {
                'metric': metric_value,
                'epoch': epoch,
                'metric_name': metric_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save best model
            best_model_path = self.best_models_dir / f"best_stage_{stage}.pth"
            
            model_data = {
                'model_state_dict': model.state_dict(),
                'stage': stage,
                'epoch': epoch,
                'metric_value': metric_value,
                'metric_name': metric_name,
                'timestamp': datetime.now().isoformat(),
                'experiment_name': self.experiment_name
            }
            
            torch.save(model_data, best_model_path)
            self.best_metrics[stage_key]['path'] = str(best_model_path)
            
            # Save best metrics tracking
            self._save_best_metrics()
            
            logger.info(f"🏆 New best model for stage {stage}!")
            logger.info(f"   Metric: {metric_name} = {metric_value:.6f} (epoch {epoch})")
            logger.info(f"   Saved: {best_model_path}")
            
            return True
        
        return False
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict]:
        """
        Load checkpoint dari file.
        
        Args:
            checkpoint_path: Path ke checkpoint file. Jika None, load latest.
        
        Returns:
            Checkpoint data atau None jika tidak ada
        """
        if checkpoint_path is None:
            checkpoint_path = self.latest_checkpoint_path
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found at: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            logger.info(f"  Epoch: {checkpoint_data['epoch']}")
            logger.info(f"  Stage: {checkpoint_data['stage']}")
            logger.info(f"  Timestamp: {checkpoint_data.get('timestamp', 'Unknown')}")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {str(e)}")
            return None
    
    def load_best_model(self, stage: int) -> Optional[Dict]:
        """
        Load best model untuk stage tertentu.
        
        Args:
            stage: Training stage (1, 2, or 3)
        
        Returns:
            Model data atau None jika tidak ada
        """
        stage_key = f'stage_{stage}'
        best_info = self.best_metrics[stage_key]
        
        if best_info['path'] is None:
            logger.info(f"No best model found for stage {stage}")
            return None
        
        best_model_path = Path(best_info['path'])
        
        if not best_model_path.exists():
            logger.warning(f"Best model file not found: {best_model_path}")
            return None
        
        try:
            model_data = torch.load(best_model_path, map_location='cpu')
            logger.info(f"Loaded best model for stage {stage}")
            logger.info(f"  Metric: {best_info.get('metric_name', 'unknown')} = {best_info['metric']:.6f}")
            logger.info(f"  Epoch: {best_info['epoch']}")
            
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load best model for stage {stage}: {str(e)}")
            return None
    
    def resume_training_prompt(self) -> bool:
        """
        Interactive prompt untuk resume training.
        
        Returns:
            True jika user memilih resume, False untuk start fresh
        """
        if not self.latest_checkpoint_path.exists():
            logger.info("No existing checkpoint found. Starting fresh training.")
            return False
        
        # Load checkpoint info
        checkpoint_data = self.load_checkpoint()
        if checkpoint_data is None:
            logger.info("Cannot load checkpoint. Starting fresh training.")
            return False
        
        print("\n" + "="*60)
        print("🔄 EXISTING CHECKPOINT DETECTED")
        print("="*60)
        print(f"Experiment: {checkpoint_data.get('experiment_name', 'Unknown')}")
        print(f"Last Epoch: {checkpoint_data['epoch']}")
        print(f"Last Stage: {checkpoint_data['stage']}")
        print(f"Timestamp: {checkpoint_data.get('timestamp', 'Unknown')}")
        
        if 'metrics' in checkpoint_data:
            print(f"Last Metrics:")
            for key, value in checkpoint_data['metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.6f}")
        
        print("\nOptions:")
        print("1. Resume from checkpoint (recommended)")
        print("2. Start fresh training (will backup existing)")
        print("3. Cancel")
        
        while True:
            try:
                choice = input("\nEnter your choice (1/2/3): ").strip()
                
                if choice == '1':
                    logger.info("✅ Resuming from checkpoint")
                    return True
                elif choice == '2':
                    # Backup existing checkpoint
                    backup_dir = self.checkpoint_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    backup_dir.mkdir(exist_ok=True)
                    
                    if self.latest_checkpoint_path.exists():
                        shutil.copy2(self.latest_checkpoint_path, backup_dir / "latest_checkpoint.pth")
                    
                    if self.best_models_dir.exists():
                        shutil.copytree(self.best_models_dir, backup_dir / "best_models", dirs_exist_ok=True)
                    
                    logger.info(f"📦 Existing checkpoint backed up to: {backup_dir}")
                    logger.info("🆕 Starting fresh training")
                    return False
                elif choice == '3':
                    logger.info("❌ Training cancelled by user")
                    exit(0)
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    
            except KeyboardInterrupt:
                logger.info("\n❌ Training cancelled by user")
                exit(0)
    
    def _save_best_metrics(self):
        """Save best metrics tracking to file."""
        metrics_file = self.checkpoint_dir / "best_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.best_metrics, f, indent=2)
    
    def _load_best_metrics(self):
        """Load best metrics tracking from file."""
        metrics_file = self.checkpoint_dir / "best_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    loaded_metrics = json.load(f)
                    self.best_metrics.update(loaded_metrics)
                logger.info("Loaded existing best metrics tracking")
            except Exception as e:
                logger.warning(f"Failed to load best metrics: {str(e)}")
    
    def get_checkpoint_summary(self) -> Dict:
        """Get summary of checkpoint status."""
        summary = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'latest_checkpoint_exists': self.latest_checkpoint_path.exists(),
            'best_models': {}
        }
        
        for stage in [1, 2, 3]:
            stage_key = f'stage_{stage}'
            best_info = self.best_metrics[stage_key]
            
            summary['best_models'][stage_key] = {
                'exists': best_info['path'] is not None and Path(best_info['path']).exists() if best_info['path'] else False,
                'metric': best_info['metric'] if best_info['metric'] != float('inf') else None,
                'epoch': best_info['epoch'] if best_info['epoch'] != -1 else None
            }
        
        return summary
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Cleanup old periodic checkpoints, keeping only the most recent ones.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        # Find all periodic checkpoints
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_stage_*_epoch_*.pth"))
        
        if len(checkpoint_files) <= keep_last_n:
            return
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[keep_last_n:]:
            try:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                logger.warning(f"Failed to remove {old_checkpoint}: {str(e)}")


class AutoResumeTrainer:
    """
    Wrapper class yang mengintegrasikan checkpoint system dengan trainer.
    """
    
    def __init__(self, 
                 trainer,
                 checkpoint_manager: ProductionCheckpointManager,
                 auto_resume: bool = True):
        """
        Initialize auto-resume trainer.
        
        Args:
            trainer: Base trainer object
            checkpoint_manager: Checkpoint manager
            auto_resume: Whether to automatically prompt for resume
        """
        self.trainer = trainer
        self.checkpoint_manager = checkpoint_manager
        self.auto_resume = auto_resume
        
        # Check for resume
        self.should_resume = False
        if auto_resume:
            self.should_resume = checkpoint_manager.resume_training_prompt()
    
    def load_checkpoint_if_resuming(self):
        """Load checkpoint if resuming training."""
        if not self.should_resume:
            return None
        
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        if checkpoint_data is None:
            logger.warning("Failed to load checkpoint for resume. Starting fresh.")
            self.should_resume = False
            return None
        
        return checkpoint_data
    
    def save_training_checkpoint(self, 
                                model, 
                                optimizer, 
                                scheduler, 
                                epoch, 
                                stage, 
                                metrics,
                                additional_info=None):
        """Save training checkpoint."""
        return self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            stage=stage,
            metrics=metrics,
            additional_info=additional_info
        )
    
    def save_best_model_if_improved(self, 
                                   model, 
                                   stage, 
                                   metric_value, 
                                   epoch,
                                   metric_name='val_loss'):
        """Save model if it's the best for the stage."""
        return self.checkpoint_manager.save_best_model(
            model=model,
            stage=stage,
            metric_value=metric_value,
            epoch=epoch,
            metric_name=metric_name
        )


def create_production_checkpoint_system(output_dir: str, 
                                      experiment_name: str,
                                      trainer=None) -> tuple:
    """
    Factory function untuk membuat production checkpoint system.
    
    Args:
        output_dir: Output directory
        experiment_name: Experiment name
        trainer: Optional trainer object
    
    Returns:
        Tuple of (checkpoint_manager, auto_resume_trainer)
    """
    checkpoint_dir = Path(output_dir) / experiment_name / "checkpoints"
    
    checkpoint_manager = ProductionCheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        experiment_name=experiment_name,
        save_top_k=3,
        save_every_n_epochs=1
    )
    
    auto_resume_trainer = None
    if trainer is not None:
        auto_resume_trainer = AutoResumeTrainer(
            trainer=trainer,
            checkpoint_manager=checkpoint_manager,
            auto_resume=True
        )
    
    return checkpoint_manager, auto_resume_trainer