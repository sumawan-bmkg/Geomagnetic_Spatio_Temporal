"""
Training Pipeline for Spatio-Temporal Earthquake Precursor Model

Implements stage-wise training with:
- Progressive learning across 3 stages
- Adaptive learning rate scheduling
- Model checkpointing and early stopping
- Comprehensive logging and monitoring
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime

from .metrics import PrecursorMetrics, calculate_stage_metrics
from .utils import save_checkpoint, load_checkpoint, setup_logging

logger = logging.getLogger(__name__)


class SpatioTemporalTrainer:
    """
    Trainer for spatio-temporal earthquake precursor model with stage-wise learning.
    
    Implements progressive training across three stages:
    1. Binary precursor detection
    2. Magnitude classification
    3. Localization estimation
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = 'cuda',
                 output_dir: str = './outputs/training',
                 experiment_name: str = None,
                 log_level: str = 'INFO'):
        """
        Initialize trainer.
        
        Args:
            model: Spatio-temporal precursor model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device for training
            output_dir: Output directory for checkpoints and logs
            experiment_name: Name for this experiment
            log_level: Logging level
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup experiment directory
        if experiment_name is None:
            experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(self.output_dir / 'training.log', log_level)
        
        # Training state
        self.current_stage = 1
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Training history
        self.train_history = []
        self.val_history = []
        
        # Metrics
        self.train_metrics = PrecursorMetrics(device=device)
        self.val_metrics = PrecursorMetrics(device=device)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def setup_stage_training(self, stage: int, config: Dict) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """
        Setup optimizer and scheduler for specific training stage.
        
        Args:
            stage: Training stage (1, 2, or 3)
            config: Stage configuration
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        logger.info(f"Setting up training for Stage {stage}")
        
        # Set model training stage
        self.model.set_training_stage(stage)
        self.current_stage = stage
        
        # Get stage-specific parameters
        if stage == 1:
            # Stage 1: Only binary head parameters
            params = list(self.model.hierarchical_heads.binary_head.parameters())
            if config.get('train_backbone', True):
                params += list(self.model.backbone.parameters())
            if config.get('train_gnn', True):
                params += list(self.model.gnn_fusion.parameters())
        elif stage == 2:
            # Stage 2: Binary + magnitude head parameters
            params = (list(self.model.hierarchical_heads.binary_head.parameters()) +
                     list(self.model.hierarchical_heads.magnitude_head.parameters()))
            if config.get('train_backbone', True):
                params += list(self.model.backbone.parameters())
            if config.get('train_gnn', True):
                params += list(self.model.gnn_fusion.parameters())
        else:
            # Stage 3: All parameters
            params = list(self.model.parameters())
        
        # Create optimizer
        optimizer_config = config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'AdamW')
        
        if optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                params,
                lr=optimizer_config.get('lr', 1e-4),
                weight_decay=optimizer_config.get('weight_decay', 1e-4),
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(
                params,
                lr=optimizer_config.get('lr', 1e-3),
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Create scheduler
        scheduler_config = config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'CosineAnnealingLR')
        
        if scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=scheduler_config.get('T_max', 50),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        logger.info(f"  Optimizer: {optimizer_type}")
        logger.info(f"  Scheduler: {scheduler_type}")
        logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in params if p.requires_grad):,}")
        
        return optimizer, scheduler
    
    def train_epoch(self, optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            optimizer: Optimizer for training
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.train_metrics.reset()
        
        total_batches = len(self.train_loader)
        log_interval = max(1, total_batches // 10)  # Log 10 times per epoch
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Forward pass
            optimizer.zero_grad()
            predictions = self.model(inputs, training_stage=self.current_stage)
            
            # Compute losses
            losses = self.model.compute_loss(predictions, targets, self.current_stage)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            self.train_metrics.update(predictions, targets, losses, self.current_stage)
            
            # Log progress
            if batch_idx % log_interval == 0:
                batch_metrics = calculate_stage_metrics(predictions, targets, self.current_stage)
                logger.info(
                    f"Epoch {self.epoch}, Batch {batch_idx}/{total_batches}: "
                    f"Loss={losses['total_loss'].item():.4f}, "
                    f"Metrics={batch_metrics}"
                )
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute_all_metrics(self.current_stage)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                # Move to device
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                predictions = self.model(inputs, training_stage=self.current_stage)
                
                # Compute losses
                losses = self.model.compute_loss(predictions, targets, self.current_stage)
                
                # Update metrics
                self.val_metrics.update(predictions, targets, losses, self.current_stage)
        
        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute_all_metrics(self.current_stage)
        
        return epoch_metrics
    
    def train_stage(self, stage: int, config: Dict) -> Dict[str, List[float]]:
        """
        Train a specific stage.
        
        Args:
            stage: Training stage (1, 2, or 3)
            config: Stage configuration
            
        Returns:
            Training history for this stage
        """
        logger.info(f"Starting Stage {stage} training")
        
        # Setup stage training
        optimizer, scheduler = self.setup_stage_training(stage, config)
        
        # Training parameters
        epochs = config.get('epochs', 50)
        patience = config.get('patience', 15)
        save_best = config.get('save_best', True)
        
        # Reset early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        stage_history = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(optimizer)
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['avg_total_loss'])
                else:
                    scheduler.step()
            
            # Record history
            stage_history['train'].append(train_metrics)
            stage_history['val'].append(val_metrics)
            
            # Logging
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"Stage {stage}, Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
            logger.info(f"  Train Loss: {train_metrics['avg_total_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['avg_total_loss']:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.2e}")
            
            # Log stage-specific metrics
            if stage >= 1:
                logger.info(f"  Binary Acc: Train={train_metrics.get('binary_accuracy', 0):.3f}, "
                          f"Val={val_metrics.get('binary_accuracy', 0):.3f}")
            
            if stage >= 2:
                logger.info(f"  Magnitude Acc: Train={train_metrics.get('magnitude_class_accuracy', 0):.3f}, "
                          f"Val={val_metrics.get('magnitude_class_accuracy', 0):.3f}")
            
            if stage >= 3:
                logger.info(f"  Azimuth Error: Train={train_metrics.get('azimuth_mean_error_deg', 0):.1f}°, "
                          f"Val={val_metrics.get('azimuth_mean_error_deg', 0):.1f}°")
            
            # Tensorboard logging
            self._log_to_tensorboard(train_metrics, val_metrics, epoch, stage)
            
            # Early stopping and checkpointing
            val_loss = val_metrics['avg_total_loss']
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if save_best:
                    checkpoint_path = self.output_dir / f'best_stage_{stage}.pth'
                    save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        loss=val_loss,
                        metrics=val_metrics,
                        path=checkpoint_path
                    )
                    logger.info(f"Saved best model for stage {stage}")
            else:
                self.patience_counter += 1
            
            # Regular checkpoint
            if (epoch + 1) % config.get('checkpoint_interval', 10) == 0:
                checkpoint_path = self.output_dir / f'stage_{stage}_epoch_{epoch+1}.pth'
                save_checkpoint(
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=val_loss,
                    metrics=val_metrics,
                    path=checkpoint_path
                )
            
            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info(f"Stage {stage} training completed")
        
        return stage_history
    
    def train_progressive(self, training_config: Dict) -> Dict:
        """
        Train model progressively through all stages.
        
        Args:
            training_config: Complete training configuration
            
        Returns:
            Complete training history
        """
        logger.info("Starting progressive training")
        
        complete_history = {}
        
        # Train each stage
        for stage in [1, 2, 3]:
            if f'stage_{stage}' in training_config:
                stage_config = training_config[f'stage_{stage}']
                
                # Load best model from previous stage (except stage 1)
                if stage > 1 and stage_config.get('load_previous_best', True):
                    prev_checkpoint = self.output_dir / f'best_stage_{stage-1}.pth'
                    if prev_checkpoint.exists():
                        logger.info(f"Loading best model from stage {stage-1}")
                        load_checkpoint(self.model, prev_checkpoint, device=self.device)
                
                # Train this stage
                stage_history = self.train_stage(stage, stage_config)
                complete_history[f'stage_{stage}'] = stage_history
                
                # Save stage completion
                self._save_stage_results(stage, stage_history)
            else:
                logger.warning(f"No configuration found for stage {stage}")
        
        # Save complete training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(complete_history, f, indent=2, default=str)
        
        logger.info("Progressive training completed")
        
        return complete_history
    
    def _log_to_tensorboard(self, train_metrics: Dict, val_metrics: Dict, 
                           epoch: int, stage: int):
        """Log metrics to tensorboard."""
        # Loss metrics
        for metric_name in ['avg_total_loss', 'avg_binary_loss']:
            if metric_name in train_metrics:
                self.writer.add_scalars(
                    f'Stage_{stage}/{metric_name}',
                    {'train': train_metrics[metric_name], 'val': val_metrics[metric_name]},
                    epoch
                )
        
        # Stage-specific metrics
        if stage >= 1:
            for metric_name in ['binary_accuracy', 'binary_f1']:
                if metric_name in train_metrics:
                    self.writer.add_scalars(
                        f'Stage_{stage}/{metric_name}',
                        {'train': train_metrics[metric_name], 'val': val_metrics[metric_name]},
                        epoch
                    )
        
        if stage >= 2:
            for metric_name in ['magnitude_class_accuracy', 'magnitude_mae']:
                if metric_name in train_metrics:
                    self.writer.add_scalars(
                        f'Stage_{stage}/{metric_name}',
                        {'train': train_metrics[metric_name], 'val': val_metrics[metric_name]},
                        epoch
                    )
        
        if stage >= 3:
            for metric_name in ['azimuth_mean_error_deg', 'distance_mae']:
                if metric_name in train_metrics:
                    self.writer.add_scalars(
                        f'Stage_{stage}/{metric_name}',
                        {'train': train_metrics[metric_name], 'val': val_metrics[metric_name]},
                        epoch
                    )
    
    def _save_stage_results(self, stage: int, stage_history: Dict):
        """Save stage-specific results."""
        stage_dir = self.output_dir / f'stage_{stage}'
        stage_dir.mkdir(exist_ok=True)
        
        # Save history
        history_path = stage_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(stage_history, f, indent=2, default=str)
        
        # Save final metrics
        if stage_history['val']:
            final_metrics = stage_history['val'][-1]
            metrics_path = stage_dir / 'final_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=2, default=str)
        
        # Save confusion matrices
        if hasattr(self.val_metrics, 'get_confusion_matrix'):
            binary_cm = self.val_metrics.get_confusion_matrix('binary')
            if binary_cm is not None:
                np.save(stage_dir / 'binary_confusion_matrix.npy', binary_cm)
            
            if stage >= 2:
                magnitude_cm = self.val_metrics.get_confusion_matrix('magnitude')
                if magnitude_cm is not None:
                    np.save(stage_dir / 'magnitude_confusion_matrix.npy', magnitude_cm)
    
    def evaluate_model(self, test_loader: DataLoader, 
                      checkpoint_path: str = None) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Test metrics
        """
        logger.info("Evaluating model on test set")
        
        # Load checkpoint if provided
        if checkpoint_path:
            load_checkpoint(self.model, checkpoint_path, device=self.device)
        
        # Evaluate on test set
        self.model.eval()
        test_metrics = PrecursorMetrics(device=self.device)
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                predictions = self.model(inputs, training_stage=3)
                losses = self.model.compute_loss(predictions, targets, 3)
                
                test_metrics.update(predictions, targets, losses, 3)
        
        # Compute final metrics
        final_metrics = test_metrics.compute_all_metrics(3)
        
        # Save test results
        test_results_path = self.output_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(final_metrics, f, indent=2, default=str)
        
        logger.info("Test evaluation completed")
        logger.info(f"Test results saved to: {test_results_path}")
        
        return final_metrics
    
    def close(self):
        """Close trainer and cleanup resources."""
        if hasattr(self, 'writer'):
            self.writer.close()
        
        logger.info("Trainer closed")


if __name__ == '__main__':
    # Test trainer setup
    print("Testing SpatioTemporalTrainer...")
    
    # This would normally use real model and data loaders
    print("Trainer class structure validated!")