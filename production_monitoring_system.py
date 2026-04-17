#!/usr/bin/env python3
"""
Production Monitoring & Logging System
Senior Deep Learning Engineer Protocol

Sistem monitoring real-time untuk:
1. TensorBoard integration
2. Early stopping dengan patience
3. Real-time metrics tracking
4. Performance monitoring

Author: Senior Deep Learning Engineer
Date: April 15, 2026
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping implementation dengan patience untuk production training.
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.0,
                 restore_best_weights: bool = True,
                 mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        
        if mode == 'min':
            self.best_score = np.inf
            self.is_better = lambda score, best: score < (best - min_delta)
        else:
            self.best_score = -np.inf
            self.is_better = lambda score, best: score > (best + min_delta)
        
        logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")
    
    def __call__(self, score: float, model: torch.nn.Module = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            model: Model to save best weights
        
        Returns:
            True if training should stop
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.wait = 0
            
            # Save best weights
            if model is not None and self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            return False
        
        self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            
            # Restore best weights
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights from early stopping")
            
            return True
        
        return False
    
    def get_status(self) -> Dict:
        """Get current early stopping status."""
        return {
            'wait': self.wait,
            'patience': self.patience,
            'best_score': self.best_score,
            'stopped_epoch': self.stopped_epoch
        }


class ProductionMetricsTracker:
    """
    Advanced metrics tracking untuk production training.
    """
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str,
                 track_gradients: bool = True,
                 track_weights: bool = True):
        """
        Initialize metrics tracker.
        
        Args:
            log_dir: Directory untuk logs
            experiment_name: Experiment name
            track_gradients: Whether to track gradients
            track_weights: Whether to track weights
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.track_gradients = track_gradients
        self.track_weights = track_weights
        
        # Create log directory
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(
            log_dir=str(self.tensorboard_dir / experiment_name),
            comment=f"_{experiment_name}"
        )
        
        # Metrics storage
        self.metrics_history = {
            'train': {},
            'val': {},
            'test': {}
        }
        
        self.current_epoch = 0
        self.current_stage = 1
        
        logger.info(f"MetricsTracker initialized: {self.tensorboard_dir}")
        logger.info(f"TensorBoard command: tensorboard --logdir {self.tensorboard_dir}")
    
    def log_scalar(self, 
                   tag: str, 
                   value: float, 
                   step: Optional[int] = None,
                   phase: str = 'train'):
        """
        Log scalar value to TensorBoard.
        
        Args:
            tag: Metric name
            value: Metric value
            step: Step number (uses current epoch if None)
            phase: Phase (train/val/test)
        """
        if step is None:
            step = self.current_epoch
        
        # Log to TensorBoard
        full_tag = f"{phase}/{tag}"
        self.writer.add_scalar(full_tag, value, step)
        
        # Store in history
        if tag not in self.metrics_history[phase]:
            self.metrics_history[phase][tag] = []
        
        self.metrics_history[phase][tag].append({
            'step': step,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_metrics_dict(self, 
                        metrics: Dict[str, float], 
                        step: Optional[int] = None,
                        phase: str = 'train'):
        """
        Log dictionary of metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Step number
            phase: Phase (train/val/test)
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.log_scalar(name, value, step, phase)
    
    def log_model_gradients(self, 
                           model: torch.nn.Module, 
                           step: Optional[int] = None):
        """
        Log model gradients to TensorBoard.
        
        Args:
            model: PyTorch model
            step: Step number
        """
        if not self.track_gradients:
            return
        
        if step is None:
            step = self.current_epoch
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Log gradient norm
                grad_norm = param.grad.norm().item()
                self.writer.add_scalar(f"gradients/{name}_norm", grad_norm, step)
                
                # Log gradient histogram
                self.writer.add_histogram(f"gradients/{name}", param.grad, step)
    
    def log_model_weights(self, 
                         model: torch.nn.Module, 
                         step: Optional[int] = None):
        """
        Log model weights to TensorBoard.
        
        Args:
            model: PyTorch model
            step: Step number
        """
        if not self.track_weights:
            return
        
        if step is None:
            step = self.current_epoch
        
        for name, param in model.named_parameters():
            # Log weight histogram
            self.writer.add_histogram(f"weights/{name}", param, step)
            
            # Log weight norm
            weight_norm = param.norm().item()
            self.writer.add_scalar(f"weights/{name}_norm", weight_norm, step)
    
    def log_learning_rate(self, 
                         optimizer: torch.optim.Optimizer, 
                         step: Optional[int] = None):
        """
        Log learning rate to TensorBoard.
        
        Args:
            optimizer: PyTorch optimizer
            step: Step number
        """
        if step is None:
            step = self.current_epoch
        
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.writer.add_scalar(f"learning_rate/group_{i}", lr, step)
    
    def log_stage_transition(self, new_stage: int):
        """
        Log stage transition.
        
        Args:
            new_stage: New training stage
        """
        self.current_stage = new_stage
        self.writer.add_text(
            "training/stage_transition",
            f"Transitioned to Stage {new_stage}",
            self.current_epoch
        )
        logger.info(f"📊 Logged stage transition to Stage {new_stage}")
    
    def update_epoch(self, epoch: int):
        """Update current epoch."""
        self.current_epoch = epoch
    
    def get_metrics_summary(self, phase: str = 'val', last_n: int = 10) -> Dict:
        """
        Get summary of recent metrics.
        
        Args:
            phase: Phase to summarize
            last_n: Number of recent epochs to include
        
        Returns:
            Dictionary of metric summaries
        """
        summary = {}
        
        for metric_name, history in self.metrics_history[phase].items():
            if len(history) == 0:
                continue
            
            recent_values = [h['value'] for h in history[-last_n:]]
            
            summary[metric_name] = {
                'current': history[-1]['value'] if history else None,
                'mean_recent': np.mean(recent_values),
                'std_recent': np.std(recent_values),
                'min_recent': np.min(recent_values),
                'max_recent': np.max(recent_values),
                'trend': 'improving' if len(recent_values) > 1 and recent_values[-1] < recent_values[0] else 'stable'
            }
        
        return summary
    
    def save_metrics_history(self):
        """Save metrics history to file."""
        history_file = self.log_dir / f"{self.experiment_name}_metrics_history.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        
        logger.info(f"Saved metrics history: {history_file}")
    
    def generate_training_plots(self):
        """Generate training progress plots."""
        plots_dir = self.log_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Loss plots
        self._plot_losses(plots_dir)
        
        # Metrics plots
        self._plot_metrics(plots_dir)
        
        logger.info(f"Generated training plots: {plots_dir}")
    
    def _plot_losses(self, plots_dir: Path):
        """Plot loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16)
        
        # Binary loss
        self._plot_metric('binary_loss', axes[0, 0], 'Binary Loss')
        
        # Magnitude loss
        self._plot_metric('magnitude_loss', axes[0, 1], 'Magnitude Loss')
        
        # Localization loss
        self._plot_metric('localization_loss', axes[1, 0], 'Localization Loss')
        
        # Total loss
        self._plot_metric('total_loss', axes[1, 1], 'Total Loss')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics(self, plots_dir: Path):
        """Plot performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Performance Metrics - {self.experiment_name}', fontsize=16)
        
        # Binary F1
        self._plot_metric('binary_f1', axes[0, 0], 'Binary F1-Score')
        
        # Magnitude accuracy
        self._plot_metric('magnitude_accuracy', axes[0, 1], 'Magnitude Accuracy')
        
        # Magnitude MAE
        self._plot_metric('magnitude_mae', axes[1, 0], 'Magnitude MAE')
        
        # Distance MAE
        self._plot_metric('distance_mae', axes[1, 1], 'Distance MAE (km)')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric(self, metric_name: str, ax, title: str):
        """Plot single metric."""
        train_data = self.metrics_history['train'].get(metric_name, [])
        val_data = self.metrics_history['val'].get(metric_name, [])
        
        if train_data:
            train_steps = [d['step'] for d in train_data]
            train_values = [d['value'] for d in train_data]
            ax.plot(train_steps, train_values, label='Train', alpha=0.7)
        
        if val_data:
            val_steps = [d['step'] for d in val_data]
            val_values = [d['value'] for d in val_data]
            ax.plot(val_steps, val_values, label='Validation', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def close(self):
        """Close TensorBoard writer and save final data."""
        self.save_metrics_history()
        self.generate_training_plots()
        self.writer.close()
        logger.info("MetricsTracker closed")


class ProductionMonitoringSystem:
    """
    Comprehensive monitoring system yang mengintegrasikan semua komponen.
    """
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str,
                 early_stopping_patience: int = 10,
                 track_gradients: bool = True):
        """
        Initialize monitoring system.
        
        Args:
            log_dir: Log directory
            experiment_name: Experiment name
            early_stopping_patience: Early stopping patience
            track_gradients: Whether to track gradients
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Initialize components
        self.metrics_tracker = ProductionMetricsTracker(
            log_dir=str(self.log_dir),
            experiment_name=experiment_name,
            track_gradients=track_gradients,
            track_weights=True
        )
        
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=0.001,
            restore_best_weights=True,
            mode='min'
        )
        
        # Status tracking
        self.training_status = {
            'start_time': datetime.now().isoformat(),
            'current_stage': 1,
            'current_epoch': 0,
            'best_metrics': {},
            'early_stopped': False,
            'completed_stages': []
        }
        
        logger.info(f"ProductionMonitoringSystem initialized")
        logger.info(f"TensorBoard: tensorboard --logdir {self.metrics_tracker.tensorboard_dir}")
    
    def start_epoch(self, epoch: int, stage: int):
        """Start new epoch."""
        self.training_status['current_epoch'] = epoch
        self.training_status['current_stage'] = stage
        self.metrics_tracker.update_epoch(epoch)
        
        if stage != self.training_status['current_stage']:
            self.metrics_tracker.log_stage_transition(stage)
            self.training_status['current_stage'] = stage
    
    def log_training_metrics(self, 
                           metrics: Dict[str, float], 
                           model: torch.nn.Module,
                           optimizer: torch.optim.Optimizer):
        """
        Log training metrics.
        
        Args:
            metrics: Training metrics
            model: PyTorch model
            optimizer: Optimizer
        """
        # Log metrics
        self.metrics_tracker.log_metrics_dict(metrics, phase='train')
        
        # Log gradients and weights
        self.metrics_tracker.log_model_gradients(model)
        self.metrics_tracker.log_model_weights(model)
        
        # Log learning rate
        self.metrics_tracker.log_learning_rate(optimizer)
    
    def log_validation_metrics(self, metrics: Dict[str, float]) -> bool:
        """
        Log validation metrics dan check early stopping.
        
        Args:
            metrics: Validation metrics
        
        Returns:
            True if training should stop
        """
        # Log metrics
        self.metrics_tracker.log_metrics_dict(metrics, phase='val')
        
        # Check early stopping (using validation loss)
        val_loss = metrics.get('total_loss', metrics.get('val_loss', float('inf')))
        should_stop = self.early_stopping(val_loss)
        
        if should_stop:
            self.training_status['early_stopped'] = True
            logger.info(f"🛑 Early stopping triggered at epoch {self.training_status['current_epoch']}")
        
        # Update best metrics
        stage_key = f"stage_{self.training_status['current_stage']}"
        if stage_key not in self.training_status['best_metrics']:
            self.training_status['best_metrics'][stage_key] = {}
        
        for metric_name, value in metrics.items():
            current_best = self.training_status['best_metrics'][stage_key].get(metric_name, float('inf'))
            if value < current_best:  # Assuming lower is better
                self.training_status['best_metrics'][stage_key][metric_name] = value
        
        return should_stop
    
    def complete_stage(self, stage: int):
        """Mark stage as completed."""
        if stage not in self.training_status['completed_stages']:
            self.training_status['completed_stages'].append(stage)
        
        logger.info(f"✅ Stage {stage} completed")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'training_status': self.training_status,
            'early_stopping_status': self.early_stopping.get_status(),
            'recent_metrics': self.metrics_tracker.get_metrics_summary('val', last_n=5),
            'tensorboard_dir': str(self.metrics_tracker.tensorboard_dir)
        }
        
        return summary
    
    def save_training_summary(self):
        """Save training summary to file."""
        summary = self.get_training_summary()
        summary_file = self.log_dir / f"{self.experiment_name}_training_summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved training summary: {summary_file}")
    
    def close(self):
        """Close monitoring system."""
        self.training_status['end_time'] = datetime.now().isoformat()
        self.save_training_summary()
        self.metrics_tracker.close()
        
        logger.info("ProductionMonitoringSystem closed")


def create_production_monitoring_system(log_dir: str, 
                                      experiment_name: str,
                                      early_stopping_patience: int = 10) -> ProductionMonitoringSystem:
    """
    Factory function untuk membuat production monitoring system.
    
    Args:
        log_dir: Log directory
        experiment_name: Experiment name
        early_stopping_patience: Early stopping patience
    
    Returns:
        ProductionMonitoringSystem instance
    """
    return ProductionMonitoringSystem(
        log_dir=log_dir,
        experiment_name=experiment_name,
        early_stopping_patience=early_stopping_patience,
        track_gradients=True
    )