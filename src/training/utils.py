"""
Training Utilities for Spatio-Temporal Earthquake Precursor Model

Provides utility functions for:
- Model checkpointing and loading
- Training setup and configuration
- Logging and monitoring
- Data preprocessing helpers
"""
import torch
import torch.nn as nn
import numpy as np
import os
import json
import logging
import random
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import yaml
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


def setup_training(config_path: str = None, 
                  seed: int = 42,
                  device: str = None) -> Dict:
    """
    Setup training environment and configuration.
    
    Args:
        config_path: Path to training configuration file
        seed: Random seed for reproducibility
        device: Device for training (auto-detect if None)
        
    Returns:
        Training configuration dictionary
    """
    # Set random seeds for reproducibility
    set_random_seeds(seed)
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Training setup:")
    logger.info(f"  Device: {device}")
    logger.info(f"  Random seed: {seed}")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        logger.info(f"  Config loaded from: {config_path}")
    else:
        # Default configuration
        config = get_default_training_config()
        logger.info("  Using default configuration")
    
    # Add device to config
    config['device'] = device
    config['seed'] = seed
    
    return config


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducible training.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_training_config() -> Dict:
    """
    Get default training configuration.
    
    Returns:
        Default training configuration dictionary
    """
    config = {
        'model': {
            'n_stations': 8,
            'n_components': 3,
            'efficientnet_pretrained': True,
            'gnn_hidden_dim': 256,
            'gnn_num_layers': 3,
            'dropout_rate': 0.2,
            'magnitude_classes': 5
        },
        'data': {
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'load_in_memory': False,
            'magnitude_bins': [4.0, 4.5, 5.0, 5.5, 6.0]
        },
        'stage_1': {
            'epochs': 30,
            'patience': 10,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {
                'type': 'AdamW',
                'lr': 1e-4,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'CosineAnnealingLR',
                'T_max': 30,
                'eta_min': 1e-6
            },
            'save_best': True,
            'checkpoint_interval': 10
        },
        'stage_2': {
            'epochs': 40,
            'patience': 15,
            'load_previous_best': True,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {
                'type': 'AdamW',
                'lr': 5e-5,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'CosineAnnealingLR',
                'T_max': 40,
                'eta_min': 1e-6
            },
            'save_best': True,
            'checkpoint_interval': 10
        },
        'stage_3': {
            'epochs': 50,
            'patience': 20,
            'load_previous_best': True,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {
                'type': 'AdamW',
                'lr': 2e-5,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'CosineAnnealingLR',
                'T_max': 50,
                'eta_min': 1e-6
            },
            'save_best': True,
            'checkpoint_interval': 10
        },
        'loss_weights': {
            'binary_weight': 1.0,
            'magnitude_weight': 1.0,
            'localization_weight': 1.0,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0
        }
    }
    
    return config


def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler._LRScheduler,
                   epoch: int,
                   loss: float,
                   metrics: Dict,
                   path: Union[str, Path],
                   additional_info: Dict = None):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        loss: Current loss value
        metrics: Current metrics
        path: Checkpoint save path
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    torch.save(checkpoint, path)
    
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(model: nn.Module,
                   path: Union[str, Path],
                   optimizer: torch.optim.Optimizer = None,
                   scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                   device: str = 'cpu',
                   strict: bool = True) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys
        
    Returns:
        Checkpoint information dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    # Load checkpoint with weights_only=False for compatibility
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded: {path}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Loss: {checkpoint.get('loss', 'unknown')}")
    
    return checkpoint


def setup_logging(log_file: Union[str, Path] = None, 
                 level: str = 'INFO',
                 format_string: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)
    
    return logging.getLogger(__name__)


def save_training_config(config: Dict, output_dir: Union[str, Path]):
    """
    Save training configuration to file.
    
    Args:
        config: Training configuration
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / 'training_config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Training config saved: {config_path}")


def create_experiment_directory(base_dir: Union[str, Path], 
                              experiment_name: str = None) -> Path:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Custom experiment name
        
    Returns:
        Path to created experiment directory
    """
    base_dir = Path(base_dir)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'experiment_{timestamp}'
    
    experiment_dir = base_dir / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / 'checkpoints').mkdir(exist_ok=True)
    (experiment_dir / 'logs').mkdir(exist_ok=True)
    (experiment_dir / 'plots').mkdir(exist_ok=True)
    
    logger.info(f"Experiment directory created: {experiment_dir}")
    
    return experiment_dir


def backup_code(source_dir: Union[str, Path], 
               backup_dir: Union[str, Path],
               exclude_patterns: List[str] = None):
    """
    Backup source code for experiment reproducibility.
    
    Args:
        source_dir: Source code directory
        backup_dir: Backup destination
        exclude_patterns: Patterns to exclude from backup
    """
    if exclude_patterns is None:
        exclude_patterns = ['__pycache__', '*.pyc', '.git', 'outputs', 'data']
    
    source_dir = Path(source_dir)
    backup_dir = Path(backup_dir)
    
    # Create backup directory
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy source files
    for item in source_dir.rglob('*'):
        if item.is_file():
            # Check if item should be excluded
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in str(item):
                    should_exclude = True
                    break
            
            if not should_exclude:
                # Create relative path and copy
                rel_path = item.relative_to(source_dir)
                dest_path = backup_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest_path)
    
    logger.info(f"Code backup created: {backup_dir}")


def calculate_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Calculate model size statistics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb
    }


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_gpu_memory_info() -> Dict[str, float]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with GPU memory info (in MB)
    """
    if not torch.cuda.is_available():
        return {}
    
    device = torch.cuda.current_device()
    
    return {
        'allocated_mb': torch.cuda.memory_allocated(device) / (1024 ** 2),
        'cached_mb': torch.cuda.memory_reserved(device) / (1024 ** 2),
        'max_allocated_mb': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        'total_mb': torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    }


def create_training_summary(model: nn.Module,
                          train_loader,
                          config: Dict,
                          output_dir: Union[str, Path]) -> Dict:
    """
    Create comprehensive training summary.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        config: Training configuration
        output_dir: Output directory
        
    Returns:
        Training summary dictionary
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_info': calculate_model_size(model),
        'data_info': {
            'train_samples': len(train_loader.dataset),
            'batch_size': train_loader.batch_size,
            'num_batches': len(train_loader)
        },
        'config': config,
        'system_info': {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'device': config.get('device', 'cpu')
        }
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        summary['gpu_info'] = get_gpu_memory_info()
    
    # Save summary
    output_dir = Path(output_dir)
    summary_path = output_dir / 'training_summary.json'
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Training summary saved: {summary_path}")
    
    return summary


if __name__ == '__main__':
    # Test utility functions
    print("Testing training utilities...")
    
    # Test configuration
    config = get_default_training_config()
    print(f"Default config keys: {list(config.keys())}")
    
    # Test random seed setting
    set_random_seeds(42)
    print("Random seeds set")
    
    # Test time formatting
    print(f"Time formatting: {format_time(3661.5)}")
    
    print("Training utilities test completed!")