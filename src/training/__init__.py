"""
Training Module for Spatio-Temporal Earthquake Precursor Model

This module provides training infrastructure including:
- PyTorch datasets and data loaders
- Training pipeline with stage-wise learning
- Model evaluation and validation
- Metrics and logging utilities
"""

from .dataset import SpatioTemporalDataset, create_data_loaders
from .trainer import SpatioTemporalTrainer
from .metrics import PrecursorMetrics
from .utils import setup_training, save_checkpoint, load_checkpoint

__all__ = [
    'SpatioTemporalDataset',
    'create_data_loaders', 
    'SpatioTemporalTrainer',
    'PrecursorMetrics',
    'setup_training',
    'save_checkpoint',
    'load_checkpoint'
]