"""
Models Module for Spatio-Temporal Earthquake Precursor Analysis

This module contains deep learning models for earthquake precursor detection including:
- EfficientNet-based feature extraction
- Graph Neural Network (GNN) for spatial relationships
- Hierarchical multi-task learning heads
- Conditional loss masking for progressive learning
"""

from .spatio_temporal_model import SpatioTemporalPrecursorModel
from .gnn_fusion import GNNFusionLayer
from .hierarchical_heads import HierarchicalHeads
from .losses import ConditionalLossMasking, FocalLoss
from .utils import build_station_graph, load_station_coordinates

__all__ = [
    'SpatioTemporalPrecursorModel',
    'GNNFusionLayer',
    'HierarchicalHeads',
    'ConditionalLossMasking',
    'FocalLoss',
    'build_station_graph',
    'load_station_coordinates'
]

__version__ = '1.0.0'
__author__ = 'Spatio Precursor Project Team'