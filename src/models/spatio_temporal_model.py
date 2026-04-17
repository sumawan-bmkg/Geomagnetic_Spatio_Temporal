"""
Spatio-Temporal Earthquake Precursor Model

Main model architecture combining:
- EfficientNet-B0 backbone for feature extraction
- GNN Fusion Layer for spatial relationships
- Hierarchical Heads for multi-task learning
- Conditional Loss Masking for progressive training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from .gnn_fusion import GNNFusionLayer
from .hierarchical_heads import HierarchicalHeads
from .losses import ConditionalLossMasking
from .utils import build_station_graph

logger = logging.getLogger(__name__)


class SpatioTemporalPrecursorModel(nn.Module):
    """
    Spatio-Temporal Earthquake Precursor Detection Model.
    
    Architecture:
    1. EfficientNet-B0 backbone for each station-component pair
    2. GNN Fusion Layer for spatial relationship learning
    3. Hierarchical Heads for multi-task prediction
    4. Conditional Loss Masking for progressive learning
    
    Input: (B, S=8, C=3, F=224, T=224)
    Output: Multi-task predictions (precursor, magnitude, localization)
    """
    
    def __init__(self,
                 n_stations: int = 8,
                 n_components: int = 3,
                 station_coordinates: np.ndarray = None,
                 efficientnet_pretrained: bool = True,
                 gnn_hidden_dim: int = 256,
                 gnn_num_layers: int = 3,
                 dropout_rate: float = 0.2,
                 magnitude_classes: int = 5,
                 device: str = 'cuda'):
        """
        Initialize Spatio-Temporal Precursor Model.
        
        Args:
            n_stations: Number of stations (default: 8)
            n_components: Number of components (default: 3)
            station_coordinates: Station coordinates for GNN (lat, lon)
            efficientnet_pretrained: Use pretrained EfficientNet weights
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            dropout_rate: Dropout rate for regularization
            magnitude_classes: Number of magnitude classes
            device: Device for computation
        """
        super(SpatioTemporalPrecursorModel, self).__init__()
        
        self.n_stations = n_stations
        self.n_components = n_components
        self.device = device
        self.magnitude_classes = magnitude_classes
        
        # Build station graph for GNN
        if station_coordinates is not None:
            self.station_graph = build_station_graph(station_coordinates)
        else:
            # Default graph (fully connected)
            self.station_graph = self._build_default_graph()
        
        # EfficientNet-B0 backbone (shared across all station-component pairs)
        self.backbone = self._build_efficientnet_backbone(efficientnet_pretrained)
        
        # Feature dimension from EfficientNet-B0
        self.feature_dim = 1280  # EfficientNet-B0 output dimension
        
        # GNN Fusion Layer for spatial relationships
        self.gnn_fusion = GNNFusionLayer(
            input_dim=self.feature_dim,
            hidden_dim=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            n_stations=n_stations,
            dropout_rate=dropout_rate
        )
        
        # Hierarchical Heads for multi-task learning
        self.hierarchical_heads = HierarchicalHeads(
            input_dim=gnn_hidden_dim,
            n_stations=n_stations,
            magnitude_classes=magnitude_classes,
            dropout_rate=dropout_rate
        )
        
        # Conditional Loss Masking
        self.loss_masking = ConditionalLossMasking()
        
        # Move station graph to device
        if hasattr(self.station_graph, 'edge_index'):
            self.station_graph.edge_index = self.station_graph.edge_index.to(device)
        
        logger.info(f"SpatioTemporalPrecursorModel initialized:")
        logger.info(f"  Stations: {n_stations}")
        logger.info(f"  Components: {n_components}")
        logger.info(f"  Feature dim: {self.feature_dim}")
        logger.info(f"  GNN hidden dim: {gnn_hidden_dim}")
        logger.info(f"  Device: {device}")
    
    def _build_efficientnet_backbone(self, pretrained: bool = True) -> nn.Module:
        """
        Build EfficientNet-B0 backbone for feature extraction.
        
        Args:
            pretrained: Use pretrained weights
            
        Returns:
            EfficientNet backbone without classifier
        """
        # Load EfficientNet-B0
        backbone = efficientnet_b0(pretrained=pretrained)
        
        # Remove classifier (keep only feature extractor)
        backbone.classifier = nn.Identity()
        
        # Modify first conv layer to accept 3-channel input (H, D, Z components)
        # Original: 3 channels (RGB) -> Keep as is since we have 3 components
        
        return backbone
    
    def _build_default_graph(self) -> object:
        """
        Build default fully connected graph for stations.
        
        Returns:
            Graph object with edge indices
        """
        # Create fully connected graph
        edge_list = []
        for i in range(self.n_stations):
            for j in range(self.n_stations):
                if i != j:
                    edge_list.append([i, j])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Simple graph object
        class Graph:
            def __init__(self, edge_index):
                self.edge_index = edge_index
        
        return Graph(edge_index)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using EfficientNet backbone.
        
        Args:
            x: Input tensor (B, S, C, F, T)
            
        Returns:
            Features tensor (B, S, feature_dim)
        """
        B, S, C, F, T = x.shape
        
        # Reshape for backbone processing: (B*S, C, F, T)
        x_reshaped = x.view(B * S, C, F, T)
        
        # Extract features using EfficientNet backbone
        features = self.backbone(x_reshaped)  # (B*S, feature_dim)
        
        # Reshape back to (B, S, feature_dim)
        features = features.view(B, S, self.feature_dim)
        
        return features
    
    def forward(self, x: torch.Tensor, 
                geophysical_features: Optional[torch.Tensor] = None,
                training_stage: int = 3,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (B, S=8, C=3, F=224, T=224)
            geophysical_features: Kp and Dst indices (B, 2)
            training_stage: Current training stage (1, 2, or 3)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary of predictions and optionally features
        """
        # Extract features using EfficientNet backbone
        features = self.extract_features(x)  # (B, S, feature_dim)
        
        # Apply GNN fusion for spatial relationships (now includes SE block)
        fused_features = self.gnn_fusion(features, self.station_graph)  # (B, S, gnn_hidden_dim)
        
        # Generate predictions using hierarchical heads (now includes Kp/Dst)
        predictions = self.hierarchical_heads(fused_features, geophysical_features, training_stage)
        
        # Prepare output
        output = predictions
        
        if return_features:
            output['backbone_features'] = features
            output['fused_features'] = fused_features
        
        return output

    
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    training_stage: int = 3) -> Dict[str, torch.Tensor]:
        """
        Compute conditional loss based on training stage.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            training_stage: Current training stage
            
        Returns:
            Dictionary of losses
        """
        return self.loss_masking(predictions, targets, training_stage)
    
    def get_model_summary(self) -> Dict:
        """
        Get model architecture summary.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'n_stations': self.n_stations,
            'n_components': self.n_components,
            'feature_dim': self.feature_dim,
            'magnitude_classes': self.magnitude_classes,
            'backbone': 'EfficientNet-B0',
            'gnn_layers': self.gnn_fusion.num_layers,
            'device': self.device
        }
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone parameters.
        
        Args:
            freeze: Whether to freeze backbone
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        logger.info(f"Backbone {'frozen' if freeze else 'unfrozen'}")
    
    def set_training_stage(self, stage: int):
        """
        Set training stage for conditional learning.
        
        Args:
            stage: Training stage (1, 2, or 3)
        """
        if stage not in [1, 2, 3]:
            raise ValueError("Training stage must be 1, 2, or 3")
        
        self.hierarchical_heads.set_training_stage(stage)
        logger.info(f"Training stage set to: {stage}")


def create_model(station_coordinates_path: str = None,
                config: Dict = None) -> SpatioTemporalPrecursorModel:
    """
    Factory function to create model with configuration.
    
    Args:
        station_coordinates_path: Path to station coordinates CSV
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    # Default configuration
    default_config = {
        'n_stations': 8,
        'n_components': 3,
        'efficientnet_pretrained': True,
        'gnn_hidden_dim': 256,
        'gnn_num_layers': 3,
        'dropout_rate': 0.2,
        'magnitude_classes': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Update with provided config
    if config:
        default_config.update(config)
    
    # Load station coordinates if provided
    station_coordinates = None
    if station_coordinates_path:
        from .utils import load_station_coordinates
        station_coordinates = load_station_coordinates(station_coordinates_path)
    
    # Create model
    model = SpatioTemporalPrecursorModel(
        station_coordinates=station_coordinates,
        **default_config
    )
    
    return model


if __name__ == '__main__':
    # Test model creation and forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = create_model(config={'device': device})
    model = model.to(device)
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 8, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(x, training_stage=3, return_features=True)
    
    print("Model Summary:")
    summary = model.get_model_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nPredictions:")
    for key, tensor in predictions.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: {tensor.shape}")
    
    print(f"\nModel test completed successfully!")