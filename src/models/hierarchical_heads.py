"""
Hierarchical Heads for Multi-Task Learning

Implements three-stage hierarchical learning:
1. Stage 1 (Binary): Precursor vs Solar Noise Detection
2. Stage 2 (Magnitude): Earthquake Magnitude Classification
3. Stage 3 (Localization): Azimuth and Distance Estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BinaryPrecursorHead(nn.Module):
    """
    Stage 1: Binary classification head for precursor vs solar noise detection.
    """
    
    def __init__(self, input_dim: int, n_stations: int = 8, dropout_rate: float = 0.2):
        """
        Initialize binary precursor detection head.
        
        Args:
            input_dim: Input feature dimension
            n_stations: Number of stations
            dropout_rate: Dropout rate
        """
        super(BinaryPrecursorHead, self).__init__()
        
        self.input_dim = input_dim
        self.n_stations = n_stations
        
        # Station-wise feature processing
        self.station_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Geophysical feature processing (Kp and Dst)
        self.geophysical_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )
        
        # Binary classifier (Input: pooled_features + geophysical_features)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim // 4 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)  # Binary output
        )
        
        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(input_dim // 4 + 16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, geophysical_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for binary precursor detection.
        
        Args:
            x: Input features (B, S, input_dim)
            geophysical_features: Kp and Dst indices (B, 2)
            
        Returns:
            Dictionary with binary predictions and confidence
        """
        B, S, _ = x.shape
        
        # Process each station
        station_features = self.station_mlp(x)  # (B, S, input_dim//4)
        
        # Global aggregation across stations
        pooled_features = station_features.mean(dim=1)  # (B, input_dim//4)
        
        # Process geophysical features
        if geophysical_features is None:
            geophysical_features = torch.zeros(B, 2, device=x.device)
        
        geo_emb = self.geophysical_mlp(geophysical_features)  # (B, 16)
        
        # Concatenate features
        combined_features = torch.cat([pooled_features, geo_emb], dim=1)  # (B, input_dim//4 + 16)
        
        # Binary classification
        binary_logits = self.classifier(combined_features)  # (B, 1)
        binary_probs = torch.sigmoid(binary_logits)
        
        # Confidence estimation
        confidence = self.confidence_head(combined_features)  # (B, 1)
        
        return {
            'binary_logits': binary_logits,
            'binary_probs': binary_probs,
            'binary_confidence': confidence
        }



class MagnitudeClassificationHead(nn.Module):
    """
    Stage 2: Magnitude classification head for earthquake magnitude estimation.
    """
    
    def __init__(self, input_dim: int, magnitude_classes: int = 5, 
                 n_stations: int = 8, dropout_rate: float = 0.2):
        """
        Initialize magnitude classification head.
        
        Args:
            input_dim: Input feature dimension
            magnitude_classes: Number of magnitude classes
            n_stations: Number of stations
            dropout_rate: Dropout rate
        """
        super(MagnitudeClassificationHead, self).__init__()
        
        self.input_dim = input_dim
        self.magnitude_classes = magnitude_classes
        self.n_stations = n_stations
        
        # Station-wise processing with attention
        self.station_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature processing
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Magnitude classifier
        self.magnitude_classifier = nn.Sequential(
            nn.Linear(input_dim // 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, magnitude_classes)
        )
        
        # Magnitude regression (for continuous estimation)
        self.magnitude_regressor = nn.Sequential(
            nn.Linear(input_dim // 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Continuous magnitude
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for magnitude classification.
        
        Args:
            x: Input features (B, S, input_dim)
            
        Returns:
            Dictionary with magnitude predictions
        """
        B, S, _ = x.shape
        
        # Apply attention across stations
        attended_features, attention_weights = self.station_attention(x, x, x)
        
        # Feature processing
        processed_features = self.feature_mlp(attended_features)  # (B, S, input_dim//2)
        
        # Global pooling
        global_features = processed_features.mean(dim=1)  # (B, input_dim//2)
        
        # Magnitude classification
        magnitude_logits = self.magnitude_classifier(global_features)  # (B, magnitude_classes)
        magnitude_probs = F.softmax(magnitude_logits, dim=1)
        
        # Magnitude regression
        magnitude_continuous = self.magnitude_regressor(global_features)  # (B, 1)
        
        return {
            'magnitude_logits': magnitude_logits,
            'magnitude_probs': magnitude_probs,
            'magnitude_continuous': magnitude_continuous,
            'magnitude_attention': attention_weights
        }


class LocalizationHead(nn.Module):
    """
    Stage 3: Localization head for azimuth and distance estimation.
    """
    
    def __init__(self, input_dim: int, n_stations: int = 8, dropout_rate: float = 0.2):
        """
        Initialize localization head.
        
        Args:
            input_dim: Input feature dimension
            n_stations: Number of stations
            dropout_rate: Dropout rate
        """
        super(LocalizationHead, self).__init__()
        
        self.input_dim = input_dim
        self.n_stations = n_stations
        
        # Spatial feature extraction
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(input_dim, input_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Station-wise localization features
        self.station_mlp = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Azimuth estimation (0-360 degrees)
        self.azimuth_head = nn.Sequential(
            nn.Linear(input_dim // 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # sin(azimuth), cos(azimuth) for circular regression
        )
        
        # Distance estimation (log scale)
        self.distance_head = nn.Sequential(
            nn.Linear(input_dim // 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # Log distance
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim // 4, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # Azimuth and distance uncertainties
            nn.Softplus()  # Ensure positive uncertainties
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for localization.
        
        Args:
            x: Input features (B, S, input_dim)
            
        Returns:
            Dictionary with localization predictions
        """
        B, S, _ = x.shape
        
        # Spatial convolution (treat stations as spatial dimension)
        x_conv = x.transpose(1, 2)  # (B, input_dim, S)
        spatial_features = self.spatial_conv(x_conv)  # (B, input_dim//2, S)
        spatial_features = spatial_features.transpose(1, 2)  # (B, S, input_dim//2)
        
        # Station-wise processing
        station_features = self.station_mlp(spatial_features)  # (B, S, input_dim//4)
        
        # Global pooling
        global_features = station_features.mean(dim=1)  # (B, input_dim//4)
        
        # Azimuth estimation (circular regression)
        azimuth_sincos = self.azimuth_head(global_features)  # (B, 2)
        azimuth_sin, azimuth_cos = azimuth_sincos[:, 0:1], azimuth_sincos[:, 1:2]
        
        # Convert to angle (0-360 degrees)
        azimuth_radians = torch.atan2(azimuth_sin, azimuth_cos)
        azimuth_degrees = (azimuth_radians * 180 / np.pi) % 360
        
        # Distance estimation
        log_distance = self.distance_head(global_features)  # (B, 1)
        distance = torch.exp(log_distance)  # Convert from log scale
        
        # Uncertainty estimation
        uncertainties = self.uncertainty_head(global_features)  # (B, 2)
        azimuth_uncertainty = uncertainties[:, 0:1]
        distance_uncertainty = uncertainties[:, 1:2]
        
        return {
            'azimuth_sincos': azimuth_sincos,
            'azimuth_degrees': azimuth_degrees,
            'azimuth_radians': azimuth_radians,
            'log_distance': log_distance,
            'distance': distance,
            'azimuth_uncertainty': azimuth_uncertainty,
            'distance_uncertainty': distance_uncertainty
        }


class HierarchicalHeads(nn.Module):
    """
    Hierarchical multi-task learning heads combining all three stages.
    """
    
    def __init__(self, input_dim: int, n_stations: int = 8, 
                 magnitude_classes: int = 5, dropout_rate: float = 0.2):
        """
        Initialize hierarchical heads.
        
        Args:
            input_dim: Input feature dimension
            n_stations: Number of stations
            magnitude_classes: Number of magnitude classes
            dropout_rate: Dropout rate
        """
        super(HierarchicalHeads, self).__init__()
        
        self.input_dim = input_dim
        self.n_stations = n_stations
        self.magnitude_classes = magnitude_classes
        
        # Stage 1: Binary precursor detection
        self.binary_head = BinaryPrecursorHead(
            input_dim=input_dim,
            n_stations=n_stations,
            dropout_rate=dropout_rate
        )
        
        # Stage 2: Magnitude classification
        self.magnitude_head = MagnitudeClassificationHead(
            input_dim=input_dim,
            magnitude_classes=magnitude_classes,
            n_stations=n_stations,
            dropout_rate=dropout_rate
        )
        
        # Stage 3: Localization
        self.localization_head = LocalizationHead(
            input_dim=input_dim,
            n_stations=n_stations,
            dropout_rate=dropout_rate
        )
        
        # Training stage control
        self.training_stage = 3  # Default to all stages
        
        logger.info(f"HierarchicalHeads initialized:")
        logger.info(f"  Input dim: {input_dim}")
        logger.info(f"  Magnitude classes: {magnitude_classes}")
        logger.info(f"  Training stage: {self.training_stage}")
    
    def set_training_stage(self, stage: int):
        """
        Set current training stage.
        
        Args:
            stage: Training stage (1, 2, or 3)
        """
        if stage not in [1, 2, 3]:
            raise ValueError("Training stage must be 1, 2, or 3")
        
        self.training_stage = stage
    
    def forward(self, x: torch.Tensor, 
                geophysical_features: Optional[torch.Tensor] = None,
                training_stage: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical heads.
        
        Args:
            x: Input features (B, S, input_dim)
            geophysical_features: Kp and Dst indices (B, 2)
            training_stage: Override training stage for this forward pass
            
        Returns:
            Dictionary with predictions from active stages
        """
        if training_stage is None:
            training_stage = self.training_stage
        
        predictions = {}
        
        # Stage 1: Binary precursor detection (always active)
        binary_outputs = self.binary_head(x, geophysical_features)
        predictions.update(binary_outputs)
        
        # Stage 2: Magnitude classification (active from stage 2)
        if training_stage >= 2:
            magnitude_outputs = self.magnitude_head(x)
            predictions.update(magnitude_outputs)
        
        # Stage 3: Localization (active from stage 3)
        if training_stage >= 3:
            localization_outputs = self.localization_head(x)
            predictions.update(localization_outputs)
        
        # Add training stage info
        predictions['training_stage'] = torch.tensor(training_stage, device=x.device)
        
        return predictions

    
    def get_stage_parameters(self, stage: int) -> list:
        """
        Get parameters for specific training stage.
        
        Args:
            stage: Training stage
            
        Returns:
            List of parameters for the stage
        """
        if stage == 1:
            return list(self.binary_head.parameters())
        elif stage == 2:
            return list(self.binary_head.parameters()) + list(self.magnitude_head.parameters())
        elif stage == 3:
            return list(self.parameters())
        else:
            raise ValueError("Invalid training stage")


if __name__ == '__main__':
    # Test hierarchical heads
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test parameters
    batch_size = 2
    n_stations = 8
    input_dim = 256
    magnitude_classes = 5
    
    # Create test input
    x = torch.randn(batch_size, n_stations, input_dim).to(device)
    
    # Create hierarchical heads
    heads = HierarchicalHeads(
        input_dim=input_dim,
        n_stations=n_stations,
        magnitude_classes=magnitude_classes
    ).to(device)
    
    # Test each training stage
    for stage in [1, 2, 3]:
        print(f"\nTesting Stage {stage}:")
        
        with torch.no_grad():
            predictions = heads(x, training_stage=stage)
        
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
    
    print(f"\nHierarchical heads test completed successfully!")