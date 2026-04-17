"""
Loss Functions for Hierarchical Multi-Task Learning

Implements:
- Focal Loss for magnitude classification
- Conditional Loss Masking for progressive training
- Circular regression loss for azimuth estimation
- Uncertainty-aware losses
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss for addressing class imbalance in magnitude classification.
    Particularly effective for large earthquake events (M >= 6.0).
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Enhanced for geophysical applications with large event focus.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None,
                 large_event_boost: float = 2.0):
        """
        Initialize Enhanced Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            class_weights: Optional class weights tensor
            large_event_boost: Additional boost for large events (M >= 6.0)
        """
        super(FocalLoss, self).__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.large_event_boost = large_event_boost
        
        # Enhanced class weights for large events
        if class_weights is None:
            # Default weights: [3.0-4.0M, 4.0-4.5M, 4.5-5.0M, 5.0-5.5M, 5.5M+]
            # Higher weights for larger magnitudes
            self.class_weights = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.5])
        else:
            self.class_weights = class_weights
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute enhanced focal loss with large event focus.
        
        Args:
            inputs: Predicted logits (B, num_classes)
            targets: Ground truth labels (B,)
            
        Returns:
            Enhanced focal loss value
        """
        # Compute cross entropy with class weights
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        
        # Compute probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply additional boost for large events (class 4 = M >= 5.5)
        large_event_mask = (targets >= 4)  # Assuming class 4 is large events
        if large_event_mask.any():
            focal_loss[large_event_mask] *= self.large_event_boost
        
        return focal_loss.mean()


class CircularRegressionLoss(nn.Module):
    """
    Loss function for circular regression (azimuth estimation).
    Uses sin/cos representation to handle circular nature of angles.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize circular regression loss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(CircularRegressionLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, pred_sincos: torch.Tensor, target_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute circular regression loss.
        
        Args:
            pred_sincos: Predicted sin/cos values (B, 2)
            target_angles: Target angles in radians (B,)
            
        Returns:
            Circular regression loss
        """
        # Convert target angles to sin/cos
        target_sin = torch.sin(target_angles)
        target_cos = torch.cos(target_angles)
        target_sincos = torch.stack([target_sin, target_cos], dim=1)
        
        # Compute MSE loss in sin/cos space
        loss = F.mse_loss(pred_sincos, target_sincos, reduction='none')
        
        # Sum over sin/cos dimensions
        loss = loss.sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class UncertaintyLoss(nn.Module):
    """
    Uncertainty-aware loss that incorporates prediction uncertainty.
    """
    
    def __init__(self, loss_type: str = 'mse'):
        """
        Initialize uncertainty loss.
        
        Args:
            loss_type: Base loss type ('mse', 'mae')
        """
        super(UncertaintyLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty-aware loss.
        
        Args:
            predictions: Model predictions (B, ...)
            targets: Ground truth targets (B, ...)
            uncertainties: Prediction uncertainties (B, ...)
            
        Returns:
            Uncertainty-weighted loss
        """
        # Compute base loss
        if self.loss_type == 'mse':
            base_loss = (predictions - targets) ** 2
        elif self.loss_type == 'mae':
            base_loss = torch.abs(predictions - targets)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        # Uncertainty weighting: loss / (2 * sigma^2) + log(sigma)
        # This encourages both accuracy and appropriate uncertainty estimation
        weighted_loss = base_loss / (2 * uncertainties ** 2) + torch.log(uncertainties)
        
        return weighted_loss.mean()


class ConditionalLossMasking(nn.Module):
    """
    Conditional Loss Masking for hierarchical progressive training.
    
    Implements stage-wise training:
    - Stage 1: Only binary classification loss
    - Stage 2: Binary + magnitude classification losses
    - Stage 3: All losses (binary + magnitude + localization)
    """
    
    def __init__(self, 
                 binary_weight: float = 1.0,
                 magnitude_weight: float = 1.0,
                 localization_weight: float = 1.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        """
        Initialize conditional loss masking.
        
        Args:
            binary_weight: Weight for binary classification loss
            magnitude_weight: Weight for magnitude classification loss
            localization_weight: Weight for localization loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super(ConditionalLossMasking, self).__init__()
        
        self.binary_weight = binary_weight
        self.magnitude_weight = magnitude_weight
        self.localization_weight = localization_weight
        
        # Loss functions
        self.binary_loss_fn = nn.BCEWithLogitsLoss()
        self.focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.magnitude_regression_loss_fn = nn.MSELoss()
        self.circular_loss_fn = CircularRegressionLoss()
        self.distance_loss_fn = nn.MSELoss()
        self.uncertainty_loss_fn = UncertaintyLoss()
        
        logger.info(f"ConditionalLossMasking initialized:")
        logger.info(f"  Binary weight: {binary_weight}")
        logger.info(f"  Magnitude weight: {magnitude_weight}")
        logger.info(f"  Localization weight: {localization_weight}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                training_stage: int = 3) -> Dict[str, torch.Tensor]:
        """
        Compute conditional losses based on training stage.
        
        Args:
            predictions: Model predictions dictionary
            targets: Ground truth targets dictionary
            training_stage: Current training stage (1, 2, or 3)
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}
        
        # Initialize total_loss as tensor with proper device
        device = next(iter(predictions.values())).device
        total_loss = torch.tensor(0.0, device=device)
        
        # Stage 1: Binary classification loss (always computed)
        if 'binary_logits' in predictions and 'is_precursor' in targets:
            binary_loss = self.binary_loss_fn(
                predictions['binary_logits'].squeeze(),
                targets['is_precursor'].float()
            )
            losses['binary_loss'] = binary_loss
            total_loss += self.binary_weight * binary_loss
        
        # Stage 2: Magnitude classification losses (from stage 2)
        if training_stage >= 2:
            # Focal loss for magnitude classification
            if 'magnitude_logits' in predictions and 'magnitude_class' in targets:
                magnitude_focal_loss = self.focal_loss_fn(
                    predictions['magnitude_logits'],
                    targets['magnitude_class']
                )
                losses['magnitude_focal_loss'] = magnitude_focal_loss
                total_loss += self.magnitude_weight * magnitude_focal_loss
            
            # Regression loss for continuous magnitude
            if 'magnitude_continuous' in predictions and 'magnitude_value' in targets:
                magnitude_regression_loss = self.magnitude_regression_loss_fn(
                    predictions['magnitude_continuous'].squeeze(),
                    targets['magnitude_value']
                )
                losses['magnitude_regression_loss'] = magnitude_regression_loss
                total_loss += self.magnitude_weight * 0.5 * magnitude_regression_loss
        
        # Stage 3: Localization losses (from stage 3)
        if training_stage >= 3:
            # Circular regression loss for azimuth
            if 'azimuth_sincos' in predictions and 'azimuth_radians' in targets:
                azimuth_loss = self.circular_loss_fn(
                    predictions['azimuth_sincos'],
                    targets['azimuth_radians']
                )
                losses['azimuth_loss'] = azimuth_loss
                total_loss += self.localization_weight * azimuth_loss
            
            # Distance regression loss (log scale)
            if 'log_distance' in predictions and 'log_distance' in targets:
                distance_loss = self.distance_loss_fn(
                    predictions['log_distance'].squeeze(),
                    targets['log_distance']
                )
                losses['distance_loss'] = distance_loss
                total_loss += self.localization_weight * distance_loss
            
            # Uncertainty-aware losses
            if ('azimuth_uncertainty' in predictions and 
                'azimuth_radians' in targets and 
                'azimuth_degrees' in predictions):
                
                # Convert azimuth predictions back to radians for uncertainty loss
                azimuth_pred_radians = predictions['azimuth_radians']
                azimuth_uncertainty_loss = self.uncertainty_loss_fn(
                    azimuth_pred_radians.squeeze(),
                    targets['azimuth_radians'],
                    predictions['azimuth_uncertainty'].squeeze()
                )
                losses['azimuth_uncertainty_loss'] = azimuth_uncertainty_loss
                total_loss += self.localization_weight * 0.1 * azimuth_uncertainty_loss
            
            if ('distance_uncertainty' in predictions and 
                'distance' in targets and 
                'distance' in predictions):
                
                distance_uncertainty_loss = self.uncertainty_loss_fn(
                    predictions['distance'].squeeze(),
                    targets['distance'],
                    predictions['distance_uncertainty'].squeeze()
                )
                losses['distance_uncertainty_loss'] = distance_uncertainty_loss
                total_loss += self.localization_weight * 0.1 * distance_uncertainty_loss
        
        # Add total loss
        losses['total_loss'] = total_loss
        
        # Ensure total_loss has gradients if any component losses exist
        if total_loss.item() == 0.0:
            # Create a dummy loss with gradients to prevent backward errors
            dummy_pred = next(iter(predictions.values()))
            total_loss = torch.sum(dummy_pred * 0.0)  # This will have gradients but be zero
            losses['total_loss'] = total_loss
        
        losses['training_stage'] = torch.tensor(training_stage, device=total_loss.device)
        
        return losses
    
    def get_stage_weights(self, training_stage: int) -> Dict[str, float]:
        """
        Get loss weights for current training stage.
        
        Args:
            training_stage: Current training stage
            
        Returns:
            Dictionary of loss weights
        """
        weights = {'binary': self.binary_weight}
        
        if training_stage >= 2:
            weights['magnitude'] = self.magnitude_weight
        
        if training_stage >= 3:
            weights['localization'] = self.localization_weight
        
        return weights


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts weights based on task performance.
    """
    
    def __init__(self, num_tasks: int = 3, temperature: float = 2.0):
        """
        Initialize adaptive loss weighting.
        
        Args:
            num_tasks: Number of tasks
            temperature: Temperature parameter for softmax
        """
        super(AdaptiveLossWeighting, self).__init__()
        
        self.num_tasks = num_tasks
        self.temperature = temperature
        
        # Learnable task weights
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
        # Moving averages of task losses
        self.register_buffer('loss_history', torch.zeros(num_tasks, 100))
        self.register_buffer('history_idx', torch.tensor(0))
    
    def forward(self, task_losses: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights and weighted loss.
        
        Args:
            task_losses: Individual task losses (num_tasks,)
            
        Returns:
            Weighted total loss
        """
        # Update loss history
        idx = self.history_idx % 100
        self.loss_history[:, idx] = task_losses.detach()
        self.history_idx += 1
        
        # Compute adaptive weights
        if self.history_idx > 10:  # Wait for some history
            # Use relative loss rates for weighting
            recent_losses = self.loss_history[:, :min(self.history_idx, 100)]
            loss_rates = recent_losses.mean(dim=1)
            
            # Inverse weighting (higher loss gets lower weight)
            adaptive_weights = F.softmax(-loss_rates / self.temperature, dim=0)
        else:
            # Use uniform weights initially
            adaptive_weights = F.softmax(self.task_weights / self.temperature, dim=0)
        
        # Compute weighted loss
        weighted_loss = (adaptive_weights * task_losses).sum()
        
        return weighted_loss


if __name__ == '__main__':
    # Test loss functions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    batch_size = 4
    num_classes = 5
    
    # Test Focal Loss
    print("Testing Focal Loss:")
    focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
    
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    loss = focal_loss(logits, targets)
    print(f"  Focal loss: {loss.item():.4f}")
    
    # Test Circular Regression Loss
    print("\nTesting Circular Regression Loss:")
    circular_loss = CircularRegressionLoss()
    
    pred_sincos = torch.randn(batch_size, 2)
    target_angles = torch.rand(batch_size) * 2 * np.pi
    
    loss = circular_loss(pred_sincos, target_angles)
    print(f"  Circular loss: {loss.item():.4f}")
    
    # Test Conditional Loss Masking
    print("\nTesting Conditional Loss Masking:")
    loss_masking = ConditionalLossMasking()
    
    # Create dummy predictions and targets
    predictions = {
        'binary_logits': torch.randn(batch_size, 1),
        'magnitude_logits': torch.randn(batch_size, num_classes),
        'magnitude_continuous': torch.randn(batch_size, 1),
        'azimuth_sincos': torch.randn(batch_size, 2),
        'azimuth_radians': torch.rand(batch_size) * 2 * np.pi,
        'azimuth_degrees': torch.rand(batch_size) * 360,
        'log_distance': torch.randn(batch_size, 1),
        'distance': torch.exp(torch.randn(batch_size, 1)),
        'azimuth_uncertainty': torch.rand(batch_size, 1) + 0.1,
        'distance_uncertainty': torch.rand(batch_size, 1) + 0.1
    }
    
    targets = {
        'is_precursor': torch.randint(0, 2, (batch_size,)),
        'magnitude_class': torch.randint(0, num_classes, (batch_size,)),
        'magnitude_value': torch.rand(batch_size) * 3 + 4,  # Magnitude 4-7
        'azimuth_radians': torch.rand(batch_size) * 2 * np.pi,
        'log_distance': torch.randn(batch_size),
        'distance': torch.exp(torch.randn(batch_size))
    }
    
    # Test each training stage
    for stage in [1, 2, 3]:
        losses = loss_masking(predictions, targets, training_stage=stage)
        print(f"  Stage {stage} total loss: {losses['total_loss'].item():.4f}")
    
    print(f"\nLoss functions test completed successfully!")