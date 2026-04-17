"""
Metrics and Evaluation for Spatio-Temporal Earthquake Precursor Model

Implements specialized metrics for multi-task learning including:
- Binary classification metrics (precursor detection)
- Multi-class classification metrics (magnitude estimation)
- Regression metrics (localization)
- Circular regression metrics (azimuth)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, 
    mean_absolute_error, r2_score
)
import logging

logger = logging.getLogger(__name__)


class PrecursorMetrics:
    """
    Comprehensive metrics for earthquake precursor detection model.
    
    Handles multi-task evaluation with stage-aware metric computation.
    """
    
    def __init__(self, magnitude_classes: int = 5, device: str = 'cpu'):
        """
        Initialize metrics calculator.
        
        Args:
            magnitude_classes: Number of magnitude classes
            device: Device for computation
        """
        self.magnitude_classes = magnitude_classes
        self.device = device
        
        # Metric storage
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = {
            'binary': {'logits': [], 'probs': [], 'targets': []},
            'magnitude': {'logits': [], 'continuous': [], 'class_targets': [], 'value_targets': []},
            'localization': {'azimuth_sincos': [], 'distance': [], 'azimuth_targets': [], 'distance_targets': []}
        }
        
        self.losses = {
            'binary_loss': [],
            'magnitude_focal_loss': [],
            'magnitude_regression_loss': [],
            'azimuth_loss': [],
            'distance_loss': [],
            'total_loss': []
        }
    
    def update(self, predictions: Dict[str, torch.Tensor], 
               targets: Dict[str, torch.Tensor],
               losses: Dict[str, torch.Tensor],
               training_stage: int = 3):
        """
        Update metrics with batch predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            losses: Computed losses
            training_stage: Current training stage
        """
        # Store losses
        for loss_name, loss_value in losses.items():
            if loss_name in self.losses:
                self.losses[loss_name].append(loss_value.item())
        
        # Binary classification (Stage 1+)
        if 'binary_logits' in predictions and 'is_precursor' in targets:
            self.predictions['binary']['logits'].append(
                predictions['binary_logits'].detach().cpu()
            )
            self.predictions['binary']['probs'].append(
                predictions['binary_probs'].detach().cpu()
            )
            self.predictions['binary']['targets'].append(
                targets['is_precursor'].detach().cpu()
            )
        
        # Magnitude classification (Stage 2+)
        if training_stage >= 2:
            if 'magnitude_logits' in predictions and 'magnitude_class' in targets:
                self.predictions['magnitude']['logits'].append(
                    predictions['magnitude_logits'].detach().cpu()
                )
                self.predictions['magnitude']['class_targets'].append(
                    targets['magnitude_class'].detach().cpu()
                )
            
            if 'magnitude_continuous' in predictions and 'magnitude_value' in targets:
                self.predictions['magnitude']['continuous'].append(
                    predictions['magnitude_continuous'].detach().cpu()
                )
                self.predictions['magnitude']['value_targets'].append(
                    targets['magnitude_value'].detach().cpu()
                )
        
        # Localization (Stage 3+)
        if training_stage >= 3:
            if 'azimuth_sincos' in predictions and 'azimuth_radians' in targets:
                self.predictions['localization']['azimuth_sincos'].append(
                    predictions['azimuth_sincos'].detach().cpu()
                )
                self.predictions['localization']['azimuth_targets'].append(
                    targets['azimuth_radians'].detach().cpu()
                )
            
            if 'distance' in predictions and 'distance' in targets:
                self.predictions['localization']['distance'].append(
                    predictions['distance'].detach().cpu()
                )
                self.predictions['localization']['distance_targets'].append(
                    targets['distance'].detach().cpu()
                )
    
    def compute_binary_metrics(self) -> Dict[str, float]:
        """
        Compute binary classification metrics.
        
        Returns:
            Dictionary of binary classification metrics
        """
        if not self.predictions['binary']['logits']:
            return {}
        
        # Concatenate all predictions and targets
        logits = torch.cat(self.predictions['binary']['logits'], dim=0)
        probs = torch.cat(self.predictions['binary']['probs'], dim=0)
        targets = torch.cat(self.predictions['binary']['targets'], dim=0)
        
        # Convert to numpy
        probs_np = probs.squeeze().numpy()
        targets_np = targets.numpy()
        
        # Binary predictions (threshold = 0.5)
        binary_preds = (probs_np > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(targets_np, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_np, binary_preds, average='binary', zero_division=0
        )
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(targets_np, probs_np)
        except ValueError:
            auc_roc = 0.0  # Handle case with only one class
        
        return {
            'binary_accuracy': accuracy,
            'binary_precision': precision,
            'binary_recall': recall,
            'binary_f1': f1,
            'binary_auc_roc': auc_roc
        }
    
    def compute_magnitude_metrics(self) -> Dict[str, float]:
        """
        Compute magnitude classification and regression metrics.
        
        Returns:
            Dictionary of magnitude metrics
        """
        metrics = {}
        
        # Classification metrics
        if self.predictions['magnitude']['logits']:
            logits = torch.cat(self.predictions['magnitude']['logits'], dim=0)
            class_targets = torch.cat(self.predictions['magnitude']['class_targets'], dim=0)
            
            # Convert to numpy
            probs = F.softmax(logits, dim=1).numpy()
            class_preds = np.argmax(probs, axis=1)
            class_targets_np = class_targets.numpy()
            
            # Classification accuracy
            class_accuracy = accuracy_score(class_targets_np, class_preds)
            
            # Per-class metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                class_targets_np, class_preds, average='weighted', zero_division=0
            )
            
            metrics.update({
                'magnitude_class_accuracy': class_accuracy,
                'magnitude_class_precision': precision,
                'magnitude_class_recall': recall,
                'magnitude_class_f1': f1
            })
        
        # Regression metrics
        if self.predictions['magnitude']['continuous']:
            continuous_preds = torch.cat(self.predictions['magnitude']['continuous'], dim=0)
            value_targets = torch.cat(self.predictions['magnitude']['value_targets'], dim=0)
            
            # Convert to numpy
            continuous_preds_np = continuous_preds.squeeze().numpy()
            value_targets_np = value_targets.numpy()
            
            # Regression metrics
            mse = mean_squared_error(value_targets_np, continuous_preds_np)
            mae = mean_absolute_error(value_targets_np, continuous_preds_np)
            r2 = r2_score(value_targets_np, continuous_preds_np)
            
            metrics.update({
                'magnitude_mse': mse,
                'magnitude_mae': mae,
                'magnitude_r2': r2,
                'magnitude_rmse': np.sqrt(mse)
            })
        
        return metrics
    
    def compute_localization_metrics(self) -> Dict[str, float]:
        """
        Compute localization metrics (azimuth and distance).
        
        Returns:
            Dictionary of localization metrics
        """
        metrics = {}
        
        # Azimuth metrics (circular regression)
        if self.predictions['localization']['azimuth_sincos']:
            azimuth_sincos = torch.cat(self.predictions['localization']['azimuth_sincos'], dim=0)
            azimuth_targets = torch.cat(self.predictions['localization']['azimuth_targets'], dim=0)
            
            # Convert predictions to angles
            azimuth_pred_radians = torch.atan2(azimuth_sincos[:, 0], azimuth_sincos[:, 1])
            
            # Convert to numpy
            azimuth_pred_np = azimuth_pred_radians.numpy()
            azimuth_targets_np = azimuth_targets.numpy()
            
            # Circular error metrics
            angular_error = self._circular_error(azimuth_pred_np, azimuth_targets_np)
            mean_angular_error = np.mean(angular_error)
            median_angular_error = np.median(angular_error)
            
            # Convert to degrees for interpretability
            mean_angular_error_deg = np.degrees(mean_angular_error)
            median_angular_error_deg = np.degrees(median_angular_error)
            
            metrics.update({
                'azimuth_mean_error_rad': mean_angular_error,
                'azimuth_median_error_rad': median_angular_error,
                'azimuth_mean_error_deg': mean_angular_error_deg,
                'azimuth_median_error_deg': median_angular_error_deg
            })
        
        # Distance metrics
        if self.predictions['localization']['distance']:
            distance_preds = torch.cat(self.predictions['localization']['distance'], dim=0)
            distance_targets = torch.cat(self.predictions['localization']['distance_targets'], dim=0)
            
            # Convert to numpy
            distance_preds_np = distance_preds.squeeze().numpy()
            distance_targets_np = distance_targets.numpy()
            
            # Distance metrics
            distance_mse = mean_squared_error(distance_targets_np, distance_preds_np)
            distance_mae = mean_absolute_error(distance_targets_np, distance_preds_np)
            distance_r2 = r2_score(distance_targets_np, distance_preds_np)
            
            # Relative error
            relative_error = np.abs(distance_preds_np - distance_targets_np) / (distance_targets_np + 1e-8)
            mean_relative_error = np.mean(relative_error)
            
            metrics.update({
                'distance_mse': distance_mse,
                'distance_mae': distance_mae,
                'distance_r2': distance_r2,
                'distance_rmse': np.sqrt(distance_mse),
                'distance_mean_relative_error': mean_relative_error
            })
        
        return metrics
    
    def _circular_error(self, pred_angles: np.ndarray, target_angles: np.ndarray) -> np.ndarray:
        """
        Calculate circular error between predicted and target angles.
        
        Args:
            pred_angles: Predicted angles in radians
            target_angles: Target angles in radians
            
        Returns:
            Angular errors in radians
        """
        # Calculate angular difference
        diff = pred_angles - target_angles
        
        # Wrap to [-π, π]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        
        # Return absolute error
        return np.abs(diff)
    
    def compute_loss_metrics(self) -> Dict[str, float]:
        """
        Compute average loss metrics.
        
        Returns:
            Dictionary of loss metrics
        """
        loss_metrics = {}
        
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                loss_metrics[f'avg_{loss_name}'] = np.mean(loss_values)
        
        return loss_metrics
    
    def compute_all_metrics(self, training_stage: int = 3) -> Dict[str, float]:
        """
        Compute all relevant metrics for current training stage.
        
        Args:
            training_stage: Current training stage
            
        Returns:
            Dictionary of all computed metrics
        """
        all_metrics = {}
        
        # Loss metrics (always computed)
        all_metrics.update(self.compute_loss_metrics())
        
        # Binary metrics (Stage 1+)
        all_metrics.update(self.compute_binary_metrics())
        
        # Magnitude metrics (Stage 2+)
        if training_stage >= 2:
            all_metrics.update(self.compute_magnitude_metrics())
        
        # Localization metrics (Stage 3+)
        if training_stage >= 3:
            all_metrics.update(self.compute_localization_metrics())
        
        return all_metrics
    
    def get_confusion_matrix(self, task: str = 'binary') -> Optional[np.ndarray]:
        """
        Get confusion matrix for classification tasks.
        
        Args:
            task: Task type ('binary' or 'magnitude')
            
        Returns:
            Confusion matrix or None if not available
        """
        if task == 'binary' and self.predictions['binary']['probs']:
            probs = torch.cat(self.predictions['binary']['probs'], dim=0)
            targets = torch.cat(self.predictions['binary']['targets'], dim=0)
            
            binary_preds = (probs.squeeze().numpy() > 0.5).astype(int)
            targets_np = targets.numpy()
            
            return confusion_matrix(targets_np, binary_preds)
        
        elif task == 'magnitude' and self.predictions['magnitude']['logits']:
            logits = torch.cat(self.predictions['magnitude']['logits'], dim=0)
            targets = torch.cat(self.predictions['magnitude']['class_targets'], dim=0)
            
            probs = F.softmax(logits, dim=1).numpy()
            class_preds = np.argmax(probs, axis=1)
            targets_np = targets.numpy()
            
            return confusion_matrix(targets_np, class_preds)
        
        return None
    
    def get_classification_report(self, task: str = 'binary') -> Optional[str]:
        """
        Get detailed classification report.
        
        Args:
            task: Task type ('binary' or 'magnitude')
            
        Returns:
            Classification report string or None if not available
        """
        if task == 'binary' and self.predictions['binary']['probs']:
            probs = torch.cat(self.predictions['binary']['probs'], dim=0)
            targets = torch.cat(self.predictions['binary']['targets'], dim=0)
            
            binary_preds = (probs.squeeze().numpy() > 0.5).astype(int)
            targets_np = targets.numpy()
            
            return classification_report(targets_np, binary_preds, 
                                       target_names=['Solar Noise', 'Precursor'])
        
        elif task == 'magnitude' and self.predictions['magnitude']['logits']:
            logits = torch.cat(self.predictions['magnitude']['logits'], dim=0)
            targets = torch.cat(self.predictions['magnitude']['class_targets'], dim=0)
            
            probs = F.softmax(logits, dim=1).numpy()
            class_preds = np.argmax(probs, axis=1)
            targets_np = targets.numpy()
            
            class_names = [f'Mag_{i}' for i in range(self.magnitude_classes)]
            return classification_report(targets_np, class_preds, 
                                       target_names=class_names, zero_division=0)
        
        return None


def calculate_stage_metrics(predictions: Dict[str, torch.Tensor],
                          targets: Dict[str, torch.Tensor],
                          training_stage: int = 3) -> Dict[str, float]:
    """
    Calculate metrics for a single batch based on training stage.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        training_stage: Current training stage
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Binary accuracy (Stage 1+)
    if 'binary_probs' in predictions and 'is_precursor' in targets:
        binary_preds = (predictions['binary_probs'] > 0.5).float()
        binary_accuracy = (binary_preds.squeeze() == targets['is_precursor']).float().mean()
        metrics['binary_accuracy'] = binary_accuracy.item()
    
    # Magnitude accuracy (Stage 2+)
    if training_stage >= 2 and 'magnitude_logits' in predictions and 'magnitude_class' in targets:
        magnitude_preds = torch.argmax(predictions['magnitude_logits'], dim=1)
        magnitude_accuracy = (magnitude_preds == targets['magnitude_class']).float().mean()
        metrics['magnitude_accuracy'] = magnitude_accuracy.item()
    
    # Localization errors (Stage 3+)
    if training_stage >= 3:
        # Azimuth error
        if 'azimuth_sincos' in predictions and 'azimuth_radians' in targets:
            pred_azimuth = torch.atan2(predictions['azimuth_sincos'][:, 0], 
                                     predictions['azimuth_sincos'][:, 1])
            azimuth_error = torch.abs(pred_azimuth - targets['azimuth_radians'])
            # Handle circular nature
            azimuth_error = torch.min(azimuth_error, 2 * np.pi - azimuth_error)
            metrics['azimuth_error_deg'] = torch.degrees(azimuth_error.mean()).item()
        
        # Distance error
        if 'distance' in predictions and 'distance' in targets:
            distance_error = torch.abs(predictions['distance'].squeeze() - targets['distance'])
            relative_error = distance_error / (targets['distance'] + 1e-8)
            metrics['distance_relative_error'] = relative_error.mean().item()
    
    return metrics


if __name__ == '__main__':
    # Test metrics calculation
    print("Testing PrecursorMetrics...")
    
    # Create dummy data
    batch_size = 16
    magnitude_classes = 5
    
    # Dummy predictions
    predictions = {
        'binary_logits': torch.randn(batch_size, 1),
        'binary_probs': torch.sigmoid(torch.randn(batch_size, 1)),
        'magnitude_logits': torch.randn(batch_size, magnitude_classes),
        'magnitude_continuous': torch.randn(batch_size, 1) + 5.0,
        'azimuth_sincos': torch.randn(batch_size, 2),
        'distance': torch.exp(torch.randn(batch_size, 1))
    }
    
    # Dummy targets
    targets = {
        'is_precursor': torch.randint(0, 2, (batch_size,)).float(),
        'magnitude_class': torch.randint(0, magnitude_classes, (batch_size,)),
        'magnitude_value': torch.rand(batch_size) * 3 + 4,
        'azimuth_radians': torch.rand(batch_size) * 2 * np.pi,
        'distance': torch.exp(torch.randn(batch_size))
    }
    
    # Dummy losses
    losses = {
        'binary_loss': torch.tensor(0.5),
        'magnitude_focal_loss': torch.tensor(1.2),
        'total_loss': torch.tensor(1.7)
    }
    
    # Test metrics
    metrics = PrecursorMetrics(magnitude_classes=magnitude_classes)
    metrics.update(predictions, targets, losses, training_stage=3)
    
    all_metrics = metrics.compute_all_metrics(training_stage=3)
    
    print("Computed metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("Metrics test completed successfully!")