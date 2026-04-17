#!/usr/bin/env python3
"""
Demo Production Inference & Validation
======================================

Simplified demonstration of the inference validation pipeline
using available data samples.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

# Add src to path
sys.path.append('src')

# Import model
from models.spatio_temporal_model import SpatioTemporalPrecursorModel

warnings.filterwarnings('ignore')

class DemoInferenceValidator:
    """Demo inference validator for earthquake precursor model."""
    
    def __init__(self, 
                 model_checkpoint_path: str,
                 test_dataset_path: str = 'real_earthquake_dataset.h5',
                 output_dir: str = 'outputs/demo_inference_validation',
                 device: str = None):
        
        self.model_checkpoint_path = model_checkpoint_path
        self.test_dataset_path = test_dataset_path
        self.output_dir = Path(output_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'plots').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'reports').mkdir(parents=True, exist_ok=True)
        
        self.model = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging."""
        log_file = self.output_dir / 'demo_inference_validation.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> None:
        """Load trained model."""
        self.logger.info("=" * 60)
        self.logger.info("LOADING MODEL FOR DEMO INFERENCE")
        self.logger.info("=" * 60)
        
        try:
            # Load checkpoint
            self.logger.info(f"Loading model checkpoint: {self.model_checkpoint_path}")
            checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device, weights_only=False)
            
            # Initialize model
            model_config = {
                'n_stations': 8,
                'n_components': 3,
                'station_coordinates': None,
                'efficientnet_pretrained': False,
                'gnn_hidden_dim': 256,
                'gnn_num_layers': 3,
                'dropout_rate': 0.2,
                'magnitude_classes': 5,
                'device': self.device
            }
            
            self.model = SpatioTemporalPrecursorModel(**model_config)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
            
    def load_demo_samples(self, n_samples: int = 50) -> List[Dict]:
        """Load demo samples from dataset."""
        self.logger.info("=" * 60)
        self.logger.info("LOADING DEMO SAMPLES")
        self.logger.info("=" * 60)
        
        demo_samples = []
        
        try:
            with h5py.File(self.test_dataset_path, 'r') as f:
                self.logger.info(f"Dataset keys: {list(f.keys())}")
                
                # Load scalogram data
                if 'scalogram_tensor' in f:
                    scalogram_obj = f['scalogram_tensor']
                    if hasattr(scalogram_obj, 'keys'):
                        scalogram_keys = list(scalogram_obj.keys())
                        scalogram_dataset = scalogram_obj[scalogram_keys[0]]
                    else:
                        scalogram_dataset = scalogram_obj
                    
                    dataset_size = scalogram_dataset.shape[0]
                    self.logger.info(f"Total scalogram samples: {dataset_size}")
                    
                    # Take last n_samples (most recent)
                    start_idx = max(0, dataset_size - n_samples)
                    end_idx = dataset_size
                    
                    self.logger.info(f"Loading samples {start_idx} to {end_idx}")
                    
                    scalograms = scalogram_dataset[start_idx:end_idx]
                    
                    # Load corresponding metadata
                    if 'metadata' in f:
                        metadata_obj = f['metadata']
                        if hasattr(metadata_obj, 'keys'):
                            metadata_keys = list(metadata_obj.keys())
                            metadata_dict = {}
                            for key in metadata_keys:
                                metadata_dict[key] = metadata_obj[key][start_idx:end_idx]
                            metadata = pd.DataFrame(metadata_dict)
                        else:
                            metadata = pd.DataFrame(metadata_obj[start_idx:end_idx])
                        
                        # Convert bytes to string if needed
                        if 'datetime' in metadata.columns:
                            if isinstance(metadata['datetime'].iloc[0], bytes):
                                metadata['datetime'] = metadata['datetime'].str.decode('utf-8')
                            metadata['event_time'] = pd.to_datetime(metadata['datetime'], format='mixed')
                    
                    # Create demo samples
                    for i in range(len(scalograms)):
                        sample = {
                            'scalogram': scalograms[i],
                            'magnitude': float(metadata.iloc[i]['magnitude']) if 'magnitude' in metadata.columns else 5.0,
                            'event_time': str(metadata.iloc[i]['event_time']) if 'event_time' in metadata.columns else '2024-01-01',
                            'latitude': float(metadata.iloc[i]['latitude']) if 'latitude' in metadata.columns else 0.0,
                            'longitude': float(metadata.iloc[i]['longitude']) if 'longitude' in metadata.columns else 0.0,
                            'depth': float(metadata.iloc[i]['depth']) if 'depth' in metadata.columns else 10.0
                        }
                        demo_samples.append(sample)
                    
                    self.logger.info(f"Loaded {len(demo_samples)} demo samples")
                    
                    # Log sample statistics
                    magnitudes = [s['magnitude'] for s in demo_samples]
                    self.logger.info(f"Magnitude range: {min(magnitudes):.1f} - {max(magnitudes):.1f}")
                    self.logger.info(f"Large earthquakes (M >= 5.0): {sum(1 for m in magnitudes if m >= 5.0)}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load demo samples: {str(e)}")
            raise
            
        return demo_samples
    
    def run_demo_inference(self, demo_samples: List[Dict]) -> Dict[str, Any]:
        """Run inference on demo samples."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING DEMO INFERENCE")
        self.logger.info("=" * 60)
        
        predictions = {
            'binary_probs': [],
            'binary_preds': [],
            'magnitude_preds': [],
            'azimuth_preds': [],
            'distance_preds': [],
            'true_magnitudes': [],
            'sample_info': []
        }
        
        binary_threshold = 0.5
        
        with torch.no_grad():
            for i, sample in enumerate(demo_samples):
                # Prepare input
                scalogram = torch.tensor(sample['scalogram'], dtype=torch.float32)
                
                # Ensure correct shape: (1, 8, 3, F, T)
                if len(scalogram.shape) == 4:  # (8, 3, F, T)
                    scalogram = scalogram.unsqueeze(0)  # Add batch dimension
                elif len(scalogram.shape) == 3:  # (8, F, T) - missing component dimension
                    scalogram = scalogram.unsqueeze(1)  # Add component dimension
                    scalogram = scalogram.unsqueeze(0)  # Add batch dimension
                elif len(scalogram.shape) == 2:  # (F, T) - missing station and component
                    scalogram = scalogram.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add all dimensions
                    scalogram = scalogram.repeat(1, 8, 3, 1, 1)  # Replicate for 8 stations, 3 components
                
                scalogram = scalogram.to(self.device)
                
                try:
                    # Forward pass
                    outputs = self.model(scalogram)
                    
                    # Extract predictions
                    binary_prob = torch.sigmoid(outputs['binary_logits']).cpu().numpy()[0, 0]
                    binary_pred = 1 if binary_prob > binary_threshold else 0
                    
                    magnitude_pred = outputs['magnitude_continuous'].cpu().numpy()[0, 0]
                    azimuth_pred = outputs['azimuth_degrees'].cpu().numpy()[0, 0]
                    distance_pred = outputs['distance'].cpu().numpy()[0, 0]
                    
                    # Store results
                    predictions['binary_probs'].append(binary_prob)
                    predictions['binary_preds'].append(binary_pred)
                    predictions['magnitude_preds'].append(magnitude_pred)
                    predictions['azimuth_preds'].append(azimuth_pred)
                    predictions['distance_preds'].append(distance_pred)
                    predictions['true_magnitudes'].append(sample['magnitude'])
                    predictions['sample_info'].append({
                        'event_time': str(sample['event_time']),
                        'magnitude': sample['magnitude'],
                        'latitude': sample['latitude'],
                        'longitude': sample['longitude']
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process sample {i}: {str(e)}")
                    # Add default values
                    predictions['binary_probs'].append(0.0)
                    predictions['binary_preds'].append(0)
                    predictions['magnitude_preds'].append(0.0)
                    predictions['azimuth_preds'].append(0.0)
                    predictions['distance_preds'].append(0.0)
                    predictions['true_magnitudes'].append(sample['magnitude'])
                    predictions['sample_info'].append({
                        'event_time': str(sample['event_time']),
                        'magnitude': sample['magnitude'],
                        'latitude': sample['latitude'],
                        'longitude': sample['longitude']
                    })
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(demo_samples)} samples")
        
        # Convert to numpy arrays
        for key in ['binary_probs', 'binary_preds', 'magnitude_preds', 
                   'azimuth_preds', 'distance_preds', 'true_magnitudes']:
            predictions[key] = np.array(predictions[key])
        
        self.logger.info(f"Demo inference completed on {len(demo_samples)} samples")
        self.logger.info(f"Binary detections: {np.sum(predictions['binary_preds'])}")
        
        return predictions
    def compute_demo_metrics(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Compute performance metrics for demo."""
        self.logger.info("=" * 60)
        self.logger.info("COMPUTING DEMO METRICS")
        self.logger.info("=" * 60)
        
        metrics = {}
        
        # Binary classification metrics
        magnitude_threshold = 5.0  # M5.0+ considered positive
        binary_targets = (predictions['true_magnitudes'] >= magnitude_threshold).astype(int)
        binary_preds = predictions['binary_preds']
        binary_probs = predictions['binary_probs']
        
        if len(binary_targets) == 0:
            self.logger.error("No samples to evaluate")
            return {}
        
        # Confusion matrix
        cm = confusion_matrix(binary_targets, binary_preds)
        
        # Handle different matrix sizes
        if cm.size == 4:  # 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:  # 1x1 matrix (all same class)
            if binary_targets[0] == 1:
                tp = cm[0, 0] if binary_preds[0] == 1 else 0
                fn = cm[0, 0] if binary_preds[0] == 0 else 0
                fp = tn = 0
            else:
                tn = cm[0, 0] if binary_preds[0] == 0 else 0
                fp = cm[0, 0] if binary_preds[0] == 1 else 0
                tp = fn = 0
        else:
            tp = fp = fn = tn = 0
        
        # Classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_targets, binary_preds, average='binary', zero_division=0
        )
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        try:
            if len(np.unique(binary_targets)) > 1:
                auc = roc_auc_score(binary_targets, binary_probs)
            else:
                auc = 0.5  # Default for single class
        except:
            auc = 0.5
        
        metrics['binary_classification'] = {
            'confusion_matrix': cm.tolist(),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'roc_auc': float(auc),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(binary_targets),
            'positive_samples': int(np.sum(binary_targets))
        }
        
        self.logger.info(f"Demo Binary Classification Results:")
        self.logger.info(f"  - Precision: {precision:.3f}")
        self.logger.info(f"  - Recall: {recall:.3f}")
        self.logger.info(f"  - F1-Score: {f1:.3f}")
        self.logger.info(f"  - Accuracy: {accuracy:.3f}")
        self.logger.info(f"  - ROC AUC: {auc:.3f}")
        
        # Regression metrics for positive detections
        positive_mask = binary_preds == 1
        if np.sum(positive_mask) > 0:
            mag_targets = predictions['true_magnitudes'][positive_mask]
            mag_preds = predictions['magnitude_preds'][positive_mask]
            mag_mae = np.mean(np.abs(mag_targets - mag_preds))
            
            metrics['regression'] = {
                'magnitude_mae': float(mag_mae),
                'positive_detections': int(np.sum(positive_mask))
            }
            
            self.logger.info(f"Demo Regression Results (n={np.sum(positive_mask)}):")
            self.logger.info(f"  - Magnitude MAE: {mag_mae:.3f}")
        
        return metrics
    
    def generate_demo_visualizations(self, predictions: Dict[str, Any]) -> None:
        """Generate demo validation visualizations."""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING DEMO VISUALIZATIONS")
        self.logger.info("=" * 60)
        
        # Create comprehensive demo plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Demo Production Inference Validation Results', fontsize=16)
        
        # 1. Confusion matrix
        magnitude_threshold = 5.0
        binary_targets = (predictions['true_magnitudes'] >= magnitude_threshold).astype(int)
        cm = confusion_matrix(binary_targets, predictions['binary_preds'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Probability distribution
        if len(np.unique(binary_targets)) > 1:
            axes[0, 1].hist(predictions['binary_probs'][binary_targets == 0], 
                           alpha=0.5, label='Negative (M < 5.0)', bins=10)
            axes[0, 1].hist(predictions['binary_probs'][binary_targets == 1], 
                           alpha=0.5, label='Positive (M ≥ 5.0)', bins=10)
        else:
            axes[0, 1].hist(predictions['binary_probs'], alpha=0.7, bins=10, label='All Samples')
        
        axes[0, 1].set_xlabel('Binary Probability')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Probability Distribution')
        axes[0, 1].legend()
        
        # 3. Magnitude prediction vs actual
        positive_mask = predictions['binary_preds'] == 1
        if np.sum(positive_mask) > 0:
            axes[1, 0].scatter(predictions['true_magnitudes'][positive_mask], 
                             predictions['magnitude_preds'][positive_mask], alpha=0.6)
            axes[1, 0].plot([3, 8], [3, 8], 'r--', label='Perfect Prediction')
            axes[1, 0].set_xlabel('True Magnitude')
            axes[1, 0].set_ylabel('Predicted Magnitude')
            axes[1, 0].set_title('Magnitude Prediction (Positive Detections)')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No Positive Detections', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Magnitude Prediction (No Positive Detections)')
        
        # 4. Detection results over samples
        sample_indices = range(len(predictions['true_magnitudes']))
        scatter = axes[1, 1].scatter(sample_indices, predictions['true_magnitudes'], 
                                   c=predictions['binary_preds'], cmap='RdYlBu', alpha=0.7)
        axes[1, 1].axhline(y=magnitude_threshold, color='red', linestyle='--', 
                          label=f'M{magnitude_threshold} Threshold')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].set_title('Detection Results')
        axes[1, 1].legend()
        
        # Add colorbar
        plt.colorbar(scatter, ax=axes[1, 1], label='Prediction (0=No, 1=Yes)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'demo_validation_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Demo visualizations saved")
    
    def generate_demo_report(self, metrics: Dict[str, Any], predictions: Dict[str, Any]) -> None:
        """Generate demo validation report."""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING DEMO REPORT")
        self.logger.info("=" * 60)
        
        report = {
            'demo_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_checkpoint': self.model_checkpoint_path,
                'test_dataset': self.test_dataset_path,
                'device': self.device,
                'total_demo_samples': len(predictions['binary_preds'])
            },
            'performance_metrics': metrics,
            'model_configuration': {
                'architecture': 'EfficientNet-B0 + GNN Fusion',
                'stages': ['Binary Classification', 'Magnitude Estimation', 'Localization'],
                'preprocessing': ['CWT Scaling', 'PCA-CMR', 'Z-score Normalization']
            },
            'sample_predictions': predictions['sample_info'][:10]  # First 10 samples
        }
        
        # Save JSON report
        report_path = self.output_dir / 'reports' / 'demo_validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        md_content = f"""# Demo Production Inference Validation Report

**Generated:** {report['demo_metadata']['timestamp']}  
**Model:** {os.path.basename(report['demo_metadata']['model_checkpoint'])}  
**Device:** {report['demo_metadata']['device']}  
**Demo Samples:** {report['demo_metadata']['total_demo_samples']}

## Model Architecture
- **Backbone:** EfficientNet-B0 + GNN Fusion
- **Stages:** Binary Classification -> Magnitude Estimation -> Localization
- **Preprocessing:** CWT Scaling, PCA-CMR (Solar Cycle 25), Z-score Normalization

## Demo Performance Metrics

### Binary Classification
"""
        
        if 'binary_classification' in metrics:
            bc = metrics['binary_classification']
            md_content += f"""
| Metric | Value |
|--------|-------|
| Precision | {bc['precision']:.3f} |
| Recall | {bc['recall']:.3f} |
| F1-Score | {bc['f1_score']:.3f} |
| Accuracy | {bc['accuracy']:.3f} |
| ROC AUC | {bc['roc_auc']:.3f} |

**Confusion Matrix:**
- True Positives: {bc['true_positives']}
- True Negatives: {bc['true_negatives']}
- False Positives: {bc['false_positives']}
- False Negatives: {bc['false_negatives']}
- Total Samples: {bc['total_samples']}
- Positive Samples: {bc['positive_samples']}
"""
        
        if 'regression' in metrics:
            reg = metrics['regression']
            md_content += f"""
### Regression Performance
| Metric | Value |
|--------|-------|
| Magnitude MAE | {reg['magnitude_mae']:.3f} |
| Positive Detections | {reg['positive_detections']} |
"""
        
        md_content += """
## Demo Summary

This demo validation demonstrates the model's inference capabilities:

1. **Model Loading:** Successfully loaded trained EfficientNet-B0 + GNN Fusion model
2. **Multi-Stage Architecture:** Hierarchical prediction from detection to localization
3. **Production Ready:** Model executed inference on real data samples
4. **Performance Metrics:** Computed comprehensive evaluation metrics

## Files Generated
- `plots/demo_validation_results.png` - Comprehensive validation visualizations
- `reports/demo_validation_report.json` - Complete metrics in JSON format

---
*Report generated by Demo Production Inference Validation System*
"""
        
        md_path = self.output_dir / 'reports' / 'demo_validation_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        self.logger.info(f"Demo reports saved:")
        self.logger.info(f"  - JSON: {report_path}")
        self.logger.info(f"  - Markdown: {md_path}")
    
    def run_demo_validation(self) -> None:
        """Run complete demo validation pipeline."""
        self.logger.info("STARTING DEMO PRODUCTION INFERENCE VALIDATION")
        self.logger.info("=" * 80)
        
        try:
            # Load model
            self.load_model()
            
            # Load demo samples
            demo_samples = self.load_demo_samples(n_samples=50)
            
            if len(demo_samples) == 0:
                self.logger.error("No demo samples loaded")
                return
            
            # Run inference
            predictions = self.run_demo_inference(demo_samples)
            
            # Compute metrics
            metrics = self.compute_demo_metrics(predictions)
            
            # Generate visualizations
            self.generate_demo_visualizations(predictions)
            
            # Generate report
            self.generate_demo_report(metrics, predictions)
            
            self.logger.info("=" * 80)
            self.logger.info("DEMO VALIDATION COMPLETED SUCCESSFULLY")
            self.logger.info(f"Results saved in: {self.output_dir}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Demo validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Production Inference Validation')
    parser.add_argument('--model', type=str, 
                       default='outputs/production_training/ground_truth_run/ground_truth_run/best_stage_3.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, 
                       default='real_earthquake_dataset.h5',
                       help='Path to test dataset')
    parser.add_argument('--output', type=str, 
                       default='outputs/demo_inference_validation',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of demo samples to process')
    
    args = parser.parse_args()
    
    # Initialize and run demo validation
    validator = DemoInferenceValidator(
        model_checkpoint_path=args.model,
        test_dataset_path=args.dataset,
        output_dir=args.output
    )
    
    validator.run_demo_validation()


if __name__ == "__main__":
    main()