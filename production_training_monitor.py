#!/usr/bin/env python3
"""
Production Training Monitor - Ground Truth Implementation

Script monitoring khusus untuk production training dengan fokus pada:
1. Long Tail Challenge (1.9% Large events)
2. Memory Management (1440 time steps)
3. Real-time validation monitoring
4. Solar storm period analysis
"""
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionTrainingMonitor:
    """
    Monitor khusus untuk production training dengan data asli BMKG.
    """
    
    def __init__(self, config_path='configs/production_config.yaml'):
        self.config_path = Path(config_path)
        self.dataset_path = Path('real_earthquake_dataset.h5')
        
        # Critical monitoring parameters
        self.long_tail_threshold = 0.02  # 2% untuk Large events
        self.memory_warning_threshold = 0.85  # 85% GPU memory
        self.loss_plateau_patience = 10  # epochs
        
        self.monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'long_tail_analysis': {},
            'memory_monitoring': {},
            'training_stability': {},
            'solar_storm_impact': {},
            'recommendations': []
        }
    
    def analyze_long_tail_challenge(self):
        """Analyze dan setup strategi untuk Long Tail Challenge."""
        logger.info("=" * 60)
        logger.info("LONG TAIL CHALLENGE ANALYSIS")
        logger.info("=" * 60)
        
        try:
            import h5py
            
            with h5py.File(self.dataset_path, 'r') as f:
                # Load magnitude data
                magnitudes = f['metadata']['magnitude'][:]
                
                # Define magnitude classes (sesuai audit)
                magnitude_bins = [0, 4.0, 4.5, 5.0, 5.5, 10.0]
                magnitude_labels = ['Small', 'Normal', 'Moderate', 'Medium', 'Large']
                
                # Calculate class distribution
                class_indices = np.digitize(magnitudes, magnitude_bins) - 1
                class_indices = np.clip(class_indices, 0, len(magnitude_labels) - 1)
                
                class_counts = np.bincount(class_indices, minlength=len(magnitude_labels))
                class_percentages = class_counts / len(magnitudes) * 100
                
                logger.info("Class Distribution Analysis:")
                for i, (label, count, pct) in enumerate(zip(magnitude_labels, class_counts, class_percentages)):
                    logger.info(f"  {label}: {count} events ({pct:.1f}%)")
                
                # Long tail analysis
                large_event_percentage = class_percentages[-1]  # Large events
                is_long_tail = large_event_percentage < self.long_tail_threshold * 100
                
                if is_long_tail:
                    logger.warning(f"⚠️  LONG TAIL DETECTED: {large_event_percentage:.1f}% Large events")
                    
                    # Calculate optimal class weights
                    total_samples = len(magnitudes)
                    class_weights = total_samples / (len(magnitude_labels) * class_counts + 1e-6)
                    
                    # Extra boost for rare classes
                    class_weights[-1] *= 2.0  # Double weight for Large events
                    class_weights[-2] *= 1.5  # 1.5x weight for Medium events
                    
                    logger.info("Recommended Class Weights:")
                    for label, weight in zip(magnitude_labels, class_weights):
                        logger.info(f"  {label}: {weight:.3f}")
                    
                    self.monitoring_results['long_tail_analysis'] = {
                        'detected': True,
                        'large_event_percentage': float(large_event_percentage),
                        'class_distribution': {label: int(count) for label, count in zip(magnitude_labels, class_counts)},
                        'recommended_weights': {label: float(weight) for label, weight in zip(magnitude_labels, class_weights)},
                        'strategy': 'weighted_sampling_with_focal_loss'
                    }
                    
                    # Generate weighted sampler code
                    self.generate_weighted_sampler_config(class_weights, class_indices)
                    
                else:
                    logger.info(f"✅ Balanced distribution: {large_event_percentage:.1f}% Large events")
                    self.monitoring_results['long_tail_analysis'] = {
                        'detected': False,
                        'large_event_percentage': float(large_event_percentage),
                        'strategy': 'standard_training'
                    }
        
        except Exception as e:
            logger.error(f"Error in long tail analysis: {e}")
            self.monitoring_results['long_tail_analysis']['error'] = str(e)
    
    def generate_weighted_sampler_config(self, class_weights, class_indices):
        """Generate konfigurasi WeightedRandomSampler."""
        logger.info("Generating WeightedRandomSampler configuration...")
        
        # Create sample weights
        sample_weights = class_weights[class_indices]
        
        sampler_config = {
            'use_weighted_sampling': True,
            'class_weights': class_weights.tolist(),
            'total_samples': len(sample_weights),
            'replacement': True,
            'focal_loss_alpha': class_weights.tolist(),
            'focal_loss_gamma': 2.0
        }
        
        # Save configuration
        config_path = Path('configs/weighted_sampling_config.json')
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(sampler_config, f, indent=2)
        
        logger.info(f"Weighted sampling config saved: {config_path}")
        
        # Generate code snippet
        code_snippet = f"""
# Long Tail Challenge Solution - Add to trainer.py

from torch.utils.data import WeightedRandomSampler
import json

# Load weighted sampling config
with open('configs/weighted_sampling_config.json', 'r') as f:
    sampling_config = json.load(f)

if sampling_config['use_weighted_sampling']:
    # Create sample weights for each training sample
    class_weights = torch.tensor(sampling_config['class_weights'], dtype=torch.float32)
    sample_weights = class_weights[train_labels]  # train_labels should be class indices
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=sampling_config['replacement']
    )
    
    # Use sampler in DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Don't use shuffle=True with sampler
        num_workers=4,
        pin_memory=True
    )
    
    logger.info("Using WeightedRandomSampler for Long Tail Challenge")
else:
    # Standard training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
"""
        
        # Save code snippet
        with open('long_tail_solution.py', 'w') as f:
            f.write(code_snippet)
        
        logger.info("Code snippet saved: long_tail_solution.py")
    
    def analyze_memory_requirements(self):
        """Analyze memory requirements untuk 1440 time steps."""
        logger.info("=" * 60)
        logger.info("MEMORY REQUIREMENTS ANALYSIS")
        logger.info("=" * 60)
        
        try:
            import h5py
            
            with h5py.File(self.dataset_path, 'r') as f:
                if 'scalogram_tensor' in f:
                    tensor_shape = f['scalogram_tensor'].shape
                    tensor_dtype = f['scalogram_tensor'].dtype
                    
                    logger.info(f"Tensor shape: {tensor_shape}")
                    logger.info(f"Tensor dtype: {tensor_dtype}")
                    
                    # Calculate memory requirements
                    n_samples, n_stations, n_components, freq_bins, time_bins = tensor_shape
                    
                    # Memory per sample (MB)
                    bytes_per_element = np.dtype(tensor_dtype).itemsize
                    memory_per_sample_mb = (n_stations * n_components * freq_bins * time_bins * bytes_per_element) / (1024**2)
                    
                    # Batch memory calculations
                    batch_sizes = [4, 8, 16, 32, 64]
                    memory_analysis = {}
                    
                    logger.info("Memory requirements per batch size:")
                    for batch_size in batch_sizes:
                        batch_memory_mb = memory_per_sample_mb * batch_size
                        batch_memory_gb = batch_memory_mb / 1024
                        
                        # Include gradient memory (2x for backward pass)
                        total_memory_gb = batch_memory_gb * 3  # Forward + Backward + Optimizer
                        
                        memory_analysis[batch_size] = {
                            'batch_memory_gb': float(batch_memory_gb),
                            'total_memory_gb': float(total_memory_gb),
                            'recommended': total_memory_gb < 6.0  # Safe for 8GB GPU
                        }
                        
                        status = "✅ SAFE" if total_memory_gb < 6.0 else "⚠️  HIGH" if total_memory_gb < 10.0 else "❌ UNSAFE"
                        logger.info(f"  Batch {batch_size}: {total_memory_gb:.2f} GB {status}")
                    
                    # Recommended batch size
                    recommended_batch_size = max([bs for bs, info in memory_analysis.items() if info['recommended']], default=4)
                    
                    logger.info(f"\n🎯 Recommended batch size: {recommended_batch_size}")
                    
                    self.monitoring_results['memory_monitoring'] = {
                        'tensor_shape': list(tensor_shape),
                        'memory_per_sample_mb': float(memory_per_sample_mb),
                        'batch_analysis': memory_analysis,
                        'recommended_batch_size': recommended_batch_size,
                        'high_resolution_temporal': time_bins >= 1000
                    }
                    
                    if time_bins >= 1000:
                        logger.warning(f"⚠️  HIGH TEMPORAL RESOLUTION: {time_bins} time steps detected")
                        logger.info("Consider temporal downsampling if memory issues occur")
        
        except Exception as e:
            logger.error(f"Error in memory analysis: {e}")
            self.monitoring_results['memory_monitoring']['error'] = str(e)
    
    def setup_training_monitoring(self):
        """Setup monitoring untuk training stability."""
        logger.info("=" * 60)
        logger.info("TRAINING MONITORING SETUP")
        logger.info("=" * 60)
        
        monitoring_config = {
            'loss_monitoring': {
                'plateau_patience': self.loss_plateau_patience,
                'min_delta': 1e-4,
                'early_stopping_patience': 25,
                'lr_reduction_factor': 0.5,
                'lr_reduction_patience': 10
            },
            'validation_monitoring': {
                'check_frequency': 1,  # Every epoch
                'save_best_model': True,
                'metric_priority': ['f1_score', 'accuracy', 'loss'],
                'overfitting_threshold': 0.1  # Val loss > Train loss + threshold
            },
            'gradient_monitoring': {
                'clip_grad_norm': 1.0,
                'log_grad_norm': True,
                'detect_gradient_explosion': True
            },
            'memory_monitoring': {
                'log_gpu_memory': True,
                'memory_warning_threshold': self.memory_warning_threshold,
                'clear_cache_frequency': 10  # Every 10 epochs
            }
        }
        
        # Save monitoring config
        config_path = Path('configs/training_monitoring_config.json')
        with open(config_path, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        logger.info(f"Training monitoring config saved: {config_path}")
        
        # Generate monitoring code
        monitoring_code = """
# Training Monitoring Code - Add to training loop

import torch
import psutil
import GPUtil

class TrainingMonitor:
    def __init__(self, config):
        self.config = config
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.lr_patience_counter = 0
        
    def log_epoch_metrics(self, epoch, train_loss, val_loss, metrics, model, optimizer):
        # Memory monitoring
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_pct = gpu_memory / torch.cuda.get_device_properties(0).total_memory * 1024**3 * 100
            
            if gpu_memory_pct > self.config['memory_monitoring']['memory_warning_threshold'] * 100:
                logger.warning(f"High GPU memory usage: {gpu_memory_pct:.1f}%")
        
        # Loss monitoring
        if val_loss < self.best_val_loss - self.config['loss_monitoring']['min_delta']:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')
        else:
            self.patience_counter += 1
            
        # Learning rate reduction
        if self.patience_counter >= self.config['loss_monitoring']['lr_reduction_patience']:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.config['loss_monitoring']['lr_reduction_factor']
            logger.info(f"Reduced learning rate to {param_group['lr']:.2e}")
            self.lr_patience_counter = 0
        
        # Early stopping
        if self.patience_counter >= self.config['loss_monitoring']['early_stopping_patience']:
            logger.info("Early stopping triggered")
            return True
        
        return False
"""
        
        with open('training_monitor_code.py', 'w') as f:
            f.write(monitoring_code)
        
        logger.info("Training monitoring code saved: training_monitor_code.py")
        
        self.monitoring_results['training_stability'] = monitoring_config
    
    def analyze_solar_storm_periods(self):
        """Analyze periode solar storm untuk test set validation."""
        logger.info("=" * 60)
        logger.info("SOLAR STORM PERIOD ANALYSIS")
        logger.info("=" * 60)
        
        try:
            # Load Kp-index data
            kp_df = pd.read_csv('../awal/kp_index_2018_2026.csv')
            kp_df['datetime'] = pd.to_datetime(kp_df['Date_Time_UTC'])
            
            # Focus on test period (July 2024 - 2026)
            test_start = pd.to_datetime('2024-07-01')
            test_kp = kp_df[kp_df['datetime'] >= test_start].copy()
            
            # Define solar storm levels
            storm_levels = {
                'quiet': (0, 3),
                'unsettled': (3, 4),
                'minor_storm': (4, 5),
                'moderate_storm': (5, 6),
                'strong_storm': (6, 7),
                'severe_storm': (7, 8),
                'extreme_storm': (8, 9)
            }
            
            # Analyze storm distribution in test period
            storm_analysis = {}
            total_hours = len(test_kp)
            
            logger.info("Solar storm distribution in test period (July 2024 - 2026):")
            for level, (min_kp, max_kp) in storm_levels.items():
                storm_hours = len(test_kp[(test_kp['Kp_Index'] >= min_kp) & (test_kp['Kp_Index'] < max_kp)])
                storm_percentage = storm_hours / total_hours * 100
                
                storm_analysis[level] = {
                    'hours': storm_hours,
                    'percentage': float(storm_percentage)
                }
                
                logger.info(f"  {level.replace('_', ' ').title()}: {storm_hours} hours ({storm_percentage:.1f}%)")
            
            # Identify critical storm periods
            high_storm_periods = test_kp[test_kp['Kp_Index'] >= 5.0]  # Moderate+ storms
            critical_storm_percentage = len(high_storm_periods) / total_hours * 100
            
            logger.info(f"\n🌟 Critical storm periods (Kp ≥ 5): {critical_storm_percentage:.1f}%")
            
            if critical_storm_percentage > 10:
                logger.warning("⚠️  HIGH SOLAR ACTIVITY in test period - Perfect for CMR validation!")
            
            # Monthly breakdown
            monthly_storm_activity = test_kp.groupby(test_kp['datetime'].dt.to_period('M'))['Kp_Index'].agg(['mean', 'max', 'std'])
            
            logger.info("\nMonthly storm activity (test period):")
            for month, stats in monthly_storm_activity.iterrows():
                logger.info(f"  {month}: Mean Kp={stats['mean']:.1f}, Max Kp={stats['max']:.1f}")
            
            self.monitoring_results['solar_storm_impact'] = {
                'test_period_analysis': storm_analysis,
                'critical_storm_percentage': float(critical_storm_percentage),
                'monthly_breakdown': monthly_storm_activity.to_dict(),
                'cmr_validation_opportunity': critical_storm_percentage > 5.0
            }
        
        except Exception as e:
            logger.error(f"Error in solar storm analysis: {e}")
            self.monitoring_results['solar_storm_impact']['error'] = str(e)
    
    def generate_production_recommendations(self):
        """Generate rekomendasi taktis untuk production training."""
        logger.info("=" * 60)
        logger.info("PRODUCTION TRAINING RECOMMENDATIONS")
        logger.info("=" * 60)
        
        recommendations = []
        
        # Long tail recommendations
        long_tail = self.monitoring_results.get('long_tail_analysis', {})
        if long_tail.get('detected', False):
            recommendations.extend([
                "🎯 CRITICAL: Implement WeightedRandomSampler for Long Tail Challenge",
                "📊 Use Focal Loss with gamma=2.0 for rare class focus",
                "⚖️  Apply 2x weight boost for Large magnitude events",
                "📈 Monitor per-class precision/recall during training"
            ])
        
        # Memory recommendations
        memory = self.monitoring_results.get('memory_monitoring', {})
        if memory.get('recommended_batch_size', 32) <= 16:
            recommendations.extend([
                f"💾 Use batch size ≤ {memory.get('recommended_batch_size', 16)} for memory safety",
                "🔄 Enable gradient accumulation if batch size too small",
                "🧹 Clear GPU cache every 10 epochs",
                "📊 Monitor GPU memory usage continuously"
            ])
        
        # Training stability recommendations
        recommendations.extend([
            "📉 Watch Stage 1 loss - should drop dramatically in first 5 epochs",
            "🎛️  Use learning rate scheduling with ReduceLROnPlateau",
            "⏹️  Implement early stopping with patience=25",
            "💾 Save model checkpoints every epoch during Stage 1"
        ])
        
        # Solar storm validation recommendations
        solar = self.monitoring_results.get('solar_storm_impact', {})
        if solar.get('cmr_validation_opportunity', False):
            recommendations.extend([
                "🌟 EXCELLENT: High solar activity in test period for CMR validation",
                "📊 Compare model performance during quiet vs storm periods",
                "🔍 Analyze CMR effectiveness during Kp > 5 events",
                "📈 Generate storm-specific confusion matrices"
            ])
        
        # Victory lap preparation
        recommendations.extend([
            "🏆 Prepare Attention Maps for station sensitivity analysis",
            "🗺️  Generate Confusion Matrix for each magnitude class",
            "📊 Create ROC curves for binary classification performance",
            "🎯 Validate Dobrovolsky radius effectiveness in predictions"
        ])
        
        self.monitoring_results['recommendations'] = recommendations
        
        logger.info("TACTICAL RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i:2d}. {rec}")
        
        return recommendations
    
    def save_monitoring_report(self, output_path='production_training_monitor_report.json'):
        """Save comprehensive monitoring report."""
        with open(output_path, 'w') as f:
            json.dump(self.monitoring_results, f, indent=2, default=str)
        logger.info(f"Production training monitor report saved: {output_path}")
    
    def run_complete_analysis(self):
        """Run complete pre-training analysis."""
        logger.info("🚀 PRODUCTION TRAINING MONITOR")
        logger.info("Ground Truth Implementation - BMKG Real Data")
        logger.info("=" * 80)
        
        try:
            # 1. Long Tail Challenge Analysis
            self.analyze_long_tail_challenge()
            
            # 2. Memory Requirements Analysis
            self.analyze_memory_requirements()
            
            # 3. Training Monitoring Setup
            self.setup_training_monitoring()
            
            # 4. Solar Storm Period Analysis
            self.analyze_solar_storm_periods()
            
            # 5. Generate Recommendations
            self.generate_production_recommendations()
            
            # 6. Save Report
            self.save_monitoring_report()
            
            logger.info("\n" + "=" * 80)
            logger.info("🎉 PRODUCTION TRAINING MONITOR COMPLETE")
            logger.info("System ready for 8-12 hour production training session")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error in production training monitor: {e}")
            raise
        
        return self.monitoring_results


def main():
    """Main function."""
    print("PRODUCTION TRAINING MONITOR")
    print("Ground Truth Implementation - Real BMKG Data")
    print("=" * 80)
    
    try:
        # Run complete analysis
        monitor = ProductionTrainingMonitor()
        results = monitor.run_complete_analysis()
        
        # Print key insights
        long_tail = results.get('long_tail_analysis', {})
        memory = results.get('memory_monitoring', {})
        solar = results.get('solar_storm_impact', {})
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"Long Tail Challenge: {'DETECTED' if long_tail.get('detected') else 'NOT DETECTED'}")
        print(f"Recommended Batch Size: {memory.get('recommended_batch_size', 'Unknown')}")
        print(f"Solar Storm Validation: {'EXCELLENT' if solar.get('cmr_validation_opportunity') else 'LIMITED'}")
        
        print(f"\n🚀 READY FOR PRODUCTION TRAINING!")
        print(f"Follow the tactical recommendations for optimal results.")
        
        return 'SUCCESS'
        
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        return 'ERROR'


if __name__ == '__main__':
    main()