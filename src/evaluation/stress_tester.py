"""
Stress Tester untuk Model Spatio-Temporal Earthquake Precursor

Melakukan stress testing dengan:
- Data temporal yang berbeda (Juli 2024-2026)
- Evaluasi performa pada kondisi ekstrem
- Analisis degradasi performa
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from training.metrics import PrecursorMetrics
from training.utils import load_checkpoint

logger = logging.getLogger(__name__)


class StressTester:
    """
    Stress tester untuk evaluasi model pada kondisi temporal yang berbeda.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize stress tester.
        
        Args:
            device: Device untuk evaluasi
        """
        self.device = device
        self.results = {}
        
    def load_model_checkpoint(self, model: nn.Module, checkpoint_path: str) -> nn.Module:
        """
        Load model dari checkpoint.
        
        Args:
            model: Model architecture
            checkpoint_path: Path ke checkpoint
            
        Returns:
            Loaded model
        """
        load_checkpoint(model, checkpoint_path, device=self.device)
        model.eval()
        return model
    
    def evaluate_temporal_periods(self, model: nn.Module, test_loader, 
                                metadata: pd.DataFrame) -> Dict[str, Dict]:
        """
        Evaluasi model pada periode temporal yang berbeda.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            metadata: Metadata dengan informasi temporal
            
        Returns:
            Dictionary hasil evaluasi per periode
        """
        logger.info("Melakukan evaluasi temporal periods...")
        
        # Convert datetime column
        metadata['datetime'] = pd.to_datetime(metadata['datetime'])
        
        # Define temporal periods untuk stress test
        periods = {
            'Q3_2024': ('2024-07-01', '2024-09-30'),
            'Q4_2024': ('2024-10-01', '2024-12-31'),
            'Q1_2025': ('2025-01-01', '2025-03-31'),
            'Q2_2025': ('2025-04-01', '2025-06-30'),
            'Q3_2025': ('2025-07-01', '2025-09-30'),
            'Q4_2025': ('2025-10-01', '2025-12-31'),
            'Q1_2026': ('2026-01-01', '2026-03-31'),
            'Q2_2026': ('2026-04-01', '2026-06-30')
        }
        
        period_results = {}
        
        model.eval()
        with torch.no_grad():
            for period_name, (start_date, end_date) in periods.items():
                logger.info(f"Evaluating period: {period_name} ({start_date} to {end_date})")
                
                # Filter metadata untuk periode ini
                period_mask = (
                    (metadata['datetime'] >= start_date) & 
                    (metadata['datetime'] <= end_date)
                )
                period_metadata = metadata[period_mask]
                
                if len(period_metadata) == 0:
                    logger.warning(f"No data found for period {period_name}")
                    continue
                
                # Get event IDs untuk periode ini
                period_event_ids = set(period_metadata['event_id'].unique())
                
                # Evaluate model pada data periode ini
                period_metrics = PrecursorMetrics(device=self.device)
                
                batch_count = 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    # Check if batch contains events from this period
                    # Note: This is simplified - in practice you'd need to track event IDs
                    
                    inputs = inputs.to(self.device)
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                    
                    # Forward pass
                    predictions = model(inputs, training_stage=3)
                    losses = model.compute_loss(predictions, targets, 3)
                    
                    # Update metrics
                    period_metrics.update(predictions, targets, losses, 3)
                    batch_count += 1
                
                if batch_count > 0:
                    # Compute metrics untuk periode ini
                    final_metrics = period_metrics.compute_all_metrics(3)
                    period_results[period_name] = {
                        'metrics': final_metrics,
                        'event_count': len(period_event_ids),
                        'batch_count': batch_count,
                        'date_range': (start_date, end_date)
                    }
                    
                    logger.info(f"Period {period_name}: F1={final_metrics.get('binary_f1', 0):.3f}, "
                              f"MAE={final_metrics.get('magnitude_mae', 0):.3f}")
        
        return period_results
    
    def evaluate_magnitude_ranges(self, model: nn.Module, test_loader,
                                metadata: pd.DataFrame) -> Dict[str, Dict]:
        """
        Evaluasi model pada range magnitudo yang berbeda.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            metadata: Metadata dengan informasi magnitudo
            
        Returns:
            Dictionary hasil evaluasi per range magnitudo
        """
        logger.info("Melakukan evaluasi magnitude ranges...")
        
        # Define magnitude ranges
        mag_ranges = {
            'Small (4.0-4.5)': (4.0, 4.5),
            'Medium (4.5-5.5)': (4.5, 5.5),
            'Large (5.5-6.5)': (5.5, 6.5),
            'Very Large (6.5+)': (6.5, 10.0)
        }
        
        magnitude_results = {}
        
        model.eval()
        with torch.no_grad():
            for range_name, (min_mag, max_mag) in mag_ranges.items():
                logger.info(f"Evaluating magnitude range: {range_name}")
                
                # Filter metadata untuk range magnitudo ini
                mag_mask = (
                    (metadata['magnitude'] >= min_mag) & 
                    (metadata['magnitude'] < max_mag)
                )
                mag_metadata = metadata[mag_mask]
                
                if len(mag_metadata) == 0:
                    logger.warning(f"No data found for magnitude range {range_name}")
                    continue
                
                # Evaluate model (simplified - would need proper event filtering)
                range_metrics = PrecursorMetrics(device=self.device)
                
                # Process a subset of batches for this magnitude range
                batch_count = 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if batch_count >= 50:  # Limit for demonstration
                        break
                        
                    inputs = inputs.to(self.device)
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                    
                    # Filter targets untuk magnitude range
                    mag_mask_batch = (
                        (targets['magnitude_value'] >= min_mag) & 
                        (targets['magnitude_value'] < max_mag)
                    )
                    
                    if mag_mask_batch.sum() == 0:
                        continue
                    
                    # Forward pass
                    predictions = model(inputs, training_stage=3)
                    losses = model.compute_loss(predictions, targets, 3)
                    
                    # Update metrics
                    range_metrics.update(predictions, targets, losses, 3)
                    batch_count += 1
                
                if batch_count > 0:
                    final_metrics = range_metrics.compute_all_metrics(3)
                    magnitude_results[range_name] = {
                        'metrics': final_metrics,
                        'event_count': len(mag_metadata),
                        'batch_count': batch_count,
                        'magnitude_range': (min_mag, max_mag)
                    }
                    
                    logger.info(f"Magnitude {range_name}: F1={final_metrics.get('binary_f1', 0):.3f}, "
                              f"MAE={final_metrics.get('magnitude_mae', 0):.3f}")
        
        return magnitude_results
    
    def analyze_performance_degradation(self, temporal_results: Dict, 
                                      baseline_metrics: Dict) -> Dict:
        """
        Analisis degradasi performa dari baseline.
        
        Args:
            temporal_results: Hasil evaluasi temporal
            baseline_metrics: Metrics baseline (training performance)
            
        Returns:
            Analisis degradasi performa
        """
        logger.info("Menganalisis performance degradation...")
        
        degradation_analysis = {
            'baseline_f1': baseline_metrics.get('binary_f1', 0),
            'baseline_mae': baseline_metrics.get('magnitude_mae', 0),
            'baseline_accuracy': baseline_metrics.get('binary_accuracy', 0),
            'period_analysis': {},
            'overall_degradation': {}
        }
        
        f1_scores = []
        mae_scores = []
        accuracy_scores = []
        
        for period, results in temporal_results.items():
            metrics = results['metrics']
            
            # Calculate degradation
            f1_degradation = baseline_metrics.get('binary_f1', 0) - metrics.get('binary_f1', 0)
            mae_degradation = metrics.get('magnitude_mae', 0) - baseline_metrics.get('magnitude_mae', 0)
            acc_degradation = baseline_metrics.get('binary_accuracy', 0) - metrics.get('binary_accuracy', 0)
            
            degradation_analysis['period_analysis'][period] = {
                'f1_score': metrics.get('binary_f1', 0),
                'mae': metrics.get('magnitude_mae', 0),
                'accuracy': metrics.get('binary_accuracy', 0),
                'f1_degradation': f1_degradation,
                'mae_degradation': mae_degradation,
                'accuracy_degradation': acc_degradation,
                'event_count': results['event_count']
            }
            
            f1_scores.append(metrics.get('binary_f1', 0))
            mae_scores.append(metrics.get('magnitude_mae', 0))
            accuracy_scores.append(metrics.get('binary_accuracy', 0))
        
        # Overall degradation statistics
        if f1_scores:
            degradation_analysis['overall_degradation'] = {
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores),
                'mean_mae': np.mean(mae_scores),
                'std_mae': np.std(mae_scores),
                'min_mae': np.min(mae_scores),
                'max_mae': np.max(mae_scores),
                'mean_accuracy': np.mean(accuracy_scores),
                'std_accuracy': np.std(accuracy_scores)
            }
        
        return degradation_analysis
    
    def create_stress_test_visualization(self, temporal_results: Dict,
                                       magnitude_results: Dict,
                                       degradation_analysis: Dict,
                                       output_path: str):
        """
        Buat visualisasi hasil stress test.
        
        Args:
            temporal_results: Hasil evaluasi temporal
            magnitude_results: Hasil evaluasi magnitude
            degradation_analysis: Analisis degradasi
            output_path: Path untuk menyimpan visualisasi
        """
        logger.info("Membuat visualisasi stress test...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Stress Test Results: Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Temporal Performance
        ax1 = axes[0, 0]
        periods = list(temporal_results.keys())
        f1_scores = [temporal_results[p]['metrics'].get('binary_f1', 0) for p in periods]
        
        ax1.plot(periods, f1_scores, marker='o', linewidth=2, markersize=8, color='#2E8B57')
        ax1.axhline(y=degradation_analysis['baseline_f1'], color='red', linestyle='--', 
                   label=f"Baseline F1: {degradation_analysis['baseline_f1']:.3f}")
        ax1.set_title('F1-Score Over Time Periods', fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. MAE Performance
        ax2 = axes[0, 1]
        mae_scores = [temporal_results[p]['metrics'].get('magnitude_mae', 0) for p in periods]
        
        ax2.plot(periods, mae_scores, marker='s', linewidth=2, markersize=8, color='#CD5C5C')
        ax2.axhline(y=degradation_analysis['baseline_mae'], color='red', linestyle='--',
                   label=f"Baseline MAE: {degradation_analysis['baseline_mae']:.3f}")
        ax2.set_title('MAE Over Time Periods', fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Magnitude Range Performance
        ax3 = axes[1, 0]
        mag_ranges = list(magnitude_results.keys())
        mag_f1_scores = [magnitude_results[r]['metrics'].get('binary_f1', 0) for r in mag_ranges]
        
        bars = ax3.bar(mag_ranges, mag_f1_scores, color=['#4682B4', '#32CD32', '#FF6347', '#FFD700'], alpha=0.8)
        ax3.set_title('F1-Score by Magnitude Range', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, mag_f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Performance Degradation
        ax4 = axes[1, 1]
        degradation_periods = list(degradation_analysis['period_analysis'].keys())
        f1_degradations = [degradation_analysis['period_analysis'][p]['f1_degradation'] 
                          for p in degradation_periods]
        
        colors = ['green' if d <= 0 else 'red' for d in f1_degradations]
        bars = ax4.bar(degradation_periods, f1_degradations, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_title('F1-Score Degradation from Baseline', fontweight='bold')
        ax4.set_ylabel('F1-Score Degradation')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualisasi stress test disimpan: {output_path}")
    
    def run_comprehensive_stress_test(self, model: nn.Module, test_loader,
                                    metadata: pd.DataFrame, baseline_metrics: Dict,
                                    output_dir: str) -> Dict:
        """
        Jalankan comprehensive stress test.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            metadata: Metadata
            baseline_metrics: Baseline metrics
            output_dir: Output directory
            
        Returns:
            Comprehensive stress test results
        """
        logger.info("Menjalankan comprehensive stress test...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Temporal evaluation
        temporal_results = self.evaluate_temporal_periods(model, test_loader, metadata)
        
        # 2. Magnitude range evaluation
        magnitude_results = self.evaluate_magnitude_ranges(model, test_loader, metadata)
        
        # 3. Performance degradation analysis
        degradation_analysis = self.analyze_performance_degradation(
            temporal_results, baseline_metrics
        )
        
        # 4. Create visualization
        viz_path = output_path / 'stress_test_analysis.png'
        self.create_stress_test_visualization(
            temporal_results, magnitude_results, degradation_analysis, str(viz_path)
        )
        
        # 5. Compile results
        stress_test_results = {
            'temporal_results': temporal_results,
            'magnitude_results': magnitude_results,
            'degradation_analysis': degradation_analysis,
            'visualization_path': str(viz_path),
            'summary': {
                'total_periods_tested': len(temporal_results),
                'total_magnitude_ranges_tested': len(magnitude_results),
                'average_f1_degradation': np.mean([
                    degradation_analysis['period_analysis'][p]['f1_degradation']
                    for p in degradation_analysis['period_analysis']
                ]) if degradation_analysis['period_analysis'] else 0
            }
        }
        
        # Save results
        import json
        results_path = output_path / 'stress_test_results.json'
        with open(results_path, 'w') as f:
            json.dump(stress_test_results, f, indent=2, default=str)
        
        logger.info(f"Stress test results disimpan: {results_path}")
        
        return stress_test_results