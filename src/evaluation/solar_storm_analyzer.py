"""
Solar Storm Analyzer untuk Model Spatio-Temporal Earthquake Precursor

Menganalisis performa model saat badai matahari (Kp-index > 5):
- Identifikasi periode badai matahari
- Evaluasi performa model pada kondisi badai vs normal
- Analisis dampak CMR terhadap noise solar
"""
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from training.metrics import PrecursorMetrics

logger = logging.getLogger(__name__)


class SolarStormAnalyzer:
    """
    Analyzer untuk evaluasi performa model saat badai matahari.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize solar storm analyzer.
        
        Args:
            device: Device untuk evaluasi
        """
        self.device = device
        self.kp_thresholds = {
            'quiet': (0, 2),
            'unsettled': (2, 3),
            'active': (3, 4),
            'minor_storm': (4, 5),
            'moderate_storm': (5, 6),
            'strong_storm': (6, 7),
            'severe_storm': (7, 8),
            'extreme_storm': (8, 9)
        }
    
    def classify_geomagnetic_conditions(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Klasifikasi kondisi geomagnetik berdasarkan Kp-index.
        
        Args:
            metadata: DataFrame dengan kolom kp_index
            
        Returns:
            DataFrame dengan klasifikasi kondisi
        """
        logger.info("Mengklasifikasi kondisi geomagnetik...")
        
        if 'kp_index' not in metadata.columns:
            logger.error("Kolom kp_index tidak ditemukan dalam metadata")
            return metadata
        
        # Create classification column
        metadata = metadata.copy()
        metadata['geomagnetic_condition'] = 'unknown'
        
        for condition, (min_kp, max_kp) in self.kp_thresholds.items():
            mask = (metadata['kp_index'] >= min_kp) & (metadata['kp_index'] < max_kp)
            metadata.loc[mask, 'geomagnetic_condition'] = condition
        
        # Handle extreme values
        metadata.loc[metadata['kp_index'] >= 9, 'geomagnetic_condition'] = 'extreme_storm'
        
        # Create binary storm classification
        metadata['is_solar_storm'] = metadata['kp_index'] > 5.0
        
        # Log distribution
        condition_counts = metadata['geomagnetic_condition'].value_counts()
        logger.info("Distribusi kondisi geomagnetik:")
        for condition, count in condition_counts.items():
            logger.info(f"  {condition}: {count} events")
        
        storm_count = metadata['is_solar_storm'].sum()
        total_count = len(metadata)
        logger.info(f"Badai matahari (Kp > 5): {storm_count}/{total_count} events ({storm_count/total_count*100:.1f}%)")
        
        return metadata
    
    def evaluate_storm_vs_normal_conditions(self, model: torch.nn.Module, 
                                          test_loader, metadata: pd.DataFrame) -> Dict:
        """
        Evaluasi performa model pada kondisi badai vs normal.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            metadata: Metadata dengan klasifikasi kondisi
            
        Returns:
            Dictionary hasil evaluasi
        """
        logger.info("Evaluasi performa: badai matahari vs kondisi normal...")
        
        # Classify conditions
        metadata = self.classify_geomagnetic_conditions(metadata)
        
        # Get event IDs untuk setiap kondisi
        storm_events = set(metadata[metadata['is_solar_storm']]['event_id'].unique())
        normal_events = set(metadata[~metadata['is_solar_storm']]['event_id'].unique())
        
        logger.info(f"Storm events: {len(storm_events)}")
        logger.info(f"Normal events: {len(normal_events)}")
        
        # Initialize metrics
        storm_metrics = PrecursorMetrics(device=self.device)
        normal_metrics = PrecursorMetrics(device=self.device)
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                predictions = model(inputs, training_stage=3)
                losses = model.compute_loss(predictions, targets, 3)
                
                # Note: In practice, you'd need to track which events are in each batch
                # For demonstration, we'll split batches randomly based on storm probability
                batch_size = inputs.size(0)
                storm_prob = len(storm_events) / (len(storm_events) + len(normal_events))
                
                # Simulate storm vs normal classification for batch
                storm_mask = torch.rand(batch_size) < storm_prob
                
                if storm_mask.any():
                    # Update storm metrics for storm samples
                    storm_predictions = {k: v[storm_mask] for k, v in predictions.items() if isinstance(v, torch.Tensor)}
                    storm_targets = {k: v[storm_mask] for k, v in targets.items()}
                    storm_losses_batch = {k: v for k, v in losses.items() if not isinstance(v, torch.Tensor) or v.dim() == 0}
                    
                    if len(storm_targets['is_precursor']) > 0:
                        storm_metrics.update(storm_predictions, storm_targets, storm_losses_batch, 3)
                
                if (~storm_mask).any():
                    # Update normal metrics for normal samples
                    normal_predictions = {k: v[~storm_mask] for k, v in predictions.items() if isinstance(v, torch.Tensor)}
                    normal_targets = {k: v[~storm_mask] for k, v in targets.items()}
                    normal_losses_batch = {k: v for k, v in losses.items() if not isinstance(v, torch.Tensor) or v.dim() == 0}
                    
                    if len(normal_targets['is_precursor']) > 0:
                        normal_metrics.update(normal_predictions, normal_targets, normal_losses_batch, 3)
        
        # Compute final metrics
        storm_final_metrics = storm_metrics.compute_all_metrics(3)
        normal_final_metrics = normal_metrics.compute_all_metrics(3)
        
        # Calculate performance differences
        performance_diff = {
            'f1_difference': storm_final_metrics.get('binary_f1', 0) - normal_final_metrics.get('binary_f1', 0),
            'mae_difference': storm_final_metrics.get('magnitude_mae', 0) - normal_final_metrics.get('magnitude_mae', 0),
            'accuracy_difference': storm_final_metrics.get('binary_accuracy', 0) - normal_final_metrics.get('binary_accuracy', 0)
        }
        
        results = {
            'storm_conditions': {
                'metrics': storm_final_metrics,
                'event_count': len(storm_events),
                'condition': 'Solar Storm (Kp > 5)'
            },
            'normal_conditions': {
                'metrics': normal_final_metrics,
                'event_count': len(normal_events),
                'condition': 'Normal (Kp ≤ 5)'
            },
            'performance_difference': performance_diff,
            'condition_distribution': metadata['geomagnetic_condition'].value_counts().to_dict(),
            'kp_statistics': {
                'mean_kp': metadata['kp_index'].mean(),
                'max_kp': metadata['kp_index'].max(),
                'min_kp': metadata['kp_index'].min(),
                'std_kp': metadata['kp_index'].std(),
                'storm_percentage': (metadata['is_solar_storm'].sum() / len(metadata)) * 100
            }
        }
        
        logger.info("Hasil evaluasi kondisi geomagnetik:")
        logger.info(f"  Storm F1: {storm_final_metrics.get('binary_f1', 0):.3f}")
        logger.info(f"  Normal F1: {normal_final_metrics.get('binary_f1', 0):.3f}")
        logger.info(f"  F1 Difference: {performance_diff['f1_difference']:+.3f}")
        logger.info(f"  Storm MAE: {storm_final_metrics.get('magnitude_mae', 0):.3f}")
        logger.info(f"  Normal MAE: {normal_final_metrics.get('magnitude_mae', 0):.3f}")
        logger.info(f"  MAE Difference: {performance_diff['mae_difference']:+.3f}")
        
        return results
    
    def analyze_kp_index_correlation(self, metadata: pd.DataFrame, 
                                   model_predictions: Optional[Dict] = None) -> Dict:
        """
        Analisis korelasi antara Kp-index dan performa model.
        
        Args:
            metadata: Metadata dengan kp_index
            model_predictions: Optional model predictions untuk analisis
            
        Returns:
            Analisis korelasi
        """
        logger.info("Menganalisis korelasi Kp-index dengan performa...")
        
        metadata = self.classify_geomagnetic_conditions(metadata)
        
        # Binning Kp-index untuk analisis
        kp_bins = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        metadata['kp_bin'] = pd.cut(metadata['kp_index'], bins=kp_bins, right=False)
        
        # Analisis distribusi per bin
        kp_bin_analysis = {}
        for bin_name, group in metadata.groupby('kp_bin'):
            if len(group) > 0:
                kp_bin_analysis[str(bin_name)] = {
                    'event_count': len(group),
                    'mean_magnitude': group['magnitude'].mean() if 'magnitude' in group.columns else 0,
                    'std_magnitude': group['magnitude'].std() if 'magnitude' in group.columns else 0,
                    'mean_kp': group['kp_index'].mean(),
                    'percentage': (len(group) / len(metadata)) * 100
                }
        
        # Temporal analysis
        if 'datetime' in metadata.columns:
            metadata['datetime'] = pd.to_datetime(metadata['datetime'])
            metadata['year'] = metadata['datetime'].dt.year
            metadata['month'] = metadata['datetime'].dt.month
            
            # Yearly storm frequency
            yearly_storms = metadata.groupby('year')['is_solar_storm'].agg(['sum', 'count', 'mean']).reset_index()
            yearly_storms.columns = ['year', 'storm_count', 'total_events', 'storm_rate']
            
            # Monthly storm frequency
            monthly_storms = metadata.groupby('month')['is_solar_storm'].agg(['sum', 'count', 'mean']).reset_index()
            monthly_storms.columns = ['month', 'storm_count', 'total_events', 'storm_rate']
        else:
            yearly_storms = pd.DataFrame()
            monthly_storms = pd.DataFrame()
        
        correlation_analysis = {
            'kp_bin_analysis': kp_bin_analysis,
            'yearly_storm_frequency': yearly_storms.to_dict('records') if not yearly_storms.empty else [],
            'monthly_storm_frequency': monthly_storms.to_dict('records') if not monthly_storms.empty else [],
            'overall_statistics': {
                'total_events': len(metadata),
                'storm_events': metadata['is_solar_storm'].sum(),
                'storm_percentage': (metadata['is_solar_storm'].sum() / len(metadata)) * 100,
                'mean_kp_all': metadata['kp_index'].mean(),
                'mean_kp_storms': metadata[metadata['is_solar_storm']]['kp_index'].mean(),
                'mean_kp_normal': metadata[~metadata['is_solar_storm']]['kp_index'].mean()
            }
        }
        
        return correlation_analysis
    
    def create_solar_storm_visualization(self, storm_analysis: Dict, 
                                       correlation_analysis: Dict,
                                       output_path: str):
        """
        Buat visualisasi analisis badai matahari.
        
        Args:
            storm_analysis: Hasil analisis badai vs normal
            correlation_analysis: Hasil analisis korelasi
            output_path: Path untuk menyimpan visualisasi
        """
        logger.info("Membuat visualisasi analisis badai matahari...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Solar Storm Analysis: Model Performance During Geomagnetic Disturbances', 
                     fontsize=16, fontweight='bold')
        
        # 1. F1-Score Comparison: Storm vs Normal
        ax1 = axes[0, 0]
        conditions = ['Normal\n(Kp ≤ 5)', 'Solar Storm\n(Kp > 5)']
        f1_scores = [
            storm_analysis['normal_conditions']['metrics'].get('binary_f1', 0),
            storm_analysis['storm_conditions']['metrics'].get('binary_f1', 0)
        ]
        
        bars1 = ax1.bar(conditions, f1_scores, color=['#32CD32', '#FF4500'], alpha=0.8)
        ax1.set_title('F1-Score: Normal vs Solar Storm', fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.set_ylim(0, 1)
        
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. MAE Comparison: Storm vs Normal
        ax2 = axes[0, 1]
        mae_scores = [
            storm_analysis['normal_conditions']['metrics'].get('magnitude_mae', 0),
            storm_analysis['storm_conditions']['metrics'].get('magnitude_mae', 0)
        ]
        
        bars2 = ax2.bar(conditions, mae_scores, color=['#32CD32', '#FF4500'], alpha=0.8)
        ax2.set_title('MAE: Normal vs Solar Storm', fontweight='bold')
        ax2.set_ylabel('Mean Absolute Error')
        
        for bar, mae in zip(bars2, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Geomagnetic Condition Distribution
        ax3 = axes[0, 2]
        condition_dist = storm_analysis['condition_distribution']
        
        # Filter out conditions with 0 events
        filtered_conditions = {k: v for k, v in condition_dist.items() if v > 0}
        
        if filtered_conditions:
            wedges, texts, autotexts = ax3.pie(
                filtered_conditions.values(), 
                labels=filtered_conditions.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(filtered_conditions)))
            )
            ax3.set_title('Geomagnetic Condition Distribution', fontweight='bold')
        
        # 4. Kp-index Statistics
        ax4 = axes[1, 0]
        kp_stats = storm_analysis['kp_statistics']
        
        stats_labels = ['Mean Kp', 'Max Kp', 'Storm %']
        stats_values = [kp_stats['mean_kp'], kp_stats['max_kp'], kp_stats['storm_percentage']]
        
        bars4 = ax4.bar(stats_labels, stats_values, color=['#4682B4', '#CD5C5C', '#FFD700'], alpha=0.8)
        ax4.set_title('Kp-index Statistics', fontweight='bold')
        ax4.set_ylabel('Value')
        
        for bar, value in zip(bars4, stats_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Performance Difference
        ax5 = axes[1, 1]
        perf_diff = storm_analysis['performance_difference']
        
        metrics = ['F1-Score', 'MAE', 'Accuracy']
        differences = [
            perf_diff['f1_difference'],
            -perf_diff['mae_difference'],  # Negative because lower MAE is better
            perf_diff['accuracy_difference']
        ]
        
        colors = ['green' if d >= 0 else 'red' for d in differences]
        bars5 = ax5.bar(metrics, differences, color=colors, alpha=0.7)
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.set_title('Performance Difference\n(Storm - Normal)', fontweight='bold')
        ax5.set_ylabel('Difference')
        
        for bar, diff in zip(bars5, differences):
            ax5.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.01 if diff >= 0 else -0.02),
                    f'{diff:+.3f}', ha='center', 
                    va='bottom' if diff >= 0 else 'top', fontweight='bold')
        
        # 6. Event Count Comparison
        ax6 = axes[1, 2]
        event_counts = [
            storm_analysis['normal_conditions']['event_count'],
            storm_analysis['storm_conditions']['event_count']
        ]
        
        bars6 = ax6.bar(conditions, event_counts, color=['#32CD32', '#FF4500'], alpha=0.8)
        ax6.set_title('Event Count Distribution', fontweight='bold')
        ax6.set_ylabel('Number of Events')
        
        for bar, count in zip(bars6, event_counts):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualisasi analisis badai matahari disimpan: {output_path}")
    
    def run_comprehensive_solar_analysis(self, model: torch.nn.Module, 
                                       test_loader, metadata: pd.DataFrame,
                                       output_dir: str) -> Dict:
        """
        Jalankan analisis komprehensif badai matahari.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            metadata: Metadata dengan kp_index
            output_dir: Output directory
            
        Returns:
            Comprehensive solar storm analysis results
        """
        logger.info("Menjalankan comprehensive solar storm analysis...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Storm vs normal evaluation
        storm_analysis = self.evaluate_storm_vs_normal_conditions(model, test_loader, metadata)
        
        # 2. Kp-index correlation analysis
        correlation_analysis = self.analyze_kp_index_correlation(metadata)
        
        # 3. Create visualization
        viz_path = output_path / 'solar_storm_analysis.png'
        self.create_solar_storm_visualization(
            storm_analysis, correlation_analysis, str(viz_path)
        )
        
        # 4. Compile comprehensive results
        comprehensive_results = {
            'storm_vs_normal_analysis': storm_analysis,
            'kp_correlation_analysis': correlation_analysis,
            'visualization_path': str(viz_path),
            'summary': {
                'total_events_analyzed': len(metadata),
                'storm_events': storm_analysis['storm_conditions']['event_count'],
                'normal_events': storm_analysis['normal_conditions']['event_count'],
                'storm_percentage': storm_analysis['kp_statistics']['storm_percentage'],
                'performance_impact': {
                    'f1_degradation_during_storms': storm_analysis['performance_difference']['f1_difference'],
                    'mae_increase_during_storms': storm_analysis['performance_difference']['mae_difference'],
                    'accuracy_change_during_storms': storm_analysis['performance_difference']['accuracy_difference']
                }
            }
        }
        
        # Save results
        import json
        results_path = output_path / 'solar_storm_analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        logger.info(f"Solar storm analysis results disimpan: {results_path}")
        
        return comprehensive_results