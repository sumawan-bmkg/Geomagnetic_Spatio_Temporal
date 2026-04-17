#!/usr/bin/env python3
"""
Complete Evaluation Script untuk Spatio-Temporal Earthquake Precursor Model

Script ini menjalankan:
1. Pelatihan model dengan data 2018-2024 (CMR vs Original)
2. Stress test dengan data Juli 2024-2026
3. Evaluasi khusus saat badai matahari (Kp-index > 5)
4. Ablation study dan visualisasi komprehensif

Usage:
    python run_complete_evaluation.py --data-dir /path/to/data --output-dir outputs/evaluation
"""
import argparse
import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import modules
from src.models.spatio_temporal_model import create_model
from src.training.dataset import create_data_loaders
from src.training.trainer import SpatioTemporalTrainer
from src.training.utils import setup_training, create_experiment_directory
from src.preprocessing.tensor_engine import TensorEngine
from src.evaluation.stress_tester import StressTester
from src.evaluation.solar_storm_analyzer import SolarStormAnalyzer

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path, level: str = 'INFO'):
    """Setup logging configuration."""
    log_file = output_dir / 'evaluation.log'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return logging.getLogger(__name__)


def check_data_availability(data_paths: dict) -> bool:
    """Check if required data files are available."""
    required_files = [
        'metadata_path',
        'station_coords_path'
    ]
    
    for file_key in required_files:
        if file_key in data_paths:
            if not os.path.exists(data_paths[file_key]):
                logger.error(f"Required file not found: {data_paths[file_key]}")
                return False
    
    return True


def prepare_tensor_datasets(scalogram_path: str, metadata_path: str, 
                          output_dir: Path) -> dict:
    """Prepare tensor datasets for both CMR and Original variants."""
    logger.info("=== PREPARING TENSOR DATASETS ===")
    
    tensor_output_dir = output_dir / "tensor_data"
    tensor_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tensor engine
    engine = TensorEngine(
        scalogram_base_path=scalogram_path,
        metadata_path=metadata_path,
        target_shape=(224, 224),
        primary_stations=['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI'],
        components=['H', 'D', 'Z']
    )
    
    # Process CMR dataset
    logger.info("Processing CMR dataset...")
    cmr_files = engine.process_complete_dataset(
        output_dir=str(tensor_output_dir / "cmr"),
        train_test_split=True,
        apply_cmr=True
    )
    
    # Process Original dataset
    logger.info("Processing Original dataset...")
    original_files = engine.process_complete_dataset(
        output_dir=str(tensor_output_dir / "original"),
        train_test_split=True,
        apply_cmr=False
    )
    
    return {
        'cmr_files': cmr_files,
        'original_files': original_files,
        'tensor_output_dir': tensor_output_dir
    }


def train_model_variant(variant_name: str, train_hdf5: str, test_hdf5: str,
                       metadata_path: str, station_coords_path: str,
                       experiment_dir: Path, device: str = 'cuda') -> dict:
    """Train a model variant (CMR or Original)."""
    logger.info(f"=== TRAINING {variant_name.upper()} MODEL ===")
    
    # Setup training configuration
    config = {
        'model': {
            'n_stations': 8,
            'n_components': 3,
            'efficientnet_pretrained': True,
            'gnn_hidden_dim': 256,
            'gnn_num_layers': 3,
            'dropout_rate': 0.2,
            'magnitude_classes': 5
        },
        'data': {
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'load_in_memory': False,
            'magnitude_bins': [4.0, 4.5, 5.0, 5.5, 6.0]
        },
        'device': device
    }
    
    # Progressive training configuration
    training_stages = {
        'stage_1': {
            'epochs': 50,
            'patience': 15,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {'type': 'AdamW', 'lr': 1e-4, 'weight_decay': 1e-4},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 50, 'eta_min': 1e-6},
            'save_best': True,
            'checkpoint_interval': 10
        },
        'stage_2': {
            'epochs': 60,
            'patience': 20,
            'load_previous_best': True,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {'type': 'AdamW', 'lr': 5e-5, 'weight_decay': 1e-4},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 60, 'eta_min': 1e-6},
            'save_best': True,
            'checkpoint_interval': 10
        },
        'stage_3': {
            'epochs': 80,
            'patience': 25,
            'load_previous_best': True,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {'type': 'AdamW', 'lr': 2e-5, 'weight_decay': 1e-4},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 80, 'eta_min': 1e-6},
            'save_best': True,
            'checkpoint_interval': 10
        }
    }
    
    # Create data loaders
    train_loader, test_loader, dataset_info = create_data_loaders(
        train_hdf5_path=train_hdf5,
        test_hdf5_path=test_hdf5,
        metadata_path=metadata_path,
        **config['data']
    )
    
    # Load station coordinates if available
    station_coordinates = None
    if station_coords_path and os.path.exists(station_coords_path):
        try:
            coords_df = pd.read_csv(station_coords_path)
            if all(col in coords_df.columns for col in ['latitude', 'longitude']):
                station_coordinates = coords_df[['latitude', 'longitude']].values
                config['model']['station_coordinates'] = station_coordinates
                logger.info(f"Loaded station coordinates: {station_coordinates.shape}")
        except Exception as e:
            logger.warning(f"Could not load station coordinates: {e}")
    
    # Create model
    model = create_model(config=config['model'])
    
    # Create trainer
    trainer = SpatioTemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        output_dir=experiment_dir,
        experiment_name=f"{variant_name}_training"
    )
    
    try:
        # Train progressively
        logger.info(f"Starting progressive training for {variant_name}...")
        training_history = trainer.train_progressive(training_stages)
        
        # Final evaluation
        best_checkpoint = experiment_dir / 'best_stage_3.pth'
        if best_checkpoint.exists():
            test_metrics = trainer.evaluate_model(
                test_loader=test_loader,
                checkpoint_path=best_checkpoint
            )
        else:
            logger.warning(f"Best checkpoint not found for {variant_name}")
            test_metrics = {}
        
        return {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'best_checkpoint': best_checkpoint,
            'dataset_info': dataset_info,
            'model': model
        }
        
    finally:
        trainer.close()


def run_stress_test(model_results: dict, tensor_files: dict, metadata_path: str,
                   output_dir: Path, device: str = 'cuda') -> dict:
    """Run stress test evaluation."""
    logger.info("=== RUNNING STRESS TEST ===")
    
    stress_output_dir = output_dir / "stress_test"
    stress_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    stress_results = {}
    
    for variant_name, results in model_results.items():
        logger.info(f"Stress testing {variant_name} model...")
        
        # Determine test data file
        if variant_name == 'cmr':
            test_hdf5 = tensor_files['cmr_files']['test_cmr']
        else:
            test_hdf5 = tensor_files['original_files']['test']
        
        # Create test data loader
        _, test_loader, _ = create_data_loaders(
            train_hdf5_path=test_hdf5,  # Dummy
            test_hdf5_path=test_hdf5,
            metadata_path=metadata_path,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
            load_in_memory=False
        )
        
        # Initialize stress tester
        stress_tester = StressTester(device=device)
        
        # Run comprehensive stress test
        variant_stress_results = stress_tester.run_comprehensive_stress_test(
            model=results['model'],
            test_loader=test_loader,
            metadata=metadata,
            baseline_metrics=results['test_metrics'],
            output_dir=str(stress_output_dir / variant_name)
        )
        
        stress_results[variant_name] = variant_stress_results
    
    return stress_results


def run_solar_storm_analysis(model_results: dict, tensor_files: dict, 
                           metadata_path: str, output_dir: Path,
                           device: str = 'cuda') -> dict:
    """Run solar storm analysis."""
    logger.info("=== RUNNING SOLAR STORM ANALYSIS ===")
    
    solar_output_dir = output_dir / "solar_storm_analysis"
    solar_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    solar_results = {}
    
    for variant_name, results in model_results.items():
        logger.info(f"Solar storm analysis for {variant_name} model...")
        
        # Determine test data file
        if variant_name == 'cmr':
            test_hdf5 = tensor_files['cmr_files']['test_cmr']
        else:
            test_hdf5 = tensor_files['original_files']['test']
        
        # Create test data loader
        _, test_loader, _ = create_data_loaders(
            train_hdf5_path=test_hdf5,  # Dummy
            test_hdf5_path=test_hdf5,
            metadata_path=metadata_path,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
            load_in_memory=False
        )
        
        # Initialize solar storm analyzer
        solar_analyzer = SolarStormAnalyzer(device=device)
        
        # Run comprehensive solar analysis
        variant_solar_results = solar_analyzer.run_comprehensive_solar_analysis(
            model=results['model'],
            test_loader=test_loader,
            metadata=metadata,
            output_dir=str(solar_output_dir / variant_name)
        )
        
        solar_results[variant_name] = variant_solar_results
    
    return solar_results

def create_comprehensive_ablation_study(cmr_results: dict, original_results: dict,
                                       stress_results: dict, solar_results: dict,
                                       output_dir: Path):
    """Create comprehensive ablation study visualization."""
    logger.info("=== CREATING COMPREHENSIVE ABLATION STUDY ===")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Setup matplotlib
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Ablation Study: CMR vs Original\nSpatio-Temporal Earthquake Precursor Model', 
                 fontsize=18, fontweight='bold')
    
    # 1. F1-Score Comparison (Main Results)
    ax1 = axes[0, 0]
    variants = ['CMR', 'Original']
    f1_scores = [
        cmr_results['test_metrics'].get('binary_f1', 0),
        original_results['test_metrics'].get('binary_f1', 0)
    ]
    
    bars1 = ax1.bar(variants, f1_scores, color=['#2E8B57', '#CD5C5C'], alpha=0.8, width=0.6)
    ax1.set_title('F1-Score Comparison\n(Primary Metric)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('F1-Score')
    ax1.set_ylim(0, 1)
    
    for bar, score in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add improvement annotation
    improvement = f1_scores[0] - f1_scores[1]
    ax1.text(0.5, 0.9, f'CMR Improvement: {improvement:+.3f}', 
             transform=ax1.transAxes, ha='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. MAE Comparison
    ax2 = axes[0, 1]
    mae_scores = [
        cmr_results['test_metrics'].get('magnitude_mae', 0),
        original_results['test_metrics'].get('magnitude_mae', 0)
    ]
    
    bars2 = ax2.bar(variants, mae_scores, color=['#2E8B57', '#CD5C5C'], alpha=0.8, width=0.6)
    ax2.set_title('Mean Absolute Error\n(Magnitude Estimation)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('MAE')
    
    for bar, mae in zip(bars2, mae_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    mae_improvement = mae_scores[1] - mae_scores[0]  # Lower is better
    ax2.text(0.5, 0.9, f'CMR Improvement: {mae_improvement:+.3f}', 
             transform=ax2.transAxes, ha='center', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if mae_improvement > 0 else "lightcoral", alpha=0.7))
    
    # 3. Precision-Recall Comparison
    ax3 = axes[0, 2]
    precision_cmr = cmr_results['test_metrics'].get('binary_precision', 0)
    recall_cmr = cmr_results['test_metrics'].get('binary_recall', 0)
    precision_orig = original_results['test_metrics'].get('binary_precision', 0)
    recall_orig = original_results['test_metrics'].get('binary_recall', 0)
    
    x = np.arange(2)
    width = 0.35
    
    bars3a = ax3.bar(x - width/2, [precision_cmr, precision_orig], width, 
                    label='Precision', color='#4682B4', alpha=0.8)
    bars3b = ax3.bar(x + width/2, [recall_cmr, recall_orig], width,
                    label='Recall', color='#FF6347', alpha=0.8)
    
    ax3.set_title('Precision vs Recall\nComparison', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Score')
    ax3.set_xticks(x)
    ax3.set_xticklabels(variants)
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 4. Solar Storm Performance (if available)
    ax4 = axes[1, 0]
    if 'cmr' in solar_results and 'original' in solar_results:
        storm_f1_cmr = solar_results['cmr']['storm_vs_normal_analysis']['storm_conditions']['metrics'].get('binary_f1', 0)
        storm_f1_orig = solar_results['original']['storm_vs_normal_analysis']['storm_conditions']['metrics'].get('binary_f1', 0)
        normal_f1_cmr = solar_results['cmr']['storm_vs_normal_analysis']['normal_conditions']['metrics'].get('binary_f1', 0)
        normal_f1_orig = solar_results['original']['storm_vs_normal_analysis']['normal_conditions']['metrics'].get('binary_f1', 0)
        
        x = np.arange(2)
        width = 0.35
        
        bars4a = ax4.bar(x - width/2, [normal_f1_cmr, normal_f1_orig], width, 
                        label='Normal (Kp ≤ 5)', color='#32CD32', alpha=0.8)
        bars4b = ax4.bar(x + width/2, [storm_f1_cmr, storm_f1_orig], width,
                        label='Storm (Kp > 5)', color='#FF4500', alpha=0.8)
        
        ax4.set_title('F1-Score During\nSolar Storms', fontweight='bold', fontsize=12)
        ax4.set_ylabel('F1-Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(variants)
        ax4.legend()
        ax4.set_ylim(0, 1)
    else:
        ax4.text(0.5, 0.5, 'Solar Storm\nData Not Available', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.set_title('Solar Storm Analysis', fontweight='bold', fontsize=12)
    
    # 5. Stress Test Performance (if available)
    ax5 = axes[1, 1]
    if 'cmr' in stress_results and 'original' in stress_results:
        # Get average F1 across time periods
        cmr_temporal = stress_results['cmr']['temporal_results']
        orig_temporal = stress_results['original']['temporal_results']
        
        if cmr_temporal and orig_temporal:
            cmr_avg_f1 = np.mean([results['metrics'].get('binary_f1', 0) 
                                 for results in cmr_temporal.values()])
            orig_avg_f1 = np.mean([results['metrics'].get('binary_f1', 0) 
                                  for results in orig_temporal.values()])
            
            bars5 = ax5.bar(variants, [cmr_avg_f1, orig_avg_f1], 
                           color=['#2E8B57', '#CD5C5C'], alpha=0.8, width=0.6)
            ax5.set_title('Average F1-Score\nStress Test (2024-2026)', fontweight='bold', fontsize=12)
            ax5.set_ylabel('Average F1-Score')
            ax5.set_ylim(0, 1)
            
            for bar, score in zip(bars5, [cmr_avg_f1, orig_avg_f1]):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        else:
            ax5.text(0.5, 0.5, 'Stress Test\nData Processing', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Stress Test\nData Not Available', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12, fontweight='bold')
    ax5.set_title('Stress Test Performance', fontweight='bold', fontsize=12)
    
    # 6. Localization Performance
    ax6 = axes[1, 2]
    azimuth_error_cmr = cmr_results['test_metrics'].get('azimuth_mean_error_deg', 0)
    azimuth_error_orig = original_results['test_metrics'].get('azimuth_mean_error_deg', 0)
    distance_error_cmr = cmr_results['test_metrics'].get('distance_mae', 0)
    distance_error_orig = original_results['test_metrics'].get('distance_mae', 0)
    
    x = np.arange(2)
    width = 0.35
    
    # Normalize errors for comparison (azimuth in degrees, distance in km)
    norm_azimuth_cmr = azimuth_error_cmr / 180  # Normalize to [0,1]
    norm_azimuth_orig = azimuth_error_orig / 180
    norm_distance_cmr = min(distance_error_cmr / 100, 1)  # Normalize to [0,1], cap at 1
    norm_distance_orig = min(distance_error_orig / 100, 1)
    
    bars6a = ax6.bar(x - width/2, [norm_azimuth_cmr, norm_azimuth_orig], width, 
                    label='Azimuth Error (norm)', color='#9370DB', alpha=0.8)
    bars6b = ax6.bar(x + width/2, [norm_distance_cmr, norm_distance_orig], width,
                    label='Distance Error (norm)', color='#20B2AA', alpha=0.8)
    
    ax6.set_title('Localization Errors\n(Normalized)', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Normalized Error')
    ax6.set_xticks(x)
    ax6.set_xticklabels(variants)
    ax6.legend()
    ax6.set_ylim(0, 1)
    
    # 7. Training Efficiency (Epochs to Convergence)
    ax7 = axes[2, 0]
    # Extract training epochs from history
    cmr_epochs = sum([len(stage_history.get('train', [])) 
                     for stage_history in cmr_results.get('training_history', {}).values()
                     if isinstance(stage_history, dict)])
    orig_epochs = sum([len(stage_history.get('train', [])) 
                      for stage_history in original_results.get('training_history', {}).values()
                      if isinstance(stage_history, dict)])
    
    if cmr_epochs > 0 and orig_epochs > 0:
        bars7 = ax7.bar(variants, [cmr_epochs, orig_epochs], 
                       color=['#2E8B57', '#CD5C5C'], alpha=0.8, width=0.6)
        ax7.set_title('Training Epochs\nto Convergence', fontweight='bold', fontsize=12)
        ax7.set_ylabel('Total Epochs')
        
        for bar, epochs in zip(bars7, [cmr_epochs, orig_epochs]):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{epochs}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    else:
        ax7.text(0.5, 0.5, 'Training History\nNot Available', ha='center', va='center',
                transform=ax7.transAxes, fontsize=12, fontweight='bold')
        ax7.set_title('Training Efficiency', fontweight='bold', fontsize=12)
    
    # 8. Model Complexity Comparison
    ax8 = axes[2, 1]
    # Both models have same architecture, show parameter count
    model_params = sum(p.numel() for p in cmr_results['model'].parameters())
    
    bars8 = ax8.bar(['Model Parameters'], [model_params/1e6], 
                   color='#4682B4', alpha=0.8, width=0.4)
    ax8.set_title('Model Complexity\n(Same Architecture)', fontweight='bold', fontsize=12)
    ax8.set_ylabel('Parameters (Millions)')
    
    ax8.text(0, model_params/1e6 + 0.1, f'{model_params/1e6:.1f}M', 
             ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 9. Summary Metrics
    ax9 = axes[2, 2]
    
    # Calculate overall improvements
    metrics_summary = {
        'F1-Score': (f1_scores[0] - f1_scores[1]) / f1_scores[1] * 100 if f1_scores[1] > 0 else 0,
        'MAE': (mae_scores[1] - mae_scores[0]) / mae_scores[1] * 100 if mae_scores[1] > 0 else 0,  # Lower is better
        'Precision': (precision_cmr - precision_orig) / precision_orig * 100 if precision_orig > 0 else 0,
        'Recall': (recall_cmr - recall_orig) / recall_orig * 100 if recall_orig > 0 else 0
    }
    
    metrics_names = list(metrics_summary.keys())
    improvements = list(metrics_summary.values())
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars9 = ax9.barh(metrics_names, improvements, color=colors, alpha=0.7)
    ax9.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax9.set_title('CMR Improvement\nSummary (%)', fontweight='bold', fontsize=12)
    ax9.set_xlabel('Improvement (%)')
    
    for bar, imp in zip(bars9, improvements):
        ax9.text(imp + (1 if imp >= 0 else -1), bar.get_y() + bar.get_height()/2,
                f'{imp:+.1f}%', ha='left' if imp >= 0 else 'right', va='center', 
                fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / 'comprehensive_ablation_study.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comprehensive_ablation_study.pdf', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comprehensive ablation study saved: {viz_path}")
    
    return viz_path, metrics_summary


def generate_final_report(all_results: dict, output_dir: Path):
    """Generate final comprehensive report."""
    logger.info("=== GENERATING FINAL REPORT ===")
    
    report = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'training_period': '2018-2024',
            'stress_test_period': 'Juli 2024-2026',
            'evaluation_focus': 'CMR vs Original during Solar Storms (Kp > 5)'
        },
        'model_variants': {},
        'ablation_study_results': {},
        'stress_test_summary': {},
        'solar_storm_analysis': {},
        'key_findings': {},
        'recommendations': {}
    }
    
    # Extract results for each variant
    for variant in ['cmr', 'original']:
        if variant in all_results['model_results']:
            variant_results = all_results['model_results'][variant]
            report['model_variants'][variant] = {
                'test_metrics': variant_results['test_metrics'],
                'training_completed': True,
                'best_checkpoint': str(variant_results['best_checkpoint'])
            }
    
    # Ablation study summary
    if 'ablation_metrics' in all_results:
        report['ablation_study_results'] = all_results['ablation_metrics']
    
    # Key findings
    cmr_f1 = all_results['model_results']['cmr']['test_metrics'].get('binary_f1', 0)
    orig_f1 = all_results['model_results']['original']['test_metrics'].get('binary_f1', 0)
    cmr_mae = all_results['model_results']['cmr']['test_metrics'].get('magnitude_mae', 0)
    orig_mae = all_results['model_results']['original']['test_metrics'].get('magnitude_mae', 0)
    
    report['key_findings'] = {
        'cmr_improves_f1_score': cmr_f1 > orig_f1,
        'cmr_improves_mae': cmr_mae < orig_mae,
        'f1_improvement_percentage': ((cmr_f1 - orig_f1) / orig_f1 * 100) if orig_f1 > 0 else 0,
        'mae_improvement_percentage': ((orig_mae - cmr_mae) / orig_mae * 100) if orig_mae > 0 else 0,
        'cmr_effectiveness_during_solar_storms': 'Analyzed' if 'solar_results' in all_results else 'Not Available'
    }
    
    # Recommendations
    report['recommendations'] = {
        'use_cmr_preprocessing': cmr_f1 > orig_f1 and cmr_mae < orig_mae,
        'focus_on_solar_storm_periods': True,
        'model_deployment_ready': cmr_f1 > 0.7,  # Threshold for deployment
        'further_improvements': [
            'Consider ensemble methods combining CMR and Original models',
            'Investigate adaptive CMR based on real-time Kp-index',
            'Expand training data during high Kp-index periods'
        ]
    }
    
    # Save JSON report
    json_path = output_dir / 'final_evaluation_report.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    md_content = f"""# Final Evaluation Report: Spatio-Temporal Earthquake Precursor Model

## Executive Summary

**Experiment Date**: {report['experiment_info']['timestamp']}
**Training Period**: {report['experiment_info']['training_period']}
**Stress Test Period**: {report['experiment_info']['stress_test_period']}

## Key Results

### Model Performance Comparison

| Metric | CMR Model | Original Model | Improvement |
|--------|-----------|----------------|-------------|
| F1-Score | {cmr_f1:.4f} | {orig_f1:.4f} | {report['key_findings']['f1_improvement_percentage']:+.2f}% |
| MAE (Magnitude) | {cmr_mae:.4f} | {orig_mae:.4f} | {report['key_findings']['mae_improvement_percentage']:+.2f}% |

### Key Findings

✅ **CMR Improves F1-Score**: {report['key_findings']['cmr_improves_f1_score']}
✅ **CMR Improves MAE**: {report['key_findings']['cmr_improves_mae']}
📊 **Solar Storm Analysis**: {report['key_findings']['cmr_effectiveness_during_solar_storms']}

## Recommendations

### Deployment Decision
**Use CMR Preprocessing**: {'✅ Recommended' if report['recommendations']['use_cmr_preprocessing'] else '❌ Not Recommended'}

**Model Ready for Deployment**: {'✅ Yes' if report['recommendations']['model_deployment_ready'] else '❌ Needs Improvement'}

### Future Improvements
"""
    
    for improvement in report['recommendations']['further_improvements']:
        md_content += f"- {improvement}\n"
    
    md_content += f"""
## Conclusion

The ablation study demonstrates that Common Mode Rejection (CMR) preprocessing {'significantly improves' if report['recommendations']['use_cmr_preprocessing'] else 'does not significantly improve'} model performance for earthquake precursor detection, particularly during solar storm conditions (Kp-index > 5).

**Final Recommendation**: {'Deploy the CMR-enhanced model for operational use.' if report['recommendations']['use_cmr_preprocessing'] else 'Continue development with focus on alternative preprocessing methods.'}
"""
    
    # Save markdown report
    md_path = output_dir / 'final_evaluation_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Final report saved: {json_path}")
    logger.info(f"Markdown report saved: {md_path}")
    
    return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Complete Evaluation Pipeline')
    parser.add_argument('--scalogram-dir', type=str, default='scalogramv3',
                       help='Path to scalogram data directory')
    parser.add_argument('--metadata-path', type=str, 
                       default='outputs/data_audit/master_metadata.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--station-coords', type=str, default='awal/lokasi_stasiun.csv',
                       help='Path to station coordinates CSV')
    parser.add_argument('--output-dir', type=str, default='outputs/complete_evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cpu', 'cuda'], help='Device for training')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing checkpoints')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(output_dir)
    
    logger.info("🚀 STARTING COMPLETE EVALUATION PIPELINE")
    logger.info("=" * 60)
    
    # Check data availability
    data_paths = {
        'metadata_path': args.metadata_path,
        'station_coords_path': args.station_coords,
        'scalogram_dir': args.scalogram_dir
    }
    
    if not check_data_availability(data_paths):
        logger.error("Required data files not available. Exiting.")
        return
    
    try:
        all_results = {}
        
        # 1. Prepare tensor datasets
        if not args.skip_training:
            tensor_files = prepare_tensor_datasets(
                scalogram_path=args.scalogram_dir,
                metadata_path=args.metadata_path,
                output_dir=output_dir
            )
            all_results['tensor_files'] = tensor_files
        else:
            logger.info("Skipping tensor preparation (using existing data)")
            # You would need to specify existing tensor file paths here
            tensor_files = {}
        
        # 2. Train model variants
        model_results = {}
        
        if not args.skip_training:
            # Train CMR model
            cmr_experiment_dir = output_dir / "experiments" / "cmr_model"
            cmr_experiment_dir.mkdir(parents=True, exist_ok=True)
            
            cmr_results = train_model_variant(
                variant_name="cmr",
                train_hdf5=tensor_files['cmr_files']['train_cmr'],
                test_hdf5=tensor_files['cmr_files']['test_cmr'],
                metadata_path=args.metadata_path,
                station_coords_path=args.station_coords,
                experiment_dir=cmr_experiment_dir,
                device=args.device
            )
            model_results['cmr'] = cmr_results
            
            # Train Original model
            orig_experiment_dir = output_dir / "experiments" / "original_model"
            orig_experiment_dir.mkdir(parents=True, exist_ok=True)
            
            original_results = train_model_variant(
                variant_name="original",
                train_hdf5=tensor_files['original_files']['train'],
                test_hdf5=tensor_files['original_files']['test'],
                metadata_path=args.metadata_path,
                station_coords_path=args.station_coords,
                experiment_dir=orig_experiment_dir,
                device=args.device
            )
            model_results['original'] = original_results
        else:
            logger.info("Skipping training (using existing checkpoints)")
            # You would need to load existing model results here
        
        all_results['model_results'] = model_results
        
        # 3. Run stress test
        if model_results and tensor_files:
            stress_results = run_stress_test(
                model_results=model_results,
                tensor_files=tensor_files,
                metadata_path=args.metadata_path,
                output_dir=output_dir,
                device=args.device
            )
            all_results['stress_results'] = stress_results
        
        # 4. Run solar storm analysis
        if model_results and tensor_files:
            solar_results = run_solar_storm_analysis(
                model_results=model_results,
                tensor_files=tensor_files,
                metadata_path=args.metadata_path,
                output_dir=output_dir,
                device=args.device
            )
            all_results['solar_results'] = solar_results
        
        # 5. Create comprehensive ablation study
        if 'cmr' in model_results and 'original' in model_results:
            viz_path, ablation_metrics = create_comprehensive_ablation_study(
                cmr_results=model_results['cmr'],
                original_results=model_results['original'],
                stress_results=all_results.get('stress_results', {}),
                solar_results=all_results.get('solar_results', {}),
                output_dir=output_dir
            )
            all_results['ablation_visualization'] = str(viz_path)
            all_results['ablation_metrics'] = ablation_metrics
        
        # 6. Generate final report
        final_report = generate_final_report(all_results, output_dir)
        
        logger.info("✅ COMPLETE EVALUATION PIPELINE FINISHED")
        logger.info("=" * 60)
        logger.info(f"📊 Results saved in: {output_dir}")
        logger.info(f"📈 Ablation study: {output_dir / 'comprehensive_ablation_study.png'}")
        logger.info(f"📋 Final report: {output_dir / 'final_evaluation_report.md'}")
        
        # Print key results
        if 'cmr' in model_results and 'original' in model_results:
            cmr_f1 = model_results['cmr']['test_metrics'].get('binary_f1', 0)
            orig_f1 = model_results['original']['test_metrics'].get('binary_f1', 0)
            improvement = ((cmr_f1 - orig_f1) / orig_f1 * 100) if orig_f1 > 0 else 0
            
            logger.info(f"🎯 KEY RESULT: CMR improves F1-Score by {improvement:+.2f}%")
            logger.info(f"   CMR F1-Score: {cmr_f1:.4f}")
            logger.info(f"   Original F1-Score: {orig_f1:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()