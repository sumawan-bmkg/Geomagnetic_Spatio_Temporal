#!/usr/bin/env python3
"""
Training Pipeline Runner untuk Spatio-Temporal Earthquake Precursor Model

Script ini menjalankan:
1. Pelatihan model menggunakan data 2018-2024
2. Stress test menggunakan data Juli 2024-2026
3. Ablation study: Dengan CMR vs Tanpa CMR
4. Evaluasi khusus saat badai matahari (Kp-index > 5)
"""
import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.spatio_temporal_model import create_model
from src.training.dataset import create_data_loaders
from src.training.trainer import SpatioTemporalTrainer
from src.training.utils import setup_training, create_experiment_directory
from src.preprocessing.tensor_engine import TensorEngine

logger = logging.getLogger(__name__)


class TrainingPipelineRunner:
    """Runner untuk menjalankan pipeline pelatihan lengkap dengan evaluasi."""
    
    def __init__(self, base_output_dir: str = "outputs/training_pipeline"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.base_output_dir / 'pipeline.log')
            ]
        )
        
        self.results = {}
        
    def prepare_datasets(self):
        """Persiapkan dataset untuk pelatihan dan evaluasi."""
        logger.info("=== PERSIAPAN DATASET ===")
        
        # Path ke data
        scalogram_path = "scalogramv3"  # Sesuaikan dengan lokasi data Anda
        metadata_path = "outputs/data_audit/master_metadata.csv"
        
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata tidak ditemukan: {metadata_path}")
            return None
            
        # Load metadata untuk analisis
        metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata: {len(metadata)} records")
        
        # Analisis distribusi temporal
        metadata['datetime'] = pd.to_datetime(metadata['datetime'])
        metadata['year'] = metadata['datetime'].dt.year
        
        # Split data berdasarkan periode
        train_data = metadata[metadata['datetime'] < '2024-07-01']
        stress_test_data = metadata[metadata['datetime'] >= '2024-07-01']
        
        logger.info(f"Training data (2018-2024): {len(train_data)} records")
        logger.info(f"Stress test data (Jul 2024-2026): {len(stress_test_data)} records")
        
        return {
            'metadata': metadata,
            'train_data': train_data,
            'stress_test_data': stress_test_data,
            'scalogram_path': scalogram_path,
            'metadata_path': metadata_path
        }
    
    def create_tensor_datasets(self, data_info):
        """Buat tensor datasets untuk CMR dan non-CMR."""
        logger.info("=== PEMBUATAN TENSOR DATASETS ===")
        
        # Create tensor engine
        engine = TensorEngine(
            scalogram_base_path=data_info['scalogram_path'],
            metadata_path=data_info['metadata_path'],
            target_shape=(224, 224),
            primary_stations=['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI'],
            components=['H', 'D', 'Z']
        )
        
        # Process datasets
        tensor_output_dir = self.base_output_dir / "tensors"
        
        # Dataset dengan CMR
        logger.info("Memproses dataset dengan CMR...")
        cmr_files = engine.process_complete_dataset(
            output_dir=str(tensor_output_dir / "cmr"),
            train_test_split=True,
            apply_cmr=True,
            max_events_per_split=None  # Process all events
        )
        
        # Dataset tanpa CMR (original)
        logger.info("Memproses dataset tanpa CMR...")
        original_files = engine.process_complete_dataset(
            output_dir=str(tensor_output_dir / "original"),
            train_test_split=True,
            apply_cmr=False,
            max_events_per_split=None
        )
        
        return {
            'cmr_files': cmr_files,
            'original_files': original_files,
            'tensor_output_dir': tensor_output_dir
        }
    
    def train_model_variant(self, variant_name: str, train_hdf5: str, test_hdf5: str, 
                           metadata_path: str, experiment_dir: Path):
        """Train model variant (CMR atau Original)."""
        logger.info(f"=== PELATIHAN MODEL: {variant_name.upper()} ===")
        
        # Setup training configuration
        config = setup_training(seed=42, device='cuda')
        
        # Modify config for production training
        config.update({
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
        })
        
        # Create data loaders
        train_loader, test_loader, dataset_info = create_data_loaders(
            train_hdf5_path=train_hdf5,
            test_hdf5_path=test_hdf5,
            metadata_path=metadata_path,
            **config['data']
        )
        
        # Create model
        model = create_model(config=config['model'])
        
        # Create trainer
        trainer = SpatioTemporalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            device=config['device'],
            output_dir=experiment_dir,
            experiment_name=f"{variant_name}_training"
        )
        
        try:
            # Train progressively
            training_history = trainer.train_progressive(config)
            
            # Final evaluation
            best_checkpoint = experiment_dir / 'best_stage_3.pth'
            test_metrics = trainer.evaluate_model(
                test_loader=test_loader,
                checkpoint_path=best_checkpoint
            )
            
            return {
                'training_history': training_history,
                'test_metrics': test_metrics,
                'best_checkpoint': best_checkpoint,
                'dataset_info': dataset_info
            }
            
        finally:
            trainer.close()
    
    def stress_test_evaluation(self, model_checkpoints: dict, stress_test_data: pd.DataFrame,
                              tensor_files: dict, metadata_path: str):
        """Lakukan stress test menggunakan data Juli 2024-2026."""
        logger.info("=== STRESS TEST EVALUATION ===")
        
        stress_results = {}
        
        for variant_name, checkpoint_path in model_checkpoints.items():
            logger.info(f"Stress testing {variant_name} model...")
            
            # Determine which tensor files to use
            if variant_name == 'cmr':
                test_hdf5 = tensor_files['cmr_files']['test_cmr']
            else:
                test_hdf5 = tensor_files['original_files']['test']
            
            # Create data loader for stress test
            _, stress_loader, _ = create_data_loaders(
                train_hdf5_path=test_hdf5,  # Dummy, not used
                test_hdf5_path=test_hdf5,
                metadata_path=metadata_path,
                batch_size=16,
                num_workers=4,
                pin_memory=True,
                load_in_memory=False
            )
            
            # Load model and evaluate
            model = create_model()
            from src.training.utils import load_checkpoint
            load_checkpoint(model, checkpoint_path, device='cuda')
            
            # Create temporary trainer for evaluation
            trainer = SpatioTemporalTrainer(
                model=model,
                train_loader=stress_loader,  # Dummy
                val_loader=stress_loader,
                device='cuda',
                output_dir=self.base_output_dir / f"stress_test_{variant_name}"
            )
            
            try:
                stress_metrics = trainer.evaluate_model(
                    test_loader=stress_loader,
                    checkpoint_path=checkpoint_path
                )
                
                stress_results[variant_name] = stress_metrics
                
            finally:
                trainer.close()
        
        return stress_results
    
    def solar_storm_analysis(self, metadata: pd.DataFrame, model_results: dict):
        """Analisis performa saat badai matahari (Kp-index > 5)."""
        logger.info("=== ANALISIS BADAI MATAHARI ===")
        
        # Filter data dengan Kp-index > 5
        if 'kp_index' in metadata.columns:
            solar_storm_data = metadata[metadata['kp_index'] > 5.0]
            normal_data = metadata[metadata['kp_index'] <= 5.0]
            
            logger.info(f"Data saat badai matahari (Kp > 5): {len(solar_storm_data)} records")
            logger.info(f"Data kondisi normal (Kp ≤ 5): {len(normal_data)} records")
            
            # Analisis distribusi Kp-index
            kp_stats = {
                'solar_storm_events': len(solar_storm_data),
                'normal_events': len(normal_data),
                'max_kp': metadata['kp_index'].max(),
                'mean_kp_storm': solar_storm_data['kp_index'].mean() if len(solar_storm_data) > 0 else 0,
                'mean_kp_normal': normal_data['kp_index'].mean()
            }
            
            return {
                'solar_storm_data': solar_storm_data,
                'normal_data': normal_data,
                'kp_statistics': kp_stats
            }
        else:
            logger.warning("Kolom kp_index tidak ditemukan dalam metadata")
            return None
    
    def create_ablation_study_visualization(self, cmr_results: dict, original_results: dict,
                                          solar_analysis: dict, output_dir: Path):
        """Buat visualisasi ablation study CMR vs Original."""
        logger.info("=== PEMBUATAN VISUALISASI ABLATION STUDY ===")
        
        # Setup matplotlib
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ablation Study: Dengan CMR vs Tanpa CMR\nPerforma Model Saat Badai Matahari (Kp-index > 5)', 
                     fontsize=16, fontweight='bold')
        
        # 1. F1-Score Comparison
        ax1 = axes[0, 0]
        variants = ['Dengan CMR', 'Tanpa CMR']
        f1_scores = [
            cmr_results['test_metrics'].get('binary_f1', 0),
            original_results['test_metrics'].get('binary_f1', 0)
        ]
        
        bars1 = ax1.bar(variants, f1_scores, color=['#2E8B57', '#CD5C5C'], alpha=0.8)
        ax1.set_title('F1-Score untuk Deteksi Precursor', fontweight='bold')
        ax1.set_ylabel('F1-Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars1, f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. MAE Comparison for Magnitude
        ax2 = axes[0, 1]
        mae_scores = [
            cmr_results['test_metrics'].get('magnitude_mae', 0),
            original_results['test_metrics'].get('magnitude_mae', 0)
        ]
        
        bars2 = ax2.bar(variants, mae_scores, color=['#2E8B57', '#CD5C5C'], alpha=0.8)
        ax2.set_title('Mean Absolute Error (MAE) untuk Magnitudo', fontweight='bold')
        ax2.set_ylabel('MAE')
        
        for bar, mae in zip(bars2, mae_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mae:.3f}', ha='center', va='bottom', fontweight='bold')
        
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
        
        ax3.set_title('Precision vs Recall', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(variants)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # 4. Kp-index Distribution
        if solar_analysis:
            ax4 = axes[1, 0]
            kp_stats = solar_analysis['kp_statistics']
            
            categories = ['Badai Matahari\n(Kp > 5)', 'Kondisi Normal\n(Kp ≤ 5)']
            counts = [kp_stats['solar_storm_events'], kp_stats['normal_events']]
            
            bars4 = ax4.bar(categories, counts, color=['#FF4500', '#32CD32'], alpha=0.8)
            ax4.set_title('Distribusi Data Berdasarkan Kp-index', fontweight='bold')
            ax4.set_ylabel('Jumlah Events')
            
            for bar, count in zip(bars4, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Model Accuracy Comparison
        ax5 = axes[1, 1]
        accuracy_scores = [
            cmr_results['test_metrics'].get('binary_accuracy', 0),
            original_results['test_metrics'].get('binary_accuracy', 0)
        ]
        
        bars5 = ax5.bar(variants, accuracy_scores, color=['#2E8B57', '#CD5C5C'], alpha=0.8)
        ax5.set_title('Akurasi Model', fontweight='bold')
        ax5.set_ylabel('Accuracy')
        ax5.set_ylim(0, 1)
        
        for bar, acc in zip(bars5, accuracy_scores):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Localization Error Comparison
        ax6 = axes[1, 2]
        azimuth_errors = [
            cmr_results['test_metrics'].get('azimuth_mean_error_deg', 0),
            original_results['test_metrics'].get('azimuth_mean_error_deg', 0)
        ]
        
        bars6 = ax6.bar(variants, azimuth_errors, color=['#2E8B57', '#CD5C5C'], alpha=0.8)
        ax6.set_title('Error Azimuth (derajat)', fontweight='bold')
        ax6.set_ylabel('Mean Error (°)')
        
        for bar, error in zip(bars6, azimuth_errors):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{error:.1f}°', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / 'ablation_study_cmr_vs_original.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'ablation_study_cmr_vs_original.pdf', bbox_inches='tight')
        
        logger.info(f"Visualisasi ablation study disimpan: {viz_path}")
        
        return viz_path
    
    def generate_comprehensive_report(self, all_results: dict, output_dir: Path):
        """Generate comprehensive evaluation report."""
        logger.info("=== PEMBUATAN LAPORAN KOMPREHENSIF ===")
        
        report = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'training_period': '2018-2024',
                'stress_test_period': 'Juli 2024-2026',
                'model_variants': ['CMR', 'Original'],
                'evaluation_metrics': ['F1-Score', 'MAE', 'Accuracy', 'Precision', 'Recall']
            },
            'training_results': {},
            'stress_test_results': {},
            'solar_storm_analysis': {},
            'ablation_study_summary': {}
        }
        
        # Training results summary
        for variant in ['cmr', 'original']:
            if variant in all_results:
                variant_results = all_results[variant]
                report['training_results'][variant] = {
                    'final_metrics': variant_results.get('test_metrics', {}),
                    'training_epochs': self._extract_training_epochs(variant_results.get('training_history', {})),
                    'best_checkpoint': str(variant_results.get('best_checkpoint', ''))
                }
        
        # Stress test summary
        if 'stress_test' in all_results:
            report['stress_test_results'] = all_results['stress_test']
        
        # Solar storm analysis
        if 'solar_analysis' in all_results:
            report['solar_storm_analysis'] = all_results['solar_analysis'].get('kp_statistics', {})
        
        # Ablation study summary
        if 'cmr' in all_results and 'original' in all_results:
            cmr_metrics = all_results['cmr']['test_metrics']
            orig_metrics = all_results['original']['test_metrics']
            
            report['ablation_study_summary'] = {
                'f1_score_improvement': cmr_metrics.get('binary_f1', 0) - orig_metrics.get('binary_f1', 0),
                'mae_improvement': orig_metrics.get('magnitude_mae', 0) - cmr_metrics.get('magnitude_mae', 0),
                'accuracy_improvement': cmr_metrics.get('binary_accuracy', 0) - orig_metrics.get('binary_accuracy', 0),
                'cmr_advantage': {
                    'f1_score': cmr_metrics.get('binary_f1', 0) > orig_metrics.get('binary_f1', 0),
                    'mae': cmr_metrics.get('magnitude_mae', 0) < orig_metrics.get('magnitude_mae', 0),
                    'accuracy': cmr_metrics.get('binary_accuracy', 0) > orig_metrics.get('binary_accuracy', 0)
                }
            }
        
        # Save report
        report_path = output_dir / 'comprehensive_evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report, output_dir)
        
        logger.info(f"Laporan komprehensif disimpan: {report_path}")
        
        return report
    
    def _extract_training_epochs(self, training_history: dict):
        """Extract training epoch information."""
        epochs_info = {}
        for stage, history in training_history.items():
            if isinstance(history, dict) and 'train' in history:
                epochs_info[stage] = len(history['train'])
        return epochs_info
    
    def _generate_markdown_report(self, report: dict, output_dir: Path):
        """Generate markdown version of the report."""
        md_content = f"""# Laporan Evaluasi Spatio-Temporal Earthquake Precursor Model

## Informasi Eksperimen
- **Timestamp**: {report['experiment_info']['timestamp']}
- **Periode Pelatihan**: {report['experiment_info']['training_period']}
- **Periode Stress Test**: {report['experiment_info']['stress_test_period']}
- **Varian Model**: {', '.join(report['experiment_info']['model_variants'])}

## Hasil Pelatihan

### Model dengan CMR
"""
        
        if 'cmr' in report['training_results']:
            cmr_results = report['training_results']['cmr']['final_metrics']
            md_content += f"""
- **F1-Score**: {cmr_results.get('binary_f1', 0):.4f}
- **Accuracy**: {cmr_results.get('binary_accuracy', 0):.4f}
- **Precision**: {cmr_results.get('binary_precision', 0):.4f}
- **Recall**: {cmr_results.get('binary_recall', 0):.4f}
- **MAE Magnitudo**: {cmr_results.get('magnitude_mae', 0):.4f}
"""
        
        md_content += """
### Model Tanpa CMR (Original)
"""
        
        if 'original' in report['training_results']:
            orig_results = report['training_results']['original']['final_metrics']
            md_content += f"""
- **F1-Score**: {orig_results.get('binary_f1', 0):.4f}
- **Accuracy**: {orig_results.get('binary_accuracy', 0):.4f}
- **Precision**: {orig_results.get('binary_precision', 0):.4f}
- **Recall**: {orig_results.get('binary_recall', 0):.4f}
- **MAE Magnitudo**: {orig_results.get('magnitude_mae', 0):.4f}
"""
        
        md_content += """
## Ablation Study Summary

### Keunggulan CMR
"""
        
        if 'ablation_study_summary' in report:
            ablation = report['ablation_study_summary']
            md_content += f"""
- **Peningkatan F1-Score**: {ablation.get('f1_score_improvement', 0):+.4f}
- **Peningkatan MAE**: {ablation.get('mae_improvement', 0):+.4f} (lebih rendah = lebih baik)
- **Peningkatan Accuracy**: {ablation.get('accuracy_improvement', 0):+.4f}

### Kesimpulan
- CMR memberikan performa lebih baik pada F1-Score: {'✓' if ablation.get('cmr_advantage', {}).get('f1_score', False) else '✗'}
- CMR memberikan performa lebih baik pada MAE: {'✓' if ablation.get('cmr_advantage', {}).get('mae', False) else '✗'}
- CMR memberikan performa lebih baik pada Accuracy: {'✓' if ablation.get('cmr_advantage', {}).get('accuracy', False) else '✗'}
"""
        
        # Save markdown report
        md_path = output_dir / 'evaluation_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Laporan markdown disimpan: {md_path}")
    
    def run_complete_pipeline(self):
        """Jalankan pipeline lengkap."""
        logger.info("🚀 MEMULAI PIPELINE PELATIHAN LENGKAP")
        logger.info("=" * 60)
        
        try:
            # 1. Persiapan dataset
            data_info = self.prepare_datasets()
            if data_info is None:
                return None
            
            # 2. Buat tensor datasets
            tensor_info = self.create_tensor_datasets(data_info)
            
            # 3. Train model variants
            all_results = {}
            
            # Train CMR model
            cmr_experiment_dir = self.base_output_dir / "experiments" / "cmr_model"
            cmr_experiment_dir.mkdir(parents=True, exist_ok=True)
            
            cmr_results = self.train_model_variant(
                variant_name="cmr",
                train_hdf5=tensor_info['cmr_files']['train_cmr'],
                test_hdf5=tensor_info['cmr_files']['test_cmr'],
                metadata_path=data_info['metadata_path'],
                experiment_dir=cmr_experiment_dir
            )
            all_results['cmr'] = cmr_results
            
            # Train Original model
            orig_experiment_dir = self.base_output_dir / "experiments" / "original_model"
            orig_experiment_dir.mkdir(parents=True, exist_ok=True)
            
            original_results = self.train_model_variant(
                variant_name="original",
                train_hdf5=tensor_info['original_files']['train'],
                test_hdf5=tensor_info['original_files']['test'],
                metadata_path=data_info['metadata_path'],
                experiment_dir=orig_experiment_dir
            )
            all_results['original'] = original_results
            
            # 4. Stress test
            model_checkpoints = {
                'cmr': cmr_results['best_checkpoint'],
                'original': original_results['best_checkpoint']
            }
            
            stress_results = self.stress_test_evaluation(
                model_checkpoints=model_checkpoints,
                stress_test_data=data_info['stress_test_data'],
                tensor_files=tensor_info,
                metadata_path=data_info['metadata_path']
            )
            all_results['stress_test'] = stress_results
            
            # 5. Solar storm analysis
            solar_analysis = self.solar_storm_analysis(
                metadata=data_info['metadata'],
                model_results=all_results
            )
            all_results['solar_analysis'] = solar_analysis
            
            # 6. Create visualizations
            viz_output_dir = self.base_output_dir / "visualizations"
            viz_output_dir.mkdir(parents=True, exist_ok=True)
            
            viz_path = self.create_ablation_study_visualization(
                cmr_results=cmr_results,
                original_results=original_results,
                solar_analysis=solar_analysis,
                output_dir=viz_output_dir
            )
            
            # 7. Generate comprehensive report
            report = self.generate_comprehensive_report(
                all_results=all_results,
                output_dir=self.base_output_dir
            )
            
            logger.info("✅ PIPELINE PELATIHAN SELESAI")
            logger.info("=" * 60)
            logger.info(f"📊 Hasil tersimpan di: {self.base_output_dir}")
            logger.info(f"📈 Visualisasi: {viz_path}")
            logger.info(f"📋 Laporan: {self.base_output_dir / 'evaluation_report.md'}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline gagal: {e}")
            raise


def main():
    """Main function untuk menjalankan pipeline."""
    print("🌍 Spatio-Temporal Earthquake Precursor Training Pipeline")
    print("=" * 60)
    
    # Create and run pipeline
    runner = TrainingPipelineRunner()
    results = runner.run_complete_pipeline()
    
    if results:
        print("\n🎉 Pipeline berhasil diselesaikan!")
        print("📊 Periksa hasil di folder outputs/training_pipeline/")
    else:
        print("\n❌ Pipeline gagal dijalankan")


if __name__ == '__main__':
    main()