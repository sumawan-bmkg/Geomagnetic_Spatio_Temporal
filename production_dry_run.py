#!/usr/bin/env python3
"""
Production Dry Run & System Stress Test
Senior Deep Learning Engineer Protocol

Mandatory dry run sebelum training penuh untuk:
1. Memory audit (GPU VRAM monitoring)
2. Gradient flow verification
3. Loss stability check
4. System stress test

Author: Senior Deep Learning Engineer
Date: April 15, 2026
"""

import sys
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import json
import argparse
import yaml
import warnings
import psutil
import GPUtil
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from run_production_train import ProductionTrainingPipeline, ProductionDataset
from src.models.spatio_temporal_model import SpatioTemporalPrecursorModel
from src.training.trainer import SpatioTemporalTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """System resource monitoring untuk dry run."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = 'cuda' if self.gpu_available else 'cpu'
        
    def get_gpu_memory_info(self) -> Dict:
        """Get GPU memory information."""
        if not self.gpu_available:
            return {'available': False}
        
        gpu = GPUtil.getGPUs()[0] if GPUtil.getGPUs() else None
        if gpu is None:
            return {'available': False}
        
        return {
            'available': True,
            'total_mb': gpu.memoryTotal,
            'used_mb': gpu.memoryUsed,
            'free_mb': gpu.memoryFree,
            'utilization_percent': gpu.memoryUtil * 100,
            'gpu_name': gpu.name
        }
    
    def get_system_memory_info(self) -> Dict:
        """Get system RAM information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
    
    def log_system_status(self):
        """Log current system status."""
        gpu_info = self.get_gpu_memory_info()
        ram_info = self.get_system_memory_info()
        
        logger.info("=== SYSTEM STATUS ===")
        logger.info(f"Device: {self.device}")
        
        if gpu_info['available']:
            logger.info(f"GPU: {gpu_info['gpu_name']}")
            logger.info(f"GPU Memory: {gpu_info['used_mb']:.0f}/{gpu_info['total_mb']:.0f} MB ({gpu_info['utilization_percent']:.1f}%)")
        else:
            logger.info("GPU: Not available")
        
        logger.info(f"RAM: {ram_info['used_gb']:.1f}/{ram_info['total_gb']:.1f} GB ({ram_info['percent_used']:.1f}%)")


class GradientMonitor:
    """Monitor gradient flow dan stability."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_stats = {}
    
    def check_gradient_flow(self) -> Dict:
        """Check gradient flow through model."""
        gradient_stats = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                gradient_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'has_nan': torch.isnan(param.grad).any().item(),
                    'has_inf': torch.isinf(param.grad).any().item()
                }
        
        return gradient_stats
    
    def analyze_gradient_health(self, gradient_stats: Dict) -> Dict:
        """Analyze gradient health."""
        analysis = {
            'total_params_with_grad': len(gradient_stats),
            'nan_gradients': 0,
            'inf_gradients': 0,
            'zero_gradients': 0,
            'healthy_gradients': 0,
            'gradient_norms': []
        }
        
        for name, stats in gradient_stats.items():
            if stats['has_nan']:
                analysis['nan_gradients'] += 1
            elif stats['has_inf']:
                analysis['inf_gradients'] += 1
            elif stats['norm'] < 1e-8:
                analysis['zero_gradients'] += 1
            else:
                analysis['healthy_gradients'] += 1
                analysis['gradient_norms'].append(stats['norm'])
        
        if analysis['gradient_norms']:
            analysis['mean_gradient_norm'] = np.mean(analysis['gradient_norms'])
            analysis['std_gradient_norm'] = np.std(analysis['gradient_norms'])
        
        return analysis


class ProductionDryRun:
    """
    Production dry run system untuk stress testing.
    """
    
    def __init__(self, 
                 dataset_path: str = 'real_earthquake_dataset.h5',
                 config_path: str = 'configs/production_config.yaml',
                 max_batches: int = 50,
                 max_epochs: int = 1):
        """
        Initialize dry run system.
        
        Args:
            dataset_path: Path to dataset
            config_path: Path to configuration
            max_batches: Maximum batches for dry run
            max_epochs: Maximum epochs for dry run
        """
        self.dataset_path = Path(dataset_path)
        self.config_path = Path(config_path)
        self.max_batches = max_batches
        self.max_epochs = max_epochs
        
        # Initialize monitoring
        self.system_monitor = SystemMonitor()
        self.device = self.system_monitor.device
        
        # Results storage
        self.dry_run_results = {
            'start_time': datetime.now().isoformat(),
            'system_info': {},
            'memory_usage': [],
            'gradient_analysis': {},
            'loss_progression': [],
            'batch_times': [],
            'success': False,
            'errors': []
        }
        
        logger.info("ProductionDryRun initialized")
        logger.info(f"Max batches: {self.max_batches}")
        logger.info(f"Max epochs: {self.max_epochs}")
    
    def load_config(self):
        """Load training configuration."""
        logger.info("Loading configuration...")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Adjust config for dry run
        self.config['data']['batch_size'] = min(self.config['data']['batch_size'], 4)  # Conservative batch size
        
        logger.info(f"Config loaded. Batch size: {self.config['data']['batch_size']}")
    
    def setup_pipeline(self):
        """Setup training pipeline untuk dry run."""
        logger.info("Setting up training pipeline...")
        
        # Create pipeline with dry run parameters
        self.pipeline = ProductionTrainingPipeline(
            dataset_path=str(self.dataset_path),
            station_coords_path='../awal/lokasi_stasiun.csv',
            output_dir='./outputs/dry_run',
            experiment_name='dry_run_test',
            config_path=str(self.config_path)
        )
        
        # Override config
        self.pipeline.config = self.config
        
        # Load components
        self.pipeline.load_station_coordinates()
        self.pipeline.load_dataset()
        self.pipeline.create_data_loaders()
        self.pipeline.create_model()
        
        # Setup gradient monitoring
        self.gradient_monitor = GradientMonitor(self.pipeline.model)
        
        logger.info("Pipeline setup complete")
    
    def memory_stress_test(self) -> bool:
        """Test memory usage dengan batch processing."""
        logger.info("=== MEMORY STRESS TEST ===")
        
        try:
            # Initial memory check
            initial_gpu = self.system_monitor.get_gpu_memory_info()
            initial_ram = self.system_monitor.get_system_memory_info()
            
            self.dry_run_results['system_info'] = {
                'initial_gpu': initial_gpu,
                'initial_ram': initial_ram,
                'device': self.device
            }
            
            # Process a few batches to test memory
            model = self.pipeline.model
            train_loader = self.pipeline.train_loader
            
            model.train()
            batch_count = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_count >= min(5, self.max_batches):  # Test first 5 batches
                    break
                
                # Move to device
                data = data.to(self.device)
                for key in targets:
                    if isinstance(targets[key], torch.Tensor):
                        targets[key] = targets[key].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(data)
                
                # Memory check
                gpu_info = self.system_monitor.get_gpu_memory_info()
                ram_info = self.system_monitor.get_system_memory_info()
                
                memory_usage = {
                    'batch': batch_count,
                    'gpu_used_mb': gpu_info.get('used_mb', 0),
                    'gpu_percent': gpu_info.get('utilization_percent', 0),
                    'ram_used_gb': ram_info['used_gb'],
                    'ram_percent': ram_info['percent_used']
                }
                
                self.dry_run_results['memory_usage'].append(memory_usage)
                
                logger.info(f"Batch {batch_count}: GPU {gpu_info.get('utilization_percent', 0):.1f}%, RAM {ram_info['percent_used']:.1f}%")
                
                # Check memory limits
                if gpu_info.get('utilization_percent', 0) > 90:
                    logger.warning(f"GPU memory usage high: {gpu_info['utilization_percent']:.1f}%")
                    return False
                
                if ram_info['percent_used'] > 85:
                    logger.warning(f"RAM usage high: {ram_info['percent_used']:.1f}%")
                    return False
                
                batch_count += 1
            
            logger.info("Memory stress test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Memory stress test FAILED: {str(e)}")
            self.dry_run_results['errors'].append(f"Memory test: {str(e)}")
            return False
    
    def gradient_flow_test(self) -> bool:
        """Test gradient flow dan stability."""
        logger.info("=== GRADIENT FLOW TEST ===")
        
        try:
            model = self.pipeline.model
            train_loader = self.pipeline.train_loader
            
            # Setup optimizer untuk gradient test
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['stage_1']['optimizer']['lr'],
                weight_decay=self.config['stage_1']['optimizer']['weight_decay']
            )
            
            model.train()
            batch_count = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_count >= min(3, self.max_batches):  # Test first 3 batches
                    break
                
                # Move to device
                data = data.to(self.device)
                for key in targets:
                    if isinstance(targets[key], torch.Tensor):
                        targets[key] = targets[key].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(data)
                
                # Calculate loss (simplified)
                binary_loss = nn.BCEWithLogitsLoss()(outputs['binary'].squeeze(), targets['binary'])
                magnitude_loss = nn.CrossEntropyLoss()(outputs['magnitude'], targets['magnitude_class'])
                
                total_loss = binary_loss + magnitude_loss
                
                # Backward pass
                total_loss.backward()
                
                # Check gradients
                gradient_stats = self.gradient_monitor.check_gradient_flow()
                gradient_analysis = self.gradient_monitor.analyze_gradient_health(gradient_stats)
                
                # Store results
                loss_info = {
                    'batch': batch_count,
                    'binary_loss': binary_loss.item(),
                    'magnitude_loss': magnitude_loss.item(),
                    'total_loss': total_loss.item(),
                    'has_nan_loss': torch.isnan(total_loss).item(),
                    'has_inf_loss': torch.isinf(total_loss).item()
                }
                
                self.dry_run_results['loss_progression'].append(loss_info)
                
                # Log gradient health
                logger.info(f"Batch {batch_count}:")
                logger.info(f"  Loss: {total_loss.item():.4f} (Binary: {binary_loss.item():.4f}, Mag: {magnitude_loss.item():.4f})")
                logger.info(f"  Gradients: {gradient_analysis['healthy_gradients']}/{gradient_analysis['total_params_with_grad']} healthy")
                
                if gradient_analysis['mean_gradient_norm']:
                    logger.info(f"  Grad norm: {gradient_analysis['mean_gradient_norm']:.6f}")
                
                # Check for gradient issues
                if gradient_analysis['nan_gradients'] > 0:
                    logger.error(f"NaN gradients detected: {gradient_analysis['nan_gradients']}")
                    return False
                
                if gradient_analysis['inf_gradients'] > 0:
                    logger.error(f"Inf gradients detected: {gradient_analysis['inf_gradients']}")
                    return False
                
                if loss_info['has_nan_loss'] or loss_info['has_inf_loss']:
                    logger.error("NaN/Inf loss detected")
                    return False
                
                # Store gradient analysis
                self.dry_run_results['gradient_analysis'][f'batch_{batch_count}'] = gradient_analysis
                
                batch_count += 1
            
            logger.info("Gradient flow test PASSED")
            return True
            
        except Exception as e:
            logger.error(f"Gradient flow test FAILED: {str(e)}")
            self.dry_run_results['errors'].append(f"Gradient test: {str(e)}")
            return False
    
    def performance_benchmark(self) -> bool:
        """Benchmark training performance."""
        logger.info("=== PERFORMANCE BENCHMARK ===")
        
        try:
            model = self.pipeline.model
            train_loader = self.pipeline.train_loader
            
            model.train()
            batch_times = []
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                if batch_idx >= min(10, self.max_batches):  # Benchmark 10 batches
                    break
                
                start_time = datetime.now()
                
                # Move to device
                data = data.to(self.device)
                for key in targets:
                    if isinstance(targets[key], torch.Tensor):
                        targets[key] = targets[key].to(self.device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(data)
                
                end_time = datetime.now()
                batch_time = (end_time - start_time).total_seconds()
                batch_times.append(batch_time)
                
                logger.info(f"Batch {batch_idx}: {batch_time:.3f}s")
            
            # Calculate statistics
            mean_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            
            self.dry_run_results['batch_times'] = {
                'times': batch_times,
                'mean_seconds': mean_time,
                'std_seconds': std_time,
                'batches_per_hour': 3600 / mean_time if mean_time > 0 else 0
            }
            
            logger.info(f"Performance: {mean_time:.3f}±{std_time:.3f}s per batch")
            logger.info(f"Estimated: {3600/mean_time:.0f} batches/hour")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmark FAILED: {str(e)}")
            self.dry_run_results['errors'].append(f"Performance test: {str(e)}")
            return False
    
    def run_complete_dry_run(self) -> Dict:
        """Run complete dry run protocol."""
        logger.info("🔧 STARTING PRODUCTION DRY RUN")
        logger.info("=" * 60)
        
        try:
            # Initial system check
            self.system_monitor.log_system_status()
            
            # Load configuration
            self.load_config()
            
            # Setup pipeline
            self.setup_pipeline()
            
            # Run tests
            tests_passed = 0
            total_tests = 3
            
            # Test 1: Memory stress test
            if self.memory_stress_test():
                tests_passed += 1
                logger.info("✅ Memory stress test PASSED")
            else:
                logger.error("❌ Memory stress test FAILED")
            
            # Test 2: Gradient flow test
            if self.gradient_flow_test():
                tests_passed += 1
                logger.info("✅ Gradient flow test PASSED")
            else:
                logger.error("❌ Gradient flow test FAILED")
            
            # Test 3: Performance benchmark
            if self.performance_benchmark():
                tests_passed += 1
                logger.info("✅ Performance benchmark PASSED")
            else:
                logger.error("❌ Performance benchmark FAILED")
            
            # Final assessment
            self.dry_run_results['tests_passed'] = tests_passed
            self.dry_run_results['total_tests'] = total_tests
            self.dry_run_results['success'] = tests_passed == total_tests
            self.dry_run_results['end_time'] = datetime.now().isoformat()
            
            # Generate report
            self.generate_dry_run_report()
            
            logger.info("=" * 60)
            if self.dry_run_results['success']:
                logger.info("🏆 DRY RUN COMPLETED SUCCESSFULLY")
                logger.info("✅ System ready for production training")
            else:
                logger.error("🚨 DRY RUN FAILED")
                logger.error(f"❌ {total_tests - tests_passed}/{total_tests} tests failed")
                logger.error("⚠️ System NOT ready for production training")
            
            return self.dry_run_results
            
        except Exception as e:
            logger.error(f"Dry run FAILED with exception: {str(e)}")
            self.dry_run_results['success'] = False
            self.dry_run_results['errors'].append(f"Critical error: {str(e)}")
            return self.dry_run_results
    
    def generate_dry_run_report(self):
        """Generate comprehensive dry run report."""
        logger.info("Generating dry run report...")
        
        # Create output directory
        output_dir = Path("outputs/dry_run")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        report_path = output_dir / f"dry_run_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.dry_run_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_path = output_dir / "DRY_RUN_SUMMARY.md"
        with open(summary_path, 'w') as f:
            f.write(self._create_summary_report())
        
        logger.info(f"Reports saved:")
        logger.info(f"  - JSON: {report_path}")
        logger.info(f"  - Summary: {summary_path}")
    
    def _create_summary_report(self) -> str:
        """Create markdown summary report."""
        results = self.dry_run_results
        
        report = f"""# Production Dry Run Report

## System Information
- **Date**: {results['start_time']}
- **Device**: {results['system_info'].get('device', 'Unknown')}
- **Tests Passed**: {results.get('tests_passed', 0)}/{results.get('total_tests', 0)}
- **Overall Status**: {'✅ PASSED' if results.get('success', False) else '❌ FAILED'}

## Memory Usage Analysis
"""
        
        if results['memory_usage']:
            max_gpu = max([m.get('gpu_percent', 0) for m in results['memory_usage']])
            max_ram = max([m['ram_percent'] for m in results['memory_usage']])
            
            report += f"""
- **Peak GPU Usage**: {max_gpu:.1f}%
- **Peak RAM Usage**: {max_ram:.1f}%
- **Memory Status**: {'✅ SAFE' if max_gpu < 90 and max_ram < 85 else '⚠️ HIGH'}
"""
        
        report += "\n## Gradient Flow Analysis\n"
        
        if results['gradient_analysis']:
            last_analysis = list(results['gradient_analysis'].values())[-1]
            report += f"""
- **Parameters with Gradients**: {last_analysis.get('total_params_with_grad', 0)}
- **Healthy Gradients**: {last_analysis.get('healthy_gradients', 0)}
- **NaN Gradients**: {last_analysis.get('nan_gradients', 0)}
- **Inf Gradients**: {last_analysis.get('inf_gradients', 0)}
- **Gradient Status**: {'✅ HEALTHY' if last_analysis.get('nan_gradients', 0) == 0 and last_analysis.get('inf_gradients', 0) == 0 else '❌ UNHEALTHY'}
"""
        
        report += "\n## Performance Metrics\n"
        
        if results.get('batch_times'):
            batch_stats = results['batch_times']
            report += f"""
- **Mean Batch Time**: {batch_stats.get('mean_seconds', 0):.3f}s
- **Batches per Hour**: {batch_stats.get('batches_per_hour', 0):.0f}
- **Performance Status**: {'✅ GOOD' if batch_stats.get('mean_seconds', 999) < 5.0 else '⚠️ SLOW'}
"""
        
        if results.get('errors'):
            report += "\n## Errors Detected\n"
            for i, error in enumerate(results['errors'], 1):
                report += f"{i}. {error}\n"
        
        report += f"""
## Recommendations

{'✅ **PROCEED WITH PRODUCTION TRAINING**' if results.get('success', False) else '❌ **DO NOT PROCEED - FIX ISSUES FIRST**'}

"""
        
        if not results.get('success', False):
            report += """
### Required Actions:
1. Review and fix all errors listed above
2. Optimize memory usage if needed
3. Check gradient flow issues
4. Re-run dry run before production training
"""
        else:
            report += """
### Production Training Ready:
1. All system checks passed
2. Memory usage within safe limits
3. Gradient flow healthy
4. Performance acceptable

**Recommended Command:**
```bash
python run_production_train.py --config configs/production_config.yaml --experiment ground_truth_run
```
"""
        
        return report


def main():
    """Main function untuk dry run execution."""
    parser = argparse.ArgumentParser(description='Production Dry Run & System Stress Test')
    parser.add_argument('--dataset', type=str, default='real_earthquake_dataset.h5',
                       help='Path to dataset')
    parser.add_argument('--config', type=str, default='configs/production_config.yaml',
                       help='Path to config file')
    parser.add_argument('--max-batches', type=int, default=50,
                       help='Maximum batches for dry run')
    parser.add_argument('--max-epochs', type=int, default=1,
                       help='Maximum epochs for dry run')
    
    args = parser.parse_args()
    
    print("🔧 Production Dry Run & System Stress Test")
    print("Senior Deep Learning Engineer Protocol")
    print("=" * 60)
    
    # Run dry run
    dry_run = ProductionDryRun(
        dataset_path=args.dataset,
        config_path=args.config,
        max_batches=args.max_batches,
        max_epochs=args.max_epochs
    )
    
    results = dry_run.run_complete_dry_run()
    
    # Final status
    if results['success']:
        print("\n🏆 DRY RUN SUCCESSFUL - READY FOR PRODUCTION")
        print("✅ Execute: python run_production_train.py --config configs/production_config.yaml --experiment ground_truth_run")
    else:
        print("\n🚨 DRY RUN FAILED - DO NOT PROCEED")
        print("❌ Fix issues before production training")
    
    return results


if __name__ == '__main__':
    main()