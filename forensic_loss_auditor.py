#!/usr/bin/env python3
"""
FORENSIC LOSS AUDITOR - Deep Investigation of Loss Collapse to 0.000000
Persona: AI Debugging Expert & Senior Data Scientist

Author: Kiro AI Assistant
Date: April 16, 2026
"""
import sys
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.spatio_temporal_model import SpatioTemporalPrecursorModel
from production_dataset_adapter import ProductionDatasetAdapter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forensic_audit.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForensicLossAuditor:
    """
    Deep forensic auditor for loss collapse investigation
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audit_results = {}
        
        logger.info("=== FORENSIC LOSS AUDITOR INITIALIZED ===")
        logger.info("Investigating Loss Collapse to 0.000000 at Batch 100")
    
    def audit_1_output_label_uniformity(self):
        """AUDIT 1: Check for label uniformity and model output patterns"""
        logger.info("=== AUDIT 1: OUTPUT & LABEL UNIFORMITY CHECK ===")
        
        # Create model and dataset
        station_coordinates = np.array([
            [-6.2, 106.8], [-7.8, 110.4], [-6.9, 107.6], [-7.3, 112.7],
            [-8.7, 115.2], [-0.9, 100.4], [3.6, 98.7], [-5.1, 119.4]
        ])
        
        model = SpatioTemporalPrecursorModel(
            n_stations=8, n_components=3, station_coordinates=station_coordinates,
            efficientnet_pretrained=True, gnn_hidden_dim=256, gnn_num_layers=3,
            dropout_rate=0.3, device=str(self.device)
        ).to(self.device)
        
        # Create dataset with STRICT SHUFFLING
        full_dataset = ProductionDatasetAdapter('real_earthquake_dataset.h5', split='train')
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, _ = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loader with STRICT SHUFFLING
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, 
            num_workers=0, drop_last=True  # Drop last to ensure consistent batch sizes
        )
        
        logger.info("Analyzing first 10 batches for uniformity patterns...")
        
        model.eval()
        batch_analysis = []
        
        with torch.no_grad():
            for batch_idx, (tensors, targets) in enumerate(train_loader):
                if batch_idx >= 10:  # Analyze first 10 batches
                    break
                
                tensors = tensors.to(self.device)
                outputs = model(tensors)
                
                # Extract binary targets and predictions
                binary_targets = targets['binary'].numpy()
                binary_preds = outputs['binary_probs'].cpu().numpy().squeeze()
                
                # Analyze uniformity
                target_unique = np.unique(binary_targets)
                pred_variance = np.var(binary_preds)
                pred_mean = np.mean(binary_preds)
                
                batch_info = {
                    'batch_idx': batch_idx,
                    'target_unique_values': target_unique.tolist(),
                    'target_uniformity': len(target_unique) == 1,
                    'all_targets_same': len(target_unique) == 1,
                    'target_value_if_uniform': target_unique[0] if len(target_unique) == 1 else None,
                    'pred_mean': float(pred_mean),
                    'pred_variance': float(pred_variance),
                    'pred_min': float(np.min(binary_preds)),
                    'pred_max': float(np.max(binary_preds)),
                    'identical_predictions': pred_variance < 1e-6,
                    'binary_targets': binary_targets.tolist(),
                    'binary_predictions': binary_preds.tolist()
                }
                
                batch_analysis.append(batch_info)
                
                logger.info(f"Batch {batch_idx}:")
                logger.info(f"  Target unique values: {target_unique}")
                logger.info(f"  All targets same: {len(target_unique) == 1}")
                logger.info(f"  Pred mean: {pred_mean:.6f}, variance: {pred_variance:.6f}")
                logger.info(f"  Identical predictions: {pred_variance < 1e-6}")
                
                if len(target_unique) == 1:
                    logger.warning(f"  🚨 UNIFORM LABELS DETECTED: All targets = {target_unique[0]}")
                
                if pred_variance < 1e-6:
                    logger.warning(f"  🚨 IDENTICAL PREDICTIONS DETECTED: Variance = {pred_variance:.8f}")
        
        # Summary analysis
        uniform_batches = sum(1 for b in batch_analysis if b['target_uniformity'])
        identical_pred_batches = sum(1 for b in batch_analysis if b['identical_predictions'])
        
        self.audit_results['uniformity_check'] = {
            'total_batches_analyzed': len(batch_analysis),
            'uniform_label_batches': uniform_batches,
            'identical_prediction_batches': identical_pred_batches,
            'batch_details': batch_analysis
        }
        
        logger.info(f"UNIFORMITY AUDIT RESULTS:")
        logger.info(f"  Uniform label batches: {uniform_batches}/10")
        logger.info(f"  Identical prediction batches: {identical_pred_batches}/10")
        
        if uniform_batches > 5:
            logger.error("🚨 CRITICAL: >50% batches have uniform labels - SHUFFLING ISSUE")
        
        if identical_pred_batches > 3:
            logger.error("🚨 CRITICAL: >30% batches have identical predictions - WEIGHT COLLAPSE")
        
        return batch_analysis
    
    def audit_2_target_leakage(self):
        """AUDIT 2: Investigate data leakage in dataset adapter"""
        logger.info("=== AUDIT 2: TARGET LEAKAGE AUDIT ===")
        
        # Read and analyze production_dataset_adapter.py
        adapter_path = "production_dataset_adapter.py"
        
        with open(adapter_path, 'r') as f:
            adapter_code = f.read()
        
        # Check for potential leakage patterns
        leakage_patterns = [
            'magnitude.*normalize',
            'target.*transform',
            'binary.*tensor',
            'future.*data',
            'earthquake.*time'
        ]
        
        leakage_found = []
        for pattern in leakage_patterns:
            if pattern.lower() in adapter_code.lower():
                leakage_found.append(pattern)
        
        # Analyze target preparation
        logger.info("Analyzing target preparation in ProductionDatasetAdapter...")
        
        # Create dataset and inspect target creation
        dataset = ProductionDatasetAdapter('real_earthquake_dataset.h5', split='train')
        
        # Sample first few items to check target consistency
        sample_targets = []
        for i in range(min(100, len(dataset))):
            _, targets = dataset[i]
            sample_targets.append({
                'binary': float(targets['binary']),
                'magnitude_class': int(targets['magnitude_class']),
                'distance': float(targets['distance'])
            })
        
        # Analyze target distribution
        binary_values = [t['binary'] for t in sample_targets]
        mag_values = [t['magnitude_class'] for t in sample_targets]
        
        binary_unique = np.unique(binary_values)
        mag_unique = np.unique(mag_values)
        
        self.audit_results['target_leakage'] = {
            'leakage_patterns_found': leakage_found,
            'binary_target_unique': binary_unique.tolist(),
            'magnitude_target_unique': mag_unique.tolist(),
            'binary_distribution': {
                'mean': float(np.mean(binary_values)),
                'std': float(np.std(binary_values)),
                'all_ones': all(v == 1.0 for v in binary_values)
            },
            'sample_targets': sample_targets[:10]
        }
        
        logger.info(f"Target leakage analysis:")
        logger.info(f"  Binary target unique values: {binary_unique}")
        logger.info(f"  Magnitude target unique values: {mag_unique}")
        logger.info(f"  All binary targets = 1.0: {all(v == 1.0 for v in binary_values)}")
        
        if all(v == 1.0 for v in binary_values):
            logger.error("🚨 CRITICAL: ALL BINARY TARGETS = 1.0 - NO NEGATIVE SAMPLES!")
        
        return leakage_found
    
    def audit_3_numerical_stability(self):
        """AUDIT 3: Check numerical stability issues"""
        logger.info("=== AUDIT 3: NUMERICAL STABILITY AUDIT ===")
        
        # Test different loss functions
        test_logits = torch.tensor([10.0, -10.0, 0.0, 100.0, -100.0])
        test_targets = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0])
        
        # Test BCELoss (current)
        sigmoid_probs = torch.sigmoid(test_logits)
        bce_loss = nn.BCELoss()(sigmoid_probs, test_targets)
        
        # Test BCEWithLogitsLoss (recommended)
        bce_logits_loss = nn.BCEWithLogitsLoss()(test_logits, test_targets)
        
        # Check for numerical issues
        has_nan = torch.isnan(bce_loss) or torch.isnan(bce_logits_loss)
        has_inf = torch.isinf(bce_loss) or torch.isinf(bce_logits_loss)
        
        self.audit_results['numerical_stability'] = {
            'test_logits': test_logits.tolist(),
            'test_targets': test_targets.tolist(),
            'sigmoid_probs': sigmoid_probs.tolist(),
            'bce_loss': float(bce_loss),
            'bce_logits_loss': float(bce_logits_loss),
            'has_nan': has_nan,
            'has_inf': has_inf,
            'stability_issue': has_nan or has_inf
        }
        
        logger.info(f"Numerical stability test:")
        logger.info(f"  BCELoss: {bce_loss:.6f}")
        logger.info(f"  BCEWithLogitsLoss: {bce_logits_loss:.6f}")
        logger.info(f"  Has NaN: {has_nan}")
        logger.info(f"  Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            logger.error("🚨 CRITICAL: Numerical instability detected!")
        
        return bce_loss, bce_logits_loss
    
    def audit_4_gradient_flow(self):
        """AUDIT 4: Check gradient flow issues"""
        logger.info("=== AUDIT 4: GRADIENT FLOW AUDIT ===")
        
        # Create model
        station_coordinates = np.array([
            [-6.2, 106.8], [-7.8, 110.4], [-6.9, 107.6], [-7.3, 112.7],
            [-8.7, 115.2], [-0.9, 100.4], [3.6, 98.7], [-5.1, 119.4]
        ])
        
        model = SpatioTemporalPrecursorModel(
            n_stations=8, n_components=3, station_coordinates=station_coordinates,
            efficientnet_pretrained=True, gnn_hidden_dim=256, gnn_num_layers=3,
            dropout_rate=0.3, device=str(self.device)
        ).to(self.device)
        
        # Create dummy batch
        dummy_input = torch.randn(4, 8, 3, 224, 224).to(self.device)
        dummy_targets = torch.ones(4).to(self.device)
        
        # Forward pass
        model.train()
        outputs = model(dummy_input)
        
        # Calculate loss
        loss = nn.BCELoss()(outputs['binary_probs'].squeeze(), dummy_targets)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        gradient_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                
                gradient_stats[name] = {
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': grad_max,
                    'min': grad_min,
                    'is_zero': abs(grad_mean) < 1e-8,
                    'is_nan': torch.isnan(param.grad).any().item(),
                    'is_inf': torch.isinf(param.grad).any().item()
                }
        
        # Focus on key layers
        key_layers = ['backbone.features.0.0.weight', 'hierarchical_heads.binary_head.0.weight']
        
        self.audit_results['gradient_flow'] = {
            'loss_value': float(loss),
            'total_parameters_with_grad': len(gradient_stats),
            'key_layer_gradients': {k: gradient_stats.get(k, 'Not found') for k in key_layers},
            'gradient_summary': {
                'zero_gradients': sum(1 for g in gradient_stats.values() if g['is_zero']),
                'nan_gradients': sum(1 for g in gradient_stats.values() if g['is_nan']),
                'inf_gradients': sum(1 for g in gradient_stats.values() if g['is_inf'])
            }
        }
        
        logger.info(f"Gradient flow analysis:")
        logger.info(f"  Loss value: {loss:.6f}")
        logger.info(f"  Parameters with gradients: {len(gradient_stats)}")
        logger.info(f"  Zero gradients: {self.audit_results['gradient_flow']['gradient_summary']['zero_gradients']}")
        logger.info(f"  NaN gradients: {self.audit_results['gradient_flow']['gradient_summary']['nan_gradients']}")
        logger.info(f"  Inf gradients: {self.audit_results['gradient_flow']['gradient_summary']['inf_gradients']}")
        
        return gradient_stats
    
    def generate_rca_report(self):
        """Generate Root Cause Analysis Report"""
        logger.info("=== GENERATING ROOT CAUSE ANALYSIS REPORT ===")
        
        # Analyze all audit results
        issues_found = []
        
        # Check uniformity issues
        if self.audit_results.get('uniformity_check', {}).get('uniform_label_batches', 0) > 5:
            issues_found.append("CRITICAL: Uniform label batches detected - shuffling issue")
        
        if self.audit_results.get('uniformity_check', {}).get('identical_prediction_batches', 0) > 3:
            issues_found.append("CRITICAL: Identical predictions detected - weight collapse")
        
        # Check target leakage
        if self.audit_results.get('target_leakage', {}).get('binary_distribution', {}).get('all_ones', False):
            issues_found.append("CRITICAL: All binary targets = 1.0 - no negative samples")
        
        # Check numerical stability
        if self.audit_results.get('numerical_stability', {}).get('stability_issue', False):
            issues_found.append("CRITICAL: Numerical instability (NaN/Inf) detected")
        
        # Check gradient flow
        gradient_summary = self.audit_results.get('gradient_flow', {}).get('gradient_summary', {})
        if gradient_summary.get('zero_gradients', 0) > 10:
            issues_found.append("WARNING: Many zero gradients detected")
        
        if gradient_summary.get('nan_gradients', 0) > 0:
            issues_found.append("CRITICAL: NaN gradients detected")
        
        # Generate RCA
        rca_report = {
            'timestamp': datetime.now().isoformat(),
            'loss_collapse_location': 'Batch 100, Epoch 1',
            'primary_issues': issues_found,
            'audit_results': self.audit_results,
            'recommended_fixes': [
                "Replace nn.BCELoss() with nn.BCEWithLogitsLoss()",
                "Remove nn.Sigmoid() from model output",
                "Reduce learning rate from 0.001 to 1e-4",
                "Add gradient clipping with max_norm=1.0",
                "Fix dataset shuffling and target distribution",
                "Add label smoothing for binary classification"
            ]
        }
        
        # Save report
        with open('loss_collapse_rca_report.json', 'w') as f:
            json.dump(rca_report, f, indent=2, default=str)
        
        logger.info("=== ROOT CAUSE ANALYSIS COMPLETE ===")
        logger.info(f"Issues found: {len(issues_found)}")
        for issue in issues_found:
            logger.error(f"  🚨 {issue}")
        
        logger.info("RCA Report saved: loss_collapse_rca_report.json")
        
        return rca_report
    
    def run_full_audit(self):
        """Run complete forensic audit"""
        logger.info("=== STARTING FULL FORENSIC AUDIT ===")
        
        try:
            # Run all audits
            self.audit_1_output_label_uniformity()
            self.audit_2_target_leakage()
            self.audit_3_numerical_stability()
            self.audit_4_gradient_flow()
            
            # Generate RCA report
            rca_report = self.generate_rca_report()
            
            logger.info("=== FORENSIC AUDIT COMPLETED SUCCESSFULLY ===")
            return rca_report
            
        except Exception as e:
            logger.error(f"Forensic audit failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main forensic audit function"""
    auditor = ForensicLossAuditor()
    rca_report = auditor.run_full_audit()
    
    if rca_report:
        print("\n" + "="*80)
        print("🔍 FORENSIC AUDIT COMPLETE")
        print("="*80)
        print(f"Issues found: {len(rca_report['primary_issues'])}")
        for issue in rca_report['primary_issues']:
            print(f"  🚨 {issue}")
        print("\nRecommended fixes:")
        for fix in rca_report['recommended_fixes']:
            print(f"  ✅ {fix}")
        print("="*80)

if __name__ == "__main__":
    main()