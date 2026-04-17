#!/usr/bin/env python3
"""
Verify Loss Recovery - Quick test to confirm loss collapse is fixed
"""
import sys
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.spatio_temporal_model import SpatioTemporalPrecursorModel
from corrected_dataset_adapter import CorrectedDatasetAdapter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_recovery():
    """Verify that loss collapse has been fixed"""
    logger.info("=== VERIFYING LOSS COLLAPSE RECOVERY ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Station coordinates
    station_coordinates = np.array([
        [-6.2, 106.8], [-7.8, 110.4], [-6.9, 107.6], [-7.3, 112.7],
        [-8.7, 115.2], [-0.9, 100.4], [3.6, 98.7], [-5.1, 119.4]
    ])
    
    try:
        # Create model
        model = SpatioTemporalPrecursorModel(
            n_stations=8, n_components=3, station_coordinates=station_coordinates,
            efficientnet_pretrained=True, gnn_hidden_dim=256, gnn_num_layers=3,
            dropout_rate=0.3, device=str(device)
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Model parameters: {total_params:,}")
        
        # Create corrected dataset
        dataset = CorrectedDatasetAdapter('real_earthquake_dataset.h5', split='train', negative_ratio=0.5)
        stats = dataset.get_target_statistics()
        
        logger.info(f"✅ Dataset balance: {stats['binary_stats']['positive_ratio']:.3f}")
        logger.info(f"✅ Negative samples: {stats['binary_stats']['negative_count']}")
        logger.info(f"✅ Positive samples: {stats['binary_stats']['positive_count']}")
        
        # Create loader
        loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
        
        # Test 10 batches with BCEWithLogitsLoss
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        loss_values = []
        target_balances = []
        
        with torch.no_grad():
            for batch_idx, (tensors, targets) in enumerate(loader):
                if batch_idx >= 10:
                    break
                
                tensors = tensors.to(device)
                outputs = model(tensors)
                
                # Calculate loss with BCEWithLogitsLoss
                loss = criterion(
                    outputs['binary_logits'].squeeze(),
                    targets['binary'].to(device).float()
                )
                
                binary_targets = targets['binary'].numpy()
                target_balance = np.mean(binary_targets)
                
                loss_values.append(loss.item())
                target_balances.append(target_balance)
                
                logger.info(f"Batch {batch_idx}: Loss={loss.item():.6f}, Balance={target_balance:.3f}")
        
        # Analysis
        avg_loss = np.mean(loss_values)
        avg_balance = np.mean(target_balances)
        min_loss = np.min(loss_values)
        max_loss = np.max(loss_values)
        
        logger.info("=== RECOVERY VERIFICATION RESULTS ===")
        logger.info(f"Average loss: {avg_loss:.6f}")
        logger.info(f"Loss range: {min_loss:.6f} - {max_loss:.6f}")
        logger.info(f"Average target balance: {avg_balance:.3f}")
        
        # Success criteria
        success = True
        
        if min_loss < 1e-6:
            logger.error("❌ FAILURE: Loss collapse still detected")
            success = False
        else:
            logger.info("✅ SUCCESS: No loss collapse detected")
        
        if avg_loss > 0.1:
            logger.info("✅ SUCCESS: Loss values are healthy (>0.1)")
        else:
            logger.warning(f"⚠️  WARNING: Loss values are low ({avg_loss:.6f})")
        
        if 0.3 <= avg_balance <= 0.7:
            logger.info("✅ SUCCESS: Target balance is healthy")
        else:
            logger.error(f"❌ FAILURE: Target balance is poor ({avg_balance:.3f})")
            success = False
        
        if success:
            logger.info("🎉 RECOVERY COMPLETE: All fixes are working!")
        else:
            logger.error("💥 RECOVERY FAILED: Issues still present")
        
        return success
        
    except Exception as e:
        logger.error(f"Recovery verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_recovery()