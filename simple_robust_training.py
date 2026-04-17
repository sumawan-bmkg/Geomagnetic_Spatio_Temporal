#!/usr/bin/env python3
"""
Simple Robust Training - Minimal script untuk training yang stabil
Fokus pada stabilitas dan progress monitoring

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
        logging.FileHandler('simple_robust_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Simple robust training"""
    logger.info("=== SIMPLE ROBUST TRAINING STARTED ===")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Station coordinates (8 stations)
    station_coordinates = np.array([
        [-6.2, 106.8],  # Jakarta area
        [-7.8, 110.4],  # Yogyakarta
        [-6.9, 107.6],  # Bandung
        [-7.3, 112.7],  # Surabaya
        [-8.7, 115.2],  # Denpasar
        [-0.9, 100.4],  # Padang
        [3.6, 98.7],    # Medan
        [-5.1, 119.4]   # Makassar
    ])
    
    try:
        # Create model
        logger.info("Creating model...")
        model = SpatioTemporalPrecursorModel(
            n_stations=8,
            n_components=3,
            station_coordinates=station_coordinates,
            efficientnet_pretrained=True,
            gnn_hidden_dim=256,
            gnn_num_layers=3,
            dropout_rate=0.3,
            device=str(device)
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")
        
        if total_params < 5_000_000:
            raise ValueError(f"Model has only {total_params:,} parameters, expected >5M")
        
        # Create dataset
        logger.info("Creating dataset...")
        full_dataset = ProductionDatasetAdapter('real_earthquake_dataset.h5', split='train')
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (tensors, targets) in enumerate(train_loader):
                if batch_idx >= 1:
                    break
                
                tensors = tensors.to(device)
                outputs = model(tensors)
                
                # Calculate test loss
                binary_loss = nn.BCELoss()(
                    outputs['binary_probs'].squeeze(),
                    targets['binary'].to(device).float()
                )
                
                logger.info(f"Test loss: {binary_loss.item():.6f}")
                
                if binary_loss.item() == 0.0:
                    logger.error("Loss is zero - problem with loss calculation")
                    return
                else:
                    logger.info("Forward pass test PASSED")
                    break
        
        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        
        # Training loop - Stage 1 only (Binary Classification)
        logger.info("=== STARTING STAGE 1: BINARY CLASSIFICATION ===")
        
        # Set training stage
        if hasattr(model, 'set_training_stage'):
            model.set_training_stage(1)
        
        training_history = []
        best_val_loss = float('inf')
        
        for epoch in range(5):  # Start with just 5 epochs to test
            epoch_start = datetime.now()
            logger.info(f"Epoch {epoch+1}/5")
            
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (tensors, targets) in enumerate(train_loader):
                tensors = tensors.to(device)
                
                optimizer.zero_grad()
                outputs = model(tensors)
                
                # Binary classification loss only
                loss = nn.BCELoss()(
                    outputs['binary_probs'].squeeze(),
                    targets['binary'].to(device).float()
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Progress logging
                if batch_idx % 100 == 0:
                    logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
                
                del tensors, targets, outputs, loss
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, (tensors, targets) in enumerate(val_loader):
                    tensors = tensors.to(device)
                    outputs = model(tensors)
                    
                    loss = nn.BCELoss()(
                        outputs['binary_probs'].squeeze(),
                        targets['binary'].to(device).float()
                    )
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    del tensors, targets, outputs, loss
            
            # Calculate averages
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_simple_model.pth')
                logger.info(f"New best model saved: {avg_val_loss:.6f}")
            
            epoch_duration = datetime.now() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, LR={current_lr:.2e}, Duration={epoch_duration}")
            
            # Save progress
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'learning_rate': current_lr,
                'duration': str(epoch_duration)
            })
            
            # Save history
            with open('simple_training_history.json', 'w') as f:
                json.dump(training_history, f, indent=2, default=str)
        
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        logger.info("Model saved: best_simple_model.pth")
        logger.info("History saved: simple_training_history.json")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()