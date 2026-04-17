#!/usr/bin/env python3
"""
Robust Training Launcher dengan Resume Capability
Mengatasi masalah loss calculation dan memastikan training berjalan dengan benar

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
import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import json
import argparse
import matplotlib.pyplot as plt
import psutil
import time
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
        logging.FileHandler('robust_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustTrainer:
    """
    Robust trainer dengan error handling dan resume capability
    """
    
    def __init__(self, dataset_path: str, resume_from: str = None):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_from = resume_from
        self.start_time = datetime.now()
        
        # Load station coordinates
        self.station_coordinates = self._load_station_coordinates()
        
        logger.info("=== ROBUST TRAINING LAUNCHER INITIALIZED ===")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Resume from: {self.resume_from}")
        
    def _load_station_coordinates(self):
        """Load station coordinates"""
        try:
            df = pd.read_csv('../awal/lokasi_stasiun.csv', sep=';')
            coords = []
            for _, row in df.iterrows():
                lat = float(str(row['Latitude']).strip())
                lon = float(str(row['Longitude']).strip())
                coords.append([lat, lon])
            return np.array(coords[:8])  # Only first 8 stations
        except Exception as e:
            logger.warning(f"Could not load station coordinates: {e}")
            # Default coordinates for 8 stations
            return np.array([
                [-6.2, 106.8],  # Jakarta area
                [-7.8, 110.4],  # Yogyakarta
                [-6.9, 107.6],  # Bandung
                [-7.3, 112.7],  # Surabaya
                [-8.7, 115.2],  # Denpasar
                [-0.9, 100.4],  # Padang
                [3.6, 98.7],    # Medan
                [-5.1, 119.4]   # Makassar
            ])
    
    def create_model(self):
        """Create model with proper verification"""
        logger.info("=== CREATING MODEL ===")
        
        model = SpatioTemporalPrecursorModel(
            n_stations=8,
            n_components=3,
            station_coordinates=self.station_coordinates,
            efficientnet_pretrained=True,
            gnn_hidden_dim=256,
            gnn_num_layers=3,
            dropout_rate=0.3,
            device=str(self.device)
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        if total_params < 5_000_000:
            raise ValueError(f"Model has only {total_params:,} parameters, expected >5M")
        
        logger.info("Model parameter verification PASSED")
        return model
    
    def create_datasets(self, batch_size=16):
        """Create datasets with proper error handling"""
        logger.info("=== CREATING DATASETS ===")
        
        # Create full dataset
        full_dataset = ProductionDatasetAdapter(self.dataset_path, split='train')
        
        # Manual train/val split (80/20)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid issues
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Val dataset: {len(val_dataset)} samples")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Batch size: {batch_size}")
        
        return train_loader, val_loader
    
    def test_model_forward(self, model, train_loader):
        """Test model forward pass"""
        logger.info("=== TESTING MODEL FORWARD PASS ===")
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (tensors, targets) in enumerate(train_loader):
                if batch_idx >= 1:  # Only test first batch
                    break
                
                logger.info(f"Input tensor shape: {tensors.shape}")
                logger.info(f"Target keys: {list(targets.keys())}")
                
                # Move to device
                tensors = tensors.to(self.device)
                
                # Forward pass
                outputs = model(tensors)
                
                logger.info("Model outputs:")
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape}")
                    else:
                        logger.info(f"  {key}: {value}")
                
                # Test loss calculation
                loss = self._calculate_loss(outputs, targets, stage=1)
                logger.info(f"Test loss: {loss.item():.6f}")
                
                if loss.item() == 0.0:
                    logger.error("Loss is zero - there's a problem with loss calculation")
                    return False
                else:
                    logger.info("Forward pass test PASSED")
                    return True
        
        return False
    
    def _calculate_loss(self, outputs, targets, stage=1):
        """Calculate loss with proper error handling"""
        total_loss = 0.0
        loss_components = {}
        
        # Binary classification loss
        if 'binary_probs' in outputs and 'binary' in targets:
            binary_target = targets['binary'].to(self.device).float()
            binary_pred = outputs['binary_probs'].squeeze()
            
            # Ensure same shape
            if binary_pred.dim() == 0:
                binary_pred = binary_pred.unsqueeze(0)
            if binary_target.dim() == 0:
                binary_target = binary_target.unsqueeze(0)
            
            binary_loss = nn.BCELoss()(binary_pred, binary_target)
            total_loss += binary_loss
            loss_components['binary'] = binary_loss.item()
        
        # Magnitude classification loss (Stage 2+)
        if stage >= 2 and 'magnitude_logits' in outputs and 'magnitude_class' in targets:
            mag_target = targets['magnitude_class'].to(self.device).long()
            mag_pred = outputs['magnitude_logits']
            
            mag_loss = nn.CrossEntropyLoss()(mag_pred, mag_target)
            total_loss += mag_loss
            loss_components['magnitude'] = mag_loss.item()
        
        # Distance loss (Stage 3)
        if stage >= 3 and 'distance' in outputs and 'distance' in targets:
            dist_target = targets['distance'].to(self.device).float()
            dist_pred = outputs['distance'].squeeze()
            
            if dist_pred.dim() == 0:
                dist_pred = dist_pred.unsqueeze(0)
            if dist_target.dim() == 0:
                dist_target = dist_target.unsqueeze(0)
            
            dist_loss = nn.MSELoss()(dist_pred, dist_target)
            total_loss += dist_loss * 0.001  # Scale down
            loss_components['distance'] = dist_loss.item()
        
        # Log loss components
        if len(loss_components) > 0:
            comp_str = ", ".join([f"{k}={v:.4f}" for k, v in loss_components.items()])
            logger.debug(f"Loss components: {comp_str}")
        
        return total_loss
    
    def train_single_epoch(self, model, train_loader, optimizer, stage, epoch):
        """Train single epoch with detailed logging"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (tensors, targets) in enumerate(train_loader):
            # Move to device
            tensors = tensors.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(tensors)
            
            # Calculate loss
            loss = self._calculate_loss(outputs, targets, stage)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress logging
            if batch_idx % 50 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
            
            # Memory cleanup
            del tensors, targets, outputs, loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate_single_epoch(self, model, val_loader, stage):
        """Validate single epoch"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (tensors, targets) in enumerate(val_loader):
                tensors = tensors.to(self.device)
                outputs = model(tensors)
                
                loss = self._calculate_loss(outputs, targets, stage)
                total_loss += loss.item()
                num_batches += 1
                
                del tensors, targets, outputs, loss
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, stage, loss, filepath):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, model, optimizer, scheduler, filepath):
        """Load training checkpoint"""
        if not os.path.exists(filepath):
            logger.info(f"No checkpoint found at {filepath}")
            return 0, 1
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        stage = checkpoint['stage']
        
        logger.info(f"Checkpoint loaded: Stage {stage}, Epoch {epoch}")
        return epoch, stage
    
    def run_training(self):
        """Run complete training with error handling"""
        logger.info("=== STARTING ROBUST TRAINING ===")
        
        try:
            # Create model
            model = self.create_model()
            
            # Create datasets
            train_loader, val_loader = self.create_datasets(batch_size=16)
            
            # Test model forward pass
            if not self.test_model_forward(model, train_loader):
                logger.error("Model forward pass test failed")
                return False
            
            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )
            
            # Resume from checkpoint if specified
            start_epoch = 0
            start_stage = 1
            if self.resume_from:
                start_epoch, start_stage = self.load_checkpoint(
                    model, optimizer, scheduler, self.resume_from
                )
            
            # Training stages
            stages = [
                {'stage': 1, 'name': 'Binary Classification', 'epochs': 25},
                {'stage': 2, 'name': 'Magnitude Classification', 'epochs': 25},
                {'stage': 3, 'name': 'Localization', 'epochs': 25}
            ]
            
            training_history = []
            
            # Run training stages
            for stage_info in stages:
                stage = stage_info['stage']
                stage_name = stage_info['name']
                max_epochs = stage_info['epochs']
                
                if stage < start_stage:
                    continue
                
                logger.info(f"=== STAGE {stage}: {stage_name} ===")
                
                # Set model training stage if supported
                if hasattr(model, 'set_training_stage'):
                    model.set_training_stage(stage)
                
                best_val_loss = float('inf')
                
                for epoch in range(max_epochs):
                    if stage == start_stage and epoch < start_epoch:
                        continue
                    
                    epoch_start = datetime.now()
                    logger.info(f"Stage {stage}, Epoch {epoch+1}/{max_epochs}")
                    
                    # Training
                    train_loss = self.train_single_epoch(model, train_loader, optimizer, stage, epoch+1)
                    
                    # Validation
                    val_loss = self.validate_single_epoch(model, val_loader, stage)
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpoint_path = f'best_model_stage_{stage}.pth'
                        self.save_checkpoint(model, optimizer, scheduler, epoch+1, stage, val_loss, checkpoint_path)
                    
                    # Regular checkpoint
                    if (epoch + 1) % 5 == 0:
                        checkpoint_path = f'checkpoint_stage_{stage}_epoch_{epoch+1}.pth'
                        self.save_checkpoint(model, optimizer, scheduler, epoch+1, stage, val_loss, checkpoint_path)
                    
                    epoch_duration = datetime.now() - epoch_start
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    logger.info(f"Stage {stage}, Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.2e}, Duration={epoch_duration}")
                    
                    # Save progress
                    training_history.append({
                        'stage': stage,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                        'duration': str(epoch_duration)
                    })
                    
                    # Save history periodically
                    with open('robust_training_history.json', 'w') as f:
                        json.dump(training_history, f, indent=2, default=str)
                
                logger.info(f"Stage {stage} completed. Best val loss: {best_val_loss:.6f}")
            
            # Save final model
            final_checkpoint = 'final_robust_model.pth'
            self.save_checkpoint(model, optimizer, scheduler, max_epochs, 3, best_val_loss, final_checkpoint)
            
            total_duration = datetime.now() - self.start_time
            logger.info(f"=== TRAINING COMPLETED SUCCESSFULLY ===")
            logger.info(f"Total duration: {total_duration}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Robust Training Launcher')
    parser.add_argument('--dataset', type=str, default='real_earthquake_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    logger.info("=== ROBUST TRAINING LAUNCHER STARTED ===")
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return
    
    trainer = RobustTrainer(args.dataset, args.resume)
    success = trainer.run_training()
    
    if success:
        logger.info("Training completed successfully!")
    else:
        logger.error("Training failed!")

if __name__ == "__main__":
    main()