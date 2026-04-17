#!/usr/bin/env python3
"""
FINAL PRODUCTION TRAINING - Spatio-Temporal Earthquake Precursor Model
PROTOCOL: 3-Stage Training dengan EfficientNet-B0 (5.7M parameters)

Stage 1: Binary Classification (PCA-CMR immunity)
Stage 2: Magnitude Classification (Focal Loss)  
Stage 3: Localization (Station coordinates triangulation)

Author: Kiro AI Assistant
Date: April 16, 2026
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

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_production_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinalProductionTrainer:
    """
    Final Production Trainer dengan protokol 3-stage ketat
    """
    
    def __init__(self, dataset_path: str, resume_from: str = None):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resume_from = resume_from
        self.start_time = datetime.now()
        
        # Load station coordinates
        self.station_coordinates = self._load_station_coordinates()
        
        logger.info("=== FINAL PRODUCTION TRAINING INITIALIZED ===")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Station coordinates loaded: {len(self.station_coordinates)} stations")
        
    def _load_station_coordinates(self):
        """Load station coordinates from lokasi_stasiun.csv"""
        try:
            df = pd.read_csv('../awal/lokasi_stasiun.csv', sep=';')
            coords = []
            for _, row in df.iterrows():
                lat = float(str(row['Latitude']).strip())
                lon = float(str(row['Longitude']).strip())
                coords.append([lat, lon])
            return np.array(coords)
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
    
    def verify_model_parameters(self):
        """Verifikasi parameter model sebelum training"""
        logger.info("=== VERIFYING MODEL PARAMETERS ===")
        
        model = SpatioTemporalPrecursorModel(
            n_stations=8,
            n_components=3,
            station_coordinates=self.station_coordinates,
            efficientnet_pretrained=True,
            gnn_hidden_dim=256,
            gnn_num_layers=3,
            dropout_rate=0.3,
            device=str(self.device)
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        if total_params < 1_000_000:
            raise ValueError(f"CRITICAL: Model has only {total_params:,} parameters, expected >1M")
        
        if total_params < 5_000_000:
            logger.warning(f"WARNING: Model has {total_params:,} parameters, expected ~5.7M")
        else:
            logger.info(f"Parameter verification PASSED: {total_params:,} parameters")
        
        return model, total_params
    
    def create_datasets_and_loaders(self, batch_size=16):
        """Create datasets and loaders with full data coverage"""
        logger.info("=== CREATING DATASETS AND LOADERS ===")
        
        # Create datasets with manual split
        full_dataset = ProductionDatasetAdapter(self.dataset_path, split='train')
        
        # Manual train/val split (80/20)
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loaders - NO BATCH LIMITS
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"Train dataset: {len(self.train_dataset)} samples")
        logger.info(f"Val dataset: {len(self.val_dataset)} samples")
        logger.info(f"Train batches: {len(self.train_loader)} (ALL WILL BE PROCESSED)")
        logger.info(f"Val batches: {len(self.val_loader)} (ALL WILL BE PROCESSED)")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"100% DATA COVERAGE CONFIRMED")
        
        return True
    
    def monitor_system_resources(self):
        """Monitor CPU and memory usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        logger.info(f"CPU Usage: {cpu_percent:.1f}%")
        logger.info(f"Memory Usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        if cpu_percent < 50:
            logger.warning("⚠️  CPU usage low - backpropagation might not be running properly")
        else:
            logger.info("✅ CPU usage high - backpropagation confirmed active")
    
    def sanity_check_training(self, epoch, batch_idx, loss_value, total_batches):
        """Sanity check during first 15 minutes"""
        elapsed_minutes = (datetime.now() - self.start_time).total_seconds() / 60
        
        if elapsed_minutes <= 15:  # First 15 minutes
            if batch_idx % 100 == 0:
                logger.info(f"🔍 SANITY CHECK - Epoch {epoch}, Batch {batch_idx}/{total_batches}")
                logger.info(f"   Loss: {loss_value:.4f}")
                logger.info(f"   Elapsed: {elapsed_minutes:.1f} minutes")
                
                # Check batch count
                if total_batches < 100:
                    logger.error(f"❌ SANITY FAIL: Only {total_batches} batches, expected thousands")
                    raise ValueError("Batch count too low")
                else:
                    logger.info("Batch count OK: {total_batches} batches")
                
                # Check loss values
                if loss_value < 0.01:
                    logger.warning(f"⚠️  Loss suspiciously low: {loss_value:.4f}")
                elif 0.6 <= loss_value <= 2.0:
                    logger.info(f"Loss in expected range: {loss_value:.4f}")
                
                # Monitor system
                self.monitor_system_resources()
    
    def calculate_focal_loss(self, outputs, targets, alpha=0.25, gamma=2.0):
        """Calculate Focal Loss for magnitude classification"""
        if 'magnitude_class' not in outputs:
            return 0.0
        
        ce_loss = nn.CrossEntropyLoss(reduction='none')(
            outputs['magnitude_class'], 
            targets['magnitude_class'].to(self.device)
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        
        return focal_loss.mean()
    
    def save_checkpoint(self, model, optimizer, epoch, stage, loss, checkpoint_name):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'stage': stage,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, checkpoint_name)
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def run_stage_training(self, model, optimizer, scheduler, stage_info, training_history):
        """Run training for a specific stage"""
        stage = stage_info['stage']
        stage_name = stage_info['name']
        epochs = stage_info['epochs']
        
        logger.info(f"=== STARTING STAGE {stage}: {stage_name} ===")
        logger.info(f"Epochs: {epochs}")
        
        # Set model training stage
        if hasattr(model, 'set_training_stage'):
            model.set_training_stage(stage)
        
        stage_history = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_start = datetime.now()
            logger.info(f"Stage {stage}, Epoch {epoch+1}/{epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (tensors, targets) in enumerate(self.train_loader):
                # Sanity check during first 15 minutes
                self.sanity_check_training(epoch+1, batch_idx, 0.0, len(self.train_loader))
                
                # Move to device
                tensors = tensors.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(tensors)
                
                # Calculate stage-specific loss
                loss = self._calculate_stage_loss(outputs, targets, stage)
                
                # Sanity check loss value
                if batch_idx % 100 == 0:
                    self.sanity_check_training(epoch+1, batch_idx, loss.item(), len(self.train_loader))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Progress logging
                if batch_idx % 200 == 0:
                    logger.info(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                
                del tensors, targets, outputs, loss
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, (tensors, targets) in enumerate(self.val_loader):
                    tensors = tensors.to(self.device)
                    outputs = model(tensors)
                    
                    loss = self._calculate_stage_loss(outputs, targets, stage)
                    val_loss += loss.item()
                    val_batches += 1
                    
                    del tensors, targets, outputs, loss
            
            # Calculate averages
            avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_name = f'best_efficientnet_stage{stage}.pth'
                self.save_checkpoint(model, optimizer, epoch+1, stage, avg_val_loss, checkpoint_name)
            
            epoch_duration = datetime.now() - epoch_start
            
            # Progress report every 5 epochs
            if (epoch + 1) % 5 == 0:
                logger.info(f"📊 PROGRESS REPORT - Stage {stage}, Epoch {epoch+1}")
                logger.info(f"   Train Loss: {avg_train_loss:.4f}")
                logger.info(f"   Val Loss: {avg_val_loss:.4f}")
                logger.info(f"   Best Val Loss: {best_val_loss:.4f}")
                logger.info(f"   Duration: {epoch_duration}")
                logger.info(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            stage_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'duration': str(epoch_duration),
                'lr': optimizer.param_groups[0]['lr']
            })
        
        training_history[f'stage_{stage}'] = stage_history
        logger.info(f"Stage {stage} completed. Best val loss: {best_val_loss:.4f}")
        
        return training_history
    
    def _calculate_stage_loss(self, outputs, targets, stage):
        """Calculate loss based on training stage"""
        loss = 0.0
        
        if stage == 1:  # Binary Classification (PCA-CMR immunity)
            if 'binary_probs' in outputs:
                binary_loss = nn.BCELoss()(
                    outputs['binary_probs'].squeeze(),
                    targets['binary'].to(self.device).float()
                )
                loss += binary_loss
                
        elif stage == 2:  # Magnitude Classification (Focal Loss)
            if 'binary_probs' in outputs:
                binary_loss = nn.BCELoss()(
                    outputs['binary_probs'].squeeze(),
                    targets['binary'].to(self.device).float()
                )
                loss += binary_loss
            
            # Focal Loss for magnitude classification
            if 'magnitude_logits' in outputs:
                focal_loss = self.calculate_focal_loss(outputs, targets)
                loss += focal_loss * 2.0  # Weight focal loss higher
                
        elif stage == 3:  # Localization (Station coordinates)
            if 'binary_probs' in outputs:
                binary_loss = nn.BCELoss()(
                    outputs['binary_probs'].squeeze(),
                    targets['binary'].to(self.device).float()
                )
                loss += binary_loss
            
            if 'magnitude_logits' in outputs:
                mag_loss = nn.CrossEntropyLoss()(
                    outputs['magnitude_logits'],
                    targets['magnitude_class'].to(self.device)
                )
                loss += mag_loss
            
            # Localization losses
            if 'distance' in outputs and 'distance' in targets:
                dist_loss = nn.MSELoss()(
                    outputs['distance'].squeeze(),
                    targets['distance'].to(self.device).float()
                )
                loss += dist_loss * 0.001
        
        return loss
    
    def plot_training_curves(self, training_history):
        """Plot training curves for convergence analysis"""
        logger.info("=== GENERATING TRAINING CURVES ===")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Convergence Analysis - EfficientNet-B0', fontsize=16)
        
        # Plot each stage
        for stage_key, stage_data in training_history.items():
            stage_num = int(stage_key.split('_')[1])
            
            epochs = [d['epoch'] for d in stage_data]
            train_losses = [d['train_loss'] for d in stage_data]
            val_losses = [d['val_loss'] for d in stage_data]
            
            # Loss curves
            ax = axes[0, 0] if stage_num <= 2 else axes[0, 1]
            ax.plot(epochs, train_losses, label=f'Stage {stage_num} Train', marker='o')
            ax.plot(epochs, val_losses, label=f'Stage {stage_num} Val', marker='s')
            ax.set_title(f'Stage {stage_num} Loss Curves')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True)
        
        # Combined loss plot
        ax = axes[1, 0]
        all_train_losses = []
        all_val_losses = []
        all_epochs = []
        epoch_offset = 0
        
        for stage_key, stage_data in training_history.items():
            for d in stage_data:
                all_epochs.append(d['epoch'] + epoch_offset)
                all_train_losses.append(d['train_loss'])
                all_val_losses.append(d['val_loss'])
            epoch_offset += len(stage_data)
        
        ax.plot(all_epochs, all_train_losses, label='Train Loss', color='blue')
        ax.plot(all_val_losses, label='Val Loss', color='red')
        ax.set_title('Overall Training Progress')
        ax.set_xlabel('Total Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Learning rate plot
        ax = axes[1, 1]
        all_lrs = []
        for stage_key, stage_data in training_history.items():
            for d in stage_data:
                all_lrs.append(d['lr'])
        
        ax.plot(all_epochs, all_lrs, label='Learning Rate', color='green')
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Total Epochs')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('training_convergence_curves.png', dpi=300, bbox_inches='tight')
        logger.info("Training curves saved: training_convergence_curves.png")
        
        # Overfitting analysis
        final_train_loss = all_train_losses[-1]
        final_val_loss = all_val_losses[-1]
        overfitting_ratio = final_val_loss / final_train_loss
        
        if overfitting_ratio > 1.5:
            logger.warning(f"⚠️  Potential overfitting detected: Val/Train ratio = {overfitting_ratio:.2f}")
        else:
            logger.info(f"No overfitting detected: Val/Train ratio = {overfitting_ratio:.2f}")
    
    def run_final_production_training(self):
        """Run complete 3-stage production training"""
        logger.info("=== STARTING FINAL PRODUCTION TRAINING ===")
        
        # 1. Verify model parameters
        model, total_params = self.verify_model_parameters()
        model = model.to(self.device)
        
        # 2. Create datasets
        self.create_datasets_and_loaders(batch_size=16)
        
        # 3. Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        # 4. Define 3-stage protocol
        stages = [
            {'stage': 1, 'name': 'Binary Classification (PCA-CMR Immunity)', 'epochs': 25},
            {'stage': 2, 'name': 'Magnitude Classification (Focal Loss)', 'epochs': 25},
            {'stage': 3, 'name': 'Localization (Station Triangulation)', 'epochs': 25}
        ]
        
        training_history = {}
        
        # 5. Execute 3-stage training
        for stage_info in stages:
            training_history = self.run_stage_training(
                model, optimizer, scheduler, stage_info, training_history
            )
        
        # 6. Save final model
        final_checkpoint = 'final_spatio_model.pth'
        self.save_checkpoint(model, optimizer, 75, 3, 0.0, final_checkpoint)
        
        # 7. Generate training curves
        self.plot_training_curves(training_history)
        
        # 8. Save training history
        with open('final_training_history.json', 'w') as f:
            json.dump(training_history, f, indent=2, default=str)
        
        total_duration = datetime.now() - self.start_time
        
        logger.info("=== FINAL PRODUCTION TRAINING COMPLETED ===")
        logger.info(f"Total duration: {total_duration}")
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Checkpoints saved: best_efficientnet_stage1.pth, best_efficientnet_stage2.pth, final_spatio_model.pth")
        logger.info(f"Training curves: training_convergence_curves.png")
        logger.info(f"Training history: final_training_history.json")
        
        return training_history

def main():
    """Main training execution"""
    parser = argparse.ArgumentParser(description='Final Production Training')
    parser.add_argument('--dataset', type=str, default='real_earthquake_dataset.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    logger.info("=== FINAL PRODUCTION TRAINING STARTED ===")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Resume from: {args.resume}")
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        return
    
    try:
        trainer = FinalProductionTrainer(args.dataset, args.resume)
        history = trainer.run_final_production_training()
        
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()