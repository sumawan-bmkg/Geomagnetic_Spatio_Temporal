#!/usr/bin/env python3
"""
TUNED PRODUCTION TRAINING - Spatio-Temporal Earthquake Precursor Model
Protocol: Hybrid Ensemble + Attention Refinement + Hard Negative Mining
Features: Cosine Annealing with Warm Restarts, Kp/Dst Integration, SE Blocks

Author: Antigravity AI
Date: April 17, 2026
"""
import sys
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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
        logging.FileHandler('tuned_production_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TunedProductionTrainer:
    def __init__(self, dataset_path: str, checkpoint_dir: str = 'outputs/checkpoints'):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        
        # Load station coordinates
        self.station_coordinates = self._load_station_coordinates()
        
        logger.info("=== TUNED PRODUCTION TRAINING INITIALIZED ===")
        logger.info(f"Device: {self.device}")
        
    def _load_station_coordinates(self):
        try:
            df = pd.read_csv('../awal/lokasi_stasiun.csv', sep=';')
            coords = [[float(str(r['Latitude']).strip()), float(str(r['Longitude']).strip())] for _, r in df.iterrows()]
            return np.array(coords)
        except:
            return np.array([[-6.2, 106.8], [-7.8, 110.4], [-6.9, 107.6], [-7.3, 112.7], [-8.7, 115.2], [-0.9, 100.4], [3.6, 98.7], [-5.1, 119.4]])

    def create_model(self):
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
        return model.to(self.device)

    def run_training_cycle(self, model, train_loader, val_loader, stage, epochs, lr=1e-3, use_cosine=True):
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        if use_cosine:
            # Cosine Annealing with Warm Restarts
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_loss = float('inf')
        history = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_idx, (tensors, targets) in enumerate(train_loader):
                tensors = tensors.to(self.device)
                geo_features = targets['geophysical'].to(self.device)
                binary_label = targets['binary'].to(self.device).float()
                mag_label = targets['magnitude_class'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(tensors, geophysical_features=geo_features, training_stage=stage)
                
                # Composite Loss
                loss = 0.0
                if stage >= 1:
                    loss += nn.BCEWithLogitsLoss()(outputs['binary_logits'].squeeze(), binary_label)
                if stage >= 2:
                    # Focal Loss for magnitude (simplified)
                    mag_loss = nn.CrossEntropyLoss()(outputs['magnitude_logits'], mag_label)
                    loss += mag_loss * 2.0
                if stage >= 3:
                    dist_loss = nn.MSELoss()(outputs['distance'].squeeze(), targets['distance'].to(self.device).float())
                    loss += dist_loss * 0.01

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if use_cosine:
                    scheduler.step(epoch + batch_idx / len(train_loader))
                
                train_loss += loss.item()
                if batch_idx % 100 == 0:
                    logger.info(f"Stage {stage} Ep {epoch+1} Batch {batch_idx}: Loss {loss.item():.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for tensors, targets in val_loader:
                    tensors = tensors.to(self.device)
                    geo_features = targets['geophysical'].to(self.device)
                    outputs = model(tensors, geophysical_features=geo_features, training_stage=stage)
                    loss = nn.BCEWithLogitsLoss()(outputs['binary_logits'].squeeze(), targets['binary'].to(self.device).float())
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            if not use_cosine:
                scheduler.step(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1} Summary: Val Loss {avg_val_loss:.4f}, LR {optimizer.param_groups[0]['lr']:.6f}")
            history.append({'epoch': epoch+1, 'val_loss': avg_val_loss, 'lr': optimizer.param_groups[0]['lr']})
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.checkpoint_dir / f'best_model_stage{stage}.pth')

        return history

    def identify_hard_negatives(self, model, dataset, threshold=0.7):
        """Find 'Normal' samples misclassified as 'Precursor'"""
        logger.info("Identifying Hard Negatives (False Positives)...")
        model.eval()
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        hard_indices = []
        
        with torch.no_grad():
            for i, (tensors, targets) in enumerate(loader):
                tensors = tensors.to(self.device)
                geo_features = targets['geophysical'].to(self.device)
                binary_labels = targets['binary']
                outputs = model(tensors, geophysical_features=geo_features, training_stage=1)
                probs = outputs['binary_probs'].squeeze().cpu().numpy()
                
                for j, (prob, label) in enumerate(zip(probs, binary_labels)):
                    if label == 0 and prob > threshold:  # Normal sample with high precursor prob
                        hard_indices.append(i * 32 + j)
        
        logger.info(f"Found {len(hard_indices)} Hard Negatives.")
        return hard_indices

    def execute(self, epochs=15, dry_run=False):
        # 1. Load Datasets
        train_dataset = ProductionDatasetAdapter(self.dataset_path, split='train')
        val_dataset = ProductionDatasetAdapter(self.dataset_path, split='val')
        
        if dry_run:
            logger.info("DRY RUN: Loading only a subset of data")
            train_dataset.tensor_indices = train_dataset.tensor_indices[:32]
            val_dataset.tensor_indices = val_dataset.tensor_indices[:16]
        
        train_loader = DataLoader(train_dataset, batch_size=8 if dry_run else 16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8 if dry_run else 16)
        
        model = self.create_model()
        
        # STAGE 1: Initial Binary Training
        logger.info("--- STAGE 1: INITIAL BINARY TRAINING ---")
        self.run_training_cycle(model, train_loader, val_loader, stage=1, epochs=1 if dry_run else epochs)
        
        if dry_run:
            logger.info("Dry run completed successfully.")
            return True
            
        # HARD NEGATIVE MINING
        hard_indices = self.identify_hard_negatives(model, train_dataset)
        if len(hard_indices) > 0:
            logger.info("--- SPECIAL STAGE: HARD NEGATIVE FINE-TUNING ---")
            # Create a subset with hard negatives + equal number of positives
            pos_indices = np.where(train_dataset.targets['binary'] == 1)[0]
            sampled_pos = np.random.choice(pos_indices, size=min(len(pos_indices), len(hard_indices)*2), replace=False)
            hn_subset_indices = np.concatenate([hard_indices, sampled_pos])
            hn_dataset = Subset(train_dataset, hn_subset_indices)
            hn_loader = DataLoader(hn_dataset, batch_size=8, shuffle=True)
            
            # Fine-tune with lower learning rate
            self.run_training_cycle(model, hn_loader, val_loader, stage=1, epochs=5, lr=1e-4, use_cosine=False)
        
        # STAGE 2 & 3: Multi-task refinement
        logger.info("--- STAGE 2: MAGNITUDE REFINEMENT ---")
        self.run_training_cycle(model, train_loader, val_loader, stage=2, epochs=epochs)
        
        logger.info("--- STAGE 3: LOCALIZATION & FINAL TUNING ---")
        self.run_training_cycle(model, train_loader, val_loader, stage=3, epochs=max(1, epochs // 2))
        
        # Save Final Model
        torch.save(model.state_dict(), 'tuned_final_spatio_model.pth')
        logger.info("Tuned model saved to tuned_final_spatio_model.pth")
        
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='real_earthquake_dataset.h5')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    trainer = TunedProductionTrainer(args.dataset)
    trainer.execute(epochs=args.epochs, dry_run=args.dry_run)

