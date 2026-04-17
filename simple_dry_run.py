#!/usr/bin/env python3
"""
Simple Dry Run for Production Training
Bypasses unicode issues and focuses on core functionality
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
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from run_production_train import ProductionTrainingPipeline

# Setup logging without unicode
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def simple_dry_run():
    """Simple dry run test."""
    print("=== SIMPLE DRY RUN TEST ===")
    
    try:
        # Create pipeline
        pipeline = ProductionTrainingPipeline(
            dataset_path='outputs/corrective_actions/certified_spatio_dataset.h5',
            station_coords_path='../awal/lokasi_stasiun.csv',
            output_dir='./outputs/simple_dry_run',
            experiment_name='simple_test'
        )
        
        print("1. Loading station coordinates...")
        pipeline.load_station_coordinates()
        print(f"   Loaded {len(pipeline.station_coordinates)} stations")
        
        print("2. Loading dataset...")
        pipeline.load_dataset()
        print(f"   Train: {len(pipeline.train_dataset)} samples")
        print(f"   Val: {len(pipeline.val_dataset)} samples")
        
        print("3. Creating data loaders...")
        pipeline.create_data_loaders()
        print(f"   Train batches: {len(pipeline.train_loader)}")
        print(f"   Val batches: {len(pipeline.val_loader)}")
        
        print("4. Creating model...")
        pipeline.create_model()
        print(f"   Model parameters: {sum(p.numel() for p in pipeline.model.parameters()):,}")
        
        print("5. Testing data loading...")
        for i, (data, targets) in enumerate(pipeline.train_loader):
            print(f"   Batch {i}: Data shape {data.shape}")
            print(f"   Targets keys: {list(targets.keys())}")
            
            # Test forward pass
            pipeline.model.eval()
            with torch.no_grad():
                outputs = pipeline.model(data)
            
            print(f"   Output keys: {list(outputs.keys())}")
            
            if i >= 2:  # Test only first 3 batches
                break
        
        print("=== DRY RUN PASSED ===")
        return True
        
    except Exception as e:
        print(f"=== DRY RUN FAILED ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = simple_dry_run()
    if success:
        print("\nSUCCESS: Ready for production training")
    else:
        print("\nFAILED: Fix issues before production training")