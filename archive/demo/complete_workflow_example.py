#!/usr/bin/env python3
"""
Complete Workflow Example for Spatio-Temporal Earthquake Precursor Project

This script demonstrates the complete end-to-end workflow:
1. Process scalogram data into 5D tensors with CMR
2. Create train/test datasets
3. Train the model progressively through 3 stages
4. Evaluate and analyze results

This is a comprehensive example showing how all components work together.
"""
import sys
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.preprocessing.tensor_engine import TensorEngine
from src.models.spatio_temporal_model import create_model
from src.training.dataset import create_data_loaders
from src.training.trainer import SpatioTemporalTrainer
from src.training.utils import setup_training, create_experiment_directory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_synthetic_data(output_dir: str):
    """
    Create synthetic data for demonstration purposes.
    
    In a real scenario, this would be replaced with actual scalogram data processing.
    """
    logger.info("Creating synthetic data for demonstration...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic metadata
    n_events = 100
    stations = ['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI']
    
    metadata_records = []
    for event_id in range(1000, 1000 + n_events):
        for station in stations:
            # Random earthquake parameters
            magnitude = np.random.uniform(4.0, 7.0)
            event_lat = np.random.uniform(-10, 10)
            event_lon = np.random.uniform(95, 140)
            station_lat = np.random.uniform(-10, 10)
            station_lon = np.random.uniform(95, 140)
            
            # Calculate distance
            distance = np.sqrt((event_lat - station_lat)**2 + (event_lon - station_lon)**2) * 111  # Rough km conversion
            
            # Assign train/test split (80/20)
            split = 'train' if np.random.random() < 0.8 else 'test'
            
            metadata_records.append({
                'event_id': event_id,
                'station_code': station,
                'magnitude': magnitude,
                'event_lat': event_lat,
                'event_lon': event_lon,
                'station_lat': station_lat,
                'station_lon': station_lon,
                'distance_km': distance,
                'split': split,
                'has_scalogram_data': True
            })
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_path = output_dir / 'synthetic_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    
    # Create synthetic tensor data
    train_events = metadata_df[metadata_df['split'] == 'train']['event_id'].unique()
    test_events = metadata_df[metadata_df['split'] == 'test']['event_id'].unique()
    
    # Synthetic tensor dimensions: (B, S=8, C=3, F=224, T=224)
    for split, events in [('train', train_events), ('test', test_events)]:
        n_samples = len(events)
        
        # Create synthetic tensor data
        tensor_data = np.random.randn(n_samples, 8, 3, 224, 224).astype(np.float32)
        
        # Add some structure to make it more realistic
        for i in range(n_samples):
            # Add frequency-dependent patterns
            for f in range(224):
                freq_weight = np.exp(-f / 50.0)  # Lower frequencies have higher amplitude
                tensor_data[i, :, :, f, :] *= freq_weight
            
            # Add station correlations
            base_pattern = np.random.randn(3, 224, 224) * 0.5
            for s in range(8):
                correlation = np.random.uniform(0.3, 0.8)
                tensor_data[i, s] += base_pattern * correlation
        
        # Save as HDF5
        import h5py
        hdf5_path = output_dir / f'synthetic_{split}.h5'
        
        with h5py.File(hdf5_path, 'w') as f:
            f.create_dataset('tensor_data', data=tensor_data, compression='gzip')
            f.create_dataset('event_ids', data=events)
            f.create_dataset('stations', data=[s.encode('utf-8') for s in stations])
            f.create_dataset('components', data=[c.encode('utf-8') for c in ['H', 'D', 'Z']])
            
            # Metadata
            f.attrs['tensor_shape'] = tensor_data.shape
            f.attrs['n_stations'] = 8
            f.attrs['n_components'] = 3
            f.attrs['n_freq'] = 224
            f.attrs['n_time'] = 224
    
    logger.info(f"Synthetic data created:")
    logger.info(f"  Metadata: {metadata_path}")
    logger.info(f"  Train tensor: {output_dir / 'synthetic_train.h5'}")
    logger.info(f"  Test tensor: {output_dir / 'synthetic_test.h5'}")
    logger.info(f"  Train events: {len(train_events)}")
    logger.info(f"  Test events: {len(test_events)}")
    
    return {
        'metadata_path': metadata_path,
        'train_hdf5': output_dir / 'synthetic_train.h5',
        'test_hdf5': output_dir / 'synthetic_test.h5'
    }


def demonstrate_tensor_engine():
    """
    Demonstrate tensor engine functionality.
    
    Note: This would normally process real scalogram data.
    """
    logger.info("=== Tensor Engine Demonstration ===")
    
    # In a real scenario, you would use:
    # engine = TensorEngine(
    #     scalogram_base_path='path/to/scalogramv3',
    #     metadata_path='path/to/master_metadata.csv'
    # )
    # 
    # # Process complete dataset
    # saved_files = engine.process_complete_dataset(
    #     output_dir='outputs/tensors',
    #     train_test_split=True,
    #     apply_cmr=True
    # )
    
    logger.info("Tensor engine would process scalogram data into 5D tensors with CMR")
    logger.info("For this demo, we'll use synthetic data instead")


def demonstrate_model_training(data_paths: dict):
    """
    Demonstrate complete model training workflow.
    """
    logger.info("=== Model Training Demonstration ===")
    
    # Setup training environment
    config = setup_training(seed=42, device='cpu')  # Use CPU for demo
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(
        base_dir='outputs/demo_training',
        experiment_name='complete_workflow_demo'
    )
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_loader, test_loader, dataset_info = create_data_loaders(
        train_hdf5_path=str(data_paths['train_hdf5']),
        test_hdf5_path=str(data_paths['test_hdf5']),
        metadata_path=str(data_paths['metadata_path']),
        batch_size=4,  # Small batch size for demo
        num_workers=0,  # No multiprocessing for demo
        load_in_memory=True  # Load in memory for faster demo
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("Creating model...")
    
    model = create_model(config={
        'n_stations': 8,
        'n_components': 3,
        'gnn_hidden_dim': 128,  # Smaller for demo
        'gnn_num_layers': 2,
        'magnitude_classes': 5,
        'device': 'cpu'
    })
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    trainer = SpatioTemporalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device='cpu',
        output_dir=experiment_dir,
        experiment_name='demo'
    )
    
    # Demo training configuration (very short for demonstration)
    demo_config = {
        'stage_1': {
            'epochs': 2,
            'patience': 5,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {'type': 'AdamW', 'lr': 1e-3},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 2},
            'save_best': True,
            'checkpoint_interval': 1
        },
        'stage_2': {
            'epochs': 2,
            'patience': 5,
            'load_previous_best': True,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {'type': 'AdamW', 'lr': 5e-4},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 2},
            'save_best': True,
            'checkpoint_interval': 1
        },
        'stage_3': {
            'epochs': 2,
            'patience': 5,
            'load_previous_best': True,
            'train_backbone': True,
            'train_gnn': True,
            'optimizer': {'type': 'AdamW', 'lr': 2e-4},
            'scheduler': {'type': 'CosineAnnealingLR', 'T_max': 2},
            'save_best': True,
            'checkpoint_interval': 1
        }
    }
    
    try:
        # Train model progressively
        logger.info("Starting progressive training (demo with 2 epochs per stage)...")
        
        training_history = trainer.train_progressive(demo_config)
        
        logger.info("Training completed successfully!")
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        
        test_metrics = trainer.evaluate_model(
            test_loader=test_loader,
            checkpoint_path=experiment_dir / 'best_stage_3.pth'
        )
        
        logger.info("Test evaluation completed:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
        
        return training_history, test_metrics
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        trainer.close()


def demonstrate_model_inference():
    """
    Demonstrate model inference on new data.
    """
    logger.info("=== Model Inference Demonstration ===")
    
    # This would show how to use the trained model for inference
    logger.info("Model inference would involve:")
    logger.info("1. Loading trained model checkpoint")
    logger.info("2. Processing new scalogram data through tensor engine")
    logger.info("3. Running inference to get precursor predictions")
    logger.info("4. Interpreting results (binary, magnitude, localization)")


def main():
    """
    Main function demonstrating the complete workflow.
    """
    logger.info("Starting Complete Workflow Demonstration")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create synthetic data (replace with real data processing)
        logger.info("Step 1: Data Preparation")
        data_paths = create_synthetic_data('outputs/demo_data')
        
        # Step 2: Demonstrate tensor engine (would process real scalograms)
        logger.info("\nStep 2: Tensor Engine Processing")
        demonstrate_tensor_engine()
        
        # Step 3: Demonstrate model training
        logger.info("\nStep 3: Model Training")
        training_history, test_metrics = demonstrate_model_training(data_paths)
        
        # Step 4: Demonstrate inference
        logger.info("\nStep 4: Model Inference")
        demonstrate_model_inference()
        
        logger.info("\n" + "=" * 60)
        logger.info("Complete Workflow Demonstration Completed Successfully!")
        logger.info("=" * 60)
        
        # Summary
        logger.info("\nWorkflow Summary:")
        logger.info("✓ Data preparation and tensor creation")
        logger.info("✓ Model architecture with EfficientNet + GNN")
        logger.info("✓ Progressive training (3 stages)")
        logger.info("✓ Multi-task learning (binary + magnitude + localization)")
        logger.info("✓ Model evaluation and metrics")
        
        logger.info("\nFor real usage:")
        logger.info("1. Replace synthetic data with actual scalogram processing")
        logger.info("2. Use real station coordinates from lokasi_stasiun.csv")
        logger.info("3. Adjust training configuration for your dataset size")
        logger.info("4. Run full training with more epochs")
        
    except Exception as e:
        logger.error(f"Workflow demonstration failed: {e}")
        raise


if __name__ == '__main__':
    main()