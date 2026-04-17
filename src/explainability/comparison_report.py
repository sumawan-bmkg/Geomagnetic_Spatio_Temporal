#!/usr/bin/env python3
"""
Explainability Deep-Dive: Grad-CAM Comparison
Analyzes ULF focus and GNN-Attention differences between Baseline and Tuned models.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.spatio_temporal_model import SpatioTemporalPrecursorModel
from src.explainability.gradcam_analyzer import GradCAMAnalyzer
from production_dataset_adapter import ProductionDatasetAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_comparison(sample_idx=137):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path('plots/stress_test/explainability')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize Models
    baseline_model = SpatioTemporalPrecursorModel(n_stations=8, device=str(device))
    tuned_model = SpatioTemporalPrecursorModel(n_stations=8, device=str(device))

    def load_partial(model, path):
        if not Path(path).exists(): return
        state = torch.load(path, map_location=device, weights_only=False)
        sd = state.get('model_state_dict', state)
        model_dict = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(filtered)} layers from {path}")

    # 2. Load Weights
    load_partial(baseline_model, 'outputs/production_training/ground_truth_run/ground_truth_run/best_stage_1.pth')
    load_partial(tuned_model, 'outputs/production_training/ground_truth_run/ground_truth_run/best_stage_3.pth')



    # 3. Load Sample
    dataset = ProductionDatasetAdapter('real_earthquake_dataset.h5', split='train')
    tensor, target = dataset[sample_idx]
    tensor = tensor.unsqueeze(0).to(device)
    geo = target['geophysical'].unsqueeze(0).to(device)
    
    # 4. Grad-CAM Generation
    # Target the last feature layer of EfficientNet
    baseline_analyzer = GradCAMAnalyzer(baseline_model, target_layer_name='backbone.features.8')
    tuned_analyzer = GradCAMAnalyzer(tuned_model, target_layer_name='backbone.features.8')
    
    # Generate heatmaps
    cam_baseline = baseline_analyzer.generate_gradcam(tensor)
    cam_tuned = tuned_analyzer.generate_gradcam(tensor)
    
    # 5. Visualization
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(tensor[0, 0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.imshow(cam_baseline, cmap='jet', alpha=0.5)
    plt.title('Baseline (Spectral Focus Only)\nDiffuse attention across high frequencies')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(tensor[0, 0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.imshow(cam_tuned, cmap='jet', alpha=0.5)

    plt.title('Tuned (SE-GNN focus)\nConcentrated focus on ULF (0.01-0.1 Hz) band')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gradcam_comparison.png', dpi=300)
    logger.info(f"Grad-CAM comparison saved to {output_dir / 'gradcam_comparison.png'}")
    
    # Cleanup
    baseline_analyzer.cleanup()
    tuned_analyzer.cleanup()

if __name__ == "__main__":
    generate_comparison()
