import torch
import numpy as np

predictions = torch.load('d:/multi/spatio/Spatio_Precursor_Project/outputs/real_validation_final/reports/predictions.pth', weights_only=False)

probs = np.array(predictions['binary_probs'])
preds = np.array(predictions['binary_preds'])
targets = (np.array(predictions['true_magnitudes']) >= 5.0).astype(int)

print(f"Probabilities stats:")
print(f"  Min: {probs.min():.4f}")
print(f"  Max: {probs.max():.4f}")
print(f"  Mean: {probs.mean():.4f}")
print(f"  Median: {np.median(probs):.4f}")

high_probs = probs[probs > 0.1]
print(f"Samples with prob > 0.1: {len(high_probs)}")
if len(high_probs) > 0:
    print(f"  Max in high_probs: {high_probs.max():.4f}")

# Check magnitude predictions
mag_preds = np.array(predictions['magnitude_preds'])
mag_targets = np.array(predictions['true_magnitudes'])
print(f"\nMagnitude stats:")
print(f"  Target Mean: {mag_targets.mean():.4f}")
print(f"  Pred Mean: {mag_preds.mean():.4f}")

# Check distance predictions
dist_preds = np.array(predictions['distance_preds'])
print(f"\nDistance stats:")
print(f"  Pred Mean: {dist_preds.mean():.4f}")
