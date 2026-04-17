import matplotlib.pyplot as plt
import numpy as np
import os

# Data from previous turn
thresholds = [0.12, 0.35, 0.4526, 0.68]
recalls = [0.92, 0.61, 0.381, 0.12]
precisions = [0.21, 0.45, 0.305, 0.58]

plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label='SE-GNN PRC')
plt.scatter([0.381], [0.305], color='red', s=100, zorder=5, label='Operating Point (τ=0.4526)')

plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve (BMKG Certified Data)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Style like a scientific paper
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

os.makedirs('plots/submission', exist_ok=True)
plt.savefig('plots/submission/pr_curve.png', dpi=300, bbox_inches='tight')
print("Generated plots/submission/pr_curve.png")
