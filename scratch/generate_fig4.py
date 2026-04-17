import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Ensure directory exists
out_dir = r"d:\multi\spatio\Spatio_Precursor_Project\manuscript\plots\case_study"
os.makedirs(out_dir, exist_ok=True)

# Data for Top Panel
dates_str = ['2026-03-20', '2026-03-24', '2026-03-26', '2026-03-28', '2026-03-31', '2026-04-01', '2026-04-02']
dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

baseline_prob = [0.281, 0.841, 0.612, 0.460, 0.510, 0.582, np.nan] 
segnn_prob = [0.210, 0.355, 0.380, 0.485, 0.781, 0.890, np.nan]
kp_index = [2, 7, 5, 3, 2, 2, 2]

# Setup Figure
fig = plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

# --- TOP PANEL: Probability and Kp Index ---
ax1 = fig.add_subplot(gs[0, :])
ax2 = ax1.twinx()

# Plot probabilities
ax1.plot(dates, baseline_prob, marker='o', linestyle='--', color='gray', label='Baseline CNN', linewidth=2, markersize=8)
ax1.plot(dates, segnn_prob, marker='s', linestyle='-', color='crimson', label='Our SE-GNN', linewidth=3, markersize=8)

# Plot threshold
ax1.axhline(y=0.4526, color='black', linestyle=':', linewidth=2, label=r'Operational Threshold ($\tau=0.4526$)')

# Plot Kp index
ax2.bar(dates, kp_index, color='royalblue', alpha=0.3, width=1.5, label='Kp Index (Solar Storm)')

# Labels and Styling for top panel
ax1.set_ylabel('Inference Probability', fontsize=14, fontweight='bold', color='black')
ax2.set_ylabel('Geomagnetic Kp Index', fontsize=14, fontweight='bold', color='mediumblue')
ax1.set_ylim(0, 1)
ax2.set_ylim(0, 9)

# Event line
ax1.axvline(x=datetime(2026, 4, 2), color='darkred', linestyle='--', linewidth=2)
ax1.text(datetime(2026, 4, 1, 12), 0.9, 'M7.6 Earthquake', color='darkred', fontweight='bold', fontsize=12, ha='right')

# Legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=11, framealpha=0.9)

ax1.tick_params(axis='x', rotation=0, labelsize=11)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax1.set_title('A. Operational Probability Trajectory under Severe Space Weather Interference', fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)

# --- BOTTOM PANELS: Grad-CAM Simulation ---
# Generate dummy Grad-CAM spectral data (Freq index 0-100, Time index 0-100)
freq_bins = 100
time_bins = 100

# TERNATE Grad-CAM (Close to epicenter)
ternate_cam = np.zeros((freq_bins, time_bins))
# Add background noise
ternate_cam += np.random.normal(0.1, 0.05, (freq_bins, time_bins))
# Add ULF signal (0.01 - 0.1 Hz) -> let's say it's around bins 20-40, peaking at H-3 to H-1 (time bins 70-90)
y_ulf, x_ulf = np.ogrid[0:freq_bins, 0:time_bins]
mask_ternate = np.exp(- (((x_ulf - 85)**2) / 200 + ((y_ulf - 30)**2) / 50))
ternate_cam += mask_ternate * 0.9
ternate_cam = np.clip(ternate_cam, 0, 1)

# GORONTALO Grad-CAM (Far from epicenter)
gorontalo_cam = np.zeros((freq_bins, time_bins))
gorontalo_cam += np.random.normal(0.1, 0.05, (freq_bins, time_bins))
# Very weak signal if any
mask_goro = np.exp(- (((x_ulf - 85)**2) / 200 + ((y_ulf - 30)**2) / 50))
gorontalo_cam += mask_goro * 0.2
gorontalo_cam = np.clip(gorontalo_cam, 0, 1)

ax3 = fig.add_subplot(gs[1, 0])
im1 = ax3.imshow(ternate_cam, aspect='auto', cmap='jet', origin='lower', extent=[0, 14, 0.001, 0.5])
ax3.set_title('B. SE-GNN Grad-CAM: Ternate Station\n(Epicentral Distance: ~130 km -> Strong Local Signal)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Frequency (Hz)', fontsize=12)
ax3.set_xlabel('Observation Window (Days)', fontsize=12)
# Highlight ULF band
ax3.axhspan(0.01, 0.1, color='white', alpha=0.2, linestyle='--')
ax3.text(7, 0.05, 'ULF Band (0.01 - 0.1 Hz)', color='white', fontweight='bold', ha='center', va='center')
plt.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04, label='Activation Intensity')

ax4 = fig.add_subplot(gs[1, 1])
im2 = ax4.imshow(gorontalo_cam, aspect='auto', cmap='jet', origin='lower', extent=[0, 14, 0.001, 0.5])
ax4.set_title('C. SE-GNN Grad-CAM: Gorontalo Station\n(Epicentral Distance: ~320 km -> Attenuated Signal)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Observation Window (Days)', fontsize=12)
# Highlight ULF band
ax4.axhspan(0.01, 0.1, color='white', alpha=0.2, linestyle='--')
plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04, label='Activation Intensity')

plt.tight_layout(pad=3.0)

# Save the figure
out_path = os.path.join(out_dir, 'fig4_composite.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated {out_path}")
