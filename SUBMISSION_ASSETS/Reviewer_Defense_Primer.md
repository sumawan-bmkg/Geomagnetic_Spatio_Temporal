# Peer-Review Defense Primer (Scientific Rationale)

This document provides formal technical responses to potential critical questions from reviewers, specifically regarding data processing and physical validity.

### Question 1: Does PCA-CMR (Common Mode Rejection) remove precursor signals?
**Defense**: 
PCA-CMR is designed to identify and subtract the primary eigenvector ($PC_1$), which represents the spatially uniform "Common Mode" signal across the entire network. In seismology, solar-induced noise (geomagnetic storms) behaves as a global plane-wave, hitting all stations simultaneously with zero spatial decay. Precursors, however, are point-source-like signals that exhibit significant spatial decay ($1/r^2$) within the **Dobrovolsky Radius** ($R = 10^{0.43M}$). 

Because the precursor signal is localized and spatially heterogeneous, it is mathematically relegated to the higher-order principal components ($PC_{2-N}$), which are **preserved** during $PC_1$ rejection. Thus, PCA-CMR acts as a spatial high-pass filter that suppresses global noise while maintaining the integrity of local lithospheric transients.

### Question 2: How distinguish precursors from ULF Solar harmonics?
**Defense**: 
While both signals share the ULF band ($0.01 - 0.1$ Hz), they differ in **Vertical-to-Horizontal ($Z/H$) coherence**. Magnetospheric induction (Solar Storms) exhibits high coherence and predictable phase-shifts between components. By integrating **Kp and Dst indices** into the SE-GNN architecture, the multi-task heads learn the "Normal" correlation pattern of solar storms. Deviations from this global correlation—specifically localized phase lags and tilted polarization vectors—are classified by the GNN as lithospheric.

### Question 3: Why SE-GNN instead of simple CNN?
**Defense**: 
Earthquake preparation is a spatio-temporal process. A standard CNN treats station data as independent images. The **GNN Attention** layer explicitly models the adjacency and distance between stations, allowing the model to perform "Geophysical Consensus." The **Squeeze-and-Excitation (SE)** block further optimizes this by adaptively weighting stations with high SNR, suppressing those affected by local anthropogenic noise.
