# Panduan Evaluasi: Spatio-Temporal Earthquake Precursor Model

## Ringkasan Evaluasi

Panduan ini menjelaskan cara menjalankan evaluasi lengkap model spatio-temporal earthquake precursor dengan fokus pada:

1. **Pelatihan Model** menggunakan data 2018-2024
2. **Stress Test** menggunakan data Juli 2024-2026  
3. **Evaluasi Badai Matahari** (Kp-index > 5)
4. **Ablation Study** CMR vs Original
5. **Visualisasi Komprehensif** hasil evaluasi

## Persyaratan Sistem

### Hardware
- **GPU**: NVIDIA GPU dengan CUDA support (recommended)
- **RAM**: Minimum 16GB, recommended 32GB
- **Storage**: Minimum 50GB free space

### Software
- **Python**: 3.8 atau lebih baru
- **CUDA**: 11.0+ (jika menggunakan GPU)
- **Dependencies**: Lihat `requirements.txt`

### Data Requirements
- **Scalogram Data**: Folder `scalogramv3` dengan data scalogram
- **Earthquake Catalog**: `earthquake_catalog_2018_2025_merged.csv`
- **Kp-index Data**: `kp_index_2018_2026.csv`
- **Station Coordinates**: `lokasi_stasiun.csv`

## Quick Start

### 1. Instalasi Dependencies

```bash
# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scikit-learn h5py opencv-python
pip install tensorboard pyyaml
```

### 2. Persiapan Data

Pastikan struktur data sebagai berikut:
```
Spatio_Precursor_Project/
├── awal/
│   ├── earthquake_catalog_2018_2025_merged.csv
│   ├── kp_index_2018_2026.csv
│   └── lokasi_stasiun.csv
├── scalogramv3/
│   └── [scalogram files organized by station]
└── [project files]
```

### 3. Jalankan Evaluasi Lengkap

```bash
# Masuk ke direktori proyek
cd Spatio_Precursor_Project

# Jalankan evaluasi lengkap
python execute_evaluation.py
```

## Evaluasi Manual Step-by-Step

### Step 1: Data Auditor

```bash
# Jalankan data auditor untuk membuat master metadata
python run_data_audit.py \
    --earthquake-catalog awal/earthquake_catalog_2018_2025_merged.csv \
    --kp-index awal/kp_index_2018_2026.csv \
    --station-locations awal/lokasi_stasiun.csv \
    --scalogram-base scalogramv3 \
    --output-dir outputs/data_audit
```

### Step 2: Tensor Engine

```bash
# Proses scalogram menjadi tensor 5D dengan CMR
python src/preprocessing/tensor_engine.py \
    --scalogram-path scalogramv3 \
    --metadata-path outputs/data_audit/master_metadata.csv \
    --output-dir outputs/tensors \
    --stations ALR TND PLU GTO LWK GSI LWA SMI \
    --components H D Z \
    --target-shape 224 224
```

### Step 3: Training Model CMR

```bash
# Train model dengan CMR preprocessing
python train_model.py \
    --train-data outputs/tensors/cmr/train_cmr.h5 \
    --test-data outputs/tensors/cmr/test_cmr.h5 \
    --metadata outputs/data_audit/master_metadata.csv \
    --station-coords awal/lokasi_stasiun.csv \
    --config configs/training_config.yaml \
    --experiment-name cmr_model_2024 \
    --device cuda
```

### Step 4: Training Model Original

```bash
# Train model tanpa CMR preprocessing
python train_model.py \
    --train-data outputs/tensors/original/train.h5 \
    --test-data outputs/tensors/original/test.h5 \
    --metadata outputs/data_audit/master_metadata.csv \
    --station-coords awal/lokasi_stasiun.csv \
    --config configs/training_config.yaml \
    --experiment-name original_model_2024 \
    --device cuda
```

### Step 5: Evaluasi Lengkap

```bash
# Jalankan evaluasi lengkap dengan ablation study
python run_complete_evaluation.py \
    --scalogram-dir scalogramv3 \
    --metadata-path outputs/data_audit/master_metadata.csv \
    --station-coords awal/lokasi_stasiun.csv \
    --output-dir outputs/complete_evaluation \
    --device cuda
```

## Konfigurasi Training

### Training Configuration (configs/training_config.yaml)

```yaml
# Model Architecture
model:
  n_stations: 8
  n_components: 3
  efficientnet_pretrained: true
  gnn_hidden_dim: 256
  gnn_num_layers: 3
  dropout_rate: 0.2
  magnitude_classes: 5

# Data Configuration
data:
  batch_size: 16
  num_workers: 4
  pin_memory: true
  load_in_memory: false
  magnitude_bins: [4.0, 4.5, 5.0, 5.5, 6.0]

# Progressive Training Stages
stage_1:  # Binary Classification
  epochs: 50
  patience: 15
  optimizer:
    type: AdamW
    lr: 1.0e-4
    weight_decay: 1.0e-4

stage_2:  # Magnitude Classification
  epochs: 60
  patience: 20
  optimizer:
    type: AdamW
    lr: 5.0e-5
    weight_decay: 1.0e-4

stage_3:  # Localization
  epochs: 80
  patience: 25
  optimizer:
    type: AdamW
    lr: 2.0e-5
    weight_decay: 1.0e-4
```

## Hasil Evaluasi

### Output Structure

```
outputs/complete_evaluation/
├── experiments/
│   ├── cmr_model/
│   │   ├── best_stage_1.pth
│   │   ├── best_stage_2.pth
│   │   ├── best_stage_3.pth
│   │   └── training.log
│   └── original_model/
│       ├── best_stage_1.pth
│       ├── best_stage_2.pth
│       ├── best_stage_3.pth
│       └── training.log
├── stress_test/
│   ├── cmr/
│   │   └── stress_test_results.json
│   └── original/
│       └── stress_test_results.json
├── solar_storm_analysis/
│   ├── cmr/
│   │   └── solar_storm_analysis_results.json
│   └── original/
│       └── solar_storm_analysis_results.json
├── comprehensive_ablation_study.png
├── comprehensive_ablation_study.pdf
├── final_evaluation_report.json
├── final_evaluation_report.md
└── evaluation.log
```

### Key Metrics

#### Primary Metrics
- **F1-Score**: Untuk deteksi precursor (binary classification)
- **MAE**: Mean Absolute Error untuk estimasi magnitudo
- **Accuracy**: Akurasi keseluruhan model
- **Precision/Recall**: Untuk analisis detail performa

#### Specialized Metrics
- **Azimuth Error**: Error estimasi arah gempa (dalam derajat)
- **Distance Error**: Error estimasi jarak gempa (dalam km)
- **Solar Storm Impact**: Performa saat Kp-index > 5

### Visualisasi

#### 1. Comprehensive Ablation Study
- Perbandingan F1-Score CMR vs Original
- Perbandingan MAE untuk estimasi magnitudo
- Analisis performa saat badai matahari
- Summary improvement metrics

#### 2. Stress Test Analysis
- Performa temporal (Q3 2024 - Q2 2026)
- Performa per range magnitudo
- Analisis degradasi performa

#### 3. Solar Storm Analysis
- Performa saat kondisi normal vs badai matahari
- Distribusi kondisi geomagnetik
- Korelasi Kp-index dengan performa model

## Interpretasi Hasil

### Kriteria Keberhasilan

#### CMR Effectiveness
- **F1-Score Improvement**: CMR > Original
- **MAE Improvement**: CMR < Original (lower is better)
- **Solar Storm Robustness**: Performa stabil saat Kp > 5

#### Model Deployment Readiness
- **F1-Score**: > 0.70 untuk deployment
- **MAE**: < 0.5 untuk estimasi magnitudo
- **Consistency**: Performa stabil across time periods

### Expected Results

Berdasarkan teori CMR, diharapkan:

1. **CMR Model** menunjukkan performa lebih baik saat badai matahari
2. **F1-Score improvement** 5-15% dibanding model original
3. **MAE reduction** 10-20% untuk estimasi magnitudo
4. **Robustness** terhadap noise solar yang tinggi

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--batch-size 8

# Or use CPU
--device cpu
```

#### 2. Data Not Found
```bash
# Check data paths
ls awal/
ls scalogramv3/

# Verify metadata
head outputs/data_audit/master_metadata.csv
```

#### 3. Training Slow
```bash
# Reduce model size
# Edit configs/training_config.yaml:
model:
  gnn_hidden_dim: 128
  gnn_num_layers: 2
```

#### 4. Insufficient Disk Space
```bash
# Clean up intermediate files
rm -rf outputs/tensors/*/temp/
rm -rf outputs/training/*/tensorboard/
```

### Performance Optimization

#### For Limited Resources
```yaml
# Quick training config
data:
  batch_size: 8
  load_in_memory: false

stage_1:
  epochs: 20
stage_2:
  epochs: 25
stage_3:
  epochs: 30
```

#### For High Performance
```yaml
# Production config
data:
  batch_size: 32
  load_in_memory: true
  num_workers: 8

stage_1:
  epochs: 80
stage_2:
  epochs: 100
stage_3:
  epochs: 120
```

## Synthetic Demo

Jika data real tidak tersedia, jalankan demo dengan data sintetis:

```bash
# Demo dengan data sintetis
python examples/complete_workflow_example.py
```

Demo ini akan:
1. Generate data sintetis yang realistis
2. Menjalankan pipeline lengkap
3. Menghasilkan visualisasi contoh
4. Mendemonstrasikan semua fitur evaluasi

## Kesimpulan

Evaluasi ini memberikan analisis komprehensif tentang efektivitas Common Mode Rejection (CMR) dalam meningkatkan performa model deteksi precursor gempa, khususnya saat kondisi badai matahari (Kp-index > 5).

**Expected Outcome**: CMR preprocessing akan menunjukkan peningkatan signifikan dalam F1-Score dan pengurangan MAE, terutama saat kondisi geomagnetik yang terganggu.

Untuk pertanyaan atau masalah teknis, periksa log file di `outputs/complete_evaluation/evaluation.log`.