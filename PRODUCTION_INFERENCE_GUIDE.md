# Production Inference & Validation Guide
## Spatio-Temporal Earthquake Precursor Model

**Version:** 1.0  
**Date:** April 16, 2026  
**Status:** Production Ready ✅

---

## 🎯 Overview

This guide provides comprehensive instructions for running production-grade inference validation on the Spatio-Temporal Earthquake Precursor Model. The system implements the complete Master Prompt requirements for Q1 journal standards validation.

### Key Features
- **Multi-Stage Architecture:** Binary Classification → Magnitude Estimation → Localization
- **Scientific Rigor:** Follows Q1 journal validation methodology
- **Production Ready:** Robust error handling and comprehensive reporting
- **Explainability:** Grad-CAM analysis and attention visualizations

---

## 🚀 Quick Start

### 1. Test Model Loading
```bash
python run_production_inference.py --test-only
```

### 2. Run Demo Validation (50 samples)
```bash
python run_production_inference.py
```

### 3. Run Extended Validation (100 samples)
```bash
python run_production_inference.py --samples 100
```

### 4. Use GPU Acceleration
```bash
python run_production_inference.py --gpu --samples 100
```

---

## 📋 Prerequisites

### System Requirements
- **Python:** 3.8+
- **Memory:** 8GB+ RAM recommended
- **Storage:** 30GB+ free space
- **GPU:** Optional (CUDA-compatible for acceleration)

### Dependencies
```bash
pip install torch torchvision matplotlib seaborn scikit-learn h5py pandas numpy
```

### Required Files
- **Model Checkpoint:** `best_stage_3.pth` (trained model weights)
- **Dataset:** `real_earthquake_dataset.h5` (28.6 GB earthquake data)

---

## 🔧 Available Scripts

### 1. `run_production_inference.py` (Recommended)
**Main production launcher with comprehensive options**

```bash
# Basic usage
python run_production_inference.py

# Advanced options
python run_production_inference.py \
    --model path/to/model.pth \
    --dataset path/to/dataset.h5 \
    --samples 100 \
    --gpu \
    --output custom_output_dir
```

**Options:**
- `--model`: Path to model checkpoint
- `--dataset`: Path to HDF5 dataset
- `--samples`: Number of samples to process
- `--gpu`: Use GPU if available
- `--test-only`: Only test model loading
- `--output`: Custom output directory

### 2. `demo_inference_validation.py`
**Direct demo validation script**

```bash
python demo_inference_validation.py --samples 50
```

### 3. `test_model_loading.py`
**Model loading verification**

```bash
python test_model_loading.py
```

---

## 📊 Output Structure

### Generated Files
```
outputs/demo_inference_validation/
├── reports/
│   ├── demo_validation_report.json    # Detailed metrics (JSON)
│   └── demo_validation_report.md      # Human-readable report
├── plots/
│   └── demo_validation_results.png    # Comprehensive visualizations
└── demo_inference_validation.log      # Execution log
```

### Report Contents
- **Performance Metrics:** Precision, Recall, F1-Score, ROC AUC
- **Confusion Matrix:** Detailed classification results
- **Model Architecture:** Technical specifications
- **Sample Analysis:** Individual prediction results
- **Visualizations:** Probability distributions, scatter plots

---

## 🔬 Validation Methodology

### Stage 1: Model Loading & Configuration
- Load EfficientNet-B0 + GNN Fusion architecture
- Verify model weights and parameters
- Set evaluation mode and disable gradients
- Validate model components and dimensions

### Stage 2: Data Preprocessing Validation
- Load earthquake dataset (HDF5 format)
- Apply preprocessing pipeline:
  - CWT Scaling
  - PCA-CMR (Solar Cycle 25 cleaning)
  - Z-score normalization
  - Dobrovolsky Radius Filter

### Stage 3: Multi-Stage Inference Execution
- **Binary Classification:** Precursor detection (threshold: 0.5)
- **Magnitude Estimation:** Continuous magnitude prediction
- **Localization:** Azimuth and distance estimation
- Process samples with comprehensive error handling

### Stage 4: Performance Metrics Computation
- Binary classification metrics (Precision, Recall, F1, AUC)
- Regression metrics for positive detections
- Confusion matrix analysis
- Statistical significance testing

### Stage 5: Explainability Analysis
- Probability distribution visualization
- Magnitude prediction accuracy plots
- Detection results over time
- Model confidence analysis

---

## 📈 Performance Interpretation

### Binary Classification Metrics
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **Accuracy:** Overall correct predictions
- **ROC AUC:** Area under receiver operating characteristic curve

### Expected Performance Ranges
- **High Precision (>0.8):** Low false positive rate
- **Moderate Recall (0.6-0.8):** Reasonable detection rate
- **Balanced F1-Score (>0.7):** Good overall performance
- **High Accuracy (>0.9):** Reliable predictions

### Conservative Model Behavior
The model may show conservative behavior (low recall, high precision) which is appropriate for earthquake precursor detection where false alarms have significant consequences.

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```
Error: Model checkpoint not found
```
**Solution:** Check model path and ensure training completed successfully
```bash
python run_production_inference.py --test-only
```

#### 2. Dataset Access Issues
```
Error: Dataset not found
```
**Solution:** Verify dataset path and file permissions
```bash
ls -la real_earthquake_dataset.h5
```

#### 3. Memory Issues
```
Error: Out of memory
```
**Solution:** Reduce sample count or use CPU mode
```bash
python run_production_inference.py --samples 25
```

#### 4. GPU Issues
```
Error: CUDA out of memory
```
**Solution:** Use CPU mode or reduce batch size
```bash
python run_production_inference.py  # Uses CPU by default
```

### Debug Mode
Enable detailed logging by checking the log file:
```bash
tail -f outputs/demo_inference_validation/demo_inference_validation.log
```

---

## 🔍 Advanced Usage

### Custom Model Checkpoints
```bash
python run_production_inference.py \
    --model outputs/custom_training/best_model.pth \
    --samples 200
```

### Batch Processing
```bash
# Process multiple sample sizes
for samples in 50 100 200; do
    python run_production_inference.py \
        --samples $samples \
        --output "outputs/validation_${samples}_samples"
done
```

### Performance Profiling
```bash
# Time the execution
time python run_production_inference.py --samples 100
```

---

## 📚 Scientific Validation Standards

### Q1 Journal Requirements ✅
- **Blind Test:** Uses completely unseen test data
- **Reproducibility:** Detailed logging and random seed control
- **Statistical Rigor:** Comprehensive metrics and significance testing
- **Explainability:** Visual evidence and model interpretation
- **Documentation:** Complete methodology and results reporting

### Publication-Ready Outputs
- **Figures:** High-resolution PNG plots (300 DPI)
- **Tables:** Formatted metrics in Markdown and JSON
- **Methodology:** Detailed validation procedure documentation
- **Results:** Statistical analysis with confidence intervals

---

## 🎉 Success Indicators

### Model Readiness Checklist
- ✅ Model loads without errors
- ✅ Forward pass executes successfully
- ✅ Multi-stage outputs generated
- ✅ Metrics computed correctly
- ✅ Visualizations rendered
- ✅ Reports generated successfully

### Production Deployment Readiness
- ✅ Robust error handling
- ✅ Comprehensive logging
- ✅ Performance monitoring
- ✅ Scientific validation
- ✅ Documentation completeness

---

## 📞 Support

### Getting Help
1. **Check Logs:** Review execution logs for detailed error information
2. **Test Components:** Use `--test-only` flag to isolate issues
3. **Reduce Complexity:** Start with smaller sample sizes
4. **Verify Environment:** Ensure all dependencies are installed

### Common Solutions
- **Import Errors:** Reinstall dependencies with pip
- **Path Issues:** Use absolute paths for model and dataset
- **Memory Issues:** Reduce sample count or use CPU mode
- **Performance Issues:** Enable GPU acceleration if available

---

*Generated by Production Inference & Validation System*  
*Spatio-Temporal Earthquake Precursor Project*  
*April 16, 2026*