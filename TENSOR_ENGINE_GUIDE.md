# Tensor Engine Guide
## 5D Tensor Construction with PCA-based Common Mode Rejection

### Overview

The Tensor Engine module (`tensor_engine.py`) converts scalogram data into 5D tensors optimized for spatio-temporal earthquake precursor analysis. It implements PCA-based Common Mode Rejection (CMR) to reduce global noise and enhance local precursor signals.

## Key Features

### 1. 5D Tensor Construction
- **Tensor Shape**: (B, S=8, C=3, F=224, T=224)
- **B**: Batch size (number of earthquake events)
- **S**: Stations (8 primary geomagnetic stations)
- **C**: Components (H, D, Z geomagnetic field components)
- **F**: Frequency bins (224 bins, ULF range 0.01-0.1 Hz)
- **T**: Time bins (224 bins, 24-hour coverage)

### 2. PCA-based Common Mode Rejection (CMR)
- **Global Signal Detection**: Extracts PC1 from 8 stations per component
- **Noise Reduction**: Removes solar noise and global disturbances
- **Local Enhancement**: Preserves and enhances local precursor signals
- **Component-wise Processing**: Separate CMR for H, D, Z components

### 3. HDF5 Export
- **Efficient Storage**: Compressed HDF5 format for ML training
- **Metadata Preservation**: Embedded processing parameters
- **Fast Loading**: Optimized for deep learning frameworks
- **Cross-platform**: Compatible across different systems

## Tensor Structure Details

### Dimension Breakdown

```
Tensor Shape: (B, S=8, C=3, F=224, T=224)

B (Batch): Variable number of earthquake events
├── Event 1: Earthquake ID 35402
├── Event 2: Earthquake ID 47722
└── Event N: Earthquake ID XXXXX

S (Stations): 8 primary geomagnetic stations
├── ALR: Alor (Eastern Indonesia)
├── TND: Ternate (North Maluku)
├── PLU: Palu (Central Sulawesi)
├── GTO: Gorontalo (North Sulawesi)
├── LWK: Luwuk (Central Sulawesi)
├── GSI: Gunung Sitoli (North Sumatra)
├── LWA: Liwa (South Sumatra)
└── SMI: Sorong (West Papua)

C (Components): 3 geomagnetic field components
├── H: Horizontal intensity (nT)
├── D: Declination (degrees)
└── Z: Vertical intensity (nT)

F (Frequency): 224 frequency bins
├── Range: 0.01 - 0.1 Hz (ULF band)
├── Resolution: ~0.0004 Hz per bin
└── Focus: Earthquake precursor frequencies

T (Time): 224 time bins
├── Range: 24 hours
├── Resolution: ~6.4 minutes per bin
└── Coverage: Complete daily cycle
```

### Memory Requirements

| Batch Size | Memory Usage | Use Case |
|------------|--------------|----------|
| B=1 | 9.2 MB | Single event analysis |
| B=10 | 92 MB | Small batch training |
| B=100 | 920 MB | Medium batch training |
| B=1000 | 9.2 GB | Large batch training |

## PCA-based CMR Process

### 1. Global Signal Extraction
```python
# For each component (H, D, Z):
# 1. Reshape tensor: (B*F*T, S) - flatten spatial-temporal
# 2. Standardize across stations
# 3. Apply PCA to extract principal components
# 4. PC1 represents global signal (solar noise)
```

### 2. Common Mode Rejection
```python
# 5. Reconstruct global signal from PC1
# 6. Subtract global signal from original data
# 7. Result: Enhanced local precursor signals
```

### 3. CMR Effectiveness Analysis
- **Variance Reduction**: Measures noise reduction percentage
- **Spatial Correlation**: Analyzes inter-station correlation changes
- **Signal Preservation**: Ensures local signals are preserved

## Usage Examples

### Basic Usage
```python
from preprocessing.tensor_engine import TensorEngine

# Initialize engine
engine = TensorEngine(
    scalogram_base_path='../scalogramv3',
    metadata_path='outputs/data_audit/master_metadata.csv',
    target_shape=(224, 224),
    primary_stations=['ALR', 'TND', 'PLU', 'GTO', 'LWK', 'GSI', 'LWA', 'SMI'],
    components=['H', 'D', 'Z']
)

# Load metadata
engine.load_metadata()

# Build tensor dataset
tensor_data = engine.build_tensor_dataset(event_ids=[35402, 47722, 9180])

# Apply CMR
cmr_tensor = engine.apply_pca_cmr(tensor_data)

# Save to HDF5
engine.save_to_hdf5('output.h5', cmr_tensor)
```

### Complete Processing Pipeline
```python
# Process complete dataset with train/test splitting
saved_files = engine.process_complete_dataset(
    output_dir='outputs/tensor_datasets',
    train_test_split=True,
    apply_cmr=True,
    max_events_per_split=1000
)
```

### Loading HDF5 Data
```python
import h5py
import numpy as np

# Load tensor data for ML training
with h5py.File('train_cmr.h5', 'r') as f:
    tensor_data = f['tensor_data'][:]  # Shape: (B, 8, 3, 224, 224)
    stations = [s.decode() for s in f['stations'][:]]
    components = [c.decode() for c in f['components'][:]]
    event_ids = f['event_ids'][:]
    
    # Access CMR analysis results
    cmr_group = f['cmr_analysis']
    pc1_components = {}
    for component in ['H', 'D', 'Z']:
        pc1_components[component] = cmr_group['pc1_components'][component][:]
```

## Command Line Usage

### Basic Processing
```bash
# Run with default settings
python run_tensor_engine.py

# Process specific dataset
python -m preprocessing.tensor_engine \
    --scalogram-path ../scalogramv3 \
    --metadata-path outputs/data_audit/master_metadata.csv \
    --output-dir outputs/tensor_datasets \
    --stations ALR TND PLU GTO LWK GSI LWA SMI \
    --components H D Z \
    --target-shape 224 224
```

### Advanced Options
```bash
# Disable CMR processing
python -m preprocessing.tensor_engine \
    --scalogram-path ../scalogramv3 \
    --metadata-path outputs/data_audit/master_metadata.csv \
    --output-dir outputs/tensor_datasets \
    --no-cmr

# Limit events for testing
python -m preprocessing.tensor_engine \
    --scalogram-path ../scalogramv3 \
    --metadata-path outputs/data_audit/master_metadata.csv \
    --output-dir outputs/tensor_datasets \
    --max-events 100
```

## Output Files

### HDF5 Dataset Structure
```
dataset.h5
├── tensor_data          # Main 5D tensor (B, S, C, F, T)
├── stations            # Station codes ['ALR', 'TND', ...]
├── components          # Component names ['H', 'D', 'Z']
├── event_ids           # Corresponding event IDs
├── cmr_analysis/       # CMR processing results
│   ├── pc1_components/ # PC1 weights for each component
│   ├── explained_variance/ # Variance explained by PC1
│   └── global_signals/ # Extracted global signals
└── metadata           # Processing parameters
```

### File Types Generated

1. **train_original.h5** - Training set without CMR
2. **train_cmr.h5** - Training set with CMR applied
3. **test_original.h5** - Test set without CMR
4. **test_cmr.h5** - Test set with CMR applied
5. **train_cmr_analysis.json** - CMR effectiveness analysis
6. **test_cmr_analysis.json** - CMR effectiveness analysis

## CMR Analysis Results

### Example CMR Effectiveness
```json
{
  "noise_reduction": {
    "H": 0.85,  // 85% variance reduction
    "D": 0.82,  // 82% variance reduction
    "Z": 0.88   // 88% variance reduction
  },
  "spatial_correlation": {
    "H": {
      "original_mean_correlation": 0.75,
      "cmr_mean_correlation": 0.25,
      "correlation_reduction": 0.50
    }
  }
}
```

### Interpretation
- **High variance reduction**: Effective global noise removal
- **Reduced spatial correlation**: Less common signal across stations
- **Preserved local variations**: Enhanced precursor detectability

## Integration with ML Frameworks

### PyTorch Example
```python
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

class ScalogramDataset(Dataset):
    def __init__(self, hdf5_path):
        self.h5_file = h5py.File(hdf5_path, 'r')
        self.tensor_data = self.h5_file['tensor_data']
        self.event_ids = self.h5_file['event_ids']
    
    def __len__(self):
        return len(self.tensor_data)
    
    def __getitem__(self, idx):
        # Return tensor: (S=8, C=3, F=224, T=224)
        tensor = torch.from_numpy(self.tensor_data[idx])
        event_id = self.event_ids[idx]
        return tensor, event_id

# Create data loader
dataset = ScalogramDataset('train_cmr.h5')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch_tensors, batch_event_ids in dataloader:
    # batch_tensors shape: (32, 8, 3, 224, 224)
    # Process with your model
    pass
```

### TensorFlow Example
```python
import tensorflow as tf
import h5py

def load_hdf5_dataset(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        tensor_data = f['tensor_data'][:]
        event_ids = f['event_ids'][:]
    return tensor_data, event_ids

# Load data
train_tensors, train_event_ids = load_hdf5_dataset('train_cmr.h5')

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices(train_tensors)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Model input shape: (None, 8, 3, 224, 224)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(8, 3, 224, 224)),
    # Add your layers here
])
```

## Performance Optimization

### Memory Management
- **Chunked Processing**: Process events in batches to manage memory
- **Lazy Loading**: Load scalograms on-demand during tensor construction
- **Compression**: Use HDF5 compression for storage efficiency
- **Caching**: Cache frequently accessed scalograms

### Processing Speed
- **Parallel Loading**: Multi-threaded scalogram loading
- **Vectorized Operations**: NumPy optimizations for PCA
- **Efficient Interpolation**: Fast missing data handling
- **Batch Processing**: Process multiple events simultaneously

## Quality Control

### Data Validation
- **Completeness Check**: Minimum station/component coverage
- **Range Validation**: Scalogram value ranges
- **Shape Consistency**: Uniform tensor dimensions
- **Missing Data Handling**: Interpolation strategies

### CMR Validation
- **PC1 Variance**: Ensure PC1 captures significant variance (>50%)
- **Correlation Reduction**: Verify reduced inter-station correlation
- **Signal Preservation**: Check local signal integrity
- **Stability Analysis**: Consistent CMR across different events

## Troubleshooting

### Common Issues

1. **Missing Scalogram Files**
   - Check scalogram directory structure
   - Verify file naming conventions
   - Ensure metadata paths are correct

2. **Memory Errors**
   - Reduce batch size or max_events
   - Use chunked processing
   - Monitor system memory usage

3. **CMR Failures**
   - Check for sufficient valid data
   - Verify station coverage
   - Ensure numeric stability

4. **HDF5 Errors**
   - Check disk space availability
   - Verify write permissions
   - Ensure HDF5 library compatibility

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check tensor construction step by step
engine = TensorEngine(...)
engine.load_metadata()

# Test single event
tensor = engine.build_tensor_for_event(35402)
if tensor is not None:
    print(f"Tensor shape: {tensor.shape}")
    print(f"Data range: {tensor.min():.3f} to {tensor.max():.3f}")
```

## Future Enhancements

### Planned Features
1. **Multi-resolution Tensors**: Different F/T resolutions
2. **Adaptive CMR**: Event-specific CMR parameters
3. **Real-time Processing**: Streaming tensor construction
4. **GPU Acceleration**: CUDA-based processing
5. **Advanced Interpolation**: ML-based missing data filling

### Integration Roadmap
1. **Deep Learning Models**: CNN/RNN architectures for precursor detection
2. **Attention Mechanisms**: Spatial-temporal attention for station weighting
3. **Transfer Learning**: Pre-trained models for different regions
4. **Ensemble Methods**: Multiple model combination strategies
5. **Real-time Deployment**: Production-ready inference pipelines

This comprehensive tensor engine provides the foundation for advanced spatio-temporal earthquake precursor analysis using deep learning techniques.