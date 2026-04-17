# Data Auditor Guide
## Comprehensive Data Auditing for Earthquake Precursor Analysis

### Overview

The Data Auditor module (`data_auditor.py`) provides comprehensive data auditing capabilities for spatio-temporal earthquake precursor analysis. It performs chronological splitting, station mapping based on Dobrovolsky radius, and generates master metadata for scalogram datasets.

## Key Features

### 1. Chronological Event Splitting
- **Train Set**: 2018-01-01 to 2024-06-30
- **Test Set**: 2024-07-01 to 2026-12-31
- Ensures temporal separation for proper model validation

### 2. Dobrovolsky Radius Calculation
- **Formula**: R = 10^(0.43 × M) km
- **Purpose**: Determines precursor detection radius for each earthquake
- **Application**: Maps stations within precursor range for each event

### 3. Station-Event Mapping
- Maps 14 primary stations to earthquake events
- Calculates great circle distances using Haversine formula
- Identifies stations within Dobrovolsky radius for each event

### 4. Master Metadata Generation
- Creates comprehensive dataset catalog
- Links earthquake events to station data
- Includes Kp index synchronization
- Provides scalogram file path mapping (when available)

## Results Summary

### Data Processing Results

```
📊 AUDIT RESULTS SUMMARY
========================

📅 Chronological Split:
   - Train Events: 13,807 (2018-2024)
   - Test Events: 1,974 (2024-2026)

🗺️ Station Mapping:
   - Total Stations: 14
   - Station-Event Mappings: 1,447
   - Events with Station Coverage: 1,219

🌍 Dobrovolsky Radius Examples:
   - M4.0: 52.5 km precursor radius
   - M5.0: 141.3 km precursor radius  
   - M6.0: 380.2 km precursor radius
   - M7.0: 1,023.3 km precursor radius
```

### Station Coverage Analysis

| Station | Mappings | Coverage (%) | Region |
|---------|----------|--------------|---------|
| TND | 189 | 13.1% | North Sulawesi |
| PLU | 188 | 13.0% | Central Sulawesi |
| GTO | 187 | 12.9% | North Sulawesi |
| JYP | 143 | 9.9% | Papua |
| LWK | 141 | 9.7% | Central Sulawesi |
| GSI | 109 | 7.5% | North Sumatra |
| LWA | 76 | 5.3% | South Sumatra |
| SMI | 74 | 5.1% | Maluku |
| MLB | 73 | 5.0% | North Sumatra |
| CLP | 64 | 4.4% | Central Java |
| LPS | 62 | 4.3% | South Sumatra |
| ALR | 56 | 3.9% | Alor |
| SCN | 53 | 3.7% | West Sumatra |
| YOG | 32 | 2.2% | Yogyakarta |

### Magnitude Distribution

#### Train Set (13,807 events)
- **<4.0**: 2,029 events (14.7%)
- **4.0-4.9**: 10,752 events (77.9%)
- **5.0-5.9**: 949 events (6.9%)
- **6.0-6.9**: 65 events (0.5%)
- **≥7.0**: 12 events (0.1%)

#### Test Set (1,974 events)
- **<4.0**: 0 events (0.0%)
- **4.0-4.9**: 1,796 events (91.0%)
- **5.0-5.9**: 167 events (8.5%)
- **6.0-6.9**: 11 events (0.6%)
- **≥7.0**: 0 events (0.0%)

#### Mapped Events (1,219 events)
- **<4.0**: 42 events (3.4%)
- **4.0-4.9**: 723 events (59.3%)
- **5.0-5.9**: 380 events (31.2%)
- **6.0-6.9**: 62 events (5.1%)
- **≥7.0**: 12 events (1.0%)

## Output Files

### 1. master_metadata.csv (274.5 KB)
**Primary output file containing complete dataset information**

**Columns:**
- `event_id`: Unique earthquake identifier
- `station_code`: Station code (e.g., ALR, TND, PLU)
- `datetime`: Event datetime
- `magnitude`: Earthquake magnitude
- `event_lat`, `event_lon`: Event coordinates
- `station_lat`, `station_lon`: Station coordinates
- `distance_km`: Distance from event to station
- `dobrovolsky_radius_km`: Precursor detection radius
- `split`: Train/test designation
- `scalogram_files_count`: Number of associated scalogram files
- `scalogram_files`: Paths to scalogram files (semicolon-separated)
- `has_scalogram_data`: Boolean flag for data availability
- `data_synchronized`: Boolean flag for synchronization status
- `kp_index`: Geomagnetic Kp index value
- `kp_datetime`: Kp index timestamp
- `kp_time_diff_hours`: Time difference to nearest Kp measurement

### 2. station_event_mappings.csv (189.6 KB)
**Station-to-event mappings with spatial relationships**

Contains all station-event pairs within Dobrovolsky radius, including distance calculations and precursor range verification.

### 3. train_events.csv (1,054.5 KB)
**Training set events (2018-2024)**

Complete earthquake catalog for training period with chronological split labels.

### 4. test_events.csv (215.1 KB)
**Test set events (2024-2026)**

Complete earthquake catalog for test period with chronological split labels.

### 5. audit_summary.json (0.7 KB)
**Comprehensive audit statistics in JSON format**

Machine-readable summary of all audit results and statistics.

### 6. detailed_audit_report.txt
**Human-readable detailed report**

Comprehensive text report with all statistics, distributions, and analysis results.

## Usage Examples

### Basic Usage
```python
from preprocessing.data_auditor import DataAuditor

# Initialize auditor
auditor = DataAuditor(
    earthquake_catalog_path='../awal/earthquake_catalog_2018_2025_merged.csv',
    kp_index_path='../awal/kp_index_2018_2026.csv',
    station_locations_path='../awal/lokasi_stasiun.csv',
    scalogram_base_path='../scalogramv3'  # Optional
)

# Run complete audit
saved_files = auditor.run_complete_audit('outputs/data_audit')
```

### Command Line Usage
```bash
# Run automated audit
python run_data_audit.py

# Run with custom paths
python -m preprocessing.data_auditor \
    --earthquake-catalog path/to/catalog.csv \
    --kp-index path/to/kp_index.csv \
    --station-locations path/to/stations.csv \
    --scalogram-path path/to/scalograms \
    --output-dir outputs/custom_audit
```

## Technical Implementation

### Dobrovolsky Radius Calculation
```python
def calculate_dobrovolsky_radius(magnitude):
    """Calculate precursor detection radius"""
    return 10 ** (0.43 * magnitude)
```

### Distance Calculation
Uses Haversine formula for great circle distance:
```python
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return 6371.0 * c  # Earth radius in km
```

### Chronological Splitting
- **Train Period**: 2018-01-01 00:00:00 to 2024-06-30 23:59:59
- **Test Period**: 2024-07-01 00:00:00 to 2026-12-31 23:59:59
- Ensures no temporal overlap between training and testing data

## Data Quality Assurance

### Station Data Validation
- Removes empty station codes
- Validates coordinate ranges
- Handles missing data gracefully

### Event Data Validation
- Validates datetime formats
- Checks magnitude ranges
- Ensures coordinate validity

### Kp Index Synchronization
- Matches events to nearest Kp measurements (within 3 hours)
- Handles timezone differences
- Provides time difference metrics

## Integration with Scalogram Processing

The master metadata file is designed to integrate seamlessly with scalogram processing pipelines:

1. **File Path Mapping**: Links events to scalogram files
2. **Synchronization Status**: Indicates data availability
3. **Spatial Context**: Provides station-event relationships
4. **Temporal Context**: Includes chronological split information

## Next Steps

1. **Review master_metadata.csv** for complete dataset information
2. **Use station_event_mappings.csv** for spatial analysis
3. **Process train/test splits** for model development
4. **Integrate with scalogram processing** pipeline
5. **Implement quality control** based on audit results

## Performance Notes

- **Processing Time**: ~25 seconds for 15,781 events and 14 stations
- **Memory Usage**: Efficient pandas operations for large datasets
- **Scalability**: Designed to handle larger station networks and event catalogs
- **Output Size**: Compressed CSV format for efficient storage

## Validation and Quality Control

The audit includes comprehensive validation:
- ✅ Data completeness checks
- ✅ Coordinate validation
- ✅ Temporal consistency verification
- ✅ Magnitude distribution analysis
- ✅ Station coverage assessment
- ✅ Dobrovolsky radius validation

This comprehensive data audit provides the foundation for reliable spatio-temporal earthquake precursor analysis.