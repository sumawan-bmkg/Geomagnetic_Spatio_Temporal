# Changelog

All notable changes to the Spatio-Temporal Earthquake Precursor Analysis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-15

### Added
- **Initial Release**: Complete refactoring from original `awal` folder scripts
- **GeomagneticDataReader**: Refactored from `read_mdata.py` with enhanced functionality
  - Class-based architecture with improved error handling
  - Support for compressed (.gz) and uncompressed files
  - Automatic X/Y component calculation from H/D
  - Quality control validation and thresholds
  - NPZ export functionality
  - Comprehensive logging and documentation

- **GeomagneticSignalProcessor**: Enhanced version of `signal_processing.py`
  - ULF frequency band processing (0.01-0.1 Hz) optimized for earthquake precursors
  - PC3 pulsation band processing (0.022-0.1 Hz)
  - Dual filter support (ULF + PC3)
  - Improved Z/H ratio calculation and analysis
  - Enhanced visualization with comparison plots
  - Power spectrum analysis capabilities

- **ScalogramProcessor**: New CWT-based scalogram analysis module
  - Continuous Wavelet Transform using Morlet wavelet
  - Z/H ratio scalogram generation for precursor detection
  - ULF frequency band feature extraction
  - Time-frequency analysis with logarithmic scaling
  - High-quality visualization with ULF band highlighting
  - Complete daily data processing pipeline

- **Configuration Management**
  - YAML-based configuration system (`preprocessing_config.yaml`)
  - Configurable parameters for all processing modules
  - Output directory and format management
  - Logging configuration

- **Testing and Validation**
  - Comprehensive installation test suite (`test_installation.py`)
  - Module import validation
  - Functionality testing for all components
  - Integration testing with synthetic data

- **Documentation and Examples**
  - Complete README with installation and usage instructions
  - Workflow example (`preprocessing_example.py`)
  - Configuration documentation
  - API documentation in docstrings

- **Project Structure**
  - Organized directory structure with clear separation
  - Requirements management (`requirements.txt`)
  - Setup script for easy installation (`setup.py`)
  - Git ignore file for version control
  - Professional project organization

### Technical Improvements
- **Architecture**: Object-oriented design with class-based modules
- **Error Handling**: Comprehensive exception handling throughout
- **Logging**: Structured logging with configurable levels
- **Performance**: Optimized NumPy operations and memory management
- **Compatibility**: Support for modern NumPy/SciPy versions
- **Visualization**: High-resolution plots (150 DPI) with professional styling

### Features for Earthquake Precursor Analysis
- **ULF Focus**: Optimized processing for 0.01-0.1 Hz frequency range
- **Z/H Ratio Analysis**: Enhanced ratio calculation for anomaly detection
- **Scalogram Generation**: Consistent CWT-based time-frequency analysis
- **Feature Extraction**: ULF-specific features for machine learning applications
- **Multi-Filter Comparison**: Side-by-side analysis of different frequency bands

### Output Capabilities
- **Signal Processing Plots**: Component comparisons and Z/H ratio analysis
- **Scalogram Visualizations**: Time-frequency power maps with ULF highlighting
- **Feature Plots**: Extracted ULF features for precursor analysis
- **Data Export**: NPZ format for further analysis
- **Log Files**: Detailed processing logs for debugging and validation

### Dependencies
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- PyWavelets >= 1.3.0
- pandas >= 1.3.0
- python-dateutil >= 2.8.0

### Compatibility
- Python 3.7+
- Windows, macOS, Linux
- Modern NumPy/SciPy versions with backward compatibility

## [Unreleased]

### Planned Features
- Multi-station spatial analysis capabilities
- Real-time data processing support
- Machine learning model integration
- Database connectivity
- Web-based interface
- Parallel processing optimization
- Cloud deployment support

### Known Issues
- None currently identified

### Notes
- This release represents a complete refactoring and modernization of the original codebase
- All original functionality has been preserved and enhanced
- The new architecture supports future extensions and improvements
- Comprehensive testing ensures reliability and correctness