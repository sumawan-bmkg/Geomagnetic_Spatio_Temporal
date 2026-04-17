"""
Example usage of the preprocessing modules for geomagnetic data analysis.
Demonstrates the complete workflow from data reading to scalogram generation.
"""
import os
import sys
import numpy as np
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import GeomagneticDataReader, GeomagneticSignalProcessor, ScalogramProcessor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing_example.log')
        ]
    )


def example_data_reading():
    """Example of reading geomagnetic data."""
    print("\n=== Data Reading Example ===")
    
    # Initialize data reader
    reader = GeomagneticDataReader()
    
    # Example parameters (adjust as needed)
    year = 2023
    month = 1
    day = 1
    station = 'ALR'
    data_path = '../data/raw'  # Adjust path as needed
    
    try:
        # Read daily data
        data = reader.read_daily_data(year, month, day, station, data_path)
        
        print(f"Successfully read data for {station} on {year}-{month:02d}-{day:02d}")
        print(f"Number of records: {data['n_records']}")
        print(f"H component range: {np.nanmin(data['H']):.2f} to {np.nanmax(data['H']):.2f} nT")
        print(f"Z component range: {np.nanmin(data['Z']):.2f} to {np.nanmax(data['Z']):.2f} nT")
        
        # Save to NPZ format
        output_file = reader.save_to_npz(data)
        print(f"Data saved to: {output_file}")
        
        return data
        
    except FileNotFoundError:
        print(f"Data file not found for {station} on {year}-{month:02d}-{day:02d}")
        print("Generating synthetic data for demonstration...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic geomagnetic data for demonstration."""
    print("\n=== Generating Synthetic Data ===")
    
    # Create 24-hour synthetic data (1 Hz sampling)
    n_samples = 86400  # 24 hours * 3600 seconds
    t = np.arange(n_samples)
    
    # Base field values (typical for mid-latitude station)
    h_base = 40000  # nT
    z_base = 30000  # nT
    
    # Add ULF pulsations and noise
    np.random.seed(42)  # For reproducible results
    
    # ULF components (0.01-0.1 Hz)
    h_ulf = (50 * np.sin(2 * np.pi * 0.05 * t / 3600) +  # 0.05 Hz
             30 * np.sin(2 * np.pi * 0.02 * t / 3600) +  # 0.02 Hz
             20 * np.sin(2 * np.pi * 0.08 * t / 3600))   # 0.08 Hz
    
    z_ulf = (40 * np.sin(2 * np.pi * 0.04 * t / 3600) +  # 0.04 Hz
             25 * np.sin(2 * np.pi * 0.03 * t / 3600) +  # 0.03 Hz
             15 * np.sin(2 * np.pi * 0.07 * t / 3600))   # 0.07 Hz
    
    # Add noise
    h_noise = 10 * np.random.randn(n_samples)
    z_noise = 8 * np.random.randn(n_samples)
    
    # Combine components
    h_data = h_base + h_ulf + h_noise
    z_data = z_base + z_ulf + z_noise
    d_data = 2.0 + 0.5 * np.sin(2 * np.pi * 0.01 * t / 3600) + 0.1 * np.random.randn(n_samples)
    
    # Create time array
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    times = np.array([start_time + np.timedelta64(i, 's') for i in range(n_samples)])
    
    # Compute X, Y components
    d_rad = np.deg2rad(d_data)
    x_data = h_data * np.cos(d_rad)
    y_data = h_data * np.sin(d_rad)
    
    synthetic_data = {
        'H': h_data,
        'D': d_data,
        'Z': z_data,
        'X': x_data,
        'Y': y_data,
        'Time': times,
        'station': 'SYNTHETIC',
        'date': '2023-01-01',
        'n_records': n_samples,
        'filename': 'synthetic_data.dat'
    }
    
    print(f"Generated {n_samples} synthetic data points")
    print(f"H range: {np.min(h_data):.2f} to {np.max(h_data):.2f} nT")
    print(f"Z range: {np.min(z_data):.2f} to {np.max(z_data):.2f} nT")
    
    return synthetic_data


def example_signal_processing(data):
    """Example of signal processing with ULF filters."""
    print("\n=== Signal Processing Example ===")
    
    # Initialize signal processor
    processor = GeomagneticSignalProcessor(sampling_rate=1.0)
    
    # Process components with ULF and PC3 filters
    processed = processor.process_components(
        data['H'], data['D'], data['Z'],
        apply_ulf=True, apply_pc3=True
    )
    
    print("Signal processing completed:")
    print(f"- Raw Z/H ratio std: {np.nanstd(processed['zh_ratio_raw']):.4f}")
    if 'zh_ratio_ulf' in processed:
        print(f"- ULF Z/H ratio std: {np.nanstd(processed['zh_ratio_ulf']):.4f}")
    if 'zh_ratio_pc3' in processed:
        print(f"- PC3 Z/H ratio std: {np.nanstd(processed['zh_ratio_pc3']):.4f}")
    
    # Generate comparison plots
    output_dir = '../outputs/signal_processing'
    os.makedirs(output_dir, exist_ok=True)
    
    station = data.get('station', 'Unknown')
    date_str = data.get('date', 'Unknown')
    
    processor.plot_components_comparison(
        processed,
        title=f"{station} - {date_str} - Signal Processing Comparison",
        save_path=os.path.join(output_dir, f"components_comparison_{station}_{date_str}.png")
    )
    
    processor.plot_zh_ratio_comparison(
        processed,
        title=f"{station} - {date_str} - Z/H Ratio Comparison",
        save_path=os.path.join(output_dir, f"zh_ratio_comparison_{station}_{date_str}.png")
    )
    
    print(f"Signal processing plots saved to: {output_dir}")
    
    return processed


def example_scalogram_analysis(data):
    """Example of scalogram analysis with CWT."""
    print("\n=== Scalogram Analysis Example ===")
    
    # Initialize scalogram processor
    processor = ScalogramProcessor(sampling_rate=1.0, wavelet='morl')
    
    # Process daily data
    output_dir = '../outputs/scalograms'
    os.makedirs(output_dir, exist_ok=True)
    
    station = data.get('station', 'Unknown')
    date_str = data.get('date', 'Unknown').replace('-', '')
    
    results = processor.process_daily_data(
        data['H'], data['Z'],
        output_dir=output_dir,
        station=station,
        date_str=date_str
    )
    
    print("Scalogram analysis completed:")
    print(f"- Scalogram shape: {results['scalogram_data']['zh_ratio_power'].shape}")
    print(f"- Frequency range: {results['scalogram_data']['frequencies'][-1]:.4f} to {results['scalogram_data']['frequencies'][0]:.4f} Hz")
    
    if results['ulf_features']:
        print(f"- ULF features extracted: {len(results['ulf_features'])} features")
        print(f"- ULF frequency range: {results['ulf_features']['ulf_freq_min']:.3f} to {results['ulf_features']['ulf_freq_max']:.3f} Hz")
        
        # Print some statistics
        mean_power = np.mean(results['ulf_features']['ulf_mean_power'])
        max_power = np.max(results['ulf_features']['ulf_max_power'])
        print(f"- Mean ULF power: {mean_power:.2e}")
        print(f"- Maximum ULF power: {max_power:.2e}")
    
    print(f"Scalogram plots saved to: {output_dir}")
    
    return results


def example_complete_workflow():
    """Complete workflow example from data reading to scalogram analysis."""
    print("=" * 60)
    print("SPATIO-TEMPORAL EARTHQUAKE PRECURSOR ANALYSIS")
    print("Complete Preprocessing Workflow Example")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting complete workflow example")
    
    try:
        # Step 1: Read data
        data = example_data_reading()
        
        # Step 2: Signal processing
        processed_data = example_signal_processing(data)
        
        # Step 3: Scalogram analysis
        scalogram_results = example_scalogram_analysis(data)
        
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nOutput files generated:")
        print("- Signal processing plots: ../outputs/signal_processing/")
        print("- Scalogram plots: ../outputs/scalograms/")
        print("- Log file: preprocessing_example.log")
        
        logger.info("Complete workflow example finished successfully")
        
        return {
            'raw_data': data,
            'processed_data': processed_data,
            'scalogram_results': scalogram_results
        }
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        print(f"\nWorkflow failed with error: {e}")
        raise


if __name__ == '__main__':
    # Run complete workflow example
    results = example_complete_workflow()