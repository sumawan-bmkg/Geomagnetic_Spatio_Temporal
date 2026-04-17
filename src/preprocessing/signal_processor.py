"""
Signal Processing Module for Geomagnetic Data
Refactored from awal/signal_processing.py with enhanced ULF frequency processing.
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class GeomagneticSignalProcessor:
    """Process geomagnetic signals with ULF frequency filters."""
    
    def __init__(self, sampling_rate=1.0):
        """
        Initialize processor.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 1.0 for 1-second data)
        """
        self.fs = sampling_rate
        
        # ULF frequency range: 0.01-0.1 Hz (optimized for earthquake precursors)
        self.ulf_low = 0.01   # Hz
        self.ulf_high = 0.1   # Hz
        
        # PC3 pulsation frequency range: 22-100 mHz (0.022-0.1 Hz)
        self.pc3_low = 0.022  # Hz
        self.pc3_high = 0.1   # Hz
    
    def bandpass_filter(self, data, low_freq=None, high_freq=None, order=4):
        """
        Apply bandpass filter to data.
        
        Args:
            data: Input signal array
            low_freq: Low cutoff frequency (default: ULF low)
            high_freq: High cutoff frequency (default: ULF high)
            order: Filter order
            
        Returns:
            Filtered signal
        """
        if low_freq is None:
            low_freq = self.ulf_low
        if high_freq is None:
            high_freq = self.ulf_high
        
        # Remove NaN values
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            logger.warning("All data is NaN, returning zeros")
            return np.zeros_like(data)
        
        # Interpolate NaN values
        data_clean = np.array(data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(data))
            data_clean[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], data[valid_mask])
        
        # Design Butterworth bandpass filter
        nyquist = self.fs / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = max(0.001, min(low, 0.999))
        high = max(low + 0.001, min(high, 0.999))
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            
            # Apply filter (use filtfilt for zero-phase filtering)
            filtered = signal.filtfilt(b, a, data_clean)
            
            logger.info(f"Bandpass filter applied: {low_freq:.3f}-{high_freq:.3f} Hz")
            return filtered
            
        except Exception as e:
            logger.error(f"Filter error: {e}")
            return data_clean
    
    def calculate_zh_ratio(self, z_data, h_data):
        """
        Calculate Z/H ratio for earthquake precursor analysis.
        
        Args:
            z_data: Z component array
            h_data: H component array
            
        Returns:
            Z/H ratio array
        """
        # Avoid division by zero
        h_safe = np.where(np.abs(h_data) < 1e-10, 1e-10, h_data)
        ratio = z_data / h_safe
        
        return ratio
    
    def process_components(self, h_data, d_data, z_data, apply_ulf=True, apply_pc3=False):
        """
        Process all three components with ULF and/or PC3 filters.
        
        Args:
            h_data: H component
            d_data: D component
            z_data: Z component
            apply_ulf: Apply ULF bandpass filter (0.01-0.1 Hz)
            apply_pc3: Apply PC3 bandpass filter (0.022-0.1 Hz)
            
        Returns:
            dict with processed data
        """
        result = {
            'h_raw': h_data,
            'd_raw': d_data,
            'z_raw': z_data
        }
        
        if apply_ulf:
            result['h_ulf'] = self.bandpass_filter(h_data, self.ulf_low, self.ulf_high)
            result['d_ulf'] = self.bandpass_filter(d_data, self.ulf_low, self.ulf_high)
            result['z_ulf'] = self.bandpass_filter(z_data, self.ulf_low, self.ulf_high)
            
            # Calculate Z/H ratio for ULF filtered data
            result['zh_ratio_ulf'] = self.calculate_zh_ratio(
                result['z_ulf'], 
                result['h_ulf']
            )
        
        if apply_pc3:
            result['h_pc3'] = self.bandpass_filter(h_data, self.pc3_low, self.pc3_high)
            result['d_pc3'] = self.bandpass_filter(d_data, self.pc3_low, self.pc3_high)
            result['z_pc3'] = self.bandpass_filter(z_data, self.pc3_low, self.pc3_high)
            
            # Calculate Z/H ratio for PC3 filtered data
            result['zh_ratio_pc3'] = self.calculate_zh_ratio(
                result['z_pc3'], 
                result['h_pc3']
            )
        
        # Calculate Z/H ratio for raw data
        result['zh_ratio_raw'] = self.calculate_zh_ratio(z_data, h_data)
        
        return result
    
    def calculate_power_spectrum(self, data, nperseg=1024):
        """
        Calculate power spectral density.
        
        Args:
            data: Input signal
            nperseg: Length of each segment
            
        Returns:
            frequencies, power spectral density
        """
        # Remove NaN
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return np.array([]), np.array([])
        
        data_clean = np.array(data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(data))
            data_clean[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], data[valid_mask])
        
        # Calculate PSD
        freqs, psd = signal.welch(data_clean, fs=self.fs, nperseg=nperseg)
        
        return freqs, psd
    
    def plot_components_comparison(self, processed_data, title=None, save_path=None):
        """
        Plot comparison of raw, ULF, and PC3 filtered components.
        
        Args:
            processed_data: Dict from process_components()
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        time_axis = np.arange(len(processed_data['h_raw'])) / self.fs / 3600  # Hours
        
        components = [
            ('h_raw', 'h_ulf', 'h_pc3', 'H Component', 'red'),
            ('d_raw', 'd_ulf', 'd_pc3', 'D Component', 'green'),
            ('z_raw', 'z_ulf', 'z_pc3', 'Z Component', 'blue')
        ]
        
        for idx, (raw_key, ulf_key, pc3_key, label, color) in enumerate(components):
            ax = axes[idx]
            
            # Plot raw data (lightest)
            ax.plot(time_axis, processed_data[raw_key], 
                   color=color, alpha=0.3, linewidth=0.5, label='Raw')
            
            # Plot ULF filtered (medium)
            if ulf_key in processed_data:
                ax.plot(time_axis, processed_data[ulf_key], 
                       color=color, alpha=0.7, linewidth=0.8, label='ULF (0.01-0.1 Hz)')
            
            # Plot PC3 filtered (darkest)
            if pc3_key in processed_data:
                ax.plot(time_axis, processed_data[pc3_key], 
                       color=color, linewidth=1.0, label='PC3 (0.022-0.1 Hz)')
            
            ax.set_ylabel(f'{label}\n(nT)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Time (hours)', fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Components comparison plot saved to {save_path}")
        
        return fig
    
    def plot_zh_ratio_comparison(self, processed_data, title=None, save_path=None):
        """
        Plot Z/H ratio comparison for different frequency bands.
        
        Args:
            processed_data: Dict from process_components()
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        
        time_axis = np.arange(len(processed_data['zh_ratio_raw'])) / self.fs / 3600
        
        # Raw Z/H ratio
        axes[0].plot(time_axis, processed_data['zh_ratio_raw'], 
                    color='purple', linewidth=0.5, alpha=0.6, label='Raw Z/H')
        axes[0].set_ylabel('Z/H Ratio\n(Raw)', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        axes[0].legend(loc='upper right', fontsize=8)
        
        # ULF filtered Z/H ratio
        if 'zh_ratio_ulf' in processed_data:
            axes[1].plot(time_axis, processed_data['zh_ratio_ulf'], 
                        color='darkviolet', linewidth=0.8, label='ULF Z/H (0.01-0.1 Hz)')
            axes[1].set_ylabel('Z/H Ratio\n(ULF Filtered)', fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[1].legend(loc='upper right', fontsize=8)
        
        # PC3 filtered Z/H ratio
        if 'zh_ratio_pc3' in processed_data:
            axes[2].plot(time_axis, processed_data['zh_ratio_pc3'], 
                        color='indigo', linewidth=1.0, label='PC3 Z/H (0.022-0.1 Hz)')
            axes[2].set_ylabel('Z/H Ratio\n(PC3 Filtered)', fontsize=10)
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[2].legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Time (hours)', fontsize=10)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Z/H ratio comparison plot saved to {save_path}")
        
        return fig


if __name__ == '__main__':
    # Test signal processor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test signal with ULF components
    t = np.linspace(0, 86400, 86400)  # 1 day, 1 Hz
    
    # Simulate geomagnetic data with ULF and PC3 pulsations
    h_test = 40000 + 100 * np.sin(2 * np.pi * 0.05 * t) + 20 * np.random.randn(len(t))
    d_test = 50 * np.cos(2 * np.pi * 0.03 * t) + 10 * np.random.randn(len(t))
    z_test = 30000 + 80 * np.sin(2 * np.pi * 0.04 * t) + 15 * np.random.randn(len(t))
    
    processor = GeomagneticSignalProcessor()
    
    # Process with both ULF and PC3 filters
    result = processor.process_components(h_test, d_test, z_test, 
                                        apply_ulf=True, apply_pc3=True)
    
    # Plot comparisons
    processor.plot_components_comparison(result, 
                                       title='Test Signal - ULF vs PC3 Filters', 
                                       save_path='test_components_comparison.png')
    processor.plot_zh_ratio_comparison(result, 
                                     title='Test Signal - Z/H Ratio Comparison', 
                                     save_path='test_zh_ratio_comparison.png')
    
    print("Test plots generated!")