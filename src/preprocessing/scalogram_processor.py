"""
Scalogram Processor Module
Continuous Wavelet Transform (CWT) analysis for geomagnetic data with focus on ULF frequencies.
Generates Z/H ratio scalograms for earthquake precursor detection.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pywt
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional, Union

logger = logging.getLogger(__name__)


class ScalogramProcessor:
    """
    Process geomagnetic data using Continuous Wavelet Transform (CWT) 
    with focus on ULF frequencies (0.01-0.1 Hz) for earthquake precursor analysis.
    """
    
    def __init__(self, sampling_rate: float = 1.0, wavelet: str = 'morl'):
        """
        Initialize ScalogramProcessor.
        
        Args:
            sampling_rate: Sampling rate in Hz (default: 1.0 for 1-second data)
            wavelet: Wavelet type for CWT (default: 'morl' - Morlet wavelet)
        """
        self.fs = sampling_rate
        self.wavelet = wavelet
        
        # ULF frequency range for earthquake precursors
        self.ulf_freq_min = 0.01   # Hz (100 seconds period)
        self.ulf_freq_max = 0.1    # Hz (10 seconds period)
        
        # Default frequency range for scalogram
        self.freq_min = 0.005      # Hz (200 seconds period)
        self.freq_max = 0.2        # Hz (5 seconds period)
        
        # Number of scales for CWT
        self.n_scales = 50
        
    def _generate_scales(self, freq_min: float = None, freq_max: float = None) -> np.ndarray:
        """
        Generate scales for CWT based on frequency range.
        
        Args:
            freq_min: Minimum frequency (Hz)
            freq_max: Maximum frequency (Hz)
            
        Returns:
            Array of scales for CWT
        """
        if freq_min is None:
            freq_min = self.freq_min
        if freq_max is None:
            freq_max = self.freq_max
            
        # Convert frequencies to periods
        period_min = 1.0 / freq_max
        period_max = 1.0 / freq_min
        
        # Generate logarithmically spaced periods
        periods = np.logspace(np.log10(period_min), np.log10(period_max), self.n_scales)
        
        # Convert periods to scales (depends on wavelet)
        if self.wavelet == 'morl':
            # For Morlet wavelet, scale ≈ period / (2π)
            scales = periods / (2 * np.pi)
        else:
            # General approximation
            scales = periods / 2.0
            
        return scales
    
    def _frequencies_from_scales(self, scales: np.ndarray) -> np.ndarray:
        """
        Convert scales to frequencies.
        
        Args:
            scales: Array of scales
            
        Returns:
            Array of frequencies (Hz)
        """
        if self.wavelet == 'morl':
            # For Morlet wavelet
            frequencies = 1.0 / (2 * np.pi * scales * self.fs)
        else:
            # General approximation
            frequencies = 1.0 / (2.0 * scales * self.fs)
            
        return frequencies
    
    def compute_cwt(self, data: np.ndarray, scales: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Continuous Wavelet Transform.
        
        Args:
            data: Input signal
            scales: Scales for CWT (optional)
            
        Returns:
            Tuple of (coefficients, frequencies)
        """
        if scales is None:
            scales = self._generate_scales()
            
        # Remove NaN values and interpolate
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            logger.warning("All data is NaN, returning zeros")
            return np.zeros((len(scales), len(data))), self._frequencies_from_scales(scales)
        
        data_clean = np.array(data, dtype=float)
        if np.any(~valid_mask):
            x = np.arange(len(data))
            data_clean[~valid_mask] = np.interp(x[~valid_mask], x[valid_mask], data[valid_mask])
        
        # Compute CWT
        try:
            coefficients, _ = pywt.cwt(data_clean, scales, self.wavelet, sampling_period=1.0/self.fs)
            frequencies = self._frequencies_from_scales(scales)
            
            logger.info(f"CWT computed: {len(scales)} scales, frequency range: {frequencies[-1]:.4f}-{frequencies[0]:.4f} Hz")
            
            return coefficients, frequencies
            
        except Exception as e:
            logger.error(f"CWT computation error: {e}")
            return np.zeros((len(scales), len(data))), self._frequencies_from_scales(scales)
    
    def compute_scalogram_power(self, coefficients: np.ndarray) -> np.ndarray:
        """
        Compute scalogram power from CWT coefficients.
        
        Args:
            coefficients: CWT coefficients (complex)
            
        Returns:
            Power scalogram (real, positive)
        """
        return np.abs(coefficients) ** 2
    
    def compute_zh_ratio_scalogram(self, z_data: np.ndarray, h_data: np.ndarray, 
                                  scales: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Compute Z/H ratio scalogram for earthquake precursor analysis.
        
        Args:
            z_data: Z component data
            h_data: H component data
            scales: Scales for CWT (optional)
            
        Returns:
            Dictionary containing Z, H, and Z/H ratio scalograms
        """
        if scales is None:
            scales = self._generate_scales()
        
        # Compute CWT for both components
        z_coeffs, frequencies = self.compute_cwt(z_data, scales)
        h_coeffs, _ = self.compute_cwt(h_data, scales)
        
        # Compute power scalograms
        z_power = self.compute_scalogram_power(z_coeffs)
        h_power = self.compute_scalogram_power(h_coeffs)
        
        # Compute Z/H ratio scalogram
        # Avoid division by zero
        h_power_safe = np.where(h_power < 1e-10, 1e-10, h_power)
        zh_ratio_power = z_power / h_power_safe
        
        # Also compute complex ratio for phase information
        h_coeffs_safe = np.where(np.abs(h_coeffs) < 1e-10, 1e-10, h_coeffs)
        zh_ratio_complex = z_coeffs / h_coeffs_safe
        zh_ratio_phase = np.angle(zh_ratio_complex)
        
        return {
            'z_power': z_power,
            'h_power': h_power,
            'zh_ratio_power': zh_ratio_power,
            'zh_ratio_phase': zh_ratio_phase,
            'frequencies': frequencies,
            'scales': scales,
            'z_coeffs': z_coeffs,
            'h_coeffs': h_coeffs,
            'zh_ratio_complex': zh_ratio_complex
        }
    
    def extract_ulf_features(self, scalogram_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract ULF frequency band features from scalogram.
        
        Args:
            scalogram_data: Output from compute_zh_ratio_scalogram()
            
        Returns:
            Dictionary of ULF features
        """
        frequencies = scalogram_data['frequencies']
        zh_ratio_power = scalogram_data['zh_ratio_power']
        
        # Find ULF frequency indices
        ulf_mask = (frequencies >= self.ulf_freq_min) & (frequencies <= self.ulf_freq_max)
        ulf_indices = np.where(ulf_mask)[0]
        
        if len(ulf_indices) == 0:
            logger.warning("No ULF frequencies found in scalogram")
            return {}
        
        # Extract ULF band data
        ulf_frequencies = frequencies[ulf_indices]
        ulf_zh_power = zh_ratio_power[ulf_indices, :]
        
        # Compute time-series features
        ulf_mean_power = np.mean(ulf_zh_power, axis=0)  # Average across frequencies
        ulf_max_power = np.max(ulf_zh_power, axis=0)    # Maximum across frequencies
        # Use scipy.integrate.trapezoid for newer NumPy versions, fallback to np.trapz
        try:
            from scipy.integrate import trapezoid
            ulf_integrated_power = trapezoid(ulf_zh_power, ulf_frequencies, axis=0)  # Integrated power
        except ImportError:
            # Fallback for older scipy versions
            try:
                ulf_integrated_power = np.trapz(ulf_zh_power, ulf_frequencies, axis=0)
            except AttributeError:
                # For very new NumPy versions where trapz is removed
                from scipy.integrate import simpson
                ulf_integrated_power = simpson(ulf_zh_power, ulf_frequencies, axis=0)
        
        # Compute frequency-averaged features
        ulf_temporal_mean = np.mean(ulf_zh_power, axis=1)  # Average across time
        ulf_temporal_std = np.std(ulf_zh_power, axis=1)    # Std across time
        
        return {
            'ulf_frequencies': ulf_frequencies,
            'ulf_zh_power': ulf_zh_power,
            'ulf_mean_power': ulf_mean_power,
            'ulf_max_power': ulf_max_power,
            'ulf_integrated_power': ulf_integrated_power,
            'ulf_temporal_mean': ulf_temporal_mean,
            'ulf_temporal_std': ulf_temporal_std,
            'ulf_freq_min': self.ulf_freq_min,
            'ulf_freq_max': self.ulf_freq_max
        }
    
    def plot_scalogram(self, scalogram_data: Dict[str, np.ndarray], 
                      component: str = 'zh_ratio_power',
                      time_hours: np.ndarray = None,
                      title: str = None, save_path: str = None,
                      vmin: float = None, vmax: float = None) -> plt.Figure:
        """
        Plot scalogram with proper time and frequency axes.
        
        Args:
            scalogram_data: Output from compute_zh_ratio_scalogram()
            component: Component to plot ('z_power', 'h_power', 'zh_ratio_power')
            time_hours: Time axis in hours (optional)
            title: Plot title
            save_path: Path to save figure
            vmin, vmax: Color scale limits
            
        Returns:
            matplotlib Figure
        """
        if component not in scalogram_data:
            raise ValueError(f"Component '{component}' not found in scalogram data")
        
        data = scalogram_data[component]
        frequencies = scalogram_data['frequencies']
        
        if time_hours is None:
            time_hours = np.arange(data.shape[1]) / 3600.0  # Convert seconds to hours
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot scalogram
        if vmin is None or vmax is None:
            # Use percentiles for robust color scaling
            vmin = np.percentile(data, 5)
            vmax = np.percentile(data, 95)
        
        # Use log scale for better visualization
        data_log = np.log10(np.maximum(data, vmin))
        vmin_log = np.log10(vmin)
        vmax_log = np.log10(vmax)
        
        im = ax.pcolormesh(time_hours, frequencies, data_log, 
                          shading='auto', cmap='jet',
                          vmin=vmin_log, vmax=vmax_log)
        
        # Set frequency axis to log scale
        ax.set_yscale('log')
        ax.set_ylim(frequencies[-1], frequencies[0])  # Reverse for conventional display
        
        # Highlight ULF frequency band
        ax.axhline(y=self.ulf_freq_min, color='white', linestyle='--', 
                  linewidth=2, alpha=0.8, label=f'ULF Band ({self.ulf_freq_min}-{self.ulf_freq_max} Hz)')
        ax.axhline(y=self.ulf_freq_max, color='white', linestyle='--', 
                  linewidth=2, alpha=0.8)
        
        # Labels and formatting
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_xlim(time_hours[0], time_hours[-1])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        if component == 'zh_ratio_power':
            cbar.set_label('log₁₀(Z/H Power Ratio)', fontsize=11, fontweight='bold')
        elif component == 'z_power':
            cbar.set_label('log₁₀(Z Power)', fontsize=11, fontweight='bold')
        elif component == 'h_power':
            cbar.set_label('log₁₀(H Power)', fontsize=11, fontweight='bold')
        
        # Title
        if title is None:
            if component == 'zh_ratio_power':
                title = 'Z/H Ratio Scalogram (CWT Power)'
            elif component == 'z_power':
                title = 'Z Component Scalogram (CWT Power)'
            elif component == 'h_power':
                title = 'H Component Scalogram (CWT Power)'
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Scalogram saved to {save_path}")
        
        return fig
    
    def plot_ulf_features(self, ulf_features: Dict[str, np.ndarray],
                         time_hours: np.ndarray = None,
                         title: str = None, save_path: str = None) -> plt.Figure:
        """
        Plot ULF frequency band features.
        
        Args:
            ulf_features: Output from extract_ulf_features()
            time_hours: Time axis in hours
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure
        """
        if not ulf_features:
            raise ValueError("No ULF features provided")
        
        if time_hours is None:
            time_hours = np.arange(len(ulf_features['ulf_mean_power'])) / 3600.0
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: ULF power time series
        axes[0].plot(time_hours, ulf_features['ulf_mean_power'], 
                    color='blue', linewidth=1.0, label='Mean ULF Power')
        axes[0].plot(time_hours, ulf_features['ulf_max_power'], 
                    color='red', linewidth=1.0, alpha=0.7, label='Max ULF Power')
        axes[0].set_ylabel('Z/H Power Ratio\n(ULF Band)', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].set_yscale('log')
        
        # Plot 2: Integrated ULF power
        axes[1].plot(time_hours, ulf_features['ulf_integrated_power'], 
                    color='green', linewidth=1.0, label='Integrated ULF Power')
        axes[1].set_ylabel('Integrated Power\n(ULF Band)', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].set_yscale('log')
        
        # Plot 3: ULF frequency spectrum (time-averaged)
        axes[2].semilogx(ulf_features['ulf_frequencies'], ulf_features['ulf_temporal_mean'], 
                        color='purple', linewidth=1.5, label='Time-averaged ULF Spectrum')
        axes[2].fill_between(ulf_features['ulf_frequencies'], 
                           ulf_features['ulf_temporal_mean'] - ulf_features['ulf_temporal_std'],
                           ulf_features['ulf_temporal_mean'] + ulf_features['ulf_temporal_std'],
                           alpha=0.3, color='purple', label='±1σ')
        axes[2].set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Mean Z/H Power', fontsize=11, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='upper right', fontsize=10)
        axes[2].set_xlim(ulf_features['ulf_freq_min'], ulf_features['ulf_freq_max'])
        
        if title is None:
            title = f'ULF Features ({ulf_features["ulf_freq_min"]:.3f}-{ulf_features["ulf_freq_max"]:.3f} Hz)'
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ULF features plot saved to {save_path}")
        
        return fig
    
    def process_daily_data(self, h_data: np.ndarray, z_data: np.ndarray,
                          output_dir: str = None, station: str = 'Unknown',
                          date_str: str = None) -> Dict[str, np.ndarray]:
        """
        Complete processing pipeline for daily geomagnetic data.
        
        Args:
            h_data: H component data
            z_data: Z component data
            output_dir: Directory to save plots (optional)
            station: Station name for plots
            date_str: Date string for plots
            
        Returns:
            Dictionary containing all scalogram results and ULF features
        """
        logger.info(f"Processing daily data: {len(h_data)} samples")
        
        # Compute scalograms
        scalogram_data = self.compute_zh_ratio_scalogram(z_data, h_data)
        
        # Extract ULF features
        ulf_features = self.extract_ulf_features(scalogram_data)
        
        # Create time axis
        time_hours = np.arange(len(h_data)) / 3600.0
        
        # Generate plots if output directory specified
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            plot_title_base = f"{station} - {date_str}" if date_str else station
            
            # Plot Z/H ratio scalogram
            self.plot_scalogram(scalogram_data, 'zh_ratio_power', time_hours,
                              title=f"{plot_title_base} - Z/H Ratio Scalogram",
                              save_path=os.path.join(output_dir, f"scalogram_zh_ratio_{station}_{date_str}.png"))
            
            # Plot ULF features
            if ulf_features:
                self.plot_ulf_features(ulf_features, time_hours,
                                     title=f"{plot_title_base} - ULF Features",
                                     save_path=os.path.join(output_dir, f"ulf_features_{station}_{date_str}.png"))
        
        # Combine results
        results = {
            'scalogram_data': scalogram_data,
            'ulf_features': ulf_features,
            'time_hours': time_hours,
            'processing_params': {
                'sampling_rate': self.fs,
                'wavelet': self.wavelet,
                'ulf_freq_range': (self.ulf_freq_min, self.ulf_freq_max),
                'n_scales': self.n_scales
            }
        }
        
        logger.info("Daily data processing completed")
        return results


if __name__ == '__main__':
    # Test scalogram processor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test signal with ULF components
    t = np.linspace(0, 86400, 86400)  # 1 day, 1 Hz sampling
    
    # Simulate geomagnetic data with ULF pulsations
    h_test = 40000 + 100 * np.sin(2 * np.pi * 0.05 * t) + 50 * np.sin(2 * np.pi * 0.02 * t) + 20 * np.random.randn(len(t))
    z_test = 30000 + 80 * np.sin(2 * np.pi * 0.04 * t) + 60 * np.sin(2 * np.pi * 0.03 * t) + 15 * np.random.randn(len(t))
    
    # Initialize processor
    processor = ScalogramProcessor(sampling_rate=1.0, wavelet='morl')
    
    # Process data
    results = processor.process_daily_data(h_test, z_test, 
                                         output_dir='test_output',
                                         station='TEST', 
                                         date_str='20240101')
    
    print("Test scalogram processing completed!")
    print(f"ULF features extracted: {len(results['ulf_features'])} features")
    print(f"Scalogram shape: {results['scalogram_data']['zh_ratio_power'].shape}")