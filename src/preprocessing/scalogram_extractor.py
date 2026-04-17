"""
CWT Scalogram Extractor Module
Extracted and refactored from ScalogramProcessor for multi-station architecture.
Focuses on core CWT functionality with corrected frequency ranges for geomagnetic ULF/Pc3-Pc4 analysis.
"""
import numpy as np
import pywt
import logging
from typing import Tuple, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CWTResult:
    """
    Container for CWT computation results.
    
    Attributes:
        coefficients: Complex CWT coefficients (F, T)
        frequencies: Frequency array (F,) in Hz
        scales: Scale array (F,) used for CWT
        power: Power scalogram |coefficients|² (F, T)
        sampling_rate: Original sampling rate in Hz
        wavelet: Wavelet type used
    """
    coefficients: np.ndarray
    frequencies: np.ndarray
    scales: np.ndarray
    power: np.ndarray
    sampling_rate: float
    wavelet: str
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return (F, T) shape of the scalogram."""
        return self.coefficients.shape
    
    @property
    def frequency_range(self) -> Tuple[float, float]:
        """Return (min_freq, max_freq) in Hz."""
        return (self.frequencies.min(), self.frequencies.max())


class CWTScalogramExtractor:
    """
    Continuous Wavelet Transform untuk ekstraksi scalogram geomagnetik ULF/Pc3-Pc4.
    
    KOREKSI FREKUENSI: Rentang dikunci pada 0.01–0.5 Hz (ULF/Pc3-Pc4),
    bukan 0.5-25 Hz (seismik broadband). Sampling rate BMKG = 1 Hz,
    sehingga Nyquist = 0.5 Hz adalah batas fisik maksimum.
    
    Band targets:
    - ULF:  0.01–0.05 Hz (10–50 mHz)  — resonansi litosfer pra-gempa
    - Pc3:  0.022–0.1 Hz (22–100 mHz) — pulsasi geomagnetik Pc3
    - Pc4:  0.007–0.022 Hz (7–22 mHz) — pulsasi geomagnetik Pc4
    
    Refactored from ScalogramProcessor for multi-station architecture compatibility.
    """
    
    def __init__(
        self,
        wavelet: str = 'morl',
        freq_range: Tuple[float, float] = (0.01, 0.5),   # Hz — ULF/Pc3-Pc4, BUKAN seismik broadband
        n_freqs: int = 64,
        sampling_rate: float = 1.0          # Hz — BMKG geomagnetic data
    ):
        """
        Initialize CWT Scalogram Extractor.
        
        Args:
            wavelet: Wavelet type ('morl' for Morlet, optimal for geomagnetic ULF)
            freq_range: (min_freq, max_freq) in Hz. Default (0.01, 0.5) for ULF/Pc3-Pc4
            n_freqs: Number of frequency bins (logarithmic spacing)
            sampling_rate: Sampling rate in Hz (BMKG standard: 1 Hz)
        
        Note:
            Frequency range corrected from seismic broadband (0.5-25 Hz) to 
            geomagnetic ULF/Pc3-Pc4 (0.01-0.5 Hz) based on physical requirements.
        """
        self.wavelet = wavelet
        self.freq_range = freq_range
        self.n_freqs = n_freqs
        self.sampling_rate = sampling_rate
        
        # Validate frequency range against Nyquist limit
        nyquist_freq = sampling_rate / 2.0
        if freq_range[1] > nyquist_freq:
            logger.warning(
                f"Maximum frequency {freq_range[1]} Hz exceeds Nyquist frequency "
                f"{nyquist_freq} Hz. Clamping to Nyquist limit."
            )
            self.freq_range = (freq_range[0], nyquist_freq)
        
        # Pre-compute scales for efficiency
        self._scales = self._generate_scales()
        self._frequencies = self._frequencies_from_scales(self._scales)
        
        logger.info(
            f"CWTScalogramExtractor initialized: {wavelet} wavelet, "
            f"freq_range={self.freq_range} Hz, n_freqs={n_freqs}, fs={sampling_rate} Hz"
        )
    
    def _generate_scales(self) -> np.ndarray:
        """
        Generate scales for CWT based on frequency range.
        Uses logarithmic spacing for better frequency resolution at low frequencies.
        
        Returns:
            Array of scales for CWT computation
        """
        freq_min, freq_max = self.freq_range
        
        # Convert frequencies to periods
        period_min = 1.0 / freq_max
        period_max = 1.0 / freq_min
        
        # Generate logarithmically spaced periods
        periods = np.logspace(
            np.log10(period_min), 
            np.log10(period_max), 
            self.n_freqs
        )
        
        # Convert periods to scales (wavelet-dependent)
        if self.wavelet == 'morl':
            # For Morlet wavelet: scale ≈ period / (2π)
            scales = periods / (2 * np.pi)
        else:
            # General approximation for other wavelets
            scales = periods / 2.0
        
        return scales
    
    def _frequencies_from_scales(self, scales: np.ndarray) -> np.ndarray:
        """
        Convert scales to frequencies.
        
        Args:
            scales: Array of scales
            
        Returns:
            Array of frequencies in Hz
        """
        if self.wavelet == 'morl':
            # For Morlet wavelet
            frequencies = 1.0 / (2 * np.pi * scales * (1.0 / self.sampling_rate))
        else:
            # General approximation
            frequencies = 1.0 / (2.0 * scales * (1.0 / self.sampling_rate))
        
        return frequencies
    
    def extract(self, waveform: np.ndarray) -> CWTResult:
        """
        Extract CWT scalogram from input waveform.
        
        Args:
            waveform: Input signal (T,) - single channel geomagnetic trace
                     (H, D, or Z component)
        
        Returns:
            CWTResult containing coefficients, frequencies, scales, and power scalogram
            
        Raises:
            ValueError: If waveform is empty or all NaN
            RuntimeError: If CWT computation fails
        """
        if len(waveform) == 0:
            raise ValueError("Input waveform is empty")
        
        # Handle NaN values
        valid_mask = ~np.isnan(waveform)
        if not np.any(valid_mask):
            raise ValueError("All waveform values are NaN")
        
        # Clean and interpolate missing data
        waveform_clean = self._clean_waveform(waveform, valid_mask)
        
        try:
            # Compute CWT using pre-computed scales
            coefficients, _ = pywt.cwt(
                waveform_clean, 
                self._scales, 
                self.wavelet, 
                sampling_period=1.0 / self.sampling_rate
            )
            
            # Compute power scalogram
            power = np.abs(coefficients) ** 2
            
            # Normalize power to [0, 1] range for consistency
            power_normalized = self._normalize_power(power)
            
            logger.debug(
                f"CWT extracted: shape={coefficients.shape}, "
                f"freq_range=({self._frequencies[-1]:.4f}, {self._frequencies[0]:.4f}) Hz"
            )
            
            return CWTResult(
                coefficients=coefficients,
                frequencies=self._frequencies,
                scales=self._scales,
                power=power_normalized,
                sampling_rate=self.sampling_rate,
                wavelet=self.wavelet
            )
            
        except Exception as e:
            logger.error(f"CWT computation failed: {e}")
            raise RuntimeError(f"CWT computation failed: {e}") from e
    
    def _clean_waveform(self, waveform: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Clean waveform by interpolating NaN values.
        
        Args:
            waveform: Original waveform with potential NaN values
            valid_mask: Boolean mask indicating valid (non-NaN) values
            
        Returns:
            Cleaned waveform with interpolated values
        """
        waveform_clean = np.array(waveform, dtype=float)
        
        if np.any(~valid_mask):
            # Interpolate missing values
            x = np.arange(len(waveform))
            waveform_clean[~valid_mask] = np.interp(
                x[~valid_mask], 
                x[valid_mask], 
                waveform[valid_mask]
            )
            
            n_interpolated = np.sum(~valid_mask)
            logger.debug(f"Interpolated {n_interpolated} NaN values in waveform")
        
        return waveform_clean
    
    def _normalize_power(self, power: np.ndarray) -> np.ndarray:
        """
        Normalize power scalogram to [0, 1] range.
        
        Args:
            power: Raw power scalogram
            
        Returns:
            Normalized power scalogram
        """
        # Use robust normalization to handle outliers
        p_min = np.percentile(power, 1)   # 1st percentile
        p_max = np.percentile(power, 99)  # 99th percentile
        
        if p_max > p_min:
            power_normalized = (power - p_min) / (p_max - p_min)
            power_normalized = np.clip(power_normalized, 0, 1)
        else:
            # Handle edge case where all values are similar
            power_normalized = np.zeros_like(power)
        
        return power_normalized
    
    def extract_multi_channel(
        self, 
        waveforms: np.ndarray
    ) -> Tuple[CWTResult, ...]:
        """
        Extract CWT scalograms from multiple channels simultaneously.
        
        Args:
            waveforms: Multi-channel waveforms (C, T) where C is number of channels
                      For geomagnetic: C=3 for (H, D, Z) components
        
        Returns:
            Tuple of CWTResult objects, one per channel
            
        Note:
            This method processes each channel independently, maintaining
            compatibility with the multi-station tensor architecture.
        """
        if waveforms.ndim != 2:
            raise ValueError(f"Expected 2D array (C, T), got shape {waveforms.shape}")
        
        n_channels, n_samples = waveforms.shape
        results = []
        
        for ch_idx in range(n_channels):
            try:
                result = self.extract(waveforms[ch_idx])
                results.append(result)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Failed to extract CWT for channel {ch_idx}: {e}")
                # Create dummy result for failed channel
                dummy_result = self._create_dummy_result(n_samples)
                results.append(dummy_result)
        
        return tuple(results)
    
    def _create_dummy_result(self, n_samples: int) -> CWTResult:
        """
        Create dummy CWT result for failed extractions.
        
        Args:
            n_samples: Number of time samples
            
        Returns:
            CWTResult with zero coefficients
        """
        dummy_coeffs = np.zeros((len(self._scales), n_samples), dtype=complex)
        dummy_power = np.zeros((len(self._scales), n_samples))
        
        return CWTResult(
            coefficients=dummy_coeffs,
            frequencies=self._frequencies,
            scales=self._scales,
            power=dummy_power,
            sampling_rate=self.sampling_rate,
            wavelet=self.wavelet
        )
    
    def get_ulf_band_indices(
        self, 
        ulf_range: Tuple[float, float] = (0.01, 0.1)
    ) -> np.ndarray:
        """
        Get frequency indices corresponding to ULF band.
        
        Args:
            ulf_range: (min_freq, max_freq) for ULF band in Hz
                      Default (0.01, 0.1) covers ULF + Pc3 bands
        
        Returns:
            Array of frequency indices within ULF range
        """
        ulf_min, ulf_max = ulf_range
        ulf_mask = (self._frequencies >= ulf_min) & (self._frequencies <= ulf_max)
        return np.where(ulf_mask)[0]
    
    def extract_ulf_power(
        self, 
        cwt_result: CWTResult,
        ulf_range: Tuple[float, float] = (0.01, 0.1)
    ) -> np.ndarray:
        """
        Extract ULF band power time series from CWT result.
        
        Args:
            cwt_result: CWT result from extract() method
            ulf_range: (min_freq, max_freq) for ULF band in Hz
        
        Returns:
            ULF power time series (T,) - mean power across ULF frequencies
        """
        ulf_indices = self.get_ulf_band_indices(ulf_range)
        
        if len(ulf_indices) == 0:
            logger.warning(f"No frequencies found in ULF range {ulf_range}")
            return np.zeros(cwt_result.shape[1])
        
        # Extract ULF band and compute mean across frequencies
        ulf_power = cwt_result.power[ulf_indices, :]
        ulf_mean_power = np.mean(ulf_power, axis=0)
        
        return ulf_mean_power
    
    @property
    def frequency_resolution(self) -> float:
        """Get average frequency resolution in Hz."""
        if len(self._frequencies) > 1:
            # For logarithmic spacing, resolution varies
            return np.mean(np.diff(self._frequencies))
        return 0.0
    
    @property
    def scales(self) -> np.ndarray:
        """Get pre-computed scales array."""
        return self._scales.copy()
    
    @property
    def frequencies(self) -> np.ndarray:
        """Get pre-computed frequencies array."""
        return self._frequencies.copy()
    
    def __repr__(self) -> str:
        return (
            f"CWTScalogramExtractor("
            f"wavelet='{self.wavelet}', "
            f"freq_range={self.freq_range}, "
            f"n_freqs={self.n_freqs}, "
            f"sampling_rate={self.sampling_rate})"
        )


# Backward compatibility alias
ScalogramExtractor = CWTScalogramExtractor


if __name__ == '__main__':
    # Test the CWT scalogram extractor
    import matplotlib.pyplot as plt
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate test geomagnetic signal with ULF components
    t = np.linspace(0, 3600, 3600)  # 1 hour, 1 Hz sampling
    
    # Simulate geomagnetic H component with ULF pulsations
    h_signal = (
        40000 +  # DC offset
        100 * np.sin(2 * np.pi * 0.05 * t) +  # 50 mHz (ULF)
        50 * np.sin(2 * np.pi * 0.02 * t) +   # 20 mHz (Pc4)
        30 * np.sin(2 * np.pi * 0.08 * t) +   # 80 mHz (Pc3)
        20 * np.random.randn(len(t))           # Noise
    )
    
    # Initialize extractor with ULF/Pc3-Pc4 frequency range
    extractor = CWTScalogramExtractor(
        wavelet='morl',
        freq_range=(0.01, 0.5),  # ULF/Pc3-Pc4 range
        n_freqs=64,
        sampling_rate=1.0
    )
    
    print(f"Extractor: {extractor}")
    print(f"Frequency range: {extractor.frequency_range} Hz")
    print(f"Frequency resolution: {extractor.frequency_resolution:.6f} Hz")
    
    # Extract CWT scalogram
    cwt_result = extractor.extract(h_signal)
    
    print(f"CWT result shape: {cwt_result.shape}")
    print(f"Power range: [{cwt_result.power.min():.3f}, {cwt_result.power.max():.3f}]")
    
    # Extract ULF power time series
    ulf_power = extractor.extract_ulf_power(cwt_result, ulf_range=(0.01, 0.1))
    
    print(f"ULF power time series shape: {ulf_power.shape}")
    print(f"ULF power range: [{ulf_power.min():.6f}, {ulf_power.max():.6f}]")
    
    # Test multi-channel extraction
    multi_channel_data = np.array([h_signal, h_signal * 0.8, h_signal * 1.2])  # 3 channels
    multi_results = extractor.extract_multi_channel(multi_channel_data)
    
    print(f"Multi-channel results: {len(multi_results)} channels")
    print(f"Channel shapes: {[result.shape for result in multi_results]}")
    
    print("CWT Scalogram Extractor test completed successfully!")