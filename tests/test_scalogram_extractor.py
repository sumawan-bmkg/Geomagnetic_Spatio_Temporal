"""
Unit tests for CWTScalogramExtractor module.
Tests core CWT functionality, frequency range validation, and multi-channel processing.
"""
import pytest
import numpy as np
import logging
from unittest.mock import patch

# Import the module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.scalogram_extractor import CWTScalogramExtractor, CWTResult


class TestCWTScalogramExtractor:
    """Test suite for CWTScalogramExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = CWTScalogramExtractor(
            wavelet='morl',
            freq_range=(0.01, 0.5),
            n_freqs=32,  # Smaller for faster tests
            sampling_rate=1.0
        )
        
        # Generate test signal
        self.t = np.linspace(0, 1000, 1000)  # 1000 seconds, 1 Hz
        self.test_signal = (
            100 * np.sin(2 * np.pi * 0.05 * self.t) +  # 50 mHz
            50 * np.sin(2 * np.pi * 0.02 * self.t) +   # 20 mHz
            10 * np.random.randn(len(self.t))           # Noise
        )
    
    def test_initialization(self):
        """Test extractor initialization and parameter validation."""
        # Test default initialization
        extractor = CWTScalogramExtractor()
        assert extractor.wavelet == 'morl'
        assert extractor.freq_range == (0.01, 0.5)
        assert extractor.n_freqs == 64
        assert extractor.sampling_rate == 1.0
        
        # Test custom parameters
        extractor_custom = CWTScalogramExtractor(
            wavelet='mexh',
            freq_range=(0.005, 0.2),
            n_freqs=128,
            sampling_rate=2.0
        )
        assert extractor_custom.wavelet == 'mexh'
        assert extractor_custom.freq_range == (0.005, 0.2)
        assert extractor_custom.n_freqs == 128
        assert extractor_custom.sampling_rate == 2.0
    
    def test_nyquist_frequency_validation(self):
        """Test validation against Nyquist frequency limit."""
        # Test frequency range exceeding Nyquist limit
        with patch('preprocessing.scalogram_extractor.logger') as mock_logger:
            extractor = CWTScalogramExtractor(
                freq_range=(0.01, 1.0),  # Max freq = Nyquist for 1 Hz sampling
                sampling_rate=1.0
            )
            # Should clamp to Nyquist frequency (0.5 Hz)
            assert extractor.freq_range[1] == 0.5
            mock_logger.warning.assert_called_once()
    
    def test_scale_generation(self):
        """Test scale generation for different wavelets."""
        # Test Morlet wavelet scales
        scales = self.extractor._generate_scales()
        assert len(scales) == self.extractor.n_freqs
        assert np.all(scales > 0)
        assert np.all(np.diff(scales) > 0)  # Should be increasing
        
        # Test other wavelet
        extractor_mexh = CWTScalogramExtractor(wavelet='mexh', n_freqs=16)
        scales_mexh = extractor_mexh._generate_scales()
        assert len(scales_mexh) == 16
        assert np.all(scales_mexh > 0)
    
    def test_frequency_conversion(self):
        """Test conversion from scales to frequencies."""
        scales = self.extractor._scales
        frequencies = self.extractor._frequencies
        
        # Check frequency array properties
        assert len(frequencies) == len(scales)
        assert np.all(frequencies > 0)
        assert frequencies[0] > frequencies[-1]  # Should be decreasing (high to low freq)
        
        # Check frequency range
        freq_min, freq_max = self.extractor.freq_range
        assert frequencies[-1] >= freq_min * 0.9  # Allow small tolerance
        assert frequencies[0] <= freq_max * 1.1
    
    def test_extract_basic(self):
        """Test basic CWT extraction functionality."""
        result = self.extractor.extract(self.test_signal)
        
        # Check result type and structure
        assert isinstance(result, CWTResult)
        assert result.coefficients.dtype == complex
        assert result.power.dtype == float
        
        # Check shapes
        expected_shape = (self.extractor.n_freqs, len(self.test_signal))
        assert result.coefficients.shape == expected_shape
        assert result.power.shape == expected_shape
        assert len(result.frequencies) == self.extractor.n_freqs
        assert len(result.scales) == self.extractor.n_freqs
        
        # Check metadata
        assert result.sampling_rate == self.extractor.sampling_rate
        assert result.wavelet == self.extractor.wavelet
    
    def test_extract_with_nan_values(self):
        """Test CWT extraction with NaN values in input."""
        # Create signal with NaN values
        signal_with_nan = self.test_signal.copy()
        signal_with_nan[100:110] = np.nan  # 10 NaN values
        
        result = self.extractor.extract(signal_with_nan)
        
        # Should succeed and interpolate NaN values
        assert isinstance(result, CWTResult)
        assert not np.any(np.isnan(result.coefficients))
        assert not np.any(np.isnan(result.power))
    
    def test_extract_edge_cases(self):
        """Test edge cases for CWT extraction."""
        # Test empty array
        with pytest.raises(ValueError, match="Input waveform is empty"):
            self.extractor.extract(np.array([]))
        
        # Test all NaN array
        with pytest.raises(ValueError, match="All waveform values are NaN"):
            self.extractor.extract(np.full(100, np.nan))
        
        # Test single value
        result = self.extractor.extract(np.array([1.0]))
        assert result.shape == (self.extractor.n_freqs, 1)
        
        # Test very short signal
        short_signal = np.array([1.0, 2.0, 3.0])
        result = self.extractor.extract(short_signal)
        assert result.shape == (self.extractor.n_freqs, 3)
    
    def test_power_normalization(self):
        """Test power scalogram normalization."""
        result = self.extractor.extract(self.test_signal)
        
        # Power should be normalized to [0, 1] range
        assert np.all(result.power >= 0)
        assert np.all(result.power <= 1)
        
        # Should have some variation (not all zeros or ones)
        assert np.std(result.power) > 0.01
    
    def test_multi_channel_extraction(self):
        """Test multi-channel CWT extraction."""
        # Create 3-channel data (H, D, Z components)
        multi_channel = np.array([
            self.test_signal,
            self.test_signal * 0.8,  # Scaled version
            self.test_signal * 1.2   # Another scaled version
        ])
        
        results = self.extractor.extract_multi_channel(multi_channel)
        
        # Check results
        assert len(results) == 3
        assert all(isinstance(result, CWTResult) for result in results)
        
        # All results should have same shape
        expected_shape = (self.extractor.n_freqs, len(self.test_signal))
        assert all(result.shape == expected_shape for result in results)
        
        # Results should be different (due to scaling)
        assert not np.allclose(results[0].power, results[1].power)
        assert not np.allclose(results[0].power, results[2].power)
    
    def test_multi_channel_with_failures(self):
        """Test multi-channel extraction with some channels failing."""
        # Create data with one problematic channel
        multi_channel = np.array([
            self.test_signal,
            np.full_like(self.test_signal, np.nan),  # All NaN channel
            self.test_signal * 0.5
        ])
        
        with patch('preprocessing.scalogram_extractor.logger') as mock_logger:
            results = self.extractor.extract_multi_channel(multi_channel)
        
        # Should return 3 results (including dummy for failed channel)
        assert len(results) == 3
        assert all(isinstance(result, CWTResult) for result in results)
        
        # Failed channel should have zero power
        assert np.all(results[1].power == 0)
        
        # Should log warning for failed channel
        mock_logger.warning.assert_called_once()
    
    def test_multi_channel_wrong_shape(self):
        """Test multi-channel extraction with wrong input shape."""
        # Test 1D array (should fail)
        with pytest.raises(ValueError, match="Expected 2D array"):
            self.extractor.extract_multi_channel(self.test_signal)
        
        # Test 3D array (should fail)
        with pytest.raises(ValueError, match="Expected 2D array"):
            self.extractor.extract_multi_channel(self.test_signal.reshape(1, 1, -1))
    
    def test_ulf_band_extraction(self):
        """Test ULF band frequency indices and power extraction."""
        # Test default ULF range
        ulf_indices = self.extractor.get_ulf_band_indices()
        assert len(ulf_indices) > 0
        
        # Check that indices correspond to ULF frequencies
        ulf_freqs = self.extractor.frequencies[ulf_indices]
        assert np.all(ulf_freqs >= 0.01)
        assert np.all(ulf_freqs <= 0.1)
        
        # Test custom ULF range
        custom_indices = self.extractor.get_ulf_band_indices((0.02, 0.05))
        custom_freqs = self.extractor.frequencies[custom_indices]
        assert np.all(custom_freqs >= 0.02)
        assert np.all(custom_freqs <= 0.05)
        assert len(custom_indices) <= len(ulf_indices)  # Should be subset
    
    def test_ulf_power_extraction(self):
        """Test ULF power time series extraction."""
        result = self.extractor.extract(self.test_signal)
        ulf_power = self.extractor.extract_ulf_power(result)
        
        # Check output shape and properties
        assert len(ulf_power) == len(self.test_signal)
        assert np.all(ulf_power >= 0)
        assert np.std(ulf_power) > 0  # Should have variation
        
        # Test with custom ULF range
        ulf_power_custom = self.extractor.extract_ulf_power(
            result, ulf_range=(0.02, 0.05)
        )
        assert len(ulf_power_custom) == len(self.test_signal)
        
        # Test with no frequencies in range
        with patch('preprocessing.scalogram_extractor.logger') as mock_logger:
            ulf_power_empty = self.extractor.extract_ulf_power(
                result, ulf_range=(0.6, 0.8)  # Outside available range
            )
            assert np.all(ulf_power_empty == 0)
            mock_logger.warning.assert_called_once()
    
    def test_properties(self):
        """Test extractor properties."""
        # Test frequency resolution
        freq_res = self.extractor.frequency_resolution
        assert freq_res > 0
        
        # Test scales and frequencies properties
        scales = self.extractor.scales
        frequencies = self.extractor.frequencies
        
        assert len(scales) == self.extractor.n_freqs
        assert len(frequencies) == self.extractor.n_freqs
        assert not np.shares_memory(scales, self.extractor._scales)  # Should be copy
        assert not np.shares_memory(frequencies, self.extractor._frequencies)
    
    def test_cwt_result_properties(self):
        """Test CWTResult dataclass properties."""
        result = self.extractor.extract(self.test_signal)
        
        # Test shape property
        assert result.shape == result.coefficients.shape
        
        # Test frequency_range property
        freq_min, freq_max = result.frequency_range
        assert freq_min == result.frequencies.min()
        assert freq_max == result.frequencies.max()
    
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.extractor)
        assert 'CWTScalogramExtractor' in repr_str
        assert 'morl' in repr_str
        assert '(0.01, 0.5)' in repr_str
        assert '32' in repr_str
        assert '1.0' in repr_str
    
    def test_backward_compatibility_alias(self):
        """Test backward compatibility alias."""
        from preprocessing.scalogram_extractor import ScalogramExtractor
        
        # Should be the same class
        assert ScalogramExtractor is CWTScalogramExtractor
        
        # Should work identically
        extractor_alias = ScalogramExtractor()
        result_alias = extractor_alias.extract(self.test_signal)
        assert isinstance(result_alias, CWTResult)


class TestCWTResultDataclass:
    """Test suite for CWTResult dataclass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.coefficients = np.random.complex128((32, 100))
        self.frequencies = np.logspace(-2, -0.3, 32)
        self.scales = np.logspace(0, 2, 32)
        self.power = np.abs(self.coefficients) ** 2
        
        self.result = CWTResult(
            coefficients=self.coefficients,
            frequencies=self.frequencies,
            scales=self.scales,
            power=self.power,
            sampling_rate=1.0,
            wavelet='morl'
        )
    
    def test_dataclass_creation(self):
        """Test CWTResult dataclass creation."""
        assert isinstance(self.result, CWTResult)
        assert np.array_equal(self.result.coefficients, self.coefficients)
        assert np.array_equal(self.result.frequencies, self.frequencies)
        assert np.array_equal(self.result.scales, self.scales)
        assert np.array_equal(self.result.power, self.power)
        assert self.result.sampling_rate == 1.0
        assert self.result.wavelet == 'morl'
    
    def test_shape_property(self):
        """Test shape property."""
        assert self.result.shape == (32, 100)
        assert self.result.shape == self.coefficients.shape
    
    def test_frequency_range_property(self):
        """Test frequency_range property."""
        freq_min, freq_max = self.result.frequency_range
        assert freq_min == self.frequencies.min()
        assert freq_max == self.frequencies.max()


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])