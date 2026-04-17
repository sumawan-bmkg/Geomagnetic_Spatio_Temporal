"""
Preprocessing Module for Spatio-Temporal Earthquake Precursor Analysis

This module provides tools for processing geomagnetic data including:
- Data reading from FRG604RC binary files
- Signal processing with ULF frequency filters
- Continuous Wavelet Transform (CWT) analysis
- Z/H ratio scalogram generation for earthquake precursor detection
- Comprehensive data auditing and chronological splitting
"""

from .data_reader import GeomagneticDataReader
from .signal_processor import GeomagneticSignalProcessor
from .scalogram_processor import ScalogramProcessor
from .data_auditor import DataAuditor
from .tensor_engine import TensorEngine

__all__ = [
    'GeomagneticDataReader',
    'GeomagneticSignalProcessor', 
    'ScalogramProcessor',
    'DataAuditor',
    'TensorEngine'
]

__version__ = '1.0.0'
__author__ = 'Spatio Precursor Project Team'