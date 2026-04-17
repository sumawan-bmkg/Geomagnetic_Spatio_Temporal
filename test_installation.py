#!/usr/bin/env python3
"""
Installation Test Script for Spatio-Temporal Earthquake Precursor Analysis Project

This script tests the installation and basic functionality of all preprocessing modules.
Run this script to verify that the refactored modules work correctly.
"""
import sys
import os
import traceback
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all modules can be imported successfully."""
    print("Testing module imports...")
    
    try:
        from preprocessing import GeomagneticDataReader, GeomagneticSignalProcessor, ScalogramProcessor
        print("✓ All preprocessing modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_reader():
    """Test GeomagneticDataReader with synthetic data."""
    print("\nTesting GeomagneticDataReader...")
    
    try:
        from preprocessing import GeomagneticDataReader
        
        # Initialize reader
        reader = GeomagneticDataReader()
        
        # Test internal methods with dummy data
        test_data = np.array([1, 2, 3], dtype=np.uint8)
        
        # Test two's complement conversion
        result = reader._twos_complement(np.array([65535, 32768, 1000]), 16)
        # 65535 should become -1 in 16-bit two's complement
        # 32768 should become -32768 in 16-bit two's complement
        expected_negative = result[0] == -1 and result[1] == -32768
        
        if expected_negative:
            print("✓ GeomagneticDataReader basic functionality works")
            return True
        else:
            print(f"✗ GeomagneticDataReader two's complement conversion failed: got {result}")
            return False
            
    except Exception as e:
        print(f"✗ GeomagneticDataReader test failed: {e}")
        traceback.print_exc()
        return False

def test_signal_processor():
    """Test GeomagneticSignalProcessor with synthetic data."""
    print("\nTesting GeomagneticSignalProcessor...")
    
    try:
        from preprocessing import GeomagneticSignalProcessor
        
        # Initialize processor
        processor = GeomagneticSignalProcessor(sampling_rate=1.0)
        
        # Generate test data
        t = np.linspace(0, 3600, 3600)  # 1 hour of data
        h_test = 40000 + 50 * np.sin(2 * np.pi * 0.05 * t) + 10 * np.random.randn(len(t))
        d_test = 2.0 + 0.5 * np.sin(2 * np.pi * 0.03 * t) + 0.1 * np.random.randn(len(t))
        z_test = 30000 + 40 * np.sin(2 * np.pi * 0.04 * t) + 8 * np.random.randn(len(t))
        
        # Test processing
        result = processor.process_components(h_test, d_test, z_test, apply_ulf=True, apply_pc3=True)
        
        # Verify results
        required_keys = ['h_raw', 'd_raw', 'z_raw', 'zh_ratio_raw']
        if all(key in result for key in required_keys):
            print("✓ GeomagneticSignalProcessor basic functionality works")
            
            # Test filtering
            if 'h_ulf' in result and 'zh_ratio_ulf' in result:
                print("✓ ULF filtering works")
            
            if 'h_pc3' in result and 'zh_ratio_pc3' in result:
                print("✓ PC3 filtering works")
                
            return True
        else:
            print("✗ GeomagneticSignalProcessor missing required output keys")
            return False
            
    except Exception as e:
        print(f"✗ GeomagneticSignalProcessor test failed: {e}")
        traceback.print_exc()
        return False

def test_scalogram_processor():
    """Test ScalogramProcessor with synthetic data."""
    print("\nTesting ScalogramProcessor...")
    
    try:
        from preprocessing import ScalogramProcessor
        
        # Initialize processor
        processor = ScalogramProcessor(sampling_rate=1.0, wavelet='morl')
        
        # Generate test data with ULF components
        t = np.linspace(0, 3600, 3600)  # 1 hour of data
        h_test = 40000 + 100 * np.sin(2 * np.pi * 0.05 * t) + 20 * np.random.randn(len(t))
        z_test = 30000 + 80 * np.sin(2 * np.pi * 0.04 * t) + 15 * np.random.randn(len(t))
        
        # Test CWT computation
        scales = processor._generate_scales()
        coeffs, freqs = processor.compute_cwt(h_test, scales)
        
        if coeffs.shape[0] == len(scales) and coeffs.shape[1] == len(h_test):
            print("✓ CWT computation works")
        else:
            print("✗ CWT computation failed - wrong output shape")
            return False
        
        # Test scalogram computation
        scalogram_data = processor.compute_zh_ratio_scalogram(z_test, h_test)
        
        required_keys = ['z_power', 'h_power', 'zh_ratio_power', 'frequencies']
        if all(key in scalogram_data for key in required_keys):
            print("✓ Z/H ratio scalogram computation works")
        else:
            print("✗ Scalogram computation missing required keys")
            return False
        
        # Test ULF feature extraction
        ulf_features = processor.extract_ulf_features(scalogram_data)
        
        if ulf_features and 'ulf_mean_power' in ulf_features:
            print("✓ ULF feature extraction works")
            return True
        else:
            print("✗ ULF feature extraction failed")
            return False
            
    except Exception as e:
        print(f"✗ ScalogramProcessor test failed: {e}")
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'pywt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def test_directory_structure():
    """Test if the directory structure is correct."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'src',
        'src/preprocessing',
        'configs',
        'examples',
        'data',
        'data/raw',
        'data/processed',
        'outputs'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
            # Create missing directories
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ Created missing directory: {dir_path}")
        else:
            print(f"✓ {dir_path} exists")
    
    return True

def run_all_tests():
    """Run all installation tests."""
    print("=" * 60)
    print("SPATIO-TEMPORAL EARTHQUAKE PRECURSOR ANALYSIS PROJECT")
    print("Installation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Directory Structure", test_directory_structure),
        ("Module Imports", test_imports),
        ("Data Reader", test_data_reader),
        ("Signal Processor", test_signal_processor),
        ("Scalogram Processor", test_scalogram_processor)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Run the example: python examples/preprocessing_example.py")
        print("2. Check the documentation in README.md")
        print("3. Start processing your geomagnetic data!")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you're running from the project root directory")
        print("3. Verify Python version compatibility (Python 3.7+)")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)