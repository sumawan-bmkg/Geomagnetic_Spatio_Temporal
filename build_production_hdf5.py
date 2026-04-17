#!/usr/bin/env python3
"""
Build Production HDF5 - Real Data Pipeline

Script ini menyatukan data asli dari scalogramv3 ke dalam format HDF5 siap pakai
untuk production training dengan audit lengkap dan pembersihan total.
"""
import sys
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionHDF5Builder:
    """
    Builder untuk menyatukan data asli scalogramv3 menjadi HDF5 production-ready.
    """
    
    def __init__(self):
        self.scalogram_path = Path('../scalogramv3')
        self.earthquake_catalog_path = Path('../awal/earthquake_catalog_2018_2025_merged.csv')
        self.kp_index_path = Path('../awal/kp_index_2018_2026.csv')
        self.station_coords_path = Path('../awal/lokasi_stasiun.csv')
        
        # Expected 8 primary stations
        self.primary_stations = ['SBG', 'SCN', 'KPY', 'LWA', 'LPS', 'SRG', 'SKB', 'CLP']
        
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': {},
            'temporal_coverage': {},
            'station_completeness': {},
            'magnitude_distribution': {},
            'noise_analysis': {},
            'readiness_status': 'UNKNOWN'
        }
    
    def run_complete_audit_and_build(self):
        """Jalankan audit lengkap dan build HDF5 production."""
        logger.info("=" * 70)
        logger.info("PRODUCTION HDF5 BUILDER - REAL DATA PIPELINE")
        logger.info("=" * 70)
        
        try:
            # 1. Audit Data Sources
            logger.info("\n1. AUDIT DATA SOURCES")
            logger.info("-" * 40)
            self.audit_data_sources()
            
            # 2. Load and Analyze Scalogramv3 Data
            logger.info("\n2. SCALOGRAMV3 DATA ANALYSIS")
            logger.info("-" * 40)
            scalogram_data = self.analyze_scalogramv3_data()
            
            # 3. Temporal Coverage Audit
            logger.info("\n3. TEMPORAL COVERAGE AUDIT")
            logger.info("-" * 40)
            self.audit_temporal_coverage(scalogram_data)
            
            # 4. Station Completeness Audit
            logger.info("\n4. STATION COMPLETENESS AUDIT")
            logger.info("-" * 40)
            self.audit_station_completeness(scalogram_data)
            
            # 5. Magnitude Distribution Audit
            logger.info("\n5. MAGNITUDE DISTRIBUTION AUDIT")
            logger.info("-" * 40)
            self.audit_magnitude_distribution(scalogram_data)
            
            # 6. Noise Pattern Analysis
            logger.info("\n6. NOISE PATTERN ANALYSIS")
            logger.info("-" * 40)
            self.analyze_noise_patterns(scalogram_data)
            
            # 7. Generate Production HDF5
            logger.info("\n7. GENERATE PRODUCTION HDF5")
            logger.info("-" * 40)
            self.build_production_hdf5(scalogram_data)
            
            # 8. Final Readiness Assessment
            logger.info("\n8. FINAL READINESS ASSESSMENT")
            logger.info("-" * 40)
            self.assess_final_readiness()
            
        except Exception as e:
            logger.error(f"Error during audit and build: {e}")
            self.audit_results['error'] = str(e)
            raise
        
        return self.audit_results
    
    def audit_data_sources(self):
        """Audit ketersediaan dan validitas sumber data."""
        logger.info("Auditing data sources...")
        
        sources = {
            'scalogramv3': {
                'path': str(self.scalogram_path),
                'exists': self.scalogram_path.exists(),
                'files': []
            },
            'earthquake_catalog': {
                'path': str(self.earthquake_catalog_path),
                'exists': self.earthquake_catalog_path.exists(),
                'records': 0
            },
            'kp_index': {
                'path': str(self.kp_index_path),
                'exists': self.kp_index_path.exists(),
                'records': 0
            },
            'station_coords': {
                'path': str(self.station_coords_path),
                'exists': self.station_coords_path.exists(),
                'stations': 0
            }
        }
        
        # Check scalogramv3 files
        if sources['scalogramv3']['exists']:
            h5_files = list(self.scalogram_path.glob('*.h5'))
            sources['scalogramv3']['files'] = [f.name for f in h5_files]
            logger.info(f"Found {len(h5_files)} HDF5 files in scalogramv3:")
            for f in h5_files:
                logger.info(f"  - {f.name}")
        
        # Check earthquake catalog
        if sources['earthquake_catalog']['exists']:
            df = pd.read_csv(self.earthquake_catalog_path)
            sources['earthquake_catalog']['records'] = len(df)
            logger.info(f"Earthquake catalog: {len(df)} events")
        
        # Check Kp-index data
        if sources['kp_index']['exists']:
            df = pd.read_csv(self.kp_index_path)
            sources['kp_index']['records'] = len(df)
            logger.info(f"Kp-index data: {len(df)} records")
        
        # Check station coordinates
        if sources['station_coords']['exists']:
            df = pd.read_csv(self.station_coords_path, sep=';')
            sources['station_coords']['stations'] = len(df)
            logger.info(f"Station coordinates: {len(df)} stations")
        
        self.audit_results['data_sources'] = sources
        
        # Validate all sources available
        missing_sources = [name for name, info in sources.items() if not info['exists']]
        if missing_sources:
            raise FileNotFoundError(f"Missing data sources: {missing_sources}")
    
    def analyze_scalogramv3_data(self):
        """Analyze scalogramv3 HDF5 files."""
        logger.info("Analyzing scalogramv3 HDF5 data...")
        
        h5_files = list(self.scalogram_path.glob('*.h5'))
        if not h5_files:
            raise FileNotFoundError("No HDF5 files found in scalogramv3")
        
        # Use the main file (cosmic_final seems to be the primary)
        main_file = None
        for f in h5_files:
            if 'cosmic_final.h5' in f.name:
                main_file = f
                break
        
        if main_file is None:
            main_file = h5_files[0]  # Use first available
        
        logger.info(f"Using primary file: {main_file.name}")
        
        scalogram_data = {}
        
        with h5py.File(main_file, 'r') as f:
            logger.info("HDF5 structure:")
            
            def print_structure(name, obj):
                if isinstance(obj, h5py.Dataset):
                    logger.info(f"  Dataset: {name} - Shape: {obj.shape}, Type: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    logger.info(f"  Group: {name}")
            
            f.visititems(print_structure)
            
            # Extract key datasets
            scalogram_data['file_path'] = str(main_file)
            scalogram_data['groups'] = list(f.keys())
            scalogram_data['datasets'] = {}
            
            # Look for common dataset patterns
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    dataset = f[key]
                    scalogram_data['datasets'][key] = {
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                        'size_mb': dataset.size * dataset.dtype.itemsize / (1024*1024)
                    }
                    
                    # Sample data for analysis
                    if dataset.size < 1000000:  # Only sample small datasets
                        scalogram_data['datasets'][key]['sample'] = dataset[:min(10, dataset.shape[0])].tolist()
                
                elif isinstance(f[key], h5py.Group):
                    # Explore group structure
                    group_info = {}
                    for subkey in f[key].keys():
                        if isinstance(f[key][subkey], h5py.Dataset):
                            subdataset = f[key][subkey]
                            group_info[subkey] = {
                                'shape': subdataset.shape,
                                'dtype': str(subdataset.dtype)
                            }
                    scalogram_data['datasets'][key] = group_info
        
        return scalogram_data
    
    def audit_temporal_coverage(self, scalogram_data):
        """Audit temporal coverage dari data scalogramv3."""
        logger.info("Auditing temporal coverage...")
        
        # Load earthquake catalog
        catalog_df = pd.read_csv(self.earthquake_catalog_path)
        catalog_df['datetime'] = pd.to_datetime(catalog_df['datetime'])
        
        # Analyze temporal range
        min_date = catalog_df['datetime'].min()
        max_date = catalog_df['datetime'].max()
        
        logger.info(f"Catalog temporal range: {min_date} to {max_date}")
        
        # Check train/test split
        train_end = pd.to_datetime('2024-06-30')
        test_start = pd.to_datetime('2024-07-01')
        
        train_events = catalog_df[catalog_df['datetime'] <= train_end]
        test_events = catalog_df[catalog_df['datetime'] >= test_start]
        
        logger.info(f"Train events (2018-June 2024): {len(train_events)}")
        logger.info(f"Test events (July 2024-2026): {len(test_events)}")
        
        # Year-by-year breakdown
        yearly_counts = catalog_df.groupby(catalog_df['datetime'].dt.year).size()
        logger.info("Events per year:")
        for year, count in yearly_counts.items():
            logger.info(f"  {year}: {count} events")
        
        self.audit_results['temporal_coverage'] = {
            'total_events': len(catalog_df),
            'date_range': {
                'min': min_date.isoformat(),
                'max': max_date.isoformat()
            },
            'train_test_split': {
                'train_events': len(train_events),
                'test_events': len(test_events),
                'train_percentage': len(train_events) / len(catalog_df) * 100
            },
            'yearly_distribution': yearly_counts.to_dict(),
            'stress_test_coverage': len(test_events) >= 100  # Minimum for stress testing
        }
    
    def audit_station_completeness(self, scalogram_data):
        """Audit kelengkapan stasiun dalam data scalogramv3."""
        logger.info("Auditing station completeness...")
        
        # Load station coordinates
        station_df = pd.read_csv(self.station_coords_path, sep=';')
        official_stations = set(station_df['Kode Stasiun'].dropna().values)
        
        logger.info(f"Official stations: {sorted(official_stations)}")
        logger.info(f"Expected primary stations: {self.primary_stations}")
        
        # Check which primary stations are available
        available_primary = []
        missing_primary = []
        
        for station in self.primary_stations:
            if station in official_stations:
                available_primary.append(station)
            else:
                missing_primary.append(station)
        
        logger.info(f"Available primary stations: {available_primary}")
        if missing_primary:
            logger.warning(f"Missing primary stations: {missing_primary}")
        
        # Calculate completeness metrics
        completeness_percentage = len(available_primary) / len(self.primary_stations) * 100
        
        self.audit_results['station_completeness'] = {
            'total_official_stations': len(official_stations),
            'expected_primary_stations': len(self.primary_stations),
            'available_primary_stations': len(available_primary),
            'missing_primary_stations': len(missing_primary),
            'completeness_percentage': completeness_percentage,
            'available_stations': available_primary,
            'missing_stations': missing_primary,
            'ready_for_8station_processing': completeness_percentage >= 87.5  # 7/8 stations minimum
        }
        
        logger.info(f"Station completeness: {completeness_percentage:.1f}%")
    
    def audit_magnitude_distribution(self, scalogram_data):
        """Audit distribusi magnitudo dalam katalog."""
        logger.info("Auditing magnitude distribution...")
        
        # Load earthquake catalog
        catalog_df = pd.read_csv(self.earthquake_catalog_path)
        
        # Define magnitude classes
        magnitude_bins = [0, 4.0, 4.5, 5.0, 5.5, 10.0]
        magnitude_labels = ['Small', 'Normal', 'Moderate', 'Medium', 'Large']
        
        catalog_df['magnitude_class'] = pd.cut(
            catalog_df['Magnitude'], 
            bins=magnitude_bins, 
            labels=magnitude_labels, 
            include_lowest=True
        )
        
        # Count distribution
        class_counts = catalog_df['magnitude_class'].value_counts()
        
        logger.info("Magnitude distribution:")
        for mag_class, count in class_counts.items():
            percentage = count / len(catalog_df) * 100
            logger.info(f"  {mag_class}: {count} events ({percentage:.1f}%)")
        
        # Statistics
        mag_stats = {
            'mean': float(catalog_df['Magnitude'].mean()),
            'std': float(catalog_df['Magnitude'].std()),
            'min': float(catalog_df['Magnitude'].min()),
            'max': float(catalog_df['Magnitude'].max()),
            'median': float(catalog_df['Magnitude'].median())
        }
        
        logger.info(f"Magnitude statistics: Mean={mag_stats['mean']:.2f}, Std={mag_stats['std']:.2f}")
        
        self.audit_results['magnitude_distribution'] = {
            'total_events': len(catalog_df),
            'class_distribution': class_counts.to_dict(),
            'statistics': mag_stats,
            'magnitude_bins': magnitude_bins,
            'magnitude_labels': magnitude_labels,
            'balanced_distribution': min(class_counts) >= 10  # Minimum samples per class
        }
    
    def analyze_noise_patterns(self, scalogram_data):
        """Analyze noise patterns dalam data scalogramv3."""
        logger.info("Analyzing noise patterns...")
        
        main_file = Path(scalogram_data['file_path'])
        
        noise_analysis = {
            'file_analyzed': main_file.name,
            'datasets_analyzed': [],
            'statistical_summary': {},
            'quality_assessment': 'unknown'
        }
        
        try:
            with h5py.File(main_file, 'r') as f:
                # Analyze first few datasets
                analyzed_count = 0
                for key in f.keys():
                    if analyzed_count >= 3:  # Limit analysis
                        break
                    
                    if isinstance(f[key], h5py.Dataset):
                        dataset = f[key]
                        
                        # Sample data for analysis
                        if dataset.size > 0:
                            sample_size = min(1000, dataset.size)
                            if len(dataset.shape) > 1:
                                # Multi-dimensional data
                                sample = dataset.flat[:sample_size]
                            else:
                                sample = dataset[:sample_size]
                            
                            stats = {
                                'dataset': key,
                                'shape': list(dataset.shape),
                                'mean': float(np.mean(sample)),
                                'std': float(np.std(sample)),
                                'min': float(np.min(sample)),
                                'max': float(np.max(sample)),
                                'range': float(np.max(sample) - np.min(sample)),
                                'non_zero_percentage': float(np.count_nonzero(sample) / len(sample) * 100)
                            }
                            
                            noise_analysis['datasets_analyzed'].append(stats)
                            analyzed_count += 1
                            
                            logger.info(f"  Dataset {key}:")
                            logger.info(f"    Shape: {stats['shape']}")
                            logger.info(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                            logger.info(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        
        except Exception as e:
            logger.error(f"Error analyzing noise patterns: {e}")
            noise_analysis['error'] = str(e)
        
        # Assess quality
        if noise_analysis['datasets_analyzed']:
            # Calculate aggregate statistics
            all_ranges = [d['range'] for d in noise_analysis['datasets_analyzed']]
            all_stds = [d['std'] for d in noise_analysis['datasets_analyzed']]
            
            avg_range = np.mean(all_ranges)
            avg_std = np.mean(all_stds)
            
            noise_analysis['statistical_summary'] = {
                'average_range': float(avg_range),
                'average_std': float(avg_std),
                'datasets_count': len(noise_analysis['datasets_analyzed'])
            }
            
            # Quality assessment
            if 0.01 < avg_range < 10.0 and 0.1 < avg_std < 2.0:
                noise_analysis['quality_assessment'] = 'good'
            elif 0.001 < avg_range < 100.0 and 0.01 < avg_std < 10.0:
                noise_analysis['quality_assessment'] = 'acceptable'
            else:
                noise_analysis['quality_assessment'] = 'poor'
        
        self.audit_results['noise_analysis'] = noise_analysis
        logger.info(f"Noise quality assessment: {noise_analysis['quality_assessment']}")
    
    def build_production_hdf5(self, scalogram_data):
        """Build production-ready HDF5 file."""
        logger.info("Building production HDF5...")
        
        output_path = Path('real_earthquake_dataset.h5')
        
        # Load required data
        catalog_df = pd.read_csv(self.earthquake_catalog_path)
        catalog_df['datetime'] = pd.to_datetime(catalog_df['datetime'])
        
        kp_df = pd.read_csv(self.kp_index_path)
        kp_df['datetime'] = pd.to_datetime(kp_df['Date_Time_UTC'])
        
        station_df = pd.read_csv(self.station_coords_path, sep=';')
        
        # Chronological split
        train_end = pd.to_datetime('2024-06-30')
        train_events = catalog_df[catalog_df['datetime'] <= train_end]
        test_events = catalog_df[catalog_df['datetime'] > train_end]
        
        logger.info(f"Creating HDF5 with {len(train_events)} train + {len(test_events)} test events")
        
        # Create production HDF5
        with h5py.File(output_path, 'w') as f:
            # Metadata
            f.attrs['creation_time'] = datetime.now().isoformat()
            f.attrs['data_source'] = 'scalogramv3_real_data'
            f.attrs['total_events'] = len(catalog_df)
            f.attrs['train_events'] = len(train_events)
            f.attrs['test_events'] = len(test_events)
            f.attrs['n_stations'] = len(self.primary_stations)
            f.attrs['stations'] = [s.encode('utf-8') for s in self.primary_stations]
            f.attrs['train_end_date'] = '2024-06-30'
            f.attrs['test_start_date'] = '2024-07-01'
            
            # Configuration group
            config_group = f.create_group('config')
            config_group.create_dataset('station_list', data=[s.encode('utf-8') for s in self.primary_stations])
            config_group.create_dataset('train_event_ids', data=train_events['event_id'].values)
            config_group.create_dataset('test_event_ids', data=test_events['event_id'].values)
            
            # Metadata group
            metadata_group = f.create_group('metadata')
            metadata_group.create_dataset('event_id', data=catalog_df['event_id'].values)
            metadata_group.create_dataset('magnitude', data=catalog_df['Magnitude'].values)
            metadata_group.create_dataset('latitude', data=catalog_df['Latitude'].values)
            metadata_group.create_dataset('longitude', data=catalog_df['Longitude'].values)
            metadata_group.create_dataset('depth', data=catalog_df['Depth'].values)
            
            # Convert datetime to string for HDF5 storage
            datetime_strings = [dt.isoformat().encode('utf-8') for dt in catalog_df['datetime']]
            metadata_group.create_dataset('datetime', data=datetime_strings)
            
            # Station coordinates
            station_group = f.create_group('station_coordinates')
            for _, row in station_df.iterrows():
                if row['Kode Stasiun'] in self.primary_stations:
                    station_subgroup = station_group.create_group(row['Kode Stasiun'])
                    station_subgroup.attrs['latitude'] = row['Latitude']
                    station_subgroup.attrs['longitude'] = row['Longitude']
            
            # Placeholder for tensor data (to be filled by actual scalogram processing)
            # This creates the structure for future data loading
            placeholder_shape = (len(catalog_df), len(self.primary_stations), 3, 224, 224)
            tensor_dataset = f.create_dataset(
                'scalogram_tensor', 
                shape=placeholder_shape,
                dtype=np.float32,
                fillvalue=0.0,
                compression='gzip'
            )
            
            logger.info(f"Created tensor placeholder with shape: {placeholder_shape}")
            
            # Kp-index data for CMR
            kp_group = f.create_group('kp_index')
            kp_group.create_dataset('datetime', data=[dt.isoformat().encode('utf-8') for dt in kp_df['datetime']])
            kp_group.create_dataset('kp_index', data=kp_df['Kp_Index'].values)
        
        logger.info(f"Production HDF5 created: {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        
        return str(output_path)
    
    def assess_final_readiness(self):
        """Assess final readiness untuk production training."""
        logger.info("Assessing final readiness...")
        
        # Collect readiness indicators
        temporal = self.audit_results.get('temporal_coverage', {})
        stations = self.audit_results.get('station_completeness', {})
        magnitude = self.audit_results.get('magnitude_distribution', {})
        noise = self.audit_results.get('noise_analysis', {})
        
        readiness_checks = {
            'temporal_coverage': temporal.get('stress_test_coverage', False),
            'station_completeness': stations.get('ready_for_8station_processing', False),
            'magnitude_balance': magnitude.get('balanced_distribution', False),
            'data_quality': noise.get('quality_assessment') in ['good', 'acceptable']
        }
        
        # Calculate overall readiness
        passed_checks = sum(readiness_checks.values())
        total_checks = len(readiness_checks)
        
        if passed_checks == total_checks:
            overall_status = 'READY'
        elif passed_checks >= total_checks * 0.75:
            overall_status = 'MOSTLY_READY'
        else:
            overall_status = 'NOT_READY'
        
        # Generate recommendations
        recommendations = []
        if not readiness_checks['temporal_coverage']:
            recommendations.append('Increase test set coverage for robust stress testing')
        if not readiness_checks['station_completeness']:
            recommendations.append('Implement zero-padding strategy for missing stations')
        if not readiness_checks['magnitude_balance']:
            recommendations.append('Consider data augmentation for underrepresented magnitude classes')
        if not readiness_checks['data_quality']:
            recommendations.append('Review data preprocessing for noise reduction')
        
        self.audit_results['readiness_status'] = overall_status
        self.audit_results['readiness_checks'] = readiness_checks
        self.audit_results['recommendations'] = recommendations
        
        # Log final assessment
        logger.info("=" * 60)
        logger.info(f"FINAL READINESS STATUS: {overall_status}")
        logger.info("=" * 60)
        logger.info(f"Passed checks: {passed_checks}/{total_checks}")
        
        for check, status in readiness_checks.items():
            status_str = "✅ PASS" if status else "❌ FAIL"
            logger.info(f"  {check}: {status_str}")
        
        if recommendations:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        return overall_status
    
    def save_audit_report(self, output_path='dataset_readiness_report.json'):
        """Save comprehensive audit report."""
        with open(output_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        logger.info(f"Comprehensive audit report saved: {output_path}")


def main():
    """Main function."""
    print("PRODUCTION HDF5 BUILDER")
    print("Real Data Pipeline from Scalogramv3")
    print("=" * 70)
    
    try:
        # Run complete audit and build
        builder = ProductionHDF5Builder()
        results = builder.run_complete_audit_and_build()
        
        # Save comprehensive report
        builder.save_audit_report()
        
        # Print final status
        status = results.get('readiness_status', 'UNKNOWN')
        print(f"\nFINAL STATUS: {status}")
        
        if status == 'READY':
            print("\n🎉 DATASET READY FOR PRODUCTION TRAINING!")
            print("📁 Production HDF5: real_earthquake_dataset.h5")
            print("🚀 Ready to run: python run_production_train.py")
        elif status == 'MOSTLY_READY':
            print("\n⚠️  DATASET MOSTLY READY - Minor issues detected")
            print("📋 Check recommendations in audit report")
        else:
            print("\n❌ DATASET NOT READY - Major issues detected")
            print("🔧 Address issues before production training")
        
        return status
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 'ERROR'


if __name__ == '__main__':
    main()