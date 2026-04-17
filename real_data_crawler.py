#!/usr/bin/env python3
"""
Real Data Crawler - Scalogramv3 Data Mapping & Audit

Script ini melakukan crawling dan audit terhadap folder scalogramv3 untuk:
1. Memetakan seluruh file .npy atau .png yang tersedia
2. Sinkronisasi dengan earthquake_catalog_2018_2025_merged.csv
3. Audit kelengkapan stasiun dan temporal coverage
4. Audit distribusi kelas dan noise patterns
"""
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataCrawler:
    """
    Crawler untuk memetakan dan mengaudit data asli dari scalogramv3.
    """
    
    def __init__(self, scalogram_base_path='../scalogramv3'):
        self.scalogram_base_path = Path(scalogram_base_path)
        self.earthquake_catalog_path = Path('../awal/earthquake_catalog_2018_2025_merged.csv')
        self.station_coords_path = Path('../awal/lokasi_stasiun.csv')
        
        # Expected 8 primary stations
        self.primary_stations = ['SBG', 'SCN', 'KPY', 'LWA', 'LPS', 'SRG', 'SKB', 'CLP']
        
        self.crawl_results = {
            'timestamp': datetime.now().isoformat(),
            'scalogram_base_path': str(self.scalogram_base_path),
            'file_mapping': {},
            'temporal_audit': {},
            'station_completeness': {},
            'magnitude_distribution': {},
            'noise_audit': {},
            'readiness_summary': {}
        }
    
    def run_complete_crawl(self):
        """Jalankan crawling dan audit lengkap."""
        logger.info("=" * 60)
        logger.info("REAL DATA CRAWLER - SCALOGRAMV3 AUDIT")
        logger.info("=" * 60)
        
        # Check if scalogramv3 exists
        if not self.scalogram_base_path.exists():
            logger.error(f"Scalogramv3 folder tidak ditemukan: {self.scalogram_base_path}")
            logger.info("Membuat struktur folder simulasi untuk testing...")
            self.create_test_structure()
            return self.crawl_results
        
        try:
            # 1. File Mapping & Discovery
            logger.info("\n1. FILE MAPPING & DISCOVERY")
            logger.info("-" * 40)
            self.map_scalogram_files()
            
            # 2. Temporal Coverage Audit
            logger.info("\n2. TEMPORAL COVERAGE AUDIT")
            logger.info("-" * 40)
            self.audit_temporal_coverage()
            
            # 3. Station Completeness Audit
            logger.info("\n3. STATION COMPLETENESS AUDIT")
            logger.info("-" * 40)
            self.audit_station_completeness()
            
            # 4. Magnitude Distribution Audit
            logger.info("\n4. MAGNITUDE DISTRIBUTION AUDIT")
            logger.info("-" * 40)
            self.audit_magnitude_distribution()
            
            # 5. Noise Pattern Audit
            logger.info("\n5. NOISE PATTERN AUDIT")
            logger.info("-" * 40)
            self.audit_noise_patterns()
            
            # 6. Generate Readiness Report
            logger.info("\n6. DATASET READINESS SUMMARY")
            logger.info("-" * 40)
            self.generate_readiness_summary()
            
        except Exception as e:
            logger.error(f"Error during crawling: {e}")
            self.crawl_results['error'] = str(e)
        
        return self.crawl_results
    
    def create_test_structure(self):
        """Buat struktur folder test jika scalogramv3 tidak ada."""
        logger.info("Membuat struktur test untuk demonstrasi...")
        
        # Create basic structure
        test_path = Path('test_scalogramv3')
        test_path.mkdir(exist_ok=True)
        
        for station in self.primary_stations:
            station_path = test_path / station
            station_path.mkdir(exist_ok=True)
            
            # Create some test files
            for i in range(5):
                test_file = station_path / f"scalogram_{station}_event_{35400+i}.npy"
                np.save(test_file, np.random.randn(224, 224))
        
        logger.info(f"Test structure created: {test_path}")
        self.scalogram_base_path = test_path
    
    def map_scalogram_files(self):
        """Memetakan semua file scalogram yang tersedia."""
        logger.info("Scanning scalogram files...")
        
        file_mapping = defaultdict(lambda: defaultdict(list))
        total_files = 0
        
        # Scan for .npy and .png files
        for file_path in self.scalogram_base_path.rglob('*'):
            if file_path.suffix.lower() in ['.npy', '.png']:
                # Extract station and event info from path/filename
                station = self.extract_station_from_path(file_path)
                event_id = self.extract_event_id_from_path(file_path)
                
                if station and event_id:
                    file_mapping[station][event_id].append({
                        'path': str(file_path),
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                    total_files += 1
        
        logger.info(f"Found {total_files} scalogram files")
        logger.info(f"Stations found: {list(file_mapping.keys())}")
        
        # Convert to regular dict for JSON serialization
        self.crawl_results['file_mapping'] = {
            station: dict(events) for station, events in file_mapping.items()
        }
        
        # Summary statistics
        station_counts = {station: len(events) for station, events in file_mapping.items()}
        logger.info("Files per station:")
        for station, count in station_counts.items():
            logger.info(f"  {station}: {count} events")
        
        return file_mapping
    
    def extract_station_from_path(self, file_path):
        """Extract station code from file path."""
        path_parts = file_path.parts
        filename = file_path.name
        
        # Check if any primary station is in the path
        for station in self.primary_stations:
            if station in str(file_path).upper():
                return station
        
        # Try to extract from filename patterns
        for station in self.primary_stations:
            if station.lower() in filename.lower():
                return station
        
        return None
    
    def extract_event_id_from_path(self, file_path):
        """Extract event ID from file path."""
        filename = file_path.stem
        
        # Look for numeric patterns that could be event IDs
        import re
        numbers = re.findall(r'\d+', filename)
        
        # Assume event IDs are 4-5 digit numbers
        for num in numbers:
            if len(num) >= 4 and len(num) <= 6:
                return int(num)
        
        return None
    
    def audit_temporal_coverage(self):
        """Audit temporal coverage dari 2018-2026."""
        logger.info("Auditing temporal coverage...")
        
        # Load earthquake catalog
        if not self.earthquake_catalog_path.exists():
            logger.error(f"Earthquake catalog not found: {self.earthquake_catalog_path}")
            return
        
        catalog_df = pd.read_csv(self.earthquake_catalog_path)
        catalog_df['datetime'] = pd.to_datetime(catalog_df['datetime'])
        
        # Get available event IDs from file mapping
        available_events = set()
        for station_data in self.crawl_results['file_mapping'].values():
            available_events.update(station_data.keys())
        
        # Filter catalog for available events
        available_catalog = catalog_df[catalog_df['event_id'].isin(available_events)]
        
        if len(available_catalog) == 0:
            logger.warning("No matching events found between catalog and scalogram files")
            return
        
        # Temporal analysis
        min_date = available_catalog['datetime'].min()
        max_date = available_catalog['datetime'].max()
        
        logger.info(f"Temporal range: {min_date} to {max_date}")
        
        # Check train/test split coverage
        train_end = pd.to_datetime('2024-06-30')
        test_start = pd.to_datetime('2024-07-01')
        
        train_events = available_catalog[available_catalog['datetime'] <= train_end]
        test_events = available_catalog[available_catalog['datetime'] >= test_start]
        
        logger.info(f"Train events (2018-June 2024): {len(train_events)}")
        logger.info(f"Test events (July 2024-2026): {len(test_events)}")
        
        # Year-by-year breakdown
        yearly_counts = available_catalog.groupby(available_catalog['datetime'].dt.year).size()
        logger.info("Events per year:")
        for year, count in yearly_counts.items():
            logger.info(f"  {year}: {count} events")
        
        self.crawl_results['temporal_audit'] = {
            'total_events': len(available_catalog),
            'date_range': {
                'min': min_date.isoformat(),
                'max': max_date.isoformat()
            },
            'train_test_split': {
                'train_events': len(train_events),
                'test_events': len(test_events),
                'train_percentage': len(train_events) / len(available_catalog) * 100
            },
            'yearly_distribution': yearly_counts.to_dict()
        }
    
    def audit_station_completeness(self):
        """Audit kelengkapan data per stasiun."""
        logger.info("Auditing station completeness...")
        
        file_mapping = self.crawl_results['file_mapping']
        
        # Get all unique event IDs
        all_events = set()
        for station_data in file_mapping.values():
            all_events.update(station_data.keys())
        
        logger.info(f"Total unique events: {len(all_events)}")
        
        # Check completeness per station
        completeness_stats = {}
        for station in self.primary_stations:
            if station in file_mapping:
                station_events = set(file_mapping[station].keys())
                completeness = len(station_events) / len(all_events) * 100
                completeness_stats[station] = {
                    'available_events': len(station_events),
                    'completeness_percentage': completeness,
                    'missing_events': len(all_events) - len(station_events)
                }
                logger.info(f"  {station}: {len(station_events)}/{len(all_events)} events ({completeness:.1f}%)")
            else:
                completeness_stats[station] = {
                    'available_events': 0,
                    'completeness_percentage': 0.0,
                    'missing_events': len(all_events)
                }
                logger.info(f"  {station}: 0/{len(all_events)} events (0.0%) - MISSING")
        
        # Calculate events with complete 8-station coverage
        complete_events = []
        partial_events = []
        
        for event_id in all_events:
            stations_with_data = []
            for station in self.primary_stations:
                if station in file_mapping and str(event_id) in file_mapping[station]:
                    stations_with_data.append(station)
            
            if len(stations_with_data) == 8:
                complete_events.append(event_id)
            else:
                partial_events.append({
                    'event_id': event_id,
                    'available_stations': stations_with_data,
                    'missing_stations': [s for s in self.primary_stations if s not in stations_with_data]
                })
        
        logger.info(f"Events with complete 8-station data: {len(complete_events)}")
        logger.info(f"Events requiring zero-padding: {len(partial_events)}")
        
        self.crawl_results['station_completeness'] = {
            'total_events': len(all_events),
            'complete_8station_events': len(complete_events),
            'partial_events': len(partial_events),
            'completeness_percentage': len(complete_events) / len(all_events) * 100,
            'station_stats': completeness_stats,
            'complete_event_ids': complete_events[:10],  # Sample
            'partial_event_sample': partial_events[:5]   # Sample
        }
    
    def audit_magnitude_distribution(self):
        """Audit distribusi kelas magnitudo."""
        logger.info("Auditing magnitude distribution...")
        
        # Load earthquake catalog
        if not self.earthquake_catalog_path.exists():
            logger.error(f"Earthquake catalog not found: {self.earthquake_catalog_path}")
            return
        
        catalog_df = pd.read_csv(self.earthquake_catalog_path)
        
        # Get available event IDs
        available_events = set()
        for station_data in self.crawl_results['file_mapping'].values():
            available_events.update(int(eid) for eid in station_data.keys())
        
        # Filter catalog for available events
        available_catalog = catalog_df[catalog_df['event_id'].isin(available_events)]
        
        if len(available_catalog) == 0:
            logger.warning("No matching events for magnitude analysis")
            return
        
        # Define magnitude classes
        magnitude_bins = [0, 4.0, 4.5, 5.0, 5.5, 10.0]
        magnitude_labels = ['Small', 'Normal', 'Moderate', 'Medium', 'Large']
        
        available_catalog['magnitude_class'] = pd.cut(
            available_catalog['magnitude'], 
            bins=magnitude_bins, 
            labels=magnitude_labels, 
            include_lowest=True
        )
        
        # Count distribution
        class_counts = available_catalog['magnitude_class'].value_counts()
        
        logger.info("Magnitude distribution:")
        for mag_class, count in class_counts.items():
            percentage = count / len(available_catalog) * 100
            logger.info(f"  {mag_class}: {count} events ({percentage:.1f}%)")
        
        # Statistics
        mag_stats = {
            'mean': float(available_catalog['magnitude'].mean()),
            'std': float(available_catalog['magnitude'].std()),
            'min': float(available_catalog['magnitude'].min()),
            'max': float(available_catalog['magnitude'].max()),
            'median': float(available_catalog['magnitude'].median())
        }
        
        logger.info(f"Magnitude statistics: Mean={mag_stats['mean']:.2f}, Std={mag_stats['std']:.2f}")
        
        self.crawl_results['magnitude_distribution'] = {
            'total_events': len(available_catalog),
            'class_distribution': class_counts.to_dict(),
            'statistics': mag_stats,
            'magnitude_bins': magnitude_bins,
            'magnitude_labels': magnitude_labels
        }
    
    def audit_noise_patterns(self):
        """Audit pola noise pada sampel acak."""
        logger.info("Auditing noise patterns (random sampling)...")
        
        file_mapping = self.crawl_results['file_mapping']
        
        # Collect sample files for analysis
        sample_files = []
        for station, events in file_mapping.items():
            for event_id, files in events.items():
                if len(sample_files) < 10:  # Limit to 10 samples
                    sample_files.extend(files[:1])  # Take first file per event
        
        if not sample_files:
            logger.warning("No files available for noise analysis")
            return
        
        logger.info(f"Analyzing {len(sample_files)} sample files...")
        
        noise_stats = []
        for i, file_info in enumerate(sample_files[:10]):
            try:
                file_path = Path(file_info['path'])
                
                if file_path.suffix.lower() == '.npy':
                    data = np.load(file_path)
                elif file_path.suffix.lower() == '.png':
                    data = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                    data = data.astype(np.float32) / 255.0
                else:
                    continue
                
                # Calculate statistics
                stats = {
                    'file': file_path.name,
                    'shape': list(data.shape),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'range': float(np.max(data) - np.min(data)),
                    'non_zero_percentage': float(np.count_nonzero(data) / data.size * 100)
                }
                
                noise_stats.append(stats)
                
                logger.info(f"  Sample {i+1}: {stats['file']}")
                logger.info(f"    Shape: {stats['shape']}, Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                logger.info(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
                
            except Exception as e:
                logger.error(f"Error analyzing {file_info['path']}: {e}")
        
        # Aggregate statistics
        if noise_stats:
            aggregate_stats = {
                'sample_count': len(noise_stats),
                'mean_range': {
                    'min': min(s['mean'] for s in noise_stats),
                    'max': max(s['mean'] for s in noise_stats),
                    'avg': sum(s['mean'] for s in noise_stats) / len(noise_stats)
                },
                'std_range': {
                    'min': min(s['std'] for s in noise_stats),
                    'max': max(s['std'] for s in noise_stats),
                    'avg': sum(s['std'] for s in noise_stats) / len(noise_stats)
                },
                'data_range': {
                    'min': min(s['range'] for s in noise_stats),
                    'max': max(s['range'] for s in noise_stats),
                    'avg': sum(s['range'] for s in noise_stats) / len(noise_stats)
                }
            }
            
            # Assessment
            assessment = self.assess_noise_quality(aggregate_stats)
            
            self.crawl_results['noise_audit'] = {
                'sample_statistics': noise_stats,
                'aggregate_statistics': aggregate_stats,
                'quality_assessment': assessment
            }
        
    def assess_noise_quality(self, stats):
        """Assess data quality based on noise statistics."""
        assessment = {
            'quality_indicators': [],
            'warning_indicators': [],
            'overall_quality': 'unknown'
        }
        
        avg_mean = stats['mean_range']['avg']
        avg_std = stats['std_range']['avg']
        avg_range = stats['data_range']['avg']
        
        # Quality indicators
        if 0.01 < avg_range < 10.0:
            assessment['quality_indicators'].append('Realistic data range for geomagnetic scalograms')
        
        if abs(avg_mean) < 1.0:
            assessment['quality_indicators'].append('Reasonable mean values')
        
        if 0.1 < avg_std < 2.0:
            assessment['quality_indicators'].append('Appropriate noise distribution')
        
        # Warning indicators
        if avg_range < 0.01:
            assessment['warning_indicators'].append('Suspiciously small data range')
        
        if avg_range > 10.0:
            assessment['warning_indicators'].append('Unusually large data range')
        
        if abs(avg_mean) > 5.0:
            assessment['warning_indicators'].append('Extreme mean values detected')
        
        # Overall assessment
        quality_score = len(assessment['quality_indicators'])
        warning_score = len(assessment['warning_indicators'])
        
        if quality_score >= 2 and warning_score == 0:
            assessment['overall_quality'] = 'good'
        elif quality_score >= 1 and warning_score <= 1:
            assessment['overall_quality'] = 'acceptable'
        else:
            assessment['overall_quality'] = 'poor'
        
        return assessment
    
    def generate_readiness_summary(self):
        """Generate dataset readiness summary."""
        logger.info("Generating dataset readiness summary...")
        
        # Collect key metrics
        temporal_audit = self.crawl_results.get('temporal_audit', {})
        station_audit = self.crawl_results.get('station_completeness', {})
        magnitude_audit = self.crawl_results.get('magnitude_distribution', {})
        noise_audit = self.crawl_results.get('noise_audit', {})
        
        # Calculate readiness scores
        readiness_scores = {}
        
        # Temporal readiness
        train_events = temporal_audit.get('train_test_split', {}).get('train_events', 0)
        test_events = temporal_audit.get('train_test_split', {}).get('test_events', 0)
        
        if train_events >= 50 and test_events >= 10:
            readiness_scores['temporal'] = 'ready'
        elif train_events >= 20 and test_events >= 5:
            readiness_scores['temporal'] = 'limited'
        else:
            readiness_scores['temporal'] = 'insufficient'
        
        # Station completeness readiness
        complete_percentage = station_audit.get('completeness_percentage', 0)
        
        if complete_percentage >= 70:
            readiness_scores['station_completeness'] = 'ready'
        elif complete_percentage >= 40:
            readiness_scores['station_completeness'] = 'limited'
        else:
            readiness_scores['station_completeness'] = 'insufficient'
        
        # Data quality readiness
        noise_quality = noise_audit.get('quality_assessment', {}).get('overall_quality', 'unknown')
        readiness_scores['data_quality'] = noise_quality
        
        # Overall readiness
        ready_count = sum(1 for score in readiness_scores.values() if score in ['ready', 'good'])
        total_checks = len(readiness_scores)
        
        if ready_count == total_checks:
            overall_readiness = 'READY'
        elif ready_count >= total_checks * 0.7:
            overall_readiness = 'MOSTLY_READY'
        else:
            overall_readiness = 'NOT_READY'
        
        # Summary
        summary = {
            'overall_readiness': overall_readiness,
            'readiness_scores': readiness_scores,
            'key_metrics': {
                'total_events': temporal_audit.get('total_events', 0),
                'train_events': train_events,
                'test_events': test_events,
                'complete_8station_events': station_audit.get('complete_8station_events', 0),
                'station_completeness_percentage': complete_percentage,
                'data_quality': noise_quality
            },
            'recommendations': []
        }
        
        # Generate recommendations
        if readiness_scores.get('temporal') != 'ready':
            summary['recommendations'].append('Increase temporal coverage, especially for test period (July 2024-2026)')
        
        if readiness_scores.get('station_completeness') != 'ready':
            summary['recommendations'].append('Improve station data completeness or implement robust zero-padding strategy')
        
        if readiness_scores.get('data_quality') not in ['ready', 'good']:
            summary['recommendations'].append('Review data preprocessing pipeline for noise reduction')
        
        self.crawl_results['readiness_summary'] = summary
        
        # Log summary
        logger.info("=" * 50)
        logger.info(f"DATASET READINESS: {overall_readiness}")
        logger.info("=" * 50)
        logger.info(f"Total Events: {summary['key_metrics']['total_events']}")
        logger.info(f"Train/Test Split: {train_events}/{test_events}")
        logger.info(f"Complete 8-Station Events: {summary['key_metrics']['complete_8station_events']}")
        logger.info(f"Station Completeness: {complete_percentage:.1f}%")
        logger.info(f"Data Quality: {noise_quality}")
        
        if summary['recommendations']:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
    
    def save_crawl_report(self, output_path='dataset_readiness_report.json'):
        """Save crawl report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.crawl_results, f, indent=2, default=str)
        logger.info(f"Dataset readiness report saved: {output_path}")


def main():
    """Main function."""
    print("REAL DATA CRAWLER")
    print("Scalogramv3 Data Mapping & Audit")
    print("=" * 60)
    
    # Run crawling
    crawler = RealDataCrawler()
    results = crawler.run_complete_crawl()
    
    # Save report
    crawler.save_crawl_report()
    
    # Print summary
    readiness = results.get('readiness_summary', {})
    overall_status = readiness.get('overall_readiness', 'UNKNOWN')
    
    print(f"\nDATASET READINESS STATUS: {overall_status}")
    
    if overall_status == 'NOT_READY':
        print("\n" + "!" * 60)
        print("PERINGATAN: Dataset belum siap untuk production training!")
        print("Periksa laporan detail untuk rekomendasi perbaikan.")
        print("!" * 60)
    
    return overall_status


if __name__ == '__main__':
    main()