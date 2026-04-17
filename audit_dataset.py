#!/usr/bin/env python3
"""
Dataset Audit Script - Deep Verification of Real vs Synthetic Data

Script ini melakukan audit mendalam terhadap spatio_earthquake_dataset.h5
untuk memverifikasi keaslian data (Real BMKG vs Synthetic Demo).
"""
import sys
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetAuditor:
    """
    Auditor untuk memverifikasi keaslian dataset HDF5.
    """
    
    def __init__(self, dataset_path='real_earthquake_dataset.h5'):
        self.dataset_path = Path(dataset_path)
        self.earthquake_catalog_path = Path('../awal/earthquake_catalog_2018_2025_merged.csv')
        self.station_coords_path = Path('../awal/lokasi_stasiun.csv')
        
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'file_exists': False,
            'metadata_check': {},
            'event_id_check': {},
            'statistical_check': {},
            'station_mapping_check': {},
            'final_verdict': 'UNKNOWN'
        }
    
    def run_complete_audit(self):
        """Jalankan audit lengkap."""
        logger.info("=" * 60)
        logger.info("DATASET AUDIT - REAL vs SYNTHETIC VERIFICATION")
        logger.info("=" * 60)
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            logger.error(f"Dataset tidak ditemukan: {self.dataset_path}")
            self.audit_results['final_verdict'] = 'FILE_NOT_FOUND'
            return self.audit_results
        
        self.audit_results['file_exists'] = True
        logger.info(f"Dataset ditemukan: {self.dataset_path}")
        
        try:
            # 1. Metadata Inspection
            logger.info("\n1. METADATA INSPECTION")
            logger.info("-" * 30)
            self.inspect_metadata()
            
            # 2. Event ID Verification
            logger.info("\n2. EVENT ID VERIFICATION")
            logger.info("-" * 30)
            self.verify_event_ids()
            
            # 3. Statistical Pattern Check
            logger.info("\n3. STATISTICAL PATTERN CHECK")
            logger.info("-" * 30)
            self.check_statistical_patterns()
            
            # 4. Station Mapping Check
            logger.info("\n4. STATION MAPPING CHECK")
            logger.info("-" * 30)
            self.check_station_mapping()
            
            # 5. Final Verdict
            logger.info("\n5. FINAL VERDICT")
            logger.info("-" * 30)
            self.determine_final_verdict()
            
        except Exception as e:
            logger.error(f"Error during audit: {e}")
            self.audit_results['final_verdict'] = 'ERROR'
        
        return self.audit_results
    
    def inspect_metadata(self):
        """Periksa metadata dan atribut HDF5."""
        with h5py.File(self.dataset_path, 'r') as f:
            # Check file attributes
            file_attrs = dict(f.attrs)
            logger.info("File attributes:")
            for key, value in file_attrs.items():
                logger.info(f"  {key}: {value}")
            
            # Check for synthetic flags
            synthetic_flags = ['is_synthetic', 'data_source', 'synthetic', 'demo', 'test']
            found_flags = []
            
            for flag in synthetic_flags:
                if flag in file_attrs:
                    found_flags.append((flag, file_attrs[flag]))
            
            # Check creation method from attributes
            creation_time = file_attrs.get('creation_time', 'unknown')
            tensor_shape = file_attrs.get('tensor_shape', 'unknown')
            
            # Check groups and datasets
            logger.info("\nDataset structure:")
            def print_structure(name, obj):
                logger.info(f"  {name}: {type(obj).__name__}")
            
            f.visititems(print_structure)
            
            # Store results
            self.audit_results['metadata_check'] = {
                'file_attributes': file_attrs,
                'synthetic_flags_found': found_flags,
                'creation_time': creation_time,
                'tensor_shape': tensor_shape,
                'has_suspicious_flags': len(found_flags) > 0
            }
            
            if found_flags:
                logger.warning(f"SUSPICIOUS: Found synthetic flags: {found_flags}")
            else:
                logger.info("No obvious synthetic flags found in metadata")
    
    def verify_event_ids(self):
        """Verifikasi Event ID dengan katalog resmi BMKG."""
        # Load earthquake catalog
        if not self.earthquake_catalog_path.exists():
            logger.error(f"Katalog gempa tidak ditemukan: {self.earthquake_catalog_path}")
            self.audit_results['event_id_check']['catalog_available'] = False
            return
        
        catalog_df = pd.read_csv(self.earthquake_catalog_path)
        official_event_ids = set(catalog_df['event_id'].values)
        logger.info(f"Katalog resmi memiliki {len(official_event_ids)} event IDs")
        
        # Load dataset event IDs
        with h5py.File(self.dataset_path, 'r') as f:
            if 'config' in f and 'valid_events' in f['config']:
                dataset_event_ids = f['config']['valid_events'][:]
            elif 'metadata' in f and 'event_id' in f['metadata']:
                dataset_event_ids = f['metadata']['event_id'][:]
            else:
                logger.error("Tidak dapat menemukan event IDs dalam dataset")
                self.audit_results['event_id_check']['ids_available'] = False
                return
        
        logger.info(f"Dataset memiliki {len(dataset_event_ids)} event IDs")
        
        # Sample 5 random event IDs for verification
        sample_size = min(5, len(dataset_event_ids))
        sample_indices = np.random.choice(len(dataset_event_ids), sample_size, replace=False)
        sample_event_ids = [dataset_event_ids[i] for i in sample_indices]
        
        logger.info(f"Memeriksa {sample_size} event IDs secara acak:")
        
        verification_results = []
        for event_id in sample_event_ids:
            is_in_catalog = event_id in official_event_ids
            logger.info(f"  Event ID {event_id}: {'FOUND' if is_in_catalog else 'NOT FOUND'} in catalog")
            verification_results.append({
                'event_id': int(event_id),
                'found_in_catalog': is_in_catalog
            })
        
        # Calculate statistics
        found_count = sum(1 for r in verification_results if r['found_in_catalog'])
        found_percentage = (found_count / len(verification_results)) * 100
        
        self.audit_results['event_id_check'] = {
            'catalog_available': True,
            'ids_available': True,
            'total_catalog_events': len(official_event_ids),
            'total_dataset_events': len(dataset_event_ids),
            'sample_size': sample_size,
            'sample_results': verification_results,
            'found_in_catalog': found_count,
            'found_percentage': found_percentage,
            'likely_real': found_percentage >= 80  # 80% threshold
        }
        
        if found_percentage >= 80:
            logger.info(f"GOOD: {found_percentage:.1f}% event IDs ditemukan dalam katalog resmi")
        else:
            logger.warning(f"SUSPICIOUS: Hanya {found_percentage:.1f}% event IDs ditemukan dalam katalog resmi")
    
    def check_statistical_patterns(self):
        """Periksa pola statistik data tensor."""
        with h5py.File(self.dataset_path, 'r') as f:
            # Load tensor data
            if 'cmr_tensor' in f:
                tensor_data = f['cmr_tensor']
            elif 'original_tensor' in f:
                tensor_data = f['original_tensor']
            else:
                logger.error("Tidak dapat menemukan tensor data")
                self.audit_results['statistical_check']['tensor_available'] = False
                return
            
            # Get tensor shape
            tensor_shape = tensor_data.shape
            logger.info(f"Tensor shape: {tensor_shape}")
            
            # Load station list to find GTO index
            station_list = []
            if 'config' in f and 'station_list' in f['config']:
                station_list = [s.decode('utf-8') for s in f['config']['station_list'][:]]
            
            logger.info(f"Available stations: {station_list}")
            
            # Find GTO station index
            gto_index = None
            if 'GTO' in station_list:
                gto_index = station_list.index('GTO')
                logger.info(f"GTO station found at index: {gto_index}")
            else:
                logger.warning("GTO station not found, using first available station")
                gto_index = 0
                if station_list:
                    logger.info(f"Using station: {station_list[0]} at index 0")
            
            # Sample random event and component
            if len(tensor_shape) >= 4:  # (B, S, C, F, T) or similar
                random_event = np.random.randint(0, tensor_shape[0])
                random_component = np.random.randint(0, min(3, tensor_shape[2]))  # Assume max 3 components
                
                # Extract sample tensor
                sample_tensor = tensor_data[random_event, gto_index, random_component, :, :]
                
                # Calculate statistics
                stats = {
                    'mean': float(np.mean(sample_tensor)),
                    'std': float(np.std(sample_tensor)),
                    'min': float(np.min(sample_tensor)),
                    'max': float(np.max(sample_tensor)),
                    'shape': sample_tensor.shape
                }
                
                logger.info(f"Sample tensor statistics (Event {random_event}, Station {gto_index}, Component {random_component}):")
                logger.info(f"  Shape: {stats['shape']}")
                logger.info(f"  Mean: {stats['mean']:.6f}")
                logger.info(f"  Std: {stats['std']:.6f}")
                logger.info(f"  Min: {stats['min']:.6f}")
                logger.info(f"  Max: {stats['max']:.6f}")
                
                # Analyze patterns
                analysis = self.analyze_statistical_patterns(stats)
                
                self.audit_results['statistical_check'] = {
                    'tensor_available': True,
                    'tensor_shape': tensor_shape,
                    'sample_event': random_event,
                    'sample_station_index': gto_index,
                    'sample_station_name': station_list[gto_index] if gto_index < len(station_list) else 'unknown',
                    'sample_component': random_component,
                    'statistics': stats,
                    'analysis': analysis
                }
                
            else:
                logger.error(f"Unexpected tensor shape: {tensor_shape}")
                self.audit_results['statistical_check']['tensor_available'] = False
    
    def analyze_statistical_patterns(self, stats):
        """Analisis pola statistik untuk menentukan real vs synthetic."""
        analysis = {
            'likely_synthetic_indicators': [],
            'likely_real_indicators': [],
            'overall_assessment': 'unknown'
        }
        
        mean = stats['mean']
        std = stats['std']
        min_val = stats['min']
        max_val = stats['max']
        
        # Check for synthetic patterns
        # 1. Perfect normal distribution (mean ~0, std ~1)
        if abs(mean) < 0.1 and abs(std - 1.0) < 0.1:
            analysis['likely_synthetic_indicators'].append('Perfect normal distribution (mean≈0, std≈1)')
        
        # 2. Too perfect range
        if abs(min_val + max_val) < 0.1:  # Symmetric around zero
            analysis['likely_synthetic_indicators'].append('Perfectly symmetric range around zero')
        
        # 3. Suspiciously round numbers
        if all(abs(val - round(val, 1)) < 1e-6 for val in [mean, std, min_val, max_val]):
            analysis['likely_synthetic_indicators'].append('Suspiciously round statistical values')
        
        # 4. Unrealistic range for geomagnetic data
        if max_val - min_val < 0.01:  # Too small range
            analysis['likely_synthetic_indicators'].append('Unrealistically small data range')
        
        # Check for real data patterns
        # 1. Complex statistical distribution
        if abs(mean) > 0.01 or abs(std - 1.0) > 0.2:
            analysis['likely_real_indicators'].append('Non-perfect statistical distribution')
        
        # 2. Realistic geomagnetic range
        if 0.1 < (max_val - min_val) < 10.0:
            analysis['likely_real_indicators'].append('Realistic geomagnetic data range')
        
        # 3. Asymmetric distribution
        if abs(min_val + max_val) > 0.1:
            analysis['likely_real_indicators'].append('Asymmetric data distribution')
        
        # Overall assessment
        synthetic_score = len(analysis['likely_synthetic_indicators'])
        real_score = len(analysis['likely_real_indicators'])
        
        if synthetic_score > real_score:
            analysis['overall_assessment'] = 'likely_synthetic'
        elif real_score > synthetic_score:
            analysis['overall_assessment'] = 'likely_real'
        else:
            analysis['overall_assessment'] = 'inconclusive'
        
        logger.info(f"Statistical analysis:")
        logger.info(f"  Synthetic indicators: {synthetic_score}")
        logger.info(f"  Real indicators: {real_score}")
        logger.info(f"  Assessment: {analysis['overall_assessment']}")
        
        return analysis
    
    def check_station_mapping(self):
        """Periksa mapping stasiun dengan koordinat resmi."""
        # Load official station coordinates
        if not self.station_coords_path.exists():
            logger.error(f"File koordinat stasiun tidak ditemukan: {self.station_coords_path}")
            self.audit_results['station_mapping_check']['coords_available'] = False
            return
        
        coords_df = pd.read_csv(self.station_coords_path, sep=';')
        official_stations = set(coords_df['Kode Stasiun'].dropna().values)
        logger.info(f"Koordinat resmi memiliki {len(official_stations)} stasiun")
        
        # Load dataset stations
        with h5py.File(self.dataset_path, 'r') as f:
            if 'config' in f and 'station_list' in f['config']:
                dataset_stations = [s.decode('utf-8') for s in f['config']['station_list'][:]]
            else:
                logger.error("Tidak dapat menemukan daftar stasiun dalam dataset")
                self.audit_results['station_mapping_check']['stations_available'] = False
                return
        
        logger.info(f"Dataset memiliki {len(dataset_stations)} stasiun: {dataset_stations}")
        
        # Check mapping
        mapping_results = []
        for station in dataset_stations:
            is_official = station in official_stations
            mapping_results.append({
                'station_code': station,
                'found_in_official': is_official
            })
            logger.info(f"  {station}: {'OFFICIAL' if is_official else 'NOT FOUND'}")
        
        # Calculate statistics
        found_count = sum(1 for r in mapping_results if r['found_in_official'])
        found_percentage = (found_count / len(mapping_results)) * 100 if mapping_results else 0
        
        self.audit_results['station_mapping_check'] = {
            'coords_available': True,
            'stations_available': True,
            'official_stations_count': len(official_stations),
            'dataset_stations_count': len(dataset_stations),
            'mapping_results': mapping_results,
            'found_in_official': found_count,
            'found_percentage': found_percentage,
            'mapping_valid': found_percentage >= 80
        }
        
        if found_percentage >= 80:
            logger.info(f"GOOD: {found_percentage:.1f}% stasiun ditemukan dalam koordinat resmi")
        else:
            logger.warning(f"SUSPICIOUS: Hanya {found_percentage:.1f}% stasiun ditemukan dalam koordinat resmi")
    
    def determine_final_verdict(self):
        """Tentukan verdict akhir berdasarkan semua pemeriksaan."""
        synthetic_indicators = 0
        real_indicators = 0
        
        # Check metadata
        if self.audit_results['metadata_check'].get('has_suspicious_flags', False):
            synthetic_indicators += 2
            logger.info("[-] Metadata: Found synthetic flags")
        else:
            real_indicators += 1
            logger.info("[+] Metadata: No synthetic flags")
        
        # Check event IDs
        event_check = self.audit_results['event_id_check']
        if event_check.get('likely_real', False):
            real_indicators += 2
            logger.info("[+] Event IDs: Found in official catalog")
        else:
            synthetic_indicators += 2
            logger.info("[-] Event IDs: Not found in official catalog")
        
        # Check statistics
        stat_check = self.audit_results['statistical_check']
        if stat_check.get('analysis', {}).get('overall_assessment') == 'likely_real':
            real_indicators += 1
            logger.info("[+] Statistics: Patterns suggest real data")
        elif stat_check.get('analysis', {}).get('overall_assessment') == 'likely_synthetic':
            synthetic_indicators += 1
            logger.info("[-] Statistics: Patterns suggest synthetic data")
        
        # Check station mapping
        station_check = self.audit_results['station_mapping_check']
        if station_check.get('mapping_valid', False):
            real_indicators += 1
            logger.info("[+] Stations: Valid mapping to official coordinates")
        else:
            synthetic_indicators += 1
            logger.info("[-] Stations: Invalid mapping")
        
        # Final decision
        logger.info(f"\nScore: Real={real_indicators}, Synthetic={synthetic_indicators}")
        
        if synthetic_indicators > real_indicators:
            self.audit_results['final_verdict'] = 'DEMO'
        elif real_indicators > synthetic_indicators:
            self.audit_results['final_verdict'] = 'REAL'
        else:
            self.audit_results['final_verdict'] = 'INCONCLUSIVE'
        
        # Log final verdict
        verdict = self.audit_results['final_verdict']
        logger.info("=" * 60)
        logger.info(f"DATASET STATUS: {verdict}")
        logger.info("=" * 60)
        
        if verdict == 'DEMO':
            logger.warning("DATASET TERDETEKSI SEBAGAI SYNTHETIC/DEMO!")
            logger.warning("Perlu menjalankan generate_final_dataset.py dengan data asli.")
        elif verdict == 'REAL':
            logger.info("Dataset terverifikasi sebagai data asli BMKG.")
        else:
            logger.warning("Status dataset tidak dapat dipastikan.")
    
    def save_audit_report(self, output_path='dataset_audit_report.json'):
        """Simpan laporan audit."""
        with open(output_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        logger.info(f"Laporan audit disimpan: {output_path}")


def main():
    """Main function."""
    print("DATASET AUDIT TOOL")
    print("Verifikasi Real BMKG vs Synthetic Demo Data")
    print("=" * 60)
    
    # Run audit
    auditor = DatasetAuditor()
    results = auditor.run_complete_audit()
    
    # Save report
    auditor.save_audit_report()
    
    # Print final verdict
    verdict = results['final_verdict']
    print(f"\nFINAL VERDICT: {verdict}")
    
    if verdict == 'DEMO':
        print("\n" + "!" * 60)
        print("PERINGATAN: Dataset terdeteksi sebagai SYNTHETIC/DEMO!")
        print("Untuk menggunakan data asli, jalankan:")
        print("python generate_final_dataset.py")
        print("!" * 60)
    
    return verdict


if __name__ == '__main__':
    main()