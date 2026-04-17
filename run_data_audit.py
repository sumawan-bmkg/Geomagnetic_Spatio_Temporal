#!/usr/bin/env python3
"""
Data Audit Runner Script
Runs comprehensive data audit for earthquake precursor analysis with predefined paths.
"""
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.data_auditor import DataAuditor


def main():
    """Main function to run data audit with predefined paths."""
    print("=" * 80)
    print("SPATIO-TEMPORAL EARTHQUAKE PRECURSOR DATA AUDIT")
    print("Automated Data Auditing Script")
    print("=" * 80)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'data_audit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting automated data audit")
    
    # Define data paths (relative to project root)
    data_paths = {
        'earthquake_catalog': '../awal/earthquake_catalog_2018_2025_merged.csv',
        'kp_index': '../awal/kp_index_2018_2026.csv',
        'station_locations': '../awal/lokasi_stasiun.csv',
        'scalogram_base': '../scalogramv3'  # Will check if exists
    }
    
    # Verify data files
    print("\n📋 VERIFYING DATA FILES...")
    missing_files = []
    existing_files = []
    
    for key, path in data_paths.items():
        if os.path.exists(path):
            file_size = os.path.getsize(path) / 1024  # KB
            existing_files.append(f"✅ {key}: {path} ({file_size:.1f} KB)")
        else:
            missing_files.append(f"❌ {key}: {path}")
    
    for file_info in existing_files:
        print(f"   {file_info}")
    
    if missing_files:
        print(f"\n⚠️  Missing files:")
        for file_info in missing_files:
            print(f"   {file_info}")
        
        # Check if critical files are missing
        critical_files = ['earthquake_catalog', 'kp_index', 'station_locations']
        missing_critical = [key for key in critical_files if not os.path.exists(data_paths[key])]
        
        if missing_critical:
            print(f"\n❌ Critical files missing: {missing_critical}")
            print("Cannot proceed without these files.")
            return False
        else:
            print(f"\n✅ All critical files present. Proceeding without scalogram data...")
    
    try:
        # Initialize DataAuditor
        print(f"\n🔧 INITIALIZING DATA AUDITOR...")
        
        scalogram_path = data_paths['scalogram_base'] if os.path.exists(data_paths['scalogram_base']) else None
        if scalogram_path is None:
            print("   Note: Scalogram directory not found, will skip file matching")
        
        auditor = DataAuditor(
            earthquake_catalog_path=data_paths['earthquake_catalog'],
            kp_index_path=data_paths['kp_index'],
            station_locations_path=data_paths['station_locations'],
            scalogram_base_path=scalogram_path
        )
        
        # Create output directory
        output_dir = 'outputs/data_audit'
        os.makedirs(output_dir, exist_ok=True)
        print(f"   Output directory: {output_dir}")
        
        # Run complete audit
        print(f"\n🚀 RUNNING COMPLETE DATA AUDIT...")
        saved_files = auditor.run_complete_audit(output_dir)
        
        # Generate detailed report
        print(f"\n📊 GENERATING DETAILED REPORT...")
        generate_detailed_report(auditor, output_dir)
        
        # Display final summary
        print_final_summary(auditor, saved_files)
        
        logger.info("Data audit completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Data audit failed: {e}")
        print(f"\n❌ DATA AUDIT FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_detailed_report(auditor, output_dir):
    """Generate detailed audit report."""
    report_path = os.path.join(output_dir, 'detailed_audit_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SPATIO-TEMPORAL EARTHQUAKE PRECURSOR DATA AUDIT REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data sources
        f.write("DATA SOURCES:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Earthquake Catalog: {auditor.earthquake_catalog_path}\n")
        f.write(f"Kp Index: {auditor.kp_index_path}\n")
        f.write(f"Station Locations: {auditor.station_locations_path}\n")
        f.write(f"Scalogram Base Path: {auditor.scalogram_base_path or 'Not provided'}\n\n")
        
        # Chronological split details
        f.write("CHRONOLOGICAL SPLIT:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Train Period: 2018-01-01 to 2024-06-30\n")
        f.write(f"Test Period: 2024-07-01 to 2026-12-31\n")
        f.write(f"Train Events: {len(auditor.train_events)}\n")
        f.write(f"Test Events: {len(auditor.test_events)}\n\n")
        
        # Station mapping details
        f.write("STATION MAPPING:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Stations: {len(auditor.station_data)}\n")
        f.write(f"Station Codes: {', '.join(auditor.station_data['Kode Stasiun'].tolist())}\n")
        f.write(f"Total Mappings: {len(auditor.station_event_mapping)}\n")
        f.write(f"Unique Events Mapped: {len(auditor.station_event_mapping['event_id'].unique())}\n\n")
        
        # Magnitude distribution
        f.write("MAGNITUDE DISTRIBUTION:\n")
        f.write("-" * 20 + "\n")
        import pandas as pd
        mag_bins = [0, 4.0, 5.0, 6.0, 7.0, 10.0]
        mag_labels = ['<4.0', '4.0-4.9', '5.0-5.9', '6.0-6.9', '>=7.0']
        
        for split_name, split_data in [("Train", auditor.train_events), ("Test", auditor.test_events)]:
            if len(split_data) > 0:
                f.write(f"\n{split_name} Set Magnitude Distribution:\n")
                mag_dist = pd.cut(split_data['Magnitude'], bins=mag_bins, labels=mag_labels, include_lowest=True)
                for label, count in mag_dist.value_counts().sort_index().items():
                    percentage = (count / len(split_data)) * 100
                    f.write(f"  {label}: {count} events ({percentage:.1f}%)\n")
        
        # Station coverage
        f.write("\nSTATION COVERAGE:\n")
        f.write("-" * 20 + "\n")
        station_counts = auditor.station_event_mapping['station_code'].value_counts()
        for station, count in station_counts.items():
            percentage = (count / len(auditor.station_event_mapping)) * 100
            f.write(f"{station}: {count} mappings ({percentage:.1f}%)\n")
        
        # Data availability
        f.write(f"\nDATA AVAILABILITY:\n")
        f.write("-" * 20 + "\n")
        total_records = len(auditor.master_metadata)
        with_data = len(auditor.master_metadata[auditor.master_metadata['has_scalogram_data']])
        synchronized = len(auditor.master_metadata[auditor.master_metadata['data_synchronized']])
        
        f.write(f"Total Metadata Records: {total_records}\n")
        f.write(f"Records with Scalogram Data: {with_data} ({with_data/total_records*100:.1f}%)\n")
        f.write(f"Synchronized Records: {synchronized} ({synchronized/total_records*100:.1f}%)\n")
    
    print(f"   Detailed report saved: {report_path}")


def print_final_summary(auditor, saved_files):
    """Print final summary of audit results."""
    print(f"\n" + "="*80)
    print("🎉 DATA AUDIT COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Key statistics
    print(f"\n📈 KEY STATISTICS:")
    print(f"   📅 Train Events: {len(auditor.train_events)} (2018-2024)")
    print(f"   📅 Test Events: {len(auditor.test_events)} (2024-2026)")
    print(f"   🗺️  Station-Event Mappings: {len(auditor.station_event_mapping)}")
    print(f"   📊 Metadata Records: {len(auditor.master_metadata)}")
    
    # Data availability
    total_records = len(auditor.master_metadata)
    with_data = len(auditor.master_metadata[auditor.master_metadata['has_scalogram_data']])
    print(f"   📁 Records with Data: {with_data}/{total_records} ({with_data/total_records*100:.1f}%)")
    
    # Dobrovolsky radius examples
    print(f"\n🌍 DOBROVOLSKY RADIUS EXAMPLES:")
    example_mags = [4.0, 5.0, 6.0, 7.0]
    for mag in example_mags:
        radius = auditor.calculate_dobrovolsky_radius(mag)
        print(f"   M{mag}: {radius:.1f} km precursor radius")
    
    # Output files
    print(f"\n📁 OUTPUT FILES:")
    for key, path in saved_files.items():
        file_size = os.path.getsize(path) / 1024
        print(f"   {key}: {path} ({file_size:.1f} KB)")
    
    # Next steps
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Review master_metadata.csv for complete dataset information")
    print(f"   2. Use station_event_mappings.csv for spatial analysis")
    print(f"   3. Process train/test splits for model development")
    print(f"   4. Integrate with scalogram processing pipeline")
    print(f"   5. Check detailed_audit_report.txt for comprehensive analysis")
    
    print(f"\n✅ Master metadata file ready: outputs/data_audit/master_metadata.csv")


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)