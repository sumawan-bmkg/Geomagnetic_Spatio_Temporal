"""
Example usage of DataAuditor for earthquake precursor analysis.
Demonstrates complete data auditing workflow including chronological splitting,
station mapping, and master metadata generation.
"""
import os
import sys
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.data_auditor import DataAuditor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_audit_example.log')
        ]
    )


def example_data_audit():
    """Example of complete data audit workflow."""
    print("=" * 80)
    print("SPATIO-TEMPORAL EARTHQUAKE PRECURSOR DATA AUDIT")
    print("Complete Data Auditing Workflow Example")
    print("=" * 80)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting data audit example")
    
    # Define data paths (adjust these paths according to your setup)
    data_paths = {
        'earthquake_catalog': '../../awal/earthquake_catalog_2018_2025_merged.csv',
        'kp_index': '../../awal/kp_index_2018_2026.csv',
        'station_locations': '../../awal/lokasi_stasiun.csv',
        'scalogram_base': '../../scalogramv3'  # Adjust if scalogramv3 exists
    }
    
    # Check if files exist
    missing_files = []
    for key, path in data_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{key}: {path}")
    
    if missing_files:
        print("\n⚠️  Some data files are missing:")
        for missing in missing_files:
            print(f"   - {missing}")
        print("\nThe audit will continue with available files...")
    
    try:
        # Initialize DataAuditor
        print("\n" + "="*60)
        print("INITIALIZING DATA AUDITOR")
        print("="*60)
        
        auditor = DataAuditor(
            earthquake_catalog_path=data_paths['earthquake_catalog'],
            kp_index_path=data_paths['kp_index'],
            station_locations_path=data_paths['station_locations'],
            scalogram_base_path=data_paths['scalogram_base'] if os.path.exists(data_paths['scalogram_base']) else None
        )
        
        # Run complete audit
        output_dir = '../outputs/data_audit'
        saved_files = auditor.run_complete_audit(output_dir)
        
        # Display results summary
        print("\n" + "="*80)
        print("DATA AUDIT RESULTS SUMMARY")
        print("="*80)
        
        # Chronological split summary
        print(f"\n📅 CHRONOLOGICAL SPLIT:")
        print(f"   Train Events: {len(auditor.train_events)} (2018-01-01 to 2024-06-30)")
        print(f"   Test Events: {len(auditor.test_events)} (2024-07-01 to 2026-12-31)")
        
        # Station mapping summary
        print(f"\n🗺️  STATION MAPPING:")
        print(f"   Total Stations: {len(auditor.station_data)}")
        print(f"   Station-Event Mappings: {len(auditor.station_event_mapping)}")
        print(f"   Events with Station Coverage: {len(auditor.station_event_mapping['event_id'].unique())}")
        
        # Data availability summary
        print(f"\n📊 DATA AVAILABILITY:")
        total_records = len(auditor.master_metadata)
        with_data = len(auditor.master_metadata[auditor.master_metadata['has_scalogram_data']])
        synchronized = len(auditor.master_metadata[auditor.master_metadata['data_synchronized']])
        
        print(f"   Total Metadata Records: {total_records}")
        print(f"   Records with Scalogram Data: {with_data} ({with_data/total_records*100:.1f}%)")
        print(f"   Synchronized Records: {synchronized} ({synchronized/total_records*100:.1f}%)")
        
        # Dobrovolsky radius examples
        print(f"\n🌍 DOBROVOLSKY RADIUS EXAMPLES:")
        example_magnitudes = [4.0, 5.0, 6.0, 7.0]
        for mag in example_magnitudes:
            radius = auditor.calculate_dobrovolsky_radius(mag)
            print(f"   M{mag}: {radius:.1f} km")
        
        # Station coverage by magnitude
        print(f"\n📍 STATION COVERAGE BY MAGNITUDE:")
        mag_bins = [0, 4.0, 5.0, 6.0, 7.0, 10.0]
        mag_labels = ['<4.0', '4.0-4.9', '5.0-5.9', '6.0-6.9', '≥7.0']
        
        unique_events = auditor.station_event_mapping.drop_duplicates('event_id')
        import pandas as pd
        mag_dist = pd.cut(unique_events['magnitude'], bins=mag_bins, labels=mag_labels, include_lowest=True)
        
        for label in mag_labels:
            events_in_range = unique_events[mag_dist == label]
            if len(events_in_range) > 0:
                avg_stations = auditor.station_event_mapping[
                    auditor.station_event_mapping['event_id'].isin(events_in_range['event_id'])
                ].groupby('event_id').size().mean()
                print(f"   {label}: {len(events_in_range)} events, avg {avg_stations:.1f} stations/event")
        
        # Output files
        print(f"\n📁 OUTPUT FILES:")
        for key, path in saved_files.items():
            file_size = os.path.getsize(path) / 1024  # KB
            print(f"   {key}: {path} ({file_size:.1f} KB)")
        
        # Sample data preview
        print(f"\n🔍 SAMPLE MASTER METADATA (first 5 records):")
        sample_cols = ['event_id', 'station_code', 'magnitude', 'distance_km', 
                      'dobrovolsky_radius_km', 'has_scalogram_data', 'split']
        print(auditor.master_metadata[sample_cols].head().to_string(index=False))
        
        print(f"\n" + "="*80)
        print("✅ DATA AUDIT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Review master_metadata.csv for dataset paths")
        print(f"2. Use station_event_mappings.csv for spatial analysis")
        print(f"3. Process train/test splits for model development")
        print(f"4. Check audit_summary.json for detailed statistics")
        
        logger.info("Data audit example completed successfully")
        
        return auditor, saved_files
        
    except Exception as e:
        logger.error(f"Data audit failed: {e}")
        print(f"\n❌ Data audit failed with error: {e}")
        raise


def demonstrate_dobrovolsky_calculations():
    """Demonstrate Dobrovolsky radius calculations for different magnitudes."""
    print(f"\n" + "="*60)
    print("DOBROVOLSKY RADIUS DEMONSTRATION")
    print("="*60)
    
    print(f"\nFormula: R = 10^(0.43 * M) km")
    print(f"Where R is the precursor detection radius and M is magnitude")
    
    magnitudes = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    
    print(f"\n{'Magnitude':<10} {'Radius (km)':<12} {'Coverage Area (km²)':<20}")
    print("-" * 45)
    
    for mag in magnitudes:
        radius = 10 ** (0.43 * mag)
        area = 3.14159 * radius ** 2
        print(f"{mag:<10.1f} {radius:<12.1f} {area:<20,.0f}")
    
    print(f"\nNote: Larger earthquakes have larger precursor detection zones,")
    print(f"allowing stations further away to potentially detect precursory signals.")


def analyze_station_coverage():
    """Analyze theoretical station coverage for Indonesian region."""
    print(f"\n" + "="*60)
    print("STATION COVERAGE ANALYSIS")
    print("="*60)
    
    # Sample station coordinates (from lokasi_stasiun.csv)
    stations = {
        'SBG': (5.87679, 95.3382),
        'SCN': (-0.545875, 100.298),
        'KPY': (-3.67999, 102.582),
        'LWA': (-5.01744, 104.058),
        'LPS': (-5.7887, 105.583),
        'SRG': (-6.17132, 106.051),
        'SKB': (-7.07442, 106.531),
        'CLP': (-7.7194, 109.015),
        'YOG': (-7.73119, 110.354)
    }
    
    print(f"\nStation Network Coverage:")
    print(f"{'Station':<8} {'Latitude':<10} {'Longitude':<11} {'Region':<15}")
    print("-" * 50)
    
    for code, (lat, lon) in stations.items():
        if lat > 0:
            region = "Northern Sumatra"
        elif lat > -3:
            region = "Central Sumatra"
        elif lat > -6:
            region = "Southern Sumatra"
        else:
            region = "Java"
        
        print(f"{code:<8} {lat:<10.3f} {lon:<11.3f} {region:<15}")
    
    print(f"\nNetwork spans approximately:")
    lat_range = max(lat for lat, lon in stations.values()) - min(lat for lat, lon in stations.values())
    lon_range = max(lon for lat, lon in stations.values()) - min(lon for lat, lon in stations.values())
    print(f"- Latitude range: {lat_range:.1f}° ({lat_range * 111:.0f} km)")
    print(f"- Longitude range: {lon_range:.1f}° ({lon_range * 111:.0f} km)")


if __name__ == '__main__':
    # Run demonstrations
    demonstrate_dobrovolsky_calculations()
    analyze_station_coverage()
    
    # Run main audit example
    try:
        auditor, saved_files = example_data_audit()
    except Exception as e:
        print(f"Example failed: {e}")
        sys.exit(1)