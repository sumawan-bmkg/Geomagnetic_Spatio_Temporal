"""
Data Auditor Module for Spatio-Temporal Earthquake Precursor Analysis

This module performs comprehensive data auditing including:
- Chronological splitting of earthquake events (Train/Test)
- Station mapping based on Dobrovolsky radius
- Master metadata generation for scalogram datasets
- Data synchronization verification
"""
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DataAuditor:
    """
    Comprehensive data auditor for earthquake precursor analysis.
    
    Handles chronological splitting, station mapping, and metadata generation
    for spatio-temporal earthquake precursor datasets.
    """
    
    def __init__(self, 
                 earthquake_catalog_path: str,
                 kp_index_path: str,
                 station_locations_path: str,
                 scalogram_base_path: str = None):
        """
        Initialize DataAuditor.
        
        Args:
            earthquake_catalog_path: Path to earthquake catalog CSV
            kp_index_path: Path to Kp index CSV
            station_locations_path: Path to station locations CSV
            scalogram_base_path: Base path to scalogram datasets
        """
        self.earthquake_catalog_path = earthquake_catalog_path
        self.kp_index_path = kp_index_path
        self.station_locations_path = station_locations_path
        self.scalogram_base_path = scalogram_base_path
        
        # Chronological split parameters
        self.train_start = datetime(2018, 1, 1)
        self.train_end = datetime(2024, 6, 30, 23, 59, 59)
        self.test_start = datetime(2024, 7, 1)
        self.test_end = datetime(2026, 12, 31, 23, 59, 59)
        
        # Data containers
        self.earthquake_data = None
        self.kp_data = None
        self.station_data = None
        self.master_metadata = None
        
        # Results
        self.train_events = None
        self.test_events = None
        self.station_event_mapping = None
        
    def load_data(self) -> None:
        """Load all required datasets."""
        logger.info("Loading earthquake catalog...")
        self.earthquake_data = pd.read_csv(self.earthquake_catalog_path)
        
        # Convert datetime column
        self.earthquake_data['datetime'] = pd.to_datetime(self.earthquake_data['datetime'])
        
        logger.info(f"Loaded {len(self.earthquake_data)} earthquake events")
        
        logger.info("Loading Kp index data...")
        self.kp_data = pd.read_csv(self.kp_index_path)
        self.kp_data['Date_Time_UTC'] = pd.to_datetime(self.kp_data['Date_Time_UTC'])
        
        logger.info(f"Loaded {len(self.kp_data)} Kp index records")
        
        logger.info("Loading station locations...")
        # Handle different possible separators
        try:
            self.station_data = pd.read_csv(self.station_locations_path, sep=';')
        except:
            self.station_data = pd.read_csv(self.station_locations_path, sep=',')
        
        # Clean column names and data
        self.station_data.columns = self.station_data.columns.str.strip()
        
        # Remove rows with missing station codes
        self.station_data = self.station_data.dropna(subset=['Kode Stasiun'])
        self.station_data = self.station_data[self.station_data['Kode Stasiun'].str.strip() != '']
        
        # Clean and convert numeric columns
        for col in ['Latitude', 'Longitude']:
            if col in self.station_data.columns:
                self.station_data[col] = pd.to_numeric(self.station_data[col], errors='coerce')
        
        # Remove rows with invalid coordinates
        self.station_data = self.station_data.dropna(subset=['Latitude', 'Longitude'])
        
        logger.info(f"Loaded {len(self.station_data)} stations")
        
        # Display data summaries
        self._display_data_summary()
    
    def _display_data_summary(self) -> None:
        """Display summary of loaded data."""
        print("\n" + "="*60)
        print("DATA SUMMARY")
        print("="*60)
        
        print(f"\nEarthquake Catalog:")
        print(f"- Total events: {len(self.earthquake_data)}")
        print(f"- Date range: {self.earthquake_data['datetime'].min()} to {self.earthquake_data['datetime'].max()}")
        print(f"- Magnitude range: {self.earthquake_data['Magnitude'].min():.1f} to {self.earthquake_data['Magnitude'].max():.1f}")
        
        print(f"\nKp Index Data:")
        print(f"- Total records: {len(self.kp_data)}")
        print(f"- Date range: {self.kp_data['Date_Time_UTC'].min()} to {self.kp_data['Date_Time_UTC'].max()}")
        print(f"- Kp range: {self.kp_data['Kp_Index'].min():.1f} to {self.kp_data['Kp_Index'].max():.1f}")
        
        print(f"\nStation Data:")
        print(f"- Total stations: {len(self.station_data)}")
        # Handle potential NaN values in station codes
        valid_stations = self.station_data['Kode Stasiun'].dropna().tolist()
        print("- Stations:", ", ".join(str(s) for s in valid_stations))
    
    def chronological_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split earthquake events chronologically into train and test sets.
        
        Returns:
            Tuple of (train_events, test_events) DataFrames
        """
        logger.info("Performing chronological split...")
        
        # Train set: 2018 - June 2024
        train_mask = (self.earthquake_data['datetime'] >= self.train_start) & \
                    (self.earthquake_data['datetime'] <= self.train_end)
        self.train_events = self.earthquake_data[train_mask].copy()
        
        # Test set: July 2024 - 2026
        test_mask = (self.earthquake_data['datetime'] >= self.test_start) & \
                   (self.earthquake_data['datetime'] <= self.test_end)
        self.test_events = self.earthquake_data[test_mask].copy()
        
        # Add split label
        self.train_events['split'] = 'train'
        self.test_events['split'] = 'test'
        
        logger.info(f"Train events: {len(self.train_events)} ({self.train_start.date()} to {self.train_end.date()})")
        logger.info(f"Test events: {len(self.test_events)} ({self.test_start.date()} to {self.test_end.date()})")
        
        # Display magnitude distribution
        self._display_split_statistics()
        
        return self.train_events, self.test_events
    
    def _display_split_statistics(self) -> None:
        """Display statistics for train/test split."""
        print(f"\n" + "="*60)
        print("CHRONOLOGICAL SPLIT STATISTICS")
        print("="*60)
        
        for split_name, split_data in [("TRAIN", self.train_events), ("TEST", self.test_events)]:
            if len(split_data) > 0:
                print(f"\n{split_name} SET:")
                print(f"- Events: {len(split_data)}")
                print(f"- Date range: {split_data['datetime'].min().date()} to {split_data['datetime'].max().date()}")
                print(f"- Magnitude range: {split_data['Magnitude'].min():.1f} to {split_data['Magnitude'].max():.1f}")
                print(f"- Mean magnitude: {split_data['Magnitude'].mean():.2f}")
                
                # Magnitude distribution
                mag_bins = [0, 4.0, 5.0, 6.0, 7.0, 10.0]
                mag_labels = ['<4.0', '4.0-4.9', '5.0-5.9', '6.0-6.9', '≥7.0']
                mag_dist = pd.cut(split_data['Magnitude'], bins=mag_bins, labels=mag_labels, include_lowest=True)
                print("- Magnitude distribution:")
                for label, count in mag_dist.value_counts().sort_index().items():
                    print(f"  {label}: {count} events ({count/len(split_data)*100:.1f}%)")
    
    def calculate_dobrovolsky_radius(self, magnitude: float) -> float:
        """
        Calculate Dobrovolsky radius for earthquake precursor detection.
        
        Formula: R = 10^(0.43*M) km
        
        Args:
            magnitude: Earthquake magnitude
            
        Returns:
            Dobrovolsky radius in kilometers
        """
        return 10 ** (0.43 * magnitude)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate great circle distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of first point (degrees)
            lat2, lon2: Latitude and longitude of second point (degrees)
            
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        R = 6371.0
        
        return R * c
    
    def map_stations_to_events(self) -> pd.DataFrame:
        """
        Map stations to earthquake events based on Dobrovolsky radius.
        
        Returns:
            DataFrame with station-event mappings
        """
        logger.info("Mapping stations to earthquake events...")
        
        # Combine train and test events
        all_events = pd.concat([self.train_events, self.test_events], ignore_index=True)
        
        mappings = []
        
        for _, event in all_events.iterrows():
            event_lat = event['Latitude']
            event_lon = event['Longitude']
            magnitude = event['Magnitude']
            
            # Calculate Dobrovolsky radius
            dobrovolsky_radius = self.calculate_dobrovolsky_radius(magnitude)
            
            # Check each station
            stations_in_range = []
            station_distances = []
            
            for _, station in self.station_data.iterrows():
                station_lat = station['Latitude']
                station_lon = station['Longitude']
                station_code = station['Kode Stasiun']
                
                # Calculate distance
                distance = self.calculate_distance(event_lat, event_lon, station_lat, station_lon)
                
                # Check if station is within precursor range
                if distance <= dobrovolsky_radius:
                    stations_in_range.append(station_code)
                    station_distances.append(distance)
                    
                    # Create mapping record
                    mapping = {
                        'event_id': event['event_id'],
                        'datetime': event['datetime'],
                        'magnitude': magnitude,
                        'event_lat': event_lat,
                        'event_lon': event_lon,
                        'dobrovolsky_radius_km': dobrovolsky_radius,
                        'station_code': station_code,
                        'station_lat': station_lat,
                        'station_lon': station_lon,
                        'distance_km': distance,
                        'within_precursor_range': True,
                        'split': event['split']
                    }
                    mappings.append(mapping)
        
        self.station_event_mapping = pd.DataFrame(mappings)
        
        logger.info(f"Generated {len(self.station_event_mapping)} station-event mappings")
        
        # Display mapping statistics
        self._display_mapping_statistics()
        
        return self.station_event_mapping
    
    def _display_mapping_statistics(self) -> None:
        """Display station-event mapping statistics."""
        print(f"\n" + "="*60)
        print("STATION-EVENT MAPPING STATISTICS")
        print("="*60)
        
        # Overall statistics
        total_events = len(self.station_event_mapping['event_id'].unique())
        total_mappings = len(self.station_event_mapping)
        avg_stations_per_event = total_mappings / total_events if total_events > 0 else 0
        
        print(f"\nOverall Statistics:")
        print(f"- Total events with station mappings: {total_events}")
        print(f"- Total station-event mappings: {total_mappings}")
        print(f"- Average stations per event: {avg_stations_per_event:.2f}")
        
        # Statistics by split
        for split in ['train', 'test']:
            split_data = self.station_event_mapping[self.station_event_mapping['split'] == split]
            if len(split_data) > 0:
                split_events = len(split_data['event_id'].unique())
                split_mappings = len(split_data)
                split_avg = split_mappings / split_events if split_events > 0 else 0
                
                print(f"\n{split.upper()} Set:")
                print(f"- Events: {split_events}")
                print(f"- Mappings: {split_mappings}")
                print(f"- Avg stations per event: {split_avg:.2f}")
        
        # Statistics by station
        print(f"\nMappings by Station:")
        station_counts = self.station_event_mapping['station_code'].value_counts()
        for station, count in station_counts.items():
            percentage = (count / total_mappings) * 100
            print(f"- {station}: {count} mappings ({percentage:.1f}%)")
        
        # Magnitude distribution of mapped events
        print(f"\nMagnitude Distribution of Mapped Events:")
        mag_bins = [0, 4.0, 5.0, 6.0, 7.0, 10.0]
        mag_labels = ['<4.0', '4.0-4.9', '5.0-5.9', '6.0-6.9', '≥7.0']
        unique_events = self.station_event_mapping.drop_duplicates('event_id')
        mag_dist = pd.cut(unique_events['magnitude'], bins=mag_bins, labels=mag_labels, include_lowest=True)
        for label, count in mag_dist.value_counts().sort_index().items():
            print(f"- {label}: {count} events ({count/len(unique_events)*100:.1f}%)")
    
    def find_scalogram_files(self, base_path: str = None) -> Dict[str, List[str]]:
        """
        Find scalogram files for each station.
        
        Args:
            base_path: Base path to search for scalogram files
            
        Returns:
            Dictionary mapping station codes to lists of file paths
        """
        if base_path is None:
            base_path = self.scalogram_base_path
        
        if base_path is None:
            logger.warning("No scalogram base path provided")
            return {}
        
        logger.info(f"Searching for scalogram files in: {base_path}")
        
        scalogram_files = {}
        
        # Search patterns for scalogram files
        patterns = [
            "**/*scalogram*.png",
            "**/*scalogram*.jpg", 
            "**/*scalogram*.npz",
            "**/*cwt*.png",
            "**/*cwt*.npz",
            "**/scalogram_*.png",
            "**/scalogram_*.npz"
        ]
        
        for station_code in self.station_data['Kode Stasiun']:
            station_files = []
            
            # Search in station-specific directories
            station_patterns = [
                f"**/{station_code}/**/*scalogram*",
                f"**/{station_code}/**/*cwt*",
                f"**/scalogram*{station_code}*",
                f"**/cwt*{station_code}*"
            ]
            
            all_patterns = patterns + station_patterns
            
            for pattern in all_patterns:
                search_path = os.path.join(base_path, pattern)
                files = glob.glob(search_path, recursive=True)
                station_files.extend(files)
            
            # Remove duplicates and sort
            station_files = sorted(list(set(station_files)))
            scalogram_files[station_code] = station_files
            
            logger.info(f"Found {len(station_files)} scalogram files for station {station_code}")
        
        return scalogram_files
    
    def generate_master_metadata(self, scalogram_files: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Generate master metadata CSV with synchronized dataset paths.
        
        Args:
            scalogram_files: Dictionary of station codes to file paths
            
        Returns:
            Master metadata DataFrame
        """
        logger.info("Generating master metadata...")
        
        if scalogram_files is None:
            scalogram_files = self.find_scalogram_files()
        
        metadata_records = []
        
        # Process each station-event mapping
        for _, mapping in self.station_event_mapping.iterrows():
            event_id = mapping['event_id']
            station_code = mapping['station_code']
            event_datetime = mapping['datetime']
            
            # Find corresponding scalogram files for this station
            station_files = scalogram_files.get(station_code, [])
            
            # Try to match files by date or event ID
            matched_files = self._match_scalogram_files(station_files, event_datetime, event_id)
            
            # Create metadata record
            record = {
                'event_id': event_id,
                'station_code': station_code,
                'datetime': event_datetime,
                'magnitude': mapping['magnitude'],
                'event_lat': mapping['event_lat'],
                'event_lon': mapping['event_lon'],
                'station_lat': mapping['station_lat'],
                'station_lon': mapping['station_lon'],
                'distance_km': mapping['distance_km'],
                'dobrovolsky_radius_km': mapping['dobrovolsky_radius_km'],
                'split': mapping['split'],
                'scalogram_files_count': len(matched_files),
                'scalogram_files': ';'.join(matched_files) if matched_files else '',
                'has_scalogram_data': len(matched_files) > 0,
                'data_synchronized': len(matched_files) > 0  # Simplified check
            }
            
            # Add Kp index information
            kp_info = self._get_kp_index_for_event(event_datetime)
            record.update(kp_info)
            
            metadata_records.append(record)
        
        self.master_metadata = pd.DataFrame(metadata_records)
        
        logger.info(f"Generated master metadata with {len(self.master_metadata)} records")
        
        # Display metadata statistics
        self._display_metadata_statistics()
        
        return self.master_metadata
    
    def _match_scalogram_files(self, files: List[str], event_datetime: datetime, event_id: int) -> List[str]:
        """
        Match scalogram files to specific event based on datetime or event ID.
        
        Args:
            files: List of file paths
            event_datetime: Event datetime
            event_id: Event ID
            
        Returns:
            List of matched file paths
        """
        matched_files = []
        
        # Extract date string for matching
        date_str = event_datetime.strftime('%Y%m%d')
        date_str_alt = event_datetime.strftime('%Y-%m-%d')
        
        for file_path in files:
            file_name = os.path.basename(file_path)
            
            # Check if file contains event ID
            if str(event_id) in file_name:
                matched_files.append(file_path)
                continue
            
            # Check if file contains date
            if date_str in file_name or date_str_alt in file_name:
                matched_files.append(file_path)
                continue
        
        return matched_files
    
    def _get_kp_index_for_event(self, event_datetime: datetime) -> Dict:
        """
        Get Kp index information for event datetime.
        
        Args:
            event_datetime: Event datetime
            
        Returns:
            Dictionary with Kp index information
        """
        # Convert both to timezone-naive for comparison
        kp_datetime_naive = pd.to_datetime(self.kp_data['Date_Time_UTC']).dt.tz_localize(None)
        
        # Ensure event_datetime is timezone-naive
        if hasattr(event_datetime, 'tz_localize'):
            event_datetime_naive = event_datetime.tz_localize(None)
        elif hasattr(event_datetime, 'replace') and event_datetime.tzinfo is not None:
            event_datetime_naive = event_datetime.replace(tzinfo=None)
        else:
            event_datetime_naive = event_datetime
        
        # Find closest Kp index record (within 3 hours)
        time_diff = abs(kp_datetime_naive - event_datetime_naive)
        closest_idx = time_diff.idxmin()
        
        if time_diff.iloc[closest_idx] <= timedelta(hours=3):
            kp_record = self.kp_data.iloc[closest_idx]
            return {
                'kp_index': kp_record['Kp_Index'],
                'kp_datetime': kp_record['Date_Time_UTC'],
                'kp_time_diff_hours': time_diff.iloc[closest_idx].total_seconds() / 3600
            }
        else:
            return {
                'kp_index': np.nan,
                'kp_datetime': pd.NaT,
                'kp_time_diff_hours': np.nan
            }
    
    def _display_metadata_statistics(self) -> None:
        """Display master metadata statistics."""
        print(f"\n" + "="*60)
        print("MASTER METADATA STATISTICS")
        print("="*60)
        
        total_records = len(self.master_metadata)
        with_scalogram = len(self.master_metadata[self.master_metadata['has_scalogram_data']])
        synchronized = len(self.master_metadata[self.master_metadata['data_synchronized']])
        
        print(f"\nOverall Statistics:")
        print(f"- Total metadata records: {total_records}")
        print(f"- Records with scalogram data: {with_scalogram} ({with_scalogram/total_records*100:.1f}%)")
        print(f"- Synchronized records: {synchronized} ({synchronized/total_records*100:.1f}%)")
        
        # Statistics by split
        for split in ['train', 'test']:
            split_data = self.master_metadata[self.master_metadata['split'] == split]
            if len(split_data) > 0:
                split_total = len(split_data)
                split_with_data = len(split_data[split_data['has_scalogram_data']])
                split_sync = len(split_data[split_data['data_synchronized']])
                
                print(f"\n{split.upper()} Set:")
                print(f"- Total records: {split_total}")
                print(f"- With scalogram data: {split_with_data} ({split_with_data/split_total*100:.1f}%)")
                print(f"- Synchronized: {split_sync} ({split_sync/split_total*100:.1f}%)")
        
        # Statistics by station
        print(f"\nData Availability by Station:")
        for station in self.station_data['Kode Stasiun']:
            station_data = self.master_metadata[self.master_metadata['station_code'] == station]
            if len(station_data) > 0:
                station_total = len(station_data)
                station_with_data = len(station_data[station_data['has_scalogram_data']])
                print(f"- {station}: {station_with_data}/{station_total} ({station_with_data/station_total*100:.1f}%)")
    
    def save_results(self, output_dir: str = "outputs/data_audit") -> Dict[str, str]:
        """
        Save all audit results to files.
        
        Args:
            output_dir: Output directory for results
            
        Returns:
            Dictionary of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Save master metadata
        metadata_path = os.path.join(output_dir, "master_metadata.csv")
        self.master_metadata.to_csv(metadata_path, index=False)
        saved_files['master_metadata'] = metadata_path
        logger.info(f"Saved master metadata to: {metadata_path}")
        
        # Save station-event mappings
        mapping_path = os.path.join(output_dir, "station_event_mappings.csv")
        self.station_event_mapping.to_csv(mapping_path, index=False)
        saved_files['station_mappings'] = mapping_path
        logger.info(f"Saved station mappings to: {mapping_path}")
        
        # Save train/test splits
        train_path = os.path.join(output_dir, "train_events.csv")
        self.train_events.to_csv(train_path, index=False)
        saved_files['train_events'] = train_path
        
        test_path = os.path.join(output_dir, "test_events.csv")
        self.test_events.to_csv(test_path, index=False)
        saved_files['test_events'] = test_path
        
        logger.info(f"Saved train/test splits to: {train_path}, {test_path}")
        
        # Save audit summary
        summary_path = os.path.join(output_dir, "audit_summary.json")
        summary = self._generate_audit_summary()
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        saved_files['audit_summary'] = summary_path
        logger.info(f"Saved audit summary to: {summary_path}")
        
        return saved_files
    
    def _generate_audit_summary(self) -> Dict:
        """Generate comprehensive audit summary."""
        return {
            'audit_timestamp': datetime.now().isoformat(),
            'data_sources': {
                'earthquake_catalog': self.earthquake_catalog_path,
                'kp_index': self.kp_index_path,
                'station_locations': self.station_locations_path,
                'scalogram_base_path': self.scalogram_base_path
            },
            'chronological_split': {
                'train_period': f"{self.train_start.date()} to {self.train_end.date()}",
                'test_period': f"{self.test_start.date()} to {self.test_end.date()}",
                'train_events': len(self.train_events),
                'test_events': len(self.test_events)
            },
            'station_mapping': {
                'total_stations': len(self.station_data),
                'total_mappings': len(self.station_event_mapping),
                'events_with_stations': len(self.station_event_mapping['event_id'].unique())
            },
            'data_availability': {
                'total_metadata_records': len(self.master_metadata),
                'records_with_scalogram_data': len(self.master_metadata[self.master_metadata['has_scalogram_data']]),
                'synchronized_records': len(self.master_metadata[self.master_metadata['data_synchronized']])
            }
        }
    
    def run_complete_audit(self, output_dir: str = "outputs/data_audit") -> Dict[str, str]:
        """
        Run complete data audit pipeline.
        
        Args:
            output_dir: Output directory for results
            
        Returns:
            Dictionary of saved file paths
        """
        logger.info("Starting complete data audit...")
        
        # Load all data
        self.load_data()
        
        # Perform chronological split
        self.chronological_split()
        
        # Map stations to events
        self.map_stations_to_events()
        
        # Generate master metadata
        self.generate_master_metadata()
        
        # Save results
        saved_files = self.save_results(output_dir)
        
        logger.info("Data audit completed successfully!")
        
        return saved_files


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Auditor for Earthquake Precursor Analysis')
    parser.add_argument('--earthquake-catalog', required=True, help='Path to earthquake catalog CSV')
    parser.add_argument('--kp-index', required=True, help='Path to Kp index CSV')
    parser.add_argument('--station-locations', required=True, help='Path to station locations CSV')
    parser.add_argument('--scalogram-path', help='Base path to scalogram datasets')
    parser.add_argument('--output-dir', default='outputs/data_audit', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create auditor and run
    auditor = DataAuditor(
        earthquake_catalog_path=args.earthquake_catalog,
        kp_index_path=args.kp_index,
        station_locations_path=args.station_locations,
        scalogram_base_path=args.scalogram_path
    )
    
    saved_files = auditor.run_complete_audit(args.output_dir)
    
    print(f"\n" + "="*60)
    print("DATA AUDIT COMPLETED")
    print("="*60)
    print("\nSaved files:")
    for key, path in saved_files.items():
        print(f"- {key}: {path}")


if __name__ == '__main__':
    main()