"""
Geomagnetic Data Reader Module
Refactored from awal/read_mdata.py with enhanced error handling and validation.
"""
import os
import gzip
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


class GeomagneticDataReader:
    """Reader for FRG604RC 1-second geomagnetic data files."""
    
    def __init__(self):
        """Initialize the data reader."""
        self.block_size = 30 + 17 * 600  # 10230 bytes per block
        self.records_per_block = 600
        self.record_size = 17  # bytes per record
    
    def _read_file(self, filepath: str) -> bytes:
        """
        Read file data, handling both compressed and uncompressed files.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Raw file data as bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Try compressed file first
        if filepath.endswith('.gz'):
            if os.path.exists(filepath):
                with gzip.open(filepath, 'rb') as f:
                    return f.read()
        else:
            # Try both compressed and uncompressed
            gz_path = filepath + '.gz'
            if os.path.exists(gz_path):
                with gzip.open(gz_path, 'rb') as f:
                    return f.read()
            elif os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    return f.read()
        
        raise FileNotFoundError(f"File not found: {filepath} or {filepath}.gz")
    
    def _parse_blocks(self, data: np.ndarray) -> tuple:
        """
        Parse data into blocks and extract voltage information.
        
        Args:
            data: Raw file data as uint8 array
            
        Returns:
            Tuple of (blocks, voltage_list)
        """
        n_blocks = data.size // self.block_size
        blocks = []
        volt_list = []
        
        for i in range(n_blocks):
            start = i * self.block_size
            block = data[start:start + self.block_size]
            
            if block.size < self.block_size:
                logger.warning(f"Incomplete block {i}, size: {block.size}")
                break
                
            blocks.append(block)
            
            # Extract voltage from byte position 28 (1-based) -> index 27 (0-based)
            if block.size > 27:
                volt_list.append(int(block[27]))
            else:
                volt_list.append(0)
                logger.warning(f"Block {i} too small for voltage extraction")
        
        logger.info(f"Parsed {len(blocks)} complete blocks")
        return blocks, volt_list
    
    def _extract_payload(self, blocks: list) -> np.ndarray:
        """
        Extract payload data by removing headers from blocks.
        
        Args:
            blocks: List of data blocks
            
        Returns:
            Concatenated payload data
        """
        # Remove first 30 bytes (header) from each block
        payload_parts = [block[30:] for block in blocks]
        payload = np.concatenate(payload_parts)
        
        # Ensure payload size is multiple of record size
        n_records = payload.size // self.record_size
        if payload.size != n_records * self.record_size:
            payload = payload[:n_records * self.record_size]
            logger.warning(f"Truncated payload to {n_records} complete records")
        
        return payload.reshape((n_records, self.record_size))
    
    def _read_uint24_le(self, arr: np.ndarray) -> np.ndarray:
        """
        Read 24-bit little-endian unsigned integers.
        
        Args:
            arr: Array of 3 bytes per value
            
        Returns:
            Array of uint32 values
        """
        return (arr[:, 0].astype(np.uint32) + 
                (arr[:, 1].astype(np.uint32) << 8) + 
                (arr[:, 2].astype(np.uint32) << 16))
    
    def _twos_complement(self, vals: np.ndarray, bits: int) -> np.ndarray:
        """
        Convert unsigned values to signed using two's complement.
        
        Args:
            vals: Unsigned values
            bits: Number of bits
            
        Returns:
            Signed values
        """
        vals_signed = vals.copy().astype(np.int64)
        over = vals >= (1 << (bits - 1))
        vals_signed[over] = vals_signed[over] - (1 << bits)
        return vals_signed
    
    def _parse_records(self, records: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parse individual records into geomagnetic components.
        
        Args:
            records: Array of record data
            
        Returns:
            Dictionary of parsed components
        """
        # Extract raw values (little-endian)
        H_raw = self._read_uint24_le(records[:, 0:3])
        D_raw = (records[:, 3].astype(np.uint32) + 
                (records[:, 4].astype(np.uint32) << 8) + 
                (records[:, 5].astype(np.uint32) << 16))
        Z_raw = (records[:, 6].astype(np.uint32) + 
                (records[:, 7].astype(np.uint32) << 8) + 
                (records[:, 8].astype(np.uint32) << 16))
        
        IX_raw = (records[:, 9].astype(np.uint32) + 
                 (records[:, 10].astype(np.uint32) << 8))
        IY_raw = (records[:, 11].astype(np.uint32) + 
                 (records[:, 12].astype(np.uint32) << 8))
        TempS_raw = (records[:, 13].astype(np.uint32) + 
                    (records[:, 14].astype(np.uint32) << 8))
        TempP_raw = (records[:, 15].astype(np.uint32) + 
                    (records[:, 16].astype(np.uint32) << 8))
        
        # Apply two's complement and scaling
        H_signed = self._twos_complement(H_raw, 24)
        D_signed = self._twos_complement(D_raw, 24)
        Z_signed = self._twos_complement(Z_raw, 24)
        
        IX_signed = self._twos_complement(IX_raw, 16)
        IY_signed = self._twos_complement(IY_raw, 16)
        TempS_signed = self._twos_complement(TempS_raw, 16)
        TempP_signed = self._twos_complement(TempP_raw, 16)
        
        # Scale to physical units
        H = H_signed.astype(np.float64) * 0.01  # nT
        D = D_signed.astype(np.float64) * 0.01  # degrees
        Z = Z_signed.astype(np.float64) * 0.01  # nT
        
        IX = IX_signed.astype(np.float64) * 0.1  # nT
        IY = IY_signed.astype(np.float64) * 0.1  # nT
        TempS = TempS_signed.astype(np.float64) * 0.01  # °C
        TempP = TempP_signed.astype(np.float64) * 0.01  # °C
        
        return {
            'H': H, 'D': D, 'Z': Z,
            'IX': IX, 'IY': IY,
            'TempS': TempS, 'TempP': TempP
        }
    
    def _apply_quality_control(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply quality control thresholds and set invalid values to NaN.
        
        Args:
            data: Dictionary of geomagnetic components
            
        Returns:
            Quality-controlled data
        """
        # Apply invalid thresholds
        data['H'][np.abs(data['H']) > 80000] = np.nan
        data['D'][np.abs(data['D']) > 80000] = np.nan
        data['Z'][np.abs(data['Z']) > 80000] = np.nan
        
        data['IX'][np.abs(data['IX']) > 3000] = np.nan
        data['IY'][np.abs(data['IY']) > 3000] = np.nan
        
        data['TempS'][np.abs(data['TempS']) > 300] = np.nan
        data['TempP'][np.abs(data['TempP']) > 300] = np.nan
        
        # Log quality statistics
        for key in ['H', 'D', 'Z']:
            valid_count = np.sum(np.isfinite(data[key]))
            total_count = len(data[key])
            valid_percent = (valid_count / total_count) * 100
            logger.info(f"{key}: {valid_count}/{total_count} valid ({valid_percent:.1f}%)")
        
        return data
    
    def _compute_cartesian_components(self, H: np.ndarray, D: np.ndarray) -> tuple:
        """
        Compute X and Y components from H and D.
        
        Args:
            H: H component (nT)
            D: D component (degrees)
            
        Returns:
            Tuple of (X, Y) components in nT
        """
        D_rad = np.deg2rad(D)
        X = np.full_like(H, np.nan, dtype=np.float64)
        Y = np.full_like(H, np.nan, dtype=np.float64)
        
        valid_idx = np.isfinite(H) & np.isfinite(D_rad)
        if np.any(valid_idx):
            X[valid_idx] = H[valid_idx] * np.cos(D_rad[valid_idx])
            Y[valid_idx] = H[valid_idx] * np.sin(D_rad[valid_idx])
        
        return X, Y
    
    def _process_voltage(self, volt_list: list, n_records: int) -> np.ndarray:
        """
        Process voltage data from block headers.
        
        Args:
            volt_list: List of voltage values from block headers
            n_records: Total number of records
            
        Returns:
            Voltage array for all records
        """
        V1 = np.array(volt_list, dtype=np.float64)
        
        # Repeat each block voltage value 600 times
        Voltage = np.repeat(V1, self.records_per_block)
        
        # Adjust size to match records
        if Voltage.size > n_records:
            Voltage = Voltage[:n_records]
        elif Voltage.size < n_records:
            Voltage = np.pad(Voltage, (0, n_records - Voltage.size), 'edge')
        
        # Scale and apply quality control
        Voltage = Voltage * 0.1
        Voltage[Voltage > 24] = np.nan
        
        return Voltage
    
    def read_daily_data(self, year: int, month: int, day: int, 
                       station: str, data_path: str) -> Dict[str, Union[np.ndarray, str]]:
        """
        Read daily geomagnetic data for specified date and station.
        
        Args:
            year: Year (4-digit)
            month: Month (1-12)
            day: Day (1-31)
            station: Station code (e.g., 'ALR')
            data_path: Path to data directory
            
        Returns:
            Dictionary containing all geomagnetic components and metadata
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data parsing fails
        """
        # Convert to 2-digit year for filename
        yy = year if year <= 2000 else year - 2000
        
        # Construct filename
        filename = os.path.join(data_path, station, 
                               f"S{yy:02d}{month:02d}{day:02d}.{station}")
        
        logger.info(f"Reading data from {filename}")
        
        try:
            # Read file data
            raw_data = self._read_file(filename)
            data_array = np.frombuffer(raw_data, dtype=np.uint8)
            
            # Parse blocks and extract payload
            blocks, volt_list = self._parse_blocks(data_array)
            if not blocks:
                raise ValueError("No complete blocks found in file")
            
            records = self._extract_payload(blocks)
            n_records = len(records)
            
            # Parse geomagnetic components
            components = self._parse_records(records)
            
            # Apply quality control
            components = self._apply_quality_control(components)
            
            # Compute Cartesian components
            X, Y = self._compute_cartesian_components(components['H'], components['D'])
            components['X'] = X
            components['Y'] = Y
            
            # Process voltage data
            components['Voltage'] = self._process_voltage(volt_list, n_records)
            
            # Create time vector
            start_time = datetime(year, month, day, 0, 0, 0)
            components['Time'] = np.array([
                start_time + timedelta(seconds=i) for i in range(n_records)
            ])
            
            # Add metadata
            components['filename'] = filename
            components['station'] = station
            components['date'] = f"{year:04d}-{month:02d}-{day:02d}"
            components['n_records'] = n_records
            
            logger.info(f"Successfully read {n_records} records from {station}")
            
            return components
            
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            raise
    
    def save_to_npz(self, data: Dict[str, Union[np.ndarray, str]], 
                   output_path: Optional[str] = None) -> str:
        """
        Save data to NPZ format.
        
        Args:
            data: Data dictionary from read_daily_data()
            output_path: Output file path (optional)
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            # Extract date info from data
            date_str = data['date'].replace('-', '')
            station = data['station']
            output_path = f"{date_str}.{station}.npz"
        
        # Prepare data for saving (exclude non-array metadata)
        save_data = {k: v for k, v in data.items() 
                    if isinstance(v, np.ndarray)}
        
        np.savez_compressed(output_path, **save_data)
        logger.info(f"Data saved to {output_path}")
        
        return output_path


if __name__ == '__main__':
    import argparse
    
    # Command line interface
    parser = argparse.ArgumentParser(description='Read geomagnetic data files')
    parser.add_argument('--year', type=int, required=True, help='Year')
    parser.add_argument('--month', type=int, required=True, help='Month')
    parser.add_argument('--day', type=int, required=True, help='Day')
    parser.add_argument('--station', type=str, required=True, help='Station code')
    parser.add_argument('--path', type=str, default='.', help='Data path')
    parser.add_argument('--output', type=str, help='Output NPZ file path')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Read data
    reader = GeomagneticDataReader()
    data = reader.read_daily_data(args.year, args.month, args.day, 
                                 args.station, args.path)
    
    # Print summary
    print(f"\nData Summary:")
    print(f"Station: {data['station']}")
    print(f"Date: {data['date']}")
    print(f"Records: {data['n_records']}")
    print(f"H range: {np.nanmin(data['H']):.3f} to {np.nanmax(data['H']):.3f} nT")
    print(f"Voltage range: {np.nanmin(data['Voltage']):.2f} to {np.nanmax(data['Voltage']):.2f} V")
    
    # Save to NPZ
    output_file = reader.save_to_npz(data, args.output)
    print(f"Saved to: {output_file}")