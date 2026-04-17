#!/usr/bin/env python3
"""
Execute Evaluation Script - Simplified Runner

Script untuk menjalankan evaluasi lengkap dengan konfigurasi yang sudah disesuaikan
untuk data Indonesia dan kondisi spesifik yang diminta.
"""
import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all requirements are met."""
    logger.info("Checking requirements...")
    
    # Check Python packages
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'scikit-learn', 'h5py', 'opencv-python'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("✅ All required packages are available")
    return True


def setup_data_paths():
    """Setup and verify data paths."""
    logger.info("Setting up data paths...")
    
    # Default paths (adjust according to your data location)
    data_paths = {
        'scalogram_dir': 'scalogramv3',
        'metadata_path': 'outputs/data_audit/master_metadata.csv',
        'station_coords': 'awal/lokasi_stasiun.csv',
        'output_dir': 'outputs/complete_evaluation'
    }
    
    # Check if critical files exist
    critical_files = ['metadata_path', 'station_coords']
    
    for file_key in critical_files:
        if not os.path.exists(data_paths[file_key]):
            logger.warning(f"File not found: {data_paths[file_key]}")
            
            # Try alternative locations
            if file_key == 'metadata_path':
                alternatives = [
                    'Spatio_Precursor_Project/outputs/data_audit/master_metadata.csv',
                    'data/master_metadata.csv',
                    'master_metadata.csv'
                ]
                for alt in alternatives:
                    if os.path.exists(alt):
                        data_paths[file_key] = alt
                        logger.info(f"Found alternative: {alt}")
                        break
            
            elif file_key == 'station_coords':
                alternatives = [
                    'awal/lokasi_stasiun.csv',
                    'data/lokasi_stasiun.csv',
                    'lokasi_stasiun.csv'
                ]
                for alt in alternatives:
                    if os.path.exists(alt):
                        data_paths[file_key] = alt
                        logger.info(f"Found alternative: {alt}")
                        break
    
    return data_paths


def run_data_preparation():
    """Run data preparation steps."""
    logger.info("=== RUNNING DATA PREPARATION ===")
    
    # Check if data auditor has been run
    metadata_path = 'outputs/data_audit/master_metadata.csv'
    
    if not os.path.exists(metadata_path):
        logger.info("Running data auditor...")
        try:
            subprocess.run([
                sys.executable, 'run_data_audit.py',
                '--earthquake-catalog', 'awal/earthquake_catalog_2018_2025_merged.csv',
                '--kp-index', 'awal/kp_index_2018_2026.csv',
                '--station-locations', 'awal/lokasi_stasiun.csv',
                '--scalogram-base', 'scalogramv3',
                '--output-dir', 'outputs/data_audit'
            ], check=True, cwd='Spatio_Precursor_Project')
            logger.info("✅ Data auditor completed")
        except subprocess.CalledProcessError as e:
            logger.error(f"Data auditor failed: {e}")
            return False
        except FileNotFoundError:
            logger.error("run_data_audit.py not found. Please ensure you're in the correct directory.")
            return False
    else:
        logger.info("✅ Data auditor output already exists")
    
    return True


def run_evaluation_pipeline(data_paths: dict, quick_mode: bool = False):
    """Run the complete evaluation pipeline."""
    logger.info("=== RUNNING EVALUATION PIPELINE ===")
    
    # Prepare command
    cmd = [
        sys.executable, 'run_complete_evaluation.py',
        '--scalogram-dir', data_paths['scalogram_dir'],
        '--metadata-path', data_paths['metadata_path'],
        '--station-coords', data_paths['station_coords'],
        '--output-dir', data_paths['output_dir'],
        '--device', 'cuda' if os.system('nvidia-smi') == 0 else 'cpu'
    ]
    
    if quick_mode:
        logger.info("Running in quick mode (skip training if checkpoints exist)")
        cmd.append('--skip-training')
    
    try:
        # Change to project directory
        project_dir = Path('Spatio_Precursor_Project')
        if project_dir.exists():
            os.chdir(project_dir)
        
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        logger.info("✅ Evaluation pipeline completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("run_complete_evaluation.py not found. Please ensure you're in the correct directory.")
        return False


def create_synthetic_demo():
    """Create synthetic demo if real data is not available."""
    logger.info("=== CREATING SYNTHETIC DEMO ===")
    
    try:
        # Change to project directory
        project_dir = Path('Spatio_Precursor_Project')
        if project_dir.exists():
            os.chdir(project_dir)
        
        # Run the complete workflow example with synthetic data
        cmd = [sys.executable, 'examples/complete_workflow_example.py']
        
        logger.info("Running synthetic demo...")
        result = subprocess.run(cmd, check=True)
        
        logger.info("✅ Synthetic demo completed")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Synthetic demo failed: {e}")
        return False
    except FileNotFoundError:
        logger.error("Synthetic demo script not found")
        return False


def main():
    """Main execution function."""
    print("🌍 Spatio-Temporal Earthquake Precursor Evaluation")
    print("=" * 60)
    print("Evaluasi Model: CMR vs Original saat Badai Matahari (Kp > 5)")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not met. Please install missing packages.")
        return
    
    # Setup data paths
    data_paths = setup_data_paths()
    
    # Check if we have real data or need to use synthetic
    has_real_data = (
        os.path.exists(data_paths['metadata_path']) or 
        os.path.exists('awal/earthquake_catalog_2018_2025_merged.csv')
    )
    
    if has_real_data:
        print("📊 Real data detected. Running complete evaluation...")
        
        # Run data preparation
        if not run_data_preparation():
            print("❌ Data preparation failed")
            return
        
        # Run evaluation pipeline
        success = run_evaluation_pipeline(data_paths, quick_mode=False)
        
        if success:
            print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")
            print(f"📊 Results available in: {data_paths['output_dir']}")
            print("📈 Check the ablation study visualization")
            print("📋 Read the final evaluation report")
        else:
            print("\n❌ Evaluation failed. Check logs for details.")
    
    else:
        print("📝 Real data not found. Running synthetic demo...")
        
        success = create_synthetic_demo()
        
        if success:
            print("\n🎉 SYNTHETIC DEMO COMPLETED!")
            print("📊 Demo results available in: outputs/demo_training/")
            print("💡 Replace with real data for actual evaluation")
        else:
            print("\n❌ Demo failed. Check logs for details.")
    
    print("\n" + "=" * 60)
    print("Evaluation script finished.")


if __name__ == '__main__':
    main()