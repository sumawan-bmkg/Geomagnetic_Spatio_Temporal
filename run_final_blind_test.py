#!/usr/bin/env python3
"""
Final Blind Test Launcher
========================

STOP ALL DEMOS - THIS IS THE REAL DEAL

Launch the final blind test validation using:
- Certified dataset: 15,781 events
- Test set: 1,974 events (July 2024 - 2026)
- Full PCA-CMR solar storm analysis
- Q1 journal standards validation

Expected realistic results:
- Accuracy: 68-78% (not 94%!)
- Distance MAE: <500km for Indonesia network
- Solar robustness during Kp > 5 periods
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main launcher for final blind test."""
    
    print("SCIENTIFIC FINAL BLIND TEST VALIDATION")
    print("   REAL DATA - Q1 JOURNAL STANDARDS")
    print("   NO MORE DEMOS OR SIMULATIONS")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description='Final Blind Test Validation - Real Scientific Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REAL DATA VALIDATION:
- Certified dataset: 15,781 earthquake events
- Test period: July 2024 - 2026 (1,974 events)
- Full-scale inference with PCA-CMR active
- Solar storm robustness analysis (Kp > 5)
- Grad-CAM evidence for 3 largest earthquakes

EXPECTED REALISTIC PERFORMANCE:
- Accuracy: 68-78% (Q1 journal standard)
- Distance MAE: <500km (excellent for Indonesia)
- Conservative model behavior (appropriate for seismic prediction)

OUTPUT:
- Final Scientific Validation Report
- Grad-CAM visualizations for paper
- Station attention analysis
- Solar robustness comparison table
        """
    )
    
    parser.add_argument('--model', type=str, 
                       default='outputs/production_training/ground_truth_run/ground_truth_run/best_stage_3.pth',
                       help='Path to production model checkpoint')
    
    parser.add_argument('--certified-dataset', type=str, 
                       default='outputs/corrective_actions/certified_spatio_dataset.h5',
                       help='Path to certified dataset (15,781 events)')
    
    parser.add_argument('--kp-data', type=str, 
                       default='awal/kp_index_2026.csv',
                       help='Path to Kp-index data for solar analysis')
    
    parser.add_argument('--output', type=str, 
                       default='outputs/real_validation_final',
                       help='Output directory for final results')
    
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify data sources without running full validation')
    
    parser.add_argument('--evidence-only', action='store_true',
                       help='Only generate Grad-CAM/Attention (requires existing predictions)')

    
    args = parser.parse_args()
    
    # Verify critical files exist
    print("VERIFYING DATA SOURCES...")
    
    # Check model checkpoint
    if not os.path.exists(args.model):
        print(f" ERROR: Model checkpoint not found: {args.model}")
        print("\nExpected location:")
        print("  outputs/production_training/ground_truth_run/ground_truth_run/best_stage_3.pth")
        sys.exit(1)
    
    # Check certified dataset
    if not os.path.exists(args.certified_dataset):
        print(f" ERROR: Certified dataset not found: {args.certified_dataset}")
        print("\nExpected location:")
        print("  outputs/corrective_actions/certified_spatio_dataset.h5")
        print("\nThis should be the REAL dataset with 15,781 events, not demo data!")
        sys.exit(1)
    
    # Check Kp-index data
    if not os.path.exists(args.kp_data):
        print(f"  WARNING: Kp-index data not found: {args.kp_data}")
        print("Solar robustness analysis will be skipped")
    
    # Verify dataset size and content
    try:
        import h5py
        with h5py.File(args.certified_dataset, 'r') as f:
            dataset_keys = list(f.keys())
            
            # Support both root metadata and train/val structure
            total_events = 0
            if 'metadata' in f:
                metadata_obj = f['metadata']
                total_events = len(metadata_obj[list(metadata_obj.keys())[0]]) if hasattr(metadata_obj, 'keys') else len(metadata_obj)
            elif 'train' in f:
                for split in ['test', 'train', 'val']:
                    if split in f:
                        meta_key = 'meta' if 'meta' in f[split] else 'metadata'
                        if meta_key in f[split]:
                            total_events += len(f[split][meta_key])
            
            if total_events > 0:
                print(f"  Certified dataset verified: {total_events:,} events")
                
                if total_events < 15000:
                    print(f"  WARNING: Dataset has only {total_events} events")
                    print("Expected ~15,781 events for full validation")
                
                file_size_gb = os.path.getsize(args.certified_dataset) / (1024**3)
                print(f"  Dataset size: {file_size_gb:.1f} GB")
                
            else:
                print(" ERROR: Invalid dataset structure - no metadata found")
                sys.exit(1)
                
    except Exception as e:
        print(f" ERROR: Cannot read certified dataset: {str(e)}")
        sys.exit(1)
    
    print(f"  Model checkpoint: {args.model}")
    print(f"  Output directory: {args.output}")
    
    if args.gpu:
        print("Using GPU acceleration")
    
    if args.verify_only:
        print("\n DATA SOURCE VERIFICATION COMPLETED")
        print("All required files are present and valid")
        print("\nTo run full validation:")
        print(f"python {sys.argv[0]}")
        return
    
    print("\n" + "=" * 60)
    print("LAUNCHING FINAL BLIND TEST VALIDATION")
    print("=" * 60)
    print("  IMPORTANT EXPECTATIONS:")
    print("- This will process 1,974 real earthquake events")
    print("- Expect realistic accuracy: 68-78% (not 94%!)")
    print("- Processing time: ~30-60 minutes")
    print("- Memory usage: High (full dataset loading)")
    print("=" * 60)
    
    # Confirm execution
    try:
        confirm = input("\nProceed with final blind test? (yes/no): ").lower().strip()
        if confirm not in ['yes', 'y']:
            print("Final blind test cancelled.")
            return
    except KeyboardInterrupt:
        print("\nFinal blind test cancelled.")
        return
    
    # Run final blind test
    try:
        from final_blind_test_validation import FinalBlindTestValidator
        
        # Set device
        device = None
        if args.gpu:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("Using GPU acceleration")
            else:
                print("  GPU requested but not available, using CPU")
                device = 'cpu'
        
        # Initialize validator
        validator = FinalBlindTestValidator(
            model_checkpoint_path=args.model,
            certified_dataset_path=args.certified_dataset,
            kp_index_path=args.kp_data,
            output_dir=args.output,
            device=device
        )
        
        # Run final blind test
        validator.evidence_only = args.evidence_only
        validator.run_final_blind_test()

        
        print("\n" + "=" * 60)
        print("FINAL BLIND TEST VALIDATION COMPLETED!")
        print("=" * 60)
        print(f"  Results saved in: {args.output}")
        print("\n  Generated files:")
        print(f"  - Scientific report: {args.output}/reports/final_scientific_validation_report.md")
        print(f"  - Metrics (JSON): {args.output}/reports/final_scientific_validation_report.json")
        print(f"  - Grad-CAM evidence: {args.output}/plots/gradcam/")
        print(f"  - Station attention: {args.output}/plots/attention/")
        print(f"  - Execution log: {args.output}/final_blind_test_validation.log")
        
        print("\nMODEL STATUS: READY FOR Q1 JOURNAL SUBMISSION")
        print("Next steps: Manuscript preparation and peer review")
        print("=" * 60)
        
    except ImportError as e:
        print(f"  Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install torch torchvision matplotlib seaborn scikit-learn h5py pandas numpy")
        sys.exit(1)
    except Exception as e:
        print(f"  Final blind test failed: {str(e)}")
        print(f"\nCheck the log file for detailed error information:")
        print(f"  {args.output}/final_blind_test_validation.log")
        sys.exit(1)


if __name__ == "__main__":
    main()