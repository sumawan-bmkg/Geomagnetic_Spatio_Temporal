#!/usr/bin/env python3
"""
Production Inference Launcher
=============================

Easy-to-use launcher for running production inference validation
on the Spatio-Temporal Earthquake Precursor Model.

Usage:
    python run_production_inference.py
    python run_production_inference.py --samples 100
    python run_production_inference.py --model path/to/model.pth
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Main launcher function."""
    
    print("🚀 SPATIO-TEMPORAL EARTHQUAKE PRECURSOR MODEL")
    print("   Production Inference & Validation System")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(
        description='Production Inference & Validation for Earthquake Precursor Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_production_inference.py                    # Run with defaults
  python run_production_inference.py --samples 100     # Process 100 samples
  python run_production_inference.py --gpu             # Use GPU if available
  
Output:
  Results will be saved in outputs/demo_inference_validation/
  - Comprehensive validation report (JSON + Markdown)
  - Performance visualizations (PNG)
  - Detailed execution logs
        """
    )
    
    parser.add_argument('--model', type=str, 
                       default='outputs/production_training/ground_truth_run/ground_truth_run/best_stage_3.pth',
                       help='Path to model checkpoint (default: best_stage_3.pth)')
    
    parser.add_argument('--dataset', type=str, 
                       default='real_earthquake_dataset.h5',
                       help='Path to test dataset (default: real_earthquake_dataset.h5)')
    
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples to process (default: 50)')
    
    parser.add_argument('--output', type=str, 
                       default='outputs/demo_inference_validation',
                       help='Output directory (default: outputs/demo_inference_validation)')
    
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU usage if available')
    
    parser.add_argument('--test-only', action='store_true',
                       help='Only test model loading without full inference')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ ERROR: Model checkpoint not found: {args.model}")
        print("\nAvailable model checkpoints:")
        
        # Search for available models
        search_patterns = [
            "outputs/production_training/*/ground_truth_run/best_stage_3.pth",
            "outputs/*/best_stage_3.pth",
            "best_stage_3.pth"
        ]
        
        import glob
        found_models = []
        for pattern in search_patterns:
            found_models.extend(glob.glob(pattern))
        
        if found_models:
            for model in found_models:
                print(f"  - {model}")
            print(f"\nTry: python {sys.argv[0]} --model {found_models[0]}")
        else:
            print("  No model checkpoints found!")
            print("  Please train the model first or check the path.")
        
        sys.exit(1)
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"❌ ERROR: Dataset not found: {args.dataset}")
        print("\nExpected dataset locations:")
        print("  - real_earthquake_dataset.h5")
        print("  - data/real_earthquake_dataset.h5")
        sys.exit(1)
    
    print(f"✓ Model checkpoint: {args.model}")
    print(f"✓ Dataset: {args.dataset}")
    print(f"✓ Samples to process: {args.samples}")
    print(f"✓ Output directory: {args.output}")
    
    if args.gpu:
        print("✓ GPU usage requested")
    
    print("\n" + "=" * 60)
    
    # Run test-only mode
    if args.test_only:
        print("🧪 RUNNING MODEL LOADING TEST ONLY")
        print("=" * 60)
        
        try:
            from test_model_loading import test_model_loading
            success = test_model_loading()
            
            if success:
                print("\n✅ MODEL LOADING TEST PASSED!")
                print("🚀 Ready for full inference validation!")
                print(f"\nTo run full validation:")
                print(f"python {sys.argv[0]} --samples {args.samples}")
            else:
                print("\n❌ MODEL LOADING TEST FAILED!")
                sys.exit(1)
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            sys.exit(1)
        
        return
    
    # Run full inference validation
    print("🔬 RUNNING PRODUCTION INFERENCE VALIDATION")
    print("=" * 60)
    
    try:
        # Import and run demo validation
        from demo_inference_validation import DemoInferenceValidator
        
        # Set device
        device = None
        if args.gpu:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                print("✓ Using GPU acceleration")
            else:
                print("⚠️  GPU requested but not available, using CPU")
                device = 'cpu'
        
        # Initialize validator
        validator = DemoInferenceValidator(
            model_checkpoint_path=args.model,
            test_dataset_path=args.dataset,
            output_dir=args.output,
            device=device
        )
        
        # Override sample count
        validator.load_demo_samples = lambda: validator.load_demo_samples(n_samples=args.samples)
        
        # Run validation
        validator.run_demo_validation()
        
        print("\n" + "=" * 60)
        print("✅ PRODUCTION INFERENCE VALIDATION COMPLETED!")
        print("=" * 60)
        print(f"📊 Results saved in: {args.output}")
        print("\n📋 Generated files:")
        print(f"  - Validation report: {args.output}/reports/demo_validation_report.md")
        print(f"  - Metrics (JSON): {args.output}/reports/demo_validation_report.json")
        print(f"  - Visualizations: {args.output}/plots/demo_validation_results.png")
        print(f"  - Execution log: {args.output}/demo_inference_validation.log")
        
        print("\n🎯 Key Results:")
        
        # Try to read and display key metrics
        try:
            import json
            with open(f"{args.output}/reports/demo_validation_report.json", 'r') as f:
                report = json.load(f)
            
            if 'performance_metrics' in report and 'binary_classification' in report['performance_metrics']:
                bc = report['performance_metrics']['binary_classification']
                print(f"  - Samples processed: {bc['total_samples']}")
                print(f"  - Accuracy: {bc['accuracy']:.3f}")
                print(f"  - F1-Score: {bc['f1_score']:.3f}")
                print(f"  - ROC AUC: {bc['roc_auc']:.3f}")
        except:
            pass
        
        print("\n🚀 System is ready for operational deployment!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install torch torchvision matplotlib seaborn scikit-learn h5py pandas numpy")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        print("\nCheck the log file for detailed error information:")
        print(f"  {args.output}/demo_inference_validation.log")
        sys.exit(1)


if __name__ == "__main__":
    main()