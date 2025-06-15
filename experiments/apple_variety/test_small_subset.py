"""Test apple variety experiment with a small subset for quick validation."""

import yaml
from pathlib import Path
import shutil

def create_test_config():
    """Create a test configuration with reduced parameters."""
    # Load original config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify for quick testing
    config['dataset']['n_varieties'] = 3  # Only top 3 varieties
    config['training']['epochs'] = 10     # Reduced epochs
    config['training']['batch_size'] = 16 # Smaller batch size
    config['clustering']['k_max'] = 4     # Smaller k range
    config['visualization']['sankey']['top_n_paths'] = 5  # Fewer paths
    config['experiment']['output_dir'] = 'results/test_run'
    
    # Save test config
    with open('test_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return 'test_config.yaml'

def run_test():
    """Run the experiment with test configuration."""
    print("Creating test configuration...")
    test_config = create_test_config()
    
    print(f"Running experiment with {test_config}...")
    print("This will test:")
    print("- Data loading and preprocessing")
    print("- Model training (10 epochs)")
    print("- Activation collection")
    print("- Clustering with gap statistic")
    print("- Trajectory analysis")
    print("- Visualization generation")
    
    # Import and run experiment
    from run_experiment import AppleVarietyExperiment
    
    try:
        # Create experiment
        experiment = AppleVarietyExperiment(test_config)
        
        # Run experiment
        results = experiment.run()
        
        print("\nTest completed successfully!")
        print(f"Results saved to: {experiment.output_dir}")
        
        # List generated files
        output_files = list(Path(experiment.output_dir).glob('*'))
        print(f"\nGenerated {len(output_files)} files:")
        for f in sorted(output_files)[:10]:  # Show first 10
            print(f"  - {f.name}")
        
        # Check key metrics
        if 'test_accuracy' in results:
            print(f"\nTest Accuracy: {results['test_accuracy']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up test config
        if Path(test_config).exists():
            Path(test_config).unlink()

def clean_test_results():
    """Clean up test results directory."""
    test_dir = Path('results/test_run')
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test apple variety experiment")
    parser.add_argument('--clean', action='store_true', 
                       help='Clean up test results after running')
    args = parser.parse_args()
    
    # Run test
    success = run_test()
    
    # Clean up if requested
    if args.clean and success:
        clean_test_results()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)