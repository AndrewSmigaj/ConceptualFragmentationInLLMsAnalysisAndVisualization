#!/usr/bin/env python
"""
Run script to generate all visualizations for the Results section of the paper.

This script:
1. Ensures all requirements are installed
2. Runs the visualization generation script
3. Copies the output to the appropriate location for the paper

Usage:
    python run_visualizations.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR]
                                [--datasets DATASETS] [--seeds SEEDS] [--static_only]

Example:
    python run_visualizations.py --datasets titanic heart --seeds 0 1 2
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def check_requirements():
    """Check if required packages are installed and install if needed."""
    required_packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "plotly",
        "networkx",
        "scikit-learn"
    ]
    
    try:
        # Try importing packages
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import plotly
        import networkx
        import sklearn
        print("All required packages are installed.")
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Installing required packages...")
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade"] + required_packages, 
                      check=True)
        print("Requirements installed successfully.")


def copy_to_paper_directory(output_dir, paper_dir="docs/figures"):
    """Copy generated visualizations to the paper directory."""
    # Create paper figures directory if it doesn't exist
    os.makedirs(paper_dir, exist_ok=True)
    
    # Copy all PNG files recursively
    output_path = Path(output_dir)
    paper_path = Path(paper_dir)
    
    for png_file in output_path.glob("**/*.png"):
        # Create relative path structure in paper directory
        rel_path = png_file.relative_to(output_path)
        target_path = paper_path / rel_path
        
        # Create parent directories if needed
        os.makedirs(target_path.parent, exist_ok=True)
        
        # Copy file
        shutil.copy2(png_file, target_path)
    
    print(f"Copied all visualizations to {paper_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run visualization generation for Results section.")
    parser.add_argument("--data_dir", type=str, default="data/cluster_paths",
                       help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Directory to save output figures")
    parser.add_argument("--datasets", type=str, nargs="+", default=["titanic", "heart"],
                       help="List of datasets to generate visualizations for")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                       help="List of random seeds to use")
    parser.add_argument("--static_only", action="store_true",
                       help="Generate only static visualizations (no interactive)")
    parser.add_argument("--paper_dir", type=str, default="docs/figures",
                       help="Directory to copy figures for paper inclusion")
    
    args = parser.parse_args()
    
    # Check requirements
    check_requirements()
    
    # Run visualization script
    cmd = [
        sys.executable, "visualization/generate_all_visualizations.py",
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--datasets"
    ] + args.datasets + [
        "--seeds"
    ] + [str(seed) for seed in args.seeds]
    
    if args.static_only:
        cmd.append("--static_only")
    
    print(f"Running visualization generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Copy to paper directory
    copy_to_paper_directory(args.output_dir, args.paper_dir)
    
    print("\nVisualization generation complete!")
    print(f"Results saved to:")
    print(f"  - {args.output_dir} (all visualizations)")
    print(f"  - {args.paper_dir} (paper figures)")


if __name__ == "__main__":
    main()