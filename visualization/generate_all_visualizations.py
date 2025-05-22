"""
Generate all visualizations for the Results section of the paper.

This script runs all the individual visualization scripts to generate
the complete set of figures referenced in the Results section.
"""

import os
import subprocess
import argparse
from typing import List, Dict, Any, Optional


def run_script(script_path: str, args: List[str]):
    """
    Run a Python script with the given arguments.
    
    Args:
        script_path: Path to the script to run
        args: List of command-line arguments
    """
    cmd = ["python", script_path] + args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def generate_visualizations(data_dir: str,
                           output_dir: str,
                           datasets: List[str] = ["titanic", "heart"],
                           seeds: List[int] = [0],
                           interactive: bool = True):
    """
    Generate all visualizations for the Results section.
    
    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save output figures
        datasets: List of datasets to generate visualizations for
        seeds: List of random seeds to use
        interactive: Whether to generate interactive visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset-specific directories
    for dataset in datasets:
        os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
    
    print("\n=== Generating Stepped-Layer Visualizations ===")
    for dataset in datasets:
        for seed in seeds:
            args = [
                "--data_dir", data_dir,
                "--output_dir", os.path.join(output_dir, dataset),
                "--dataset", dataset,
                "--seed", str(seed)
            ]
            if interactive:
                args.append("--interactive")
            
            run_script("visualization/generate_stepped_layer_viz.py", args)
    
    print("\n=== Generating Sankey Diagrams ===")
    for dataset in datasets:
        for seed in seeds:
            args = [
                "--data_dir", data_dir,
                "--output_dir", os.path.join(output_dir, dataset),
                "--dataset", dataset,
                "--seed", str(seed)
            ]
            
            run_script("visualization/generate_sankey_diagram.py", args)
    
    print("\n=== Generating ETS Visualizations ===")
    for dataset in datasets:
        for seed in seeds:
            for layer in ["layer1", "layer2", "layer3"]:
                args = [
                    "--data_dir", data_dir,
                    "--output_dir", os.path.join(output_dir, dataset),
                    "--dataset", dataset,
                    "--layer", layer
                ]
                if not interactive:
                    args.append("--static")
                
                run_script("visualization/generate_ets_visualization.py", args)
    
    print("\n=== All visualizations generated successfully! ===")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate all visualizations for the Results section.")
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
    
    args = parser.parse_args()
    
    generate_visualizations(args.data_dir, args.output_dir, args.datasets, args.seeds, 
                           interactive=not args.static_only)


if __name__ == "__main__":
    main()