#!/usr/bin/env python
"""
Analysis Runner Script for Conceptual Fragmentation Project

This script provides a clean way to run analysis scripts without path hacks.
It automatically sets up the correct Python path and forwards arguments.

Usage:
    python run_analysis.py <script_name> [script_arguments]

Available Scripts:
    cluster_paths - Generate cluster paths data
    aggregate_results - Aggregate results from multiple runs
    similarity_metrics - Compute similarity metrics
    cross_layer_metrics - Compute cross-layer metrics
    cluster_stats - Compute cluster statistics
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# Get the project root directory (where this script is located)
PROJECT_ROOT = Path(__file__).resolve().parent

# Define mapping of script names to their relative paths
SCRIPTS = {
    "cluster_paths": "concept_fragmentation/analysis/cluster_paths.py",
    "aggregate_results": "concept_fragmentation/analysis/aggregate_results.py",
    "similarity_metrics": "concept_fragmentation/analysis/similarity_metrics.py",
    "cross_layer_metrics": "concept_fragmentation/analysis/cross_layer_metrics.py",
    "cluster_stats": "concept_fragmentation/analysis/cluster_stats.py",
}

def list_available_scripts():
    """Print a list of available scripts with their descriptions."""
    print("Available scripts:")
    for name, path in SCRIPTS.items():
        # Try to extract a description from the script's docstring
        script_path = PROJECT_ROOT / path
        if script_path.exists():
            try:
                with open(script_path, 'r') as f:
                    first_lines = ''.join([f.readline() for _ in range(10)])
                    if '"""' in first_lines:
                        description = first_lines.split('"""')[1].strip().split('\n')[0]
                        print(f"  {name} - {description}")
                    else:
                        print(f"  {name}")
            except Exception:
                print(f"  {name}")
        else:
            print(f"  {name} (script not found)")

def run_script(script_name, args):
    """Run the specified script with the given arguments."""
    if script_name not in SCRIPTS:
        print(f"Error: Unknown script '{script_name}'")
        list_available_scripts()
        return 1
    
    # Get the absolute path to the script
    script_path = PROJECT_ROOT / SCRIPTS[script_name]
    
    if not script_path.exists():
        print(f"Error: Script file '{script_path}' not found")
        return 1
    
    # Construct the command
    cmd = [sys.executable, str(script_path)] + args
    
    # Run the script with the correct Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    print(f"Running: {' '.join(cmd)}")
    print(f"With PYTHONPATH={env['PYTHONPATH']}")
    
    try:
        # Use run instead of call for better error handling in Python 3
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error executing script: {e}")
        return 1

def main():
    """Parse arguments and run the specified script."""
    # Create a parser for the runner script
    parser = argparse.ArgumentParser(
        description="Run analysis scripts with the correct Python path",
        usage="%(prog)s <script_name> [script_arguments]"
    )
    parser.add_argument("script", nargs="?", help="Script to run")
    
    # Parse just the first argument (script name)
    args, remaining_args = parser.parse_known_args()
    
    if not args.script:
        parser.print_help()
        list_available_scripts()
        return 0
    
    # Run the specified script
    return run_script(args.script, remaining_args)

if __name__ == "__main__":
    sys.exit(main())