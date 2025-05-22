#!/usr/bin/env python
"""
Test script for path metrics module.

This script tests the functionality of the path_metrics.py module
to ensure it can load data and calculate metrics correctly.
"""

import os
import sys
import json
from pprint import pprint

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main test function."""
    print("Testing path_metrics.py module...")
    
    try:
        from visualization.path_metrics import (
            load_path_data,
            calculate_all_path_metrics
        )
        
        # Test loading data
        dataset = "heart"
        seed = 0
        
        paths_data, paths_with_centroids_data, cluster_stats_data = load_path_data(dataset, seed)
        
        # Check if data was loaded
        print("\nData loading results:")
        print(f"  paths_data: {'Loaded successfully' if paths_data else 'Failed to load'}")
        print(f"  paths_with_centroids_data: {'Loaded successfully' if paths_with_centroids_data else 'Failed to load'}")
        print(f"  cluster_stats_data: {'Loaded successfully' if cluster_stats_data else 'Failed to load'}")
        
        if paths_data:
            print(f"\nLoaded paths data for {dataset} dataset (seed {seed}):")
            print(f"  Layers: {paths_data.get('layers', [])}")
            print(f"  Number of unique paths: {len(paths_data.get('unique_paths', []))}")
        
        # Calculate metrics
        print("\nCalculating all path metrics...")
        metrics = calculate_all_path_metrics(dataset, seed)
        
        if metrics:
            print("\nCalculated metrics:")
            for metric_name, metric_values in metrics.items():
                if metric_name == "layers":
                    print(f"  Layers: {metric_values}")
                else:
                    print(f"  {metric_name}: {type(metric_values).__name__} with {len(metric_values)} layer entries")
            
            # Show sample values for each metric
            print("\nSample metric values:")
            for metric_name, metric_values in metrics.items():
                if metric_name != "layers" and isinstance(metric_values, dict):
                    print(f"  {metric_name}:")
                    for layer, value in list(metric_values.items())[:3]:  # Show first 3 layers
                        print(f"    {layer}: {value:.4f}")
                    print("    ...")
        else:
            print("\nNo metrics were calculated.")
        
        # Test plotting functionality
        print("\nTesting plot_path_metrics function...")
        try:
            from visualization.path_metrics import plot_path_metrics
            
            figures = plot_path_metrics(metrics)
            print(f"  Created {len(figures)} figures")
            print("  Figure names:")
            for fig_name in figures.keys():
                print(f"    - {fig_name}")
            
            print("\nPath metrics module tested successfully!")
            return True
            
        except Exception as e:
            import traceback
            print(f"\nError testing plot_path_metrics: {e}")
            print(traceback.format_exc())
            return False
            
    except Exception as e:
        import traceback
        print(f"\nError testing path_metrics.py: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)