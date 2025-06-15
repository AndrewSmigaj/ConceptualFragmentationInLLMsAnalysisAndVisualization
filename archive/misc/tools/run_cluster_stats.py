#!/usr/bin/env python
"""
CLI wrapper to generate cluster statistics files correctly.

This script loads cluster paths and demographic data, then calls the 
write_cluster_stats function to generate proper statistics files.
"""

import os
import sys
import json
import pandas as pd
import argparse
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the cluster stats function
from concept_fragmentation.analysis.cluster_stats import write_cluster_stats


def load_paths_json(dataset: str, seed: int) -> Dict[str, Any]:
    """Load the cluster paths JSON file."""
    json_path = f"data/cluster_paths/{dataset}_seed_{seed}_paths.json"
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cluster paths file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        paths_data = json.load(f)
        
    print(f"Loaded cluster paths from {json_path}")
    return paths_data


def load_demographic_data(dataset: str) -> pd.DataFrame:
    """Load demographic data from CSV file."""
    if dataset == "titanic":
        csv_path = "data/titanic/train.csv"
    elif dataset == "heart":
        csv_path = "data/heart/heart.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Demographic data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded demographic data from {csv_path}: {len(df)} rows")
    return df


def convert_paths_to_layer_clusters(paths_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert paths data to the layer_clusters format expected by write_cluster_stats.
    Extract cluster labels for each layer from the human_readable_paths.
    """
    if "layers" not in paths_data:
        raise ValueError("Paths data missing 'layers' key")
    
    if "human_readable_paths" not in paths_data:
        raise ValueError("Paths data missing 'human_readable_paths' key")
    
    # Human readable paths contain the full path with cluster IDs
    # Example: "L0C3→L1C0→L2C1→L3C0"
    # We want to extract the cluster IDs for each layer

    layers = paths_data["layers"]
    human_paths = paths_data["human_readable_paths"]
    
    # Initialize results
    layer_clusters = {layer: {"labels": [], "k": 0} for layer in layers}
    
    # Convert all the raw paths to layer-specific cluster assignments
    path_segments_by_datapoint = []
    
    for path in human_paths:
        # Extract the cluster IDs from the path
        # Example: "L0C3→L1C0→L2C1→L3C0" -> [3, 0, 1, 0]
        segments = []
        for segment in path.split("→"):
            # Extract the cluster ID (e.g., "L0C3" -> 3)
            cluster_id = int(segment.split("C")[1])
            segments.append(cluster_id)
        
        path_segments_by_datapoint.append(segments)
    
    # Convert to layer-specific cluster assignments
    for i, layer in enumerate(layers):
        # Extract the cluster assignment for this layer from each path
        labels = [segments[i] for segments in path_segments_by_datapoint]
        k = max(labels) + 1  # Number of clusters = max cluster ID + 1
        
        layer_clusters[layer] = {
            "labels": labels,
            "k": k,
        }
        
        print(f"Layer {layer}: {k} clusters, {len(labels)} data points")
    
    return layer_clusters


def main():
    parser = argparse.ArgumentParser(description="Generate cluster statistics files")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["titanic", "heart", "adult"],
                        help="Dataset name")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Load the paths data
    paths_data = load_paths_json(args.dataset, args.seed)
    
    # Load demographic data
    df = load_demographic_data(args.dataset)
    
    # Convert paths to layer_clusters format
    layer_clusters = convert_paths_to_layer_clusters(paths_data)
    
    # Match the length of the dataframe to the number of paths
    num_paths = len(paths_data.get("human_readable_paths", []))
    if len(df) > num_paths:
        print(f"Trimming dataframe from {len(df)} to {num_paths} rows to match paths")
        df = df.iloc[:num_paths].reset_index(drop=True)
    
    # Write cluster statistics
    output_path = write_cluster_stats(args.dataset, args.seed, layer_clusters, df)
    
    print(f"Wrote cluster statistics to {output_path}")
    # Verify the file exists
    if os.path.exists(output_path):
        print(f"✓ Verified: Cluster statistics file exists at {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"  File size: {file_size} bytes")
        
        # Check content
        try:
            with open(output_path, 'r') as f:
                stats_data = json.load(f)
            print(f"  ✓ File contains valid JSON")
            print(f"  ✓ Contains statistics for {len(stats_data.get('layers', {}))} layers")
        except Exception as e:
            print(f"  ✗ Error reading statistics file: {e}")
    else:
        print(f"✗ Error: Could not verify statistics file at {output_path}")


if __name__ == "__main__":
    main() 