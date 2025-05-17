import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any


def compute_statistics(data, cluster_labels):
    numeric_stats = {}
    categorical_stats = {}
    for cluster in np.unique(cluster_labels):
        cluster_data = data[cluster_labels == cluster]
        numeric_stats[f'cluster_{cluster}'] = cluster_data.describe().to_dict()
        categorical_stats[f'cluster_{cluster}'] = {
            col: cluster_data[col].value_counts(normalize=True).to_dict()
            for col in data.select_dtypes(include=['object']).columns
        }
    return {'numeric_stats': numeric_stats, 'categorical_stats': categorical_stats}


def write_to_json(data, dataset_name, seed):
    path = f"data/cluster_stats/{dataset_name}_seed_{seed}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    return path


def write_cluster_stats(dataset_name: str, seed: int, layer_clusters: Dict[str, Dict[str, Any]], df: pd.DataFrame) -> str:
    """
    Compute and save descriptive statistics for clusters in the embedded space
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed used
        layer_clusters: Dictionary of layer clusters from the embedded space visualization
                        Structure: {layer_name: {"labels": labels_array, "k": num_clusters, ...}}
        df: DataFrame containing the original data points
        
    Returns:
        Path to the written JSON file
    """
    layer_stats = {}
    
    # Process each layer's clusters
    for layer_name, cluster_info in layer_clusters.items():
        # Get cluster labels from the embedded-space clustering
        cluster_labels = cluster_info["labels"]
        
        # Ensure we have the right number of rows
        if len(cluster_labels) > len(df):
            # Trim cluster labels if needed
            cluster_labels = cluster_labels[:len(df)]
        elif len(cluster_labels) < len(df):
            # Trim dataframe if needed
            df = df.iloc[:len(cluster_labels)]
            
        # Compute statistics for this layer's clusters
        stats = compute_statistics(df, cluster_labels)
        layer_stats[layer_name] = stats
    
    # Prepare final data structure
    data = {
        "seed": seed,
        "dataset": dataset_name,
        "layers": layer_stats
    }
    
    # Write to JSON and return path
    return write_to_json(data, dataset_name, seed) 