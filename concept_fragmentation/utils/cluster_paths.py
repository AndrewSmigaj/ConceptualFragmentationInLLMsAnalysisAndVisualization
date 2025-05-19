"""
Cluster Path Utilities for Concept Fragmentation analysis.

This module provides functions to track individual sample trajectories through different
clusters across network layers, enabling a more interpretable analysis of concept
fragmentation by showing how samples move through representation space.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Union, Optional
import pandas as pd


def build_cluster_paths(
    layer_cluster_labels: Dict[str, List[int]], 
    ordered_layers: Optional[List[str]] = None
) -> List[str]:
    """
    Build path strings showing cluster membership across layers for each sample.
    
    Args:
        layer_cluster_labels: Dictionary mapping layer names to lists of cluster assignments
        ordered_layers: Optional list specifying order of layers (if None, sorted alphabetically)
        
    Returns:
        List of path strings, one per sample
    """
    if ordered_layers is None:
        ordered_layers = sorted(layer_cluster_labels.keys())
    
    # Ensure all layers have the same number of samples
    n_samples = len(layer_cluster_labels[ordered_layers[0]])
    for layer in ordered_layers:
        if len(layer_cluster_labels[layer]) != n_samples:
            raise ValueError(f"Layer {layer} has {len(layer_cluster_labels[layer])} samples, "
                            f"but expected {n_samples}")
    
    # Build path for each sample
    paths = []
    for i in range(n_samples):
        path = "→".join(str(layer_cluster_labels[layer][i]) for layer in ordered_layers)
        paths.append(path)
    
    return paths


def find_representative_sample(
    path_group: List[Dict], 
    final_layer_activations: Optional[np.ndarray] = None,
    method: str = "random"
) -> int:
    """
    Find a representative sample for a group of samples with the same path.
    
    Args:
        path_group: List of sample dictionaries with the same path
        final_layer_activations: Optional activations to find closest to centroid
        method: Method to select representative ('random', 'first', or 'centroid')
        
    Returns:
        Index of the representative sample
    """
    if len(path_group) == 0:
        return -1
    
    if method == "first":
        return path_group[0]["row_index"]
    
    if method == "random":
        import random
        return random.choice(path_group)["row_index"]
    
    if method == "centroid" and final_layer_activations is not None:
        # Get indices of samples in this group
        indices = [s["row_index"] for s in path_group]
        group_activations = final_layer_activations[indices]
        
        # Compute centroid
        centroid = np.mean(group_activations, axis=0)
        
        # Find sample closest to centroid
        distances = np.linalg.norm(group_activations - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        return path_group[closest_idx]["row_index"]
    
    # Default to random if method not supported or activations not provided
    import random
    return random.choice(path_group)["row_index"]


def summarize_paths(
    samples: List[Dict], 
    final_layer_activations: Optional[np.ndarray] = None,
    demographic_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create summary statistics for each unique path.
    
    Args:
        samples: List of sample dictionaries with path and demographics
        final_layer_activations: Optional activations to find representative samples
        demographic_keys: List of demographic keys to include in summary
        
    Returns:
        Dictionary mapping paths to summary statistics
    """
    # Group by path
    path_groups = {}
    for sample in samples:
        path = sample["path"]
        if path not in path_groups:
            path_groups[path] = []
        path_groups[path].append(sample)
    
    # Compute statistics for each path
    paths_summary = {}
    for path, group in path_groups.items():
        count = len(group)
        
        # Compute class distribution
        class_counts = {}
        for s in group:
            label = s["survived"]  # Titanic-specific, can be generalized
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Calculate positive class rate for binary classification
        positive_rate = (class_counts.get(1, 0) / count) if count > 0 else 0
        
        # Demographic statistics
        demo_stats = {}
        if demographic_keys:
            for key in demographic_keys:
                values = [s["demographics"].get(key) for s in group if key in s["demographics"]]
                if values:
                    if isinstance(values[0], (int, float)):
                        # Numerical features
                        demo_stats[f"mean_{key}"] = np.mean([v for v in values if v is not None])
                        demo_stats[f"median_{key}"] = np.median([v for v in values if v is not None])
                    else:
                        # Categorical features
                        counts = {}
                        for v in values:
                            if v not in counts:
                                counts[v] = 0
                            counts[v] += 1
                        demo_stats[f"{key}_distribution"] = {
                            str(k): v / len(values) for k, v in counts.items()
                        }
        
        # Find representative sample
        representative_idx = find_representative_sample(
            group, final_layer_activations, method="random" if final_layer_activations is None else "centroid"
        )
        
        paths_summary[path] = {
            "count": count,
            "class_distribution": {str(k): v / count for k, v in class_counts.items()},
            "positive_rate": positive_rate,
            "representative_idx": representative_idx,
            **demo_stats
        }
    
    return paths_summary


def prepare_cluster_path_data(
    layer_cluster_labels: Dict[str, List[int]],
    test_df_raw: pd.DataFrame,
    y_test: np.ndarray,
    ordered_layers: Optional[List[str]] = None,
    demographic_keys: Optional[List[str]] = None,
    final_layer_activations: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Prepare the complete cluster path data structure for analysis.
    
    Args:
        layer_cluster_labels: Dictionary mapping layer names to lists of cluster assignments
        test_df_raw: Raw test dataframe with original features
        y_test: Ground truth labels
        ordered_layers: Optional list specifying layer order
        demographic_keys: Keys for demographic information to include
        final_layer_activations: Optional activations for finding representative samples
    
    Returns:
        Complete cluster paths data structure ready for export
    """
    if ordered_layers is None:
        ordered_layers = sorted(layer_cluster_labels.keys())
    
    if demographic_keys is None and isinstance(test_df_raw, pd.DataFrame):
        # Titanic-specific default demographics
        demographic_keys = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
        # Filter to only include keys that exist in the dataframe
        demographic_keys = [k for k in demographic_keys if k in test_df_raw.columns]
    
    # Build path for each sample
    samples = []
    for i in range(len(y_test)):
        # Get demographics if dataframe is provided
        demographics = {}
        if isinstance(test_df_raw, pd.DataFrame) and i < len(test_df_raw):
            for key in demographic_keys:
                if key in test_df_raw.columns:
                    value = test_df_raw.iloc[i][key]
                    # Convert numpy types to Python native types for JSON
                    if isinstance(value, (np.integer, np.floating)):
                        value = value.item()
                    demographics[key] = value
        
        # Build sample info
        sample_info = {
            "row_index": int(i),
            "survived": int(y_test[i]),  # Titanic-specific, can be generalized
            "path": "→".join(str(layer_cluster_labels[layer][i]) for layer in ordered_layers),
            "demographics": demographics
        }
        samples.append(sample_info)
    
    # Create path summaries
    paths_summary = summarize_paths(samples, final_layer_activations, demographic_keys)
    
    # Compile full data structure
    cluster_paths_data = {
        "layers": ordered_layers,
        "layer_clusters": layer_cluster_labels,
        "samples": samples,
        "paths_summary": paths_summary
    }
    
    return cluster_paths_data


def save_cluster_paths(
    cluster_paths_data: Dict[str, Any], 
    output_path: str
) -> None:
    """
    Save the cluster paths data to a JSON file.
    
    Args:
        cluster_paths_data: Complete cluster paths data structure
        output_path: Path to save the JSON file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert any remaining non-serializable types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.number):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_data = convert_to_serializable(cluster_paths_data)
    
    # Save to file
    with open(output_path, "w") as f:
        json.dump(serializable_data, f, indent=4) 