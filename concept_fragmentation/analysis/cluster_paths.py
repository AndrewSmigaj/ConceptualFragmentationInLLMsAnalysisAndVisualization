"""
Module for computing and analyzing cluster paths across neural network layers.

This module provides functions to:
1. Compute cluster paths (how points move between clusters in different layers)
2. Add survival information for each sample path (for Titanic dataset)
3. Compute demographic archetype statistics for frequent paths
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import warnings
import pickle

# Import sklearn for our own clustering if needed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _natural_layer_sort(layer_name: str) -> Tuple[str, int]:
    """
    Sort layer names naturally, handling numeric parts correctly.
    E.g., 'layer10' should come after 'layer9', not 'layer1'.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Tuple for sorting: (prefix, number) or (layer_name, 0) if no number
    """
    if layer_name == "input":
        return ("", -1)  # Place input at the beginning
        
    # Extract prefix and numeric part
    match = re.match(r'([a-zA-Z_]+)(\d+)', layer_name)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    return (layer_name, 0)  # Default case


def load_clusters_from_cache(dataset_name: str, config_id: str, seed: int, max_k: int = 10) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Load precomputed clusters from the visualization cache.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'titanic')
        config_id: Configuration ID (e.g., 'baseline')
        seed: Random seed
        max_k: Maximum number of clusters used
        
    Returns:
        Dictionary of clusters by layer, or None if not found
    """
    # Path to cache file
    cache_path = f"visualization/cache/embedded_clusters/{dataset_name}_{config_id}_seed{seed}_maxk{max_k}.pkl"
    
    if os.path.exists(cache_path):
        try:
            print(f"Loading precomputed clusters from {cache_path}")
            with open(cache_path, 'rb') as f:
                layer_clusters = pickle.load(f)
            return layer_clusters
        except Exception as e:
            print(f"Error loading cached clusters: {e}")
    
    return None


def load_experiment_activations(results_dir: str) -> Dict[str, np.ndarray]:
    """
    Load activations from a previously run experiment.
    
    Args:
        results_dir: Path to the experiment results directory
        
    Returns:
        Dictionary mapping layer names to activation arrays
    """
    # Try to load from layer_activations.pkl
    activations_path = os.path.join(results_dir, "layer_activations.pkl")
    if os.path.exists(activations_path):
        print(f"Loading activations from {activations_path}")
        with open(activations_path, 'rb') as f:
            layer_activations = pickle.load(f)
        
        # Extract the last epoch's test activations for each layer
        activations = {}
        for layer_name in layer_activations:
            if layer_name != "epoch" and layer_name != "labels":
                if layer_activations[layer_name]["test"]:
                    # Get the last epoch's activations
                    activations[layer_name] = layer_activations[layer_name]["test"][-1].numpy()
        
        return activations
    
    # If that fails, try to load specific activation files
    activations = {}
    for layer_name in ["input", "layer1", "layer2", "layer3", "output"]:
        layer_file = os.path.join(results_dir, f"{layer_name}_activations.npy")
        if os.path.exists(layer_file):
            print(f"Loading {layer_name} activations from {layer_file}")
            activations[layer_name] = np.load(layer_file)
    
    if not activations:
        raise FileNotFoundError(f"No activation files found in {results_dir}")
    
    return activations


def compute_clusters_for_layer(
    activations: np.ndarray, 
    max_k: int = 10, 
    random_state: int = 42
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find optimal number of clusters for a layer's activations using silhouette score.
    This function follows the same pattern as visualization/_silhouette_search.
    
    Args:
        activations: Numpy array of activations (n_samples, n_features)
        max_k: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (optimal_k, cluster_centers, cluster_labels)
    """
    # Same logic as visualization/_silhouette_search
    best_k = 2
    best_score = -1.0
    best_labels = None
    best_centers = None
    
    # Try different k values
    for k in range(2, min(max_k, activations.shape[0]//2) + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(activations)
        
        # Skip if we have clusters with only one point (silhouette undefined)
        # or if only one cluster is formed
        if len(set(labels)) < 2 or np.min(np.bincount(labels)) < 2:
            continue
            
        try:
            score = silhouette_score(activations, labels)
            if score > best_score:
                best_k = k
                best_score = score
                best_labels = labels
                best_centers = kmeans.cluster_centers_
        except Exception as e:
            warnings.warn(f"Error computing silhouette for k={k}: {str(e)}")
            continue
    
    # Fall back to k=2 if no valid clustering found
    if best_centers is None:
        # Attempt to force k=2 if possible
        if activations.shape[0] >= 4:
            num_clusters_fallback = 2
        elif activations.shape[0] >=2:
            num_clusters_fallback = 1
        else:
            return 0, np.array([]), np.array([])

        if num_clusters_fallback == 1 and activations.shape[0] > 0:
            best_labels = np.zeros(activations.shape[0], dtype=int)
            best_centers = np.mean(activations, axis=0, keepdims=True)
            best_k = 1
        elif num_clusters_fallback == 2:
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=random_state)
            try:
                best_labels = kmeans.fit_predict(activations)
                # Check if k-means actually produced 2 clusters
                if len(np.unique(best_labels)) < 2:
                    # If k-means collapsed to 1 cluster, set labels to 0s and center to mean
                    best_labels = np.zeros(activations.shape[0], dtype=int)
                    best_centers = np.mean(activations, axis=0, keepdims=True)
                    best_k = 1
                else:
                    best_centers = kmeans.cluster_centers_
                    best_k = 2
            except Exception as e:
                warnings.warn(f"Error during fallback KMeans (k=2): {e}")
                return 0, np.array([]), np.array([])
        else:
            return 0, np.array([]), np.array([])

    return best_k, best_centers, best_labels


def compute_cluster_paths(layer_clusters: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute cluster paths based on layer-wise cluster assignments.
    
    Args:
        layer_clusters: Dictionary of cluster information per layer
                       {layer_name: {"k": k_opt, "centers": centers, "labels": labels}}
                       
    Returns:
        Tuple of (paths, layer_names)
        - paths: Array of shape (n_samples, n_layers) where paths[i,j] is the 
                cluster of sample i in layer j
        - layer_names: List of layer names in the order used for paths
    """
    # Sort layers naturally
    layer_names = sorted(layer_clusters.keys(), key=_natural_layer_sort)
    
    # Ensure all layers have the same number of samples
    first_layer = layer_names[0]
    n_samples = len(layer_clusters[first_layer]["labels"])
    
    # Collect labels for each layer
    labels_per_layer = []
    for layer in layer_names:
        layer_labels = layer_clusters[layer]["labels"]
        if len(layer_labels) != n_samples:
            raise ValueError(f"Layer {layer} has {len(layer_labels)} samples, expected {n_samples}")
        labels_per_layer.append(layer_labels)
    
    # Transpose so rows are samples, columns are layers
    paths = np.vstack(labels_per_layer).T  # Shape: (n_samples, n_layers)
    
    return paths, layer_names


def compute_path_archetypes(
    paths: np.ndarray, 
    layer_names: List[str], 
    df: pd.DataFrame, 
    target_column: Optional[str] = None,
    demographic_columns: Optional[List[str]] = None,
    top_k: int = 3,
    max_members: int = 50
) -> List[Dict[str, Any]]:
    """
    Compute archetypes (representative statistics) for the most common paths.
    
    Args:
        paths: Array of shape (n_samples, n_layers) containing cluster IDs
        layer_names: Names of the layers corresponding to path columns
        df: Original dataframe with demographic information
        target_column: Name of the target column (e.g., 'survived')
        demographic_columns: Columns to include in demographic statistics
        top_k: Number of most frequent paths to analyze
        max_members: Maximum number of member indices to include
        
    Returns:
        List of dictionaries, each representing a path archetype
    """
    # Convert paths to string representation for counting
    path_strings = []
    for path in paths:
        path_str = "→".join(str(cluster_id) for cluster_id in path)
        path_strings.append(path_str)
    
    # Count path frequencies
    path_counts = {}
    for i, path_str in enumerate(path_strings):
        if path_str not in path_counts:
            path_counts[path_str] = {"count": 0, "indices": []}
        path_counts[path_str]["count"] += 1
        path_counts[path_str]["indices"].append(i)
    
    # Sort paths by frequency
    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1]["count"], reverse=True)
    
    # Get demographic columns if not specified
    if demographic_columns is None:
        # Default demographics for Titanic
        demographic_columns = ["age", "sex", "pclass", "fare"]
        # Filter to include only columns that exist in df
        demographic_columns = [col for col in demographic_columns if col in df.columns]
    
    # Compute statistics for top-k paths
    archetypes = []
    for i, (path_str, path_info) in enumerate(sorted_paths[:top_k]):
        # Get indices of members with this path
        member_indices = path_info["indices"]
        
        # Create archetype with basic info
        archetype = {
            "path": path_str,
            "count": path_info["count"],
            "demo_stats": {},
            "member_indices": member_indices[:max_members]  # Limit number of indices
        }
        
        # Extract relevant subset of the dataframe
        members_df = df.iloc[member_indices].copy()
        
        # Compute target column statistics if provided
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # For numeric targets (like survival)
                archetype[f"{target_column}_rate"] = float(members_df[target_column].mean())
            else:
                # For categorical targets
                target_counts = members_df[target_column].value_counts(normalize=True).to_dict()
                archetype[f"{target_column}_distribution"] = target_counts
        
        # Compute demographic statistics
        for col in demographic_columns:
            if col not in df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns
                col_stats = {
                    "mean": float(members_df[col].mean()),
                    "std": float(members_df[col].std()),
                    "min": float(members_df[col].min()),
                    "max": float(members_df[col].max()),
                }
                archetype["demo_stats"][col] = col_stats
            else:
                # For categorical columns
                value_counts = members_df[col].value_counts(normalize=True).to_dict()
                # Convert keys to strings for JSON compatibility
                value_counts = {str(k): float(v) for k, v in value_counts.items()}
                archetype["demo_stats"][col] = value_counts
        
        archetypes.append(archetype)
    
    return archetypes


def write_cluster_paths(
    dataset_name: str,
    seed: int,
    layer_clusters: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    demographic_columns: Optional[List[str]] = None,
    output_dir: str = "data/cluster_paths",
    top_k: int = 3,
    max_members: int = 50
) -> str:
    """
    Compute and save cluster paths and associated information.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed used for experiment
        layer_clusters: Dictionary of cluster information per layer
        df: Original dataframe with demographic information
        target_column: Name of the target column (e.g., 'survived' for Titanic)
        demographic_columns: Columns to include in demographic statistics
        output_dir: Directory to save the output JSON
        top_k: Number of most frequent paths to analyze
        max_members: Maximum number of member indices to include
        
    Returns:
        Path to the written JSON file
    """
    # Compute cluster paths
    paths, layer_names = compute_cluster_paths(layer_clusters)
    
    # Create output data structure
    data = {
        "dataset": dataset_name,
        "seed": seed,
        "layers": layer_names,
        "paths": paths.tolist()  # Convert numpy array to list for JSON
    }
    
    # Add target column values if specified
    if target_column and target_column in df.columns:
        # Ensure df has the right number of rows
        if len(df) >= len(paths):
            # Only take the rows corresponding to the paths
            data[target_column] = df[target_column].values[:len(paths)].tolist()
    
    # Compute path archetypes
    archetypes = compute_path_archetypes(
        paths, 
        layer_names, 
        df, 
        target_column=target_column, 
        demographic_columns=demographic_columns,
        top_k=top_k,
        max_members=max_members
    )
    data["path_archetypes"] = archetypes
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to JSON
    output_path = os.path.join(output_dir, f"{dataset_name}_seed_{seed}_paths.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path


if __name__ == "__main__":
    import argparse
    import sys
    
    # Add project root to path if running as main script
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from concept_fragmentation.data.loaders import get_dataset_loader
    
    parser = argparse.ArgumentParser(description="Compute and save cluster paths.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., titanic)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum number of clusters")
    parser.add_argument("--output_dir", type=str, default="data/cluster_paths", help="Output directory")
    parser.add_argument("--top_k", type=int, default=3, help="Number of path archetypes")
    parser.add_argument("--max_members", type=int, default=50, help="Maximum member indices per archetype")
    parser.add_argument("--target_column", type=str, help="Target column (e.g., 'survived' for Titanic)")
    parser.add_argument("--demographic_columns", type=str, nargs="+", help="Demographic columns to include")
    parser.add_argument("--config_id", type=str, default="baseline", help="Configuration ID to use (e.g., 'baseline')")
    parser.add_argument("--use_cached_clusters", action="store_true", help="Use clusters from visualization cache")
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_loader = get_dataset_loader(args.dataset)
    train_df, test_df = dataset_loader.load_data()
    
    # Combine train and test for complete data
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Set default demographic columns based on dataset if not specified
    if args.demographic_columns is None:
        if args.dataset == "titanic":
            args.demographic_columns = ["age", "sex", "pclass", "fare"]
        elif args.dataset == "adult":
            args.demographic_columns = ["age", "education", "occupation", "race", "sex"]
        elif args.dataset == "heart":
            args.demographic_columns = ["age", "sex", "cp", "trestbps", "chol"]
    
    # Set default target column based on dataset if not specified
    if args.target_column is None:
        if args.dataset == "titanic":
            args.target_column = "survived"
        elif args.dataset == "adult":
            args.target_column = "income"
        elif args.dataset == "heart":
            args.target_column = "target"
    
    # First try to load precomputed clusters from cache if requested
    layer_clusters = None
    if args.use_cached_clusters:
        layer_clusters = load_clusters_from_cache(args.dataset, args.config_id, args.seed, args.max_k)
    
    # If no cached clusters found, compute them from activations
    if layer_clusters is None:
        print(f"No cached clusters found. Computing from activations...")
        # Load activations from experiment
        results_dir = os.path.join("results", "baselines", args.dataset, f"baseline_seed{args.seed}")
        print(f"Loading activations from {results_dir}")
        try:
            activations = load_experiment_activations(results_dir)
        except FileNotFoundError:
            print(f"No activation files found. Please run the baseline experiment for {args.dataset} seed {args.seed} first.")
            sys.exit(1)
        
        # Compute clusters for each layer
        layer_clusters = {}
        for layer_name, layer_activations in activations.items():
            print(f"Computing clusters for layer {layer_name}...")
            k, centers, labels = compute_clusters_for_layer(
                layer_activations, max_k=args.max_k, random_state=args.seed
            )
            print(f"  Found {k} clusters")
            layer_clusters[layer_name] = {
                "k": k,
                "centers": centers,
                "labels": labels
            }
    
    # Write cluster paths and archetypes
    output_path = write_cluster_paths(
        args.dataset,
        args.seed,
        layer_clusters,
        df,
        target_column=args.target_column,
        demographic_columns=args.demographic_columns,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_members=args.max_members
    )
    
    print(f"Cluster paths written to: {output_path}") 