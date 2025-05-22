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
import glob
import shutil
from datetime import datetime

# Import sklearn for our own clustering if needed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import our similarity metrics module
from concept_fragmentation.analysis.similarity_metrics import (
    compute_centroid_similarity,
    normalize_similarity_matrix,
    compute_layer_similarity_matrix,
    get_top_similar_clusters,
    serialize_similarity_matrix,
    deserialize_similarity_matrix
)


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


def load_experiment_activations(results_dir: str, use_full_dataset: bool = False) -> Dict[str, np.ndarray]:
    """
    Load activations from a previously run experiment.
    
    Args:
        results_dir: Path to the experiment results directory
        use_full_dataset: Force use of full dataset even if dimensions don't match
        
    Returns:
        Dictionary mapping layer names to activation arrays
    """
    # Try to load from layer_activations.pkl
    activations_path = os.path.join(results_dir, "layer_activations.pkl")
    if os.path.exists(activations_path):
        print(f"Loading activations from {activations_path}")
        with open(activations_path, 'rb') as f:
            layer_activations = pickle.load(f)
        
        # First scan all layers to determine the activation strategy
        # We'll use the same strategy for all layers to maintain sample count consistency
        use_combined = True  # Default: try to use combined data
        use_test_only = False
        train_test_dimension_mismatch = False
        sample_sizes = {}
        
        # Check feature dimensions for all layers
        for layer_name in layer_activations:
            if layer_name != "epoch" and layer_name != "labels":
                if layer_activations[layer_name]["train"] and layer_activations[layer_name]["test"]:
                    train_activations = layer_activations[layer_name]["train"][-1].numpy()
                    test_activations = layer_activations[layer_name]["test"][-1].numpy()
                    
                    # Store sample sizes
                    sample_sizes[layer_name] = {
                        "train": train_activations.shape[0],
                        "test": test_activations.shape[0],
                        "combined": train_activations.shape[0] + test_activations.shape[0]
                    }
                    
                    # Check if any layer has dimension mismatch
                    if train_activations.shape[1:] != test_activations.shape[1:]:
                        train_test_dimension_mismatch = True
                        
                        # If we can't fix with use_full_dataset, we must use test only
                        if not use_full_dataset:
                            use_combined = False
                            use_test_only = True
                            break
        
        # Determine the global strategy based on scanned information
        if train_test_dimension_mismatch:
            if use_full_dataset:
                print(f"  ⚠ Dimension mismatch detected in one or more layers")
                print(f"  Using --use_full_dataset flag to truncate to common feature dimensions")
            else:
                print(f"  ⚠ Dimension mismatch detected in one or more layers")
                print(f"  Using test data only for all layers to ensure consistent sample counts")
                print(f"  Use --use_full_dataset flag to use both train and test data with truncated features")
        
        # Extract activations consistently across all layers
        activations = {}
        for layer_name in layer_activations:
            if layer_name != "epoch" and layer_name != "labels":
                # Ensure we have both train and test activations
                if layer_activations[layer_name]["train"] and layer_activations[layer_name]["test"]:
                    # Get the last epoch's activations from both train and test
                    train_activations = layer_activations[layer_name]["train"][-1].numpy()
                    test_activations = layer_activations[layer_name]["test"][-1].numpy()
                    
                    if use_test_only:
                        # Use only test data for all layers for consistency
                        activations[layer_name] = test_activations
                        print(f"  Layer {layer_name}: Using {test_activations.shape[0]} test samples with shape {test_activations.shape}")
                    elif use_combined and train_activations.shape[1:] == test_activations.shape[1:]:
                        # No dimension mismatch, can concatenate directly
                        activations[layer_name] = np.concatenate([train_activations, test_activations], axis=0)
                        print(f"  Layer {layer_name}: Combined {train_activations.shape[0]} train + {test_activations.shape[0]} test = {activations[layer_name].shape[0]} total samples")
                    elif use_combined and use_full_dataset:
                        # Dimension mismatch but using full dataset, truncate features
                        min_features = min(train_activations.shape[1], test_activations.shape[1])
                        train_trunc = train_activations[:, :min_features]
                        test_trunc = test_activations[:, :min_features]
                        
                        # Concatenate the truncated arrays
                        activations[layer_name] = np.concatenate([train_trunc, test_trunc], axis=0)
                        print(f"  Layer {layer_name}: Truncated to common feature size {min_features}")
                        print(f"  Layer {layer_name}: Combined {train_trunc.shape[0]} train + {test_trunc.shape[0]} test = {activations[layer_name].shape[0]} total samples")
                    else:
                        # Fallback to test-only data for consistency with other layers
                        activations[layer_name] = test_activations
                        print(f"  Layer {layer_name}: Using {test_activations.shape[0]} test samples with shape {test_activations.shape}")
                elif layer_activations[layer_name]["test"]:
                    # Fallback to just test if train isn't available
                    activations[layer_name] = layer_activations[layer_name]["test"][-1].numpy()
                    print(f"  Layer {layer_name}: Using {activations[layer_name].shape[0]} test samples (train data not available)")
                elif layer_activations[layer_name]["train"]:
                    # Fallback to just train if test isn't available
                    activations[layer_name] = layer_activations[layer_name]["train"][-1].numpy()
                    print(f"  Layer {layer_name}: Using {activations[layer_name].shape[0]} train samples (test data not available)")
        
        # Final sanity check: verify all layers have the same number of samples
        sample_count = None
        for layer_name, activation in activations.items():
            if sample_count is None:
                sample_count = activation.shape[0]
            elif sample_count != activation.shape[0]:
                print(f"  ⚠ WARNING: Inconsistent sample counts across layers!")
                print(f"  Layer {layer_name} has {activation.shape[0]} samples, expected {sample_count}")
                print(f"  This will cause errors in subsequent processing steps")
        
        if sample_count is not None:
            print(f"  ✓ All layers have consistent sample count: {sample_count}")
        
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


def assign_unique_cluster_ids(layer_clusters: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, Tuple[str, int, int]], Dict[str, Dict[int, int]]]:
    """
    Assign unique IDs to all clusters across all layers.
    
    This function preserves the original cluster labels while adding unique IDs.
    
    Args:
        layer_clusters: Dictionary of cluster information per layer
        
    Returns:
        Tuple of (updated_layer_clusters, id_to_layer_cluster, cluster_to_unique_id)
        - updated_layer_clusters: Same dictionary with added unique IDs
        - id_to_layer_cluster: Mapping from unique ID to (layer_name, original_id, layer_idx)
        - cluster_to_unique_id: Mapping from (layer_name, original_id) to unique ID
    """
    # Sort layers to ensure consistent ordering
    layer_names = sorted(layer_clusters.keys(), key=_natural_layer_sort)
    
    # Create mapping from unique ID to (layer, original_cluster_id, layer_idx)
    id_to_layer_cluster = {}
    cluster_to_unique_id = {}
    next_id = 0
    
    # Assign unique IDs to each cluster in each layer
    for layer_idx, layer_name in enumerate(layer_names):
        original_labels = layer_clusters[layer_name]["labels"]
        unique_clusters = np.unique(original_labels)
        
        # Create mapping for this layer's clusters
        layer_mapping = {}
        for original_id in unique_clusters:
            unique_id = next_id
            layer_mapping[original_id] = unique_id
            id_to_layer_cluster[unique_id] = (layer_name, original_id, layer_idx)
            next_id += 1
        
        cluster_to_unique_id[layer_name] = layer_mapping
        
        # Create new labels array with unique IDs
        unique_labels = np.zeros_like(original_labels)
        for orig_id, unique_id in layer_mapping.items():
            unique_labels[original_labels == orig_id] = unique_id
        
        # Update layer clusters with unique IDs while preserving original
        layer_clusters[layer_name]["unique_labels"] = unique_labels
        layer_clusters[layer_name]["id_mapping"] = {
            unique_id: (original_id, layer_idx) for original_id, unique_id in layer_mapping.items()
        }
    
    return layer_clusters, id_to_layer_cluster, cluster_to_unique_id


def get_human_readable_path(path: np.ndarray, id_to_layer_cluster: Dict[int, Tuple[str, int, int]]) -> str:
    """
    Convert a path of unique IDs to a human-readable string.
    
    Args:
        path: List of unique cluster IDs
        id_to_layer_cluster: Mapping from unique IDs to layer information
        
    Returns:
        Human-readable path string (e.g., "L0C1→L1C2→L2C0")
    """
    parts = []
    for cluster_id in path:
        if cluster_id in id_to_layer_cluster:
            layer_name, original_id, layer_idx = id_to_layer_cluster[cluster_id]
            parts.append(f"L{layer_idx}C{original_id}")
        else:
            parts.append(f"Unknown({cluster_id})")
    
    return "→".join(parts)


def compute_fragmentation_score(
    path: np.ndarray,
    similarity_matrix: Dict[Tuple[int, int], float],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]]
) -> float:
    """
    Compute a fragmentation score for a path through the network.
    
    A high fragmentation score means that clusters in the path have low similarity
    across consecutive layers, indicating that concepts are being fragmented.
    
    Args:
        path: Array of cluster IDs representing a path through the network
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        id_to_layer_cluster: Mapping from unique ID to layer information
        
    Returns:
        Fragmentation score in [0, 1], where 0 is perfectly coherent (no fragmentation)
        and 1 is completely fragmented
    """
    if len(path) < 2:
        return 0.0  # Can't compute fragmentation for a single point
    
    # Convert similarity matrix to lookup for faster access
    similarity_lookup = {}
    for (id1, id2), sim in similarity_matrix.items():
        if id1 not in similarity_lookup:
            similarity_lookup[id1] = {}
        if id2 not in similarity_lookup:
            similarity_lookup[id2] = {}
        similarity_lookup[id1][id2] = sim
        similarity_lookup[id2][id1] = sim  # Ensure symmetry
    
    # Compute average similarity between consecutive pairs
    total_similarity = 0.0
    valid_pairs = 0
    
    for i in range(len(path) - 1):
        id1 = path[i]
        id2 = path[i + 1]
        
        # Skip invalid pairs (missing cluster IDs)
        if id1 not in id_to_layer_cluster or id2 not in id_to_layer_cluster:
            continue
        
        # Get similarity between consecutive clusters
        similarity = similarity_lookup.get(id1, {}).get(id2, 0.0)
        total_similarity += similarity
        valid_pairs += 1
    
    if valid_pairs == 0:
        # Fallback: proportion of layer transitions that change cluster
        changes = sum(1 for i in range(len(path) - 1) if path[i] != path[i + 1])
        if len(path) > 1:
            return changes / (len(path) - 1)
        return 0.0
    
    avg_similarity = total_similarity / valid_pairs
    
    # Convert similarity to fragmentation (1 - similarity)
    fragmentation = 1.0 - avg_similarity
    
    return fragmentation


def identify_similarity_convergent_paths(
    unique_paths: np.ndarray,
    similarity_matrix: Dict[Tuple[int, int], float],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]],
    min_similarity: float = 0.6,
    max_layer_distance: int = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Identify paths where clusters in later layers have high similarity to clusters in earlier layers.
    
    Args:
        unique_paths: Array where paths[i,j] is the unique cluster ID of sample i in layer j
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        id_to_layer_cluster: Mapping from unique ID to layer information
        min_similarity: Minimum similarity threshold to consider
        max_layer_distance: Maximum distance between layers to consider (None for any distance)
        
    Returns:
        Dictionary mapping path_idx to list of detected similarity convergences
    """
    if not similarity_matrix:
        return {}
    
    # Convert sparse similarity matrix to more efficient lookup
    similarity_lookup = {}
    for (id1, id2), sim in similarity_matrix.items():
        if id1 not in similarity_lookup:
            similarity_lookup[id1] = {}
        if id2 not in similarity_lookup:
            similarity_lookup[id2] = {}
        similarity_lookup[id1][id2] = sim
        similarity_lookup[id2][id1] = sim  # Ensure symmetry
    
    convergent_paths = {}
    
    # For each path
    for path_idx, path in enumerate(unique_paths):
        convergences = []
        
        # For each pair of clusters in the path
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                id1 = path[i]
                id2 = path[j]
                
                # Skip if we don't have information about these clusters
                if id1 not in id_to_layer_cluster or id2 not in id_to_layer_cluster:
                    continue
                
                # Get layer indices
                _, _, layer_idx1 = id_to_layer_cluster[id1]
                _, _, layer_idx2 = id_to_layer_cluster[id2]
                
                # Skip if layers are too far apart
                if max_layer_distance is not None and (layer_idx2 - layer_idx1) > max_layer_distance:
                    continue
                
                # Check similarity
                similarity = similarity_lookup.get(id1, {}).get(id2, 0.0)
                
                if similarity >= min_similarity:
                    convergences.append({
                        "early_layer": layer_idx1,
                        "late_layer": layer_idx2,
                        "early_cluster": id1,
                        "late_cluster": id2,
                        "similarity": similarity
                    })
        
        # Add to results if we found any convergences in this path
        if convergences:
            convergent_paths[path_idx] = convergences
    
    return convergent_paths


def compute_cluster_paths(layer_clusters: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[str], Dict[int, Tuple[str, int, int]], np.ndarray, List[str]]:
    """
    Compute cluster paths based on layer-wise cluster assignments.
    
    Args:
        layer_clusters: Dictionary of cluster information per layer
                       {layer_name: {"k": k_opt, "centers": centers, "labels": labels}}
                       
    Returns:
        Tuple of (unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths)
        - unique_paths: Array where paths[i,j] is the unique cluster ID of sample i in layer j
        - layer_names: List of layer names in the order used for paths
        - id_to_layer_cluster: Mapping from unique ID to (layer_name, original_id, layer_idx)
        - original_paths: Array where paths[i,j] is the original cluster ID (0-based within layer)
        - human_readable_paths: List of human-readable path strings for each sample
    """
    # Assign unique IDs if not already done
    if not all("unique_labels" in layer_info for layer_info in layer_clusters.values()):
        layer_clusters, id_to_layer_cluster, _ = assign_unique_cluster_ids(layer_clusters)
    else:
        # Extract mapping from existing data
        id_to_layer_cluster = {}
        for layer_idx, (layer_name, layer_info) in enumerate(sorted(layer_clusters.items(), key=lambda x: _natural_layer_sort(x[0]))):
            if "id_mapping" in layer_info:
                for unique_id, (original_id, _) in layer_info["id_mapping"].items():
                    id_to_layer_cluster[unique_id] = (layer_name, original_id, layer_idx)
    
    # Sort layers naturally
    layer_names = sorted(layer_clusters.keys(), key=_natural_layer_sort)
    
    # Collect both original and unique labels for each layer
    unique_labels_per_layer = []
    original_labels_per_layer = []
    for layer in layer_names:
        unique_labels_per_layer.append(layer_clusters[layer]["unique_labels"])
        original_labels_per_layer.append(layer_clusters[layer]["labels"])
    
    # Transpose so rows are samples, columns are layers
    unique_paths = np.vstack(unique_labels_per_layer).T  # Shape: (n_samples, n_layers)
    original_paths = np.vstack(original_labels_per_layer).T  # Shape: (n_samples, n_layers)
    
    # Create human-readable versions of paths
    human_readable_paths = [get_human_readable_path(path, id_to_layer_cluster) for path in unique_paths]
    
    return unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths


def compute_path_archetypes(
    paths: np.ndarray, 
    layer_names: List[str], 
    df: pd.DataFrame, 
    dataset_name: str,
    id_to_layer_cluster: Optional[Dict[int, Tuple[str, int, int]]] = None,
    human_readable_paths: Optional[List[str]] = None,
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
        id_to_layer_cluster: Mapping from unique IDs to layer information
        human_readable_paths: Pre-computed human-readable paths (if available)
        target_column: Name of the target column (e.g., 'survived')
        demographic_columns: Columns to include in demographic statistics
        top_k: Number of most frequent paths to analyze
        max_members: Maximum number of member indices to include
        
    Returns:
        List of dictionaries, each representing a path archetype
    """
    print(f"[DEBUG compute_path_archetypes] Dataset: {dataset_name}") # DEBUG
    print(f"[DEBUG compute_path_archetypes] Demographic columns: {demographic_columns}") # DEBUG
    print(f"[DEBUG compute_path_archetypes] Target column: {target_column}") # DEBUG
    print(f"[DEBUG compute_path_archetypes] Top K archetypes to compute: {top_k}") # DEBUG
    
    # Generate path strings for counting
    if human_readable_paths is not None:
        # Use pre-computed human-readable paths if provided
        path_strings = human_readable_paths
    elif id_to_layer_cluster is not None:
        # Generate human-readable paths from unique IDs
        path_strings = [get_human_readable_path(path, id_to_layer_cluster) for path in paths]
    else:
        # Fall back to simple numeric representation
        path_strings = ["→".join(str(cluster_id) for cluster_id in path) for path in paths]
    
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
        
        print(f"\n[DEBUG compute_path_archetypes] Processing archetype {i+1}/{top_k} for path: {path_str}") # DEBUG
        print(f"[DEBUG compute_path_archetypes] Member count for this path: {len(member_indices)}") # DEBUG
        
        # Get sample path to extract numeric cluster IDs (for future reference)
        sample_idx = member_indices[0]
        numeric_path = paths[sample_idx].tolist()
        
        # Create archetype with basic info
        archetype = {
            "path": path_str,  # Human-readable path
            "numeric_path": numeric_path,  # Underlying numeric IDs
            "count": path_info["count"],
            "percentage": 100 * path_info["count"] / len(paths),  # Add percentage of total
            "member_indices": member_indices[:max_members]  # Limit number of indices
        }
        
        # Extract relevant subset of the dataframe
        members_df = df.iloc[member_indices].copy()
        print(f"[DEBUG compute_path_archetypes] members_df shape: {members_df.shape}") # DEBUG
        if not members_df.empty and demographic_columns:
            # Print head for specified demographic columns if they exist in members_df
            cols_to_print = [col for col in demographic_columns if col in members_df.columns]
            if cols_to_print:
                 print(f"[DEBUG compute_path_archetypes] members_df head for relevant columns:\n{members_df[cols_to_print].head()}") # DEBUG
            else:
                print("[DEBUG compute_path_archetypes] No specified demographic columns found in members_df to print head.")
        elif demographic_columns:
            print("[DEBUG compute_path_archetypes] members_df is empty, cannot print head.")
        
        # Compute target column statistics if provided
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # For numeric targets (like survival)
                archetype[f"{target_column}_rate"] = float(members_df[target_column].mean())
            else:
                # For categorical targets
                target_counts = members_df[target_column].value_counts(normalize=True).to_dict()
                archetype[f"{target_column}_distribution"] = target_counts
        
        # Compute demographic statistics and store them as top-level keys
        print(f"[DEBUG compute_path_archetypes] Starting demographic calculations for path: {path_str}") # DEBUG
        for col in demographic_columns:
            print(f"[DEBUG compute_path_archetypes]   Processing column: {col}") # DEBUG
            if col not in df.columns:
                print(f"[DEBUG compute_path_archetypes]     Column {col} not in df. Skipping.") # DEBUG
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"[DEBUG compute_path_archetypes]     Column {col} is numeric.") # DEBUG
                mean_val = float(members_df[col].mean()) if not members_df[col].dropna().empty else None
                std_val = float(members_df[col].std()) if not members_df[col].dropna().empty else None
                min_val = float(members_df[col].min()) if not members_df[col].dropna().empty else None
                max_val = float(members_df[col].max()) if not members_df[col].dropna().empty else None
                archetype[f"{col}_mean"] = mean_val
                archetype[f"{col}_std"] = std_val
                archetype[f"{col}_min"] = min_val
                archetype[f"{col}_max"] = max_val
                print(f"[DEBUG compute_path_archetypes]       {col}_mean: {mean_val}, {col}_std: {std_val}, {col}_min: {min_val}, {col}_max: {max_val}") # DEBUG
            else:
                print(f"[DEBUG compute_path_archetypes]     Column {col} is categorical.") # DEBUG
                value_counts = members_df[col].value_counts(normalize=True).to_dict()
                dist_val = {str(k): float(v) for k, v in value_counts.items()}
                archetype[f"{col}_distribution"] = dist_val
                print(f"[DEBUG compute_path_archetypes]       {col}_distribution: {dist_val}") # DEBUG
                                
                if col == "sex": 
                    if dataset_name == "titanic": 
                        male_pct = float(100 * (members_df[col].str.lower() == "male").mean())
                        archetype["male_percentage"] = male_pct
                        print(f"[DEBUG compute_path_archetypes]       titanic male_percentage: {male_pct}") # DEBUG
                    elif dataset_name == "heart": 
                        if 1 in members_df[col].unique(): 
                            male_pct = float(100 * (members_df[col] == 1).mean())
                            archetype["male_percentage"] = male_pct
                            print(f"[DEBUG compute_path_archetypes]       heart male_percentage (1 is male): {male_pct}") # DEBUG
                        else: 
                            archetype["male_percentage"] = 0.0
                            print(f"[DEBUG compute_path_archetypes]       heart male_percentage (no 1s found, set to 0.0)") # DEBUG
        
        print(f"[DEBUG compute_path_archetypes] Final archetype for path {path_str}:\n{json.dumps(archetype, indent=2)}") # DEBUG
        archetypes.append(archetype)
    
    return archetypes


def extract_unique_centroids(layer_clusters: Dict[str, Dict[str, Any]], id_to_layer_cluster: Dict[int, Tuple[str, int, int]]) -> Dict[str, List[float]]:
    """
    Extract centroids for each unique cluster ID.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster information
        id_to_layer_cluster: Mapping from unique ID to (layer_name, original_id, layer_idx)
        
    Returns:
        Dictionary mapping unique cluster IDs to their centroids
    """
    unique_centroids = {}
    
    # Process each unique cluster ID
    for unique_id, (layer_name, original_id, _) in id_to_layer_cluster.items():
        # Get the layer's cluster information
        if layer_name in layer_clusters and "centers" in layer_clusters[layer_name]:
            # Find the centroid for this cluster
            cluster_centers = layer_clusters[layer_name]["centers"]
            
            # Ensure original_id is within range
            if original_id < len(cluster_centers):
                # Get the centroid as a Python list (for JSON serialization)
                centroid = cluster_centers[original_id].tolist() if hasattr(cluster_centers[original_id], "tolist") else list(cluster_centers[original_id])
                
                # Store with unique_id as a string key (for JSON compatibility)
                unique_centroids[str(unique_id)] = centroid
    
    return unique_centroids


def write_cluster_paths(
    dataset_name: str,
    seed: int,
    layer_clusters: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    demographic_columns: Optional[List[str]] = None,
    output_dir: str = "data/cluster_paths",
    top_k: int = 3,
    max_members: int = 50,
    compute_similarity: bool = True,
    similarity_metric: str = 'cosine',
    min_similarity: float = 0.3
) -> str:
    """
    Compute and save cluster paths and associated information using unique cluster IDs.
    
    Args:
        dataset_name: Name of the dataset
        seed: Random seed used for experiment
        layer_clusters: Dictionary of cluster information per layer
        df: Original dataframe with demographic information (already filtered and column names lowercased)
        target_column: Name of the target column (e.g., 'survived') (already lowercased)
        demographic_columns: Columns to include in demographic statistics (already lowercased and validated)
        output_dir: Directory to save the output JSON
        top_k: Number of most frequent paths to analyze
        max_members: Maximum number of member indices to include
        
    Returns:
        Path to the written JSON file
    """
    # Compute cluster paths with unique IDs
    unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths = compute_cluster_paths(layer_clusters)
    
    # Create serializable version of id_to_layer_cluster
    serializable_mapping = {}
    for unique_id, (layer_name, original_id, layer_idx) in id_to_layer_cluster.items():
        serializable_mapping[str(unique_id)] = {
            "layer_name": layer_name,
            "original_id": int(original_id),
            "layer_idx": int(layer_idx)
        }
    
    # Compute similarity matrix between clusters across layers if requested
    similarity_data = {}
    similarity_matrix = {}
    normalized_matrix = {}
    
    if compute_similarity:
        print(f"Computing centroid similarity matrix using {similarity_metric} similarity...")
        similarity_matrix = compute_centroid_similarity(
            layer_clusters,
            id_to_layer_cluster,
            metric=similarity_metric,
            min_similarity=0.0, # Capture all similarities
            same_layer=True 
        )
        
        normalized_matrix = normalize_similarity_matrix(similarity_matrix, metric=similarity_metric)
        
        layer_similarity = compute_layer_similarity_matrix(
            normalized_matrix,
            id_to_layer_cluster
        )
        
        top_similar_clusters = get_top_similar_clusters(
            normalized_matrix, 
            id_to_layer_cluster,
            top_k=min(5, top_k), # Use a reasonable number for top_k
            min_similarity=min_similarity
        )
        
        convergent_paths = identify_similarity_convergent_paths(
            unique_paths,
            normalized_matrix,
            id_to_layer_cluster,
            min_similarity=min_similarity,
            max_layer_distance=None
        )
        
        print("Computing fragmentation scores for all paths...")
        fragmentation_scores = np.zeros(len(unique_paths))
        for path_idx, path in enumerate(unique_paths):
            fragmentation_scores[path_idx] = compute_fragmentation_score(
                path, normalized_matrix, id_to_layer_cluster
            )
        
        low_frag_threshold = np.percentile(fragmentation_scores, 25)
        high_frag_threshold = np.percentile(fragmentation_scores, 75)
        
        high_fragmentation_paths = np.where(fragmentation_scores >= high_frag_threshold)[0].tolist()
        low_fragmentation_paths = np.where(fragmentation_scores <= low_frag_threshold)[0].tolist()
        
        print(f"Identified {len(high_fragmentation_paths)} paths with high fragmentation (score >= {high_frag_threshold:.3f})")
        print(f"Identified {len(low_fragmentation_paths)} paths with low fragmentation (score <= {low_frag_threshold:.3f})")
        
        human_readable_convergent_paths = {}
        for path_idx, convergences in convergent_paths.items():
            if path_idx < len(human_readable_paths):
                path_str = human_readable_paths[path_idx]
                readable_convergences = []
                for conv in convergences:
                    readable_conv = conv.copy()
                    if conv["early_cluster"] in id_to_layer_cluster:
                        _, early_original_id, early_layer_idx = id_to_layer_cluster[conv["early_cluster"]]
                        readable_conv["early_cluster_str"] = f"L{early_layer_idx}C{early_original_id}"
                    if conv["late_cluster"] in id_to_layer_cluster:
                        _, late_original_id, late_layer_idx = id_to_layer_cluster[conv["late_cluster"]]
                        readable_conv["late_cluster_str"] = f"L{late_layer_idx}C{late_original_id}"
                    readable_convergences.append(readable_conv)
                human_readable_convergent_paths[path_str] = readable_convergences
        
        similarity_data = {
            "raw_similarity": serialize_similarity_matrix(similarity_matrix),
            "normalized_similarity": serialize_similarity_matrix(normalized_matrix),
            "layer_similarity": {f"{l1},{l2}": stats for (l1, l2), stats in layer_similarity.items()},
            "top_similar_clusters": {str(cluster_id): [(int(similar_id), float(sim)) 
                                   for similar_id, sim in similarities] 
                               for cluster_id, similarities in top_similar_clusters.items()},
            "convergent_paths": {str(path_idx): convergences for path_idx, convergences in convergent_paths.items()},
            "human_readable_convergent_paths": human_readable_convergent_paths,
            "fragmentation_scores": {
                "scores": fragmentation_scores.tolist(),
                "high_fragmentation_paths": high_fragmentation_paths,
                "low_fragmentation_paths": low_fragmentation_paths,
                "high_threshold": float(high_frag_threshold),
                "low_threshold": float(low_frag_threshold),
                "mean": float(np.mean(fragmentation_scores)),
                "median": float(np.median(fragmentation_scores)),
                "std": float(np.std(fragmentation_scores))
            }
        }
        
        print(f"Found {len(similarity_matrix)} similarity relationships between clusters")
        print(f"Identified {len(convergent_paths)} paths with similarity convergence")
    
    print("Extracting centroids for unique cluster IDs...")
    unique_centroids = extract_unique_centroids(layer_clusters, id_to_layer_cluster)
    print(f"Extracted centroids for {len(unique_centroids)} unique clusters")
    
    metrics_block = {}
    try:
        import torch
        from concept_fragmentation.metrics.cluster_entropy import compute_cluster_entropy
        from concept_fragmentation.metrics.subspace_angle import compute_subspace_angle

        if target_column and target_column in df.columns: # target_column is now lowercased
            labels_vector = df[target_column].values

            entropy_per_layer = {}
            angle_per_layer = {}
            kstar_per_layer = {}

            for layer_name, layer_info in layer_clusters.items():
                if "activations" not in layer_info:
                    continue
                activations = layer_info["activations"]
                n_samples = activations.shape[0]
                if len(labels_vector) < n_samples:
                    activations = activations[: len(labels_vector)]
                    n_samples = activations.shape[0]
                layer_labels = labels_vector[:n_samples]

                act_tensor = torch.tensor(activations, dtype=torch.float32)
                lbl_tensor = torch.tensor(layer_labels, dtype=torch.long)

                ce_res = compute_cluster_entropy(act_tensor, lbl_tensor, k_selection="auto", normalize=True)
                entropy_per_layer[layer_name] = float(ce_res.get("mean_entropy", 0.0))
                kstar_per_layer[layer_name] = int(ce_res.get("chosen_k", layer_info.get("k", 1)))
                try:
                    sa_res = compute_subspace_angle(act_tensor, lbl_tensor, bootstrap_samples=3)
                    angle_per_layer[layer_name] = float(sa_res.get("mean_angle", 0.0))
                except Exception as e:
                    print(f"  ⚠ Sub-space angle failed for {layer_name}: {e}")
            if entropy_per_layer:
                entropy_vals = list(entropy_per_layer.values())
                metrics_block["entropy_fragmentation"] = {"per_layer": entropy_per_layer, "mean": float(np.mean(entropy_vals)), "std": float(np.std(entropy_vals))}
            if angle_per_layer:
                angle_vals = list(angle_per_layer.values())
                metrics_block["angle_fragmentation"] = {"per_layer": angle_per_layer, "mean": float(np.mean(angle_vals)), "std": float(np.std(angle_vals))}
            if kstar_per_layer:
                metrics_block["k_star"] = kstar_per_layer
    except Exception as e:
        print(f"⚠ Label-aware metrics could not be computed: {e}")
    
    data = {
        "dataset": dataset_name,
        "seed": seed,
        "layers": layer_names,
        "unique_paths": unique_paths.tolist(),
        "original_paths": original_paths.tolist(),
        "human_readable_paths": human_readable_paths,
        "id_mapping": serializable_mapping,
        "unique_centroids": unique_centroids,
        "similarity": similarity_data,
        "metrics": metrics_block
    }
    
    if target_column and target_column in df.columns: # target_column is now lowercased
        if len(df) >= len(unique_paths): # df here is the filtered one
            data[target_column] = df[target_column].values[:len(unique_paths)].tolist() # Use the already lowercased target_column
    
    archetypes = compute_path_archetypes(
        unique_paths, 
        layer_names, 
        df, # df is already filtered and column names lowercased
        dataset_name,
        id_to_layer_cluster=id_to_layer_cluster,
        human_readable_paths=human_readable_paths,
        target_column=target_column, # Already lowercased
        demographic_columns=demographic_columns, # Already lowercased and validated
        top_k=top_k,
        max_members=max_members
    )
    data["path_archetypes"] = archetypes
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_seed_{seed}_paths.json")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    with_centroids = output_path.replace("_paths.json", "_paths_with_centroids.json")
    shutil.copy(output_path, with_centroids)
    
    # Save copies in commonly accessed locations
    data_output_dir = os.path.join("data", "cluster_paths")
    os.makedirs(data_output_dir, exist_ok=True)
    data_output_path = os.path.join(data_output_dir, f"{dataset_name}_seed_{seed}_paths.json")
    with open(data_output_path, 'w') as f: json.dump(data, f, indent=2)
    shutil.copy(data_output_path, data_output_path.replace("_paths.json", "_paths_with_centroids.json"))
        
    vis_data_dir = os.path.join("visualization", "data", "cluster_paths") 
    os.makedirs(vis_data_dir, exist_ok=True)
    vis_data_path = os.path.join(vis_data_dir, f"{dataset_name}_seed_{seed}_paths.json")
    with open(vis_data_path, 'w') as f: json.dump(data, f, indent=2)
    shutil.copy(vis_data_path, vis_data_path.replace("_paths.json", "_paths_with_centroids.json"))
        
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    abs_data_path = os.path.join(project_root, "data", "cluster_paths", f"{dataset_name}_seed_{seed}_paths.json")
    os.makedirs(os.path.dirname(abs_data_path), exist_ok=True)
    with open(abs_data_path, 'w') as f: json.dump(data, f, indent=2)
    shutil.copy(abs_data_path, abs_data_path.replace("_paths.json", "_paths_with_centroids.json"))
    
    print(f"Saved cluster paths with unique IDs to:")
    for p in [output_path, data_output_path, vis_data_path, abs_data_path]: print(f"  - {p}")
    
    for path_to_check in [output_path, data_output_path, vis_data_path, abs_data_path]:
        if os.path.exists(path_to_check):
            print(f"✓ Verified: {path_to_check} exists")
            try:
                with open(path_to_check, 'r') as f: verify_data = json.load(f)
                if "unique_centroids" in verify_data:
                    print(f"  ✓ Verified: {path_to_check} contains {len(verify_data['unique_centroids'])} unique centroids")
                else: print(f"  ⚠ Warning: {path_to_check} does not contain unique_centroids field")
            except Exception as e: print(f"  ⚠ Warning: Could not verify content of {path_to_check}: {e}")
        else: print(f"⚠ Warning: {path_to_check} was not created!")
    
    return output_path


if __name__ == "__main__":
    import argparse
    import sys
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from concept_fragmentation.data.loaders import get_dataset_loader
    from concept_fragmentation.config import RESULTS_DIR # Ensure this is correctly imported/defined
    
    parser = argparse.ArgumentParser(description="Compute and save cluster paths using unique cluster IDs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., titanic)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--max_k", type=int, default=10, help="Maximum number of clusters")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: [RESULTS_DIR]/cluster_paths)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of path archetypes")
    parser.add_argument("--compute_similarity", action="store_true", help="Compute similarity matrix between clusters")
    parser.add_argument("--similarity_metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Similarity metric to use")
    parser.add_argument("--min_similarity", type=float, default=0.3, help="Minimum similarity threshold")
    parser.add_argument("--max_members", type=int, default=50, help="Maximum member indices per archetype")
    parser.add_argument("--target_column", type=str, help="Target column (e.g., 'survived' for Titanic)")
    parser.add_argument("--demographic_columns", type=str, nargs="+", help="Demographic columns to include")
    parser.add_argument("--config_id", type=str, default="baseline", help="Configuration ID to use (e.g., 'baseline')")
    parser.add_argument("--use_cached_clusters", action="store_true", help="Use clusters from visualization cache")
    parser.add_argument("--use_full_dataset", action="store_true", help="Force use of full dataset")
    
    args = parser.parse_args()
    
    dataset_loader = get_dataset_loader(args.dataset)
    train_df, test_df = dataset_loader.load_data()
    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    
    # Normalize df column names to lowercase for consistent access
    df.columns = [col.lower() for col in df.columns]
    
    # Determine and normalize target_column
    if args.target_column is None:
        if args.dataset == "titanic": args.target_column = "survived"
        elif args.dataset == "adult": args.target_column = "income"
        elif args.dataset == "heart": args.target_column = "target"
    if args.target_column:
        args.target_column = args.target_column.lower()

    # Determine, normalize, and validate demographic_columns
    if args.demographic_columns is None:
        if args.dataset == "titanic": args.demographic_columns = ["age", "sex", "pclass", "fare"]
        elif args.dataset == "adult": args.demographic_columns = ["age", "education", "occupation", "race", "sex"]
        elif args.dataset == "heart": args.demographic_columns = ["age", "sex", "cp", "trestbps", "chol"]
    
    actual_demographic_columns = []
    if args.demographic_columns:
        for col_name in args.demographic_columns:
            lower_col_name = col_name.lower()
            if lower_col_name in df.columns:
                actual_demographic_columns.append(lower_col_name)
            else:
                print(f"[Info] Demographic column '{col_name}' (as '{lower_col_name}') not found in DataFrame. Excluding.")
    args.demographic_columns = actual_demographic_columns # Update args with validated, lowercased columns

    # Prepare a minimal df with only necessary columns for compute_path_archetypes
    # This df is passed to write_cluster_paths which then passes it to compute_path_archetypes
    columns_to_keep_for_archetypes = []
    if args.target_column and args.target_column in df.columns:
        columns_to_keep_for_archetypes.append(args.target_column)
    if args.demographic_columns:
        columns_to_keep_for_archetypes.extend(args.demographic_columns)
    
    # Ensure unique columns, especially if target_column is also in demographic_columns
    columns_to_keep_for_archetypes = list(set(columns_to_keep_for_archetypes)) 
    
    # Create the filtered df to be passed to write_cluster_paths
    # If no relevant columns, pass the original df (now with lowercased columns)
    # compute_path_archetypes should handle cases where df might not have all expected columns.
    if columns_to_keep_for_archetypes:
        archetype_df = df[columns_to_keep_for_archetypes].copy()
    else:
        print("[Warning] No target or demographic columns found for archetypes. Passing full (lowercased) DataFrame.")
        archetype_df = df.copy() # Pass the full df (with lowercased columns)

    print(f"[Info] DataFrame for archetypes will contain columns: {list(archetype_df.columns)}")
    print(f"[Info] Effective demographic columns for archetypes: {args.demographic_columns}")
    print(f"[Info] Effective target column for archetypes: {args.target_column}")

    layer_clusters = None
    if args.use_cached_clusters:
        layer_clusters = load_clusters_from_cache(args.dataset, args.config_id, args.seed, args.max_k)
    
    if layer_clusters is None:
        print(f"No cached clusters found or not using cache. Computing from activations...")
        search_pattern = os.path.join(RESULTS_DIR, "baselines", args.dataset, f"{args.dataset}_baseline_seed{args.seed}_*")
        matching_dirs = glob.glob(search_pattern)
        if not matching_dirs:
            print(f"No experiment directories found matching: {search_pattern}")
            sys.exit(1)
        results_dir = sorted(matching_dirs)[-1]
        print(f"Loading activations from {results_dir}")
        try:
            activations = load_experiment_activations(results_dir, use_full_dataset=args.use_full_dataset)
        except FileNotFoundError:
            print(f"No activation files found in {results_dir}.")
            sys.exit(1)
        
        layer_clusters = {}
        for layer_name, layer_activations in activations.items():
            print(f"Computing clusters for layer {layer_name}...")
            k, centers, labels = compute_clusters_for_layer(layer_activations, max_k=args.max_k, random_state=args.seed)
            print(f"  Found {k} clusters")
            layer_clusters[layer_name] = {"k": k, "centers": centers, "labels": labels, "activations": layer_activations}
    
    print("Assigning unique cluster IDs across all layers...")
    layer_clusters, id_to_layer_cluster, cluster_to_unique_id = assign_unique_cluster_ids(layer_clusters)
    
    default_output_dir_base = RESULTS_DIR if RESULTS_DIR else "results" # Fallback for RESULTS_DIR
    if args.output_dir is None:
        args.output_dir = os.path.join(default_output_dir_base, "cluster_paths")
    
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output_dir from args is created
    
    output_path = write_cluster_paths(
        args.dataset,
        args.seed,
        layer_clusters,
        archetype_df, # Pass the filtered DataFrame
        target_column=args.target_column, # Already lowercased
        demographic_columns=args.demographic_columns, # Already lowercased and validated
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_members=args.max_members,
        compute_similarity=args.compute_similarity,
        similarity_metric=args.similarity_metric,
        min_similarity=args.min_similarity
    )
    
    total_clusters = sum(len(np.unique(layer_info["labels"])) for layer_info in layer_clusters.values())
    unique_ids_count = len(id_to_layer_cluster)
    
    sample_counts = []
    for layer_name, layer_info in layer_clusters.items():
        if "labels" in layer_info: sample_counts.append((layer_name, len(layer_info["labels"])))
    
    print(f"Summary:")
    print(f"- Dataset: {args.dataset}")
    print(f"- Layers: {', '.join(sorted(layer_clusters.keys(), key=_natural_layer_sort))}")
    print(f"- Total clusters across all layers: {total_clusters}")
    print(f"- Unique cluster IDs assigned: {unique_ids_count}")
    print(f"- Top {args.top_k} paths analyzed")
    
    print(f"\nSanity Check - Sample counts by layer:")
    for layer_name, count in sorted(sample_counts, key=lambda x: _natural_layer_sort(x[0])):
        print(f"  {layer_name}: {count} samples")
    
    if sample_counts:
        first_layer_count = sample_counts[0][1]
        all_same_count = all(count == first_layer_count for _, count in sample_counts)
        if all_same_count:
            # Use the original full df for this comparison, not archetype_df
            full_df_row_count = len(df) # df here is concat of train_df and test_df with lowercased columns
            print(f"✓ All layers have the same number of samples ({first_layer_count})")
            if abs(first_layer_count - full_df_row_count) < 5:
                print(f"✓ Using full dataset - Sample count ({first_layer_count}) matches combined train+test dataframe size ({full_df_row_count})")
            elif first_layer_count < 0.5 * full_df_row_count:
                print(f"⚠ WARNING: May be using partial dataset - Sample count ({first_layer_count}) is smaller than combined dataset size ({full_df_row_count})")
            else:
                print(f"ℹ️ Sample count: {first_layer_count}, Combined Dataset size: {full_df_row_count}")
        else:
            print(f"⚠ WARNING: Inconsistent sample counts across layers!")
            
    total_datapoints = sum(count for _, count in sample_counts)
    points_per_layer_val = sample_counts[0][1] if sample_counts else 0
    expected_total_val = points_per_layer_val * len(sample_counts)
    print(f"- Total datapoints across all clusters: {total_datapoints}")
    print(f"- Expected total ({points_per_layer_val} samples × {len(sample_counts)} layers): {expected_total_val}")
    
    print(f"\n- Cluster paths written to: {output_path}")