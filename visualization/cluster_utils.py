"""
Utility functions for clustering in the embedded space.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def compute_layer_clusters_embedded(
    embeddings_dict: Dict[str, np.ndarray],
    max_k: int = 10,
    random_state: int = 42,
    cache_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute k-means clusters in the embedded 3D space.
    
    Args:
        embeddings_dict: Dictionary mapping layer names to 3D embedding arrays (shape: [n_samples, 3])
        max_k: Maximum number of clusters to try
        random_state: Random seed for reproducibility
        cache_path: Path to save/load cached results (if None, no caching)
        
    Returns:
        Dictionary mapping layer names to clustering info:
            {layer_name: {"k": optimal_k, "centers": cluster_centers, "labels": cluster_assignments}}
    """
    # Check cache if provided
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
                print(f"Loaded cached embedded clusters from {cache_path}")
                # Debug the structure
                for layer, info in result.items():
                    print(f"DEBUG layer: {layer} has keys: {list(info.keys())}")
                    print(f"DEBUG layer: {layer} has k={info.get('k')}, labels.shape={info.get('labels').shape if 'labels' in info else 'No labels'}")
                return result
        except Exception as e:
            print(f"Error loading cached clusters: {e}")
    
    # Initialize results dictionary
    layer_clusters = {}
    
    # Process each layer
    for layer_name, embeddings in embeddings_dict.items():
        print(f"DEBUG: Processing layer {layer_name} with embeddings shape {embeddings.shape}")
        if embeddings.shape[0] < 4:  # Need at least a few points
            continue
            
        # Find optimal k and cluster assignments
        best_k, best_centers, best_labels = _silhouette_search(
            embeddings, max_k, random_state)
        
        layer_clusters[layer_name] = {
            "k": best_k,
            "centers": best_centers,
            "labels": best_labels,
        }
    
    # Cache results if requested
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(layer_clusters, f)
            print(f"Cached embedded clusters to {cache_path}")
    
    return layer_clusters

def _silhouette_search(X: np.ndarray, max_k: int, random_state: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        X: Data matrix (n_samples, n_features)
        max_k: Maximum number of clusters to try
        random_state: Random seed
        
    Returns:
        (optimal_k, cluster_centers, cluster_labels)
    """
    best_k = 2
    best_score = -1.0
    best_labels = None
    best_centers = None
    
    # Try different k values
    for k in range(2, min(max_k, X.shape[0]//2) + 1):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X)
        
        # Skip if we have clusters with only one point (silhouette undefined)
        # or if only one cluster is formed
        if len(set(labels)) < 2 or np.min(np.bincount(labels)) < 2:
            continue
            
        try:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k = k
                best_score = score
                best_labels = labels
                best_centers = kmeans.cluster_centers_
        except Exception as e:
            print(f"Error computing silhouette for k={k}: {e}")
            continue
    
    # Fall back to k=2 if no valid clustering found (e.g., all silhouette scores failed or were < -1)
    if best_centers is None:
        # Attempt to force k=2 if possible
        if X.shape[0] >= 4: # Ensure enough samples for 2 clusters with at least 2 points each if possible
            num_clusters_fallback = 2
        elif X.shape[0] >=2: # If less than 4 but at least 2, try k=1 essentially (or all points in one cluster)
             num_clusters_fallback = 1 # KMeans might not like k=1, handle this
        else: # Not enough points to cluster
            return 0, np.array([]), np.array([])


        if num_clusters_fallback == 1 and X.shape[0] > 0:
            best_labels = np.zeros(X.shape[0], dtype=int)
            best_centers = np.mean(X, axis=0, keepdims=True)
            best_k = 1
        elif num_clusters_fallback == 2:
            kmeans = KMeans(n_clusters=2, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'warn' else 10, random_state=random_state)
            try:
                best_labels = kmeans.fit_predict(X)
                # Check if k-means actually produced 2 clusters
                if len(np.unique(best_labels)) < 2 :
                    # If k-means collapsed to 1 cluster, set labels to 0s and center to mean
                    best_labels = np.zeros(X.shape[0], dtype=int)
                    best_centers = np.mean(X, axis=0, keepdims=True)
                    best_k = 1
                else:
                    best_centers = kmeans.cluster_centers_
                    best_k = 2
            except Exception as e:
                 print(f"Error during fallback KMeans (k=2): {e}") # Still might fail for very few points
                 return 0, np.array([]), np.array([]) # No valid clusters
        else: # Not enough points
             return 0, np.array([]), np.array([])


    return best_k, best_centers, best_labels

def get_embedded_clusters_cache_path(dataset: str, config_id: str, seed: int, max_k: int, 
                                    cache_dir: str = "cache/embedded_clusters") -> str:
    """Generate cache path for embedded clusters."""
    # Ensure the base cache directory structure exists.
    # e.g., workspace_root/visualization/cache/embedded_clusters
    # os.makedirs(cache_dir, exist_ok=True) # This might be better placed in dash_app.py or main script
    
    # The cache_path in compute_layer_clusters_embedded handles os.makedirs(os.path.dirname(cache_path))
    # so the direct cache_dir itself doesn't strictly need to be created here,
    # but the parent of the specific file will be.
    
    # Using a relative path from the workspace root if cache_dir starts with "cache/"
    # otherwise, assume it's an absolute path or relative path from where Python is run.
    # For consistency with CACHE_DIR in dash_app.py, assume cache_dir is relative to visualization module.
    
    # Find the root of the workspace to correctly join paths if cache_dir is relative
    # This function might be called from different locations, so make path handling robust.
    # However, the plan specifies cache_dir = "cache/embedded_clusters", implying relative to `visualization`
    
    # The plan for dash_app.py passes `os.path.join(CACHE_DIR, "embedded_clusters")`
    # where CACHE_DIR is `os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")`
    # So the path passed to this function will be absolute or relative to `visualization` dir.
    
    # If cache_dir is relative like "cache/embedded_clusters", it needs to be joined
    # with the directory of *this file* (cluster_utils.py) if we want it inside `visualization`.
    # However, the caller `dash_app.py` resolves `CACHE_DIR` relative to `dash_app.py`.
    # So, it's simpler if `cache_dir` is passed as an absolute path or a path that `os.makedirs` can handle.
    # The `os.makedirs(os.path.dirname(cache_path), exist_ok=True)` in `compute_layer_clusters_embedded`
    # should handle creation if `cache_path` is well-formed.
    
    # For now, let's assume cache_dir is correctly passed as an absolute path or a resolvable relative one.
    # The main responsibility of this function is constructing the filename.
    
    # The `os.makedirs` in `compute_layer_clusters_embedded` takes care of creating the directory.
    
    filename = f"{dataset}_{config_id}_seed{seed}_maxk{max_k}.pkl"
    full_path = os.path.join(cache_dir, filename)
    
    # It's good practice for the function that *uses* the path for writing to ensure the dir exists.
    # So, `os.makedirs(os.path.dirname(full_path), exist_ok=True)` is correctly placed in `compute_layer_clusters_embedded`.
    # This function primarily constructs the name.
    
    return full_path 