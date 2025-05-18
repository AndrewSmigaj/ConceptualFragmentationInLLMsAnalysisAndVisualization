"""
Explainable Threshold Similarity (ETS) Metric for Concept Fragmentation analysis.

This module implements the ETS clustering approach described in the paper
"Foundations of Archetypal Path Analysis: Toward a Principled Geometry for Cluster-Based Interpretability".

ETS declares two activations similar if |aᵢˡⱼ − aₖˡⱼ| ≤ τⱼ for every dimension j.
This creates transparent, dimension-wise explainable clusters where membership
can be verbalized as "neuron j differs by less than τⱼ."
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import time
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def compute_dimension_thresholds(
    activations: np.ndarray,
    threshold_percentile: float = 0.1,
    min_threshold: float = 1e-5
) -> np.ndarray:
    """
    Compute thresholds for each dimension based on pairwise differences.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        threshold_percentile: Percentile of pairwise differences to use as threshold (0.0-1.0)
        min_threshold: Minimum threshold value to avoid numerical issues
        
    Returns:
        Array of thresholds, one per dimension
    """
    n_samples, n_features = activations.shape
    thresholds = np.zeros(n_features)
    
    # Handle small datasets
    if n_samples <= 1:
        return np.full(n_features, min_threshold)
    
    # Compute thresholds for each dimension
    for j in range(n_features):
        # Extract this dimension's values
        values = activations[:, j]
        
        # Compute pairwise absolute differences
        # For memory efficiency, don't create the full matrix for large datasets
        if n_samples <= 1000:
            # Reshape to allow broadcasting
            col = values.reshape(-1, 1)
            diffs = np.abs(col - col.T)
            # Extract upper triangle to avoid counting differences twice
            triu_indices = np.triu_indices(n_samples, k=1)
            diff_values = diffs[triu_indices]
        else:
            # For large datasets, sample pairs to estimate percentile
            n_pairs = min(1000000, n_samples * (n_samples - 1) // 2)
            diff_values = []
            
            # Generate random pairs
            np.random.seed(42)  # For reproducibility
            for _ in range(n_pairs):
                i, j = np.random.randint(0, n_samples, 2)
                while i == j:  # Ensure different indices
                    j = np.random.randint(0, n_samples)
                diff_values.append(abs(values[i] - values[j]))
            
            diff_values = np.array(diff_values)
        
        # Filter out zeros before computing percentile
        non_zero_diffs = diff_values[diff_values > 0]
        
        if len(non_zero_diffs) > 0:
            # Use percentile as threshold
            threshold = np.percentile(non_zero_diffs, threshold_percentile * 100)
            # Ensure minimum threshold
            thresholds[j] = max(threshold, min_threshold)
        else:
            # If all differences are zero, use minimum threshold
            thresholds[j] = min_threshold
    
    return thresholds

def compute_similarity_matrix(
    activations: np.ndarray,
    thresholds: np.ndarray,
    batch_size: int = 1000,
    verbose: bool = False
) -> np.ndarray:
    """
    Compute similarity matrix where two points are similar if all dimension differences
    are below their respective thresholds.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        thresholds: Per-dimension thresholds
        batch_size: Size of batches for processing large datasets
        verbose: Whether to print progress updates
        
    Returns:
        Boolean similarity matrix where True indicates points are similar
    """
    n_samples, n_features = activations.shape
    similarity_matrix = np.zeros((n_samples, n_samples), dtype=bool)
    
    # Handle empty dataset
    if n_samples == 0:
        return similarity_matrix
    
    # All points are similar to themselves
    np.fill_diagonal(similarity_matrix, True)
    
    # For tiny datasets, compute directly
    if n_samples <= 1:
        return similarity_matrix
    
    # Start timing if verbose
    if verbose:
        start_time = time.time()
        print(f"Computing similarity matrix for {n_samples} samples with batch size {batch_size}")
    
    # Process similarity matrix in batches
    for i in range(0, n_samples, batch_size):
        i_end = min(i + batch_size, n_samples)
        chunk_i = activations[i:i_end]
        
        # Print progress
        if verbose and i % (10 * batch_size) == 0:
            elapsed = time.time() - start_time
            progress = i / n_samples
            if progress > 0:
                est_total = elapsed / progress
                est_remain = est_total - elapsed
                print(f"  Processing batch starting at {i}/{n_samples} ({progress:.1%}) - "
                      f"Est. time remaining: {est_remain:.1f}s")
        
        # Only compute upper triangle to exploit symmetry
        for j in range(i, n_samples, batch_size):
            j_end = min(j + batch_size, n_samples)
            chunk_j = activations[j:j_end]
            
            # For small enough blocks, use vectorized operations
            if (i_end - i) * (j_end - j) * n_features < 1e8:  # Memory threshold
                # Compute all pairwise differences for this block at once
                # Reshape for broadcasting: (batch_i, 1, features) - (1, batch_j, features)
                # Results in: (batch_i, batch_j, features)
                try:
                    # Reshape chunks for broadcasting
                    chunk_i_reshaped = chunk_i.reshape(i_end - i, 1, n_features)
                    chunk_j_reshaped = chunk_j.reshape(1, j_end - j, n_features)
                    
                    # Compute absolute differences
                    diffs = np.abs(chunk_i_reshaped - chunk_j_reshaped)
                    
                    # Compare with thresholds
                    is_within_threshold = diffs <= thresholds
                    
                    # Points are similar if all dimension differences are within thresholds
                    is_similar = np.all(is_within_threshold, axis=2)
                    
                    # Update similarity matrix
                    similarity_matrix[i:i_end, j:j_end] = is_similar
                    if i != j:  # Avoid double-setting the diagonal blocks
                        similarity_matrix[j:j_end, i:i_end] = is_similar.T
                except MemoryError:
                    # Fallback to non-vectorized approach if memory error occurs
                    _compute_similarity_block_nonvectorized(
                        chunk_i, chunk_j, thresholds, similarity_matrix, i, j
                    )
            else:
                # For large blocks, use non-vectorized approach to save memory
                _compute_similarity_block_nonvectorized(
                    chunk_i, chunk_j, thresholds, similarity_matrix, i, j
                )
    
    if verbose:
        total_time = time.time() - start_time
        print(f"Similarity matrix computation completed in {total_time:.2f}s")
    
    return similarity_matrix

def _compute_similarity_block_nonvectorized(
    chunk_i: np.ndarray,
    chunk_j: np.ndarray,
    thresholds: np.ndarray,
    similarity_matrix: np.ndarray,
    i_offset: int,
    j_offset: int
):
    """
    Compute similarity matrix block without vectorization to save memory.
    
    Args:
        chunk_i: First batch of samples
        chunk_j: Second batch of samples
        thresholds: Per-dimension thresholds
        similarity_matrix: Full similarity matrix to update
        i_offset: Row offset in the full matrix
        j_offset: Column offset in the full matrix
    """
    for idx_i in range(len(chunk_i)):
        global_i = i_offset + idx_i
        for idx_j in range(len(chunk_j)):
            global_j = j_offset + idx_j
            
            # Skip lower triangle to avoid redundant computation
            if global_i <= global_j:
                # Compute absolute differences
                diffs = np.abs(chunk_i[idx_i] - chunk_j[idx_j])
                
                # Early stopping: if any dimension exceeds threshold, points are not similar
                is_similar = True
                for d in range(len(diffs)):
                    if diffs[d] > thresholds[d]:
                        is_similar = False
                        break
                
                # Update similarity matrix
                similarity_matrix[global_i, global_j] = is_similar
                if global_i != global_j:  # Avoid setting diagonal twice
                    similarity_matrix[global_j, global_i] = is_similar

def compute_ets_clustering(
    activations: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    threshold_percentile: float = 0.1,
    min_threshold: float = 1e-5,
    batch_size: int = 1000,
    verbose: bool = False,
    return_similarity: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Cluster activations using Explainable Threshold Similarity (ETS).
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        thresholds: Optional array of thresholds per dimension
        threshold_percentile: Percentile to use for automatic threshold calculation (0.0-1.0)
        min_threshold: Minimum threshold value to avoid numerical issues
        batch_size: Batch size for processing large datasets
        verbose: Whether to print progress updates
        return_similarity: Whether to return the similarity matrix as well
        
    Returns:
        Tuple of (cluster_labels, thresholds) or (cluster_labels, thresholds, similarity_matrix)
    """
    n_samples, n_features = activations.shape
    
    # Handle empty dataset
    if n_samples == 0:
        return np.array([]), np.zeros(n_features)
    
    # Handle single datapoint
    if n_samples == 1:
        return np.array([0]), np.full(n_features, min_threshold)
    
    # Convert to numpy array if needed
    if not isinstance(activations, np.ndarray):
        activations = np.array(activations)
    
    # Ensure activations are float type for numerical stability
    if not np.issubdtype(activations.dtype, np.floating):
        activations = activations.astype(np.float32)
    
    # Compute thresholds if not provided
    if thresholds is None:
        if verbose:
            print("Computing dimension thresholds...")
        thresholds = compute_dimension_thresholds(
            activations, threshold_percentile, min_threshold
        )
    else:
        # Ensure thresholds are numpy array
        thresholds = np.array(thresholds)
        
        # Verify thresholds shape
        if thresholds.shape[0] != n_features:
            raise ValueError(f"Thresholds must have the same size as the number of features: "
                           f"got {thresholds.shape[0]}, expected {n_features}")
    
    if verbose:
        print(f"Using thresholds with mean: {thresholds.mean():.5f}, "
              f"min: {thresholds.min():.5f}, max: {thresholds.max():.5f}")
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(
        activations, thresholds, batch_size, verbose
    )
    
    # Form clusters using connected components
    if verbose:
        print("Finding connected components...")
    
    # Convert to sparse matrix for efficiency with connected_components
    similarity_sparse = csr_matrix(similarity_matrix)
    n_clusters, cluster_labels = connected_components(
        similarity_sparse, directed=False, return_labels=True
    )
    
    if verbose:
        print(f"Found {n_clusters} clusters")
        # Print cluster sizes
        cluster_sizes = np.bincount(cluster_labels)
        print(f"Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, "
              f"mean={cluster_sizes.mean():.1f}")
    
    if return_similarity:
        return cluster_labels, thresholds, similarity_matrix
    else:
        return cluster_labels, thresholds

def explain_ets_similarity(
    point1: np.ndarray,
    point2: np.ndarray,
    thresholds: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Explain why two points are similar or different based on ETS thresholds.
    
    Args:
        point1: First point's features
        point2: Second point's features
        thresholds: Per-dimension thresholds
        feature_names: Optional names for the features/dimensions
        
    Returns:
        Dictionary with explanation about similarity
    """
    if len(point1) != len(thresholds) or len(point2) != len(thresholds):
        raise ValueError("Points and thresholds must have the same dimension")
    
    # Compute absolute differences
    diffs = np.abs(point1 - point2)
    
    # Compare with thresholds
    is_within_threshold = diffs <= thresholds
    
    # Overall similarity
    is_similar = np.all(is_within_threshold)
    
    # Prepare dimension-wise explanation
    dimensions = []
    for i in range(len(thresholds)):
        dim_name = feature_names[i] if feature_names is not None and i < len(feature_names) else f"Dimension {i}"
        
        dimensions.append({
            "name": dim_name,
            "value1": float(point1[i]),
            "value2": float(point2[i]),
            "difference": float(diffs[i]),
            "threshold": float(thresholds[i]),
            "within_threshold": bool(is_within_threshold[i])
        })
    
    # Find dimensions that cause dissimilarity
    if not is_similar:
        distinguishing_dims = [d["name"] for d in dimensions if not d["within_threshold"]]
    else:
        distinguishing_dims = []
    
    return {
        "is_similar": is_similar,
        "dimensions": dimensions,
        "distinguishing_dimensions": distinguishing_dims,
        "num_dimensions_compared": len(thresholds),
        "num_dimensions_within_threshold": np.sum(is_within_threshold)
    }

def compute_ets_statistics(
    activations: np.ndarray,
    cluster_labels: np.ndarray,
    thresholds: np.ndarray
) -> Dict:
    """
    Compute statistics about ETS clustering result.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        cluster_labels: Cluster assignments from ETS clustering
        thresholds: Per-dimension thresholds used for clustering
        
    Returns:
        Dictionary of statistics about the clustering
    """
    n_samples, n_features = activations.shape
    n_clusters = len(np.unique(cluster_labels))
    
    # Compute cluster sizes
    cluster_sizes = np.bincount(cluster_labels)
    
    # Compute threshold utilization
    # For each cluster, count how many dimensions are "active" in cluster definition
    # A dimension is active if it's necessary for cluster integrity
    active_dimensions = np.zeros((n_clusters, n_features), dtype=bool)
    
    # For each cluster, check which dimensions are near the threshold boundary
    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_points = activations[cluster_mask]
        
        if len(cluster_points) <= 1:
            continue
        
        # For each dimension, compute max pairwise difference within cluster
        for j in range(n_features):
            values = cluster_points[:, j]
            max_diff = np.max(values) - np.min(values)
            
            # If max difference is close to threshold, this dimension is active
            if max_diff > thresholds[j] * 0.8:
                active_dimensions[cluster_id, j] = True
    
    # Count active dimensions per cluster
    active_dims_count = np.sum(active_dimensions, axis=1)
    
    # Compute dimension importance (how often each dimension is active)
    dimension_importance = np.mean(active_dimensions, axis=0)
    
    return {
        "n_clusters": n_clusters,
        "cluster_sizes": {
            "min": int(np.min(cluster_sizes)),
            "max": int(np.max(cluster_sizes)),
            "mean": float(np.mean(cluster_sizes)),
            "median": float(np.median(cluster_sizes))
        },
        "active_dimensions": {
            "min": int(np.min(active_dims_count)),
            "max": int(np.max(active_dims_count)),
            "mean": float(np.mean(active_dims_count)),
            "median": float(np.median(active_dims_count))
        },
        "dimension_importance": {
            str(i): float(importance) 
            for i, importance in enumerate(dimension_importance)
        },
        "threshold_stats": {
            "min": float(np.min(thresholds)),
            "max": float(np.max(thresholds)),
            "mean": float(np.mean(thresholds)),
            "median": float(np.median(thresholds))
        }
    }