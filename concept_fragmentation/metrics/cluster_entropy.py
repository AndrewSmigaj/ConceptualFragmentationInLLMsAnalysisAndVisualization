"""
Cluster Entropy Metric for Concept Fragmentation analysis.

This module provides functions to compute the cluster entropy metric,
which measures how dispersed class activations are in the representation space.
Higher entropy indicates more fragmentation (activations for a class are spread 
across many clusters), while lower entropy indicates more cohesion.
"""

import torch
import numpy as np
import warnings
import math
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.stats as stats

from ..config import METRICS, RANDOM_SEED


def _find_best_k(
    activations: np.ndarray,
    k_max: Optional[int] = None,
    n_init: int = METRICS["cluster_entropy"]["n_init"],
    max_iter: int = METRICS["cluster_entropy"]["max_iter"],
    random_state: int = RANDOM_SEED,
    sample_size: Optional[int] = 1000
) -> Tuple[np.ndarray, int, float]:
    """
    Find the best number of clusters K using silhouette score.
    
    Args:
        activations: Numpy array of shape (n_samples, n_features)
        k_max: Maximum number of clusters to try
        n_init: Number of initializations for KMeans
        max_iter: Maximum number of iterations for KMeans
        random_state: Random seed for reproducibility
        sample_size: If n_samples > sample_size, subsample for efficiency
        
    Returns:
        Tuple of (cluster_assignments, best_k, best_score)
    """
    n_samples = activations.shape[0]
    
    # Handle small datasets (need at least 3 points for silhouette)
    if n_samples < 3:
        return np.zeros(n_samples, dtype=int), 1, 1.0
    
    # Default k_max if not provided
    if k_max is None:
        k_max = min(12, math.ceil(math.sqrt(n_samples)))
    
    # Subsample for efficiency when dataset is large
    use_sampling = sample_size is not None and n_samples > sample_size
    if use_sampling:
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_acts = activations[indices]
    else:
        sample_acts = activations
    
    # Try different K values and compute silhouette score
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n_samples, dtype=int)
    
    # K must be at least 2 and at most n_samples-1
    min_k = 2
    max_k = min(k_max, n_samples - 1)
    
    if max_k < min_k:  # Not enough samples for clustering
        return best_labels, best_k, best_score
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        
        try:
            if use_sampling:
                sample_labels = kmeans.fit_predict(sample_acts)
                if len(np.unique(sample_labels)) < 2:
                    continue  # Skip if we got only one cluster
                score = silhouette_score(sample_acts, sample_labels)
            else:
                labels = kmeans.fit_predict(activations)
                if len(np.unique(labels)) < 2:
                    continue  # Skip if we got only one cluster
                score = silhouette_score(activations, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
                if not use_sampling:
                    best_labels = labels
        except Exception as e:
            # Skip when silhouette score fails (e.g., one cluster)
            continue
    
    # If we used sampling, refit with the best K on full data
    if use_sampling and best_k > 1:
        kmeans = KMeans(
            n_clusters=best_k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        best_labels = kmeans.fit_predict(activations)
    
    return best_labels, best_k, best_score


def compute_cluster_assignments(
    activations: np.ndarray,
    n_clusters: int,
    n_init: int,
    max_iter: int,
    random_state: int
) -> Tuple[np.ndarray, KMeans]:
    """Fit K-means once on the whole activation matrix and return assignments."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    cluster_labels = kmeans.fit_predict(activations)
    return cluster_labels, kmeans


def compute_cluster_entropy(
    activations: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = METRICS["cluster_entropy"]["default_k"],
    n_init: int = METRICS["cluster_entropy"]["n_init"],
    max_iter: int = METRICS["cluster_entropy"]["max_iter"],
    random_state: int = RANDOM_SEED,
    normalize: bool = True,
    layer_name: Optional[str] = None,
    return_clusters: bool = False,
    k_selection: str = 'auto',
    k_max: Optional[int] = None,
    sample_size: Optional[int] = 1000
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute the cluster entropy metric for concept fragmentation.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
                    or dictionary mapping layer names to activations
        labels: Tensor of shape (n_samples,) containing class labels
        n_clusters: Number of clusters for KMeans (used only when k_selection='fixed')
        n_init: Number of initializations for KMeans
        max_iter: Maximum number of iterations for KMeans
        random_state: Random seed for reproducibility
        normalize: Whether to normalize entropy by log2(n_clusters)
        layer_name: Layer name to use when activations is a dictionary
        return_clusters: Whether to return cluster assignments
        k_selection: Method to select number of clusters ('auto' or 'fixed')
        k_max: Maximum number of clusters to try when k_selection='auto'
        sample_size: Subsample size for silhouette computation when n_samples is large
        
    Returns:
        Dictionary containing:
            - 'mean_entropy': Mean entropy across all classes
            - 'class_entropies': Dictionary mapping class labels to entropy values
            - 'chosen_k': The selected number of clusters (only when k_selection='auto')
    """
    # Handle dictionary input
    if isinstance(activations, dict):
        if layer_name is None:
            raise ValueError("layer_name must be provided when activations is a dictionary")
        activations = activations[layer_name]
    
    # Convert tensors to numpy arrays
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Backward compatibility check
    if n_clusters is not METRICS["cluster_entropy"]["default_k"] and k_selection == 'auto':
        warnings.warn(
            "Passing n_clusters without setting k_selection='fixed' is deprecated. "
            "The function will use fixed k, but please update your code.",
            DeprecationWarning
        )
        k_selection = 'fixed'
    
    # Get unique class labels
    unique_labels = np.unique(labels)
    
    # Get cluster assignments based on selection method
    chosen_k = None
    
    if k_selection == 'auto':
        global_assignments, chosen_k, _ = _find_best_k(
            activations, k_max, n_init, max_iter, random_state, sample_size
        )
    else:  # 'fixed'
        # Handle edge case where n_clusters > n_samples
        effective_n_clusters = min(n_clusters, len(activations))
        if effective_n_clusters < 2:
            global_assignments = np.zeros(len(activations), dtype=int)
            chosen_k = 1
        else:
            global_assignments, kmeans = compute_cluster_assignments(
                activations, effective_n_clusters, n_init, max_iter, random_state
            )
            chosen_k = effective_n_clusters
    
    # Compute entropy for each class
    class_entropies = {}
    clusters_by_class = {}
    
    for label in unique_labels:
        # Get indices belonging to this class
        class_indices = np.where(labels == label)[0]
        class_clusters = global_assignments[class_indices]
        
        # Compute distribution across clusters
        if chosen_k == 1:
            # Only one cluster, entropy is 0
            entropy = 0.0
        else:
            distribution = np.bincount(class_clusters, minlength=chosen_k).astype(float)
            if np.sum(distribution) > 0:
                distribution = distribution / np.sum(distribution)
            
            # Compute entropy
            nonzero = distribution[distribution > 0]
            if len(nonzero) > 0:
                entropy = -np.sum(nonzero * np.log2(nonzero))
                # Normalize if requested
                if normalize and chosen_k > 1:
                    max_entropy = np.log2(chosen_k)
                    if max_entropy > 0:
                        entropy = entropy / max_entropy
            else:
                entropy = 0.0
        
        # Store result (use abs to avoid -0.0 display)
        class_entropies[int(label)] = float(abs(entropy))
        
        # Store clusters if requested
        if return_clusters:
            clusters_by_class[int(label)] = class_clusters
    
    # Compute mean entropy across all classes
    mean_entropy = float(np.mean(list(class_entropies.values())))
    
    # Create result dictionary
    result = {
        'mean_entropy': mean_entropy,
        'class_entropies': class_entropies,
        # For backward compatibility with tests
        'mean': mean_entropy,
        'per_class': class_entropies,
        'max': float(max(class_entropies.values())) if class_entropies else 0.0
    }
    
    # Add cluster assignments if requested
    if return_clusters:
        result['cluster_assignments'] = clusters_by_class
    
    # Add chosen_k when using auto selection
    if k_selection == 'auto':
        result['chosen_k'] = chosen_k
    
    return result


def compute_fragmentation_score(
    activations: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: Optional[int] = METRICS["cluster_entropy"]["default_k"],
    n_init: int = METRICS["cluster_entropy"]["n_init"],
    max_iter: int = METRICS["cluster_entropy"]["max_iter"],
    random_state: int = RANDOM_SEED,
    k_selection: str = 'auto'
) -> float:
    """
    Compute a single fragmentation score based on cluster entropy.
    Higher values indicate more fragmentation.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
        labels: Tensor of shape (n_samples,) containing class labels
        n_clusters: Number of clusters for KMeans (used only when k_selection='fixed')
        n_init: Number of initializations for KMeans
        max_iter: Maximum number of iterations for KMeans
        random_state: Random seed for reproducibility
        k_selection: Method to select number of clusters ('auto' or 'fixed')
        
    Returns:
        Fragmentation score (normalized mean entropy)
    """
    result = compute_cluster_entropy(
        activations, labels, n_clusters, n_init, max_iter, random_state, 
        normalize=True, k_selection=k_selection
    )
    
    return result['mean_entropy']
