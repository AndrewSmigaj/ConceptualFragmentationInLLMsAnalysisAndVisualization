"""
Optimal Number of Clusters (k*) Metric for Concept Fragmentation analysis.

This module provides functions to compute the optimal number of clusters for each class,
which directly measures the number of distinct subgroups or 'islands' within a class.
Higher k* indicates more fragmentation (class is split into more clusters).
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

from ..config import METRICS, RANDOM_SEED


def _find_optimal_k(
    activations: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    n_init: int = METRICS["cluster_entropy"]["n_init"],
    max_iter: int = METRICS["cluster_entropy"]["max_iter"],
    random_state: int = RANDOM_SEED,
    sample_size: Optional[int] = 1000
) -> Tuple[int, float]:
    """
    Find the optimal number of clusters using silhouette score.
    
    Args:
        activations: Numpy array of shape (n_samples, n_features)
        k_range: Tuple of (min_k, max_k) to try
        n_init: Number of initializations for KMeans
        max_iter: Maximum number of iterations for KMeans
        random_state: Random seed for reproducibility
        sample_size: If n_samples > sample_size, subsample for efficiency
        
    Returns:
        Tuple of (optimal_k, best_silhouette_score)
    """
    n_samples = activations.shape[0]
    
    # Handle small datasets
    if n_samples < k_range[0] + 1:
        return 1, 0.0
    
    # Adjust max_k if needed
    min_k, max_k = k_range
    max_k = min(max_k, n_samples - 1)
    
    if max_k < min_k:
        return 1, 0.0
    
    # Subsample for efficiency when dataset is large
    use_sampling = sample_size is not None and n_samples > sample_size
    if use_sampling:
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_acts = activations[indices]
    else:
        sample_acts = activations
    
    # Try different k values
    best_k = 1
    best_score = -1.0
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state
        )
        
        try:
            labels = kmeans.fit_predict(sample_acts)
            
            # Ensure we have at least 2 clusters
            if len(np.unique(labels)) < 2:
                continue
            
            score = silhouette_score(sample_acts, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            # Skip if something goes wrong
            warnings.warn(f"Error computing silhouette for k={k}: {str(e)}")
            continue
    
    return best_k, best_score


def compute_optimal_k(
    activations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_name: Optional[str] = None,
    k_range: Tuple[int, int] = (2, 10),
    n_init: int = METRICS["cluster_entropy"]["n_init"],
    max_iter: int = METRICS["cluster_entropy"]["max_iter"],
    random_state: int = RANDOM_SEED,
    sample_size: Optional[int] = 1000
) -> Dict[str, Union[float, Dict[int, int]]]:
    """
    Compute the optimal number of clusters (k*) for each class.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
                    or dictionary mapping layer names to activations
        labels: Tensor of shape (n_samples,) containing class labels
        layer_name: Layer name to use when activations is a dictionary
        k_range: Tuple of (min_k, max_k) to try
        n_init: Number of initializations for KMeans
        max_iter: Maximum number of iterations for KMeans
        random_state: Random seed for reproducibility
        sample_size: Subsample size for silhouette computation when n_samples is large
        
    Returns:
        Dictionary containing:
            - 'mean_k': Mean optimal k across all classes
            - 'class_k': Dictionary mapping class labels to optimal k values
            - 'silhouette_scores': Dictionary mapping class labels to silhouette scores
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
    
    # Get unique class labels
    unique_labels = np.unique(labels)
    
    # Compute optimal k for each class
    class_k = {}
    silhouette_scores = {}
    
    for label in unique_labels:
        # Get activations for this class
        class_indices = np.where(labels == label)[0]
        
        # Skip if too few samples
        if len(class_indices) <= k_range[0]:
            class_k[int(label)] = 1
            silhouette_scores[int(label)] = 0.0
            continue
        
        class_acts = activations[class_indices]
        
        # Find optimal k
        optimal_k, score = _find_optimal_k(
            class_acts, k_range, n_init, max_iter, random_state, sample_size
        )
        
        # Store results
        class_k[int(label)] = int(optimal_k)
        silhouette_scores[int(label)] = float(score)
    
    # Compute mean k across all classes
    mean_k = float(np.mean(list(class_k.values()))) if class_k else 1.0
    
    # Create result dictionary
    result = {
        'mean_k': mean_k,
        'class_k': class_k,
        'silhouette_scores': silhouette_scores,
        # For consistency with other metrics
        'mean': mean_k,
        'per_class': class_k,
        'max': float(max(class_k.values())) if class_k else 1.0
    }
    
    return result


def compute_fragmentation_score(
    activations: torch.Tensor,
    labels: torch.Tensor,
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = RANDOM_SEED
) -> float:
    """
    Compute a single fragmentation score based on optimal number of clusters.
    Higher values indicate more fragmentation.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
        labels: Tensor of shape (n_samples,) containing class labels
        k_range: Tuple of (min_k, max_k) to try
        random_state: Random seed for reproducibility
        
    Returns:
        Fragmentation score (mean optimal k across classes)
    """
    result = compute_optimal_k(
        activations, labels, k_range=k_range, random_state=random_state
    )
    
    return result['mean_k'] 