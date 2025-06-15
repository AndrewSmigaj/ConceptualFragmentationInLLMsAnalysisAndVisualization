"""K-selection methods for clustering algorithms.

This module provides various methods for selecting the optimal number of clusters (k),
including Gap Statistic, Elbow Method, Silhouette Analysis, and Davies-Bouldin Index.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
import logging

logger = logging.getLogger(__name__)


def calculate_gap_statistic(
    data: np.ndarray,
    k: int,
    n_refs: int = 10,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Calculate gap statistic for given k.
    
    The gap statistic compares the total within-cluster variation for different
    values of k with their expected values under null reference distribution.
    
    Gap(k) = E*[log(W_k)] - log(W_k)
    where W_k is within-cluster sum of squares
    E* is expectation under null reference distribution
    
    Args:
        data: Data matrix (n_samples, n_features)
        k: Number of clusters
        n_refs: Number of reference datasets to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (gap_value, gap_std)
        
    Raises:
        ValueError: If k is invalid for the given data
    """
    # Validate inputs
    if len(data) == 0:
        raise ValueError("Cannot calculate gap statistic on empty data")
    if k < 1 or k > len(data):
        raise ValueError(f"k must be between 1 and {len(data)}, got {k}")
    
    # Fit clustering on actual data
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(data)
    
    # Use inertia directly (it's the within-cluster sum of squares)
    W_k = kmeans.inertia_
    
    # Generate reference datasets and calculate expected W_k
    n_samples, n_features = data.shape
    ref_W_ks = []
    
    # Get data range for uniform distribution
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    
    np.random.seed(random_state)
    for ref_idx in range(n_refs):
        # Generate uniform random data in same range
        ref_data = np.random.uniform(
            data_min, data_max, 
            size=(n_samples, n_features)
        )
        
        # Cluster reference data
        ref_kmeans = KMeans(n_clusters=k, random_state=random_state + ref_idx, n_init=3)
        ref_kmeans.fit(ref_data)
        
        # Get inertia for reference
        ref_W_ks.append(ref_kmeans.inertia_)
    
    # Calculate gap statistic
    ref_W_ks = np.array(ref_W_ks)
    gap = np.mean(np.log(ref_W_ks + 1e-10)) - np.log(W_k + 1e-10)
    gap_std = np.std(np.log(ref_W_ks + 1e-10))
    
    return gap, gap_std


def calculate_elbow_score(
    data: np.ndarray,
    k: int,
    random_state: int = 42
) -> float:
    """
    Calculate within-cluster sum of squares (WCSS) for elbow method.
    
    Args:
        data: Data matrix (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed
        
    Returns:
        WCSS (inertia) value
        
    Raises:
        ValueError: If k is invalid for the given data
    """
    if len(data) == 0:
        raise ValueError("Cannot calculate elbow score on empty data")
    if k < 1 or k > len(data):
        raise ValueError(f"k must be between 1 and {len(data)}, got {k}")
        
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    kmeans.fit(data)
    return kmeans.inertia_


def calculate_silhouette(
    data: np.ndarray,
    k: int,
    random_state: int = 42,
    sample_size: Optional[int] = 1000
) -> float:
    """
    Calculate silhouette score for given k.
    
    Args:
        data: Data matrix (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed
        sample_size: If specified and data has more samples, subsample for efficiency
        
    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    if len(data) == 0:
        return -1.0
    if k < 2 or k >= len(data):
        return -1.0
    
    # Subsample if needed
    if sample_size and len(data) > sample_size:
        indices = np.random.RandomState(random_state).choice(
            len(data), sample_size, replace=False
        )
        sample_data = data[indices]
    else:
        sample_data = data
    
    # Fit clustering
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(sample_data)
    
    # Check if all points are in one cluster (can happen with bad initialization)
    if len(np.unique(labels)) < 2:
        return -1.0
    
    # Calculate silhouette
    try:
        score = silhouette_score(sample_data, labels)
    except Exception as e:
        logger.warning(f"Silhouette calculation failed for k={k}: {e}")
        score = -1.0
    
    return score


def calculate_davies_bouldin(
    data: np.ndarray,
    k: int,
    random_state: int = 42
) -> float:
    """
    Calculate Davies-Bouldin index for given k.
    
    Lower values indicate better clustering.
    
    Args:
        data: Data matrix (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed
        
    Returns:
        Davies-Bouldin index (lower is better)
    """
    if len(data) == 0:
        return float('inf')
    if k < 2 or k >= len(data):
        return float('inf')
    
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    
    # Check if all points are in one cluster
    if len(np.unique(labels)) < 2:
        return float('inf')
    
    try:
        score = davies_bouldin_score(data, labels)
    except Exception as e:
        logger.warning(f"Davies-Bouldin calculation failed for k={k}: {e}")
        score = float('inf')
    
    return score


def select_optimal_k(
    data: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    method: str = 'gap',
    random_state: int = 42,
    return_scores: bool = False,
    **kwargs
) -> Union[int, Tuple[int, Dict[int, float]]]:
    """
    Select optimal number of clusters using specified method.
    
    Args:
        data: Data matrix (n_samples, n_features)
        k_range: Range of k values to test (min_k, max_k)
        method: Selection method ('gap', 'elbow', 'silhouette', 'davies_bouldin', 'combined')
        random_state: Random seed
        return_scores: If True, return all scores along with optimal k
        **kwargs: Additional arguments for specific methods
            - n_refs: Number of reference datasets for gap statistic
            - sample_size: Subsample size for silhouette
            - weights: Dict of weights for combined method
        
    Returns:
        Optimal k value, or tuple of (optimal_k, scores_dict) if return_scores=True
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Validate inputs
    if len(data) == 0:
        raise ValueError("Cannot select optimal k for empty data")
    
    min_k, max_k = k_range
    max_k = min(max_k, len(data) - 1)
    
    if max_k < min_k:
        raise ValueError(f"Invalid k_range: {k_range}. Ensure min_k <= max_k <= n_samples-1")
    
    scores = {}
    
    if method == 'gap':
        # Gap statistic method
        n_refs = kwargs.get('n_refs', 10)
        gaps = []
        gap_stds = []
        
        for k in range(min_k, max_k + 1):
            gap, gap_std = calculate_gap_statistic(data, k, n_refs, random_state)
            gaps.append(gap)
            gap_stds.append(gap_std)
            scores[k] = gap
        
        # Find optimal k using gap statistic rule
        # Choose smallest k such that Gap(k) >= Gap(k+1) - s_{k+1}
        optimal_k = min_k
        for i in range(len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - gap_stds[i + 1]:
                optimal_k = min_k + i
                break
        else:
            # If no k satisfies the rule, choose k with maximum gap
            optimal_k = min_k + np.argmax(gaps)
    
    elif method == 'elbow':
        # Elbow method - find point of maximum curvature
        wcss = []
        for k in range(min_k, max_k + 1):
            score = calculate_elbow_score(data, k, random_state)
            wcss.append(score)
            scores[k] = score
        
        # Calculate second derivative to find elbow
        if len(wcss) > 2:
            second_diff = np.diff(np.diff(wcss))
            elbow_idx = np.argmax(second_diff) + 1  # +1 because of double diff
            optimal_k = min_k + elbow_idx
        else:
            optimal_k = min_k
    
    elif method == 'silhouette':
        # Silhouette method - choose k with highest silhouette score
        sample_size = kwargs.get('sample_size', 1000)
        
        for k in range(min_k, max_k + 1):
            score = calculate_silhouette(data, k, random_state, sample_size)
            scores[k] = score
        
        optimal_k = max(scores, key=scores.get)
    
    elif method == 'davies_bouldin':
        # Davies-Bouldin method - choose k with lowest DB index
        for k in range(min_k, max_k + 1):
            score = calculate_davies_bouldin(data, k, random_state)
            scores[k] = score
        
        optimal_k = min(scores, key=scores.get)
    
    elif method == 'combined':
        # Combined method using multiple metrics
        weights = kwargs.get('weights', {
            'gap': 0.3,
            'silhouette': 0.4,
            'davies_bouldin': 0.3
        })
        
        # Calculate all metrics
        all_scores = {
            'gap': {},
            'silhouette': {},
            'davies_bouldin': {}
        }
        
        for k in range(min_k, max_k + 1):
            gap, _ = calculate_gap_statistic(data, k, random_state=random_state)
            all_scores['gap'][k] = gap
            all_scores['silhouette'][k] = calculate_silhouette(
                data, k, random_state, kwargs.get('sample_size', 1000)
            )
            all_scores['davies_bouldin'][k] = calculate_davies_bouldin(
                data, k, random_state
            )
        
        # Normalize scores to [0, 1]
        normalized_scores = {}
        for metric, metric_scores in all_scores.items():
            values = list(metric_scores.values())
            min_val, max_val = min(values), max(values)
            
            if metric == 'davies_bouldin':
                # Lower is better for DB index, so invert
                if max_val > min_val:
                    normalized = {
                        k: 1 - (v - min_val) / (max_val - min_val)
                        for k, v in metric_scores.items()
                    }
                else:
                    normalized = {k: 1.0 for k in metric_scores}
            else:
                # Higher is better for gap and silhouette
                if max_val > min_val:
                    normalized = {
                        k: (v - min_val) / (max_val - min_val)
                        for k, v in metric_scores.items()
                    }
                else:
                    normalized = {k: 1.0 for k in metric_scores}
            normalized_scores[metric] = normalized
        
        # Calculate weighted combined score
        combined_scores = {}
        for k in range(min_k, max_k + 1):
            score = sum(
                weights.get(metric, 0) * normalized_scores[metric][k]
                for metric in normalized_scores
            )
            combined_scores[k] = score
            scores[k] = score
        
        optimal_k = max(combined_scores, key=combined_scores.get)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if return_scores:
        return optimal_k, scores
    else:
        return optimal_k