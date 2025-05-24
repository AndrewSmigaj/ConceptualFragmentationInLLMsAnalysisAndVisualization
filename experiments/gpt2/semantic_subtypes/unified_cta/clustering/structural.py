"""
Structural clustering with gap statistic and multi-metric optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import warnings

from logging_config import setup_logging

logger = setup_logging(__name__)

# Layer-specific k ranges
K_RANGES = {
    'early': range(2, 6),    # Layers 0-3
    'middle': range(3, 8),   # Layers 4-8
    'late': range(4, 10)     # Layers 9-12
}


def get_layer_type(layer_idx: int) -> str:
    """Determine layer type (early/middle/late) from index."""
    if layer_idx <= 3:
        return 'early'
    elif layer_idx <= 8:
        return 'middle'
    else:
        return 'late'


def calculate_gap_statistic(
    data: np.ndarray,
    k: int,
    n_refs: int = 10,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Calculate gap statistic for given k.
    
    Gap(k) = E*[log(W_k)] - log(W_k)
    where W_k is within-cluster sum of squares
    E* is expectation under null reference distribution
    
    Args:
        data: Data matrix (n_samples, n_features)
        k: Number of clusters
        n_refs: Number of reference datasets to generate
        random_state: Random seed
        
    Returns:
        Tuple of (gap_value, gap_std)
    """
    # Fit clustering on actual data
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(data)
    
    # Calculate within-cluster sum of squares
    W_k = 0
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            center = cluster_points.mean(axis=0)
            W_k += np.sum((cluster_points - center) ** 2)
    
    # Generate reference datasets
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
        ref_labels = ref_kmeans.fit_predict(ref_data)
        
        # Calculate W_k for reference
        ref_W_k = 0
        for i in range(k):
            ref_cluster = ref_data[ref_labels == i]
            if len(ref_cluster) > 0:
                ref_center = ref_cluster.mean(axis=0)
                ref_W_k += np.sum((ref_cluster - ref_center) ** 2)
        
        ref_W_ks.append(ref_W_k)
    
    # Calculate gap statistic
    ref_W_ks = np.array(ref_W_ks)
    gap = np.mean(np.log(ref_W_ks)) - np.log(W_k)
    gap_std = np.std(np.log(ref_W_ks))
    
    return gap, gap_std


def calculate_semantic_purity(
    labels: np.ndarray,
    word_subtypes: List[str]
) -> float:
    """
    Calculate semantic purity of clustering.
    
    Purity = fraction of words in each cluster belonging to dominant subtype
    
    Args:
        labels: Cluster assignments
        word_subtypes: Semantic subtype for each word
        
    Returns:
        Purity score (0-1)
    """
    if len(set(labels)) == 1:  # Only one cluster
        return 0.0
    
    total_correct = 0
    
    # For each cluster
    for cluster_id in np.unique(labels):
        # Get words in this cluster
        cluster_mask = labels == cluster_id
        cluster_subtypes = [word_subtypes[i] for i in range(len(labels)) if cluster_mask[i]]
        
        if cluster_subtypes:
            # Count most common subtype
            subtype_counts = Counter(cluster_subtypes)
            most_common_count = subtype_counts.most_common(1)[0][1]
            total_correct += most_common_count
    
    purity = total_correct / len(labels)
    return purity


def optimize_k_selection(
    data: np.ndarray,
    word_subtypes: List[str],
    layer_idx: int,
    min_purity: float = 0.65,
    weights: Dict[str, float] = None
) -> Dict[int, Dict]:
    """
    Find optimal k using multiple metrics.
    
    Args:
        data: Preprocessed activation data
        word_subtypes: Semantic subtypes for each word
        layer_idx: Layer index (determines k range)
        min_purity: Minimum acceptable purity
        weights: Metric weights (default: silhouette=0.4, gap=0.3, purity=0.3)
        
    Returns:
        Dict mapping k to metrics and scores
    """
    if weights is None:
        weights = {
            'silhouette': 0.4,
            'gap': 0.3,
            'purity': 0.3
        }
    
    # Get appropriate k range
    layer_type = get_layer_type(layer_idx)
    k_range = K_RANGES[layer_type]
    
    logger.info(f"Optimizing k for layer {layer_idx} ({layer_type}), "
               f"k_range={list(k_range)}")
    
    results = {}
    
    # Calculate metrics for each k
    for k in k_range:
        logger.info(f"  Testing k={k}")
        
        # Fit clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        
        # Calculate metrics
        metrics = {}
        
        # Silhouette score
        if k > 1:
            metrics['silhouette'] = silhouette_score(data, labels)
        else:
            metrics['silhouette'] = 0.0
        
        # Gap statistic
        gap, gap_std = calculate_gap_statistic(data, k, n_refs=10)
        metrics['gap'] = gap
        metrics['gap_std'] = gap_std
        
        # Semantic purity
        purity = calculate_semantic_purity(labels, word_subtypes)
        metrics['purity'] = purity
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = counts.tolist()
        metrics['size_ratio'] = counts.max() / counts.min()
        
        # Store clustering results
        metrics['labels'] = labels
        metrics['centers'] = kmeans.cluster_centers_
        metrics['inertia'] = kmeans.inertia_
        
        results[k] = metrics
        
        logger.info(f"    silhouette={metrics['silhouette']:.3f}, "
                   f"gap={metrics['gap']:.3f}, "
                   f"purity={metrics['purity']:.3f}")
    
    # Find optimal k
    best_k = None
    best_score = -float('inf')
    best_with_purity = None
    best_score_with_purity = -float('inf')
    
    for k, metrics in results.items():
        # Normalize metrics to 0-1 range
        # Silhouette is already in [-1, 1], shift to [0, 1]
        norm_silhouette = (metrics['silhouette'] + 1) / 2
        
        # Gap statistic - normalize by max gap
        max_gap = max(m['gap'] for m in results.values())
        norm_gap = metrics['gap'] / max_gap if max_gap > 0 else 0
        
        # Purity is already in [0, 1]
        norm_purity = metrics['purity']
        
        # Calculate weighted score
        score = (
            weights['silhouette'] * norm_silhouette +
            weights['gap'] * norm_gap +
            weights['purity'] * norm_purity
        )
        
        # Penalize very imbalanced clusters
        if metrics['size_ratio'] > 10:
            score *= 0.9
        
        results[k]['weighted_score'] = score
        
        # Track best overall
        if score > best_score:
            best_score = score
            best_k = k
        
        # Track best with purity constraint
        if metrics['purity'] >= min_purity and score > best_score_with_purity:
            best_score_with_purity = score
            best_with_purity = k
    
    # Use best with purity constraint if available
    if best_with_purity is not None:
        optimal_k = best_with_purity
        logger.info(f"Optimal k={optimal_k} (meets purity constraint)")
    else:
        optimal_k = best_k
        logger.warning(f"No k meets purity constraint {min_purity}, "
                      f"using best k={optimal_k}")
    
    results['optimal_k'] = optimal_k
    return results


class StructuralClusterer:
    """
    Structural clustering for macro-level organization.
    """
    
    def __init__(self, min_purity: float = 0.65):
        """
        Initialize structural clusterer.
        
        Args:
            min_purity: Minimum acceptable semantic purity
        """
        self.min_purity = min_purity
        self.clustering_results = {}
        self.optimal_k_per_layer = {}
        
    def fit(self, 
            preprocessed_data: Dict[str, np.ndarray],
            word_subtypes: List[str]) -> Dict[str, Dict]:
        """
        Fit structural clustering on all layers.
        
        Args:
            preprocessed_data: Preprocessed activations by layer
            word_subtypes: Semantic subtypes for each word
            
        Returns:
            Clustering results for all layers
        """
        logger.info("Starting structural clustering")
        
        for layer_name, data in preprocessed_data.items():
            layer_idx = int(layer_name.split('_')[1])
            
            # Optimize k selection
            k_results = optimize_k_selection(
                data, word_subtypes, layer_idx, 
                self.min_purity
            )
            
            optimal_k = k_results['optimal_k']
            self.optimal_k_per_layer[layer_name] = optimal_k
            
            # Store full results
            self.clustering_results[layer_name] = k_results[optimal_k]
            self.clustering_results[layer_name]['k'] = optimal_k
            self.clustering_results[layer_name]['all_k_results'] = k_results
            
            logger.info(f"{layer_name}: optimal k={optimal_k}, "
                       f"purity={k_results[optimal_k]['purity']:.3f}")
        
        return self.clustering_results
    
    def get_macro_clusters(self, layer_name: str) -> np.ndarray:
        """Get cluster labels for a specific layer."""
        if layer_name not in self.clustering_results:
            raise ValueError(f"Layer {layer_name} not found in results")
        return self.clustering_results[layer_name]['labels']
    
    def get_cluster_centers(self, layer_name: str) -> np.ndarray:
        """Get cluster centers for a specific layer."""
        if layer_name not in self.clustering_results:
            raise ValueError(f"Layer {layer_name} not found in results")
        return self.clustering_results[layer_name]['centers']
    
    def find_optimal_k_gap(self, data: np.ndarray, k_range: Tuple[int, int], n_refs: int = 10) -> Tuple[int, List[Dict]]:
        """
        Find optimal k using gap statistic.
        
        Args:
            data: Data to cluster
            k_range: (min_k, max_k) range to test
            n_refs: Number of reference datasets
            
        Returns:
            Tuple of (optimal_k, gap_results)
        """
        results = []
        k_values = list(range(k_range[0], k_range[1] + 1))
        
        for k in k_values:
            gap, s_k = calculate_gap_statistic(data, k, n_refs)
            results.append({
                'k': k,
                'gap': gap,
                's_k': s_k
            })
        
        # Find optimal k using gap statistic criterion
        optimal_k = k_values[0]
        for i in range(len(results) - 1):
            if results[i]['gap'] >= results[i + 1]['gap'] - results[i + 1]['s_k']:
                optimal_k = results[i]['k']
                break
        
        return optimal_k, results
    
    def cluster_with_k(self, data: np.ndarray, k: int) -> np.ndarray:
        """
        Perform clustering with specific k.
        
        Args:
            data: Data to cluster
            k: Number of clusters
            
        Returns:
            Cluster labels
        """
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data)
        return labels