"""
Clustering configuration for Concept MRI.
Wraps existing clustering infrastructure from concept_fragmentation.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Add parent directory to path to import concept_fragmentation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_fragmentation.clustering.base import BaseClusterer
from concept_fragmentation.metrics.optimal_num_clusters import (
    compute_gap_statistic, find_optimal_k_binary_search
)

class ClusteringConfig:
    """
    Configuration and execution of clustering analysis.
    Wraps existing clustering infrastructure for UI integration.
    """
    
    def __init__(self):
        self.algorithm = 'kmeans'
        self.metric = 'gap'
        self.distance_metric = 'euclidean'
        self.random_seed = 42
        self.results = {}
        
    def configure(self, **kwargs):
        """
        Configure clustering parameters.
        
        Args:
            algorithm: 'kmeans' or 'dbscan'
            metric: 'gap', 'silhouette', 'elbow'
            distance_metric: 'euclidean', 'cosine', 'manhattan'
            random_seed: Random seed for reproducibility
            manual_k: Manual number of clusters (if specified)
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN min_samples parameter
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def run_clustering(self, activations: Dict[int, np.ndarray], **kwargs) -> Dict[str, Any]:
        """
        Run clustering on activations for each layer.
        
        Args:
            activations: Dictionary mapping layer index to activation arrays
            **kwargs: Additional clustering parameters
            
        Returns:
            Dictionary containing clustering results
        """
        self.configure(**kwargs)
        
        results = {
            'algorithm': self.algorithm,
            'metric': self.metric,
            'distance_metric': self.distance_metric,
            'clusters_per_layer': {},
            'metrics': {}
        }
        
        # Cluster each layer
        for layer_idx, layer_activations in activations.items():
            layer_results = self._cluster_layer(layer_activations, layer_idx, **kwargs)
            results['clusters_per_layer'][f'layer_{layer_idx}'] = layer_results
        
        return results
    
    def _cluster_layer(self, activations: np.ndarray, layer_idx: int, **kwargs) -> Dict[str, Any]:
        """
        Cluster activations for a single layer.
        
        Args:
            activations: Activation array for the layer
            layer_idx: Index of the layer
            **kwargs: Clustering parameters
            
        Returns:
            Dictionary containing clustering results for the layer
        """
        if self.algorithm == 'kmeans':
            return self._cluster_kmeans(activations, layer_idx, **kwargs)
        elif self.algorithm == 'dbscan':
            return self._cluster_dbscan(activations, layer_idx, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _cluster_kmeans(self, activations: np.ndarray, layer_idx: int, **kwargs) -> Dict[str, Any]:
        """Run K-means clustering."""
        manual_k = kwargs.get('manual_k')
        
        # Determine optimal k if not manually specified
        if manual_k:
            optimal_k = manual_k
            metric_scores = None
        else:
            optimal_k, metric_scores = self._find_optimal_k(activations)
        
        # Run clustering with optimal k
        clusterer = KMeans(
            n_clusters=optimal_k,
            random_state=self.random_seed,
            n_init=10
        )
        labels = clusterer.fit_predict(activations)
        
        # Calculate metrics
        silhouette = silhouette_score(activations, labels) if len(np.unique(labels)) > 1 else 0
        inertia = clusterer.inertia_
        
        return {
            'n_clusters': optimal_k,
            'labels': labels.tolist(),
            'centroids': clusterer.cluster_centers_.tolist(),
            'silhouette_score': float(silhouette),
            'inertia': float(inertia),
            'metric_scores': metric_scores
        }
    
    def _cluster_dbscan(self, activations: np.ndarray, layer_idx: int, **kwargs) -> Dict[str, Any]:
        """Run DBSCAN clustering."""
        eps = kwargs.get('dbscan_eps', 0.5)
        min_samples = kwargs.get('dbscan_min_samples', 5)
        
        # Normalize if using cosine distance
        if self.distance_metric == 'cosine':
            from sklearn.preprocessing import normalize
            activations = normalize(activations)
            metric = 'euclidean'  # After normalization
        else:
            metric = self.distance_metric
        
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )
        labels = clusterer.fit_predict(activations)
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        silhouette = silhouette_score(activations, labels) if n_clusters > 1 else 0
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'labels': labels.tolist(),
            'silhouette_score': float(silhouette),
            'eps': eps,
            'min_samples': min_samples
        }
    
    def _find_optimal_k(self, activations: np.ndarray) -> Tuple[int, Dict[str, List[float]]]:
        """
        Find optimal k using selected metric.
        
        Args:
            activations: Activation array
            
        Returns:
            Tuple of (optimal_k, metric_scores)
        """
        min_k = 2
        max_k = min(20, activations.shape[0] // 10)  # Reasonable upper bound
        
        if self.metric == 'gap':
            # Use existing gap statistic implementation
            optimal_k = find_optimal_k_binary_search(
                activations,
                min_k=min_k,
                max_k=max_k,
                n_refs=10,
                random_state=self.random_seed
            )
            
            # Calculate gap scores for visualization
            k_values = list(range(min_k, min(max_k, optimal_k + 5)))
            gap_scores = []
            for k in k_values:
                gap, _ = compute_gap_statistic(activations, k, n_refs=5)
                gap_scores.append(float(gap))
            
            return optimal_k, {'k_values': k_values, 'gap_values': gap_scores}
            
        elif self.metric == 'silhouette':
            # Calculate silhouette scores
            k_values = list(range(min_k, max_k + 1))
            silhouette_scores = []
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
                labels = kmeans.fit_predict(activations)
                score = silhouette_score(activations, labels)
                silhouette_scores.append(float(score))
            
            # Find k with highest silhouette score
            optimal_k = k_values[np.argmax(silhouette_scores)]
            
            return optimal_k, {
                'k_values': k_values,
                'silhouette_scores': silhouette_scores
            }
            
        elif self.metric == 'elbow':
            # Calculate inertia (within-cluster sum of squares)
            k_values = list(range(min_k, max_k + 1))
            inertia_values = []
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
                kmeans.fit(activations)
                inertia_values.append(float(kmeans.inertia_))
            
            # Simple elbow detection (find point of maximum curvature)
            # In practice, this is often selected visually
            optimal_k = self._detect_elbow(k_values, inertia_values)
            
            return optimal_k, {
                'k_values': k_values,
                'inertia_values': inertia_values
            }
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _detect_elbow(self, k_values: List[int], scores: List[float]) -> int:
        """
        Simple elbow detection using angle-based method.
        
        Args:
            k_values: List of k values
            scores: List of corresponding scores (e.g., inertia)
            
        Returns:
            Optimal k at the elbow point
        """
        if len(k_values) < 3:
            return k_values[0]
        
        # Calculate angles between consecutive points
        angles = []
        for i in range(1, len(k_values) - 1):
            # Vectors from point i to i-1 and i to i+1
            v1 = np.array([k_values[i] - k_values[i-1], scores[i] - scores[i-1]])
            v2 = np.array([k_values[i+1] - k_values[i], scores[i+1] - scores[i]])
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
        
        # Elbow is at minimum angle (sharpest turn)
        elbow_idx = np.argmin(angles) + 1
        return k_values[elbow_idx]
    
    def get_cluster_statistics(self, activations: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistics for clusters.
        
        Args:
            activations: Activation array
            labels: Cluster labels
            
        Returns:
            Dictionary of cluster statistics
        """
        unique_labels = np.unique(labels)
        stats = {}
        
        for label in unique_labels:
            mask = labels == label
            cluster_activations = activations[mask]
            
            stats[f'cluster_{label}'] = {
                'size': int(np.sum(mask)),
                'mean_activation': float(np.mean(cluster_activations)),
                'std_activation': float(np.std(cluster_activations)),
                'centroid': np.mean(cluster_activations, axis=0).tolist()
            }
        
        return stats