"""
Simple clusterer for GPT-2 pivot experiment activations.

This applies k-means clustering with silhouette-based k selection
to each layer of GPT-2 activations for the pivot experiment.
"""

import pickle
import json
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class GPT2PivotClusterer:
    """
    Simple clusterer for GPT-2 pivot experiment.
    
    Applies k-means clustering to each layer with optimal k selection
    based on silhouette scores.
    """
    
    def __init__(self, k_range: Tuple[int, int] = (2, 15), random_state: int = 42, 
                 clustering_method: str = 'kmeans', threshold_percentile: float = 0.1):
        """
        Initialize the clusterer.
        
        Args:
            k_range: Range of k values to test (min, max)
            random_state: Random seed for reproducibility
            clustering_method: Clustering method to use ('kmeans', 'hdbscan', or 'ets')
            threshold_percentile: Percentile for ETS threshold computation (0.0-1.0)
        """
        self.k_range = k_range
        self.random_state = random_state
        self.clustering_method = clustering_method
        self.threshold_percentile = threshold_percentile
        self.sklearn_available = False
        self.ets_available = False
        
    def _setup_sklearn(self):
        """Setup sklearn and HDBSCAN imports."""
        try:
            global KMeans, silhouette_score, HDBSCAN
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            self.sklearn_available = True
            
            # Try to import HDBSCAN
            try:
                from hdbscan import HDBSCAN
                self.hdbscan_available = True
                print("Both sklearn and HDBSCAN available")
            except ImportError:
                self.hdbscan_available = False
                print("sklearn available, but HDBSCAN not available")
                if self.clustering_method == 'hdbscan':
                    print("Warning: HDBSCAN requested but not available, will fall back to k-means")
            
            # Try to import ETS functions
            try:
                import sys
                from pathlib import Path
                # Add root to path for imports
                root_dir = Path(__file__).parent.parent.parent.parent
                if str(root_dir) not in sys.path:
                    sys.path.insert(0, str(root_dir))
                
                global compute_dimension_thresholds, compute_similarity_matrix, extract_clusters
                from concept_fragmentation.metrics.explainable_threshold_similarity import (
                    compute_dimension_thresholds,
                    compute_similarity_matrix,
                    extract_clusters
                )
                self.ets_available = True
                print("ETS clustering functions available")
            except ImportError as e:
                self.ets_available = False
                print(f"ETS not available: {e}")
                if self.clustering_method == 'ets':
                    print("Warning: ETS requested but not available, will fall back to k-means")
            
            return True
        except ImportError:
            print("sklearn not available, cannot perform real clustering")
            self.sklearn_available = False
            self.hdbscan_available = False
            self.ets_available = False
            return False
    
    def _find_optimal_k(self, activations: List[List[float]]) -> int:
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            activations: List of activation vectors
            
        Returns:
            Optimal k value
        """
        if not self.sklearn_available:
            return self.k_range[0]  # Default to minimum k
            
        if len(activations) < self.k_range[1]:
            # Not enough samples for max k
            return min(len(activations) - 1, self.k_range[0])
        
        best_k = self.k_range[0]
        best_score = -1
        
        for k in range(self.k_range[0], self.k_range[1] + 1):
            if k >= len(activations):
                break
                
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(activations)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(activations, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        return best_k
    
    def _cluster_with_hdbscan(self, activations: List[List[float]]) -> Tuple[List[int], List[List[float]], int, float]:
        """
        Cluster activations using HDBSCAN with automatic parameter optimization.
        
        Args:
            activations: List of activation vectors
            
        Returns:
            Tuple of (cluster_labels, cluster_centers, num_clusters, silhouette_score)
        """
        if not hasattr(self, 'hdbscan_available') or not self.hdbscan_available:
            print("HDBSCAN not available, falling back to k-means")
            optimal_k = self._find_optimal_k(activations)
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(activations)
            cluster_centers = kmeans.cluster_centers_.tolist()
            silhouette = silhouette_score(activations, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
            return cluster_labels.tolist(), cluster_centers, optimal_k, silhouette
        
        import numpy as np
        
        # Try different min_cluster_size values to find optimal clustering
        min_samples_range = range(max(2, len(activations) // 50), max(3, len(activations) // 10))
        min_cluster_sizes = range(max(5, len(activations) // 100), max(6, len(activations) // 20))
        
        best_score = -1
        best_labels = None
        best_clusterer = None
        
        for min_cluster_size in min_cluster_sizes:
            for min_samples in min_samples_range:
                if min_samples >= min_cluster_size:
                    continue
                    
                try:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=0.0,
                        alpha=1.0
                    )
                    cluster_labels = clusterer.fit_predict(activations)
                    
                    # Check if we have valid clusters (not all noise)
                    unique_labels = set(cluster_labels)
                    if len(unique_labels) > 1 and -1 not in unique_labels:
                        # Calculate silhouette score
                        score = silhouette_score(activations, cluster_labels)
                        if score > best_score:
                            best_score = score
                            best_labels = cluster_labels
                            best_clusterer = clusterer
                            
                except Exception as e:
                    continue
        
        # If no good clustering found, fall back to k-means
        if best_labels is None or best_score < 0.1:
            print(f"HDBSCAN failed to find good clusters (best score: {best_score}), falling back to k-means")
            optimal_k = self._find_optimal_k(activations)
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(activations)
            cluster_centers = kmeans.cluster_centers_.tolist()
            silhouette = silhouette_score(activations, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
            return cluster_labels.tolist(), cluster_centers, optimal_k, silhouette
        
        # Calculate cluster centers for HDBSCAN results
        cluster_centers = []
        unique_labels = sorted(set(best_labels))
        
        for label in unique_labels:
            if label != -1:  # Skip noise points
                mask = np.array(best_labels) == label
                cluster_activations = np.array(activations)[mask]
                center = np.mean(cluster_activations, axis=0)
                cluster_centers.append(center.tolist())
        
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        return best_labels.tolist(), cluster_centers, num_clusters, best_score
    
    def _cluster_with_ets(self, activations: List[List[float]]) -> Tuple[List[int], List[List[float]], int, float]:
        """
        Cluster activations using ETS (Explainable Threshold Similarity).
        
        Args:
            activations: List of activation vectors
            
        Returns:
            Tuple of (cluster_labels, cluster_centers, num_clusters, silhouette_score)
        """
        if not hasattr(self, 'ets_available') or not self.ets_available:
            print("ETS not available, falling back to k-means")
            optimal_k = self._find_optimal_k(activations)
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(activations)
            cluster_centers = kmeans.cluster_centers_.tolist()
            silhouette = silhouette_score(activations, cluster_labels) if len(set(cluster_labels)) > 1 else 0.0
            return cluster_labels.tolist(), cluster_centers, optimal_k, silhouette
        
        import numpy as np
        
        # Convert to numpy array for ETS
        activations_array = np.array(activations)
        
        # Compute dimension thresholds
        thresholds = compute_dimension_thresholds(
            activations_array, 
            threshold_percentile=self.threshold_percentile,
            min_threshold=1e-5
        )
        
        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(
            activations_array,
            thresholds,
            verbose=False
        )
        
        # Extract clusters from similarity matrix
        cluster_labels = extract_clusters(similarity_matrix)
        
        # Calculate cluster centers
        unique_labels = sorted(set(cluster_labels))
        cluster_centers = []
        
        for label in unique_labels:
            mask = cluster_labels == label
            cluster_activations = activations_array[mask]
            center = np.mean(cluster_activations, axis=0)
            cluster_centers.append(center.tolist())
        
        num_clusters = len(unique_labels)
        
        # Calculate silhouette score if we have more than one cluster
        if num_clusters > 1:
            silhouette = silhouette_score(activations_array, cluster_labels)
        else:
            silhouette = 0.0
        
        # Store thresholds for interpretability
        self.last_ets_thresholds = thresholds
        
        print(f"ETS clustering found {num_clusters} clusters with percentile={self.threshold_percentile}")
        
        return cluster_labels.tolist(), cluster_centers, num_clusters, silhouette
    
    def _cluster_layer(self, layer_activations: Dict[int, Dict[int, List[float]]], layer_idx: int) -> Dict[str, Any]:
        """
        Cluster activations for a single layer.
        
        Args:
            layer_activations: Dict[sentence_idx][token_idx] = activation_vector
            layer_idx: Layer index
            
        Returns:
            Dictionary with clustering results
        """
        print(f"Clustering layer {layer_idx}...")
        
        # Collect all activations and track indices
        all_activations = []
        activation_indices = []  # (sentence_idx, token_idx)
        
        for sent_idx, tokens in layer_activations.items():
            for token_idx, activation in tokens.items():
                all_activations.append(activation)
                activation_indices.append((sent_idx, token_idx))
        
        if len(all_activations) == 0:
            return {
                'cluster_labels': {},
                'cluster_centers': [],
                'optimal_k': 0,
                'silhouette_score': 0.0,
                'layer_idx': layer_idx
            }
        
        if not self.sklearn_available:
            # Simple fallback clustering (just assign based on position)
            optimal_k = self.k_range[0]
            cluster_labels_list = [i % optimal_k for i in range(len(all_activations))]
            cluster_centers = []
            silhouette = 0.0
        elif self.clustering_method == 'kmeans':
            # K-means clustering with optimal k selection
            optimal_k = self._find_optimal_k(all_activations)
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            cluster_labels_list = kmeans.fit_predict(all_activations)
            cluster_centers = kmeans.cluster_centers_.tolist()
            
            # Calculate silhouette score
            if len(set(cluster_labels_list)) > 1:
                silhouette = silhouette_score(all_activations, cluster_labels_list)
            else:
                silhouette = 0.0
        elif self.clustering_method == 'hdbscan':
            # HDBSCAN clustering with automatic parameter optimization
            cluster_labels_list, cluster_centers, optimal_k, silhouette = self._cluster_with_hdbscan(all_activations)
        elif self.clustering_method == 'ets':
            # ETS clustering with threshold-based similarity
            cluster_labels_list, cluster_centers, optimal_k, silhouette = self._cluster_with_ets(all_activations)
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
        
        # Map back to sentence/token structure with layer-specific labels
        cluster_labels = {}
        for idx, (sent_idx, token_idx) in enumerate(activation_indices):
            cluster_id = cluster_labels_list[idx]
            cluster_label = f"L{layer_idx}C{cluster_id}"
            
            if sent_idx not in cluster_labels:
                cluster_labels[sent_idx] = {}
            cluster_labels[sent_idx][token_idx] = cluster_label
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'optimal_k': optimal_k,
            'silhouette_score': silhouette,
            'layer_idx': layer_idx
        }
    
    def cluster_all_layers(self, activations_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cluster activations for all layers.
        
        Args:
            activations_data: Dictionary from activation extractor
            
        Returns:
            Dictionary with clustering results for all layers
        """
        if not self._setup_sklearn():
            print("Proceeding with fallback clustering...")
        
        print(f"Clustering activations for {len(activations_data['sentences'])} sentences")
        print(f"Model: {activations_data['metadata']['model_name']}")
        print(f"Layers: {activations_data['metadata']['num_layers']}")
        
        # Reorganize data by layer
        num_layers = activations_data['metadata']['num_layers']
        layer_results = {}
        
        for layer_idx in range(num_layers):
            print(f"\nProcessing layer {layer_idx}...")
            
            # Extract activations for this layer across all sentences/tokens
            layer_activations = {}
            
            for sent_idx, token_data in activations_data['activations'].items():
                layer_activations[sent_idx] = {}
                for token_idx, layer_data in token_data.items():
                    if layer_idx in layer_data:
                        layer_activations[sent_idx][token_idx] = layer_data[layer_idx]
            
            # Cluster this layer
            layer_result = self._cluster_layer(layer_activations, layer_idx)
            layer_results[f"layer_{layer_idx}"] = layer_result
        
        # Create paths for each token
        token_paths = self._create_token_paths(layer_results, activations_data)
        
        results = {
            'layer_results': layer_results,
            'token_paths': token_paths,
            'sentences': activations_data['sentences'],
            'labels': activations_data.get('labels', None),  # Labels are optional
            'metadata': {
                **activations_data['metadata'],
                'clustering_method': self.clustering_method,
                'k_range': self.k_range,
                'sklearn_available': self.sklearn_available,
                'hdbscan_available': getattr(self, 'hdbscan_available', False)
            }
        }
        
        return results
    
    def _create_token_paths(self, layer_results: Dict[str, Any], activations_data: Dict[str, Any]) -> Dict[int, Dict[int, List[str]]]:
        """
        Create token paths across layers.
        
        Args:
            layer_results: Results from clustering each layer
            activations_data: Original activation data
            
        Returns:
            Dict[sentence_idx][token_idx] = [layer0_cluster, layer1_cluster, ...]
        """
        print("Creating token paths across layers...")
        
        token_paths = {}
        num_layers = activations_data['metadata']['num_layers']
        
        for sent_idx in activations_data['activations'].keys():
            token_paths[sent_idx] = {}
            
            # Get tokens for this sentence
            for token_idx in activations_data['activations'][sent_idx].keys():
                token_paths[sent_idx][token_idx] = []
                
                # Collect cluster labels across all layers
                for layer_idx in range(num_layers):
                    layer_key = f"layer_{layer_idx}"
                    if (layer_key in layer_results and 
                        sent_idx in layer_results[layer_key]['cluster_labels'] and
                        token_idx in layer_results[layer_key]['cluster_labels'][sent_idx]):
                        
                        cluster_label = layer_results[layer_key]['cluster_labels'][sent_idx][token_idx]
                        token_paths[sent_idx][token_idx].append(cluster_label)
                    else:
                        # Missing data, use default
                        token_paths[sent_idx][token_idx].append(f"L{layer_idx}C0")
        
        return token_paths
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save clustering results to file."""
        print(f"Saving clustering results to {filename}...")
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary as JSON
        summary_file = filename.replace('.pkl', '_summary.json')
        summary = {
            'num_sentences': len(results['sentences']),
            'num_layers': len(results['layer_results']),
            'metadata': results['metadata']
        }
        
        # Add layer-wise statistics
        summary['layer_stats'] = {}
        for layer_key, layer_data in results['layer_results'].items():
            summary['layer_stats'][layer_key] = {
                'optimal_k': int(layer_data['optimal_k']),
                'silhouette_score': float(layer_data['silhouette_score'])
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Results saved to {filename}")
        print(f"Summary saved to {summary_file}")


def cluster_pivot_activations():
    """Main function to cluster pivot experiment activations."""
    # Load activations
    print("Loading GPT-2 pivot activations...")
    with open("gpt2_pivot_activations.pkl", "rb") as f:
        activations_data = pickle.load(f)
    
    # Create clusterer and run clustering
    clusterer = GPT2PivotClusterer(k_range=(4, 6), random_state=42)
    results = clusterer.cluster_all_layers(activations_data)
    
    # Save results
    clusterer.save_results(results, "gpt2_pivot_clustering_results.pkl")
    
    return results


if __name__ == "__main__":
    results = cluster_pivot_activations()