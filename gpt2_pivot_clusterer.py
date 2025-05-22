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
    
    def __init__(self, k_range: Tuple[int, int] = (2, 15), random_state: int = 42):
        """
        Initialize the clusterer.
        
        Args:
            k_range: Range of k values to test (min, max)
            random_state: Random seed for reproducibility
        """
        self.k_range = k_range
        self.random_state = random_state
        self.sklearn_available = False
        
    def _setup_sklearn(self):
        """Setup sklearn imports."""
        try:
            global KMeans, silhouette_score
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            self.sklearn_available = True
            return True
        except ImportError:
            print("sklearn not available, cannot perform real clustering")
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
        
        # Find optimal k
        optimal_k = self._find_optimal_k(all_activations)
        
        if not self.sklearn_available:
            # Simple fallback clustering (just assign based on position)
            cluster_labels_list = [i % optimal_k for i in range(len(all_activations))]
            cluster_centers = []
            silhouette = 0.0
        else:
            # Real k-means clustering
            kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
            cluster_labels_list = kmeans.fit_predict(all_activations)
            cluster_centers = kmeans.cluster_centers_.tolist()
            
            # Calculate silhouette score
            if len(set(cluster_labels_list)) > 1:
                silhouette = silhouette_score(all_activations, cluster_labels_list)
            else:
                silhouette = 0.0
        
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
        num_layers = 13  # embedding + 12 transformer layers  
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
            'labels': activations_data['labels'],
            'metadata': {
                **activations_data['metadata'],
                'clustering_method': 'kmeans_with_silhouette',
                'k_range': self.k_range,
                'sklearn_available': self.sklearn_available
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
        
        for sent_idx in activations_data['activations'].keys():
            token_paths[sent_idx] = {}
            
            # Get tokens for this sentence
            for token_idx in activations_data['activations'][sent_idx].keys():
                token_paths[sent_idx][token_idx] = []
                
                # Collect cluster labels across all layers
                for layer_idx in range(13):
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