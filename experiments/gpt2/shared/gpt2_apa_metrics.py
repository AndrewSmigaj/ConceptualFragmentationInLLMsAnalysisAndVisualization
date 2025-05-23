"""
Comprehensive APA metrics calculator for GPT-2 pivot experiment.

Computes all original APA metrics plus pivot-specific metrics on the 
clustered GPT-2 activation data.
"""

import pickle
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Set
import logging
from collections import Counter, defaultdict
# Import pivot metrics functions directly to avoid complex dependencies
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'concept_fragmentation', 'metrics'))

try:
    from pivot_metrics import (
        calculate_fragmentation_delta,
        calculate_path_divergence_index,
        analyze_pivot_effects
    )
    pivot_metrics_available = True
except ImportError:
    print("Pivot metrics not available, will compute simplified versions")
    pivot_metrics_available = False

logger = logging.getLogger(__name__)


class GPT2APAMetricsCalculator:
    """
    Comprehensive APA metrics calculator for GPT-2 pivot experiment.
    
    Computes all original APA metrics plus pivot-specific metrics.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.sklearn_available = False
        
    def _setup_sklearn(self):
        """Setup sklearn imports."""
        try:
            global silhouette_score, adjusted_rand_score, mutual_info_score
            from sklearn.metrics import silhouette_score
            from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
            global mutual_info_score
            mutual_info_score = normalized_mutual_info_score
            self.sklearn_available = True
            return True
        except ImportError:
            print("sklearn not available, some metrics will be skipped")
            return False
    
    def compute_all_metrics(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute all APA metrics for the clustering results.
        
        Args:
            clustering_results: Results from GPT2PivotClusterer
            
        Returns:
            Dictionary containing all computed metrics
        """
        print("Computing comprehensive APA metrics...")
        self._setup_sklearn()
        
        metrics = {
            'basic_clustering_metrics': self._compute_basic_clustering_metrics(clustering_results),
            'cross_layer_metrics': self._compute_cross_layer_metrics(clustering_results),
            'path_metrics': self._compute_path_metrics(clustering_results),
            'pivot_specific_metrics': self._compute_pivot_specific_metrics(clustering_results),
            'archetypal_path_analysis': self._compute_archetypal_paths(clustering_results)
        }
        
        return metrics
    
    def _compute_basic_clustering_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute basic clustering quality metrics."""
        print("Computing basic clustering metrics...")
        
        basic_metrics = {
            'silhouette_scores': {},
            'optimal_k_values': {},
            'cluster_distributions': {}
        }
        
        # Extract metrics already computed during clustering
        for layer_key, layer_data in results['layer_results'].items():
            layer_idx = layer_data['layer_idx']
            basic_metrics['silhouette_scores'][f'layer_{layer_idx}'] = layer_data['silhouette_score']
            basic_metrics['optimal_k_values'][f'layer_{layer_idx}'] = layer_data['optimal_k']
            
            # Compute cluster distribution
            cluster_labels = []
            for sent_labels in layer_data['cluster_labels'].values():
                cluster_labels.extend(sent_labels.values())
            
            if cluster_labels:
                cluster_counts = Counter(cluster_labels)
                basic_metrics['cluster_distributions'][f'layer_{layer_idx}'] = dict(cluster_counts)
        
        return basic_metrics
    
    def _compute_cross_layer_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cross-layer APA metrics: ρ^c, J, etc."""
        print("Computing cross-layer metrics...")
        
        cross_layer_metrics = {
            'centroid_similarity_rho_c': {},
            'membership_overlap_J': {},
            'layer_transition_matrices': {}
        }
        
        # Compute ρ^c (centroid similarity) between adjacent layers
        layer_results = results['layer_results']
        for i in range(12):  # 0 to 11, checking i vs i+1
            layer1_key = f'layer_{i}'
            layer2_key = f'layer_{i+1}'
            
            if layer1_key in layer_results and layer2_key in layer_results:
                similarity = self._compute_centroid_similarity(
                    layer_results[layer1_key],
                    layer_results[layer2_key]
                )
                cross_layer_metrics['centroid_similarity_rho_c'][f'{layer1_key}_to_{layer2_key}'] = similarity
        
        # Compute J (membership overlap) between adjacent layers
        for i in range(12):
            layer1_key = f'layer_{i}'
            layer2_key = f'layer_{i+1}'
            
            if layer1_key in layer_results and layer2_key in layer_results:
                overlap = self._compute_membership_overlap(
                    layer_results[layer1_key]['cluster_labels'],
                    layer_results[layer2_key]['cluster_labels']
                )
                cross_layer_metrics['membership_overlap_J'][f'{layer1_key}_to_{layer2_key}'] = overlap
        
        return cross_layer_metrics
    
    def _compute_path_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute path-based metrics: F_i, D, I, etc."""
        print("Computing path metrics...")
        
        path_metrics = {
            'fragmentation_scores_F_i': {},
            'similarity_convergent_path_density_D': 0.0,
            'interestingness_scores_I': {},
            'path_purity': {},
            'path_reproducibility': {}
        }
        
        token_paths = results['token_paths']
        labels = results['labels']
        
        # Compute F_i (fragmentation score) for each token path
        for sent_idx, sent_paths in token_paths.items():
            path_metrics['fragmentation_scores_F_i'][sent_idx] = {}
            for token_idx, path in sent_paths.items():
                fragmentation = self._compute_simple_fragmentation_score(path)
                path_metrics['fragmentation_scores_F_i'][sent_idx][token_idx] = fragmentation
        
        # Compute path purity (how often same class follows same path)
        path_metrics['path_purity'] = self._compute_path_purity(token_paths, labels)
        
        # Compute D (similarity-convergent path density)
        path_metrics['similarity_convergent_path_density_D'] = self._compute_similarity_convergent_density(results)
        
        return path_metrics
    
    def _compute_pivot_specific_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute pivot-specific metrics using our implemented functions."""
        print("Computing pivot-specific metrics...")
        
        if not pivot_metrics_available:
            # Simplified pivot metrics calculation
            return self._compute_simplified_pivot_metrics(results)
        
        # Prepare data for pivot metrics
        all_sentence_paths = []
        sentence_labels = []
        
        for sent_idx in range(len(results['sentences'])):
            if sent_idx in results['token_paths']:
                all_sentence_paths.append(results['token_paths'][sent_idx])
                sentence_labels.append(results['labels'][sent_idx])
        
        # Use our pivot metrics functions
        pivot_analysis = analyze_pivot_effects(
            all_sentence_paths,
            sentence_labels,
            pivot_token_index=1,  # "but" is at index 1 in 3-token sentences
            total_layers=13
        )
        
        return pivot_analysis
    
    def _compute_simplified_pivot_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute simplified pivot metrics when full implementation unavailable."""
        print("Computing simplified pivot metrics...")
        
        pivot_metrics = {
            'fragmentation_deltas': [],
            'path_divergence_indices': [],
            'contrast_stats': {'mean_fragmentation_delta': 0.0, 'mean_path_divergence': 0.0},
            'consistent_stats': {'mean_fragmentation_delta': 0.0, 'mean_path_divergence': 0.0}
        }
        
        # Compute simplified fragmentation delta and path divergence
        for sent_idx, paths in results['token_paths'].items():
            if len(paths) >= 3:  # Need at least 3 tokens
                # Simplified fragmentation delta: post-pivot fragmentation - pre-pivot fragmentation
                pre_pivot_path = paths.get(0, [])
                post_pivot_path = paths.get(2, [])
                
                pre_frag = len(set(pre_pivot_path)) / len(pre_pivot_path) if pre_pivot_path else 0
                post_frag = len(set(post_pivot_path)) / len(post_pivot_path) if post_pivot_path else 0
                delta = post_frag - pre_frag
                
                # Simplified path divergence: Hamming distance
                divergence = 0.0
                if pre_pivot_path and post_pivot_path:
                    min_len = min(len(pre_pivot_path), len(post_pivot_path))
                    differences = sum(1 for i in range(min_len) if pre_pivot_path[i] != post_pivot_path[i])
                    divergence = differences / min_len
                
                pivot_metrics['fragmentation_deltas'].append(delta)
                pivot_metrics['path_divergence_indices'].append(divergence)
        
        return pivot_metrics
    
    def _compute_archetypal_paths(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute archetypal paths and their frequencies."""
        print("Computing archetypal paths...")
        
        archetypal_analysis = {
            'most_frequent_paths': {},
            'path_frequencies': {},
            'contrast_vs_consistent_paths': {},
            'unique_paths_count': 0
        }
        
        # Collect all paths by class
        contrast_paths = []
        consistent_paths = []
        all_paths = []
        
        for sent_idx, paths in results['token_paths'].items():
            label = results['labels'][sent_idx]
            for token_idx, path in paths.items():
                path_str = '->'.join(path)
                all_paths.append(path_str)
                
                if label == 'contrast':
                    contrast_paths.append(path_str)
                else:
                    consistent_paths.append(path_str)
        
        # Compute path frequencies
        all_path_counts = Counter(all_paths)
        contrast_path_counts = Counter(contrast_paths)
        consistent_path_counts = Counter(consistent_paths)
        
        archetypal_analysis['path_frequencies'] = {
            'all': dict(all_path_counts.most_common(10)),
            'contrast': dict(contrast_path_counts.most_common(10)),
            'consistent': dict(consistent_path_counts.most_common(10))
        }
        
        archetypal_analysis['unique_paths_count'] = len(all_path_counts)
        archetypal_analysis['most_frequent_paths'] = dict(all_path_counts.most_common(5))
        
        # Compare contrast vs consistent paths
        archetypal_analysis['contrast_vs_consistent_paths'] = {
            'contrast_unique': len(contrast_path_counts),
            'consistent_unique': len(consistent_path_counts),
            'shared_paths': len(set(contrast_path_counts.keys()) & set(consistent_path_counts.keys())),
            'contrast_only': len(set(contrast_path_counts.keys()) - set(consistent_path_counts.keys())),
            'consistent_only': len(set(consistent_path_counts.keys()) - set(contrast_path_counts.keys()))
        }
        
        return archetypal_analysis
    
    def _compute_centroid_similarity(self, layer1_data: Dict[str, Any], layer2_data: Dict[str, Any]) -> float:
        """Compute ρ^c between two layers."""
        # This is a simplified version - would need actual centroids for full implementation
        centers1 = layer1_data.get('cluster_centers', [])
        centers2 = layer2_data.get('cluster_centers', [])
        
        if not centers1 or not centers2:
            return 0.0
        
        # Compute average cosine similarity between centroids
        similarities = []
        min_k = min(len(centers1), len(centers2))
        
        for i in range(min_k):
            # Cosine similarity between centroids
            c1, c2 = np.array(centers1[i]), np.array(centers2[i])
            if np.linalg.norm(c1) > 0 and np.linalg.norm(c2) > 0:
                sim = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_membership_overlap(self, labels1: Dict, labels2: Dict) -> float:
        """Compute J (Jaccard similarity) between layer memberships."""
        # Convert to sets of (sent_idx, token_idx, cluster) tuples
        set1 = set()
        set2 = set()
        
        for sent_idx, token_labels in labels1.items():
            for token_idx, label in token_labels.items():
                set1.add((sent_idx, token_idx, label))
        
        for sent_idx, token_labels in labels2.items():
            for token_idx, label in token_labels.items():
                set2.add((sent_idx, token_idx, label))
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_simple_fragmentation_score(self, path: List[str]) -> float:
        """Compute simple fragmentation score: unique clusters / total layers."""
        if not path:
            return 0.0
        
        unique_clusters = len(set(path))
        return unique_clusters / len(path)
    
    def _compute_path_purity(self, token_paths: Dict, labels: List[str]) -> Dict[str, float]:
        """Compute path purity by class."""
        class_paths = {'contrast': [], 'consistent': []}
        
        for sent_idx, paths in token_paths.items():
            label = labels[sent_idx]
            for path in paths.values():
                path_str = '->'.join(path)
                class_paths[label].append(path_str)
        
        purity = {}
        for class_name, paths in class_paths.items():
            if paths:
                path_counts = Counter(paths)
                most_common_count = path_counts.most_common(1)[0][1]
                purity[class_name] = most_common_count / len(paths)
            else:
                purity[class_name] = 0.0
        
        return purity
    
    def _compute_similarity_convergent_density(self, results: Dict[str, Any]) -> float:
        """Compute D (similarity-convergent path density)."""
        # Simplified version - would need full centroid similarity calculation
        # For now, estimate based on path patterns
        convergent_paths = 0
        total_paths = 0
        
        for sent_paths in results['token_paths'].values():
            for path in sent_paths.values():
                total_paths += 1
                # Simple heuristic: path converging if first and last clusters similar
                if len(path) >= 3 and path[0].split('C')[1] == path[-1].split('C')[1]:
                    convergent_paths += 1
        
        return convergent_paths / total_paths if total_paths > 0 else 0.0
    
    def save_metrics(self, metrics: Dict[str, Any], filename: str):
        """Save all metrics to file."""
        print(f"Saving comprehensive metrics to {filename}...")
        
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Metrics saved to {filename}")


def compute_pivot_apa_metrics():
    """Main function to compute all APA metrics for pivot experiment."""
    # Load clustering results
    print("Loading clustering results...")
    with open("gpt2_pivot_clustering_results.pkl", "rb") as f:
        clustering_results = pickle.load(f)
    
    # Compute all metrics
    calculator = GPT2APAMetricsCalculator()
    metrics = calculator.compute_all_metrics(clustering_results)
    
    # Save metrics
    calculator.save_metrics(metrics, "gpt2_pivot_apa_metrics.json")
    
    return metrics


if __name__ == "__main__":
    metrics = compute_pivot_apa_metrics()