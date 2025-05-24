"""
Centroid-based ETS for explainable micro-clusters within macro clusters
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path
from collections import Counter

# Add parent directories to path for imports
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_similarity_matrix,
    compute_ets_clustering
)

from logging_config import setup_logging

logger = setup_logging(__name__)


class CentroidBasedETS:
    """
    Explainable micro-clustering within macro clusters.
    
    Uses centroid-relative thresholds for interpretability.
    """
    
    def __init__(self,
                 initial_percentile: float = 30.0,
                 min_cluster_size: int = 3,
                 merge_threshold: float = 1.0,
                 coverage_target: float = 0.8,
                 purity_target: float = 0.7):
        """
        Initialize centroid-based ETS.
        
        Args:
            initial_percentile: Starting percentile for thresholds (20-40 recommended)
            min_cluster_size: Minimum micro-cluster size
            merge_threshold: Merge clusters if centroids within this many σ
            coverage_target: Target coverage of macro cluster
            purity_target: Target purity within micro-clusters
        """
        self.initial_percentile = initial_percentile
        self.min_cluster_size = min_cluster_size
        self.merge_threshold = merge_threshold
        self.coverage_target = coverage_target
        self.purity_target = purity_target
        
        logger.info(f"Initialized CentroidBasedETS: percentile={initial_percentile}, "
                   f"min_size={min_cluster_size}, coverage={coverage_target}")
    
    def compute_centroid_thresholds(self, 
                                   points: np.ndarray,
                                   centroid: np.ndarray,
                                   percentile: float) -> np.ndarray:
        """
        Compute per-dimension thresholds relative to centroid.
        
        Args:
            points: Points in the macro cluster
            centroid: Cluster centroid
            percentile: Percentile for threshold calculation
            
        Returns:
            Per-dimension thresholds
        """
        # Calculate distances from centroid
        distances = np.abs(points - centroid)
        
        # Compute percentile per dimension
        thresholds = np.percentile(distances, percentile, axis=0)
        
        # Ensure minimum threshold
        min_threshold = 1e-5
        thresholds = np.maximum(thresholds, min_threshold)
        
        return thresholds
    
    def optimize_alpha(self,
                      points: np.ndarray,
                      centroid: np.ndarray,
                      base_thresholds: np.ndarray,
                      word_subtypes: Optional[List[str]] = None) -> Tuple[float, Dict]:
        """
        Find optimal α scaling factor for coverage-purity trade-off.
        
        Args:
            points: Points in macro cluster
            centroid: Cluster centroid
            base_thresholds: Base thresholds from percentile
            word_subtypes: Optional semantic subtypes for purity calculation
            
        Returns:
            Tuple of (optimal_alpha, metrics_by_alpha)
        """
        alpha_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
        results = {}
        
        for alpha in alpha_values:
            # Scale thresholds
            scaled_thresholds = base_thresholds * alpha
            
            # Compute similarity relative to centroid
            # Points are similar to centroid if within thresholds
            distances = np.abs(points - centroid)
            within_threshold = distances <= scaled_thresholds
            covered = np.all(within_threshold, axis=1)
            
            coverage = np.sum(covered) / len(points)
            
            # If we have subtypes, calculate purity
            if word_subtypes is not None:
                # Get indices of covered points
                covered_indices = np.where(covered)[0]
                if len(covered_indices) > 0:
                    covered_subtypes = [word_subtypes[i] for i in covered_indices]
                    subtype_counts = Counter(covered_subtypes)
                    most_common = subtype_counts.most_common(1)[0][1]
                    purity = most_common / len(covered_indices)
                else:
                    purity = 0.0
            else:
                purity = 1.0  # No subtypes to check
            
            results[alpha] = {
                'coverage': coverage,
                'purity': purity,
                'n_covered': np.sum(covered)
            }
            
            logger.debug(f"  alpha={alpha}: coverage={coverage:.3f}, purity={purity:.3f}")
        
        # Find best alpha
        best_alpha = None
        best_score = -float('inf')
        
        for alpha, metrics in results.items():
            # Must meet coverage requirement
            if metrics['coverage'] >= self.coverage_target:
                # Prefer higher purity
                score = metrics['purity']
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
        
        # If no alpha meets coverage, take the one with highest coverage
        if best_alpha is None:
            best_alpha = max(results.keys(), key=lambda a: results[a]['coverage'])
            logger.warning(f"No alpha meets coverage target, using alpha={best_alpha} "
                          f"with coverage={results[best_alpha]['coverage']:.3f}")
        
        return best_alpha, results
    
    def create_micro_clusters(self,
                            points: np.ndarray,
                            centroid: np.ndarray,
                            macro_indices: np.ndarray,
                            word_subtypes: Optional[List[str]] = None) -> Dict:
        """
        Create micro-clusters within a macro cluster.
        
        Args:
            points: Points in the macro cluster
            centroid: Macro cluster centroid
            macro_indices: Original indices of points in full dataset
            word_subtypes: Optional semantic subtypes
            
        Returns:
            Dictionary with micro-clustering results
        """
        n_points = len(points)
        logger.info(f"Creating micro-clusters for macro cluster with {n_points} points")
        
        # Step 1: Compute base thresholds
        base_thresholds = self.compute_centroid_thresholds(
            points, centroid, self.initial_percentile
        )
        
        # Step 2: Optimize alpha
        if word_subtypes is not None:
            # word_subtypes is already filtered for this cluster
            cluster_subtypes = word_subtypes
        else:
            cluster_subtypes = None
            
        optimal_alpha, alpha_results = self.optimize_alpha(
            points, centroid, base_thresholds, cluster_subtypes
        )
        
        # Step 3: Create micro-clusters with optimal thresholds
        optimal_thresholds = base_thresholds * optimal_alpha
        
        # Use ETS clustering with these thresholds
        # Note: We're clustering relative to centroid, not between all points
        micro_labels, _ = compute_ets_clustering(
            points,
            thresholds=optimal_thresholds,
            threshold_percentile=None,  # We're providing thresholds directly
            verbose=False
        )
        
        # Step 4: Handle small clusters
        unique_labels, counts = np.unique(micro_labels, return_counts=True)
        
        # Mark small clusters for anomaly bucket
        anomaly_mask = np.zeros(n_points, dtype=bool)
        for label, count in zip(unique_labels, counts):
            if count < self.min_cluster_size:
                label_mask = micro_labels == label
                anomaly_mask |= label_mask
                micro_labels[label_mask] = -1  # Mark as anomaly
        
        # Step 5: Merge similar clusters
        micro_labels = self.merge_similar_clusters(
            points, micro_labels, self.merge_threshold
        )
        
        # Step 6: Calculate final metrics
        final_unique, final_counts = np.unique(micro_labels, return_counts=True)
        n_micro = len(final_unique[final_unique >= 0])  # Exclude anomalies
        n_anomalies = np.sum(micro_labels == -1)
        
        # Coverage (non-anomaly points)
        coverage = (n_points - n_anomalies) / n_points
        
        # Purity if we have subtypes
        if cluster_subtypes is not None:
            total_purity = 0
            total_weight = 0
            
            for label in final_unique[final_unique >= 0]:
                mask = micro_labels == label
                micro_subtypes = [cluster_subtypes[i] for i in range(n_points) if mask[i]]
                if micro_subtypes:
                    subtype_counts = Counter(micro_subtypes)
                    most_common = subtype_counts.most_common(1)[0][1]
                    purity = most_common / len(micro_subtypes)
                    total_purity += purity * len(micro_subtypes)
                    total_weight += len(micro_subtypes)
            
            avg_purity = total_purity / total_weight if total_weight > 0 else 0
        else:
            avg_purity = 1.0
        
        results = {
            'micro_labels': micro_labels,
            'thresholds': optimal_thresholds,
            'alpha': optimal_alpha,
            'n_micro_clusters': n_micro,
            'n_anomalies': n_anomalies,
            'coverage': coverage,
            'purity': avg_purity,
            'micro_sizes': final_counts.tolist(),
            'alpha_search_results': alpha_results
        }
        
        logger.info(f"  Created {n_micro} micro-clusters, {n_anomalies} anomalies, "
                   f"coverage={coverage:.3f}, purity={avg_purity:.3f}")
        
        return results
    
    def merge_similar_clusters(self,
                              points: np.ndarray,
                              labels: np.ndarray,
                              threshold_sigma: float) -> np.ndarray:
        """
        Merge micro-clusters with centroids within threshold_sigma standard deviations.
        
        Args:
            points: Data points
            labels: Current cluster labels
            threshold_sigma: Merge if centroids within this many σ
            
        Returns:
            Updated labels after merging
        """
        unique_labels = np.unique(labels[labels >= 0])  # Exclude anomalies
        
        if len(unique_labels) <= 1:
            return labels
        
        # Calculate centroids
        centroids = {}
        for label in unique_labels:
            mask = labels == label
            centroids[label] = points[mask].mean(axis=0)
        
        # Calculate overall std per dimension
        std_per_dim = points.std(axis=0)
        
        # Find pairs to merge
        merged_labels = labels.copy()
        label_mapping = {label: label for label in unique_labels}
        
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                # Skip if already merged
                if label_mapping[label1] != label1 or label_mapping[label2] != label2:
                    continue
                    
                # Check distance between centroids
                cent1 = centroids[label1]
                cent2 = centroids[label2]
                
                # Normalized distance
                normalized_dist = np.abs(cent1 - cent2) / (std_per_dim + 1e-10)
                mean_normalized_dist = np.mean(normalized_dist)
                
                if mean_normalized_dist <= threshold_sigma:
                    # Merge label2 into label1
                    label_mapping[label2] = label1
                    merged_labels[labels == label2] = label1
                    
                    logger.debug(f"    Merged micro-cluster {label2} into {label1} "
                                f"(distance={mean_normalized_dist:.3f}σ)")
        
        return merged_labels
    
    def fit_transform(self,
                     preprocessed_data: Dict[str, np.ndarray],
                     macro_clusters: Dict[str, np.ndarray],
                     word_subtypes: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Create micro-clusters for all macro clusters across all layers.
        
        Args:
            preprocessed_data: Preprocessed activations by layer
            macro_clusters: Macro cluster assignments by layer
            word_subtypes: Optional semantic subtypes
            
        Returns:
            Micro-clustering results by layer and macro cluster
        """
        all_results = {}
        
        for layer_name, data in preprocessed_data.items():
            logger.info(f"\nProcessing {layer_name}")
            
            macro_labels = macro_clusters[layer_name]
            unique_macros = np.unique(macro_labels)
            
            layer_results = {}
            
            for macro_id in unique_macros:
                # Get points in this macro cluster
                macro_mask = macro_labels == macro_id
                macro_indices = np.where(macro_mask)[0]
                macro_points = data[macro_mask]
                
                # Calculate macro centroid
                centroid = macro_points.mean(axis=0)
                
                # Create micro-clusters
                micro_results = self.create_micro_clusters(
                    macro_points, centroid, macro_indices, word_subtypes
                )
                
                # Add macro cluster info
                micro_results['macro_id'] = macro_id
                micro_results['macro_size'] = len(macro_points)
                micro_results['macro_indices'] = macro_indices
                micro_results['centroid'] = centroid
                
                layer_results[f'macro_{macro_id}'] = micro_results
            
            all_results[layer_name] = layer_results
        
        return all_results