#!/usr/bin/env python3
"""
Wrapper for revised ETS implementation following the comprehensive recipe.
This uses the existing ETS functions but adds proper preprocessing and threshold search.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import pickle
from datetime import datetime
import sys

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Add parent directory to path for imports
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_dimension_thresholds,
    compute_similarity_matrix
)


class ETSPreprocessor:
    """Handle preprocessing for ETS clustering."""
    
    def __init__(self, pca_dims: int = 128):
        self.pca_dims = pca_dims
        self.scaler = StandardScaler()
        self.pca = None
        
    def fit_transform(self, activations: np.ndarray) -> np.ndarray:
        """
        Apply standardization and PCA.
        
        Args:
            activations: Raw activations (n_samples, n_features)
            
        Returns:
            Preprocessed activations (n_samples, pca_dims)
        """
        # Step 1: Standardize
        standardized = self.scaler.fit_transform(activations)
        
        # Step 2: PCA if needed
        n_samples, n_features = standardized.shape
        
        if n_features > self.pca_dims:
            self.pca = PCA(n_components=self.pca_dims)
            reduced = self.pca.fit_transform(standardized)
            print(f"PCA: {n_features} -> {self.pca_dims} dims, "
                  f"explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
            return reduced
        else:
            print(f"No PCA needed: {n_features} features <= {self.pca_dims} target dims")
            return standardized


def multi_scale_ets_search(activations: np.ndarray, 
                          percentiles: List[float] = None,
                          verbose: bool = True) -> Dict[float, Dict]:
    """
    Strategy 1: Multi-scale threshold search using existing ETS functions.
    
    Args:
        activations: Preprocessed activations
        percentiles: List of percentiles to try
        verbose: Print progress
        
    Returns:
        Dictionary of results for each percentile
    """
    if percentiles is None:
        # Try a wider range including very high percentiles
        percentiles = [85, 90, 92, 94, 95, 96, 97, 98, 99]
    
    results = {}
    
    for p in percentiles:
        # Convert percentile to 0-1 scale for the ETS function
        threshold_percentile = p / 100.0
        
        if verbose:
            print(f"\nTesting percentile {p}% (threshold_percentile={threshold_percentile:.3f})")
            
        # First compute thresholds to see what we're getting
        thresholds = compute_dimension_thresholds(activations, threshold_percentile)
        
        if verbose:
            print(f"  Threshold stats: mean={thresholds.mean():.4f}, std={thresholds.std():.4f}, "
                  f"min={thresholds.min():.4f}, max={thresholds.max():.4f}")
        
        # Use existing ETS clustering
        cluster_labels, thresholds = compute_ets_clustering(
            activations,
            threshold_percentile=threshold_percentile,
            verbose=False
        )
        
        n_clusters = len(np.unique(cluster_labels))
        
        # Calculate metrics
        if n_clusters > 1 and n_clusters < len(activations):
            silhouette = silhouette_score(activations, cluster_labels)
        else:
            silhouette = -1.0
        
        # Cluster size statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        size_std = np.std(counts)
        size_ratio = counts.max() / counts.min() if n_clusters > 1 else float('inf')
        
        results[p] = {
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'size_std': size_std,
            'size_ratio': size_ratio,
            'cluster_sizes': counts.tolist(),
            'threshold_percentile': threshold_percentile,
            'thresholds_mean': thresholds.mean(),
            'thresholds_std': thresholds.std()
        }
        
        if verbose:
            print(f"  -> {n_clusters} clusters, silhouette={silhouette:.3f}, "
                  f"sizes={counts.tolist()}")
    
    return results


def find_best_percentile(results: Dict[float, Dict], 
                        min_clusters: int = 2,
                        max_clusters: int = 10) -> float:
    """
    Find best percentile based on silhouette score and cluster balance.
    
    Args:
        results: Results from multi_scale_ets_search
        min_clusters: Minimum acceptable number of clusters
        max_clusters: Maximum acceptable number of clusters
        
    Returns:
        Best percentile value
    """
    best_percentile = None
    best_score = -float('inf')
    
    for p, res in results.items():
        n_clusters = res['n_clusters']
        
        # Skip if outside acceptable range
        if n_clusters < min_clusters or n_clusters > max_clusters:
            continue
        
        # Combine silhouette with size balance penalty
        silhouette = res['silhouette']
        size_ratio = res['size_ratio']
        
        # Penalize very imbalanced clusters
        balance_penalty = np.log(size_ratio) if size_ratio > 1 else 0
        score = silhouette - 0.1 * balance_penalty
        
        if score > best_score:
            best_score = score
            best_percentile = p
    
    # Fallback if no good clustering found
    if best_percentile is None:
        # Try to find one with reasonable number of clusters
        for p in [85, 80, 75, 70]:
            if p in results and min_clusters <= results[p]['n_clusters'] <= max_clusters:
                best_percentile = p
                break
    
    return best_percentile or 85  # Default fallback


def find_percentile_for_target_k(activations: np.ndarray, 
                                target_k: int,
                                search_range: Tuple[float, float] = (85, 99.5),
                                max_iterations: int = 20,
                                verbose: bool = True) -> Tuple[float, Dict]:
    """
    Binary search for percentile that gives target k clusters.
    
    Args:
        activations: Preprocessed activations
        target_k: Target number of clusters
        search_range: Percentile range to search
        max_iterations: Maximum binary search iterations
        verbose: Print progress
        
    Returns:
        Tuple of (best_percentile, clustering_result)
    """
    low, high = search_range
    best_percentile = None
    best_result = None
    best_diff = float('inf')
    
    for iteration in range(max_iterations):
        mid = (low + high) / 2
        
        # Test this percentile
        threshold_percentile = mid / 100.0
        
        cluster_labels, thresholds = compute_ets_clustering(
            activations,
            threshold_percentile=threshold_percentile,
            verbose=False
        )
        
        n_clusters = len(np.unique(cluster_labels))
        
        if verbose:
            print(f"  Iteration {iteration}: percentile={mid:.1f}%, "
                  f"clusters={n_clusters} (target={target_k})")
        
        # Update best if closer to target
        diff = abs(n_clusters - target_k)
        if diff < best_diff:
            best_diff = diff
            best_percentile = mid
            best_result = {
                'labels': cluster_labels,
                'thresholds': thresholds,
                'n_clusters': n_clusters
            }
        
        # Check if exact match
        if n_clusters == target_k:
            break
        
        # Binary search update
        if n_clusters > target_k:
            # Too many clusters, need higher percentile (looser threshold)
            low = mid
        else:
            # Too few clusters, need lower percentile (tighter threshold)
            high = mid
        
        # Stop if range is too small
        if high - low < 0.5:
            break
    
    return best_percentile, best_result


def run_revised_ets_experiment(test_mode=False):
    """Run the revised ETS experiment with proper preprocessing and threshold search."""
    
    print("="*60)
    print("REVISED ETS EXPERIMENT WITH PREPROCESSING")
    print("="*60)
    
    # Use the reconstructed activations file
    activations_file = Path("activations_by_layer.pkl")
    
    if not activations_file.exists():
        print("Error: activations_by_layer.pkl not found!")
        print("Please run reconstruct_activations.py first")
        return
    
    print(f"\nUsing reconstructed activations from: {activations_file}")
    
    # Load data
    with open(activations_file, 'rb') as f:
        activations_data = pickle.load(f)
    
    # Load optimal k values
    if Path("layer_clustering_config.json").exists():
        with open("layer_clustering_config.json", 'r') as f:
            layer_config = json.load(f)
    else:
        # Fallback to default k=3
        print("Warning: layer_clustering_config.json not found, using k=3 for all layers")
        layer_config = {f"layer_{i}": {"optimal_k": 3} for i in range(13)}
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"semantic_subtypes_revised_ets_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = ETSPreprocessor(pca_dims=128)
    
    # Results storage
    all_results = {}
    
    # Process each layer
    max_layers = 1 if test_mode else 13
    for layer_idx in range(max_layers):
        print(f"\n{'='*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*60}")
        
        layer_key = f'layer_{layer_idx}'
        activations = activations_data[layer_key]['activations']
        
        # Preprocess
        print(f"\nPreprocessing {activations.shape[0]} samples, {activations.shape[1]} features...")
        preprocessed = preprocessor.fit_transform(activations)
        
        # Get target k
        target_k = layer_config.get(f'layer_{layer_idx}', {}).get('optimal_k', 3)
        print(f"\nTarget k from K-means elbow analysis: {target_k}")
        
        # Strategy 1: Multi-scale search
        print("\nRunning multi-scale ETS search...")
        multi_results = multi_scale_ets_search(preprocessed, verbose=True)
        
        # Find best percentile
        best_p = find_best_percentile(multi_results)
        print(f"\nBest percentile from multi-scale search: {best_p}%")
        
        # Strategy 2: Target k search
        print(f"\nFinding percentile for target k={target_k}...")
        target_p, target_result = find_percentile_for_target_k(
            preprocessed, target_k, verbose=True
        )
        
        # Use target k result if it found close to the right number
        if target_result and abs(target_result['n_clusters'] - target_k) <= 1:
            print(f"\nUsing target k result: {target_p}% -> {target_result['n_clusters']} clusters (target was {target_k})")
            final_labels = target_result['labels']
            final_percentile = target_p
            final_thresholds = target_result['thresholds']
        else:
            print(f"\nUsing multi-scale best: {best_p}% -> {multi_results[best_p]['n_clusters']} clusters")
            # Re-run with best percentile
            final_labels, final_thresholds = compute_ets_clustering(
                preprocessed,
                threshold_percentile=best_p/100.0,
                verbose=False
            )
            final_percentile = best_p
        
        # Calculate final metrics
        n_clusters = len(np.unique(final_labels))
        if n_clusters > 1:
            silhouette = silhouette_score(preprocessed, final_labels)
        else:
            silhouette = -1.0
        
        # Store results
        all_results[layer_key] = {
            'target_k': target_k,
            'final_percentile': final_percentile,
            'final_k': n_clusters,
            'silhouette': silhouette,
            'labels': final_labels.tolist(),
            'preprocessing': {
                'original_shape': activations.shape,
                'preprocessed_shape': preprocessed.shape,
                'pca_variance_explained': preprocessor.pca.explained_variance_ratio_.sum() if preprocessor.pca else 1.0
            },
            'multi_scale_results': multi_results
        }
        
        print(f"\nFinal: {n_clusters} clusters, silhouette={silhouette:.3f}")
    
    # Save results
    with open(output_dir / "revised_ets_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n\nResults saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Layer':<6} {'Target k':<10} {'ETS k':<8} {'Percentile':<12} {'Silhouette':<10}")
    print("-"*60)
    
    for layer_idx in range(max_layers):
        layer_key = f'layer_{layer_idx}'
        res = all_results[layer_key]
        print(f"{layer_idx:<6} {res['target_k']:<10} {res['final_k']:<8} "
              f"{res['final_percentile']:<12.1f} {res['silhouette']:<10.3f}")


if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    run_revised_ets_experiment(test_mode=test_mode)