#!/usr/bin/env python3
"""
Find optimal clusters per layer using K-means elbow method,
then find ETS thresholds that match those cluster counts.
"""

import pickle
import numpy as np
from pathlib import Path
import sys
import time
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_dimension_thresholds,
    compute_similarity_matrix,
    compute_ets_clustering
)

def find_kmeans_elbow(activations, k_range=(2, 10), verbose=True):
    """Find optimal k using elbow method."""
    
    inertias = []
    silhouettes = []
    k_values = list(range(k_range[0], min(k_range[1] + 1, len(activations))))
    
    if verbose:
        print(f"  Testing k values: {k_values}")
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activations)
        
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        if k > 1 and k < len(activations):
            silhouette = silhouette_score(activations, labels)
            silhouettes.append(silhouette)
        else:
            silhouettes.append(0)
    
    # Find elbow point
    if len(k_values) < 3:
        optimal_k = k_values[0]
    else:
        # Normalize inertias
        inertias_norm = np.array(inertias)
        inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min() + 1e-10)
        
        # Calculate second derivative
        second_derivative = []
        for i in range(1, len(inertias_norm) - 1):
            d2 = inertias_norm[i+1] - 2*inertias_norm[i] + inertias_norm[i-1]
            second_derivative.append(d2)
        
        # Find the point with maximum second derivative
        if second_derivative:
            elbow_idx = np.argmax(second_derivative) + 1
            optimal_k = k_values[elbow_idx]
        else:
            optimal_k = k_values[0]
    
    # Get silhouette for optimal k
    optimal_silhouette = silhouettes[k_values.index(optimal_k)]
    
    if verbose:
        print(f"  Elbow method suggests k={optimal_k} (silhouette={optimal_silhouette:.3f})")
    
    return optimal_k, optimal_silhouette, k_values, inertias, silhouettes

def find_ets_threshold_for_k(activations, target_k, threshold_range=(0.99, 0.999), max_iterations=20):
    """Binary search to find ETS threshold that gives target_k clusters."""
    
    print(f"  Finding ETS threshold for {target_k} clusters...")
    
    left, right = threshold_range[0], threshold_range[1]
    best_threshold = None
    best_n_clusters = None
    best_diff = float('inf')
    
    iteration = 0
    while iteration < max_iterations and left < right:
        mid = (left + right) / 2
        
        try:
            cluster_labels, _ = compute_ets_clustering(
                activations,
                threshold_percentile=mid,
                min_threshold=1e-5,
                verbose=False,
                batch_size=100
            )
            
            n_clusters = len(set(cluster_labels))
            
            # Update best if closer to target
            diff = abs(n_clusters - target_k)
            if diff < best_diff:
                best_diff = diff
                best_threshold = mid
                best_n_clusters = n_clusters
            
            # Perfect match
            if n_clusters == target_k:
                print(f"  Found exact match: threshold={mid:.6f} -> {n_clusters} clusters")
                return mid, n_clusters
            
            # Adjust search range
            if n_clusters > target_k:
                # Too many clusters, need higher threshold
                left = mid + 0.0001
            else:
                # Too few clusters, need lower threshold
                right = mid - 0.0001
                
        except Exception as e:
            print(f"  Error at threshold {mid}: {e}")
            break
        
        iteration += 1
    
    print(f"  Best match: threshold={best_threshold:.6f} -> {best_n_clusters} clusters (target was {target_k})")
    return best_threshold, best_n_clusters

def main():
    """Find optimal clusters per layer."""
    
    print("Loading activations...")
    
    # Load saved activations
    activations_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl")
    with open(activations_path, 'rb') as f:
        activations_data = pickle.load(f)
    
    activations = activations_data['activations']
    sentences = activations_data['sentences']
    
    print(f"Loaded {len(sentences)} words")
    print("="*80)
    
    # Store results
    results = {
        'layer_configs': {},
        'summary': {
            'total_words': len(sentences),
            'num_layers': 13
        }
    }
    
    # Process each layer
    for layer_idx in range(13):  # GPT-2 has 13 layers (0-12)
        print(f"\nLayer {layer_idx}:")
        print("-" * 40)
        
        # Extract layer activations
        layer_activations = []
        
        for sent_idx in sorted(activations.keys()):
            if isinstance(sent_idx, int):
                sent_data = activations[sent_idx]
                if 0 in sent_data:  # Token position 0
                    token_data = sent_data[0]
                    if layer_idx in token_data:
                        layer_activations.append(token_data[layer_idx])
        
        layer_activations = np.array(layer_activations)
        print(f"  Activation shape: {layer_activations.shape}")
        
        # Find optimal k using elbow method
        optimal_k, optimal_silhouette, k_values, inertias, silhouettes = find_kmeans_elbow(
            layer_activations, k_range=(2, 8)
        )
        
        # Find ETS threshold for same number of clusters
        ets_threshold, ets_n_clusters = find_ets_threshold_for_k(
            layer_activations, optimal_k, threshold_range=(0.98, 0.999)
        )
        
        # Store results
        results['layer_configs'][f'layer_{layer_idx}'] = {
            'optimal_k': optimal_k,
            'kmeans_silhouette': optimal_silhouette,
            'ets_threshold': ets_threshold,
            'ets_n_clusters': ets_n_clusters,
            'elbow_data': {
                'k_values': [int(k) for k in k_values],
                'inertias': [float(i) for i in inertias],
                'silhouettes': [float(s) for s in silhouettes]
            }
        }
    
    # Summary statistics
    all_ks = [config['optimal_k'] for config in results['layer_configs'].values()]
    results['summary']['k_distribution'] = {
        'min': min(all_ks),
        'max': max(all_ks),
        'mean': np.mean(all_ks),
        'mode': max(set(all_ks), key=all_ks.count)
    }
    
    # Save results
    output_file = "optimal_clusters_per_layer.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nOptimal k per layer (K-means elbow method):")
    for layer_idx in range(13):
        config = results['layer_configs'][f'layer_{layer_idx}']
        print(f"  Layer {layer_idx:2d}: k={config['optimal_k']} "
              f"(silhouette={config['kmeans_silhouette']:.3f}), "
              f"ETS threshold={config['ets_threshold']:.6f} -> {config['ets_n_clusters']} clusters")
    
    print(f"\nK distribution:")
    print(f"  Range: {results['summary']['k_distribution']['min']} - {results['summary']['k_distribution']['max']}")
    print(f"  Mean: {results['summary']['k_distribution']['mean']:.1f}")
    print(f"  Mode: {results['summary']['k_distribution']['mode']}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()