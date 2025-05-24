#!/usr/bin/env python3
"""
Compare elbow method for K-means with ETS threshold search.
"""

import pickle
import numpy as np
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_dimension_thresholds,
    compute_similarity_matrix,
    compute_ets_clustering
)

def test_kmeans_elbow(activations, k_range=(2, 15)):
    """Test K-means with different k values for elbow method."""
    
    print("\nTesting K-means elbow method...")
    print(f"K range: {k_range[0]} to {k_range[1]}")
    
    inertias = []
    silhouettes = []
    k_values = list(range(k_range[0], k_range[1] + 1))
    
    for k in k_values:
        print(f"  Testing k={k}...", end='', flush=True)
        start_time = time.time()
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activations)
        
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        if k > 1 and k < len(activations):
            silhouette = silhouette_score(activations, labels)
            silhouettes.append(silhouette)
        else:
            silhouettes.append(0)
        
        elapsed = time.time() - start_time
        print(f" inertia={inertia:.2f}, silhouette={silhouettes[-1]:.3f} ({elapsed:.2f}s)")
    
    return k_values, inertias, silhouettes

def find_elbow_point(k_values, inertias):
    """Find elbow point using the elbow method."""
    # Calculate the rate of change
    if len(k_values) < 3:
        return k_values[0]
    
    # Normalize inertias
    inertias_norm = np.array(inertias)
    inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min())
    
    # Calculate second derivative
    second_derivative = []
    for i in range(1, len(inertias_norm) - 1):
        d2 = inertias_norm[i+1] - 2*inertias_norm[i] + inertias_norm[i-1]
        second_derivative.append(d2)
    
    # Find the point with maximum second derivative
    elbow_idx = np.argmax(second_derivative) + 1
    return k_values[elbow_idx]

def test_ets_clusters(activations, threshold_range=(0.99, 0.999), num_tests=10):
    """Test ETS with different thresholds."""
    
    print("\nTesting ETS threshold range...")
    print(f"Threshold range: {threshold_range[0]} to {threshold_range[1]}")
    
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_tests)
    results = []
    
    for threshold in thresholds:
        print(f"  Testing threshold={threshold:.4f}...", end='', flush=True)
        start_time = time.time()
        
        try:
            cluster_labels, _ = compute_ets_clustering(
                activations,
                threshold_percentile=threshold,
                min_threshold=1e-5,
                verbose=False,
                batch_size=100
            )
            
            n_clusters = len(set(cluster_labels))
            
            # Calculate silhouette if valid
            if n_clusters > 1 and n_clusters < len(activations):
                silhouette = silhouette_score(activations, cluster_labels)
            else:
                silhouette = 0
            
            elapsed = time.time() - start_time
            print(f" {n_clusters} clusters, silhouette={silhouette:.3f} ({elapsed:.2f}s)")
            
            results.append({
                'threshold': threshold,
                'n_clusters': n_clusters,
                'silhouette': silhouette
            })
            
        except Exception as e:
            print(f" Error: {e}")
    
    return results

def main():
    """Compare elbow methods."""
    
    print("Loading activations...")
    
    # Load saved activations
    activations_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl")
    with open(activations_path, 'rb') as f:
        activations_data = pickle.load(f)
    
    activations = activations_data['activations']
    
    # Test on layer 5
    test_layer = 5
    layer_activations = []
    
    for sent_idx in sorted(activations.keys()):
        if isinstance(sent_idx, int):
            sent_data = activations[sent_idx]
            if 0 in sent_data:  # Token position 0
                token_data = sent_data[0]
                if test_layer in token_data:
                    layer_activations.append(token_data[test_layer])
    
    layer_activations = np.array(layer_activations)
    
    print(f"\nTesting on Layer {test_layer}")
    print(f"Activation shape: {layer_activations.shape}")
    print("="*60)
    
    # Test K-means elbow
    k_values, inertias, silhouettes = test_kmeans_elbow(layer_activations, k_range=(2, 10))
    
    # Find elbow point
    elbow_k = find_elbow_point(k_values, inertias)
    print(f"\nK-means elbow point: k={elbow_k}")
    
    # Find best k by silhouette
    best_k_idx = np.argmax(silhouettes)
    best_k = k_values[best_k_idx]
    print(f"K-means best silhouette: k={best_k} (score={silhouettes[best_k_idx]:.3f})")
    
    # Test ETS
    ets_results = test_ets_clusters(layer_activations, threshold_range=(0.995, 0.998), num_tests=6)
    
    # Find ETS threshold that gives 3 clusters
    threshold_for_3 = None
    for result in ets_results:
        if result['n_clusters'] == 3:
            threshold_for_3 = result['threshold']
            print(f"\nETS threshold for 3 clusters: {threshold_for_3:.4f}")
            break
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\nK-means:")
    print(f"  Elbow point: k={elbow_k}")
    print(f"  Best silhouette: k={best_k} (score={silhouettes[best_k_idx]:.3f})")
    print(f"  For k=3: silhouette={silhouettes[3-k_values[0]]:.3f}")
    
    print("\nETS:")
    if threshold_for_3:
        ets_3 = next(r for r in ets_results if r['threshold'] == threshold_for_3)
        print(f"  Threshold for 3 clusters: {threshold_for_3:.4f}")
        print(f"  Silhouette at 3 clusters: {ets_3['silhouette']:.3f}")
    
    # Save plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, 'bo-')
    plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Elbow: k={elbow_k}')
    plt.axvline(x=3, color='g', linestyle='--', label='k=3')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('K-means Elbow Method')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, 'ro-')
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best: k={best_k}')
    plt.axvline(x=3, color='g', linestyle='--', label='k=3')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('K-means Silhouette Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_elbow_analysis.png', dpi=150)
    print("\nPlot saved to: kmeans_elbow_analysis.png")

if __name__ == "__main__":
    main()