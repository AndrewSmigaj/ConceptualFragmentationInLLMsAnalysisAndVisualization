#!/usr/bin/env python3
"""
Test multiple ETS thresholds to find ones that give 2-4 clusters.
"""

import pickle
import numpy as np
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_dimension_thresholds,
    compute_similarity_matrix,
    compute_ets_clustering
)

def test_threshold(layer_activations, threshold_percentile, sentences=None):
    """Test a single threshold and return cluster count."""
    print(f"\nTesting threshold_percentile = {threshold_percentile}")
    start_time = time.time()
    
    try:
        cluster_labels, dimension_thresholds = compute_ets_clustering(
            layer_activations,
            threshold_percentile=threshold_percentile,
            min_threshold=1e-5,
            verbose=False,
            batch_size=100
        )
        
        # Count unique clusters
        unique_clusters = len(set(cluster_labels))
        elapsed = time.time() - start_time
        
        # Show cluster sizes and identify singleton clusters
        cluster_counts = {}
        cluster_members = {}
        for idx, label in enumerate(cluster_labels):
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
            if label not in cluster_members:
                cluster_members[label] = []
            cluster_members[label].append(idx)
        
        largest_clusters = sorted(cluster_counts.values(), reverse=True)[:5]
        
        print(f"  -> {unique_clusters} clusters (computed in {elapsed:.2f}s)")
        print(f"  -> Largest cluster sizes: {largest_clusters}")
        
        # Show singleton clusters
        if sentences:
            for label, count in cluster_counts.items():
                if count == 1:
                    word_idx = cluster_members[label][0]
                    word = sentences[word_idx] if word_idx < len(sentences) else f"index_{word_idx}"
                    print(f"  -> Singleton cluster: '{word}'")
        
        return unique_clusters, elapsed
        
    except Exception as e:
        print(f"  -> Error: {e}")
        return None, None

def main():
    """Test multiple thresholds."""
    
    print("Loading activations...")
    
    # Load saved activations
    activations_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl")
    with open(activations_path, 'rb') as f:
        activations_data = pickle.load(f)
    
    activations = activations_data['activations']
    sentences = activations_data['sentences']
    
    # Extract layer 5 activations
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
    print("="*50)
    
    # Test thresholds between 0.995 and 0.998 for better balance
    test_thresholds = [0.995, 0.996, 0.997, 0.9975, 0.998]
    
    results = []
    for threshold in test_thresholds:
        n_clusters, elapsed = test_threshold(layer_activations, threshold, sentences)
        if n_clusters is not None:
            results.append({
                'threshold': threshold,
                'n_clusters': n_clusters,
                'time': elapsed
            })
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print(f"{'Threshold':<12} {'Clusters':<10} {'Time (s)':<10}")
    print("-" * 32)
    for r in results:
        print(f"{r['threshold']:<12.4f} {r['n_clusters']:<10} {r['time']:<10.2f}")
    
    # Find thresholds that give 2-4 clusters
    good_thresholds = [r for r in results if 2 <= r['n_clusters'] <= 4]
    if good_thresholds:
        print(f"\nThresholds giving 2-4 clusters:")
        for r in good_thresholds:
            print(f"  {r['threshold']:.4f} -> {r['n_clusters']} clusters")
    else:
        print("\nNo thresholds found that give 2-4 clusters")
        print("You may need to test higher thresholds (closer to 1.0)")

if __name__ == "__main__":
    main()