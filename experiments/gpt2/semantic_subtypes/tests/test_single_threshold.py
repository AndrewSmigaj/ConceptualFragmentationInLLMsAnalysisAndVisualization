#!/usr/bin/env python3
"""
Test a single ETS threshold to debug performance.
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

def main():
    """Test a single threshold."""
    
    print("Loading activations...")
    
    # Load saved activations
    activations_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl")
    with open(activations_path, 'rb') as f:
        activations_data = pickle.load(f)
    
    activations = activations_data['activations']
    
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
    
    # Test a single threshold
    threshold_percentile = 0.99
    
    print(f"\nTesting threshold_percentile = {threshold_percentile}")
    print("Computing dimension thresholds...")
    start_time = time.time()
    
    # Step 1: Compute dimension thresholds
    dimension_thresholds = compute_dimension_thresholds(
        layer_activations,
        threshold_percentile=threshold_percentile,
        min_threshold=1e-5
    )
    
    print(f"Dimension thresholds computed in {time.time() - start_time:.2f}s")
    print(f"Threshold stats: mean={dimension_thresholds.mean():.5f}, min={dimension_thresholds.min():.5f}, max={dimension_thresholds.max():.5f}")
    
    # Step 2: Compute similarity matrix
    print("\nComputing similarity matrix...")
    sim_start = time.time()
    
    similarity_matrix = compute_similarity_matrix(
        layer_activations,
        dimension_thresholds,
        batch_size=100,  # Small batch size
        verbose=True
    )
    
    print(f"Similarity matrix computed in {time.time() - sim_start:.2f}s")
    
    # Step 3: Find connected components
    print("\nFinding connected components...")
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    similarity_sparse = csr_matrix(similarity_matrix)
    n_clusters, cluster_labels = connected_components(
        similarity_sparse, directed=False, return_labels=True
    )
    
    print(f"\nFound {n_clusters} clusters")
    print(f"Total time: {time.time() - start_time:.2f}s")
    
    # Show cluster sizes
    cluster_counts = {}
    for label in cluster_labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1
    print(f"Cluster sizes: {sorted(cluster_counts.values(), reverse=True)[:10]}")

if __name__ == "__main__":
    main()