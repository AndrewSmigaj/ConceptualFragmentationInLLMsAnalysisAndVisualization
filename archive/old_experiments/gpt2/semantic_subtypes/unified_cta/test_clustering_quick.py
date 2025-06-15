#!/usr/bin/env python3
"""Quick test to verify clustering produces multiple clusters."""

import sys
from pathlib import Path
import numpy as np

# Add unified_cta directory
unified_cta_dir = Path(__file__).parent
sys.path.insert(0, str(unified_cta_dir))

print("Testing clustering functionality...")
print("=" * 60)

try:
    from clustering.structural import StructuralClusterer
    
    # Create test data - 3 clear clusters
    np.random.seed(42)
    cluster1 = np.random.randn(50, 10) + np.array([0, 0, 0, 0, 0, 5, 5, 5, 5, 5])
    cluster2 = np.random.randn(50, 10) + np.array([5, 5, 5, 5, 5, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(50, 10) + np.array([2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5])
    data = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"Test data shape: {data.shape}")
    
    # Test gap statistic
    clusterer = StructuralClusterer()
    optimal_k, gap_results = clusterer.find_optimal_k_gap(data, k_range=(2, 6))
    
    print(f"\nGap statistic results:")
    for result in gap_results:
        print(f"  k={result['k']}: gap={result['gap']:.3f}")
    print(f"\nOptimal k: {optimal_k}")
    
    # Perform clustering
    labels = clusterer.cluster_with_k(data, optimal_k)
    unique_labels = np.unique(labels)
    print(f"Unique clusters: {len(unique_labels)}")
    
    # Check cluster sizes
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"  Cluster {label}: {count} points")
    
    if len(unique_labels) > 1:
        print("\nSUCCESS: Multiple clusters found!")
    else:
        print("\nFAILED: Only one cluster found")
        
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()