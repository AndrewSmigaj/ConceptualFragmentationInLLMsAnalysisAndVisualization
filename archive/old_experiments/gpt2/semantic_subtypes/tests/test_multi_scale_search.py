#!/usr/bin/env python3
"""Test the multi_scale_ets_search function."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ets_revised_wrapper import ETSPreprocessor, multi_scale_ets_search

def test_multi_scale_search():
    """Test multi-scale ETS search with synthetic data."""
    print("Testing multi_scale_ets_search...")
    
    # Create synthetic data with clear clusters
    np.random.seed(42)
    
    # 3 clusters of 50 points each
    cluster1 = np.random.randn(50, 100) + np.array([0] * 100)
    cluster2 = np.random.randn(50, 100) + np.array([5] * 100)
    cluster3 = np.random.randn(50, 100) + np.array([10] * 100)
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    # Preprocess
    preprocessor = ETSPreprocessor(pca_dims=50)
    preprocessed = preprocessor.fit_transform(data)
    
    print(f"Data shape: {preprocessed.shape}")
    
    # Run multi-scale search
    print("\nRunning multi-scale search...")
    results = multi_scale_ets_search(preprocessed, percentiles=[70, 80, 90, 95])
    
    # Check results
    print("\nResults summary:")
    for p, res in sorted(results.items()):
        print(f"Percentile {p}%: {res['n_clusters']} clusters, "
              f"silhouette={res['silhouette']:.3f}, "
              f"sizes={res['cluster_sizes']}")
    
    # Verify we get different numbers of clusters
    n_clusters_set = set(res['n_clusters'] for res in results.values())
    print(f"\nUnique cluster counts: {sorted(n_clusters_set)}")
    
    # Should have at least 2 different cluster counts
    assert len(n_clusters_set) >= 2, "Should get different cluster counts for different percentiles"
    
    print("\nTest passed!")

if __name__ == "__main__":
    test_multi_scale_search()