#!/usr/bin/env python3
"""Test ETS with structured data that has natural clusters."""

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def test_ets_with_clusters():
    """Test ETS with data that has clear clusters."""
    try:
        from gpt2_pivot_clusterer import GPT2PivotClusterer
        
        print("Testing ETS with Structured Data")
        print("=" * 50)
        
        # Create test data with 3 clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(30, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster3 = np.random.randn(30, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
        test_data = np.vstack([cluster1, cluster2, cluster3])
        
        # Test different threshold percentiles
        for percentile in [0.1, 0.2, 0.3, 0.5]:
            print(f"\nTesting with threshold_percentile={percentile}")
            
            clusterer = GPT2PivotClusterer(clustering_method='ets', threshold_percentile=percentile)
            clusterer._setup_sklearn()
            
            if clusterer.ets_available:
                labels, centers, n_clusters, score = clusterer._cluster_with_ets(test_data.tolist())
                print(f"  - Found {n_clusters} clusters")
                if n_clusters > 1:
                    print(f"  - Silhouette score: {score:.3f}")
                print(f"  - Cluster sizes: {np.bincount(labels)}")
                
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ets_with_clusters()
    sys.exit(0 if success else 1)