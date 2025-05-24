#!/usr/bin/env python3
"""Test that ETS clustering now works correctly."""

import sys
from pathlib import Path
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def test_ets_clustering():
    """Test ETS clustering functionality."""
    try:
        from gpt2_pivot_clusterer import GPT2PivotClusterer
        
        print("Testing Fixed ETS Implementation")
        print("=" * 50)
        
        # Create test data
        np.random.seed(42)
        test_data = np.random.randn(100, 10)  # 100 samples, 10 dimensions
        
        # Create clusterer with ETS (use higher percentile for test data)
        clusterer = GPT2PivotClusterer(clustering_method='ets', threshold_percentile=0.5)
        clusterer._setup_sklearn()
        
        print(f"ETS available: {clusterer.ets_available}")
        
        if clusterer.ets_available:
            # Test clustering
            labels, centers, n_clusters, score = clusterer._cluster_with_ets(test_data.tolist())
            print(f"ETS clustering successful!")
            print(f"  - Found {n_clusters} clusters")
            print(f"  - Silhouette score: {score:.3f}")
            print(f"  - Cluster sizes: {np.bincount(labels)}")
        else:
            print("ETS still not available")
            
        return clusterer.ets_available
        
    except Exception as e:
        print(f"ETS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ets_clustering()
    sys.exit(0 if success else 1)