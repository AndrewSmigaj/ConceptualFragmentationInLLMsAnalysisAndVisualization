#!/usr/bin/env python3
"""Test ETS integration in GPT2PivotClusterer."""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def test_ets_integration():
    """Test that ETS clustering is properly integrated."""
    try:
        from gpt2_pivot_clusterer import GPT2PivotClusterer
        
        print("Testing ETS Integration")
        print("=" * 50)
        
        # Test 1: Can create clusterer with ETS
        clusterer_ets = GPT2PivotClusterer(clustering_method='ets', threshold_percentile=0.1)
        print("✓ Created GPT2PivotClusterer with ETS method")
        
        # Test 2: Check if ETS is available after setup
        clusterer_ets._setup_sklearn()
        print(f"✓ ETS available: {clusterer_ets.ets_available}")
        
        # Test 3: Can create with different threshold percentiles
        for percentile in [0.05, 0.1, 0.15, 0.2]:
            clusterer = GPT2PivotClusterer(
                clustering_method='ets', 
                threshold_percentile=percentile
            )
            print(f"✓ Created clusterer with threshold_percentile={percentile}")
        
        # Test 4: Verify all methods are supported
        for method in ['kmeans', 'hdbscan', 'ets']:
            clusterer = GPT2PivotClusterer(clustering_method=method)
            print(f"✓ Method '{method}' is supported")
        
        print("\n✓ ETS integration successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ ETS integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ets_integration()
    sys.exit(0 if success else 1)