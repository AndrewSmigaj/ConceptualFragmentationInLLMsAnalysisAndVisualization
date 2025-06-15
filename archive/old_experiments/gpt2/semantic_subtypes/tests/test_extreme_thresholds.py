#!/usr/bin/env python3
"""
Test very high threshold percentiles to see if we can get reasonable cluster counts.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def test_extreme_thresholds():
    """Test very high percentiles to find reasonable clustering."""
    from gpt2_pivot_clusterer import GPT2PivotClusterer
    
    # Load activations
    results_dir = Path("semantic_subtypes_experiment_20250523_111112")
    with open(results_dir / "semantic_subtypes_activations.pkl", 'rb') as f:
        activations_data = pickle.load(f)
    
    # Extract layer 11 activations
    layer_11_activations = []
    for sent_idx in sorted(activations_data['activations'].keys()):
        if 0 in activations_data['activations'][sent_idx]:
            if 11 in activations_data['activations'][sent_idx][0]:
                layer_11_activations.append(
                    activations_data['activations'][sent_idx][0][11]
                )
    
    print(f"Testing extreme thresholds on {len(layer_11_activations)} samples")
    print("\nPercentile | Clusters | Notes")
    print("-" * 40)
    
    # Test very high percentiles
    for percentile in [0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999]:
        clusterer = GPT2PivotClusterer(
            clustering_method='ets',
            threshold_percentile=percentile,
            random_state=42
        )
        
        if clusterer._setup_sklearn() and clusterer.ets_available:
            try:
                labels, centers, n_clusters, silhouette = clusterer._cluster_with_ets(layer_11_activations)
                
                # Check if we're getting reasonable numbers
                if n_clusters <= 100:
                    print(f"  {percentile:.3f}   |   {n_clusters:4d}   | *** Reasonable range! ***")
                else:
                    print(f"  {percentile:.3f}   |   {n_clusters:4d}   |")
                    
            except Exception as e:
                print(f"  {percentile:.3f}   |  Error  | {str(e)[:40]}")

if __name__ == "__main__":
    test_extreme_thresholds()