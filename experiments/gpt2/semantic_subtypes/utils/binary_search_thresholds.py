#!/usr/bin/env python3
"""
Binary search for ETS threshold that gives reasonable cluster counts.
Testing in the range 0.95 to 0.999
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def test_threshold(percentile, layer_11_activations):
    """Test a single threshold and return cluster count."""
    from gpt2_pivot_clusterer import GPT2PivotClusterer
    
    clusterer = GPT2PivotClusterer(
        clustering_method='ets',
        threshold_percentile=percentile,
        random_state=42
    )
    
    if clusterer._setup_sklearn() and clusterer.ets_available:
        try:
            labels, centers, n_clusters, silhouette = clusterer._cluster_with_ets(layer_11_activations)
            return n_clusters
        except Exception as e:
            print(f"Error at {percentile}: {str(e)[:40]}")
            return None
    return None

def binary_search_thresholds():
    """Binary search for threshold giving 10-50 clusters."""
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
    
    print(f"Binary search for optimal threshold ({len(layer_11_activations)} samples)")
    print("\nPercentile | Clusters | Status")
    print("-" * 50)
    
    # Start with key test points
    test_points = [
        0.95,    # Start high
        0.98,    # Higher
        0.99,    # Even higher
        0.995,   # Very high
        0.997,   # Extremely high
        0.998,   # Near maximum
        0.999,   # Almost maximum
        0.9995,  # Super high
        0.9998,  # Ultra high
        0.9999   # Extreme
    ]
    
    results = []
    
    for percentile in test_points:
        print(f"  {percentile:.4f}   | ", end="", flush=True)
        n_clusters = test_threshold(percentile, layer_11_activations)
        
        if n_clusters is not None:
            status = ""
            if 10 <= n_clusters <= 50:
                status = "*** OPTIMAL RANGE ***"
            elif 50 < n_clusters <= 100:
                status = "* Acceptable *"
            
            print(f"{n_clusters:4d}    | {status}")
            results.append((percentile, n_clusters))
            
            # If we found a good range, search more finely around it
            if 10 <= n_clusters <= 100:
                print("\nRefining search around this value...")
                # Test a few more points nearby
                if percentile > 0.99:
                    for delta in [-0.0005, -0.0002, 0.0002, 0.0005]:
                        test_p = percentile + delta
                        if 0.95 < test_p < 0.9999:
                            print(f"  {test_p:.4f}   | ", end="", flush=True)
                            n = test_threshold(test_p, layer_11_activations)
                            if n:
                                status = ""
                                if 10 <= n <= 50:
                                    status = "*** OPTIMAL RANGE ***"
                                print(f"{n:4d}    | {status}")
                                results.append((test_p, n))
        else:
            print("Failed")
    
    # Find best result
    optimal_results = [(p, n) for p, n in results if 10 <= n <= 50]
    if optimal_results:
        # Prefer around 20-30 clusters
        best = min(optimal_results, key=lambda x: abs(x[1] - 25))
        print(f"\n🎯 BEST THRESHOLD: {best[0]:.4f} with {best[1]} clusters")
    else:
        acceptable = [(p, n) for p, n in results if 10 <= n <= 100]
        if acceptable:
            best = min(acceptable, key=lambda x: abs(x[1] - 50))
            print(f"\n✓ Best available: {best[0]:.4f} with {best[1]} clusters")

if __name__ == "__main__":
    binary_search_thresholds()