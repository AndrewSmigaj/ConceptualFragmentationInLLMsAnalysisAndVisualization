#!/usr/bin/env python3
"""
Test ETS thresholds on a single layer to find the range that gives 2-4 clusters.
"""

import pickle
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_dimension_thresholds,
    compute_similarity_matrix,
    compute_ets_clustering
)

def test_threshold_range(activations: np.ndarray, threshold_start: float = 0.95, increment: float = 0.01, num_steps: int = 5):
    """Test a range of thresholds and report cluster counts."""
    
    thresholds = [threshold_start + i * increment for i in range(num_steps)]
    results = []
    
    for threshold_percentile in thresholds:
        # Use compute_ets_clustering with threshold_percentile
        cluster_labels, dimension_thresholds = compute_ets_clustering(
            activations,
            threshold_percentile=threshold_percentile,
            min_threshold=1e-5,
            verbose=True,
            batch_size=500  # Smaller batch size
        )
        
        # Count unique clusters
        unique_clusters = len(set(cluster_labels))
        
        results.append({
            'threshold': threshold_percentile,
            'num_clusters': unique_clusters
        })
        
        print(f"Threshold percentile: {threshold_percentile:.6f} -> {unique_clusters} clusters")
    
    return results

def find_threshold_bounds_for_range(results, target_min=2, target_max=4):
    """Find the threshold bounds that give clusters in the target range."""
    
    valid_thresholds = []
    
    for r in results:
        if target_min <= r['num_clusters'] <= target_max:
            valid_thresholds.append(r['threshold'])
    
    if valid_thresholds:
        return min(valid_thresholds), max(valid_thresholds)
    else:
        # Find closest thresholds
        closest = min(results, key=lambda x: abs(x['num_clusters'] - (target_min + target_max) / 2))
        return closest['threshold'], closest['threshold']

def main():
    """Test ETS on a single layer."""
    
    print("Loading activations...")
    
    # Load saved activations
    activations_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl")
    if not activations_path.exists():
        print(f"Error: Activations file not found at {activations_path}")
        return
    
    with open(activations_path, 'rb') as f:
        activations_data = pickle.load(f)
    
    # Check structure of activations data
    print(f"Keys in activations_data: {list(activations_data.keys())}")
    
    if 'activations' in activations_data:
        activations = activations_data['activations']
        print(f"Activations type: {type(activations)}")
        if isinstance(activations, np.ndarray):
            print(f"Activations shape: {activations.shape}")
        elif isinstance(activations, dict):
            print(f"Activations is a dict with keys: {list(activations.keys())[:5]}...")
    
    # Handle different data structures
    test_layer = 5
    if isinstance(activations, np.ndarray) and len(activations.shape) == 4:
        # Format: (num_sentences, num_layers, num_tokens, hidden_dim)
        layer_activations = activations[:, test_layer, 0, :]
    elif isinstance(activations, dict):
        # Data is organized by sentence -> token -> layer -> activation
        # Extract activations for all sentences at specific layer and token position 0
        layer_activations = []
        
        for sent_idx in sorted(activations.keys()):
            if isinstance(sent_idx, int):  # Skip any non-integer keys
                sent_data = activations[sent_idx]
                if 0 in sent_data:  # Token position 0
                    token_data = sent_data[0]
                    if test_layer in token_data:
                        layer_activations.append(token_data[test_layer])
        
        if len(layer_activations) == 0:
            print(f"Error: No activations found for layer {test_layer}")
            # Try to show structure
            sample_sent = list(activations.keys())[0]
            if isinstance(activations[sample_sent], dict):
                print(f"Available layers for sentence {sample_sent}: {list(activations[sample_sent].keys())}")
                # Check if it's token -> layer structure
                sample_token = list(activations[sample_sent].keys())[0]
                if isinstance(activations[sample_sent][sample_token], dict):
                    print(f"Available layers for token {sample_token}: {list(activations[sample_sent][sample_token].keys())}")
                elif isinstance(activations[sample_sent][sample_token], list):
                    print(f"Data at token {sample_token} is a list of length: {len(activations[sample_sent][sample_token])}")
            return
            
        layer_activations = np.array(layer_activations)
    else:
        print(f"Error: Unexpected activations format")
        return
    
    print(f"\nTesting on Layer {test_layer}")
    print(f"Activation shape: {layer_activations.shape}")
    print("="*50)
    
    # Test starting from 0.95 with 0.01 increments (fewer tests for speed)
    print("\nTesting thresholds starting from 0.95:")
    results = test_threshold_range(layer_activations, threshold_start=0.95, increment=0.01, num_steps=5)
    
    # If we need more granularity, test smaller increments
    min_bound, max_bound = find_threshold_bounds_for_range(results, 2, 4)
    
    if min_bound == max_bound:
        # Need finer search
        print(f"\nNeed finer search around {min_bound:.6f}")
        # Try smaller increments around this value
        fine_start = max(0.95, min_bound - 0.01)
        fine_results = test_threshold_range(layer_activations, threshold_start=fine_start, increment=0.001, num_steps=20)
        results.extend(fine_results)
    else:
        fine_results = results
    
    # Find final bounds
    final_min, final_max = find_threshold_bounds_for_range(fine_results, 2, 4)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"For 2-4 clusters on layer {test_layer}:")
    print(f"Threshold range: {final_min:.6f} - {final_max:.6f}")
    
    # Find specific thresholds for 2, 3, and 4 clusters
    print("\nSpecific thresholds:")
    for target in [2, 3, 4]:
        matching = [r for r in fine_results if r['num_clusters'] == target]
        if matching:
            print(f"  {target} clusters: {matching[0]['threshold']:.6f} - {matching[-1]['threshold']:.6f}")
    
    # Recommend a single threshold
    recommended = None
    for r in fine_results:
        if r['num_clusters'] == 3:  # Prefer 3 clusters
            recommended = r['threshold']
            break
    
    if not recommended:
        for r in fine_results:
            if r['num_clusters'] == 2:  # Fall back to 2 clusters
                recommended = r['threshold']
                break
    
    if recommended:
        print(f"\nRecommended threshold for all layers: {recommended:.6f}")
        print("(This should give 2-4 clusters on most layers)")

if __name__ == "__main__":
    main()