#!/usr/bin/env python3
"""
Find optimal clusters for a sample of layers using K-means elbow method.
"""

import pickle
import numpy as np
from pathlib import Path
import sys
import time
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

def find_kmeans_elbow(activations, k_range=(2, 8)):
    """Find optimal k using elbow method with timing."""
    
    start_time = time.time()
    inertias = []
    silhouettes = []
    k_values = list(range(k_range[0], min(k_range[1] + 1, len(activations))))
    
    print(f"    Testing k values: {k_values}")
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activations)
        
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        if k > 1 and k < len(activations):
            silhouette = silhouette_score(activations, labels)
            silhouettes.append(silhouette)
        else:
            silhouettes.append(0)
        
        print(f"      k={k}: inertia={inertia:.0f}, silhouette={silhouettes[-1]:.3f}")
    
    # Find elbow point
    if len(k_values) < 3:
        optimal_k = k_values[0]
    else:
        # Normalize inertias
        inertias_norm = np.array(inertias)
        inertias_norm = (inertias_norm - inertias_norm.min()) / (inertias_norm.max() - inertias_norm.min() + 1e-10)
        
        # Calculate differences
        diffs = []
        for i in range(1, len(inertias_norm)):
            diffs.append(inertias_norm[i-1] - inertias_norm[i])
        
        # Find elbow as point where improvement drops significantly
        threshold = np.mean(diffs) * 0.5
        elbow_idx = 0
        for i, diff in enumerate(diffs):
            if diff < threshold:
                elbow_idx = i
                break
        
        optimal_k = k_values[elbow_idx + 1]
    
    # Get silhouette for optimal k
    optimal_silhouette = silhouettes[k_values.index(optimal_k)]
    
    elapsed = time.time() - start_time
    print(f"    Elbow method suggests k={optimal_k} (silhouette={optimal_silhouette:.3f}) - took {elapsed:.1f}s")
    
    return optimal_k, optimal_silhouette

def main():
    """Find optimal clusters for sample layers."""
    
    print("Loading activations...")
    
    # Load saved activations
    activations_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl")
    with open(activations_path, 'rb') as f:
        activations_data = pickle.load(f)
    
    activations = activations_data['activations']
    sentences = activations_data['sentences']
    
    print(f"Loaded {len(sentences)} words")
    print("="*80)
    
    # Test on a few key layers
    test_layers = [0, 2, 5, 8, 11]  # Early, early-mid, mid, late-mid, late
    
    results = {}
    
    for layer_idx in test_layers:
        print(f"\nLayer {layer_idx}:")
        print("-" * 40)
        
        # Extract layer activations
        layer_activations = []
        
        for sent_idx in sorted(activations.keys()):
            if isinstance(sent_idx, int):
                sent_data = activations[sent_idx]
                if 0 in sent_data:  # Token position 0
                    token_data = sent_data[0]
                    if layer_idx in token_data:
                        layer_activations.append(token_data[layer_idx])
        
        layer_activations = np.array(layer_activations)
        print(f"  Activation shape: {layer_activations.shape}")
        
        # Find optimal k using elbow method
        optimal_k, optimal_silhouette = find_kmeans_elbow(layer_activations, k_range=(2, 6))
        
        results[f'layer_{layer_idx}'] = {
            'optimal_k': optimal_k,
            'silhouette': optimal_silhouette
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Optimal k per layer (K-means elbow method)")
    print("="*80)
    
    for layer in test_layers:
        layer_key = f'layer_{layer}'
        print(f"  Layer {layer:2d}: k={results[layer_key]['optimal_k']} "
              f"(silhouette={results[layer_key]['silhouette']:.3f})")
    
    # Save results
    with open("sample_layer_optimal_k.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to: sample_layer_optimal_k.json")
    
    # Suggest next steps
    all_ks = [r['optimal_k'] for r in results.values()]
    if len(set(all_ks)) == 1:
        print(f"\nAll tested layers suggest k={all_ks[0]}")
        print("Consider using this value for all layers")
    else:
        print(f"\nK values vary from {min(all_ks)} to {max(all_ks)}")
        print("Consider testing all layers or using the mode/median")

if __name__ == "__main__":
    main()