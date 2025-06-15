#!/usr/bin/env python3
"""Check structure of optimal clustering files."""

import pickle
from pathlib import Path

optimal_dir = Path("semantic_subtypes_optimal_experiment_20250523_182344")

# Check K-means file
kmeans_file = optimal_dir / "semantic_subtypes_kmeans_optimal.pkl"
if kmeans_file.exists():
    print("Checking K-means clustering file...")
    with open(kmeans_file, 'rb') as f:
        kmeans_data = pickle.load(f)
    
    print(f"Type: {type(kmeans_data)}")
    if isinstance(kmeans_data, dict):
        print(f"Keys: {list(kmeans_data.keys())}")
        
        # Check layer_results structure
        if 'layer_results' in kmeans_data:
            layer_results = kmeans_data['layer_results']
            print(f"\nlayer_results type: {type(layer_results)}")
            if isinstance(layer_results, dict):
                print(f"layer_results keys: {list(layer_results.keys())}")
                
                # Check first layer
                if 'layer_0' in layer_results:
                    layer_0 = layer_results['layer_0']
                    print(f"\nlayer_0 type: {type(layer_0)}")
                    if isinstance(layer_0, dict):
                        print(f"layer_0 keys: {list(layer_0.keys())}")
                        
                        if 'activations' in layer_0:
                            print(f"✓ Has activations! Shape: {layer_0['activations'].shape}")
                        
                        if 'cluster_labels' in layer_0:
                            print(f"✓ Has cluster_labels! Length: {len(layer_0['cluster_labels'])}")
                
# Now we need to get activations from the original experiment
print("\n" + "="*60)
print("Looking for original activations...")

# Let's extract activations from layer_results
if 'layer_results' in kmeans_data:
    layer_results = kmeans_data['layer_results']
    
    # Check if we have activations in layer_results
    has_activations = False
    for layer_idx in range(13):
        layer_key = f'layer_{layer_idx}'
        if layer_key in layer_results and 'activations' in layer_results[layer_key]:
            has_activations = True
            break
    
    if has_activations:
        print("Found activations in layer_results!")
        
        # Create a proper activations structure
        activations_by_layer = {}
        for layer_idx in range(13):
            layer_key = f'layer_{layer_idx}'
            if layer_key in layer_results and 'activations' in layer_results[layer_key]:
                activations_by_layer[layer_key] = {
                    'activations': layer_results[layer_key]['activations']
                }
                print(f"Extracted layer {layer_idx}: shape {layer_results[layer_key]['activations'].shape}")
        
        # Save for our use
        with open("activations_by_layer.pkl", 'wb') as f:
            pickle.dump(activations_by_layer, f)
        print(f"\nSaved activations to activations_by_layer.pkl")
    else:
        print("No activations found in layer_results")