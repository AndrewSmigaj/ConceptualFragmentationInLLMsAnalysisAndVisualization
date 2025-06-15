#!/usr/bin/env python3
"""Find activations file with proper layer structure."""

import pickle
from pathlib import Path

experiment_dirs = sorted([d for d in Path(".").glob("semantic_subtypes_*") 
                        if d.is_dir() and (d / "semantic_subtypes_activations.pkl").exists()],
                       reverse=True)

print(f"Found {len(experiment_dirs)} experiment directories")

for test_dir in experiment_dirs:
    print(f"\nChecking: {test_dir}")
    
    try:
        with open(test_dir / "semantic_subtypes_activations.pkl", 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            # Check if it has layer_X keys
            has_layer_keys = any(k.startswith('layer_') for k in data.keys())
            
            if has_layer_keys:
                print("  ✓ Has layer_X keys!")
                print(f"  Keys: {[k for k in data.keys() if k.startswith('layer_')]}")
                
                # Check structure of a layer
                layer_0 = data.get('layer_0', {})
                if isinstance(layer_0, dict) and 'activations' in layer_0:
                    print(f"  ✓ layer_0 has 'activations' key")
                    print(f"  Activations shape: {layer_0['activations'].shape}")
                    print(f"  THIS IS THE RIGHT FILE!")
                else:
                    print(f"  layer_0 structure: {type(layer_0)}, keys: {layer_0.keys() if isinstance(layer_0, dict) else 'N/A'}")
            else:
                print(f"  No layer_X keys. Keys: {list(data.keys())[:5]}...")
                
    except Exception as e:
        print(f"  Error loading: {e}")