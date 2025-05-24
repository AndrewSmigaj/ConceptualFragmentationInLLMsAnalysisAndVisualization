#!/usr/bin/env python3
"""Inspect the exact structure of activations data."""

import pickle
from pathlib import Path

exp_dir = Path("semantic_subtypes_experiment_20250523_111112")
activations_file = exp_dir / "semantic_subtypes_activations.pkl"

with open(activations_file, 'rb') as f:
    data = pickle.load(f)

activations_dict = data['activations']

# Check a few entries
print("Checking structure of activations dict...")
print(f"Total entries: {len(activations_dict)}")

# Look at first few entries
for i, (key, value) in enumerate(activations_dict.items()):
    if i >= 3:  # Just check first 3
        break
    
    print(f"\nEntry {key}:")
    print(f"  Type: {type(value)}")
    
    if isinstance(value, dict):
        print(f"  Keys: {list(value.keys())}")
        # Check one sub-entry
        if value:
            first_subkey = list(value.keys())[0]
            first_subvalue = value[first_subkey]
            print(f"  First sub-entry [{first_subkey}]: type={type(first_subvalue)}, shape={getattr(first_subvalue, 'shape', 'N/A')}")
    elif hasattr(value, 'shape'):
        print(f"  Shape: {value.shape}")
    
# Check if the keys are layer indices
all_keys = list(activations_dict.keys())
print(f"\nAll keys (first 10): {all_keys[:10]}")
print(f"Are keys integers? {all(isinstance(k, int) for k in all_keys[:10])}")

# Check structure by following one word through layers
word_idx = 0
print(f"\n\nChecking word {word_idx} across layers:")
word_data = activations_dict.get(word_idx, {})
print(f"Type: {type(word_data)}")
if isinstance(word_data, dict):
    print(f"Layer keys: {sorted(word_data.keys())}")
    
    # Check one layer
    if 0 in word_data:
        layer_0_act = word_data[0]
        print(f"Layer 0 activation: type={type(layer_0_act)}, shape={getattr(layer_0_act, 'shape', 'N/A')}")
        
        # If it's a dict, go deeper
        if isinstance(layer_0_act, dict):
            print(f"  Layer 0 dict keys: {list(layer_0_act.keys())}")
            if layer_0_act:
                first_key = list(layer_0_act.keys())[0]
                first_val = layer_0_act[first_key]
                print(f"  First entry [{first_key}]: type={type(first_val)}, shape={getattr(first_val, 'shape', 'N/A')}")