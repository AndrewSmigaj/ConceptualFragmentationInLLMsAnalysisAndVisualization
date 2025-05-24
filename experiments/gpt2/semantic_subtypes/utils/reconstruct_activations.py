#!/usr/bin/env python3
"""Reconstruct activations from original experiment data."""

import pickle
import numpy as np
from pathlib import Path

# We need to reconstruct the activations from the original experiment
# The activations are stored per word in the experiment files

# Load from one of the experiment directories
exp_dir = Path("semantic_subtypes_experiment_20250523_111112")
activations_file = exp_dir / "semantic_subtypes_activations.pkl"

print(f"Loading activations from {exp_dir}")

with open(activations_file, 'rb') as f:
    data = pickle.load(f)

print(f"Data keys: {list(data.keys())}")

# The structure is:
# - data['activations'][word_idx][layer_idx] = activation vector

activations_dict = data['activations']
n_words = len(activations_dict)
n_layers = 13
n_features = 768  # GPT-2 hidden size

print(f"Found {n_words} words")

# Reconstruct into layer-wise format
activations_by_layer = {}

# The structure is: activations[word_idx][0][layer_idx]
for layer_idx in range(n_layers):
    # Collect all activations for this layer
    layer_activations = []
    
    for word_idx in range(n_words):
        if word_idx in activations_dict:
            word_data = activations_dict[word_idx]
            # Check if it has the expected structure
            if 0 in word_data and isinstance(word_data[0], dict) and layer_idx in word_data[0]:
                activation = word_data[0][layer_idx]
                layer_activations.append(activation)
    
    # Convert to numpy array
    layer_activations = np.array(layer_activations)
    
    activations_by_layer[f'layer_{layer_idx}'] = {
        'activations': layer_activations
    }
    
    print(f"Layer {layer_idx}: shape {layer_activations.shape}")

# Save the reconstructed activations
output_file = "activations_by_layer.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(activations_by_layer, f)

print(f"\nSaved reconstructed activations to {output_file}")

# Also save word metadata
metadata = {
    'sentences': data.get('sentences', []),
    'tokens': data.get('tokens', []),
    'semantic_subtypes': data.get('semantic_subtypes', {})
}

with open("word_metadata.pkl", 'wb') as f:
    pickle.dump(metadata, f)

print("Saved word metadata to word_metadata.pkl")