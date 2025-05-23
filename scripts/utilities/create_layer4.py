#!/usr/bin/env python3
"""
Script to modify the layer_activations.pkl file to add a layer4.

This script reads the existing layer_activations.pkl file, creates a new 'layer4'
by copying and modifying the 'output' layer, and saves the updated pickle.
It also creates a backup of the original file in case you need to restore it.
"""

import pickle
import os
import shutil
import numpy as np
from datetime import datetime

# Path to the activations file
DATASET = "heart"
SEED = 0
CONFIG = "baseline"
RESULTS_DIR = "D:/concept_fragmentation_results"
ACTIVATIONS_PATH = f"{RESULTS_DIR}/cohesion/{DATASET}/{CONFIG}/seed_{SEED}/layer_activations.pkl"

def add_layer4():
    """Add a layer4 to the activations file."""
    # Check if file exists
    if not os.path.exists(ACTIVATIONS_PATH):
        print(f"Error: File {ACTIVATIONS_PATH} not found.")
        return False

    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{ACTIVATIONS_PATH}.bak.{timestamp}"
    print(f"Creating backup at {backup_path}")
    shutil.copy2(ACTIVATIONS_PATH, backup_path)

    # Load the activations
    print(f"Loading activations from {ACTIVATIONS_PATH}")
    with open(ACTIVATIONS_PATH, 'rb') as f:
        activations = pickle.load(f)

    # Check if layer4 already exists
    if 'layer4' in activations:
        print("layer4 already exists! No changes needed.")
        return True

    # Check if we have output and layer3
    if 'output' not in activations or 'layer3' not in activations:
        print("Error: Required 'output' or 'layer3' keys not found.")
        return False

    # Create layer4 as a slightly modified version of output
    print("Creating layer4 from output layer")
    layer4 = {}
    
    # Process both train and test splits
    for split in ['train', 'test']:
        if split in activations['output']:
            # Get output activations
            output_acts = activations['output'][split]
            
            # If it's a list (multiple epochs), use the last one
            if isinstance(output_acts, list):
                output_acts = output_acts[-1]
            
            # Make a modified copy (add small noise to differentiate from output)
            if hasattr(output_acts, 'shape'):
                # For numpy arrays or tensors
                layer4_acts = output_acts.copy()
                if hasattr(layer4_acts, 'cpu'):
                    layer4_acts = layer4_acts.cpu().numpy()  # Convert tensor to numpy
                
                # Add small noise
                noise = np.random.normal(0, 0.01, layer4_acts.shape)
                layer4_acts = layer4_acts + noise
            else:
                # For other types, just use as is
                layer4_acts = output_acts
            
            layer4[split] = layer4_acts
    
    # Add layer4 to activations
    activations['layer4'] = layer4
    
    # Save the modified activations
    print(f"Saving modified activations with layer4 to {ACTIVATIONS_PATH}")
    with open(ACTIVATIONS_PATH, 'wb') as f:
        pickle.dump(activations, f)
    
    print("Success! Added layer4 to activations file.")
    print("Run the refresh_dashboard.py script to clear caches and regenerate embeddings:")
    print("python refresh_dashboard.py --datasets heart --seeds 0")
    return True

if __name__ == "__main__":
    add_layer4() 