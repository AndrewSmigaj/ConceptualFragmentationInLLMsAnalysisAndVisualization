#!/usr/bin/env python3
"""
Debug script for activation files.

This script helps diagnose issues with activation files and visualizations.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualization.data_interface import (
    inspect_activation_file, load_activations, get_best_config, get_baseline_config
)
from visualization.reducers import Embedder

def diagnose_problem():
    """
    Run diagnostics to find the source of the error.
    """
    # Test datasets
    datasets = ["titanic", "heart"]
    # Test seeds
    seeds = [0, 1, 2]
    
    print("\n=== DIAGNOSING ACTIVATION DATA ISSUES ===")
    
    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---")
        
        # Get configurations
        baseline_config = get_baseline_config(dataset)
        print(f"Baseline config: {baseline_config}")
        
        try:
            # Inspect an activation file
            print(f"\nInspecting baseline activation file:")
            inspect_activation_file(dataset, baseline_config, seeds[0])
            
            # Try loading the activations
            print(f"\nTrying to load activations:")
            activations = load_activations(dataset, baseline_config, seeds[0])
            print(f"Successfully loaded activations with keys: {list(activations.keys())}")
            
            # Check which keys might be causing issues
            print(f"\nAnalyzing keys for potential issues:")
            for key, value in activations.items():
                if not isinstance(value, np.ndarray):
                    if isinstance(value, dict) and 'test' in value and 'train' in value:
                        test_data = value['test']
                        train_data = value['train']
                        print(f"  Nested dict key: '{key}'")
                        print(f"    test: type={type(test_data)}")
                        if isinstance(test_data, list) and len(test_data) > 0:
                            print(f"      first element type: {type(test_data[0])}")
                            if isinstance(test_data[0], list) and len(test_data[0]) > 0:
                                print(f"        nested element type: {type(test_data[0][0])}")
                        if isinstance(test_data, np.ndarray):
                            print(f"      shape: {test_data.shape}, dtype: {test_data.dtype}")
                        
                        print(f"    train: type={type(train_data)}")
                        if isinstance(train_data, list) and len(train_data) > 0:
                            print(f"      first element type: {type(train_data[0])}")
                    else:
                        print(f"  Non-ndarray key: '{key}' (type: {type(value)})")
                elif value.dtype == object:
                    print(f"  Object-dtype key: '{key}' (shape: {value.shape})")
                elif value.ndim < 2:
                    print(f"  1D array key: '{key}' (shape: {value.shape})")
                else:
                    print(f"  Valid key: '{key}' (shape: {value.shape}, dtype: {value.dtype})")
            
            # Try to create an embedder and process some data
            print(f"\nTesting embedder with sample data (inner layer extraction):")
            embedder = Embedder(n_components=3, random_state=42)
            
            # Try a layer with the new nested extraction
            for layer_key in [k for k in activations.keys() if k.startswith('layer')]:
                if isinstance(activations[layer_key], dict) and 'test' in activations[layer_key]:
                    test_data = activations[layer_key]['test']
                    print(f"\nExtracting test data from '{layer_key}':")
                    
                    if isinstance(test_data, list):
                        print(f"  Converting list to array...")
                        try:
                            test_data = np.array(test_data)
                            print(f"  Converted to array with shape: {test_data.shape}")
                            
                            # Try to embed it
                            if test_data.ndim >= 2:
                                print(f"  Attempting to embed layer...")
                                try:
                                    embedded = embedder.fit_transform(test_data)
                                    print(f"  Success! Embedded to shape: {embedded.shape}")
                                    break
                                except Exception as e:
                                    print(f"  Embedding failed: {e}")
                        except Exception as e:
                            print(f"  Conversion failed: {e}")
                    elif isinstance(test_data, np.ndarray):
                        print(f"  Already an array with shape: {test_data.shape}")
                        if test_data.ndim >= 2:
                            try:
                                embedded = embedder.fit_transform(test_data)
                                print(f"  Success! Embedded to shape: {embedded.shape}")
                                break
                            except Exception as e:
                                print(f"  Embedding failed: {e}")
            
        except Exception as e:
            print(f"Error during diagnosis: {e}")
    
    print("\n=== DIAGNOSIS COMPLETE ===")
    print("Check the output above for clues about the data issue.")

if __name__ == "__main__":
    diagnose_problem() 