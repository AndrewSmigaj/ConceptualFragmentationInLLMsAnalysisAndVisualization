#!/usr/bin/env python3
"""Check the structure of activations data."""

import pickle
from pathlib import Path

# Find latest experiment directory with activations
experiment_dirs = sorted([d for d in Path(".").glob("semantic_subtypes_*") 
                        if d.is_dir() and (d / "semantic_subtypes_activations.pkl").exists()],
                       reverse=True)

if experiment_dirs:
    # Try to find one with proper structure
    for test_dir in experiment_dirs:
        print(f"\nChecking: {test_dir}")
    
    with open(latest_dir / "semantic_subtypes_activations.pkl", 'rb') as f:
        data = pickle.load(f)
    
    print(f"\nData type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())[:5]}...")  # First 5 keys
        
        # Check structure of first item
        first_key = list(data.keys())[0]
        first_value = data[first_key]
        print(f"\nFirst key: {first_key}")
        print(f"First value type: {type(first_value)}")
        
        if isinstance(first_value, dict):
            print(f"First value keys: {list(first_value.keys())}")
    
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if data:
            print(f"First item type: {type(data[0])}")