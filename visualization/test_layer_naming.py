"""
Test script to verify the layer naming and ordering functionality.
"""

import sys
import os

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualization.traj_plot import get_friendly_layer_name
from concept_fragmentation.analysis.cross_layer_metrics import validate_layer_order

def test_layer_naming():
    """Test the friendly layer name conversion."""
    print("Testing get_friendly_layer_name function")
    print("-" * 50)
    
    # Test standard layer naming
    names = [
        "input", 
        "layer1", 
        "layer2", 
        "layer3", 
        "layer4", 
        "output"
    ]
    
    for name in names:
        friendly_name = get_friendly_layer_name(name)
        print(f"{name:<10} -> {friendly_name}")
    
    # Test hidden layer naming
    names = [
        "hidden1", 
        "hidden2", 
        "hidden3", 
        "hidden4"
    ]
    
    print("\nTesting hidden layer naming")
    print("-" * 50)
    for name in names:
        friendly_name = get_friendly_layer_name(name)
        print(f"{name:<10} -> {friendly_name}")
    
    # Test mixed layer naming
    print("\nTesting mixed layer naming")
    print("-" * 50)
    mixed_names = [
        "input", 
        "layer1", 
        "hidden1", 
        "layer2", 
        "hidden2", 
        "hidden3", 
        "output"
    ]
    
    friendly_names = [get_friendly_layer_name(name) for name in mixed_names]
    print("Original names:", mixed_names)
    print("Friendly names:", friendly_names)
    
    # Test layer ordering
    print("\nTesting layer ordering")
    print("-" * 50)
    
    # Deliberately shuffle the names
    unordered = ["layer3", "hidden2", "input", "layer1", "hidden3", "output", "layer2", "hidden1", "layer4"]
    ordered = validate_layer_order(unordered)
    
    print("Unordered:", unordered)
    print("Ordered:  ", ordered)
    print("Friendly: ", [get_friendly_layer_name(name) for name in ordered])
    
    # Test only hiddenN layers
    print("\nTesting ordering with only hidden layers")
    print("-" * 50)
    
    hidden_only = ["hidden3", "hidden1", "hidden2", "hidden4"]
    ordered_hidden = validate_layer_order(hidden_only)
    
    print("Unordered:", hidden_only)
    print("Ordered:  ", ordered_hidden)
    print("Friendly: ", [get_friendly_layer_name(name) for name in ordered_hidden])

if __name__ == "__main__":
    test_layer_naming()