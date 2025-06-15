"""
Simple test to verify that our layer naming logic works correctly.
"""

import re

def get_friendly_layer_name(layer_name):
    """
    Convert internal layer names to user-friendly display names.
    """
    if layer_name == "input":
        return "Input Space"
    elif layer_name == "output":
        return "Output Layer"
    elif layer_name.startswith("layer"):
        # Extract layer number if present
        match = re.match(r'layer(\d+)', layer_name)
        if match:
            layer_num = int(match.group(1))
            # In this architecture, layer1 is effectively the input layer
            if layer_num == 1:
                return "Input Layer"
            else:
                return f"Hidden Layer {layer_num-1}"
    elif layer_name.startswith("hidden"):
        # Support explicit 'hidden1', 'hidden2', etc. format
        match = re.match(r'hidden(\d+)', layer_name)
        if match:
            hidden_num = int(match.group(1))
            return f"Hidden Layer {hidden_num}"
    
    # If no pattern matches, just capitalize and clean up the name
    return layer_name.replace('_', ' ').title()

# Test with various layer names
layer_names = [
    "input",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "hidden1",
    "hidden2",
    "hidden3",
    "output"
]

print("Layer name mappings:")
print("-" * 40)
for name in layer_names:
    print(f"{name:<10} -> {get_friendly_layer_name(name)}")

print("\nSorting example:")
print("-" * 40)

# Define a layer sorting function (similar to the one in traj_plot.py)
def layer_sort_key(name):
    if name == "input":
        return (0, 0)
    elif name == "output":
        return (999, 0)
    elif name.startswith("layer"):
        match = re.match(r'layer(\d+)', name)
        if match:
            return (1, int(match.group(1)))
    elif name.startswith("hidden"):
        match = re.match(r'hidden(\d+)', name)
        if match:
            hidden_num = int(match.group(1))
            return (2, hidden_num)
    return (3, name)

# Create a shuffled list to test sorting
shuffled = ["layer3", "hidden2", "output", "input", "layer1", "hidden3", "layer4", "hidden1", "layer2"]
print("Unsorted:", shuffled)

# Sort the layers
sorted_layers = sorted(shuffled, key=layer_sort_key)
print("Sorted:  ", sorted_layers)

# Show friendly names for the sorted layers
print("Friendly:", [get_friendly_layer_name(name) for name in sorted_layers])