"""
Trajectory plotting module for concept fragmentation visualization.

This module contains functions for creating 3D plots of sample
trajectories through neural network layers using Plotly.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

# Add parent directory to path to import data_interface
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualization.data_interface import load_dataset_metadata

# Colorblind-friendly color palette
COLORS = {
    "baseline": "#1F77B4",  # Blue
    "regularized": "#FF7F0E",  # Orange
    "highlight": "#2CA02C",  # Green
    "class0": "#D62728",  # Red
    "class1": "#9467BD",  # Purple
    "background": "#F0F0F0"  # Light gray
}

# Qualitative color palette for N classes (Plotly's default)
QUALITATIVE_COLORS = [
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
    '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'
]

# Map classes to marker symbols
# Note: Scatter3d only supports a limited set of symbols: 'circle', 'square', 'diamond', 'cross', 'x'
CLASS_TO_SYMBOL = {
    0: "circle",       # Class 0
    1: "diamond",      # Class 1
    2: "square",       # Class 2
    3: "cross",        # Class 3
    4: "x"             # Class 4
}

# Map dataset-specific class names
DATASET_CLASS_NAMES = {
    "titanic": {0: "Died", 1: "Survived"},
    "heart": {0: "Healthy", 1: "Disease"},
    # Add more datasets as needed
}

# Define a constant for layer separation in the single scene
LAYER_SEPARATION_OFFSET = 5.0 # Doubled from previous value of 2.5
DEFAULT_COLORSCALE = "Viridis" # Default divergent colorscale

def normalize_embeddings(embeddings: Dict[str, np.ndarray], scale: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Normalize embeddings to have similar scales across layers.
    Each layer is centered and scaled independently.
    
    Args:
        embeddings: Dictionary mapping layer names to embedding arrays.
        scale: Scale factor for the normalized embeddings (applied after unit std dev scaling).
        
    Returns:
        Dictionary of normalized embeddings.
    """
    normalized = {}
    for layer, emb in embeddings.items():
        if emb.shape[0] == 0: # Handle empty embeddings (e.g. from problematic train sets)
            normalized[layer] = emb
            continue
        # Center around mean
        centered = emb - np.mean(emb, axis=0)
        
        # Scale to unit standard deviation
        std_dev = np.std(centered, axis=0)
        std_dev[std_dev == 0] = 1.0  # Avoid division by zero
        normalized[layer] = (centered / std_dev) * scale
        
    return normalized

def create_arrows(
    points: np.ndarray, 
    direction_vectors: np.ndarray, # Expect pre-calculated vectors
    scale: float = 0.00825,  # Quarter of previous size (0.033 * 0.25)
    color: str = "gray", # Can also be a numeric value if using a colorscale on the main trace
    layer_names: Optional[List[str]] = None  # Add layer names parameter for hover text
) -> go.Cone:
    """
    Create arrow cones to visualize direction of movement between points.
    
    Args:
        points: Array of starting points (N, 3)
        direction_vectors: Array of direction vectors (N, 3)
        scale: Scale factor for arrows
        color: Color for arrows
        layer_names: Names of layers for hover text
    
    Returns:
        Plotly Cone trace
    """
    # Scale the direction vectors
    scaled_vectors = direction_vectors * scale
    
    # Get friendly layer names for hover text if provided
    friendly_names = None
    if layer_names is not None:
        friendly_names = [get_friendly_layer_name(layer) for layer in layer_names]
    else:
        friendly_names = [""] * len(points)
    
    # Cone objects require special handling for color
    # Create intensity values for coloring (all the same value)
    intensity = np.ones(len(points))
    
    # Create arrow trace with color
    return go.Cone(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=scaled_vectors[:, 0],
        v=scaled_vectors[:, 1],
        w=scaled_vectors[:, 2],
        sizemode="absolute",
        sizeref=0.5,
        colorscale=[[0, color], [1, color]],  # Single color across the entire scale
        cmin=0,
        cmax=1,
        showscale=False,
        opacity=0.8,
        hovertext=friendly_names,
        hoverinfo="text" if layer_names else "skip"
    )

def get_friendly_layer_name(layer_name: str) -> str:
    """
    Convert internal layer names to user-friendly display names.
    
    Args:
        layer_name: The internal layer name (e.g., 'layer1', 'input')
        
    Returns:
        A user-friendly layer name (e.g., 'Input Layer', 'Hidden Layer', 'Output Layer')
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


def get_class_name(class_id: int, dataset: str) -> str:
    """
    Get user-friendly class name based on dataset and class ID.
    
    Args:
        class_id: The class ID/index
        dataset: The dataset name (e.g., 'titanic', 'heart')
        
    Returns:
        Human-readable class name
    """
    # Debug output to see what's being requested
    print(f"DEBUG: get_class_name called with class_id={class_id}, dataset='{dataset}'")
    
    # Get dataset-specific class names if available
    if dataset in DATASET_CLASS_NAMES and class_id in DATASET_CLASS_NAMES[dataset]:
        class_name = DATASET_CLASS_NAMES[dataset][class_id]
        print(f"DEBUG: Found mapped class name: '{class_name}'")
        return class_name
    
    # Default to generic class name
    default_name = f"Class {class_id}"
    print(f"DEBUG: Using default class name: '{default_name}'")
    return default_name


def get_cluster_majority_class(
    cluster_id: int, 
    layer: str, 
    config_name: str, 
    class_labels: np.ndarray, 
    layer_clusters: Dict
) -> int:
    """
    Determine the majority class within a cluster.
    
    Args:
        cluster_id: The cluster ID
        layer: The layer name (e.g., 'layer1')
        config_name: Configuration name (e.g., 'baseline')
        class_labels: Array of class labels for all samples
        layer_clusters: Dictionary of cluster assignments
        
    Returns:
        The majority class ID within the cluster
    """
    # Validate inputs
    if not isinstance(class_labels, np.ndarray) or class_labels.size == 0:
        return 0  # Default if no class labels
        
    if not layer_clusters or config_name not in layer_clusters:
        return 0
        
    if layer not in layer_clusters[config_name]:
        return 0
        
    cluster_info = layer_clusters[config_name][layer]
    if "labels" not in cluster_info or cluster_info["labels"] is None:
        return 0
        
    # Get samples in this cluster
    cluster_labels = cluster_info["labels"]
    
    # Make sure dimensions align
    if len(class_labels) != len(cluster_labels):
        # If lengths differ, take the smaller length
        min_len = min(len(class_labels), len(cluster_labels))
        class_labels = class_labels[:min_len]
        cluster_labels = cluster_labels[:min_len]
    
    cluster_mask = cluster_labels == cluster_id
    if np.sum(cluster_mask) == 0:
        return 0  # Default if cluster is empty
    
    # Get class labels for these samples
    cluster_classes = class_labels[cluster_mask]
    
    # Count occurrences of each class (use float dtype for bincount with non-int labels)
    counts = np.bincount(cluster_classes.astype(int))
    
    # Return the class with the highest count
    return int(np.argmax(counts))

def build_single_scene_figure(
    embeddings_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    samples_to_plot: Optional[List[int]] = None,
    max_samples: int = 200,
    layer_order: Optional[List[str]] = None,
    highlight_indices: Optional[List[int]] = None,
    class_labels: Optional[np.ndarray] = None,
    sample_color_values: Optional[np.ndarray] = None,
    color_value_name: str = "Value",
    title: str = "Neural Network Layer Trajectories (Single Scene)",
    show_arrows: bool = True,
    normalize: bool = True,
    layer_separation: float = LAYER_SEPARATION_OFFSET,
    layer_clusters: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    show_cluster_centers: bool = False,
    color_by: str = "class",  # {"class", "metric", "cluster", "cluster_majority"}
    cluster_name_mapping: Optional[Dict[str, str]] = None,  # Custom cluster names
    shape_by_class: bool = True,  # Whether to use different shapes for classes
    dataset_name: str = ""  # Dataset name for class labels
) -> go.Figure:
    """
    Build a single 3D scene figure showing trajectories across layers.
    Layers are visually separated by an offset on the Y-axis.
    
    Args:
        embeddings_dict: Dict mapping config_names to seed to layer to embeddings.
        samples_to_plot: Indices of samples to include. If None, use all.
        max_samples: Maximum number of samples to plot.
        layer_order: Order of layers for plotting.
        highlight_indices: Indices of samples to highlight.
        class_labels: Class labels for coloring points.
        sample_color_values: Optional array of values for divergent line coloring.
        color_value_name: Name for the color bar if sample_color_values are used.
        title: Title for the figure.
        show_arrows: Whether to show direction arrows.
        normalize: Whether to normalize embeddings before offsetting.
        layer_separation: The amount of offset to apply between layers on the Y-axis.
        layer_clusters: Dictionary of cluster information per config, then per layer (k, centers, labels).
        show_cluster_centers: Whether to display cluster centers as spheres.
        color_by: How to color the points: "class", "metric", "cluster", or "cluster_majority".
        cluster_name_mapping: Optional dictionary mapping cluster keys to descriptive names.
        
    Returns:
        Plotly Figure object with a single 3D scene.
    """
    # Debug statements to help diagnose issues
    print(f"DEBUG: build_single_scene_figure called")
    print(f"DEBUG: class_labels type: {type(class_labels)}")
    if isinstance(class_labels, np.ndarray):
        print(f"DEBUG: class_labels shape: {class_labels.shape}")
        print(f"DEBUG: class_labels first few values: {class_labels[:5]}")
    
    print(f"DEBUG: highlight_indices type: {type(highlight_indices)}")
    if highlight_indices is not None:
        print(f"DEBUG: highlight_indices length: {len(highlight_indices)}")
        print(f"DEBUG: highlight_indices first few values: {highlight_indices[:5] if len(highlight_indices) > 0 else 'empty'}")
    
    # Initialize layer_to_colors at the top level to ensure it's available throughout
    layer_to_colors = {}
    cluster_to_color_map = {}  # Global cluster-to-color mapping
    global_class_to_color_map = {}
    
    configs = list(embeddings_dict.keys())
    if not configs:
        return go.Figure().update_layout(title="No configurations found in embeddings_dict")

    if layer_order is None:
        first_config_name = configs[0]
        if not embeddings_dict[first_config_name]:
            return go.Figure().update_layout(title=f"No seeds found for config {first_config_name}")
        first_seed_key = list(embeddings_dict[first_config_name].keys())[0]
        if not embeddings_dict[first_config_name][first_seed_key]:
            return go.Figure().update_layout(title=f"No layers found for seed {first_seed_key} in config {first_config_name}")
        raw_layer_order = list(embeddings_dict[first_config_name][first_seed_key].keys())
        
        print(f"DEBUG: Raw layer order before sorting: {raw_layer_order}")
        print(f"DEBUG: First config: {first_config_name}, first seed: {first_seed_key}")

        # Log all config/seed/layer combinations for debugging
        for config in configs:
            print(f"DEBUG: Config {config} has {len(embeddings_dict[config])} seeds")
            for seed in embeddings_dict[config]:
                print(f"DEBUG:   Seed {seed} has {len(embeddings_dict[config][seed])} layers")
                print(f"DEBUG:   Layers in seed {seed}: {list(embeddings_dict[config][seed].keys())}")
                for layer_name, layer_data in embeddings_dict[config][seed].items():
                    if isinstance(layer_data, np.ndarray):
                        print(f"DEBUG:     Layer {layer_name} is array with shape {layer_data.shape}")
                    else:
                        print(f"DEBUG:     Layer {layer_name} is {type(layer_data)}")
        
        # Sort to ensure proper layer ordering
        if "input" in raw_layer_order:
            # If input exists, put it first
            layers = ["input"]
            
            # Then add numbered layers in order (both 'layer' and 'hidden' formats)
            # First collect all numbered layers
            numbered_layers = []
            numbered_layers.extend([l for l in raw_layer_order if l.startswith("layer") and l != "input"])
            numbered_layers.extend([l for l in raw_layer_order if l.startswith("hidden")])
            
            # Custom sort function for layer ordering
            def layer_sort_key(name):
                if name.startswith("layer"):
                    # Extract number from layer name
                    match = re.match(r'layer(\d+)', name)
                    if match:
                        return (0, int(match.group(1)))
                elif name.startswith("hidden"):
                    # Extract number from hidden name
                    match = re.match(r'hidden(\d+)', name)
                    if match:
                        # Adjust for correct ordering with layer prefix
                        # hidden1 should come after layer2 (which is Hidden Layer 1)
                        # So we add an offset
                        hidden_num = int(match.group(1))
                        return (1, hidden_num)
                return (2, name)  # Default fallback
            
            # Sort all numbered layers using the custom key
            numbered_layers.sort(key=layer_sort_key)
            layers.extend(numbered_layers)
            
            # Then add output layer if it exists
            if "output" in raw_layer_order:
                layers.append("output")
                
            # Then add any other layers that don't match our patterns
            others = [l for l in raw_layer_order if not l.startswith("layer") and 
                                                  not l.startswith("hidden") and 
                                                  not l.startswith("input") and 
                                                  l != "output"]
            layers.extend(sorted(others))
            
            layer_order = layers
        else:
            # No input layer, just start with numbered layers
            numbered_layers = []
            numbered_layers.extend([l for l in raw_layer_order if l.startswith("layer")])
            numbered_layers.extend([l for l in raw_layer_order if l.startswith("hidden")])
            
            # Use the same custom sort function
            def layer_sort_key(name):
                if name.startswith("layer"):
                    match = re.match(r'layer(\d+)', name)
                    if match:
                        return (0, int(match.group(1)))
                elif name.startswith("hidden"):
                    match = re.match(r'hidden(\d+)', name)
                    if match:
                        hidden_num = int(match.group(1))
                        return (1, hidden_num)
                return (2, name)
            
            numbered_layers.sort(key=layer_sort_key)
            layers = numbered_layers
            
            # Then add output layer if it exists
            if "output" in raw_layer_order:
                layers.append("output")
                
            # Then add any other layers
            others = [l for l in raw_layer_order if not l.startswith("layer") and 
                                                  not l.startswith("hidden") and 
                                                  l != "output"]
            layers.extend(sorted(others))
            
            layer_order = layers
        
        print(f"DEBUG: Layer order after sorting: {layer_order}")

    fig = go.Figure()
    scene_annotations = []
    all_traces = [] # Collect all traces to add them at the end (can help with legend grouping)

    # Determine overall data ranges for consistent axis scaling *after* normalization and offsetting
    # These will be calculated after processing all data. Initialize with loose bounds.
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    # For color scaling if sample_color_values are provided
    min_color_val, max_color_val = float('inf'), float('-inf')
    if sample_color_values is not None:
        # Fix ambiguous truth value error: check size explicitly instead of using array in boolean context
        if isinstance(sample_color_values, np.ndarray) and sample_color_values.size > 0:
            valid_color_values = sample_color_values[~np.isnan(sample_color_values)]
            if len(valid_color_values) > 0:
                min_color_val = np.min(valid_color_values)
                max_color_val = np.max(valid_color_values)
            else: # No valid color values, disable this coloring mode
                sample_color_values = None 
        else:
            # If it's not an array or it's empty, disable coloring
            sample_color_values = None
            
    # Make sure max != min for colorscale
    if sample_color_values is not None: # Only adjust if we are actually using sample_color_values
        if max_color_val == min_color_val:
            max_color_val += 1e-9 # Add a tiny epsilon if all values are the same
            if max_color_val == min_color_val: # Still same (e.g. if min_color_val was 0 and epsilon was too small)
                 min_color_val -= 1e-9 # try subtracting

    added_clusters = set()  # Initialize a set to track added cluster centers

    for config_idx, config_name in enumerate(configs):
        config_data_per_seed = embeddings_dict[config_name]
        if not config_data_per_seed: continue
        
        seed = list(config_data_per_seed.keys())[0] # Use first available seed
        raw_seed_data = config_data_per_seed[seed]

        # Filter layers present in current seed_data and in layer_order
        current_layers = [l for l in layer_order if l in raw_seed_data]
        print(f"DEBUG: Raw layer order: {layer_order}")
        print(f"DEBUG: Available layers in data: {list(raw_seed_data.keys())}")
        print(f"DEBUG: Filtered current_layers: {current_layers}")
        
        if not current_layers: continue

        seed_data_for_plotting = {layer: raw_seed_data[layer] for layer in current_layers}

        if normalize:
            seed_data_for_plotting = normalize_embeddings(seed_data_for_plotting, scale=1.0) # Normalize before offsetting

        # Apply layer offset and collect data for plotting
        offset_seed_data = {}
        for i, layer in enumerate(current_layers):
            emb = seed_data_for_plotting[layer]
            if emb.shape[0] == 0: # Skip empty embeddings
                offset_seed_data[layer] = emb
                continue
            
            offset_emb = emb.copy()
            offset_emb[:, 1] += i * layer_separation # Offset Y-axis
            offset_seed_data[layer] = offset_emb

            # Update overall ranges for main plot axes
            min_x, max_x = min(min_x, np.min(offset_emb[:, 0])), max(max_x, np.max(offset_emb[:, 0]))
            min_y, max_y = min(min_y, np.min(offset_emb[:, 1])), max(max_y, np.max(offset_emb[:, 1]))
            min_z, max_z = min(min_z, np.min(offset_emb[:, 2])), max(max_z, np.max(offset_emb[:, 2]))

            # Add layer name annotations at the mean x,z of the layer, at its y-offset
            if config_idx == 0: # Add annotations and floor planes only once (for the first config)
                layer_mean_x = np.mean(offset_emb[:, 0])
                layer_mean_z = np.mean(offset_emb[:, 2])
                current_layer_y_base = i * layer_separation

                # Create annotation for this layer
                layer_annotation = dict(
                    showarrow=False,
                    x=layer_mean_x,
                    y=current_layer_y_base + layer_separation * 0.3, # Adjusted y for annotation relative to new offset
                    z=layer_mean_z,
                    text=f"<b>{get_friendly_layer_name(layer)}</b>",
                    xanchor="center",
                    yanchor="middle",
                    font=dict(color="black", size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)"
                )
                scene_annotations.append(layer_annotation)
                
                print(f"DEBUG: Added annotation for layer '{layer}' with friendly name '{get_friendly_layer_name(layer)}'")

                # Add a floor plane for this layer
                layer_min_x_local, layer_max_x_local = np.min(emb[:, 0]), np.max(emb[:, 0])
                layer_min_z_local, layer_max_z_local = np.min(emb[:, 2]), np.max(emb[:, 2])
                
                # Expand slightly for padding if desired, or use raw extents
                x_padding = (layer_max_x_local - layer_min_x_local) * 0.05
                z_padding = (layer_max_z_local - layer_min_z_local) * 0.05
                
                plane_x_coords = [layer_min_x_local - x_padding, layer_max_x_local + x_padding]
                plane_z_coords = [layer_min_z_local - z_padding, layer_max_z_local + z_padding]

                all_traces.append(go.Mesh3d(
                    x=[plane_x_coords[0], plane_x_coords[1], plane_x_coords[1], plane_x_coords[0]],
                    y=[current_layer_y_base, current_layer_y_base, current_layer_y_base, current_layer_y_base],
                    z=[plane_z_coords[0], plane_z_coords[0], plane_z_coords[1], plane_z_coords[1]],
                    i=[0, 0], j=[1, 2], k=[2, 3], # Define two triangles for the rectangle
                    opacity=0.1,
                    color='grey',
                    hoverinfo='skip',
                    showlegend=False,
                    name=f'{layer}_floor'
                ))

                # Add cluster centers if requested
                if show_cluster_centers and layer_clusters and config_name in layer_clusters:
                    config_specific_clusters = layer_clusters[config_name]
                    if layer in config_specific_clusters: # Check if current layer has cluster info for this config
                        cluster_info = config_specific_clusters[layer]
                        # Ensure cluster_info is not None and contains 'centers'
                        if cluster_info and "centers" in cluster_info and cluster_info["centers"] is not None:
                            centers = cluster_info["centers"]
                            if centers.ndim == 2 and centers.shape[1] == 3: # Basic check for 3D coordinates
                                # If normalization is enabled, normalize the cluster centers too
                                if normalize:
                                    # Get the original embeddings for this layer before normalization
                                    orig_emb = raw_seed_data[layer]
                                    if orig_emb.shape[0] > 0:  # Skip if no data
                                        # Apply the same normalization to centers as was applied to embeddings
                                        emb_mean = np.mean(orig_emb, axis=0)  # Mean of original embeddings
                                        emb_std = np.std(orig_emb, axis=0)  # Std of original embeddings
                                        emb_std[emb_std == 0] = 1.0  # Avoid division by zero
                                        
                                        # Normalize centers using same mean/std as the embeddings
                                        centers = (centers - emb_mean) / emb_std
                                
                                # Centers are already in the embedding space, just apply the y-offset
                                offset_centers = centers.copy()
                                offset_centers[:, 1] += i * layer_separation
                                
                                # Note: Cluster centers will be drawn later, after colors are determined
                                # This ensures correct coloring in cluster_majority mode
                            else:
                                print(f"Warning: Invalid cluster centers for {config_name} - {layer}: {centers}")
                        else:
                            print(f"Warning: No or invalid cluster_info[\'centers\'] for {config_name} - {layer}")
                    # else: print(f"Debug: Layer {layer} not in config_specific_clusters for {config_name}")
                # else: print(f"Debug: Conditions not met for show_cluster_centers for {config_name} - {layer}. show: {show_cluster_centers}, layer_clusters: {bool(layer_clusters)}, config_name in layer_clusters: {config_name in layer_clusters if layer_clusters else False}")
        
        # Determine samples to plot for this config/seed
        num_samples_in_data = offset_seed_data[current_layers[0]].shape[0]
        current_samples_to_plot = []
        if num_samples_in_data > 0:
            if samples_to_plot is None:
                current_samples_to_plot = np.arange(min(num_samples_in_data, max_samples))
            else:
                # Filter provided samples_to_plot to be within bounds of current data
                valid_samples = [s for s in samples_to_plot if s < num_samples_in_data]
                current_samples_to_plot = np.random.choice(valid_samples, min(len(valid_samples), max_samples), replace=False) if valid_samples else []
        
        # Highlight logic
        current_highlight_indices = highlight_indices if highlight_indices is not None else []
        # Filter highlights to be within current_samples_to_plot
        highlight_set = set(s for s in current_highlight_indices if s in current_samples_to_plot)

        # Class colors or cluster colors logic
        point_colors_array = None
        # layer_to_colors is already initialized at the beginning of the function
        # Fix ambiguous truth value error: check class_labels explicitly
        if color_by == "class" and class_labels is not None and isinstance(class_labels, np.ndarray) and class_labels.size > 0:
            # Check length match explicitly
            if len(class_labels) == num_samples_in_data:
                unique_classes = sorted(list(np.unique(class_labels)))  # Sort for consistent color assignment
                class_to_color_map = {
                    cls_label: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] 
                    for i, cls_label in enumerate(unique_classes)
                }
                point_colors_array = np.array([class_to_color_map.get(c, "grey") for c in class_labels])
            else:
                print(f"DEBUG: class_labels length mismatch: {len(class_labels)} vs {num_samples_in_data}")
        elif color_by == "cluster" and layer_clusters and config_name in layer_clusters:
            config_specific_clusters = layer_clusters[config_name]
            # For cluster coloring, use the last layer's cluster assignments for this config
            last_layer = current_layers[-1]
            if last_layer in config_specific_clusters:
                cluster_info = config_specific_clusters[last_layer]
                if cluster_info and "labels" in cluster_info and cluster_info["labels"] is not None:
                    cluster_labels_for_coloring = cluster_info["labels"]
                    # Ensure labels match the number of samples expected for coloring
                    if len(cluster_labels_for_coloring) == num_samples_in_data:
                        unique_clusters = sorted(list(np.unique(cluster_labels_for_coloring)))
                        cluster_to_color_map = {
                            cluster_idx: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)]
                            for i, cluster_idx in enumerate(unique_clusters)
                        }
                        point_colors_array = np.array([cluster_to_color_map.get(c, "grey") for c in cluster_labels_for_coloring])
                    else:
                        print(f"Warning: Cluster label count mismatch for {config_name}-{last_layer}. Got {len(cluster_labels_for_coloring)}, expected {num_samples_in_data}")
                else:
                    print(f"Warning: No or invalid cluster_info[\'labels\'] for {config_name}-{last_layer} for coloring")
        elif color_by == "cluster_majority" and layer_clusters and config_name in layer_clusters and class_labels is not None and isinstance(class_labels, np.ndarray) and class_labels.size > 0:
            # First, create class to color map (same as in "color by class" mode)
            if len(class_labels) == num_samples_in_data:
                unique_classes = sorted(list(np.unique(class_labels)))
                class_to_color_map = {
                    cls_label: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] 
                    for i, cls_label in enumerate(unique_classes)
                }
                
                # NEW IMPLEMENTATION: For cluster_majority, create a consistent mapping from cluster ID to majority class
                # This ensures clusters have consistent colors across layers
                
                # Initialize dictionaries to store majority class mapping across all layers
                config_specific_clusters = layer_clusters[config_name]
                
                # First, calculate the majority class for each cluster in each layer
                # This will ensure consistent coloring
                cluster_majority_class_map = {}  # Map of (layer, cluster_id) -> majority_class
                
                # Debug output for consistency checks
                print(f"DEBUG: Calculating majority class colors for {config_name}")
                
                for layer in current_layers:
                    if layer in config_specific_clusters:
                        cluster_info = config_specific_clusters[layer]
                        if cluster_info and "labels" in cluster_info and cluster_info["labels"] is not None:
                            cluster_labels = cluster_info["labels"]
                            if len(cluster_labels) == num_samples_in_data:
                                # Calculate majority class for each cluster in this layer
                                for cluster_id in np.unique(cluster_labels):
                                    cluster_mask = cluster_labels == cluster_id
                                    if np.sum(cluster_mask) > 0:  # Ensure cluster has points
                                        cluster_classes = class_labels[cluster_mask]
                                        class_counts = np.bincount(cluster_classes.astype(int))
                                        majority_class = np.argmax(class_counts)
                                        # Use config_name in the key for global consistency
                                        cluster_key = (config_name, layer, cluster_id)
                                        cluster_majority_class_map[cluster_key] = majority_class
                                        # Store color directly in global map
                                        cluster_to_color_map[cluster_key] = class_to_color_map.get(majority_class, "grey")
                                        print(f"DEBUG: {config_name}-{layer}, Cluster {cluster_id} -> Class {majority_class}, Color {cluster_to_color_map[cluster_key]}")
                                    else:
                                        # Use config_name in the key for consistency
                                        cluster_key = (config_name, layer, cluster_id)
                                        cluster_majority_class_map[cluster_key] = 0  # Default if empty
                                        cluster_to_color_map[cluster_key] = class_to_color_map.get(0, "grey")
                            else:
                                print(f"Warning: Cluster label count mismatch for {config_name}-{layer}. Got {len(cluster_labels)}, expected {num_samples_in_data}")
                        else:
                            print(f"Warning: No or invalid cluster_info['labels'] for {config_name}-{layer} for majority coloring")
                
                # Colors are already stored in cluster_to_color_map when calculating majority class
                # So we don't need to create the mapping again
                
                # Now create the layer_to_colors mapping based on this consistent mapping
                # IMPORTANT: Don't reinitialize layer_to_colors - it's already initialized at the top of the function
                for layer in current_layers:
                    if layer in config_specific_clusters:
                        cluster_info = config_specific_clusters[layer]
                        if cluster_info and "labels" in cluster_info and cluster_info["labels"] is not None:
                            cluster_labels = cluster_info["labels"]
                            if len(cluster_labels) == num_samples_in_data:
                                # Assign colors based on our consistent mapping
                                layer_colors = np.array([
                                    cluster_to_color_map.get((config_name, layer, cluster_id), "grey") 
                                    for cluster_id in cluster_labels
                                ])
                                layer_to_colors[layer] = layer_colors
                                print(f"DEBUG: Created colors for layer {layer} with {len(np.unique(layer_colors))} unique colors")
            else:
                print(f"DEBUG: class_labels length mismatch for cluster_majority: {len(class_labels)} vs {num_samples_in_data}")

        # Plot scatter points for each layer
        for layer in current_layers:
            data = offset_seed_data[layer]
            if data.shape[0] == 0: continue

            base_color = COLORS.get(config_name, "grey")
            
            # For cluster_majority mode, use the layer-specific colors
            if color_by == "cluster_majority" and layer in layer_to_colors:
                scatter_colors = layer_to_colors[layer]
            else:
                scatter_colors = point_colors_array if point_colors_array is not None else base_color
                
            # Create marker properties dict
            marker_props = dict(
                size=3,
                color=scatter_colors,
                opacity=0.1 if highlight_set and sample_color_values is None else 0.4
            )
            
            # Add shape by class if enabled and we have class labels
            if shape_by_class and isinstance(class_labels, np.ndarray) and class_labels.size > 0:
                # Pick symbols based on class for each point
                # Note: To avoid performance issues with large datasets, we'll use a single symbol
                # for the background points, and only use different symbols for highlighted points
                if len(class_labels) >= data.shape[0]:
                    # Use default symbol for all points in this layer
                    marker_props["symbol"] = "circle"

            all_traces.append(go.Scatter3d(
                x=data[:, 0], y=data[:, 1], z=data[:, 2],
                mode="markers",
                marker=marker_props,
                name=f"{config_name} - {layer} points",
                legendgroup=f"{config_name}_{layer}", 
                showlegend=False, # Too many items if shown per layer
                hoverinfo='skip' # Can enable if specific point data is useful
            ))
            
            # Highlighted points on top for this layer
            if highlight_set:
                highlight_data_indices = sorted(list(s for s in highlight_set if s < data.shape[0]))
                if highlight_data_indices:
                    h_data = data[highlight_data_indices, :]
                    h_colors = scatter_colors[highlight_data_indices] if isinstance(scatter_colors, np.ndarray) else base_color
                    
                    # Create highlight marker properties
                    h_marker_props = dict(
                        size=5, 
                        color=h_colors, 
                        opacity=0.9
                    )
                    
                    # Add class-based shapes for highlighted points if enabled
                    if shape_by_class and isinstance(class_labels, np.ndarray) and class_labels.size > 0:
                        # Make sure class_labels is large enough
                        if max(highlight_data_indices) < len(class_labels):
                            # Get class labels for the highlighted points
                            highlight_classes = class_labels[highlight_data_indices]
                            
                            # Map classes to symbols
                            h_symbols = [CLASS_TO_SYMBOL.get(int(cls), "circle") for cls in highlight_classes]
                            h_marker_props["symbol"] = h_symbols
                    
                    all_traces.append(go.Scatter3d(
                        x=h_data[:, 0], y=h_data[:, 1], z=h_data[:, 2],
                        mode="markers",
                        marker=h_marker_props,
                        name=f"{config_name} - Highlights",
                        legendgroup=f"{config_name}_highlights",
                        showlegend=(config_idx==0 and layer==current_layers[0]), # Show legend only once per config for highlights
                        hovertemplate=(
                            f"{config_name} - {get_friendly_layer_name(layer)}<br>" +
                            "Sample %{customdata[0]}<br>" +
                            "Class: %{customdata[1]}"
                        ),
                        customdata=[[idx, get_class_name(int(class_labels[idx]), dataset_name) if idx < len(class_labels) else "Unknown"] 
                                  for idx in highlight_data_indices]
                    ))

        # Plot trajectory lines and arrows for selected samples
        for sample_idx in current_samples_to_plot:
            coords_list = []
            valid_layer_count_for_sample = 0
            for layer in current_layers:
                if sample_idx < offset_seed_data[layer].shape[0]:
                    coords_list.append(offset_seed_data[layer][sample_idx])
                    valid_layer_count_for_sample +=1
            
            if valid_layer_count_for_sample < 2: continue # Need at least 2 points for a line

            coords = np.array(coords_list)
            is_highlight = sample_idx in highlight_set
            line_props = dict(width=3 if is_highlight else 2)
            trace_opacity = 0.9 if is_highlight else 0.5 # Make non-highlighted a bit more visible with divergent colors
            
            # Determine line color
            use_divergent_color = False
            numeric_color_val = np.nan # Initialize

            if sample_color_values is not None and sample_idx < len(sample_color_values) and not np.isnan(sample_color_values[sample_idx]):
                numeric_color_val = sample_color_values[sample_idx]
                line_props["color"] = numeric_color_val
                line_props["colorscale"] = DEFAULT_COLORSCALE
                # Apply cmin and cmax for consistent scaling across all lines using this colorscale
                line_props["cmin"] = min_color_val
                line_props["cmax"] = max_color_val
                # Add colorbar only for the first config and if using divergent colors
                if 'colorbar' not in fig.layout.scene and config_idx == 0 : # Check if colorbar is already planned for the scene
                     line_props["colorbar"] = dict(title=color_value_name, thickness=15, len=0.75, y=0.5)
                use_divergent_color = True
            elif scatter_colors is not None and sample_idx < len(scatter_colors):
                line_props["color"] = scatter_colors[sample_idx]
            else:
                line_props["color"] = base_color

            # If using divergent colors, marker color should also use the numeric value
            marker_props = dict(size=1) # Default small markers for line ends
            if is_highlight: # Highlighted lines get larger markers
                marker_props["size"] = 3
            
            # Set marker color based on line color determination
            if use_divergent_color: 
                marker_props["color"] = numeric_color_val # This is already set as line_props["color"]
                marker_props["colorscale"] = DEFAULT_COLORSCALE 
                marker_props["cmin"] = min_color_val
                marker_props["cmax"] = max_color_val
            else: # For class-based or default colors (non-divergent)
                 marker_props["color"] = line_props["color"]

            all_traces.append(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='lines+markers' if is_highlight or use_divergent_color else 'lines',
                marker=marker_props,
                line=line_props,
                opacity=trace_opacity,
                name=f"{config_name} - Sample {sample_idx}",
                legendgroup=f"{config_name}_trajectory_{'divergent' if use_divergent_color else 'fixed'}",
                showlegend=False # Generally too many lines for legend
            ))

            # Arrow coloring needs to be consistent
            arrow_color_for_segment = line_props["color"]
            if use_divergent_color:
                # For arrows on a divergently colored line, they should match the line's color
                # Instead of using the numeric value directly, we'll use the same color as the line
                # This ensures visual consistency between lines and arrows
                arrow_color_for_segment = line_props["color"]  # Keep the same color as the line

            if show_arrows and coords.shape[0] > 1:
                # Add arrows between segments of the trajectory
                for i in range(coords.shape[0] - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    direction = p2 - p1
                    dist = np.linalg.norm(direction)
                    if dist < 1e-6: continue # Skip if points are virtually identical
                    unit_direction = direction / dist
                    
                    arrow_scale = 0.00825 # Quarter of previous size (was 0.033)
                    # Place arrow closer to midpoint or endpoint of segment
                    arrow_tail = p1 + 0.8 * direction # Arrow closer to p2

                    all_traces.append(create_arrows(
                        points=arrow_tail.reshape(1,3),
                        direction_vectors=unit_direction.reshape(1,3),
                        scale=arrow_scale,
                        color=arrow_color_for_segment, # Use the determined color
                        layer_names=[get_friendly_layer_name(layer) for layer in current_layers]
                    ))
                    
        # Draw cluster centers (moved here to ensure they appear on top of other elements)
        if show_cluster_centers and layer_clusters and config_name in layer_clusters:
            config_specific_clusters = layer_clusters[config_name]
            for layer in current_layers:
                if layer not in config_specific_clusters:
                    continue
                cluster_info = config_specific_clusters[layer]
                if not cluster_info or "centers" not in cluster_info or cluster_info["centers"] is None:
                    continue
                centers = cluster_info["centers"]
                if normalize:
                    orig_emb = raw_seed_data[layer]
                    if orig_emb.shape[0] > 0:
                        emb_mean = np.mean(orig_emb, axis=0)
                        emb_std = np.std(orig_emb, axis=0)
                        emb_std[emb_std == 0] = 1.0
                        centers = (centers - emb_mean) / emb_std
                layer_y_offset = current_layers.index(layer) * layer_separation
                offset_centers = centers.copy()
                offset_centers[:, 1] += layer_y_offset
                for cluster_idx, center_coord in enumerate(offset_centers):
                    center_color = QUALITATIVE_COLORS[cluster_idx % len(QUALITATIVE_COLORS)]
                    if color_by == "cluster_majority":
                        cluster_key = (config_name, layer, cluster_idx)
                        if cluster_key in cluster_to_color_map:
                            center_color = cluster_to_color_map[cluster_key]
                    if center_color == '#636EFA':
                        light_center_color = 'rgba(135, 206, 250, 0.8)'
                    elif center_color == '#EF553B':
                        light_center_color = 'rgba(255, 182, 193, 0.8)'
                    else:
                        light_center_color = center_color
                    
                    # Get the descriptive name for this cluster if available
                    cluster_key = f"{config_name}-{layer}-cluster{cluster_idx}"
                    cluster_name = f"Cluster {cluster_idx} Ctr ({config_name}-{layer})"
                    if cluster_name_mapping and cluster_key in cluster_name_mapping:
                        cluster_name = cluster_name_mapping[cluster_key]
                    
                    # Determine the majority class for this cluster if we have class labels
                    majority_class = 0  # Default class
                    if shape_by_class and isinstance(class_labels, np.ndarray) and class_labels.size > 0:
                        majority_class = get_cluster_majority_class(
                            cluster_id=cluster_idx,
                            layer=layer,
                            config_name=config_name,
                            class_labels=class_labels,
                            layer_clusters=layer_clusters
                        )
                    
                    # Set marker symbol based on majority class
                    symbol = CLASS_TO_SYMBOL.get(majority_class, "circle")
                    
                    # Add class info to the cluster name if available
                    display_name = cluster_name
                    if dataset_name:
                        class_name = get_class_name(majority_class, dataset_name)
                        display_name = f"{cluster_name} ({class_name} majority)"
                    
                    all_traces.append(go.Scatter3d(
                        x=[center_coord[0]],
                        y=[center_coord[1]],
                        z=[center_coord[2]],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=light_center_color,
                            symbol=symbol,
                            opacity=0.8,
                            line=dict(width=1, color="black")
                        ),
                        name=display_name,  # Use the descriptive name with class info
                        legendgroup=f"cluster_centers_{config_name}",
                        showlegend=True,
                        customdata=[[cluster_idx, layer, config_name, get_class_name(majority_class, dataset_name)]],
                        hovertemplate="Cluster %{customdata[0]} (%{customdata[3]} majority)<br>Layer: %{customdata[1]}<extra></extra>"
                    ))
    
    for trace in all_traces:
        fig.add_trace(trace)
        
    # Add legend entries for class symbols if shape_by_class is enabled
    if shape_by_class:
        print(f"DEBUG: Adding class symbol legend for dataset='{dataset_name}'")
        
        # Get dataset-specific class mapping if available
        class_mapping = DATASET_CLASS_NAMES.get(dataset_name, {})
        if class_mapping:
            print(f"DEBUG: Found class mapping for dataset '{dataset_name}': {class_mapping}")
        else:
            print(f"DEBUG: No class mapping found for dataset '{dataset_name}'")
        
        # Collect used classes from class_labels if available
        used_classes = set()
        if isinstance(class_labels, np.ndarray) and class_labels.size > 0:
            used_classes = set(np.unique(class_labels))
            print(f"DEBUG: Used classes from class_labels: {used_classes}")
        elif class_mapping:
            used_classes = set(class_mapping.keys())
            print(f"DEBUG: Used classes from class_mapping: {used_classes}")
        else:
            # Default to just 0 and 1 if no information available
            used_classes = {0, 1}
            print(f"DEBUG: Using default classes: {used_classes}")
        
        # Add a legend entry for each class
        for class_id in sorted(used_classes):
            # Explicitly convert class_id to int to ensure it works with the mapping
            int_class_id = int(class_id)
            
            # Force lowercase for dataset name to match dictionary keys
            dataset_key = dataset_name.lower() if dataset_name else ""
            
            # Get class name directly from mapping for reliability
            if dataset_key in DATASET_CLASS_NAMES and int_class_id in DATASET_CLASS_NAMES[dataset_key]:
                class_name = DATASET_CLASS_NAMES[dataset_key][int_class_id]
            else:
                class_name = f"Class {int_class_id}"
                
            print(f"DEBUG: Adding legend for class_id={int_class_id}, class_name='{class_name}'")
            
            symbol = CLASS_TO_SYMBOL.get(int_class_id, "circle")
            
            # Add invisible trace just for class symbol legend
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode="markers",
                marker=dict(
                    size=8,
                    color="gray",
                    symbol=symbol
                ),
                name=f"{class_name} (Shape)",
                legendgroup="class_symbols",
                showlegend=True
            ))

    # Final layout adjustments
    padding = 0.1 # 10% padding for axes ranges
    x_range = [min_x - (max_x-min_x)*padding, max_x + (max_x-min_x)*padding] if min_x != float('inf') else [-1,1]
    y_range = [min_y - (max_y-min_y)*padding, max_y + (max_y-min_y)*padding] if min_y != float('inf') else [-1,1]
    z_range = [min_z - (max_z-min_z)*padding, max_z + (max_z-min_z)*padding] if min_z != float('inf') else [-1,1]

    # Print the annotations that will be added to the scene
    print(f"DEBUG: Adding {len(scene_annotations)} annotations to the scene")
    for i, annotation in enumerate(scene_annotations):
        print(f"DEBUG: Annotation {i+1}: text='{annotation['text']}', position=({annotation['x']:.2f}, {annotation['y']:.2f}, {annotation['z']:.2f})")
    
    fig.update_layout(
        title=title,
        height=800,
        width=1000,
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2 (Layers Offset)", # Y-axis now represents UMAP-2 + Layer progression
            zaxis_title="UMAP-3",
            xaxis=dict(range=x_range, autorange=False),
            yaxis=dict(range=y_range, autorange=False),
            zaxis=dict(range=z_range, autorange=False),
            aspectmode='data',  # Changed from 'cube' to respect true data scale
            camera=dict(eye=dict(x=1.0, y=1.7, z=1.8)),  # Rotated slightly counter-clockwise
            annotations=scene_annotations
        ),
        # Update legend position to the left side of the plot
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'  # Semi-transparent background
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def plot_dataset_trajectories(
    dataset: str,
    embeddings_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    n_samples: int = 100,
    class_labels: Optional[List[Any]] = None,
    point_color_metric_name: Optional[str] = None,
    layer_class_metrics_baseline: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    layer_class_metrics_regularized: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    highlight_indices: Optional[List[int]] = None,
    show_arrows: bool = True,
    normalize: bool = True,
    layer_separation: float = LAYER_SEPARATION_OFFSET
) -> go.Figure:
    """
    Plot trajectories for a dataset using the single-scene visualization.
    
    Args:
        dataset: Name of the dataset
        embeddings_dict: Dictionary of embeddings
        n_samples: Number of samples to plot
        class_labels: List of class labels for coloring lines
        point_color_metric_name: Which metric to use for point colors ("entropy" or "angle")
        layer_class_metrics_baseline: Per-layer, per-class metrics for baseline model
        layer_class_metrics_regularized: Per-layer, per-class metrics for regularized model
        highlight_indices: Indices of samples to highlight (e.g., "fractured off" samples)
        show_arrows: Whether to show direction arrows
        normalize: Whether to normalize embeddings
        layer_separation: Distance between layers in the visualization
    """
    # Get metadata for class names if needed
    metadata = load_dataset_metadata(dataset)
    if class_labels is None and metadata and "class_labels" in metadata:
        class_labels = metadata["class_labels"]

    # Create class color mapping
    unique_classes = sorted(list(set(class_labels))) if class_labels is not None else []
    class_color_map = {
        cls: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)]
        for i, cls in enumerate(unique_classes)
    }

    # Process metric values for point colors
    point_colors_by_layer = {}
    if point_color_metric_name and (layer_class_metrics_baseline or layer_class_metrics_regularized):
        for config_name in ["baseline", "regularized"]:
            metrics_dict = layer_class_metrics_baseline if config_name == "baseline" else layer_class_metrics_regularized
            if not metrics_dict:
                continue
            
            point_colors_by_layer[config_name] = {}
            for layer_name, layer_metrics in metrics_dict.items():
                layer_colors = []
                for class_label in class_labels:
                    if str(class_label) in layer_metrics and point_color_metric_name in layer_metrics[str(class_label)]:
                        metric_value = layer_metrics[str(class_label)][point_color_metric_name]
                        layer_colors.append(metric_value)
                    else:
                        layer_colors.append(np.nan)
                point_colors_by_layer[config_name][layer_name] = np.array(layer_colors)

    # Initialize figure
    fig = go.Figure()

    # Process each configuration (baseline/regularized)
    for config_name, config_data in embeddings_dict.items():
        if not config_data:
            continue

        # Get first seed's data (assuming single seed for now)
        seed = list(config_data.keys())[0]
        layer_data = config_data[seed]

        # Determine layer order
        if not layer_data:
            continue
        layer_order = sorted(layer_data.keys())

        # Normalize embeddings if requested
        if normalize:
            layer_data = normalize_embeddings(layer_data)

        # Apply layer separation
        offset_data = {}
        for i, layer in enumerate(layer_order):
            if layer not in layer_data:
                continue
            emb = layer_data[layer].copy()
            emb[:, 1] += i * layer_separation
            offset_data[layer] = emb

        # Plot trajectories for each sample
        for sample_idx in range(min(n_samples, len(class_labels) if class_labels is not None else float('inf'))):
            # Get coordinates for this sample's trajectory
            coords = []
            point_colors = []
            
            for layer in layer_order:
                if layer in offset_data and sample_idx < offset_data[layer].shape[0]:
                    coords.append(offset_data[layer][sample_idx])
                    # Get point color from metrics if available
                    if (config_name in point_colors_by_layer and 
                        layer in point_colors_by_layer[config_name]):
                        point_colors.append(point_colors_by_layer[config_name][layer][sample_idx])
                    else:
                        point_colors.append(np.nan)

            if len(coords) < 2:
                continue

            coords = np.array(coords)
            point_colors = np.array(point_colors)

            # Determine line color from class
            line_color = "grey"
            if class_labels is not None and sample_idx < len(class_labels):
                line_color = class_color_map.get(class_labels[sample_idx], "grey")

            # Check if this sample should be highlighted
            is_highlight = highlight_indices is not None and sample_idx in highlight_indices

            # Get layer names for hover text
            layers = [layer for layer in layer_order if layer in offset_data and sample_idx < offset_data[layer].shape[0]]
            friendly_layer_names = [get_friendly_layer_name(layer) for layer in layers]
            
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode='lines+markers' if show_arrows else 'lines',
                line=dict(
                    color=line_color,
                    width=3 if is_highlight else 2,
                    dash="solid"
                ),
                marker=dict(
                    size=5 if is_highlight else 3,
                    color=point_colors if not np.all(np.isnan(point_colors)) else line_color,
                    colorscale="Viridis" if not np.all(np.isnan(point_colors)) else None,
                    showscale=not np.all(np.isnan(point_colors)),
                    colorbar=dict(title=point_color_metric_name) if not np.all(np.isnan(point_colors)) and len(layers) > 0 else None,
                    opacity=0.7
                ),
                opacity=0.7,
                line_dash="solid",
                hovertemplate=(
                    f"{config_name.title()}<br>" +
                    "Sample %{customdata}<br>" +
                    "Layer: %{text}<br>" +
                    "x: %{x:.2f}<br>" +
                    "y: %{y:.2f}<br>" +
                    "z: %{z:.2f}"
                ),
                text=friendly_layer_names,  # Use friendly layer names
                customdata=[sample_idx] * len(coords),
                showlegend=False
            ))

            # Add arrows if requested
            if show_arrows and len(coords) > 1:
                for i in range(len(coords) - 1):
                    p1, p2 = coords[i], coords[i + 1]
                    direction = p2 - p1
                    dist = np.linalg.norm(direction)
                    if dist < 1e-6:
                        continue
                    
                    unit_direction = direction / dist
                    arrow_tail = p1 + 0.8 * direction

                    fig.add_trace(create_arrows(
                        points=arrow_tail.reshape(1, 3),
                        direction_vectors=unit_direction.reshape(1, 3),
                        scale=0.00825,
                        color=line_color,
                        layer_names=[get_friendly_layer_name(layer) for layer in layers]
                    ))

        # Update layout
    fig.update_layout(
        title=f"{dataset.title()} Dataset: Neural Network Layer Trajectories",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2 (Layers Offset)",
            zaxis_title="UMAP-3",
            aspectmode='data',  # Use data proportions, don't force cubic view
            camera=dict(eye=dict(x=1.0, y=1.7, z=1.8))  # Rotated slightly counter-clockwise
        ),
        showlegend=True,
        legend=dict(
            title="Trajectories",
            x=1.1,
            y=0.5
        )
    )

    return fig

def save_figure(fig: go.Figure, output_path: str, format: str = "html"):
    """
    Save the figure to a file.
    
    Args:
        fig: Plotly Figure object.
        output_path: Path to save the figure.
        format: Output format ("html", "png", "jpeg", "pdf", "svg").
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    if format == "html":
        fig.write_html(output_path, full_html=True, include_plotlyjs='cdn')
    else:
        try:
            fig.write_image(output_path, format=format)
        except ValueError as e:
            if "Kaleido" in str(e):
                print("Error: Kaleido is not installed or not found. Please install for static image export.")
                print("pip install -U kaleido")
            else:
                raise e
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    # This is a placeholder for testing
    # In a real scenario, we would load the embeddings and plot them
    print("This module provides functionality for plotting trajectories.")
    print("Import it in your scripts to use the functions.") 