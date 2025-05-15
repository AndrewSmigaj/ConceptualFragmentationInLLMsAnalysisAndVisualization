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
    scale: float = 0.1,
    color: str = "gray" # Can also be a numeric value if using a colorscale on the main trace
) -> go.Cone:
    """
    Create arrow cones at specified points with given direction vectors.
    
    Args:
        points: Points to place arrows at (tail of the cone).
        direction_vectors: Unit vectors indicating the direction of each cone.
        scale: Scale of the arrows (sizeref for cone).
        color: Color of the arrows.
        
    Returns:
        Plotly Cone object.
    """
    if points.shape[0] == 0 or direction_vectors.shape[0] == 0:
        # Return an empty cone trace if no points/vectors
        return go.Cone(x=[], y=[], z=[], u=[], v=[], w=[])

    # If color is numeric, it implies a colorscale is used by the parent trace.
    # Cones themselves don't directly use a numeric color with a global colorscale in the same way lines can.
    # We'll pass the color directly. If it's a number, plotly might handle it or default.
    # For simplicity, cone color will be a single color string here.
    # If dynamic cone colors are needed based on 'color' value, it would need more complex handling.
    
    actual_cone_color = color
    if not isinstance(color, str):
        # If 'color' is a number (from a colorscale value), default to a visible fixed color for arrows
        # or make it dependent on the number if you set up a specific colorscale for arrows too.
        actual_cone_color = 'grey' # Fallback for numeric color values

    return go.Cone(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        u=direction_vectors[:, 0],
        v=direction_vectors[:, 1],
        w=direction_vectors[:, 2],
        sizemode="absolute",
        sizeref=scale, # Adjust this for arrow size
        showscale=False,
        colorscale=[[0, actual_cone_color], [1, actual_cone_color]], # For single color cones
        anchor="tail",
        hoverinfo='skip' # Usually arrows don't need hoverinfo
    )

def build_single_scene_figure(
    embeddings_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    samples_to_plot: Optional[List[int]] = None,
    max_samples: int = 200,
    layer_order: Optional[List[str]] = None,
    highlight_indices: Optional[List[int]] = None,
    class_labels: Optional[np.ndarray] = None,
    sample_color_values: Optional[np.ndarray] = None, # New parameter
    color_value_name: str = "Value", # Name for the color bar
    title: str = "Neural Network Layer Trajectories (Single Scene)",
    show_arrows: bool = True,
    normalize: bool = True,
    layer_separation: float = LAYER_SEPARATION_OFFSET
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
        
    Returns:
        Plotly Figure object with a single 3D scene.
    """
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
        
        # Sort to ensure input -> layer1 -> layer2 -> ... ordering
        if "input" in raw_layer_order:
            layer_order = ["input"] + sorted([l for l in raw_layer_order if l.startswith("layer") and l != "input"]) + sorted([l for l in raw_layer_order if not l.startswith("layer") and not l.startswith("input")])
        else:
            layer_order = sorted([l for l in raw_layer_order if l.startswith("layer")]) + sorted([l for l in raw_layer_order if not l.startswith("layer")])

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
        valid_color_values = sample_color_values[~np.isnan(sample_color_values)]
        if len(valid_color_values) > 0:
            min_color_val = np.min(valid_color_values)
            max_color_val = np.max(valid_color_values)
        else: # No valid color values, disable this coloring mode
            sample_color_values = None 
            
    # Make sure max != min for colorscale
    if sample_color_values is not None: # Only adjust if we are actually using sample_color_values
        if max_color_val == min_color_val:
            max_color_val += 1e-9 # Add a tiny epsilon if all values are the same
            if max_color_val == min_color_val: # Still same (e.g. if min_color_val was 0 and epsilon was too small)
                 min_color_val -= 1e-9 # try subtracting

    for config_idx, config_name in enumerate(configs):
        config_data_per_seed = embeddings_dict[config_name]
        if not config_data_per_seed: continue
        
        seed = list(config_data_per_seed.keys())[0] # Use first available seed
        raw_seed_data = config_data_per_seed[seed]

        # Filter layers present in current seed_data and in layer_order
        current_layers = [l for l in layer_order if l in raw_seed_data]
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

                scene_annotations.append(dict(
                    showarrow=False,
                    x=layer_mean_x,
                    y= current_layer_y_base + layer_separation * 0.3, # Adjusted y for annotation relative to new offset
                    z=layer_mean_z,
                    text=f"<b>{layer.replace('layer', 'Layer ').title()}</b>",
                    xanchor="center",
                    yanchor="middle",
                    font=dict(color="black", size=12),
                    bgcolor="rgba(255, 255, 255, 0.7)"
                ))

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

        # Class colors
        point_colors_array = None
        if class_labels is not None and len(class_labels) == num_samples_in_data:
            unique_classes = sorted(list(np.unique(class_labels))) # Sort for consistent color assignment
            class_to_color_map = {
                cls_label: QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)] 
                for i, cls_label in enumerate(unique_classes)
            }
            point_colors_array = np.array([class_to_color_map.get(c, "grey") for c in class_labels])
            # Note: "grey" is a fallback if a class label somehow isn't in the map, ensure class_labels are clean.

        # Plot scatter points for each layer
        for layer in current_layers:
            data = offset_seed_data[layer]
            if data.shape[0] == 0: continue

            base_color = COLORS.get(config_name, "grey")
            scatter_colors = point_colors_array if point_colors_array is not None else base_color

            all_traces.append(go.Scatter3d(
                x=data[:, 0], y=data[:, 1], z=data[:, 2],
                mode="markers",
                marker=dict(size=3, color=scatter_colors, opacity=0.1 if highlight_set and sample_color_values is None else 0.4), # Reduce opacity if divergent lines are shown
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
                    h_colors = point_colors_array[highlight_data_indices] if point_colors_array is not None else COLORS["highlight"]
                    all_traces.append(go.Scatter3d(
                        x=h_data[:, 0], y=h_data[:, 1], z=h_data[:, 2],
                        mode="markers",
                        marker=dict(size=5, color=h_colors, opacity=0.9),
                        name=f"{config_name} - Highlights",
                        legendgroup=f"{config_name}_highlights",
                        showlegend=(config_idx==0 and layer==current_layers[0]) # Show legend only once per config for highlights
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
            elif point_colors_array is not None and sample_idx < len(point_colors_array):
                line_props["color"] = point_colors_array[sample_idx]
            else:
                line_props["color"] = COLORS.get(config_name, "grey")

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
            if use_divergent_color :
                # For arrows on a divergently colored line, they should probably match the line's color.
                # However, create_arrows expects a color string for its simple colorscale.
                # We could map numeric_color_val to a hex string from the colorscale, but that's complex.
                # For now, make arrows for divergently colored lines a distinct, neutral color or match highlight.
                arrow_color_for_segment = COLORS["highlight"] if is_highlight else "darkgrey"

            if show_arrows and coords.shape[0] > 1:
                # Add arrows between segments of the trajectory
                for i in range(coords.shape[0] - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    direction = p2 - p1
                    dist = np.linalg.norm(direction)
                    if dist < 1e-6: continue # Skip if points are virtually identical
                    unit_direction = direction / dist
                    
                    arrow_scale = 0.1 # Smaller arrows for segments
                    # Place arrow closer to midpoint or endpoint of segment
                    arrow_tail = p1 + 0.8 * direction # Arrow closer to p2

                    all_traces.append(create_arrows(
                        points=arrow_tail.reshape(1,3),
                        direction_vectors=unit_direction.reshape(1,3),
                        scale=arrow_scale,
                        color=arrow_color_for_segment # Use the determined color
                    ))
    
    for trace in all_traces:
        fig.add_trace(trace)

    # Final layout adjustments
    padding = 0.1 # 10% padding for axes ranges
    x_range = [min_x - (max_x-min_x)*padding, max_x + (max_x-min_x)*padding] if min_x != float('inf') else [-1,1]
    y_range = [min_y - (max_y-min_y)*padding, max_y + (max_y-min_y)*padding] if min_y != float('inf') else [-1,1]
    z_range = [min_z - (max_z-min_z)*padding, max_z + (max_z-min_z)*padding] if min_z != float('inf') else [-1,1]

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
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            annotations=scene_annotations
        ),
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
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

            # Create the trajectory trace
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="lines+markers",
                line=dict(
                    color=line_color,
                    width=3 if is_highlight else 2
                ),
                marker=dict(
                    size=5 if is_highlight else 3,
                    color=point_colors if not np.all(np.isnan(point_colors)) else line_color,
                    colorscale="Viridis" if not np.all(np.isnan(point_colors)) else None,
                    showscale=not np.all(np.isnan(point_colors)),
                    colorbar=dict(
                        title=point_color_metric_name,
                        thickness=15,
                        len=0.75
                    ) if not np.all(np.isnan(point_colors)) else None
                ),
                name=f"{config_name} Sample {sample_idx}",
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
                        scale=0.1,
                        color=line_color
                    ))

        # Update layout
    fig.update_layout(
        title=f"{dataset.title()} Dataset: Neural Network Layer Trajectories",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2 (Layers Offset)",
            zaxis_title="UMAP-3",
            aspectmode='data',  # Use data proportions, don't force cubic view
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
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