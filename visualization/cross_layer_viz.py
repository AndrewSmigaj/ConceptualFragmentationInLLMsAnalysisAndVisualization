"""
Visualization module for Cross-Layer Metrics in the concept fragmentation project.

This module provides visualization functions for various cross-layer metrics
such as centroid similarity, membership overlap, trajectory fragmentation,
and inter-cluster path density.
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import colorsys
import re

def plot_centroid_similarity_heatmap(
    similarity_matrices: Dict[Tuple[str, str], np.ndarray],
    layer_pairs: Optional[List[Tuple[str, str]]] = None,
    colorscale: str = "Viridis",
    height: int = 600,
    width: int = 800,
    get_friendly_layer_name=None
) -> go.Figure:
    """
    Create a heatmap visualization of centroid similarity between layers.
    
    Args:
        similarity_matrices: Dictionary mapping layer pairs to similarity matrices
                             from compute_centroid_similarity
        layer_pairs: Optional list of layer pairs to include (if None, use all)
        colorscale: Plotly colorscale to use
        height: Plot height in pixels
        width: Plot width in pixels
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Plotly Figure object with heatmap
    """
    if not similarity_matrices:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No centroid similarity data available",
            height=height,
            width=width
        )
        return fig
    
    # Use specified layer pairs or all available pairs
    if layer_pairs is None:
        # Ensure keys are proper tuples
        layer_pairs = []
        for key in similarity_matrices.keys():
            if isinstance(key, tuple) and len(key) == 2:
                layer_pairs.append(key)
    
    # Create subplots for each layer pair
    n_pairs = len(layer_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    # Create subplot titles
    subplot_titles = []
    for layer1, layer2 in layer_pairs:
        if get_friendly_layer_name:
            layer1_name = get_friendly_layer_name(layer1)
            layer2_name = get_friendly_layer_name(layer2)
        else:
            layer1_name = layer1
            layer2_name = layer2
        
        subplot_titles.append(f"{layer1_name} to {layer2_name}")
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add heatmaps for each layer pair
    for i, (layer1, layer2) in enumerate(layer_pairs):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if (layer1, layer2) not in similarity_matrices:
            continue
            
        sim_matrix = similarity_matrices[(layer1, layer2)]
        
        # Get cluster counts for each layer
        n_clusters1 = sim_matrix.shape[0]
        n_clusters2 = sim_matrix.shape[1]
        
        # Create x and y axis labels
        if get_friendly_layer_name:
            layer1_name = get_friendly_layer_name(layer1)
            layer2_name = get_friendly_layer_name(layer2)
        else:
            layer1_name = layer1
            layer2_name = layer2
            
        x_labels = [f"{layer2_name}<br>Cluster {j}" for j in range(n_clusters2)]
        y_labels = [f"{layer1_name}<br>Cluster {j}" for j in range(n_clusters1)]
        
        # Add heatmap
        heatmap = go.Heatmap(
            z=sim_matrix,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            colorbar=dict(
                thickness=15,
                title="Similarity",
                len=0.3,
                y=0.5
            ) if i == 0 else None,  # Only show colorbar for first heatmap
            hovertemplate=(
                "From: %{y}<br>" +
                "To: %{x}<br>" +
                "Similarity: %{z:.3f}<extra></extra>"
            )
        )
        
        fig.add_trace(heatmap, row=row, col=col)
        
        # Update axes
        fig.update_xaxes(
            title=None,
            tickangle=45,
            showgrid=False,
            row=row,
            col=col
        )
        fig.update_yaxes(
            title=None,
            showgrid=False,
            row=row,
            col=col
        )
    
    # Update layout
    fig.update_layout(
        title="Centroid Similarity Across Layers",
        height=height * n_rows / 2,
        width=width,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def extract_rgb(rgba_str):
    """Extract RGB values from an rgba string."""
    # Use regex to find the three numbers after "rgba("
    match = re.search(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', rgba_str)
    if match:
        return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
    else:
        # Fallback to a default color if parsing fails
        return [128, 128, 128]  # Gray

def plot_membership_overlap_sankey(
    overlap_matrices: Dict[Tuple[str, str], np.ndarray],
    layer_clusters: Dict[str, Dict[str, Any]],
    min_overlap: float = 0.1,
    height: int = 600,
    width: int = 1000,
    get_friendly_layer_name=None
) -> go.Figure:
    """
    Create a Sankey diagram showing cluster membership overlap between layers.
    
    Args:
        overlap_matrices: Dictionary mapping layer pairs to overlap matrices
                         from compute_membership_overlap
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries
        min_overlap: Minimum overlap required to show a connection
        height: Plot height in pixels
        width: Plot width in pixels
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Plotly Figure object with Sankey diagram
    """
    if not overlap_matrices:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No membership overlap data available",
            height=height,
            width=width
        )
        return fig
    
    # Get all layer names in the correct order
    layer_names = sorted(layer_clusters.keys())
    
    # Initialize Sankey diagram data
    node_labels = []
    node_colors = []
    link_sources = []
    link_targets = []
    link_values = []
    link_colors = []
    
    # Create color map for layers
    n_layers = len(layer_names)
    layer_to_color = {}
    
    for i, layer in enumerate(layer_names):
        # Generate colors evenly spaced in HSV space
        hue = i / max(1, n_layers)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        layer_to_color[layer] = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
    
    # Create nodes for each cluster in each layer
    node_index = {}  # Map (layer, cluster_id) to node index
    
    for layer_idx, layer in enumerate(layer_names):
        if "labels" not in layer_clusters[layer]:
            continue
            
        unique_clusters = np.unique(layer_clusters[layer]["labels"])
        
        for cluster_id in unique_clusters:
            # Get friendly name
            if get_friendly_layer_name:
                layer_name = get_friendly_layer_name(layer)
            else:
                layer_name = layer
                
            # Add node
            node_labels.append(f"{layer_name}<br>Cluster {cluster_id}")
            node_colors.append(layer_to_color[layer])
            
            # Map to index
            node_index[(layer, cluster_id)] = len(node_labels) - 1
    
    # Create links between adjacent layers
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Skip if this layer pair isn't in overlap matrices
        if (layer1, layer2) not in overlap_matrices:
            continue
            
        overlap = overlap_matrices[(layer1, layer2)]
        
        # Create links for each cluster pair with sufficient overlap
        for c1_idx, c1 in enumerate(np.unique(layer_clusters[layer1]["labels"])):
            for c2_idx, c2 in enumerate(np.unique(layer_clusters[layer2]["labels"])):
                if c1_idx < overlap.shape[0] and c2_idx < overlap.shape[1]:
                    overlap_value = overlap[c1_idx, c2_idx]
                    
                    if overlap_value >= min_overlap and (layer1, c1) in node_index and (layer2, c2) in node_index:
                        # Add link
                        link_sources.append(node_index[(layer1, c1)])
                        link_targets.append(node_index[(layer2, c2)])
                        link_values.append(overlap_value * 100)  # Scale for visibility
                        
                        # Blend colors of source and target nodes
                        color1 = layer_to_color[layer1]
                        color2 = layer_to_color[layer2]
                        
                        # Extract RGB values using the helper function
                        
                        # Extract RGB values
                        rgb1 = extract_rgb(color1)
                        rgb2 = extract_rgb(color2)
                        
                        # Blend the colors (average of each component)
                        blended_r = (rgb1[0] + rgb2[0]) // 2
                        blended_g = (rgb1[1] + rgb2[1]) // 2
                        blended_b = (rgb1[2] + rgb2[2]) // 2
                        
                        # Create blended color string
                        link_colors.append(f"rgba({blended_r}, {blended_g}, {blended_b}, 0.5)")
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color=link_colors
        )
    )])
    
    # Update layout
    fig.update_layout(
        title="Cluster Membership Flow Across Layers",
        height=height,
        width=width,
        font=dict(size=10),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def plot_trajectory_fragmentation_bars(
    fragmentation_scores: Dict[str, float],
    height: int = 400,
    width: int = 800,
    get_friendly_layer_name=None
) -> go.Figure:
    """
    Create a bar chart showing trajectory fragmentation by layer.
    
    Args:
        fragmentation_scores: Dictionary mapping layer names to fragmentation scores
                              from compute_trajectory_fragmentation
        height: Plot height in pixels
        width: Plot width in pixels
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Plotly Figure object with bar chart
    """
    if not fragmentation_scores:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No trajectory fragmentation data available",
            height=height,
            width=width
        )
        return fig
    
    # Sort layers for consistent ordering
    layers = sorted(fragmentation_scores.keys())
    scores = [fragmentation_scores[layer] for layer in layers]
    
    # Convert layer names to friendly names if function provided
    if get_friendly_layer_name:
        layer_names = [get_friendly_layer_name(layer) for layer in layers]
    else:
        layer_names = layers
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=layer_names,
            y=scores,
            marker_color='rgb(55, 83, 109)',
            text=[f"{score:.3f}" for score in scores],
            textposition='auto'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Trajectory Fragmentation by Layer",
        xaxis=dict(title="Layer"),
        yaxis=dict(
            title="Fragmentation Score",
            range=[0, 1],
            tickformat=".2f"
        ),
        height=height,
        width=width
    )
    
    return fig

def plot_path_density_network(
    path_graph: nx.Graph,
    layout: str = "multipartite",
    height: int = 600,
    width: int = 800,
    get_friendly_layer_name=None
) -> go.Figure:
    """
    Create a network visualization of inter-cluster path density.
    
    Args:
        path_graph: NetworkX graph from compute_path_density
        layout: Graph layout ('spring', 'circular', 'multipartite')
        height: Plot height in pixels
        width: Plot width in pixels
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Plotly Figure object with network graph
    """
    if not path_graph or path_graph.number_of_nodes() == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No path density data available",
            height=height,
            width=width
        )
        return fig
    
    # Get positions based on specified layout
    if layout == "multipartite":
        # Group nodes by layer
        layer_to_nodes = {}
        for node, attrs in path_graph.nodes(data=True):
            layer = attrs.get("layer", "unknown")
            if layer not in layer_to_nodes:
                layer_to_nodes[layer] = []
            layer_to_nodes[layer].append(node)
        
        # Sort layers
        sorted_layers = sorted(layer_to_nodes.keys())
        
        # Position nodes in layers
        pos = {}
        for layer_idx, layer in enumerate(sorted_layers):
            nodes = layer_to_nodes[layer]
            for node_idx, node in enumerate(nodes):
                pos[node] = (layer_idx, node_idx - len(nodes)/2)
    elif layout == "circular":
        pos = nx.circular_layout(path_graph)
    else:  # Default to spring layout
        pos = nx.spring_layout(path_graph)
    
    # Prepare node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node, attrs in path_graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Create node label
        layer = attrs.get("layer", "unknown")
        cluster_id = attrs.get("cluster_id", "?")
        
        if get_friendly_layer_name:
            layer_name = get_friendly_layer_name(layer)
        else:
            layer_name = layer
            
        node_text.append(f"{layer_name}, Cluster {cluster_id}")
        
        # Color nodes by layer
        # Convert layer name to a numeric value for coloring
        try:
            if layer.startswith("layer"):
                layer_num = int(layer.replace("layer", ""))
            else:
                layer_num = sum(ord(c) for c in layer)
            node_color.append(layer_num)
        except:
            node_color.append(0)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=15,
            colorbar=dict(
                thickness=15,
                title='Layer',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=1, color='black')
        )
    )
    
    # Prepare edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    edge_width = []
    
    for source, target, attrs in path_graph.edges(data=True):
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Add line coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge width based on weight
        weight = attrs.get("weight", 0.5)
        edge_width.append(weight * 5)
        
        # Edge hover text
        edge_text.append(f"Overlap: {weight:.3f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.7)'),
        hoverinfo='text',
        text=edge_text,
        mode='lines'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title="Inter-Cluster Path Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=height,
        width=width
    )
    
    return fig

def create_cross_layer_dashboard(
    cl_metrics: Dict[str, Any],
    layer_clusters: Dict[str, Dict[str, Any]],
    get_friendly_layer_name=None,
    min_overlap: float = 0.1
) -> Dict[str, go.Figure]:
    """
    Create a comprehensive dashboard of all cross-layer metrics.
    
    Args:
        cl_metrics: Dictionary of cross-layer metrics from analyze_cross_layer_metrics
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries
        get_friendly_layer_name: Optional function to convert layer names to display names
        min_overlap: Minimum overlap threshold for connections
        
    Returns:
        Dictionary mapping plot names to Plotly figure objects
    """
    figures = {}
    
    # Create centroid similarity heatmap
    if "centroid_similarity" in cl_metrics and not cl_metrics.get("centroid_similarity_error"):
        figures["centroid_similarity"] = plot_centroid_similarity_heatmap(
            cl_metrics["centroid_similarity"],
            get_friendly_layer_name=get_friendly_layer_name,
            height=600,
            width=1000
        )
    
    # Create membership overlap Sankey diagram
    if "membership_overlap" in cl_metrics and not cl_metrics.get("membership_overlap_error"):
        figures["membership_overlap"] = plot_membership_overlap_sankey(
            cl_metrics["membership_overlap"],
            layer_clusters,
            min_overlap=min_overlap,
            get_friendly_layer_name=get_friendly_layer_name,
            height=600,
            width=1000
        )
    
    # Create trajectory fragmentation bar chart
    if "trajectory_fragmentation" in cl_metrics and not cl_metrics.get("trajectory_fragmentation_error"):
        figures["trajectory_fragmentation"] = plot_trajectory_fragmentation_bars(
            cl_metrics["trajectory_fragmentation"],
            get_friendly_layer_name=get_friendly_layer_name,
            height=400,
            width=800
        )
    
    # Create path density network
    if "path_graph" in cl_metrics and not cl_metrics.get("path_density_error"):
        figures["path_density"] = plot_path_density_network(
            cl_metrics["path_graph"],
            layout="multipartite",
            get_friendly_layer_name=get_friendly_layer_name,
            height=600,
            width=1000
        )
    
    return figures