"""
Similarity Network Visualization for Neural Network Layer Clusters.

This module provides visualization functions for creating interactive network visualizations
that show relationships between clusters across different layers based on similarity metrics.
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import plotly.express as px
import colorsys

def build_similarity_network(
    similarity_matrix: Dict[Tuple[int, int], float],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]],
    threshold: float = 0.5,
    max_links: int = 100
) -> nx.Graph:
    """
    Build a NetworkX graph representing the similarity network.
    
    Args:
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        id_to_layer_cluster: Mapping from unique ID to (layer_name, original_id, layer_idx)
        threshold: Minimum similarity threshold for including links
        max_links: Maximum number of links to include (to avoid overplotting)
        
    Returns:
        NetworkX graph representing the similarity network
    """
    G = nx.Graph()
    
    # Add nodes (clusters)
    for unique_id, (layer_name, original_id, layer_idx) in id_to_layer_cluster.items():
        G.add_node(
            unique_id,
            layer=layer_name,
            layer_idx=layer_idx,
            cluster_id=original_id
        )
    
    # Add edges (similarity connections) above threshold
    # Sort by similarity (descending) to include strongest connections first
    similarity_items = sorted(
        similarity_matrix.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Keep only connections above threshold, up to max_links
    link_count = 0
    for (id1, id2), similarity in similarity_items:
        if similarity >= threshold and link_count < max_links:
            # Avoid self-connections
            if id1 != id2 and id1 in id_to_layer_cluster and id2 in id_to_layer_cluster:
                G.add_edge(id1, id2, weight=similarity)
                link_count += 1
    
    return G

def compute_force_directed_layout(
    G: nx.Graph,
    iterations: int = 100
) -> Dict[int, Tuple[float, float]]:
    """
    Compute force-directed layout positions for the graph.
    
    Args:
        G: NetworkX graph
        iterations: Number of layout iterations
        
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    # Force each layer into rough vertical alignment using multipartite layout as starting point
    layer_to_nodes = {}
    for node, attrs in G.nodes(data=True):
        layer_idx = attrs.get("layer_idx", 0)
        if layer_idx not in layer_to_nodes:
            layer_to_nodes[layer_idx] = []
        layer_to_nodes[layer_idx].append(node)
    
    # Define initial positions with layers arranged horizontally
    pos_initial = {}
    max_layer = max(layer_to_nodes.keys())
    
    for layer_idx, nodes in layer_to_nodes.items():
        x = layer_idx / max(1, max_layer)  # Normalize to [0,1]
        for i, node in enumerate(nodes):
            # Position nodes within each layer with slight jitter
            y = i / max(1, len(nodes)) + np.random.normal(0, 0.01)
            pos_initial[node] = (x, y)
    
    # Run spring layout with the initial positions as a starting point
    pos = nx.spring_layout(
        G, 
        pos=pos_initial,
        iterations=iterations,
        k=0.3,  # Controls repulsion strength (higher = more spread)
        weight='weight'  # Use similarity as edge weight
    )
    
    return pos

def generate_layer_colors(layer_names: List[str]) -> Dict[str, str]:
    """
    Generate an aesthetically pleasing color palette for layers.
    
    Args:
        layer_names: List of layer names
        
    Returns:
        Dictionary mapping layer names to color strings
    """
    layer_colors = {}
    n_layers = len(layer_names)
    
    for i, layer in enumerate(layer_names):
        # Generate colors evenly spaced in HSV space
        hue = i / max(1, n_layers)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)  # Reduced saturation and increased value for lighter colors
        layer_colors[layer] = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 0.8)"
    
    return layer_colors

def create_similarity_network_plot(
    G: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]],
    height: int = 700,
    width: int = 1000,
    get_friendly_layer_name=None
) -> go.Figure:
    """
    Create a Plotly figure with the similarity network visualization.
    
    Args:
        G: NetworkX graph of the similarity network
        positions: Dictionary mapping node IDs to (x,y) positions
        id_to_layer_cluster: Mapping from unique ID to (layer_name, original_id, layer_idx)
        height: Plot height in pixels
        width: Plot width in pixels
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Plotly Figure object with the network visualization
    """
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    node_layer = []
    
    # Extract all layers for color mapping
    all_layers = set()
    for _, (layer_name, _, _) in id_to_layer_cluster.items():
        all_layers.add(layer_name)
    
    # Generate colors for layers
    layer_colors = generate_layer_colors(sorted(all_layers))
    
    # Collect node data
    for node in G.nodes():
        if node not in positions:
            continue
            
        x, y = positions[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node attributes
        attrs = G.nodes[node]
        layer = attrs.get("layer", "unknown")
        cluster_id = attrs.get("cluster_id", "?")
        layer_idx = attrs.get("layer_idx", 0)
        
        # Format node label
        if get_friendly_layer_name:
            layer_name = get_friendly_layer_name(layer)
        else:
            layer_name = layer
            
        node_text.append(f"{layer_name}, Cluster {cluster_id}")
        
        # Node size based on degree (connectivity)
        degree = G.degree(node)
        node_size.append(10 + degree * 3)  # Base size + scaling factor
        
        # Node color based on layer
        node_color.append(layer_colors.get(layer, "rgb(200,200,200)"))
        node_layer.append(layer)
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color='rgba(50,50,50,0.8)')
        ),
        name='Clusters'
    )
    
    # Prepare edge data
    edge_x = []
    edge_y = []
    edge_text = []
    edge_width = []
    edge_color = []
    
    for source, target, attrs in G.edges(data=True):
        if source not in positions or target not in positions:
            continue
            
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        
        # Add line coordinates
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge hover text
        weight = attrs.get("weight", 0)
        edge_text.append(f"Similarity: {weight:.3f}")
        
        # Edge width based on similarity - ensure minimum width of 1
        width = max(1, weight * 5)  # Scale for visibility, minimum width of 1
        edge_width.extend([width, width, width])
        
        # Edge color based on similarity
        # Gradient from light gray (low) to dark blue (high)
        if weight < 0.5:
            alpha = 0.3 + weight * 0.6  # 0.3 to 0.6
            edge_color.extend([f"rgba(150,150,150,{alpha})", f"rgba(150,150,150,{alpha})", f"rgba(150,150,150,{alpha})"])
        else:
            alpha = 0.6 + (weight - 0.5) * 0.8  # 0.6 to 1.0
            edge_color.extend([f"rgba(70,130,180,{alpha})", f"rgba(70,130,180,{alpha})", f"rgba(70,130,180,{alpha})"])
    
    # Create edge traces - one for each edge to handle varying width/color
    edge_traces = []
    
    # Check if we have edge data
    if edge_x and edge_y:
        # Group edge data into triplets (each edge is 3 points: start, end, None)
        for i in range(0, len(edge_x), 3):
            if i+2 < len(edge_x):  # Ensure we have a complete triplet
                # Create individual trace for this edge
                edge_trace = go.Scatter(
                    x=edge_x[i:i+3],
                    y=edge_y[i:i+3],
                    line=dict(
                        width=max(1, edge_width[i] if i < len(edge_width) else 1),
                        color=edge_color[i] if i < len(edge_color) else "rgba(150,150,150,0.3)"
                    ),
                    hoverinfo='text',
                    text=edge_text[i//3] if i//3 < len(edge_text) else "",
                    mode='lines',
                    showlegend=False  # Don't show individual edges in legend
                )
                edge_traces.append(edge_trace)
    
    # Add a dummy trace for the legend
    if edge_traces:
        legend_edge_trace = go.Scatter(
            x=[None],
            y=[None],
            line=dict(width=2, color="rgba(70,130,180,0.8)"),  # 2px width is valid
            mode='lines',
            name='Similarity'
        )
        edge_traces.append(legend_edge_trace)
    else:
        # Empty edge trace as fallback
        edge_traces = [go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name='Similarity (none found)'
        )]
    
    # Create legend for layers
    legend_traces = []
    for layer in sorted(all_layers):
        if get_friendly_layer_name:
            layer_name = get_friendly_layer_name(layer)
        else:
            layer_name = layer
            
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=layer_colors.get(layer, "rgb(200,200,200)")
                ),
                name=layer_name
            )
        )
    
    # Create figure with all traces
    fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
    
    # Update layout
    fig.update_layout(
        title="Cluster Similarity Network",
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=max(600, height),  # Ensure minimum height
        width=max(1000, width),   # Ensure minimum width of 1000px (exceeds Plotly's 10px minimum)
        legend=dict(
            title="Layers",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bordercolor="Black",
            borderwidth=1
        )
    )
    
    return fig

def create_top_similar_clusters_table(
    top_similar_clusters: Dict[int, List[Tuple[int, float]]],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]],
    get_friendly_layer_name=None
) -> pd.DataFrame:
    """
    Create a DataFrame with the most similar clusters for each cluster.
    
    Args:
        top_similar_clusters: Dictionary mapping cluster IDs to list of (similar_cluster, similarity) tuples
        id_to_layer_cluster: Mapping from unique ID to (layer_name, original_id, layer_idx)
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        DataFrame with top similar clusters information
    """
    rows = []
    
    for cluster_id, similar_clusters in top_similar_clusters.items():
        if cluster_id not in id_to_layer_cluster:
            continue
            
        # Get current cluster info
        layer_name, original_id, layer_idx = id_to_layer_cluster[cluster_id]
        
        # Format current cluster name
        if get_friendly_layer_name:
            current_layer = get_friendly_layer_name(layer_name)
        else:
            current_layer = layer_name
            
        current_cluster = f"{current_layer} Cluster {original_id}"
        
        # Process similar clusters
        for similar_id, similarity in similar_clusters:
            if similar_id not in id_to_layer_cluster:
                continue
                
            # Get similar cluster info
            sim_layer_name, sim_original_id, sim_layer_idx = id_to_layer_cluster[similar_id]
            
            # Format similar cluster name
            if get_friendly_layer_name:
                similar_layer = get_friendly_layer_name(sim_layer_name)
            else:
                similar_layer = sim_layer_name
                
            similar_cluster = f"{similar_layer} Cluster {sim_original_id}"
            
            # Add row to results
            rows.append({
                "Source Cluster": current_cluster,
                "Similar Cluster": similar_cluster,
                "Similarity": similarity,
                "Source Layer": current_layer,
                "Similar Layer": similar_layer
            })
    
    # Create DataFrame and sort by similarity (descending)
    if rows:
        df = pd.DataFrame(rows)
        return df.sort_values("Similarity", ascending=False)
    else:
        return pd.DataFrame(columns=["Source Cluster", "Similar Cluster", "Similarity", "Source Layer", "Similar Layer"])

def create_convergent_paths_summary(
    convergent_paths: Dict[int, List[Dict[str, Any]]],
    human_readable_paths: Dict[str, List[Dict[str, Any]]],
    fragmentation_scores: Dict[str, float]
) -> pd.DataFrame:
    """
    Create a DataFrame summarizing similarity-convergent paths.
    
    Args:
        convergent_paths: Dictionary mapping path indices to list of convergences
        human_readable_paths: Dictionary mapping path strings to list of convergences
        fragmentation_scores: Dictionary mapping path indices to fragmentation scores
        
    Returns:
        DataFrame with convergent paths summary
    """
    rows = []
    
    # Process paths with numeric IDs first
    for path_idx, convergences in convergent_paths.items():
        path_idx_str = str(path_idx)
        
        # Get fragmentation score if available
        fragmentation = fragmentation_scores.get(path_idx_str, None)
        
        # Get number of convergences
        num_convergences = len(convergences)
        
        # Get average similarity of convergences
        avg_similarity = sum(c["similarity"] for c in convergences) / max(1, num_convergences)
        
        # Get the strongest convergence
        if convergences:
            strongest = max(convergences, key=lambda c: c["similarity"])
            strongest_similarity = strongest["similarity"]
            early_layer = strongest["early_layer"]
            late_layer = strongest["late_layer"]
            layer_distance = late_layer - early_layer
        else:
            strongest_similarity = 0
            early_layer = 0
            late_layer = 0
            layer_distance = 0
        
        # Add row
        rows.append({
            "Path ID": path_idx,
            "Fragmentation Score": fragmentation,
            "Number of Convergences": num_convergences,
            "Average Similarity": avg_similarity,
            "Strongest Similarity": strongest_similarity,
            "Layer Distance": layer_distance
        })
    
    # Add any human-readable paths (if not already covered)
    for path_str, convergences in human_readable_paths.items():
        # Check if we've already processed this path
        if any(r.get("Path String") == path_str for r in rows):
            continue
            
        # Get number of convergences
        num_convergences = len(convergences)
        
        # Get average similarity of convergences
        avg_similarity = sum(c["similarity"] for c in convergences) / max(1, num_convergences)
        
        # Get the strongest convergence
        if convergences:
            strongest = max(convergences, key=lambda c: c["similarity"])
            strongest_similarity = strongest["similarity"]
        else:
            strongest_similarity = 0
        
        # Add row
        rows.append({
            "Path String": path_str,
            "Number of Convergences": num_convergences,
            "Average Similarity": avg_similarity,
            "Strongest Similarity": strongest_similarity
        })
    
    # Create DataFrame and sort by strongest similarity (descending)
    if rows:
        df = pd.DataFrame(rows)
        if "Strongest Similarity" in df.columns:
            return df.sort_values("Strongest Similarity", ascending=False)
        return df
    else:
        return pd.DataFrame(columns=["Path ID", "Path String", "Fragmentation Score", 
                                    "Number of Convergences", "Average Similarity", "Strongest Similarity"])