"""
GPT-2 attention head agreement visualization.

This module provides specialized visualization tools for analyzing
agreement patterns between attention heads in GPT-2 models.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import scipy.spatial.distance as distance

# Import from transformer metrics
from concept_fragmentation.metrics.transformer_metrics import calculate_cross_head_agreement


def extract_head_agreement_data(
    attention_data: Dict[str, np.ndarray],
    agreement_metric: str = "cosine",
    layer_names: Optional[List[str]] = None,
    token_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Extract agreement data between attention heads.
    
    Args:
        attention_data: Dictionary mapping layer names to attention tensors
            [batch_size, n_heads, seq_len, seq_len] or [n_heads, seq_len, seq_len]
        agreement_metric: Metric for comparing attention distributions ("cosine", "euclidean", "kl")
        layer_names: Optional list of layer names to analyze
        token_mask: Optional mask for valid tokens
        
    Returns:
        Dictionary with head agreement data
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = sorted(list(attention_data.keys()))
    
    if not layer_names:
        raise ValueError("No layers available for analysis")
    
    # Initialize result container
    agreement_data = {
        "layers": {},
        "cross_layer": {},
        "metrics": {
            "metric_type": agreement_metric,
            "layer_avg_agreement": {},
            "head_avg_agreement": {},
            "global_avg_agreement": 0.0
        }
    }
    
    # Process each layer
    total_agreement = 0.0
    agreement_count = 0
    
    for layer in layer_names:
        if layer not in attention_data:
            continue
        
        # Get attention for this layer
        attention = attention_data[layer]
        
        # Ensure 4D tensor [batch_size, n_heads, seq_len, seq_len]
        if len(attention.shape) == 3:  # [batch_size, seq_len, seq_len]
            batch_size, seq_len, _ = attention.shape
            # Skip if no head dimension
            continue
        elif len(attention.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
            batch_size, n_heads, seq_len, _ = attention.shape
        else:
            # Skip unsupported shapes
            continue
        
        # Calculate agreement between heads in this layer
        head_agreement = calculate_cross_head_agreement(
            attention,
            agreement_metric=agreement_metric,
            token_mask=token_mask
        )
        
        # Store data
        agreement_data["layers"][layer] = head_agreement
        
        # Calculate average agreement for this layer
        if head_agreement:
            layer_avg = np.mean(list(head_agreement.values()))
            agreement_data["metrics"]["layer_avg_agreement"][layer] = float(layer_avg)
            
            # Update global average
            total_agreement += sum(head_agreement.values())
            agreement_count += len(head_agreement)
    
    # Calculate agreement between heads in different layers
    for i, layer1 in enumerate(layer_names):
        for j, layer2 in enumerate(layer_names):
            if i >= j or layer1 not in attention_data or layer2 not in attention_data:
                continue
            
            # Get attention for both layers
            attention1 = attention_data[layer1]
            attention2 = attention_data[layer2]
            
            # Ensure 4D tensor [batch_size, n_heads, seq_len, seq_len]
            if len(attention1.shape) == 3 or len(attention2.shape) == 3:
                # Skip if no head dimension
                continue
            
            batch_size1, n_heads1, seq_len1, _ = attention1.shape
            batch_size2, n_heads2, seq_len2, _ = attention2.shape
            
            # Skip if dimensions don't match
            if batch_size1 != batch_size2 or seq_len1 != seq_len2:
                continue
            
            # Calculate cross-layer agreement
            cross_agreement = {}
            
            for h1 in range(n_heads1):
                for h2 in range(n_heads2):
                    # Get attention patterns
                    attn1 = attention1[:, h1]  # [batch_size, seq_len, seq_len]
                    attn2 = attention2[:, h2]  # [batch_size, seq_len, seq_len]
                    
                    # Calculate agreement
                    batch_agreements = []
                    
                    for b in range(batch_size1):
                        # Flatten attention matrices
                        flat1 = attn1[b].flatten()
                        flat2 = attn2[b].flatten()
                        
                        # Calculate agreement based on metric
                        if agreement_metric == "cosine":
                            # Cosine similarity (1 is similar, -1 is dissimilar)
                            agreement = 1 - distance.cosine(flat1, flat2)
                        elif agreement_metric == "euclidean":
                            # Convert Euclidean distance to similarity
                            # (0 is distant, 1 is similar)
                            dist = distance.euclidean(flat1, flat2)
                            agreement = 1 / (1 + dist)
                        elif agreement_metric == "kl":
                            # Calculate KL divergence (lower is more similar)
                            # Need to ensure valid probability distributions
                            p = np.clip(flat1, 1e-10, 1.0)
                            p = p / p.sum()
                            q = np.clip(flat2, 1e-10, 1.0)
                            q = q / q.sum()
                            
                            kl_div = np.sum(p * np.log(p / q))
                            # Convert to similarity (0 to 1)
                            agreement = 1 / (1 + kl_div)
                        else:
                            # Default to correlation coefficient
                            agreement = np.corrcoef(flat1, flat2)[0, 1]
                        
                        batch_agreements.append(agreement)
                    
                    # Store average agreement
                    cross_agreement[(h1, h2)] = float(np.mean(batch_agreements))
                    
                    # Update global average
                    total_agreement += float(np.mean(batch_agreements))
                    agreement_count += 1
            
            # Store cross-layer agreement
            agreement_data["cross_layer"][(layer1, layer2)] = cross_agreement
    
    # Calculate global average agreement
    if agreement_count > 0:
        agreement_data["metrics"]["global_avg_agreement"] = float(total_agreement / agreement_count)
    
    return agreement_data


def create_head_agreement_heatmap(
    agreement_data: Dict[str, Any],
    layer_name: str,
    title: Optional[str] = None,
    colorscale: str = "RdBu",
    height: int = 500,
    width: int = 500
) -> go.Figure:
    """
    Create heatmap visualization for head agreement within a layer.
    
    Args:
        agreement_data: Head agreement data from extract_head_agreement_data
        layer_name: Name of the layer to visualize
        title: Optional title for the plot
        colorscale: Colorscale for the heatmap
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the heatmap
    """
    # Check if data exists for this layer
    if layer_name not in agreement_data["layers"]:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No agreement data available for layer {layer_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get agreement data
    layer_agreement = agreement_data["layers"][layer_name]
    
    # Extract head pairs and agreement values
    head_pairs = list(layer_agreement.keys())
    agreement_values = list(layer_agreement.values())
    
    # Determine the number of heads
    head_indices = set()
    for h1, h2 in head_pairs:
        head_indices.add(h1)
        head_indices.add(h2)
    
    n_heads = max(head_indices) + 1
    
    # Create matrix for heatmap
    matrix = np.zeros((n_heads, n_heads))
    
    # Fill in agreement values
    for (h1, h2), agreement in layer_agreement.items():
        matrix[h1, h2] = agreement
        matrix[h2, h1] = agreement  # Mirror for symmetry
    
    # Set diagonal to 1.0 (each head agrees with itself)
    np.fill_diagonal(matrix, 1.0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"Head {i}" for i in range(n_heads)],
        y=[f"Head {i}" for i in range(n_heads)],
        colorscale=colorscale,
        zmid=0.5,  # Center colorscale at 0.5
        colorbar=dict(
            title="Agreement",
            titleside="right"
        ),
        hovertemplate="Head %{x}<br>Head %{y}<br>Agreement: %{z:.4f}<extra></extra>"
    ))
    
    # Add title
    if title is None:
        title = f"Attention Head Agreement - Layer {layer_name}"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        xaxis_title="Head",
        yaxis_title="Head"
    )
    
    return fig


def create_cross_layer_agreement_heatmap(
    agreement_data: Dict[str, Any],
    layer1: str,
    layer2: str,
    title: Optional[str] = None,
    colorscale: str = "RdBu",
    height: int = 500,
    width: int = 500
) -> go.Figure:
    """
    Create heatmap visualization for head agreement between two layers.
    
    Args:
        agreement_data: Head agreement data from extract_head_agreement_data
        layer1: Name of the first layer
        layer2: Name of the second layer
        title: Optional title for the plot
        colorscale: Colorscale for the heatmap
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the heatmap
    """
    # Check if data exists for these layers
    layer_pair = (layer1, layer2)
    if layer_pair not in agreement_data["cross_layer"]:
        # Try the reverse pair
        layer_pair = (layer2, layer1)
        if layer_pair not in agreement_data["cross_layer"]:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No agreement data available between layers {layer1} and {layer2}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
    
    # Get agreement data
    cross_agreement = agreement_data["cross_layer"][layer_pair]
    
    # Extract head pairs and agreement values
    head_pairs = list(cross_agreement.keys())
    agreement_values = list(cross_agreement.values())
    
    # Determine the number of heads in each layer
    heads1 = set()
    heads2 = set()
    for h1, h2 in head_pairs:
        heads1.add(h1)
        heads2.add(h2)
    
    n_heads1 = max(heads1) + 1
    n_heads2 = max(heads2) + 1
    
    # Create matrix for heatmap
    matrix = np.zeros((n_heads1, n_heads2))
    
    # Fill in agreement values
    for (h1, h2), agreement in cross_agreement.items():
        matrix[h1, h2] = agreement
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=[f"{layer2} H{i}" for i in range(n_heads2)],
        y=[f"{layer1} H{i}" for i in range(n_heads1)],
        colorscale=colorscale,
        zmid=0.5,  # Center colorscale at 0.5
        colorbar=dict(
            title="Agreement",
            titleside="right"
        ),
        hovertemplate=f"{layer1} Head %{{y}}<br>{layer2} Head %{{x}}<br>Agreement: %{{z:.4f}}<extra></extra>"
    ))
    
    # Add title
    if title is None:
        title = f"Cross-Layer Head Agreement - {layer1} vs {layer2}"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        height=height,
        width=width,
        xaxis_title=f"{layer2} Heads",
        yaxis_title=f"{layer1} Heads"
    )
    
    return fig


def create_layer_agreement_summary(
    agreement_data: Dict[str, Any],
    colorscale: str = "RdBu",
    height: int = 400,
    width: int = 700
) -> go.Figure:
    """
    Create summary visualization for agreement across all layers.
    
    Args:
        agreement_data: Head agreement data from extract_head_agreement_data
        colorscale: Colorscale for the heatmap
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the summary
    """
    # Get layer average agreement
    layer_avg = agreement_data["metrics"]["layer_avg_agreement"]
    
    if not layer_avg:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No agreement data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get layers and values
    layers = list(layer_avg.keys())
    values = list(layer_avg.values())
    
    # Create bar chart
    fig = go.Figure(data=go.Bar(
        x=layers,
        y=values,
        marker_color=values,
        marker=dict(colorscale=colorscale, colorbar=dict(title="Agreement")),
        text=[f"{v:.4f}" for v in values],
        textposition="auto",
        hovertemplate="Layer: %{x}<br>Avg Agreement: %{y:.4f}<extra></extra>"
    ))
    
    # Get global average
    global_avg = agreement_data["metrics"]["global_avg_agreement"]
    
    # Add line for global average
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=global_avg,
        x1=len(layers) - 0.5,
        y1=global_avg,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Add annotation for global average
    fig.add_annotation(
        x=len(layers) - 1,
        y=global_avg,
        text=f"Global Avg: {global_avg:.4f}",
        showarrow=False,
        yshift=10,
        font=dict(color="red")
    )
    
    # Add title and labels
    fig.update_layout(
        title_text="Average Head Agreement by Layer",
        xaxis_title="Layer",
        yaxis_title="Average Agreement",
        height=height,
        width=width,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def create_head_agreement_network(
    agreement_data: Dict[str, Any],
    threshold: float = 0.7,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """
    Create network visualization for head agreement across layers.
    
    Args:
        agreement_data: Head agreement data from extract_head_agreement_data
        threshold: Minimum agreement value to show connections
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the network
    """
    import networkx as nx
    
    # Create graph
    G = nx.Graph()
    
    # Collect all head nodes
    head_nodes = {}
    
    # Process within-layer agreements
    for layer, layer_agreement in agreement_data["layers"].items():
        for (h1, h2), agreement in layer_agreement.items():
            # Skip if below threshold or self-connection
            if agreement < threshold or h1 == h2:
                continue
            
            # Create node IDs
            node1 = f"{layer}_H{h1}"
            node2 = f"{layer}_H{h2}"
            
            # Add nodes
            if node1 not in head_nodes:
                head_nodes[node1] = {"layer": layer, "head": h1}
            if node2 not in head_nodes:
                head_nodes[node2] = {"layer": layer, "head": h2}
            
            # Add edge
            G.add_edge(node1, node2, weight=agreement)
    
    # Process cross-layer agreements
    for (layer1, layer2), cross_agreement in agreement_data["cross_layer"].items():
        for (h1, h2), agreement in cross_agreement.items():
            # Skip if below threshold
            if agreement < threshold:
                continue
            
            # Create node IDs
            node1 = f"{layer1}_H{h1}"
            node2 = f"{layer2}_H{h2}"
            
            # Add nodes
            if node1 not in head_nodes:
                head_nodes[node1] = {"layer": layer1, "head": h1}
            if node2 not in head_nodes:
                head_nodes[node2] = {"layer": layer2, "head": h2}
            
            # Add edge
            G.add_edge(node1, node2, weight=agreement)
    
    # Skip if no edges
    if len(G.edges()) == 0:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No head agreements above threshold {threshold}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Assign positions using Fruchterman-Reingold layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    edge_color = []
    
    for edge in G.edges(data=True):
        node1, node2, data = edge
        x0, y0 = pos[node1]
        x1, y1 = pos[node2]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Add edge text
        edge_text.append(f"{node1} - {node2}<br>Agreement: {data['weight']:.4f}")
        
        # Add edge color based on weight
        edge_color.extend([data["weight"]] * 3)  # Repeat for each segment
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color=edge_color, colorscale="RdBu", cmin=0, cmax=1),
        hoverinfo="none",
        mode="lines"
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    
    # Group nodes by layer
    layer_nodes = {}
    
    for node, data in head_nodes.items():
        layer = data["layer"]
        if layer not in layer_nodes:
            layer_nodes[layer] = []
        layer_nodes[layer].append(node)
    
    # Assign colors by layer
    import colorsys
    
    layer_colors = {}
    for i, layer in enumerate(sorted(layer_nodes.keys())):
        # Create distinct HSV colors
        hue = i / max(1, len(layer_nodes))
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        
        # Convert to RGB string
        color = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
        layer_colors[layer] = color
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get layer and head
        layer, head = node.split("_H")
        
        # Assign color by layer
        node_color.append(layer_colors.get(layer, "gray"))
        
        # Calculate size based on degree
        degree = G.degree(node)
        node_size.append(10 + 5 * degree)
        
        # Add node text
        node_text.append(f"{node}<br>Connections: {degree}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="black")
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Add layer legend
    for i, (layer, color) in enumerate(layer_colors.items()):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=color),
            name=layer,
            showlegend=True
        ))
    
    # Add title
    if title is None:
        title = f"Head Agreement Network (threshold={threshold})"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        showlegend=True,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        height=height,
        width=width,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig


def create_head_agreement_visualization(
    attention_data: Dict[str, np.ndarray],
    agreement_metric: str = "cosine",
    layer_names: Optional[List[str]] = None,
    token_mask: Optional[np.ndarray] = None,
    threshold: float = 0.7,
    save_html: bool = False,
    output_dir: str = "gpt2_head_agreement"
) -> Dict[str, Any]:
    """
    Create comprehensive head agreement visualizations.
    
    Args:
        attention_data: Dictionary mapping layer names to attention tensors
        agreement_metric: Metric for comparing attention distributions
        layer_names: Optional list of layer names to analyze
        token_mask: Optional mask for valid tokens
        threshold: Threshold for network visualization
        save_html: Whether to save HTML files
        output_dir: Directory for saving HTML files
        
    Returns:
        Dictionary with visualization figures
    """
    # Extract agreement data
    agreement_data = extract_head_agreement_data(
        attention_data=attention_data,
        agreement_metric=agreement_metric,
        layer_names=layer_names,
        token_mask=token_mask
    )
    
    # Use provided layers or all available
    if layer_names is None:
        layer_names = sorted(list(agreement_data["layers"].keys()))
    
    # Create visualizations
    visualizations = {
        "summary": create_layer_agreement_summary(agreement_data),
        "network": create_head_agreement_network(agreement_data, threshold=threshold),
        "layers": {},
        "cross_layers": {}
    }
    
    # Create layer heatmaps
    for layer in layer_names:
        if layer in agreement_data["layers"]:
            visualizations["layers"][layer] = create_head_agreement_heatmap(
                agreement_data=agreement_data,
                layer_name=layer
            )
    
    # Create cross-layer heatmaps
    for (layer1, layer2) in agreement_data["cross_layer"].keys():
        visualizations["cross_layers"][(layer1, layer2)] = create_cross_layer_agreement_heatmap(
            agreement_data=agreement_data,
            layer1=layer1,
            layer2=layer2
        )
    
    # Save HTML files if requested
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary visualization
        summary_path = os.path.join(output_dir, "head_agreement_summary.html")
        visualizations["summary"].write_html(summary_path)
        
        # Save network visualization
        network_path = os.path.join(output_dir, "head_agreement_network.html")
        visualizations["network"].write_html(network_path)
        
        # Save layer heatmaps
        for layer, fig in visualizations["layers"].items():
            layer_path = os.path.join(output_dir, f"head_agreement_{layer}.html")
            fig.write_html(layer_path)
        
        # Save cross-layer heatmaps
        for (layer1, layer2), fig in visualizations["cross_layers"].items():
            cross_path = os.path.join(output_dir, f"head_agreement_{layer1}_{layer2}.html")
            fig.write_html(cross_path)
    
    # Return all visualizations
    return visualizations


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    batch_size = 2
    n_heads = 4
    seq_len = 10
    n_layers = 3
    
    # Create fake attention data
    attention_data = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        
        # Random attention matrices [batch_size, n_heads, seq_len, seq_len]
        attention = np.random.rand(batch_size, n_heads, seq_len, seq_len)
        
        # Normalize across sequence dimension
        for b in range(batch_size):
            for h in range(n_heads):
                for j in range(seq_len):
                    attention[b, h, j] = attention[b, h, j] / attention[b, h, j].sum()
        
        attention_data[layer_name] = attention
    
    # Create agreement visualizations
    visualizations = create_head_agreement_visualization(
        attention_data=attention_data,
        agreement_metric="cosine",
        save_html=False
    )
    
    # Show the summary visualization
    visualizations["summary"].show()