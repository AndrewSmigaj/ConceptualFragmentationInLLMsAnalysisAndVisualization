"""
GPT-2 attention flow Sankey diagram visualization.

This module provides specialized visualization tools for tracking attention patterns
through tokens in GPT-2 models using interactive Sankey diagrams.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import colorsys
import re
from pathlib import Path

# Import token path utilities from existing Sankey visualization
from visualization.gpt2_token_sankey import (
    DEFAULT_COLORS,
    DEFAULT_TOKEN_COLORS,
    get_token_path_stats
)

# Constants
ATTENTION_THRESHOLD = 0.05  # Minimum attention value to show in visualization
MAX_EDGES = 250  # Maximum number of attention edges to display


def extract_attention_flow(
    attention_data: Dict[str, np.ndarray],
    token_metadata: Dict[str, Any],
    min_attention: float = ATTENTION_THRESHOLD
) -> Dict[str, Any]:
    """
    Extract attention flow between tokens across layers.
    
    Args:
        attention_data: Dictionary mapping layer names to attention matrices
            [batch_size, n_heads, seq_len, seq_len] or [batch_size, seq_len, seq_len]
        token_metadata: Metadata about tokens from GPT-2 extractor
        min_attention: Minimum attention value to include
        
    Returns:
        Dictionary with attention flow information
    """
    # Extract token information
    tokens = token_metadata.get("tokens", [])
    token_ids = token_metadata.get("token_ids", [])
    attention_mask = token_metadata.get("attention_mask", [])
    
    if not tokens or not token_ids.any():
        raise ValueError("Token metadata must contain tokens and token_ids")
    
    # Get layers in order
    layers = sorted(attention_data.keys())
    
    # Prepare attention flow data structure
    attention_flow = {
        "layers": layers,
        "tokens": tokens,
        "token_ids": token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids,
        "flow_by_layer": {},
        "token_importance": {},
        "attention_stats": {}
    }
    
    # Process each layer
    for layer_idx, layer in enumerate(layers):
        if layer not in attention_data:
            continue
        
        # Get attention data for this layer
        layer_attention = attention_data[layer]
        
        # Ensure 4D tensor [batch_size, n_heads, seq_len, seq_len]
        if len(layer_attention.shape) == 3:  # [batch_size, seq_len, seq_len]
            batch_size, seq_len, _ = layer_attention.shape
            # Add dummy head dimension
            layer_attention = layer_attention.reshape(batch_size, 1, seq_len, seq_len)
        
        # Average across heads if needed
        if layer_attention.shape[1] > 1:  # Multiple heads
            layer_attention_avg = layer_attention.mean(axis=1)  # [batch_size, seq_len, seq_len]
        else:
            layer_attention_avg = layer_attention[:, 0]  # [batch_size, seq_len, seq_len]
        
        # Get dimensions
        batch_size, seq_len = token_ids.shape if hasattr(token_ids, 'shape') else (len(token_ids), len(token_ids[0]))
        
        # Initialize attention flow for this layer
        flow_data = {}
        token_importance = {}
        
        # Process each sequence in the batch
        for batch_idx in range(batch_size):
            # Get active tokens (non-padding)
            if attention_mask is not None and len(attention_mask) > batch_idx:
                active_positions = np.where(attention_mask[batch_idx] == 1)[0]
            else:
                active_positions = range(seq_len)
            
            # Process each token position
            for pos_idx in active_positions:
                token_id = token_ids[batch_idx][pos_idx]
                token_text = tokens[batch_idx][pos_idx] if tokens and batch_idx < len(tokens) else f"Token {token_id}"
                
                # Create unique identifier for this token instance
                token_key = f"batch{batch_idx}_pos{pos_idx}"
                
                # Get attention distribution for this token
                attention_dist = layer_attention_avg[batch_idx, pos_idx]
                
                # Calculate importance of this token (sum of attention it receives)
                received_attention = layer_attention_avg[batch_idx, :, pos_idx].sum()
                token_importance[token_key] = {
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "attention_received": float(received_attention),
                    "position": pos_idx
                }
                
                # Find tokens this position attends to strongly
                for target_idx in active_positions:
                    # Skip self-attention if needed
                    # if pos_idx == target_idx:
                    #     continue
                    
                    # Get attention weight
                    attention_weight = float(attention_dist[target_idx])
                    
                    # Skip if attention is below threshold
                    if attention_weight < min_attention:
                        continue
                    
                    # Get target token info
                    target_id = token_ids[batch_idx][target_idx]
                    target_text = tokens[batch_idx][target_idx] if tokens and batch_idx < len(tokens) else f"Token {target_id}"
                    target_key = f"batch{batch_idx}_pos{target_idx}"
                    
                    # Store flow data
                    edge_key = (token_key, target_key)
                    flow_data[edge_key] = {
                        "source_token": token_text,
                        "source_position": pos_idx,
                        "target_token": target_text,
                        "target_position": target_idx,
                        "attention_weight": attention_weight
                    }
        
        # Store processed flow data for this layer
        attention_flow["flow_by_layer"][layer] = flow_data
        attention_flow["token_importance"][layer] = token_importance
        
        # Calculate statistics for this layer
        total_edges = len(flow_data)
        total_attention = sum(data["attention_weight"] for data in flow_data.values())
        
        attention_flow["attention_stats"][layer] = {
            "edge_count": total_edges,
            "total_attention": float(total_attention),
            "mean_attention": float(total_attention / total_edges) if total_edges > 0 else 0,
            "max_attention": float(max([data["attention_weight"] for data in flow_data.values()])) if total_edges > 0 else 0,
            "min_attention": float(min([data["attention_weight"] for data in flow_data.values()])) if total_edges > 0 else 0
        }
    
    return attention_flow


def prepare_attention_sankey_data(
    attention_flow: Dict[str, Any],
    layer_names: Optional[List[str]] = None,
    highlight_tokens: Optional[List[str]] = None,
    max_edges: int = MAX_EDGES
) -> Dict[str, Any]:
    """
    Prepare data for attention-aware Sankey diagram.
    
    Args:
        attention_flow: Attention flow information from extract_attention_flow
        layer_names: Optional list of layer names to include (defaults to all)
        highlight_tokens: List of token keys to highlight
        max_edges: Maximum number of attention edges to display
        
    Returns:
        Dictionary with Sankey diagram data
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = attention_flow.get("layers", [])
    
    if not layer_names or len(layer_names) < 1:
        raise ValueError("At least one layer required for Sankey diagram")
    
    # Get token information
    tokens = attention_flow.get("tokens", [])
    token_ids = attention_flow.get("token_ids", [])
    
    # Create nodes for tokens in each layer
    nodes = []
    node_labels = []
    node_colors = []
    
    # Map to track node indices
    node_map = {}
    
    # Create layer-specific colors
    layer_colors = {
        layer: f"rgba({50 + i * 50}, {100 + i * 30}, {150 + i * 20}, 0.8)" 
        for i, layer in enumerate(layer_names)
    }
    
    # Create tokens to highlight
    token_colors = {}
    if highlight_tokens:
        # Limit highlight tokens
        highlight_tokens = highlight_tokens[:len(DEFAULT_TOKEN_COLORS)]
        
        # Assign colors
        for i, token in enumerate(highlight_tokens):
            token_colors[token] = DEFAULT_TOKEN_COLORS[i % len(DEFAULT_TOKEN_COLORS)]
    
    # Process each layer
    token_positions = set()
    
    # First pass: collect all token positions across layers
    for layer_idx, layer in enumerate(layer_names):
        if layer not in attention_flow["token_importance"]:
            continue
        
        layer_tokens = attention_flow["token_importance"][layer]
        
        # Add all token positions
        for token_key, token_data in layer_tokens.items():
            token_positions.add(token_key)
    
    # Second pass: create nodes for each token in each layer
    for layer_idx, layer in enumerate(layer_names):
        if layer not in attention_flow["token_importance"]:
            continue
        
        layer_tokens = attention_flow["token_importance"][layer]
        
        # Create nodes for each token position
        for token_key in sorted(token_positions):
            # Create node ID
            node_id = f"{layer}_{token_key}"
            
            # Skip if we already have this node
            if node_id in node_map:
                continue
            
            # Get token data if available
            if token_key in layer_tokens:
                token_data = layer_tokens[token_key]
                token_text = token_data["token_text"]
                position = token_data["position"]
                importance = token_data["attention_received"]
            else:
                # If token doesn't exist in this layer, use placeholder
                # (this shouldn't happen with proper data, but just in case)
                token_parts = token_key.split("_")
                if len(token_parts) > 1 and token_parts[1].startswith("pos"):
                    position = int(token_parts[1][3:])
                else:
                    position = -1
                
                token_text = f"[PAD]"
                importance = 0.0
            
            # Add node
            node_idx = len(nodes)
            node_map[node_id] = node_idx
            
            # Create label with token and position
            label = f"{layer}<br>{token_text}<br>Pos {position}"
            
            nodes.append(node_id)
            node_labels.append(label)
            
            # Determine color
            if token_key in highlight_tokens:
                node_colors.append(token_colors[token_key])
            else:
                node_colors.append(layer_colors[layer])
    
    # Create links for attention flow between layers
    sources = []
    targets = []
    values = []
    link_colors = []
    link_hovers = []
    
    # Process attention flow for consecutive layers
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Skip if we don't have flow data for either layer
        if layer1 not in attention_flow["flow_by_layer"] or layer2 not in attention_flow["flow_by_layer"]:
            continue
        
        # Get flow data
        flow_data1 = attention_flow["flow_by_layer"][layer1]
        
        # Sort by attention weight (descending)
        sorted_flows = sorted(
            flow_data1.items(), 
            key=lambda x: x[1]["attention_weight"], 
            reverse=True
        )
        
        # Limit to max_edges
        edges_added = 0
        
        for (source_key, target_key), flow_info in sorted_flows:
            if edges_added >= max_edges:
                break
            
            # Create source and target node IDs
            source_id = f"{layer1}_{source_key}"
            target_id = f"{layer2}_{target_key}"
            
            # Skip if source or target nodes don't exist in our map
            if source_id not in node_map or target_id not in node_map:
                continue
            
            # Create link
            sources.append(node_map[source_id])
            targets.append(node_map[target_id])
            values.append(flow_info["attention_weight"] * 100)  # Scale for visibility
            
            # Create hover text
            hover_text = (
                f"From: {flow_info['source_token']} (pos {flow_info['source_position']})<br>"
                f"To: {flow_info['target_token']} (pos {flow_info['target_position']})<br>"
                f"Attention: {flow_info['attention_weight']:.4f}"
            )
            
            link_hovers.append(hover_text)
            
            # Set link color
            if highlight_tokens and (source_key in highlight_tokens or target_key in highlight_tokens):
                # Use highlighted token's color
                if source_key in highlight_tokens:
                    link_colors.append(token_colors[source_key])
                else:
                    link_colors.append(token_colors[target_key])
            else:
                # Default gradient color based on attention weight
                attention = flow_info["attention_weight"]
                opacity = min(1.0, 0.3 + attention * 2)
                link_colors.append(f"rgba(100, 100, 100, {opacity})")
            
            edges_added += 1
    
    # Return prepared Sankey data
    return {
        "nodes": nodes,
        "node_labels": node_labels,
        "node_colors": node_colors,
        "sources": sources,
        "targets": targets,
        "values": values,
        "link_colors": link_colors,
        "link_hovers": link_hovers,
        "edge_count": len(sources)
    }


def generate_attention_sankey_diagram(
    attention_flow: Dict[str, Any],
    layer_names: Optional[List[str]] = None,
    highlight_tokens: Optional[List[str]] = None,
    max_edges: int = MAX_EDGES,
    title: str = "Attention Flow Between Tokens",
    height: int = 600,
    width: int = 1000
) -> go.Figure:
    """
    Generate Sankey diagram visualization for attention flow.
    
    Args:
        attention_flow: Attention flow information from extract_attention_flow
        layer_names: Optional list of layer names to include (defaults to all)
        highlight_tokens: List of token keys to highlight
        max_edges: Maximum number of attention edges to display
        title: Title for the diagram
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the Sankey diagram
    """
    # Prepare Sankey data
    sankey_data = prepare_attention_sankey_data(
        attention_flow,
        layer_names=layer_names,
        highlight_tokens=highlight_tokens,
        max_edges=max_edges
    )
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        # Define nodes
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = sankey_data["node_labels"],
            color = sankey_data["node_colors"]
        ),
        # Define links
        link = dict(
            source = sankey_data["sources"],
            target = sankey_data["targets"],
            value = sankey_data["values"],
            color = sankey_data["link_colors"],
            hovertemplate = "%{customdata}<extra></extra>",
            customdata = sankey_data["link_hovers"]
        )
    )])
    
    # Add edge count to the title
    full_title = f"{title} ({sankey_data['edge_count']} attention flows shown)"
    
    # Update layout
    fig.update_layout(
        title_text=full_title,
        font_size=12,
        height=height,
        width=width
    )
    
    return fig


def create_attention_token_comparison(
    token_paths: Dict[str, Any],
    attention_flow: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None
) -> go.Figure:
    """
    Create comparison plot between token paths and attention patterns.
    
    Args:
        token_paths: Token path information from extract_token_paths
        attention_flow: Attention flow information from extract_attention_flow
        highlight_tokens: List of token keys to highlight
        
    Returns:
        Plotly Figure object with comparison visualization
    """
    # Create comparison chart with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Token Path Distribution", "Attention Distribution"),
        vertical_spacing=0.2
    )
    
    # Get common layers
    token_layers = token_paths.get("layers", [])
    attention_layers = attention_flow.get("layers", [])
    common_layers = [layer for layer in token_layers if layer in attention_layers]
    
    if not common_layers:
        # Create empty figure with message
        fig.add_annotation(
            text="No common layers between token paths and attention data",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get or create a token-to-color mapping
    token_colors = {}
    
    if highlight_tokens:
        # Limit highlight tokens
        highlight_tokens = highlight_tokens[:len(DEFAULT_TOKEN_COLORS)]
        
        # Assign colors
        for i, token in enumerate(highlight_tokens):
            token_colors[token] = DEFAULT_TOKEN_COLORS[i % len(DEFAULT_TOKEN_COLORS)]
    
    # Sort highlighted tokens by position
    sorted_tokens = []
    
    if highlight_tokens:
        for token_key in highlight_tokens:
            # Extract position from token key
            position = -1
            if "_pos" in token_key:
                position = int(token_key.split("_pos")[1])
            
            # Get token text
            token_text = "Unknown"
            
            for layer in common_layers:
                if layer in attention_flow["token_importance"]:
                    if token_key in attention_flow["token_importance"][layer]:
                        token_text = attention_flow["token_importance"][layer][token_key]["token_text"]
                        break
            
            sorted_tokens.append((token_key, token_text, position))
        
        # Sort by position
        sorted_tokens.sort(key=lambda x: x[2])
    else:
        # If no tokens are highlighted, try to pick some interesting ones
        # based on attention received
        token_importance = {}
        
        for layer in common_layers:
            if layer in attention_flow["token_importance"]:
                for token_key, data in attention_flow["token_importance"][layer].items():
                    if token_key not in token_importance:
                        token_importance[token_key] = 0
                    token_importance[token_key] += data["attention_received"]
        
        # Pick top 5 most attention-receiving tokens
        top_tokens = sorted(
            token_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        for i, (token_key, _) in enumerate(top_tokens):
            # Get token info
            token_text = "Unknown"
            position = -1
            
            for layer in common_layers:
                if layer in attention_flow["token_importance"]:
                    if token_key in attention_flow["token_importance"][layer]:
                        token_info = attention_flow["token_importance"][layer][token_key]
                        token_text = token_info["token_text"]
                        position = token_info["position"]
                        break
            
            sorted_tokens.append((token_key, token_text, position))
            token_colors[token_key] = DEFAULT_TOKEN_COLORS[i % len(DEFAULT_TOKEN_COLORS)]
    
    # Create data for token path plot
    token_x = []
    token_y = []
    token_text = []
    token_color = []
    
    # Analyze token paths
    for token_idx, (token_key, token_name, _) in enumerate(sorted_tokens):
        # Check if token exists in token_paths
        if "paths_by_token_position" in token_paths and token_key in token_paths["paths_by_token_position"]:
            path_data = token_paths["paths_by_token_position"][token_key]
            
            # Get cluster path
            cluster_path = path_data.get("cluster_path", [])
            
            # Plot points for each layer
            for layer_idx, layer in enumerate(token_layers):
                if layer_idx < len(cluster_path):
                    cluster_id = cluster_path[layer_idx]
                    
                    # Add point for this layer/cluster
                    token_x.append(layer_idx)
                    token_y.append(token_idx)
                    token_text.append(f"Token: {token_name}<br>Layer: {layer}<br>Cluster: {cluster_id}")
                    token_color.append(token_colors.get(token_key, "gray"))
    
    # Add token path scatter plot
    if token_x:
        fig.add_trace(
            go.Scatter(
                x=token_x,
                y=token_y,
                mode="markers+lines",
                marker=dict(
                    size=12,
                    color=token_color,
                    symbol="circle"
                ),
                line=dict(
                    color="gray",
                    width=1
                ),
                text=token_text,
                hoverinfo="text",
                name="Token Paths"
            ),
            row=1, col=1
        )
    
    # Create data for attention plot
    attn_x = []
    attn_y = []
    attn_size = []
    attn_text = []
    attn_color = []
    
    # Analyze attention
    for token_idx, (token_key, token_name, _) in enumerate(sorted_tokens):
        # Plot attention received at each layer
        for layer_idx, layer in enumerate(common_layers):
            if layer in attention_flow["token_importance"]:
                if token_key in attention_flow["token_importance"][layer]:
                    token_data = attention_flow["token_importance"][layer][token_key]
                    attention = token_data["attention_received"]
                    
                    # Add point for this layer/token
                    attn_x.append(layer_idx)
                    attn_y.append(token_idx)
                    attn_size.append(attention * 30 + 5)  # Scale for visibility
                    attn_text.append(f"Token: {token_name}<br>Layer: {layer}<br>Attention: {attention:.4f}")
                    attn_color.append(token_colors.get(token_key, "gray"))
    
    # Add attention scatter plot
    if attn_x:
        fig.add_trace(
            go.Scatter(
                x=attn_x,
                y=attn_y,
                mode="markers+lines",
                marker=dict(
                    size=attn_size,
                    color=attn_color,
                    symbol="circle"
                ),
                line=dict(
                    color="gray",
                    width=1
                ),
                text=attn_text,
                hoverinfo="text",
                name="Attention Flow"
            ),
            row=2, col=1
        )
    
    # Set up axes
    # X-axis (layers)
    fig.update_xaxes(title_text="Layer Index", row=1, col=1)
    fig.update_xaxes(title_text="Layer Index", row=2, col=1)
    
    # Y-axis (tokens)
    token_labels = [f"{name} (pos {pos})" for _, name, pos in sorted_tokens]
    
    fig.update_yaxes(
        title_text="Tokens",
        tickvals=list(range(len(sorted_tokens))),
        ticktext=token_labels,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Tokens",
        tickvals=list(range(len(sorted_tokens))),
        ticktext=token_labels,
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title_text="Token Paths vs Attention Patterns",
        height=800,
        showlegend=False
    )
    
    return fig


def create_token_attention_window_visualization(
    window_data: Dict[str, Any],
    apa_results: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    min_attention: float = ATTENTION_THRESHOLD,
    max_edges: int = MAX_EDGES,
    save_html: bool = False,
    output_dir: str = "gpt2_visualizations"
) -> Dict[str, Any]:
    """
    Create window-based token attention visualizations for GPT-2.
    
    Args:
        window_data: Window data from GPT-2 analysis
        apa_results: APA analysis results
        highlight_tokens: List of token texts to highlight
        min_attention: Minimum attention value to include
        max_edges: Maximum number of attention edges to display
        save_html: Whether to save HTML files
        output_dir: Directory for saving HTML files
        
    Returns:
        Dictionary with visualization results
    """
    # Extract components
    activations = window_data.get("activations", {})
    metadata = window_data.get("metadata", {})
    window_layers = window_data.get("window_layers", [])
    
    # Extract cluster labels from APA results
    cluster_labels = {}
    if "clusters" in apa_results:
        cluster_labels = {
            layer: data.get("labels")
            for layer, data in apa_results["clusters"].items()
        }
    
    # Check if we have attention data
    attention_data = {}
    if "attention" in metadata:
        attention_data = metadata["attention"]
    
    # Check if we have token metadata
    token_metadata = {}
    if "tokens" in metadata:
        token_metadata = metadata["tokens"]
    elif "token_metadata" in metadata:
        token_metadata = metadata["token_metadata"]
    
    # Skip if no attention data or token metadata
    if not attention_data or not token_metadata:
        return {
            "error": "Missing attention data or token metadata",
            "sankey": go.Figure(),
            "comparison": go.Figure()
        }
    
    # Extract attention flow
    attention_flow = extract_attention_flow(
        attention_data=attention_data,
        token_metadata=token_metadata,
        min_attention=min_attention
    )
    
    # Process token keys to highlight
    token_keys_to_highlight = []
    
    if highlight_tokens:
        # Convert token texts to token keys
        for layer in window_layers:
            if layer in attention_flow["token_importance"]:
                for token_key, token_data in attention_flow["token_importance"][layer].items():
                    if token_data["token_text"] in highlight_tokens:
                        token_keys_to_highlight.append(token_key)
        
        # Deduplicate
        token_keys_to_highlight = list(set(token_keys_to_highlight))
    
    # Create attention Sankey diagram
    sankey_fig = generate_attention_sankey_diagram(
        attention_flow=attention_flow,
        layer_names=window_layers,
        highlight_tokens=token_keys_to_highlight,
        max_edges=max_edges,
        title=f"Attention Flow for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create comparison visualization if we have both token paths and attention
    comparison_fig = go.Figure()
    
    if activations and cluster_labels:
        # Import from token sankey
        from visualization.gpt2_token_sankey import extract_token_paths
        
        # Extract token paths
        token_paths = extract_token_paths(
            activations=activations,
            token_metadata=token_metadata,
            cluster_labels=cluster_labels
        )
        
        # Create comparison visualization
        comparison_fig = create_attention_token_comparison(
            token_paths=token_paths,
            attention_flow=attention_flow,
            highlight_tokens=token_keys_to_highlight
        )
    
    # Save HTML files if requested
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe name for window
        window_name = "_".join(window_layers)
        
        # Save Sankey diagram
        sankey_path = os.path.join(output_dir, f"attention_sankey_{window_name}.html")
        sankey_fig.write_html(sankey_path)
        
        # Save comparison visualization
        comparison_path = os.path.join(output_dir, f"attention_comparison_{window_name}.html")
        comparison_fig.write_html(comparison_path)
    
    # Return visualization results
    return {
        "sankey": sankey_fig,
        "comparison": comparison_fig,
        "stats": {
            "total_tokens": len(token_keys_to_highlight) if token_keys_to_highlight else 0,
            "layer_stats": attention_flow["attention_stats"]
        }
    }


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    batch_size = 2
    seq_len = 10
    n_layers = 3
    
    # Create fake attention data
    attention_data = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        
        # Random attention matrices [batch_size, seq_len, seq_len]
        attention = np.random.rand(batch_size, seq_len, seq_len)
        
        # Normalize across sequence dimension
        for b in range(batch_size):
            for j in range(seq_len):
                attention[b, j] = attention[b, j] / attention[b, j].sum()
        
        attention_data[layer_name] = attention
    
    # Create fake token metadata
    token_metadata = {
        "tokens": [["Token1", "Token2", "Token3", "Token4", "Token5", "Token6", "Token7", "Token8", "Token9", "Token10"] for _ in range(batch_size)],
        "token_ids": np.arange(seq_len * batch_size).reshape(batch_size, seq_len),
        "attention_mask": np.ones((batch_size, seq_len))
    }
    
    # Extract attention flow
    attention_flow = extract_attention_flow(
        attention_data=attention_data,
        token_metadata=token_metadata
    )
    
    # Create Sankey diagram
    sankey_fig = generate_attention_sankey_diagram(
        attention_flow=attention_flow,
        highlight_tokens=["batch0_pos0", "batch0_pos3", "batch1_pos5"]
    )
    
    # Show the figure
    sankey_fig.show()