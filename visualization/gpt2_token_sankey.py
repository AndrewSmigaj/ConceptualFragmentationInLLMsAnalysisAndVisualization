"""
GPT-2 token path Sankey diagram visualization.

This module provides specialized visualization tools for tracking token paths
through clusters in a 3-layer window of GPT-2 models using interactive Sankey diagrams.
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

# Define constants
DEFAULT_COLORS = px.colors.qualitative.Plotly  # Default Plotly color sequence
DEFAULT_TOKEN_COLORS = px.colors.qualitative.Bold  # Vibrant colors for token highlighting


def extract_token_paths(
    activations: Dict[str, np.ndarray],
    token_metadata: Dict[str, Any],
    cluster_labels: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """
    Extract token-specific paths through clusters.
    
    Args:
        activations: Dictionary mapping layer names to activations
        token_metadata: Metadata about tokens from GPT-2 extractor
        cluster_labels: Dictionary mapping layer names to cluster labels
        
    Returns:
        Dictionary with token path information
    """
    # Extract token information
    tokens = token_metadata.get("tokens", [])
    token_ids = token_metadata.get("token_ids", [])
    attention_mask = token_metadata.get("attention_mask", [])
    
    if not tokens or not token_ids.any():
        raise ValueError("Token metadata must contain tokens and token_ids")
    
    # Get layers in order
    layers = sorted(cluster_labels.keys())
    
    # Prepare token paths data structure
    token_paths = {
        "layers": layers,
        "tokens": tokens,
        "token_ids": token_ids.tolist() if hasattr(token_ids, 'tolist') else token_ids,
        "paths_by_token": {},
        "paths_by_token_position": {},
        "token_features": {}
    }
    
    # Track paths for each token
    batch_size, seq_len = token_ids.shape if hasattr(token_ids, 'shape') else (len(token_ids), len(token_ids[0]))
    
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
            
            # Initialize path for this token
            token_paths["paths_by_token_position"][token_key] = {
                "token_id": int(token_id),
                "token_text": token_text,
                "batch_idx": batch_idx,
                "position": pos_idx,
                "cluster_path": []
            }
            
            # Extract features for this token across layers
            token_features = {}
            
            # Track cluster path across layers
            cluster_path = []
            
            for layer in layers:
                if layer not in cluster_labels:
                    continue
                
                # Get cluster label for this token
                flattened_idx = batch_idx * seq_len + pos_idx
                if flattened_idx < len(cluster_labels[layer]):
                    cluster_id = int(cluster_labels[layer][flattened_idx])
                    cluster_path.append(cluster_id)
                    
                    # Store activation features
                    if layer in activations:
                        token_features[layer] = activations[layer][batch_idx, pos_idx].tolist()
                else:
                    # Handle case where token is out of bounds
                    cluster_path.append(-1)  # -1 indicates no cluster assigned
            
            # Store cluster path
            token_paths["paths_by_token_position"][token_key]["cluster_path"] = cluster_path
            
            # Group by actual token text (combines token instances)
            if token_text not in token_paths["paths_by_token"]:
                token_paths["paths_by_token"][token_text] = {
                    "token_id": int(token_id),
                    "occurrences": [],
                    "cluster_paths": []
                }
            
            # Add this occurrence
            token_paths["paths_by_token"][token_text]["occurrences"].append(token_key)
            token_paths["paths_by_token"][token_text]["cluster_paths"].append(cluster_path)
            
            # Store token features
            token_paths["token_features"][token_key] = token_features
    
    return token_paths


def prepare_token_sankey_data(
    token_paths: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    min_path_count: int = 1,
    max_tokens: int = 50
) -> Dict[str, Any]:
    """
    Prepare data for token-aware Sankey diagram.
    
    Args:
        token_paths: Token path information from extract_token_paths
        highlight_tokens: List of token keys to highlight
        min_path_count: Minimum number of tokens to include a path
        max_tokens: Maximum number of tokens to display
        
    Returns:
        Dictionary with Sankey diagram data
    """
    layers = token_paths.get("layers", [])
    if not layers or len(layers) < 2:
        raise ValueError("At least two layers required for Sankey diagram")
    
    # Create node lists - one for each layer's clusters
    nodes = []
    node_labels = []
    node_colors = []
    
    # Map to track node indices
    node_map = {}
    
    # Create layer-specific colors
    layer_colors = {
        layer: f"rgba({50 + i * 50}, {100 + i * 30}, {150 + i * 20}, 0.8)" 
        for i, layer in enumerate(layers)
    }
    
    # Process each layer
    max_clusters = 0
    for layer_idx, layer in enumerate(layers):
        # Find all clusters in this layer
        all_clusters = set()
        
        # Get cluster IDs from all token paths
        for token_key, token_data in token_paths["paths_by_token_position"].items():
            cluster_path = token_data["cluster_path"]
            if layer_idx < len(cluster_path):
                cluster_id = cluster_path[layer_idx]
                if cluster_id >= 0:  # Skip -1 (no cluster)
                    all_clusters.add(cluster_id)
        
        # Update max clusters
        max_clusters = max(max_clusters, max(all_clusters, default=0) + 1)
        
        # Add nodes for each cluster
        for cluster_id in sorted(all_clusters):
            # Create node ID
            node_id = f"{layer}_C{cluster_id}"
            node_idx = len(nodes)
            node_map[node_id] = node_idx
            
            # Add node
            nodes.append(node_id)
            node_labels.append(f"{layer}<br>Cluster {cluster_id}")
            node_colors.append(layer_colors[layer])
    
    # Prepare token highlight colors
    token_colors = {}
    if highlight_tokens:
        # Limit highlight tokens
        highlight_tokens = highlight_tokens[:len(DEFAULT_TOKEN_COLORS)]
        
        # Assign colors
        for i, token in enumerate(highlight_tokens):
            token_colors[token] = DEFAULT_TOKEN_COLORS[i % len(DEFAULT_TOKEN_COLORS)]
    
    # Track path counts between clusters
    path_counts = {}
    token_links = {}
    
    # Process paths for each token
    token_count = 0
    for token_key, token_data in list(token_paths["paths_by_token_position"].items())[:max_tokens]:
        cluster_path = token_data["cluster_path"]
        
        # Skip tokens with incomplete paths
        if len(cluster_path) < len(layers) or -1 in cluster_path:
            continue
        
        token_count += 1
        token_text = token_data["token_text"]
        batch_idx = token_data["batch_idx"]
        position = token_data["position"]
        
        # Process connections between layers
        for layer_idx in range(len(layers) - 1):
            layer1 = layers[layer_idx]
            layer2 = layers[layer_idx + 1]
            
            # Get clusters for consecutive layers
            cluster1 = cluster_path[layer_idx]
            cluster2 = cluster_path[layer_idx + 1]
            
            # Create source and target node IDs
            source_id = f"{layer1}_C{cluster1}"
            target_id = f"{layer2}_C{cluster2}"
            
            # Skip if nodes don't exist (should not happen if data is consistent)
            if source_id not in node_map or target_id not in node_map:
                continue
                
            # Create link key
            link_key = (source_id, target_id)
            
            # Initialize path count if needed
            if link_key not in path_counts:
                path_counts[link_key] = 0
                token_links[link_key] = []
            
            # Increment count
            path_counts[link_key] += 1
            
            # Add token to this link
            token_links[link_key].append({
                "token_key": token_key,
                "token_text": token_text,
                "batch_idx": batch_idx,
                "position": position
            })
    
    # Create links
    sources = []
    targets = []
    values = []
    link_colors = []
    link_tokens = []
    link_hovers = []
    
    # Filter paths by count and create links
    for link_key, count in path_counts.items():
        if count >= min_path_count:
            source_id, target_id = link_key
            source_idx = node_map[source_id]
            target_idx = node_map[target_id]
            
            # Get tokens for this link
            tokens_list = token_links[link_key]
            token_texts = [t["token_text"] for t in tokens_list]
            
            # Create link
            sources.append(source_idx)
            targets.append(target_idx)
            values.append(count)
            
            # Create hover text with token list
            if len(token_texts) <= 10:
                hover_text = f"Tokens: {', '.join(token_texts)}"
            else:
                hover_text = f"Tokens: {', '.join(token_texts[:10])}... ({len(token_texts)} total)"
            
            link_hovers.append(hover_text)
            link_tokens.append(token_texts)
            
            # Set link color based on highlighted tokens
            if highlight_tokens:
                # Check if any highlighted token is in this link
                highlighted = False
                for token_info in tokens_list:
                    token_key = token_info["token_key"]
                    if token_key in highlight_tokens:
                        link_colors.append(token_colors[token_key])
                        highlighted = True
                        break
                
                if not highlighted:
                    link_colors.append("rgba(200, 200, 200, 0.3)")  # Faded for non-highlighted
            else:
                # Default coloring
                link_colors.append("rgba(150, 150, 150, 0.5)")
    
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
        "link_tokens": link_tokens,
        "token_count": token_count,
        "path_counts": path_counts
    }


def generate_token_sankey_diagram(
    token_paths: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    min_path_count: int = 1,
    max_tokens: int = 50,
    title: str = "Token Path Flow Through Clusters",
    height: int = 600,
    width: int = 1000
) -> go.Figure:
    """
    Generate Sankey diagram visualization for token paths.
    
    Args:
        token_paths: Token path information from extract_token_paths
        highlight_tokens: List of token keys to highlight
        min_path_count: Minimum number of tokens to include a path
        max_tokens: Maximum number of tokens to display
        title: Title for the diagram
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the Sankey diagram
    """
    # Prepare Sankey data
    sankey_data = prepare_token_sankey_data(
        token_paths,
        highlight_tokens=highlight_tokens,
        min_path_count=min_path_count,
        max_tokens=max_tokens
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
    
    # Add token count to the title
    full_title = f"{title} ({sankey_data['token_count']} tokens shown)"
    
    # Update layout
    fig.update_layout(
        title_text=full_title,
        font_size=12,
        height=height,
        width=width
    )
    
    return fig


def get_token_path_stats(token_paths: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate statistics for token paths.
    
    Args:
        token_paths: Token path information from extract_token_paths
        
    Returns:
        Dictionary with token path statistics
    """
    stats = {
        "total_tokens": 0,
        "unique_tokens": 0,
        "paths": {
            "total": 0,
            "unique": 0
        },
        "tokens_per_cluster": {},
        "path_diversity": {},
        "most_common_paths": [],
        "most_fragmented_tokens": []
    }
    
    # Get layers
    layers = token_paths.get("layers", [])
    if not layers:
        return stats
    
    # Count tokens
    stats["total_tokens"] = len(token_paths["paths_by_token_position"])
    stats["unique_tokens"] = len(token_paths["paths_by_token"])
    
    # Count paths
    all_paths = []
    path_counts = {}
    
    # Process each token position
    for token_key, token_data in token_paths["paths_by_token_position"].items():
        cluster_path = tuple(token_data["cluster_path"])
        
        # Count this path
        if cluster_path not in path_counts:
            path_counts[cluster_path] = 0
        path_counts[cluster_path] += 1
        
        all_paths.append(cluster_path)
        
        # Count tokens per cluster
        for layer_idx, cluster_id in enumerate(cluster_path):
            if cluster_id < 0:  # Skip -1 (no cluster)
                continue
                
            layer = layers[layer_idx]
            
            # Initialize layer in stats if needed
            if layer not in stats["tokens_per_cluster"]:
                stats["tokens_per_cluster"][layer] = {}
            
            # Initialize cluster in layer if needed
            if cluster_id not in stats["tokens_per_cluster"][layer]:
                stats["tokens_per_cluster"][layer][cluster_id] = 0
            
            # Increment count
            stats["tokens_per_cluster"][layer][cluster_id] += 1
    
    # Set path counts
    stats["paths"]["total"] = len(all_paths)
    stats["paths"]["unique"] = len(path_counts)
    
    # Calculate path diversity by layer
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        
        # Get unique clusters in this layer
        clusters_in_layer = set()
        for path in all_paths:
            if layer_idx < len(path):
                cluster_id = path[layer_idx]
                if cluster_id >= 0:  # Skip -1 (no cluster)
                    clusters_in_layer.add(cluster_id)
        
        # Store diversity
        stats["path_diversity"][layer] = {
            "unique_clusters": len(clusters_in_layer),
            "clusters": sorted(list(clusters_in_layer))
        }
    
    # Calculate path fragmentation for each token
    token_fragmentation = {}
    for token_text, token_data in token_paths["paths_by_token"].items():
        # Skip tokens with only one occurrence
        if len(token_data["occurrences"]) <= 1:
            continue
            
        # Get all paths for this token
        paths = token_data["cluster_paths"]
        
        # Calculate fragmentation score based on how many different paths this token takes
        unique_paths = len(set(tuple(path) for path in paths))
        fragmentation_score = (unique_paths - 1) / max(1, len(paths) - 1)
        
        token_fragmentation[token_text] = {
            "token": token_text,
            "occurrences": len(token_data["occurrences"]),
            "unique_paths": unique_paths,
            "fragmentation_score": fragmentation_score
        }
    
    # Find most common paths
    most_common = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    stats["most_common_paths"] = [
        {"path": list(path), "count": count}
        for path, count in most_common
    ]
    
    # Find most fragmented tokens
    most_fragmented = sorted(
        token_fragmentation.values(), 
        key=lambda x: (x["fragmentation_score"], x["occurrences"]), 
        reverse=True
    )[:10]
    stats["most_fragmented_tokens"] = most_fragmented
    
    return stats


def create_token_path_comparison(
    token_paths: Dict[str, Any],
    tokens_to_compare: List[str],
    max_tokens: int = 5,
    height: int = 800,
    width: int = 1000
) -> go.Figure:
    """
    Create a visual comparison of paths for different tokens.
    
    Args:
        token_paths: Token path information from extract_token_paths
        tokens_to_compare: List of tokens to compare
        max_tokens: Maximum number of tokens to display
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with token path comparison
    """
    # Limit number of tokens
    tokens_to_compare = tokens_to_compare[:max_tokens]
    
    # Get number of tokens and layers
    n_tokens = len(tokens_to_compare)
    layers = token_paths.get("layers", [])
    n_layers = len(layers)
    
    if n_tokens == 0 or n_layers == 0:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No token data available for comparison",
            height=height,
            width=width
        )
        return fig
    
    # Create subplots
    fig = make_subplots(
        rows=n_tokens,
        cols=1,
        subplot_titles=[f"Token: '{token}'" for token in tokens_to_compare],
        vertical_spacing=0.05
    )
    
    # Create colors for layers
    layer_colors = px.colors.qualitative.Plotly[:n_layers]
    
    # Process each token
    for i, token in enumerate(tokens_to_compare):
        # Find all occurrences of this token
        token_instances = []
        for token_key, token_data in token_paths["paths_by_token_position"].items():
            if token_data["token_text"] == token:
                token_instances.append({
                    "key": token_key,
                    "path": token_data["cluster_path"],
                    "position": token_data["position"]
                })
        
        # Skip if no instances found
        if not token_instances:
            continue
        
        # Sort by position
        token_instances.sort(key=lambda x: x["position"])
        
        # Create a line for each token instance
        for j, instance in enumerate(token_instances):
            path = instance["path"]
            
            # Create x and y values
            x = list(range(len(path)))
            y = path
            
            # Add line
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=f"{token} (pos {instance['position']})",
                    line=dict(
                        width=2,
                        dash="solid" if j % 2 == 0 else "dot"
                    ),
                    marker=dict(
                        size=8,
                        symbol="circle" if j % 2 == 0 else "square"
                    ),
                    legendgroup=f"token_{i}",
                    showlegend=False,
                    hovertemplate=f"Token: {token}<br>Position: {instance['position']}<br>Layer: %{{x}}<br>Cluster: %{{y}}"
                ),
                row=i+1,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        title="Token Path Comparison",
        height=max(300, 200 * n_tokens),
        width=width
    )
    
    # Update x and y axes
    for i in range(1, n_tokens + 1):
        fig.update_xaxes(
            title="Layer",
            tickvals=list(range(n_layers)),
            ticktext=layers,
            row=i,
            col=1
        )
        fig.update_yaxes(
            title="Cluster ID",
            row=i,
            col=1
        )
    
    return fig


def create_3layer_window_sankey(
    window_data: Dict[str, Any],
    analysis_result: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    min_path_count: int = 1,
    output_dir: Optional[str] = None,
    title: str = "GPT-2 Token Path Flow (3-Layer Window)",
    save_html: bool = True
) -> Dict[str, go.Figure]:
    """
    Create a 3-layer window Sankey diagram for GPT-2 token paths.
    
    Args:
        window_data: Window data from GPT2ActivationExtractor
        analysis_result: Result of APA analysis
        highlight_tokens: Optional list of tokens to highlight
        min_path_count: Minimum number of tokens to include a path
        output_dir: Optional directory to save visualizations
        title: Title for the main diagram
        save_html: Whether to save the figures as HTML files
        
    Returns:
        Dictionary of figure objects
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract activations and metadata
    activations = window_data.get("activations", {})
    metadata = window_data.get("metadata", {})
    
    # Get cluster labels
    cluster_labels = {}
    for layer, layer_data in analysis_result.get("clusters", {}).items():
        if "labels" in layer_data:
            cluster_labels[layer] = layer_data["labels"]
    
    # Extract token paths
    token_paths = extract_token_paths(activations, metadata, cluster_labels)
    
    # Generate main Sankey diagram
    sankey_fig = generate_token_sankey_diagram(
        token_paths,
        highlight_tokens=highlight_tokens,
        min_path_count=min_path_count,
        title=title
    )
    
    # Get token path statistics
    stats = get_token_path_stats(token_paths)
    
    # Find most fragmented tokens if no highlight tokens specified
    if not highlight_tokens and stats["most_fragmented_tokens"]:
        highlight_tokens = [t["token"] for t in stats["most_fragmented_tokens"][:5]]
    
    # Create token path comparison
    comparison_fig = create_token_path_comparison(
        token_paths,
        tokens_to_compare=highlight_tokens if highlight_tokens else []
    )
    
    # Save figures if requested
    if save_html and output_dir:
        sankey_path = os.path.join(output_dir, "token_sankey_diagram.html")
        comparison_path = os.path.join(output_dir, "token_path_comparison.html")
        
        sankey_fig.write_html(sankey_path)
        comparison_fig.write_html(comparison_path)
        
        # Save token path statistics
        stats_path = os.path.join(output_dir, "token_path_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    # Return all figures
    return {
        "sankey": sankey_fig,
        "comparison": comparison_fig,
        "stats": stats
    }


# Example usage function
def example_usage():
    """
    Example usage for the token Sankey diagram visualization.
    """
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate token Sankey diagrams for GPT-2 analysis")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory with APA results")
    parser.add_argument("--window", type=str, default=None, help="Specific window to visualize (default: first available)")
    parser.add_argument("--output_dir", type=str, default="./token_visualizations", help="Output directory")
    parser.add_argument("--min_paths", type=int, default=1, help="Minimum number of tokens to include a path")
    parser.add_argument("--tokens", type=str, nargs='+', default=None, help="Specific tokens to highlight")
    
    args = parser.parse_args()
    
    # Find metadata file
    input_path = Path(args.input_dir)
    metadata_files = list(input_path.glob("*_metadata.json"))
    
    if not metadata_files:
        print(f"No metadata files found in {args.input_dir}")
        return
    
    # Load metadata
    with open(metadata_files[0], 'r') as f:
        metadata = json.load(f)
    
    # Get available windows
    windows = metadata.get("layer_files", {}).keys()
    
    if not windows:
        print(f"No windows found in metadata")
        return
    
    # Select window
    window_name = args.window if args.window and args.window in windows else list(windows)[0]
    
    # Load window data
    window_dir = input_path / window_name
    window_metadata_file = window_dir / f"{window_name}_metadata.json"
    
    if not window_metadata_file.exists():
        print(f"Window metadata file not found: {window_metadata_file}")
        return
    
    with open(window_metadata_file, 'r') as f:
        window_metadata = json.load(f)
    
    # Load activations
    window_activations = {}
    for layer_name, layer_file in metadata["layer_files"][window_name].items():
        layer_path = Path(layer_file)
        if layer_path.exists():
            window_activations[layer_name] = np.load(layer_path)
    
    window_data = {
        "activations": window_activations,
        "metadata": window_metadata.get("metadata", {}),
        "window_layers": window_metadata.get("window_layers", [])
    }
    
    # Load APA results
    results_dir = input_path / "results" / window_name
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Load cluster labels
    analysis_result = {"clusters": {}}
    for layer_name in window_activations.keys():
        labels_file = results_dir / f"{layer_name}_labels.npy"
        if labels_file.exists():
            analysis_result["clusters"][layer_name] = {
                "labels": np.load(labels_file)
            }
    
    # Create output directory
    output_dir = Path(args.output_dir) / window_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_3layer_window_sankey(
        window_data,
        analysis_result,
        highlight_tokens=args.tokens,
        min_path_count=args.min_paths,
        output_dir=str(output_dir),
        title=f"GPT-2 Token Path Flow: {window_name}"
    )
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    example_usage()