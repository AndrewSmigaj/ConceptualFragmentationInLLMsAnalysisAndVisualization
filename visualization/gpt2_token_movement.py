"""
GPT-2 token movement metrics and visualization.

This module provides tools for analyzing token movement patterns through
layers, calculating metrics about token trajectories, and creating
visualizations that show how tokens evolve across the model.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from collections import defaultdict
import scipy.spatial.distance as distance

# Import token path utilities
from visualization.gpt2_token_sankey import extract_token_paths, get_token_path_stats


def calculate_token_movement_metrics(
    token_paths: Dict[str, Any],
    activations: Dict[str, np.ndarray],
    layer_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate metrics about token movement across layers.
    
    Args:
        token_paths: Token path information from extract_token_paths
        activations: Dictionary mapping layer names to activations
        layer_names: Optional list of layer names to analyze
        
    Returns:
        Dictionary with token movement metrics
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = token_paths.get("layers", [])
    
    if not layer_names or len(layer_names) < 2:
        raise ValueError("At least two layers required for movement analysis")
    
    # Initialize result container
    movement_metrics = {
        "layers": layer_names,
        "token_metrics": {},
        "global_metrics": {
            "avg_path_length": 0.0,
            "avg_cluster_changes": 0.0,
            "avg_activation_change": 0.0,
            "avg_activation_velocity": 0.0
        }
    }
    
    # Extract paths
    if "paths_by_token_position" not in token_paths:
        return movement_metrics
    
    # Calculate metrics for each token
    total_path_length = 0.0
    total_cluster_changes = 0.0
    total_activation_change = 0.0
    total_activation_velocity = 0.0
    token_count = 0
    
    for token_key, token_data in token_paths["paths_by_token_position"].items():
        # Extract token information
        token_text = token_data.get("token_text", "")
        token_id = token_data.get("token_id", -1)
        position = token_data.get("position", -1)
        batch_idx = token_data.get("batch_idx", 0)
        cluster_path = token_data.get("cluster_path", [])
        
        # Skip if path is too short
        if len(cluster_path) < len(layer_names):
            continue
        
        # Initialize metrics for this token
        token_metrics = {
            "token_text": token_text,
            "token_id": token_id,
            "position": position,
            "cluster_path": cluster_path,
            "layer_metrics": {}
        }
        
        # Calculate Euclidean path length
        path_length = 0.0
        cluster_changes = 0
        activation_changes = []
        activation_velocities = []
        
        # Get token features for each layer
        layer_features = {}
        
        for layer_idx, layer in enumerate(layer_names):
            if layer in activations and batch_idx < activations[layer].shape[0] and position < activations[layer].shape[1]:
                layer_features[layer] = activations[layer][batch_idx, position]
        
        # Calculate metrics between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1 = layer_names[i]
            layer2 = layer_names[i + 1]
            
            # Get clusters for consecutive layers
            cluster1 = cluster_path[i] if i < len(cluster_path) else -1
            cluster2 = cluster_path[i + 1] if i + 1 < len(cluster_path) else -1
            
            # Calculate cluster change
            cluster_change = 0 if cluster1 == cluster2 else 1
            cluster_changes += cluster_change
            
            # Calculate activation change if features available
            activation_change = 0.0
            activation_velocity = 0.0
            
            if layer1 in layer_features and layer2 in layer_features:
                feat1 = layer_features[layer1]
                feat2 = layer_features[layer2]
                
                # Calculate Euclidean distance between features
                activation_change = np.linalg.norm(feat2 - feat1)
                path_length += activation_change
                
                # Calculate activation velocity (change per dimension)
                activation_velocity = activation_change / max(1, len(feat1))
            
            # Store layer metrics
            token_metrics["layer_metrics"][(layer1, layer2)] = {
                "source_cluster": int(cluster1),
                "target_cluster": int(cluster2),
                "cluster_change": cluster_change,
                "activation_change": float(activation_change),
                "activation_velocity": float(activation_velocity)
            }
            
            # Add to arrays for summary
            activation_changes.append(activation_change)
            activation_velocities.append(activation_velocity)
        
        # Store path summary metrics
        token_metrics["path_length"] = float(path_length)
        token_metrics["cluster_changes"] = int(cluster_changes)
        token_metrics["avg_activation_change"] = float(np.mean(activation_changes)) if activation_changes else 0.0
        token_metrics["avg_activation_velocity"] = float(np.mean(activation_velocities)) if activation_velocities else 0.0
        
        # Store token metrics
        movement_metrics["token_metrics"][token_key] = token_metrics
        
        # Update global totals
        total_path_length += path_length
        total_cluster_changes += cluster_changes
        total_activation_change += np.mean(activation_changes) if activation_changes else 0.0
        total_activation_velocity += np.mean(activation_velocities) if activation_velocities else 0.0
        token_count += 1
    
    # Calculate global averages
    if token_count > 0:
        movement_metrics["global_metrics"]["avg_path_length"] = float(total_path_length / token_count)
        movement_metrics["global_metrics"]["avg_cluster_changes"] = float(total_cluster_changes / token_count)
        movement_metrics["global_metrics"]["avg_activation_change"] = float(total_activation_change / token_count)
        movement_metrics["global_metrics"]["avg_activation_velocity"] = float(total_activation_velocity / token_count)
    
    # Calculate most and least mobile tokens
    token_mobility = []
    
    for token_key, metrics in movement_metrics["token_metrics"].items():
        # Calculate normalized mobility score
        # Combines path length, cluster changes, and activation velocity
        path_length = metrics["path_length"]
        cluster_changes = metrics["cluster_changes"]
        activation_velocity = metrics["avg_activation_velocity"]
        
        # Normalize by global averages
        norm_path_length = path_length / movement_metrics["global_metrics"]["avg_path_length"] if movement_metrics["global_metrics"]["avg_path_length"] > 0 else 0
        norm_cluster_changes = cluster_changes / movement_metrics["global_metrics"]["avg_cluster_changes"] if movement_metrics["global_metrics"]["avg_cluster_changes"] > 0 else 0
        norm_velocity = activation_velocity / movement_metrics["global_metrics"]["avg_activation_velocity"] if movement_metrics["global_metrics"]["avg_activation_velocity"] > 0 else 0
        
        # Combined mobility score
        mobility_score = (norm_path_length + norm_cluster_changes + norm_velocity) / 3
        
        token_mobility.append({
            "token_key": token_key,
            "token_text": metrics["token_text"],
            "position": metrics["position"],
            "mobility_score": mobility_score,
            "path_length": path_length,
            "cluster_changes": cluster_changes,
            "activation_velocity": activation_velocity
        })
    
    # Sort by mobility score (descending)
    token_mobility.sort(key=lambda x: x["mobility_score"], reverse=True)
    
    # Store most mobile tokens
    movement_metrics["most_mobile_tokens"] = token_mobility[:20]
    movement_metrics["least_mobile_tokens"] = token_mobility[-20:]
    
    return movement_metrics


def create_token_trajectory_plot(
    movement_metrics: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """
    Create visualization of token trajectories across layers.
    
    Args:
        movement_metrics: Movement metrics from calculate_token_movement_metrics
        highlight_tokens: List of token keys to highlight
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the trajectory plot
    """
    # Get layer names
    layer_names = movement_metrics.get("layers", [])
    
    if not layer_names or len(layer_names) < 2:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough layers for trajectory visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get token metrics
    token_metrics = movement_metrics.get("token_metrics", {})
    
    if not token_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No token metrics available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Choose tokens to highlight if not provided
    if highlight_tokens is None:
        # Use most and least mobile tokens
        most_mobile = movement_metrics.get("most_mobile_tokens", [])[:5]
        least_mobile = movement_metrics.get("least_mobile_tokens", [])[-5:]
        
        highlight_tokens = [t["token_key"] for t in most_mobile + least_mobile]
    
    # Limit highlight tokens
    highlight_tokens = highlight_tokens[:10]
    
    # Create color map for highlighted tokens
    token_colors = {}
    color_palette = px.colors.qualitative.Set1
    
    for i, token_key in enumerate(highlight_tokens):
        token_colors[token_key] = color_palette[i % len(color_palette)]
    
    # Add background traces for all tokens
    for token_key, metrics in token_metrics.items():
        # Skip highlighted tokens
        if token_key in highlight_tokens:
            continue
        
        # Get token information
        token_text = metrics.get("token_text", "")
        position = metrics.get("position", -1)
        cluster_path = metrics.get("cluster_path", [])
        
        # Create x values for plotting
        x = list(range(len(layer_names)))
        
        # Create y values (cluster IDs)
        y = cluster_path[:len(layer_names)]
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(
                color="rgba(200, 200, 200, 0.3)",
                width=1
            ),
            showlegend=False,
            hoverinfo="skip"
        ))
    
    # Add traces for highlighted tokens
    for token_key in highlight_tokens:
        # Skip if token not in metrics
        if token_key not in token_metrics:
            continue
        
        # Get token information
        metrics = token_metrics[token_key]
        token_text = metrics.get("token_text", "")
        position = metrics.get("position", -1)
        cluster_path = metrics.get("cluster_path", [])
        path_length = metrics.get("path_length", 0.0)
        mobility_score = next((t["mobility_score"] for t in movement_metrics.get("most_mobile_tokens", []) + movement_metrics.get("least_mobile_tokens", []) if t["token_key"] == token_key), 0.0)
        
        # Create x values for plotting
        x = list(range(len(layer_names)))
        
        # Create y values (cluster IDs)
        y = cluster_path[:len(layer_names)]
        
        # Create hover text
        hover_text = []
        for i, layer in enumerate(layer_names):
            if i < len(cluster_path):
                cluster_id = cluster_path[i]
                
                # Get layer-specific metrics
                activation_change = 0.0
                activation_velocity = 0.0
                
                if i > 0:
                    prev_layer = layer_names[i - 1]
                    layer_pair = (prev_layer, layer)
                    
                    if layer_pair in metrics.get("layer_metrics", {}):
                        layer_metrics = metrics["layer_metrics"][layer_pair]
                        activation_change = layer_metrics.get("activation_change", 0.0)
                        activation_velocity = layer_metrics.get("activation_velocity", 0.0)
                
                hover_text.append(
                    f"Token: {token_text} (pos {position})<br>"
                    f"Layer: {layer}<br>"
                    f"Cluster: {cluster_id}<br>"
                    f"Activation Change: {activation_change:.4f}<br>"
                    f"Activation Velocity: {activation_velocity:.4f}<br>"
                    f"Mobility Score: {mobility_score:.4f}"
                )
            else:
                hover_text.append(f"Token: {token_text}<br>Layer: {layer}<br>No cluster data")
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            marker=dict(
                size=8,
                color=token_colors.get(token_key, "gray")
            ),
            line=dict(
                color=token_colors.get(token_key, "gray"),
                width=2
            ),
            name=f"{token_text} (pos {position})",
            text=hover_text,
            hoverinfo="text"
        ))
    
    # Add title
    if title is None:
        title = "Token Trajectories Across Layers"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Layer Index",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(layer_names))),
            ticktext=layer_names
        ),
        yaxis_title="Cluster ID",
        height=height,
        width=width,
        legend_title="Tokens"
    )
    
    return fig


def create_token_velocity_heatmap(
    movement_metrics: Dict[str, Any],
    metric_type: str = "activation_velocity",
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """
    Create heatmap visualization of token movement metrics.
    
    Args:
        movement_metrics: Movement metrics from calculate_token_movement_metrics
        metric_type: Type of metric to visualize (activation_velocity, activation_change, cluster_change)
        title: Optional title for the plot
        colorscale: Colorscale for the heatmap
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the heatmap
    """
    # Get layer names
    layer_names = movement_metrics.get("layers", [])
    
    if not layer_names or len(layer_names) < 2:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough layers for velocity visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get token metrics
    token_metrics = movement_metrics.get("token_metrics", {})
    
    if not token_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No token metrics available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create layer pairs
    layer_pairs = []
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        layer_pairs.append((layer1, layer2))
    
    # Get tokens in order of position
    tokens_by_position = sorted(
        list(token_metrics.items()),
        key=lambda x: x[1].get("position", -1)
    )
    
    # Extract token information
    token_keys = [tk for tk, _ in tokens_by_position]
    token_texts = [tm.get("token_text", "") for _, tm in tokens_by_position]
    token_positions = [tm.get("position", -1) for _, tm in tokens_by_position]
    
    # Create labels for display
    token_labels = [f"{text} (pos {pos})" for text, pos in zip(token_texts, token_positions)]
    layer_pair_labels = [f"{l1}-{l2}" for l1, l2 in layer_pairs]
    
    # Create matrix for heatmap
    matrix = np.zeros((len(token_keys), len(layer_pairs)))
    
    # Fill in matrix with selected metric
    for i, (token_key, metrics) in enumerate(tokens_by_position):
        for j, layer_pair in enumerate(layer_pairs):
            # Get layer metrics
            if layer_pair in metrics.get("layer_metrics", {}):
                layer_metrics = metrics["layer_metrics"][layer_pair]
                
                # Get requested metric
                if metric_type == "activation_velocity":
                    value = layer_metrics.get("activation_velocity", 0.0)
                elif metric_type == "activation_change":
                    value = layer_metrics.get("activation_change", 0.0)
                elif metric_type == "cluster_change":
                    value = layer_metrics.get("cluster_change", 0)
                else:
                    value = 0.0
                
                matrix[i, j] = value
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=layer_pair_labels,
        y=token_labels,
        colorscale=colorscale,
        colorbar=dict(
            title=metric_type.replace("_", " ").title()
        ),
        hovertemplate="Token: %{y}<br>Layer Transition: %{x}<br>Value: %{z:.4f}<extra></extra>"
    ))
    
    # Add title
    if title is None:
        title = f"Token {metric_type.replace('_', ' ').title()} Across Layers"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Layer Transition",
        yaxis_title="Token",
        height=height,
        width=width
    )
    
    return fig


def create_mobility_ranking_plot(
    movement_metrics: Dict[str, Any],
    top_n: int = 10,
    title: Optional[str] = None,
    height: int = 600,
    width: int = 800
) -> go.Figure:
    """
    Create bar chart of token mobility rankings.
    
    Args:
        movement_metrics: Movement metrics from calculate_token_movement_metrics
        top_n: Number of top/bottom tokens to display
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the bar chart
    """
    # Get most and least mobile tokens
    most_mobile = movement_metrics.get("most_mobile_tokens", [])
    least_mobile = movement_metrics.get("least_mobile_tokens", [])
    
    if not most_mobile or not least_mobile:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No mobility data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Limit to top_n
    most_mobile = most_mobile[:top_n]
    least_mobile = least_mobile[-top_n:]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Most Mobile Tokens", "Least Mobile Tokens"),
        shared_yaxes=False
    )
    
    # Add most mobile tokens
    most_mobile_labels = [f"{t['token_text']} (pos {t['position']})" for t in most_mobile]
    most_mobile_scores = [t["mobility_score"] for t in most_mobile]
    
    fig.add_trace(
        go.Bar(
            y=most_mobile_labels,
            x=most_mobile_scores,
            orientation="h",
            marker_color="red",
            text=[f"{score:.4f}" for score in most_mobile_scores],
            textposition="auto",
            hovertemplate="Token: %{y}<br>Mobility Score: %{x:.4f}<br><extra></extra>"
        ),
        row=1, col=1
    )
    
    # Add least mobile tokens
    least_mobile_labels = [f"{t['token_text']} (pos {t['position']})" for t in least_mobile]
    least_mobile_scores = [t["mobility_score"] for t in least_mobile]
    
    fig.add_trace(
        go.Bar(
            y=least_mobile_labels,
            x=least_mobile_scores,
            orientation="h",
            marker_color="blue",
            text=[f"{score:.4f}" for score in least_mobile_scores],
            textposition="auto",
            hovertemplate="Token: %{y}<br>Mobility Score: %{x:.4f}<br><extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add mobility score details as annotations
    global_metrics = movement_metrics.get("global_metrics", {})
    
    detail_text = (
        f"Avg Path Length: {global_metrics.get('avg_path_length', 0.0):.4f}<br>"
        f"Avg Cluster Changes: {global_metrics.get('avg_cluster_changes', 0.0):.4f}<br>"
        f"Avg Activation Change: {global_metrics.get('avg_activation_change', 0.0):.4f}<br>"
        f"Avg Activation Velocity: {global_metrics.get('avg_activation_velocity', 0.0):.4f}"
    )
    
    fig.add_annotation(
        text=detail_text,
        xref="paper", yref="paper",
        x=0.5, y=1.05,
        showarrow=False,
        font=dict(size=12),
        align="center"
    )
    
    # Add title
    if title is None:
        title = "Token Mobility Rankings"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        height=height,
        width=width
    )
    
    fig.update_xaxes(title_text="Mobility Score", row=1, col=1)
    fig.update_xaxes(title_text="Mobility Score", row=1, col=2)
    fig.update_yaxes(title_text="Token", row=1, col=1)
    fig.update_yaxes(title_text="Token", row=1, col=2)
    
    return fig


def create_token_movement_visualization(
    window_data: Dict[str, Any],
    apa_results: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    top_n: int = 10,
    save_html: bool = False,
    output_dir: str = "gpt2_visualizations"
) -> Dict[str, Any]:
    """
    Create comprehensive token movement visualizations.
    
    Args:
        window_data: Window data from GPT-2 analysis
        apa_results: APA analysis results
        highlight_tokens: List of token texts to highlight
        top_n: Number of top tokens to display
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
    
    # Check if we have token metadata
    token_metadata = {}
    if "tokens" in metadata:
        token_metadata = metadata["tokens"]
    elif "token_metadata" in metadata:
        token_metadata = metadata["token_metadata"]
    
    # Skip if missing required data
    if not activations or not token_metadata or not cluster_labels:
        return {
            "error": "Missing required data for movement analysis",
            "trajectory_plot": go.Figure(),
            "velocity_heatmap": go.Figure(),
            "mobility_ranking": go.Figure()
        }
    
    # Extract token paths
    token_paths = extract_token_paths(
        activations=activations,
        token_metadata=token_metadata,
        cluster_labels=cluster_labels
    )
    
    # Calculate token movement metrics
    movement_metrics = calculate_token_movement_metrics(
        token_paths=token_paths,
        activations=activations,
        layer_names=window_layers
    )
    
    # Process token keys to highlight (if provided)
    token_keys_to_highlight = []
    
    if highlight_tokens:
        # Convert token texts to token keys
        for token_key, metrics in movement_metrics["token_metrics"].items():
            if metrics["token_text"] in highlight_tokens:
                token_keys_to_highlight.append(token_key)
    
    # Create trajectory plot
    trajectory_plot = create_token_trajectory_plot(
        movement_metrics=movement_metrics,
        highlight_tokens=token_keys_to_highlight,
        title=f"Token Trajectories for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create velocity heatmap
    velocity_heatmap = create_token_velocity_heatmap(
        movement_metrics=movement_metrics,
        metric_type="activation_velocity",
        title=f"Token Velocity for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create mobility ranking plot
    mobility_ranking = create_mobility_ranking_plot(
        movement_metrics=movement_metrics,
        top_n=top_n,
        title=f"Token Mobility Rankings for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Save HTML files if requested
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe name for window
        window_name = "_".join(window_layers)
        
        # Save trajectory plot
        trajectory_path = os.path.join(output_dir, f"token_trajectory_{window_name}.html")
        trajectory_plot.write_html(trajectory_path)
        
        # Save velocity heatmap
        velocity_path = os.path.join(output_dir, f"token_velocity_{window_name}.html")
        velocity_heatmap.write_html(velocity_path)
        
        # Save mobility ranking
        mobility_path = os.path.join(output_dir, f"token_mobility_{window_name}.html")
        mobility_ranking.write_html(mobility_path)
    
    # Return visualization results
    return {
        "trajectory_plot": trajectory_plot,
        "velocity_heatmap": velocity_heatmap,
        "mobility_ranking": mobility_ranking,
        "metrics": movement_metrics
    }


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    batch_size = 2
    seq_len = 10
    n_layers = 3
    n_clusters = 4
    
    # Create fake activations
    activations = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        activations[layer_name] = np.random.rand(batch_size, seq_len, 32)
    
    # Create fake cluster labels
    cluster_labels = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        cluster_labels[layer_name] = np.random.randint(0, n_clusters, size=(batch_size * seq_len))
    
    # Create fake token metadata
    token_metadata = {
        "tokens": [["Token1", "Token2", "Token3", "Token4", "Token5", "Token6", "Token7", "Token8", "Token9", "Token10"] for _ in range(batch_size)],
        "token_ids": np.arange(seq_len * batch_size).reshape(batch_size, seq_len),
        "attention_mask": np.ones((batch_size, seq_len))
    }
    
    # Create window data
    window_data = {
        "activations": activations,
        "metadata": {
            "tokens": token_metadata
        },
        "window_layers": [f"layer{i}" for i in range(n_layers)]
    }
    
    # Create APA results
    apa_results = {
        "clusters": {
            layer_name: {"labels": labels}
            for layer_name, labels in cluster_labels.items()
        }
    }
    
    # Create movement visualization
    viz_results = create_token_movement_visualization(
        window_data=window_data,
        apa_results=apa_results
    )
    
    # Show trajectory plot
    viz_results["trajectory_plot"].show()