"""
GPT-2 attention-to-token path correlation visualization.

This module provides specialized visualization tools for analyzing the
correlation between attention patterns and token paths in GPT-2 models.
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

# Import token path utilities
from visualization.gpt2_token_sankey import extract_token_paths
from visualization.gpt2_attention_sankey import extract_attention_flow


def calculate_token_path_transition_matrix(
    token_paths: Dict[str, Any],
    layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate transition matrices between consecutive layers based on token paths.
    
    Args:
        token_paths: Token path information from extract_token_paths
        layer_names: Optional list of layer names to analyze
        
    Returns:
        Dictionary mapping layer pairs to transition matrices
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = token_paths.get("layers", [])
    
    if not layer_names or len(layer_names) < 2:
        raise ValueError("At least two layers required for transition analysis")
    
    # Initialize result container
    transition_matrices = {}
    
    # Extract paths
    if "paths_by_token_position" not in token_paths:
        return transition_matrices
    
    # Count transitions between clusters
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Create layer transition key
        layer_pair = (layer1, layer2)
        
        # Count transitions
        transition_counts = defaultdict(int)
        cluster_counts = defaultdict(int)
        
        # Process each token position
        for token_key, token_data in token_paths["paths_by_token_position"].items():
            cluster_path = token_data.get("cluster_path", [])
            
            # Skip if path is too short
            if len(cluster_path) <= i + 1:
                continue
            
            # Get clusters for consecutive layers
            cluster1 = cluster_path[i]
            cluster2 = cluster_path[i + 1]
            
            # Count transition
            transition = (cluster1, cluster2)
            transition_counts[transition] += 1
            
            # Count source cluster
            cluster_counts[cluster1] += 1
        
        # Create transition matrix
        unique_clusters = set()
        for c1, c2 in transition_counts.keys():
            unique_clusters.add(c1)
            unique_clusters.add(c2)
        
        n_clusters = max(unique_clusters) + 1 if unique_clusters else 0
        
        # Create normalized transition matrix
        matrix = np.zeros((n_clusters, n_clusters))
        
        for (c1, c2), count in transition_counts.items():
            # Skip if source cluster has no tokens
            if cluster_counts[c1] == 0:
                continue
            
            # Normalize by source cluster count
            matrix[c1, c2] = count / cluster_counts[c1]
        
        # Store result
        transition_matrices[layer_pair] = matrix
    
    return transition_matrices


def calculate_attention_transition_matrix(
    attention_flow: Dict[str, Any],
    layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Calculate transition matrices between consecutive layers based on attention patterns.
    
    Args:
        attention_flow: Attention flow information from extract_attention_flow
        layer_names: Optional list of layer names to analyze
        
    Returns:
        Dictionary mapping layer pairs to transition matrices
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = attention_flow.get("layers", [])
    
    if not layer_names or len(layer_names) < 2:
        raise ValueError("At least two layers required for transition analysis")
    
    # Initialize result container
    transition_matrices = {}
    
    # Skip if no flow data
    if "flow_by_layer" not in attention_flow:
        return transition_matrices
    
    # Calculate transitions based on attention flow
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Create layer transition key
        layer_pair = (layer1, layer2)
        
        # Skip if we don't have data for both layers
        if layer1 not in attention_flow["flow_by_layer"] or layer2 not in attention_flow["flow_by_layer"]:
            continue
        
        # Get flow data
        flow_data1 = attention_flow["flow_by_layer"][layer1]
        
        # Get token positions
        positions = set()
        
        for (source_key, target_key), flow_info in flow_data1.items():
            positions.add(flow_info["source_position"])
            positions.add(flow_info["target_position"])
        
        # Create position-to-position transition matrix
        n_positions = max(positions) + 1 if positions else 0
        
        # Skip if no positions
        if n_positions == 0:
            continue
        
        # Create attention transition matrix
        matrix = np.zeros((n_positions, n_positions))
        
        # Fill in attention weights
        for (source_key, target_key), flow_info in flow_data1.items():
            source_pos = flow_info["source_position"]
            target_pos = flow_info["target_position"]
            
            matrix[source_pos, target_pos] = flow_info["attention_weight"]
        
        # Normalize by row sums
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
        
        # Store result
        transition_matrices[layer_pair] = matrix
    
    return transition_matrices


def calculate_correlation_metrics(
    token_paths: Dict[str, Any],
    attention_flow: Dict[str, Any],
    cluster_labels: Dict[str, np.ndarray],
    layer_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate correlation metrics between token paths and attention patterns.
    
    Args:
        token_paths: Token path information from extract_token_paths
        attention_flow: Attention flow information from extract_attention_flow
        cluster_labels: Dictionary mapping layer names to cluster labels
        layer_names: Optional list of layer names to analyze
        
    Returns:
        Dictionary with correlation metrics
    """
    # Use provided layers or all available
    if layer_names is None:
        layers1 = token_paths.get("layers", [])
        layers2 = attention_flow.get("layers", [])
        layer_names = [layer for layer in layers1 if layer in layers2]
    
    if not layer_names or len(layer_names) < 2:
        raise ValueError("At least two common layers required for correlation analysis")
    
    # Initialize result container
    correlation_metrics = {
        "layers": layer_names,
        "layer_transitions": {},
        "token_correlations": {},
        "global_metrics": {
            "avg_path_attention_correlation": 0.0,
            "attention_follows_paths": 0.0,
            "paths_follow_attention": 0.0
        }
    }
    
    # Calculate token path transition matrices
    path_matrices = calculate_token_path_transition_matrix(
        token_paths=token_paths,
        layer_names=layer_names
    )
    
    # Calculate attention transition matrices
    attention_matrices = calculate_attention_transition_matrix(
        attention_flow=attention_flow,
        layer_names=layer_names
    )
    
    # Calculate correlation for each layer transition
    total_correlation = 0.0
    correlation_count = 0
    
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Create layer transition key
        layer_pair = (layer1, layer2)
        
        # Skip if we don't have both matrices
        if layer_pair not in path_matrices or layer_pair not in attention_matrices:
            continue
        
        # Get matrices
        path_matrix = path_matrices[layer_pair]
        attn_matrix = attention_matrices[layer_pair]
        
        # Get cluster labels
        if layer1 in cluster_labels and layer2 in cluster_labels:
            clusters1 = cluster_labels[layer1]
            clusters2 = cluster_labels[layer2]
            
            # Calculate position-to-cluster matrices
            n_positions = attn_matrix.shape[0]
            n_clusters1 = np.max(clusters1) + 1 if clusters1.size > 0 else 0
            n_clusters2 = np.max(clusters2) + 1 if clusters2.size > 0 else 0
            
            # Skip if no clusters
            if n_clusters1 == 0 or n_clusters2 == 0:
                continue
            
            # Create position-to-cluster mapping matrices
            pos_to_cluster1 = np.zeros((n_positions, n_clusters1))
            pos_to_cluster2 = np.zeros((n_positions, n_clusters2))
            
            # Fill in mappings
            for pos, cluster in enumerate(clusters1):
                if pos < n_positions:
                    pos_to_cluster1[pos, cluster] = 1
            
            for pos, cluster in enumerate(clusters2):
                if pos < n_positions:
                    pos_to_cluster2[pos, cluster] = 1
            
            # Calculate cluster-aware attention matrix
            # A_cluster = P1' * A_pos * P2
            # where P1 and P2 are position-to-cluster matrices
            cluster_attn_matrix = pos_to_cluster1.T @ attn_matrix @ pos_to_cluster2
            
            # Normalize
            row_sums = cluster_attn_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            cluster_attn_matrix = cluster_attn_matrix / row_sums
            
            # Ensure matrices are same shape
            min_clusters1 = min(path_matrix.shape[0], cluster_attn_matrix.shape[0])
            min_clusters2 = min(path_matrix.shape[1], cluster_attn_matrix.shape[1])
            
            path_matrix_trimmed = path_matrix[:min_clusters1, :min_clusters2]
            attn_matrix_trimmed = cluster_attn_matrix[:min_clusters1, :min_clusters2]
            
            # Calculate correlation
            # Flatten matrices
            path_flat = path_matrix_trimmed.flatten()
            attn_flat = attn_matrix_trimmed.flatten()
            
            # Calculate correlation
            correlation = np.corrcoef(path_flat, attn_flat)[0, 1]
            
            # Handle NaN
            if np.isnan(correlation):
                correlation = 0.0
            
            # Store result
            correlation_metrics["layer_transitions"][layer_pair] = {
                "correlation": float(correlation),
                "path_entropy": float(-np.sum(path_flat * np.log2(path_flat + 1e-10))),
                "attention_entropy": float(-np.sum(attn_flat * np.log2(attn_flat + 1e-10)))
            }
            
            # Update total
            total_correlation += correlation
            correlation_count += 1
            
            # Calculate additional metrics
            # "Attention follows paths" - how much attention distribution matches cluster transitions
            # Higher value means attention is allocated along the same cluster transitions as tokens
            path_guided_attention = np.sum(path_matrix_trimmed * attn_matrix_trimmed)
            correlation_metrics["layer_transitions"][layer_pair]["attention_follows_paths"] = float(path_guided_attention)
            
            # "Paths follow attention" - how much token paths follow attention patterns
            # Higher value means tokens tend to move to clusters that receive more attention
            attention_guided_paths = np.sum(attn_matrix_trimmed * path_matrix_trimmed)
            correlation_metrics["layer_transitions"][layer_pair]["paths_follow_attention"] = float(attention_guided_paths)
    
    # Calculate global metrics
    if correlation_count > 0:
        correlation_metrics["global_metrics"]["avg_path_attention_correlation"] = float(total_correlation / correlation_count)
        
        # Calculate global "attention follows paths" and "paths follow attention"
        total_afp = sum([metrics["attention_follows_paths"] for metrics in correlation_metrics["layer_transitions"].values()])
        total_pfa = sum([metrics["paths_follow_attention"] for metrics in correlation_metrics["layer_transitions"].values()])
        
        correlation_metrics["global_metrics"]["attention_follows_paths"] = float(total_afp / correlation_count)
        correlation_metrics["global_metrics"]["paths_follow_attention"] = float(total_pfa / correlation_count)
    
    # Calculate token-specific correlations
    token_correlations = {}
    
    # Get token paths from paths_by_token_position
    if "paths_by_token_position" in token_paths:
        for token_key, token_data in token_paths["paths_by_token_position"].items():
            cluster_path = token_data.get("cluster_path", [])
            
            # Skip if path is too short
            if len(cluster_path) < len(layer_names):
                continue
            
            # Initialize metrics for this token
            token_metrics = {
                "token_text": token_data.get("token_text", ""),
                "token_id": token_data.get("token_id", -1),
                "position": token_data.get("position", -1),
                "layer_correlations": {}
            }
            
            # Calculate path-attention correlation for each layer transition
            for i in range(len(layer_names) - 1):
                layer1 = layer_names[i]
                layer2 = layer_names[i + 1]
                
                # Get token's clusters for consecutive layers
                cluster1 = cluster_path[i]
                cluster2 = cluster_path[i + 1]
                
                # Get token's attention from flow data (if available)
                token_attention = 0.0
                
                if token_key in attention_flow.get("token_importance", {}).get(layer1, {}):
                    token_pos = attention_flow["token_importance"][layer1][token_key]["position"]
                    
                    # Check for attention flow from this token
                    if layer1 in attention_flow["flow_by_layer"]:
                        for (source_key, target_key), flow_info in attention_flow["flow_by_layer"][layer1].items():
                            if source_key == token_key:
                                target_pos = flow_info["target_position"]
                                
                                # Get cluster for target position
                                if layer2 in cluster_labels and target_pos < len(cluster_labels[layer2]):
                                    target_cluster = cluster_labels[layer2][target_pos]
                                    
                                    # If target is in the same cluster as token's next cluster,
                                    # add to token attention
                                    if target_cluster == cluster2:
                                        token_attention += flow_info["attention_weight"]
                
                # Store token's layer correlation
                token_metrics["layer_correlations"][(layer1, layer2)] = {
                    "source_cluster": int(cluster1),
                    "target_cluster": int(cluster2),
                    "attention_to_target_cluster": float(token_attention)
                }
            
            # Calculate average correlation
            correlations = [
                metrics["attention_to_target_cluster"]
                for layer_pair, metrics in token_metrics["layer_correlations"].items()
            ]
            
            token_metrics["avg_correlation"] = float(np.mean(correlations)) if correlations else 0.0
            
            # Store token correlation
            token_correlations[token_key] = token_metrics
    
    # Store token correlations
    correlation_metrics["token_correlations"] = token_correlations
    
    return correlation_metrics


def create_correlation_heatmap(
    correlation_metrics: Dict[str, Any],
    title: Optional[str] = None,
    colorscale: str = "RdBu",
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Create heatmap visualization for path-attention correlation.
    
    Args:
        correlation_metrics: Correlation metrics from calculate_correlation_metrics
        title: Optional title for the plot
        colorscale: Colorscale for the heatmap
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the heatmap
    """
    # Get layer transitions
    layer_transitions = correlation_metrics.get("layer_transitions", {})
    
    if not layer_transitions:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No correlation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract layer names
    layer_names = correlation_metrics.get("layers", [])
    
    # Create matrix for correlation values
    n_layers = len(layer_names)
    matrix = np.zeros((n_layers - 1, 3))  # 3 columns: correlation, afp, pfa
    
    for i in range(n_layers - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Create layer transition key
        layer_pair = (layer1, layer2)
        
        # Skip if we don't have data for this transition
        if layer_pair not in layer_transitions:
            continue
        
        # Get metrics
        metrics = layer_transitions[layer_pair]
        
        # Fill in matrix
        matrix[i, 0] = metrics.get("correlation", 0.0)
        matrix[i, 1] = metrics.get("attention_follows_paths", 0.0)
        matrix[i, 2] = metrics.get("paths_follow_attention", 0.0)
    
    # Create layer pair labels
    layer_pair_labels = [f"{layer_names[i]}-{layer_names[i+1]}" for i in range(n_layers - 1)]
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Correlation", "Attention Follows Paths", "Paths Follow Attention"),
        shared_yaxes=True
    )
    
    # Add heatmaps
    for i, title in enumerate(["Correlation", "Attention Follows Paths", "Paths Follow Attention"]):
        fig.add_trace(
            go.Heatmap(
                z=matrix[:, i].reshape(-1, 1),
                y=layer_pair_labels,
                colorscale=colorscale,
                zmid=0,  # Center colorscale at 0 for correlation
                colorbar=dict(
                    title=title,
                    titleside="right",
                    x=0.33 + i * 0.33
                ),
                hovertemplate="Layer Pair: %{y}<br>Value: %{z:.4f}<extra></extra>"
            ),
            row=1, col=i + 1
        )
    
    # Add global metrics as annotations
    global_metrics = correlation_metrics.get("global_metrics", {})
    
    for i, (key, value) in enumerate(global_metrics.items()):
        fig.add_annotation(
            text=f"{key}: {value:.4f}",
            xref="paper", yref="paper",
            x=0.5, y=1.05 - 0.05 * i,
            showarrow=False,
            font=dict(size=12)
        )
    
    # Add title
    if title is None:
        title = "Path-Attention Correlation Analysis"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        height=height,
        width=width
    )
    
    return fig


def create_token_correlation_scatter(
    correlation_metrics: Dict[str, Any],
    top_n: int = 20,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 700
) -> go.Figure:
    """
    Create scatter visualization for token path-attention correlations.
    
    Args:
        correlation_metrics: Correlation metrics from calculate_correlation_metrics
        top_n: Number of top tokens to highlight
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the scatter plot
    """
    # Get token correlations
    token_correlations = correlation_metrics.get("token_correlations", {})
    
    if not token_correlations:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No token correlation data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract token metrics
    tokens = []
    
    for token_key, metrics in token_correlations.items():
        token_text = metrics.get("token_text", "")
        position = metrics.get("position", -1)
        correlation = metrics.get("avg_correlation", 0.0)
        
        tokens.append({
            "token_key": token_key,
            "token_text": token_text,
            "position": position,
            "correlation": correlation
        })
    
    # Sort tokens by correlation (highest first)
    tokens.sort(key=lambda x: x["correlation"], reverse=True)
    
    # Get top and bottom tokens
    top_tokens = tokens[:top_n]
    bottom_tokens = tokens[-top_n:]
    
    # Create scatter plot for all tokens
    x = [token["position"] for token in tokens]
    y = [token["correlation"] for token in tokens]
    text = [f"Token: {token['token_text']}<br>Position: {token['position']}<br>Correlation: {token['correlation']:.4f}" for token in tokens]
    
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=8,
            color="gray",
            opacity=0.5
        ),
        text=text,
        hoverinfo="text",
        name="All Tokens"
    ))
    
    # Add top tokens
    x_top = [token["position"] for token in top_tokens]
    y_top = [token["correlation"] for token in top_tokens]
    text_top = [f"Token: {token['token_text']}<br>Position: {token['position']}<br>Correlation: {token['correlation']:.4f}" for token in top_tokens]
    
    fig.add_trace(go.Scatter(
        x=x_top,
        y=y_top,
        mode="markers+text",
        marker=dict(
            size=10,
            color="green"
        ),
        text=[token["token_text"] for token in top_tokens],
        textposition="top center",
        hovertext=text_top,
        hoverinfo="text",
        name="Top Correlated Tokens"
    ))
    
    # Add bottom tokens
    x_bottom = [token["position"] for token in bottom_tokens]
    y_bottom = [token["correlation"] for token in bottom_tokens]
    text_bottom = [f"Token: {token['token_text']}<br>Position: {token['position']}<br>Correlation: {token['correlation']:.4f}" for token in bottom_tokens]
    
    fig.add_trace(go.Scatter(
        x=x_bottom,
        y=y_bottom,
        mode="markers+text",
        marker=dict(
            size=10,
            color="red"
        ),
        text=[token["token_text"] for token in bottom_tokens],
        textposition="bottom center",
        hovertext=text_bottom,
        hoverinfo="text",
        name="Bottom Correlated Tokens"
    ))
    
    # Add title
    if title is None:
        title = f"Token Path-Attention Correlation (Top {top_n} highlighted)"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Token Position",
        yaxis_title="Path-Attention Correlation",
        height=height,
        width=width,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_attention_path_correlation_visualization(
    window_data: Dict[str, Any],
    apa_results: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    top_n: int = 20,
    save_html: bool = False,
    output_dir: str = "gpt2_visualizations"
) -> Dict[str, Any]:
    """
    Create comprehensive attention-path correlation visualizations.
    
    Args:
        window_data: Window data from GPT-2 analysis
        apa_results: APA analysis results
        highlight_tokens: List of token texts to highlight
        top_n: Number of top tokens to highlight
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
    
    # Skip if missing required data
    if not activations or not attention_data or not token_metadata or not cluster_labels:
        return {
            "error": "Missing required data for correlation analysis",
            "correlation_heatmap": go.Figure(),
            "token_scatter": go.Figure()
        }
    
    # Extract token paths
    token_paths = extract_token_paths(
        activations=activations,
        token_metadata=token_metadata,
        cluster_labels=cluster_labels
    )
    
    # Extract attention flow
    attention_flow = extract_attention_flow(
        attention_data=attention_data,
        token_metadata=token_metadata
    )
    
    # Calculate correlation metrics
    correlation_metrics = calculate_correlation_metrics(
        token_paths=token_paths,
        attention_flow=attention_flow,
        cluster_labels=cluster_labels,
        layer_names=window_layers
    )
    
    # Create correlation heatmap
    correlation_heatmap = create_correlation_heatmap(
        correlation_metrics=correlation_metrics,
        title=f"Path-Attention Correlation for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create token correlation scatter
    token_scatter = create_token_correlation_scatter(
        correlation_metrics=correlation_metrics,
        top_n=top_n
    )
    
    # Save HTML files if requested
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe name for window
        window_name = "_".join(window_layers)
        
        # Save correlation heatmap
        heatmap_path = os.path.join(output_dir, f"correlation_heatmap_{window_name}.html")
        correlation_heatmap.write_html(heatmap_path)
        
        # Save token scatter
        scatter_path = os.path.join(output_dir, f"token_correlation_{window_name}.html")
        token_scatter.write_html(scatter_path)
    
    # Return visualization results
    return {
        "correlation_heatmap": correlation_heatmap,
        "token_scatter": token_scatter,
        "metrics": correlation_metrics
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
    
    # Create window data
    window_data = {
        "activations": activations,
        "metadata": {
            "attention": attention_data,
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
    
    # Create correlation visualization
    viz_results = create_attention_path_correlation_visualization(
        window_data=window_data,
        apa_results=apa_results
    )
    
    # Show correlation heatmap
    viz_results["correlation_heatmap"].show()