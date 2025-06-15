"""
GPT-2 concept purity visualization.

This module provides tools for visualizing concept purity metrics across
layers in GPT-2 models, showing how well-defined cluster concepts are
and how they evolve through the network.
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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Import utilities from existing visualizations
from visualization.cluster_utils import compute_layer_clusters_embedded


def calculate_concept_purity_metrics(
    activations: Dict[str, np.ndarray],
    cluster_labels: Dict[str, np.ndarray],
    layer_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate concept purity metrics for layers.
    
    Args:
        activations: Dictionary mapping layer names to activations
        cluster_labels: Dictionary mapping layer names to cluster labels
        layer_names: Optional list of layer names to analyze
        
    Returns:
        Dictionary with concept purity metrics
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = sorted(list(cluster_labels.keys()))
    
    # Initialize result container
    purity_metrics = {
        "layers": layer_names,
        "layer_metrics": {},
        "global_metrics": {
            "avg_silhouette": 0.0,
            "avg_intra_cluster_distance": 0.0,
            "avg_inter_cluster_distance": 0.0,
            "avg_purity": 0.0
        }
    }
    
    # Process each layer
    total_silhouette = 0.0
    total_intra_distance = 0.0
    total_inter_distance = 0.0
    total_purity = 0.0
    layer_count = 0
    
    for layer in layer_names:
        # Skip if we don't have activations or labels for this layer
        if layer not in activations or layer not in cluster_labels:
            continue
        
        # Get activations and labels
        layer_activations = activations[layer]
        labels = cluster_labels[layer]
        
        # Make sure activations and labels are properly shaped
        if len(layer_activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = layer_activations.shape
            
            # Reshape to 2D
            layer_activations = layer_activations.reshape(-1, hidden_dim)
        
        # Make sure labels match activations
        if len(labels) != len(layer_activations):
            # Skip if dimensions don't match
            continue
        
        # Initialize metrics for this layer
        layer_metrics = {}
        
        # Calculate silhouette score if we have at least 2 clusters
        silhouette = 0.0
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            try:
                # Silhouette score requires at least 2 samples per cluster
                # Check if all clusters have at least 2 samples
                valid_clusters = True
                for label in unique_labels:
                    if np.sum(labels == label) < 2:
                        valid_clusters = False
                        break
                
                if valid_clusters:
                    silhouette = silhouette_score(layer_activations, labels)
            except Exception as e:
                # Print error but continue
                print(f"Error calculating silhouette score for layer {layer}: {e}")
        
        # Calculate intra-cluster distances
        intra_cluster_distances = {}
        cluster_centers = {}
        
        for label in unique_labels:
            # Get samples for this cluster
            mask = (labels == label)
            samples = layer_activations[mask]
            
            if len(samples) > 0:
                # Calculate cluster center
                center = np.mean(samples, axis=0)
                cluster_centers[label] = center
                
                # Calculate mean distance to center
                distances = np.linalg.norm(samples - center, axis=1)
                intra_cluster_distances[label] = float(np.mean(distances))
        
        # Calculate mean intra-cluster distance
        mean_intra_distance = np.mean(list(intra_cluster_distances.values())) if intra_cluster_distances else 0.0
        
        # Calculate inter-cluster distances
        inter_cluster_distances = []
        
        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j and label1 in cluster_centers and label2 in cluster_centers:
                    center1 = cluster_centers[label1]
                    center2 = cluster_centers[label2]
                    
                    # Calculate distance between centers
                    distance_between = np.linalg.norm(center1 - center2)
                    inter_cluster_distances.append(distance_between)
        
        # Calculate mean inter-cluster distance
        mean_inter_distance = np.mean(inter_cluster_distances) if inter_cluster_distances else 0.0
        
        # Calculate purity score as ratio of inter to intra distances
        # Higher values mean more pure concepts (compact clusters that are far apart)
        purity = mean_inter_distance / mean_intra_distance if mean_intra_distance > 0 else 0.0
        
        # Store layer metrics
        layer_metrics["silhouette"] = float(silhouette)
        layer_metrics["intra_cluster_distance"] = float(mean_intra_distance)
        layer_metrics["inter_cluster_distance"] = float(mean_inter_distance)
        layer_metrics["purity"] = float(purity)
        layer_metrics["cluster_distances"] = {int(k): float(v) for k, v in intra_cluster_distances.items()}
        
        # Add calinski_harabasz and davies_bouldin scores if available
        if len(unique_labels) > 1:
            try:
                ch_score = calinski_harabasz_score(layer_activations, labels)
                layer_metrics["calinski_harabasz"] = float(ch_score)
            except Exception:
                pass
            
            try:
                db_score = davies_bouldin_score(layer_activations, labels)
                layer_metrics["davies_bouldin"] = float(db_score)
            except Exception:
                pass
        
        # Store layer metrics
        purity_metrics["layer_metrics"][layer] = layer_metrics
        
        # Update totals
        total_silhouette += silhouette
        total_intra_distance += mean_intra_distance
        total_inter_distance += mean_inter_distance
        total_purity += purity
        layer_count += 1
    
    # Calculate global averages
    if layer_count > 0:
        purity_metrics["global_metrics"]["avg_silhouette"] = float(total_silhouette / layer_count)
        purity_metrics["global_metrics"]["avg_intra_cluster_distance"] = float(total_intra_distance / layer_count)
        purity_metrics["global_metrics"]["avg_inter_cluster_distance"] = float(total_inter_distance / layer_count)
        purity_metrics["global_metrics"]["avg_purity"] = float(total_purity / layer_count)
    
    return purity_metrics


def create_concept_purity_plot(
    purity_metrics: Dict[str, Any],
    metric_type: str = "purity",
    title: Optional[str] = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create line plot visualization of concept purity across layers.
    
    Args:
        purity_metrics: Purity metrics from calculate_concept_purity_metrics
        metric_type: Type of metric to visualize (purity, silhouette, intra_cluster_distance, inter_cluster_distance)
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the line plot
    """
    # Get layer metrics
    layer_metrics = purity_metrics.get("layer_metrics", {})
    
    if not layer_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No concept purity metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get layers in order
    layer_names = purity_metrics.get("layers", [])
    
    # Extract metric values for each layer
    layers = []
    values = []
    
    for layer in layer_names:
        if layer in layer_metrics:
            metrics = layer_metrics[layer]
            
            # Get requested metric
            if metric_type == "purity":
                value = metrics.get("purity", 0.0)
            elif metric_type == "silhouette":
                value = metrics.get("silhouette", 0.0)
            elif metric_type == "intra_cluster_distance":
                value = metrics.get("intra_cluster_distance", 0.0)
            elif metric_type == "inter_cluster_distance":
                value = metrics.get("inter_cluster_distance", 0.0)
            elif metric_type == "calinski_harabasz":
                value = metrics.get("calinski_harabasz", 0.0)
            elif metric_type == "davies_bouldin":
                value = metrics.get("davies_bouldin", 0.0)
            else:
                value = 0.0
            
            layers.append(layer)
            values.append(value)
    
    # Create figure
    fig = go.Figure()
    
    # Add line plot
    fig.add_trace(go.Scatter(
        x=layers,
        y=values,
        mode="lines+markers",
        marker=dict(
            size=10,
            color="blue"
        ),
        line=dict(
            color="blue",
            width=2
        ),
        name=metric_type.replace("_", " ").title(),
        hovertemplate="Layer: %{x}<br>Value: %{y:.4f}<extra></extra>"
    ))
    
    # Add global average line
    global_metrics = purity_metrics.get("global_metrics", {})
    
    if global_metrics:
        # Get global average for this metric
        if metric_type == "purity":
            avg_value = global_metrics.get("avg_purity", 0.0)
        elif metric_type == "silhouette":
            avg_value = global_metrics.get("avg_silhouette", 0.0)
        elif metric_type == "intra_cluster_distance":
            avg_value = global_metrics.get("avg_intra_cluster_distance", 0.0)
        elif metric_type == "inter_cluster_distance":
            avg_value = global_metrics.get("avg_inter_cluster_distance", 0.0)
        else:
            avg_value = 0.0
        
        # Add horizontal line
        fig.add_shape(
            type="line",
            x0=layers[0],
            y0=avg_value,
            x1=layers[-1],
            y1=avg_value,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            )
        )
        
        # Add annotation
        fig.add_annotation(
            x=layers[-1],
            y=avg_value,
            text=f"Global Avg: {avg_value:.4f}",
            showarrow=False,
            yshift=10,
            font=dict(color="red")
        )
    
    # Add title and labels
    if title is None:
        title = f"Concept {metric_type.replace('_', ' ').title()} Across Layers"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Layer",
        yaxis_title=metric_type.replace("_", " ").title(),
        height=height,
        width=width,
        showlegend=True
    )
    
    return fig


def create_cluster_heatmap(
    purity_metrics: Dict[str, Any],
    layer_name: str,
    metric_type: str = "cluster_distances",
    title: Optional[str] = None,
    colorscale: str = "Viridis",
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create heatmap visualization of cluster metrics for a specific layer.
    
    Args:
        purity_metrics: Purity metrics from calculate_concept_purity_metrics
        layer_name: Name of the layer to visualize
        metric_type: Type of metric to visualize (cluster_distances)
        title: Optional title for the plot
        colorscale: Colorscale for the heatmap
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the heatmap
    """
    # Get layer metrics
    layer_metrics = purity_metrics.get("layer_metrics", {})
    
    if not layer_metrics or layer_name not in layer_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"No metrics available for layer {layer_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get metrics for this layer
    metrics = layer_metrics[layer_name]
    
    # Handle different metric types
    if metric_type == "cluster_distances":
        # Get cluster distances
        cluster_distances = metrics.get("cluster_distances", {})
        
        if not cluster_distances:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No cluster distances available for layer {layer_name}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Convert to list
        clusters = sorted(list(cluster_distances.keys()))
        values = [cluster_distances[c] for c in clusters]
        
        # Create figure with bar chart
        fig = go.Figure(data=go.Bar(
            x=[f"Cluster {c}" for c in clusters],
            y=values,
            marker_color=values,
            marker=dict(colorscale=colorscale),
            text=[f"{v:.4f}" for v in values],
            textposition="auto",
            hovertemplate="Cluster: %{x}<br>Intra-Cluster Distance: %{y:.4f}<extra></extra>"
        ))
        
        # Add title
        if title is None:
            title = f"Intra-Cluster Distances for Layer {layer_name}"
        
        # Update layout
        fig.update_layout(
            title_text=title,
            xaxis_title="Cluster",
            yaxis_title="Intra-Cluster Distance",
            height=height,
            width=width
        )
        
        return fig
    else:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Metric type {metric_type} not supported",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig


def create_concept_stability_plot(
    purity_metrics: Dict[str, Any],
    metric_types: Optional[List[str]] = None,
    title: Optional[str] = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create multi-metric line plot showing concept stability across layers.
    
    Args:
        purity_metrics: Purity metrics from calculate_concept_purity_metrics
        metric_types: List of metrics to plot (defaults to purity, silhouette, intra/inter distances)
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the multi-metric plot
    """
    # Default metrics to plot
    if metric_types is None:
        metric_types = ["purity", "silhouette", "intra_cluster_distance", "inter_cluster_distance"]
    
    # Get layer metrics
    layer_metrics = purity_metrics.get("layer_metrics", {})
    
    if not layer_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No concept purity metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get layers in order
    layer_names = purity_metrics.get("layers", [])
    
    # Create figure
    fig = go.Figure()
    
    # Color map for metrics
    color_map = {
        "purity": "blue",
        "silhouette": "green",
        "intra_cluster_distance": "red",
        "inter_cluster_distance": "purple",
        "calinski_harabasz": "orange",
        "davies_bouldin": "brown"
    }
    
    # Add each metric as a separate line
    for metric_type in metric_types:
        # Extract values for this metric
        layers = []
        values = []
        
        for layer in layer_names:
            if layer in layer_metrics:
                metrics = layer_metrics[layer]
                
                # Check if metric exists for this layer
                if metric_type in metrics:
                    layers.append(layer)
                    values.append(metrics[metric_type])
        
        # Skip if no values
        if not values:
            continue
        
        # Apply normalization for davies_bouldin (lower is better)
        if metric_type == "davies_bouldin":
            # Invert values so higher is better (for consistent presentation)
            max_val = max(values) if values else 1.0
            values = [max_val - v for v in values]
            display_name = "Davies-Bouldin (inverted)"
        else:
            display_name = metric_type.replace("_", " ").title()
        
        # Add line plot
        fig.add_trace(go.Scatter(
            x=layers,
            y=values,
            mode="lines+markers",
            marker=dict(
                size=8,
                color=color_map.get(metric_type, "gray")
            ),
            line=dict(
                color=color_map.get(metric_type, "gray"),
                width=2
            ),
            name=display_name,
            hovertemplate=f"{display_name}: %{{y:.4f}}<br>Layer: %{{x}}<extra></extra>"
        ))
    
    # Add title
    if title is None:
        title = "Concept Stability Metrics Across Layers"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Layer",
        yaxis_title="Metric Value",
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_concept_purity_visualization(
    window_data: Dict[str, Any],
    apa_results: Dict[str, Any],
    metric_types: Optional[List[str]] = None,
    save_html: bool = False,
    output_dir: str = "gpt2_visualizations"
) -> Dict[str, Any]:
    """
    Create comprehensive concept purity visualizations.
    
    Args:
        window_data: Window data from GPT-2 analysis
        apa_results: APA analysis results
        metric_types: List of metrics to include (defaults to all available)
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
    
    # Skip if missing required data
    if not activations or not cluster_labels:
        return {
            "error": "Missing required data for concept purity analysis",
            "purity_plot": go.Figure(),
            "stability_plot": go.Figure(),
            "cluster_heatmap": go.Figure()
        }
    
    # Calculate concept purity metrics
    purity_metrics = calculate_concept_purity_metrics(
        activations=activations,
        cluster_labels=cluster_labels,
        layer_names=window_layers
    )
    
    # Default metrics to display
    if metric_types is None:
        # Check which metrics are available across most layers
        available_metrics = set()
        for layer, metrics in purity_metrics["layer_metrics"].items():
            available_metrics.update(metrics.keys())
        
        # Prioritize key metrics
        metric_types = []
        for metric in ["purity", "silhouette", "intra_cluster_distance", "inter_cluster_distance"]:
            if metric in available_metrics:
                metric_types.append(metric)
        
        # Add additional metrics if available
        for metric in ["calinski_harabasz", "davies_bouldin"]:
            if metric in available_metrics:
                metric_types.append(metric)
    
    # Create purity plot
    purity_plot = create_concept_purity_plot(
        purity_metrics=purity_metrics,
        metric_type="purity",
        title=f"Concept Purity for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create stability plot with multiple metrics
    stability_plot = create_concept_stability_plot(
        purity_metrics=purity_metrics,
        metric_types=metric_types,
        title=f"Concept Stability for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create cluster heatmap for the first layer
    cluster_heatmap = create_cluster_heatmap(
        purity_metrics=purity_metrics,
        layer_name=window_layers[0] if window_layers else next(iter(cluster_labels.keys()), ""),
        metric_type="cluster_distances",
        title=f"Cluster Compactness for {window_layers[0] if window_layers else ''}"
    )
    
    # Save HTML files if requested
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe name for window
        window_name = "_".join(window_layers)
        
        # Save purity plot
        purity_path = os.path.join(output_dir, f"concept_purity_{window_name}.html")
        purity_plot.write_html(purity_path)
        
        # Save stability plot
        stability_path = os.path.join(output_dir, f"concept_stability_{window_name}.html")
        stability_plot.write_html(stability_path)
        
        # Save cluster heatmap
        heatmap_path = os.path.join(output_dir, f"cluster_compactness_{window_name}.html")
        cluster_heatmap.write_html(heatmap_path)
    
    # Return visualization results
    return {
        "purity_plot": purity_plot,
        "stability_plot": stability_plot,
        "cluster_heatmap": cluster_heatmap,
        "metrics": purity_metrics
    }


# Example usage
if __name__ == "__main__":
    # Mock data for testing
    batch_size = 2
    seq_len = 10
    n_layers = 3
    n_clusters = 4
    hidden_dim = 32
    
    # Create fake activations
    activations = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        activations[layer_name] = np.random.rand(batch_size, seq_len, hidden_dim)
    
    # Create fake cluster labels
    cluster_labels = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        cluster_labels[layer_name] = np.random.randint(0, n_clusters, size=(batch_size * seq_len))
    
    # Create window data
    window_data = {
        "activations": activations,
        "metadata": {},
        "window_layers": [f"layer{i}" for i in range(n_layers)]
    }
    
    # Create APA results
    apa_results = {
        "clusters": {
            layer_name: {"labels": labels}
            for layer_name, labels in cluster_labels.items()
        }
    }
    
    # Create purity visualization
    viz_results = create_concept_purity_visualization(
        window_data=window_data,
        apa_results=apa_results
    )
    
    # Show purity plot
    viz_results["purity_plot"].show()