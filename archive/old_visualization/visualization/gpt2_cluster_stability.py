"""
GPT-2 clustering stability visualization.

This module provides tools for visualizing cluster stability across different 
parameter settings and layers, showing how robust the clusters are to changes
in the model's parameters or data.
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
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

# Import utilities from existing visualizations
from visualization.cluster_utils import compute_layer_clusters_embedded


def calculate_clustering_stability(
    activations: Dict[str, np.ndarray],
    cluster_labels: Dict[str, np.ndarray],
    layer_names: Optional[List[str]] = None,
    n_trials: int = 5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Calculate clustering stability metrics by running multiple clustering trials.
    
    Args:
        activations: Dictionary mapping layer names to activations
        cluster_labels: Dictionary mapping layer names to cluster labels
        layer_names: Optional list of layer names to analyze
        n_trials: Number of trials to run
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with clustering stability metrics
    """
    # Use provided layers or all available
    if layer_names is None:
        layer_names = sorted(list(cluster_labels.keys()))
    
    # Initialize result container
    stability_metrics = {
        "layers": layer_names,
        "layer_metrics": {},
        "global_metrics": {
            "avg_rand_stability": 0.0,
            "avg_ami_stability": 0.0,
            "avg_label_consistency": 0.0
        }
    }
    
    # Process each layer
    total_rand = 0.0
    total_ami = 0.0
    total_consistency = 0.0
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
        
        # Get number of clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Skip if there's only one cluster
        if n_clusters <= 1:
            continue
        
        # Run multiple clustering trials
        trial_labels = []
        
        # First trial is the original labels
        trial_labels.append(labels)
        
        # Run additional trials with different random seeds
        from sklearn.cluster import KMeans
        
        for i in range(1, n_trials):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed + i)
            new_labels = kmeans.fit_predict(layer_activations)
            trial_labels.append(new_labels)
        
        # Calculate pairwise adjusted Rand index and adjusted mutual information
        rand_scores = []
        ami_scores = []
        
        for i in range(n_trials):
            for j in range(i + 1, n_trials):
                rand_score = adjusted_rand_score(trial_labels[i], trial_labels[j])
                ami_score = adjusted_mutual_info_score(trial_labels[i], trial_labels[j])
                
                rand_scores.append(rand_score)
                ami_scores.append(ami_score)
        
        # Calculate mean scores
        mean_rand = np.mean(rand_scores) if rand_scores else 0.0
        mean_ami = np.mean(ami_scores) if ami_scores else 0.0
        
        # Calculate label consistency as average proportion of matching assignments
        consistency_scores = []
        
        for i in range(1, n_trials):
            # Create label mapping from this trial to original
            best_mapping = {}
            for label in range(n_clusters):
                # Find the most common original label for this trial label
                mask = (trial_labels[i] == label)
                if np.any(mask):
                    orig_labels = trial_labels[0][mask]
                    counts = np.bincount(orig_labels, minlength=n_clusters)
                    best_match = np.argmax(counts)
                    best_mapping[label] = best_match
            
            # Apply mapping
            mapped_labels = np.array([best_mapping.get(label, label) for label in trial_labels[i]])
            
            # Calculate proportion of matching labels
            matches = np.sum(mapped_labels == trial_labels[0])
            consistency = matches / len(trial_labels[0])
            consistency_scores.append(consistency)
        
        # Calculate mean consistency
        mean_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Store layer metrics
        layer_metrics["rand_stability"] = float(mean_rand)
        layer_metrics["ami_stability"] = float(mean_ami)
        layer_metrics["label_consistency"] = float(mean_consistency)
        layer_metrics["rand_scores"] = [float(score) for score in rand_scores]
        layer_metrics["ami_scores"] = [float(score) for score in ami_scores]
        layer_metrics["consistency_scores"] = [float(score) for score in consistency_scores]
        layer_metrics["n_clusters"] = int(n_clusters)
        
        # Store layer metrics
        stability_metrics["layer_metrics"][layer] = layer_metrics
        
        # Update totals
        total_rand += mean_rand
        total_ami += mean_ami
        total_consistency += mean_consistency
        layer_count += 1
    
    # Calculate global averages
    if layer_count > 0:
        stability_metrics["global_metrics"]["avg_rand_stability"] = float(total_rand / layer_count)
        stability_metrics["global_metrics"]["avg_ami_stability"] = float(total_ami / layer_count)
        stability_metrics["global_metrics"]["avg_label_consistency"] = float(total_consistency / layer_count)
    
    return stability_metrics


def create_stability_line_plot(
    stability_metrics: Dict[str, Any],
    metric_type: str = "label_consistency",
    title: Optional[str] = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create line plot visualization of clustering stability across layers.
    
    Args:
        stability_metrics: Stability metrics from calculate_clustering_stability
        metric_type: Type of metric to visualize (rand_stability, ami_stability, label_consistency)
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the line plot
    """
    # Get layer metrics
    layer_metrics = stability_metrics.get("layer_metrics", {})
    
    if not layer_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering stability metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get layers in order
    layer_names = stability_metrics.get("layers", [])
    
    # Extract metric values for each layer
    layers = []
    values = []
    
    for layer in layer_names:
        if layer in layer_metrics:
            metrics = layer_metrics[layer]
            
            # Get requested metric
            if metric_type == "rand_stability":
                value = metrics.get("rand_stability", 0.0)
            elif metric_type == "ami_stability":
                value = metrics.get("ami_stability", 0.0)
            elif metric_type == "label_consistency":
                value = metrics.get("label_consistency", 0.0)
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
            color="green"
        ),
        line=dict(
            color="green",
            width=2
        ),
        name=metric_type.replace("_", " ").title(),
        hovertemplate="Layer: %{x}<br>Value: %{y:.4f}<extra></extra>"
    ))
    
    # Add global average line
    global_metrics = stability_metrics.get("global_metrics", {})
    
    if global_metrics:
        # Get global average for this metric
        if metric_type == "rand_stability":
            avg_value = global_metrics.get("avg_rand_stability", 0.0)
        elif metric_type == "ami_stability":
            avg_value = global_metrics.get("avg_ami_stability", 0.0)
        elif metric_type == "label_consistency":
            avg_value = global_metrics.get("avg_label_consistency", 0.0)
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
        title = f"Clustering {metric_type.replace('_', ' ').title()} Across Layers"
    
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


def create_stability_boxplot(
    stability_metrics: Dict[str, Any],
    metric_type: str = "consistency_scores",
    title: Optional[str] = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create boxplot visualization of clustering stability scores.
    
    Args:
        stability_metrics: Stability metrics from calculate_clustering_stability
        metric_type: Type of scores to visualize (consistency_scores, rand_scores, ami_scores)
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the boxplot
    """
    # Get layer metrics
    layer_metrics = stability_metrics.get("layer_metrics", {})
    
    if not layer_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering stability metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Get layers in order
    layer_names = stability_metrics.get("layers", [])
    
    # Extract score values for each layer
    layers = []
    all_scores = []
    
    for layer in layer_names:
        if layer in layer_metrics:
            metrics = layer_metrics[layer]
            
            # Get requested metric
            if metric_type == "consistency_scores":
                scores = metrics.get("consistency_scores", [])
            elif metric_type == "rand_scores":
                scores = metrics.get("rand_scores", [])
            elif metric_type == "ami_scores":
                scores = metrics.get("ami_scores", [])
            else:
                scores = []
            
            # Skip if no scores
            if not scores:
                continue
                
            layers.append(layer)
            all_scores.append(scores)
    
    # Create figure
    fig = go.Figure()
    
    # Add boxplot for each layer
    for i, layer in enumerate(layers):
        # Add boxplot
        fig.add_trace(go.Box(
            y=all_scores[i],
            name=layer,
            boxmean=True,  # Show mean
            marker_color="green",
            line=dict(color="darkgreen")
        ))
    
    # Add title and labels
    if title is None:
        title = f"Clustering {metric_type.replace('_scores', '').replace('_', ' ').title()} Stability"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Layer",
        yaxis_title=metric_type.replace("_scores", "").replace("_", " ").title(),
        height=height,
        width=width,
        showlegend=False
    )
    
    return fig


def create_cluster_size_stability_plot(
    stability_metrics: Dict[str, Any],
    title: Optional[str] = None,
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create scatter plot showing relationship between cluster size and stability.
    
    Args:
        stability_metrics: Stability metrics from calculate_clustering_stability
        title: Optional title for the plot
        height: Plot height in pixels
        width: Plot width in pixels
        
    Returns:
        Plotly Figure object with the scatter plot
    """
    # Get layer metrics
    layer_metrics = stability_metrics.get("layer_metrics", {})
    
    if not layer_metrics:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering stability metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract cluster sizes and stability scores
    layers = []
    cluster_sizes = []
    consistency_scores = []
    rand_scores = []
    ami_scores = []
    
    for layer, metrics in layer_metrics.items():
        # Skip if missing metrics
        if "n_clusters" not in metrics or "label_consistency" not in metrics:
            continue
        
        layers.append(layer)
        cluster_sizes.append(metrics["n_clusters"])
        consistency_scores.append(metrics["label_consistency"])
        rand_scores.append(metrics.get("rand_stability", 0.0))
        ami_scores.append(metrics.get("ami_stability", 0.0))
    
    # Create figure
    fig = go.Figure()
    
    # Add consistency scores
    fig.add_trace(go.Scatter(
        x=cluster_sizes,
        y=consistency_scores,
        mode="markers+text",
        marker=dict(
            size=12,
            color="green"
        ),
        text=layers,
        textposition="top center",
        name="Label Consistency",
        hovertemplate="Layer: %{text}<br>Clusters: %{x}<br>Consistency: %{y:.4f}<extra></extra>"
    ))
    
    # Add Rand scores
    fig.add_trace(go.Scatter(
        x=cluster_sizes,
        y=rand_scores,
        mode="markers",
        marker=dict(
            size=10,
            color="blue"
        ),
        text=layers,
        name="Adjusted Rand Index",
        hovertemplate="Layer: %{text}<br>Clusters: %{x}<br>ARI: %{y:.4f}<extra></extra>"
    ))
    
    # Add AMI scores
    fig.add_trace(go.Scatter(
        x=cluster_sizes,
        y=ami_scores,
        mode="markers",
        marker=dict(
            size=10,
            color="purple"
        ),
        text=layers,
        name="Adjusted Mutual Info",
        hovertemplate="Layer: %{text}<br>Clusters: %{x}<br>AMI: %{y:.4f}<extra></extra>"
    ))
    
    # Add trend line if we have enough points
    if len(cluster_sizes) >= 3:
        import numpy as np
        
        # Calculate trend line for consistency scores
        z = np.polyfit(cluster_sizes, consistency_scores, 1)
        p = np.poly1d(z)
        
        # Calculate endpoints
        x_min = min(cluster_sizes)
        x_max = max(cluster_sizes)
        x_range = np.linspace(x_min, x_max, 100)
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=x_range,
            y=p(x_range),
            mode="lines",
            line=dict(
                color="green",
                dash="dash"
            ),
            name="Consistency Trend",
            hoverinfo="none"
        ))
    
    # Add title and labels
    if title is None:
        title = "Cluster Size vs. Stability"
    
    # Update layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Number of Clusters",
        yaxis_title="Stability Score",
        height=height,
        width=width,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig


def create_clustering_stability_visualization(
    window_data: Dict[str, Any],
    apa_results: Dict[str, Any],
    n_trials: int = 5,
    save_html: bool = False,
    output_dir: str = "gpt2_visualizations"
) -> Dict[str, Any]:
    """
    Create comprehensive clustering stability visualizations.
    
    Args:
        window_data: Window data from GPT-2 analysis
        apa_results: APA analysis results
        n_trials: Number of clustering trials to run
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
            "error": "Missing required data for clustering stability analysis",
            "stability_plot": go.Figure(),
            "stability_boxplot": go.Figure(),
            "cluster_size_plot": go.Figure()
        }
    
    # Calculate clustering stability metrics
    stability_metrics = calculate_clustering_stability(
        activations=activations,
        cluster_labels=cluster_labels,
        layer_names=window_layers,
        n_trials=n_trials
    )
    
    # Create stability line plot
    stability_plot = create_stability_line_plot(
        stability_metrics=stability_metrics,
        metric_type="label_consistency",
        title=f"Clustering Stability for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create stability boxplot
    stability_boxplot = create_stability_boxplot(
        stability_metrics=stability_metrics,
        metric_type="consistency_scores",
        title=f"Stability Score Distribution for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Create cluster size vs stability plot
    cluster_size_plot = create_cluster_size_stability_plot(
        stability_metrics=stability_metrics,
        title=f"Cluster Size vs. Stability for Window {window_layers[0]}-{window_layers[-1]}"
    )
    
    # Save HTML files if requested
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create safe name for window
        window_name = "_".join(window_layers)
        
        # Save stability plot
        stability_path = os.path.join(output_dir, f"clustering_stability_{window_name}.html")
        stability_plot.write_html(stability_path)
        
        # Save boxplot
        boxplot_path = os.path.join(output_dir, f"stability_distribution_{window_name}.html")
        stability_boxplot.write_html(boxplot_path)
        
        # Save cluster size plot
        size_path = os.path.join(output_dir, f"cluster_size_stability_{window_name}.html")
        cluster_size_plot.write_html(size_path)
    
    # Return visualization results
    return {
        "stability_plot": stability_plot,
        "stability_boxplot": stability_boxplot,
        "cluster_size_plot": cluster_size_plot,
        "metrics": stability_metrics
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
    
    # Create stability visualization
    viz_results = create_clustering_stability_visualization(
        window_data=window_data,
        apa_results=apa_results,
        n_trials=3  # Use small number for testing
    )
    
    # Show stability plot
    viz_results["stability_plot"].show()