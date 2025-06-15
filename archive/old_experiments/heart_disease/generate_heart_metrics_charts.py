#!/usr/bin/env python
"""
Generate Fragmentation Metrics Charts for the Heart Dataset

This script creates visualizations of fragmentation metrics across layers
for the heart disease dataset, saving the results in the arxiv submission
figures directory.
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the paths
ARXIV_FIGURES_DIR = Path("arxiv_submission/figures")
PATHS_FILE = Path("data/cluster_paths/heart_seed_0_paths.json")
PATHS_WITH_CENTROIDS_FILE = Path("data/cluster_paths/heart_seed_0_paths_with_centroids.json")
CLUSTER_STATS_FILE = Path("data/cluster_stats/heart_seed_0.json")

def load_data(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return None

def compute_entropy_fragmentation(paths_data):
    """
    Compute entropy-based fragmentation by layer.
    
    Args:
        paths_data: Data from the paths JSON file
        
    Returns:
        Dictionary of entropy fragmentation values by layer
    """
    layers = paths_data.get("layers", [])
    unique_paths = paths_data.get("unique_paths", [])
    
    if not layers or not unique_paths:
        return {}
    
    # For each layer, compute the entropy of cluster assignments
    entropy_by_layer = {}
    
    for i, layer in enumerate(layers[:-1]):  # Skip the output layer
        # Count cluster frequency at this layer
        cluster_counts = Counter([path[i] for path in unique_paths])
        total_samples = len(unique_paths)
        
        # Compute entropy
        entropy = 0
        for count in cluster_counts.values():
            p = count / total_samples
            entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy (log2 of number of clusters)
        max_entropy = np.log2(len(cluster_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0
            
        entropy_by_layer[layer] = normalized_entropy
    
    return entropy_by_layer

def compute_path_coherence(paths_data):
    """
    Compute path coherence by layer.
    
    Args:
        paths_data: Data from the paths JSON file
        
    Returns:
        Dictionary of path coherence values by layer
    """
    layers = paths_data.get("layers", [])
    unique_paths = paths_data.get("unique_paths", [])
    
    if not layers or not unique_paths:
        return {}
    
    # For each layer, compute path coherence (opposite of fragmentation)
    coherence_by_layer = {}
    
    for i in range(len(layers) - 2):  # Look at transitions between layers
        current_layer = layers[i]
        next_layer = layers[i + 1]
        
        # Count transitions from this layer to the next
        transitions = defaultdict(Counter)
        for path in unique_paths:
            current_cluster = path[i]
            next_cluster = path[i + 1]
            transitions[current_cluster][next_cluster] += 1
        
        # Calculate coherence for each source cluster
        coherence_scores = []
        for source_cluster, targets in transitions.items():
            total = sum(targets.values())
            # Get proportion going to the most common target
            most_common = targets.most_common(1)[0][1] if targets else 0
            coherence = most_common / total if total > 0 else 0
            coherence_scores.append(coherence)
        
        # Average coherence across all source clusters
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0
        coherence_by_layer[current_layer] = avg_coherence
    
    return coherence_by_layer

def compute_cluster_stability(paths_with_centroids_data):
    """
    Compute cluster stability by layer based on centroid distances.
    
    Args:
        paths_with_centroids_data: Data from the paths with centroids JSON file
        
    Returns:
        Dictionary of stability values by layer
    """
    layers = paths_with_centroids_data.get("layers", [])
    centroids = paths_with_centroids_data.get("cluster_centroids", {})
    
    if not layers or not centroids:
        return {}
    
    stability_by_layer = {}
    
    for i, layer in enumerate(layers[:-1]):  # Skip the output layer
        if layer in centroids:
            layer_centroids = centroids[layer]
            
            # Calculate average distance between centroids (if more than one)
            if len(layer_centroids) > 1:
                distances = []
                for j, centroid1 in enumerate(layer_centroids):
                    for k in range(j + 1, len(layer_centroids)):
                        centroid2 = layer_centroids[k]
                        # Calculate Euclidean distance
                        dist = np.sqrt(np.sum(np.square(np.array(centroid1) - np.array(centroid2))))
                        distances.append(dist)
                
                # Normalize to 0-1 range (higher distance = lower stability)
                avg_distance = np.mean(distances) if distances else 0
                max_possible_dist = np.sqrt(len(layer_centroids[0]))  # Max distance in unit hypercube
                
                # Convert distance to stability (1 - normalized distance)
                norm_distance = min(avg_distance / max_possible_dist, 1) if max_possible_dist > 0 else 0
                stability = 1 - norm_distance
                
                stability_by_layer[layer] = stability
            else:
                # If only one centroid, stability is perfect
                stability_by_layer[layer] = 1.0
    
    return stability_by_layer

def compute_membership_overlap(paths_data, cluster_stats_data):
    """
    Compute membership overlap between layers.
    
    Args:
        paths_data: Data from the paths JSON file
        cluster_stats_data: Data from the cluster stats JSON file
        
    Returns:
        Dictionary of membership overlap values by layer
    """
    layers = paths_data.get("layers", [])
    unique_paths = paths_data.get("unique_paths", [])
    
    if not layers or not unique_paths or len(layers) < 2:
        return {}
    
    overlap_by_layer = {}
    
    # For each consecutive layer pair, compute the membership overlap
    for i in range(len(layers) - 2):  # Skip the output layer
        layer1 = layers[i]
        layer2 = layers[i + 1]
        
        # Count how many samples stay together
        clusters_layer1 = defaultdict(set)
        clusters_layer2 = defaultdict(set)
        
        # Assign samples to clusters in each layer
        for path_idx, path in enumerate(unique_paths):
            clusters_layer1[path[i]].add(path_idx)
            clusters_layer2[path[i + 1]].add(path_idx)
        
        # Calculate overlap between clusters
        overlap_scores = []
        for c1, samples1 in clusters_layer1.items():
            for c2, samples2 in clusters_layer2.items():
                # Jaccard overlap
                intersection = len(samples1.intersection(samples2))
                union = len(samples1.union(samples2))
                overlap = intersection / union if union > 0 else 0
                if overlap > 0:  # Only count non-zero overlaps
                    overlap_scores.append(overlap)
        
        # Average overlap
        avg_overlap = np.mean(overlap_scores) if overlap_scores else 0
        overlap_by_layer[layer1] = avg_overlap
    
    return overlap_by_layer

def compute_conceptual_purity(paths_data, cluster_stats_data):
    """
    Compute conceptual purity based on target class distribution.
    
    Args:
        paths_data: Data from the paths JSON file
        cluster_stats_data: Data from the cluster stats JSON file
        
    Returns:
        Dictionary of purity values by layer
    """
    layers = paths_data.get("layers", [])
    
    if not layers or "layers" not in cluster_stats_data:
        return {}
    
    purity_by_layer = {}
    
    # For each layer, compute purity of clusters
    for layer in layers[:-1]:  # Skip the output layer
        if layer in cluster_stats_data.get("layers", {}):
            layer_stats = cluster_stats_data["layers"][layer]
            
            purity_scores = []
            for cluster_id, stats in layer_stats.get("numeric_stats", {}).items():
                if "target" in stats:
                    # Calculate purity based on target distribution
                    mean_target = stats["target"]["mean"]
                    # Purity is how far from 0.5 the target is (either closer to 0 or 1)
                    purity = abs(mean_target - 0.5) * 2  # Scale to 0-1
                    purity_scores.append(purity)
            
            # Average purity across clusters
            avg_purity = np.mean(purity_scores) if purity_scores else 0
            purity_by_layer[layer] = avg_purity
    
    return purity_by_layer

def calculate_all_metrics(paths_data, paths_with_centroids_data, cluster_stats_data):
    """
    Calculate all fragmentation metrics from the available data.
    
    Args:
        paths_data: Data from the paths JSON file
        paths_with_centroids_data: Data from the paths with centroids JSON file
        cluster_stats_data: Data from the cluster stats JSON file
        
    Returns:
        Dictionary of metric values by layer
    """
    metrics = {}
    
    # Get layer names
    layers = paths_data.get("layers", [])[:-1]  # Skip the output layer
    
    # Add entropy fragmentation
    entropy_values = compute_entropy_fragmentation(paths_data)
    if entropy_values:
        metrics["entropy_fragmentation"] = entropy_values
    
    # Add path coherence
    coherence_values = compute_path_coherence(paths_data)
    if coherence_values:
        metrics["path_coherence"] = coherence_values
    
    # Add cluster stability
    stability_values = compute_cluster_stability(paths_with_centroids_data)
    if stability_values:
        metrics["cluster_stability"] = stability_values
    
    # Add membership overlap
    overlap_values = compute_membership_overlap(paths_data, cluster_stats_data)
    if overlap_values:
        metrics["membership_overlap"] = overlap_values
    
    # Add conceptual purity
    purity_values = compute_conceptual_purity(paths_data, cluster_stats_data)
    if purity_values:
        metrics["conceptual_purity"] = purity_values
    
    return metrics, layers

def plot_metric_trajectories(metrics, layers, output_dir):
    """
    Create plots of fragmentation metrics across layers.
    
    Args:
        metrics: Dictionary of metric dictionaries by layer
        layers: List of layer names
        output_dir: Directory to save output figures
        
    Returns:
        List of paths to saved figures
    """
    output_paths = []
    
    # Create a directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a separate figure for each metric
    for metric_name, metric_dict in metrics.items():
        # Get values for each layer, in order
        metric_values = [metric_dict.get(layer, np.nan) for layer in layers]
        
        # Skip metrics with insufficient data
        if sum(~np.isnan(metric_values)) < 2:
            logger.warning(f"Skipping {metric_name}: insufficient data")
            continue
        
        # Create figure
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Format layer names for display
        x_labels = [f"Layer {i+1}" for i in range(len(layers))]
        
        # Plot the metric
        plt.plot(
            x_labels,
            metric_values,
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8
        )
        
        # Format metric name for title and labels
        metric_display_name = metric_name.replace('_', ' ').title()
        
        # Add title and labels
        plt.title(f"Heart Disease Dataset - {metric_display_name} Across Layers", fontsize=14)
        plt.xlabel("Network Layer", fontsize=12)
        plt.ylabel(metric_display_name, fontsize=12)
        
        # Set y-axis limits based on metric type
        if 'fragmentation' in metric_name or 'overlap' in metric_name or 'coherence' in metric_name or 'stability' in metric_name or 'purity' in metric_name:
            plt.ylim(0, 1)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = f"heart_seed0_{metric_name}_trajectory.png"
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved {metric_name} trajectory to {output_path}")
        output_paths.append(output_path)
    
    # Create a combined figure with all metrics
    plt.figure(figsize=(12, 8), dpi=300)
    
    x_labels = [f"Layer {i+1}" for i in range(len(layers))]
    
    # Plot each metric
    for metric_name, metric_dict in metrics.items():
        # Get values for each layer, in order
        metric_values = [metric_dict.get(layer, np.nan) for layer in layers]
        
        # Skip metrics with insufficient data
        if sum(~np.isnan(metric_values)) < 2:
            continue
        
        metric_display_name = metric_name.replace('_', ' ').title()
        
        plt.plot(
            x_labels,
            metric_values,
            marker='o',
            linestyle='-',
            linewidth=2.5,
            markersize=8,
            label=metric_display_name
        )
    
    # Add title and labels
    plt.title("Heart Disease Dataset - Fragmentation Metrics Across Layers", fontsize=16)
    plt.xlabel("Network Layer", fontsize=14)
    plt.ylabel("Metric Value (0-1 scale)", fontsize=14)
    
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save combined figure
    output_file = "heart_seed0_all_metrics_trajectory.png"
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved combined metrics trajectory to {output_path}")
    output_paths.append(output_path)
    
    return output_paths

def create_plotly_metrics_chart(metrics, layers, output_dir):
    """
    Create an interactive Plotly chart of metrics across layers.
    
    Args:
        metrics: Dictionary of metric dictionaries by layer
        layers: List of layer names
        output_dir: Directory to save output figures
        
    Returns:
        Path to saved HTML file
    """
    # Format layer names for display
    x_labels = [f"Layer {i+1}" for i in range(len(layers))]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each metric
    for metric_name, metric_dict in metrics.items():
        # Get values for each layer, in order
        metric_values = [metric_dict.get(layer, np.nan) for layer in layers]
        
        # Skip metrics with insufficient data
        if sum(~np.isnan(metric_values)) < 2:
            continue
        
        # Create hover text with exact values
        hovertext = [f"{val:.3f}" if not np.isnan(val) else "N/A" for val in metric_values]
        
        metric_display_name = metric_name.replace('_', ' ').title()
        
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=metric_values,
            mode='lines+markers',
            name=metric_display_name,
            hoverinfo='text',
            hovertext=hovertext,
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    # Update layout
    fig.update_layout(
        title="Heart Disease Dataset - Fragmentation Metrics Across Layers",
        xaxis_title="Network Layer",
        yaxis_title="Metric Value (0-1 scale)",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        width=900,
        height=600
    )
    
    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HTML and PNG
    html_output = os.path.join(output_dir, "heart_seed0_metrics_interactive.html")
    png_output = os.path.join(output_dir, "heart_seed0_metrics_interactive.png")
    
    fig.write_html(html_output)
    fig.write_image(png_output)
    
    logger.info(f"Saved interactive metrics chart to {html_output}")
    logger.info(f"Saved static version to {png_output}")
    
    return html_output

def main():
    """Main function to create fragmentation metrics visualizations."""
    logger.info("Generating fragmentation metrics charts for Heart dataset")
    
    # Ensure the arxiv figures directory exists
    os.makedirs(ARXIV_FIGURES_DIR, exist_ok=True)
    
    # Load the data
    paths_data = load_data(PATHS_FILE)
    paths_with_centroids_data = load_data(PATHS_WITH_CENTROIDS_FILE)
    cluster_stats_data = load_data(CLUSTER_STATS_FILE)
    
    if not paths_data:
        logger.error(f"Failed to load paths data from {PATHS_FILE}")
        return 1
    
    if not paths_with_centroids_data:
        logger.warning(f"Failed to load paths with centroids data from {PATHS_WITH_CENTROIDS_FILE}")
    
    if not cluster_stats_data:
        logger.warning(f"Failed to load cluster stats data from {CLUSTER_STATS_FILE}")
    
    # Calculate all metrics
    metrics, layers = calculate_all_metrics(
        paths_data, 
        paths_with_centroids_data, 
        cluster_stats_data
    )
    
    if not metrics:
        logger.error("Failed to calculate metrics from the data")
        return 1
    
    # Plot metric trajectories
    output_paths = plot_metric_trajectories(metrics, layers, ARXIV_FIGURES_DIR)
    
    # Create interactive Plotly chart
    html_path = create_plotly_metrics_chart(metrics, layers, ARXIV_FIGURES_DIR)
    
    logger.info(f"Created {len(output_paths)} metric visualizations for Heart dataset")
    logger.info(f"Output saved to {ARXIV_FIGURES_DIR}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())