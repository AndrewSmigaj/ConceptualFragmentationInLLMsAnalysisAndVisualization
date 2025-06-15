"""
Path-based metrics for neural network trajectories.

This module provides functions to calculate path-based metrics
for neural network layer trajectories, focusing on metrics that
quantify concept fragmentation across layers.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import Counter, defaultdict

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

def compute_membership_overlap_by_layer(paths_data):
    """
    Compute membership overlap between layers.
    
    Args:
        paths_data: Data from the paths JSON file
        
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

def load_path_data(dataset, seed):
    """
    Load path data files for a dataset.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Tuple of (paths_data, paths_with_centroids_data, cluster_stats_data)
    """
    # Set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try multiple possible paths
    possible_paths = [
        # Data directory
        os.path.join(base_dir, "data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
        # Visualization data directory
        os.path.join(base_dir, "visualization", "data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
        # Results directory
        os.path.join(base_dir, "results", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
    ]
    
    possible_centroids_paths = [
        # Data directory
        os.path.join(base_dir, "data", "cluster_paths", f"{dataset}_seed_{seed}_paths_with_centroids.json"),
        # Visualization data directory
        os.path.join(base_dir, "visualization", "data", "cluster_paths", f"{dataset}_seed_{seed}_paths_with_centroids.json"),
        # Results directory
        os.path.join(base_dir, "results", "cluster_paths", f"{dataset}_seed_{seed}_paths_with_centroids.json"),
    ]
    
    possible_stats_paths = [
        # Data directory
        os.path.join(base_dir, "data", "cluster_stats", f"{dataset}_seed_{seed}.json"),
        # Results directory
        os.path.join(base_dir, "results", "cluster_stats", f"{dataset}_seed_{seed}.json"),
    ]
    
    # Load data files
    paths_data = None
    paths_with_centroids_data = None
    cluster_stats_data = None
    
    # Try to load paths data
    for path in possible_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    paths_data = json.load(f)
                print(f"Loaded paths data from {path}")
                break
        except Exception as e:
            print(f"Error loading paths data from {path}: {e}")
    
    # Try to load paths with centroids data
    for path in possible_centroids_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    paths_with_centroids_data = json.load(f)
                print(f"Loaded paths with centroids data from {path}")
                break
        except Exception as e:
            print(f"Error loading paths with centroids data from {path}: {e}")
    
    # Try to load cluster stats data
    for path in possible_stats_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    cluster_stats_data = json.load(f)
                print(f"Loaded cluster stats data from {path}")
                break
        except Exception as e:
            print(f"Error loading cluster stats data from {path}: {e}")
    
    return paths_data, paths_with_centroids_data, cluster_stats_data

def calculate_all_path_metrics(dataset, seed):
    """
    Load path data and calculate all metrics for a dataset.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Dictionary of all calculated metrics by layer
    """
    # Load data files
    paths_data, paths_with_centroids_data, cluster_stats_data = load_path_data(dataset, seed)
    
    # Calculate metrics
    metrics = {}
    
    # Add entropy fragmentation
    if paths_data:
        entropy_values = compute_entropy_fragmentation(paths_data)
        if entropy_values:
            metrics["entropy_fragmentation"] = entropy_values
    
    # Add path coherence
    if paths_data:
        coherence_values = compute_path_coherence(paths_data)
        if coherence_values:
            metrics["path_coherence"] = coherence_values
    
    # Add cluster stability
    if paths_with_centroids_data:
        stability_values = compute_cluster_stability(paths_with_centroids_data)
        if stability_values:
            metrics["cluster_stability"] = stability_values
    
    # Add membership overlap
    if paths_data:
        overlap_values = compute_membership_overlap_by_layer(paths_data)
        if overlap_values:
            metrics["path_membership_overlap"] = overlap_values
    
    # Add conceptual purity
    if paths_data and cluster_stats_data:
        purity_values = compute_conceptual_purity(paths_data, cluster_stats_data)
        if purity_values:
            metrics["conceptual_purity"] = purity_values
    
    # Get layer names for reference
    if paths_data and "layers" in paths_data:
        metrics["layers"] = paths_data["layers"]
    
    return metrics

def plot_path_metrics(metrics, get_friendly_layer_name=None, height=400, width=800):
    """
    Create plots for path-based metrics.
    
    Args:
        metrics: Dictionary of metrics by layer
        get_friendly_layer_name: Optional function to convert layer names to display names
        height: Plot height
        width: Plot width
        
    Returns:
        Dictionary of plotly figures by metric name
    """
    import plotly.graph_objects as go
    
    figures = {}
    all_layers = []
    
    # Collect all layer names from all metrics
    for metric_name, metric_values in metrics.items():
        if metric_name != "layers" and isinstance(metric_values, dict):
            all_layers.extend(metric_values.keys())
    
    # Remove duplicates and sort
    import re
    def natural_sort_key(s):
        if s == "input":
            return (0, 0)
        elif s == "output":
            return (999, 0)
        else:
            match = re.match(r'layer(\d+)', s)
            if match:
                return (1, int(match.group(1)))
            return (2, s)
    
    all_layers = sorted(list(set(all_layers)), key=natural_sort_key)
    
    # Process each metric
    for metric_name, metric_values in metrics.items():
        if metric_name == "layers" or not isinstance(metric_values, dict):
            continue
        
        # Get values for each layer
        values = []
        layers_with_data = []
        
        for layer in all_layers:
            if layer in metric_values:
                values.append(metric_values[layer])
                layers_with_data.append(layer)
        
        if not values:
            continue
            
        # Get friendly layer names if function provided
        if get_friendly_layer_name:
            x_labels = [get_friendly_layer_name(layer) for layer in layers_with_data]
        else:
            x_labels = layers_with_data
        
        # Create the figure
        fig = go.Figure()
        
        # Add the metric line
        fig.add_trace(go.Scatter(
            x=x_labels,
            y=values,
            mode='lines+markers',
            name=metric_name.replace('_', ' ').title(),
            line=dict(width=2),
            marker=dict(size=8)
        ))
        
        # Format metric name for title
        metric_title = metric_name.replace('_', ' ').title()
        
        # Update layout
        fig.update_layout(
            title=f"{metric_title} by Layer",
            xaxis_title="Layer",
            yaxis_title=metric_title,
            height=height,
            width=width
        )
        
        # For metrics that should be on 0-1 scale
        if any(term in metric_name for term in ['entropy', 'coherence', 'stability', 'overlap', 'purity']):
            fig.update_yaxes(range=[0, 1])
        
        figures[metric_name] = fig
    
    # Create a combined figure with all metrics
    if figures:
        fig = go.Figure()
        
        # Use first metric's x-axis for all
        first_metric = next(iter(figures.keys()))
        x_labels = figures[first_metric].data[0].x
        
        # Add each metric
        for metric_name, metric_fig in figures.items():
            y_values = metric_fig.data[0].y
            metric_title = metric_name.replace('_', ' ').title()
            
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=y_values,
                mode='lines+markers',
                name=metric_title,
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        # Update layout
        fig.update_layout(
            title="Path-Based Metrics by Layer",
            xaxis_title="Layer",
            yaxis_title="Metric Value (0-1 scale)",
            yaxis=dict(range=[0, 1]),
            height=height + 100,
            width=width,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        figures["combined"] = fig
    
    return figures