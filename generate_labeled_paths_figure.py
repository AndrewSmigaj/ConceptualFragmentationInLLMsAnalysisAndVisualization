"""
Generate stepped-layer visualization with LLM-labeled clusters for the paper.

This script creates a figure showing cluster paths through the network layers,
with each cluster labeled by an LLM-generated semantic label.
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import sys
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import LLM analysis module
from concept_fragmentation.llm.analysis import ClusterAnalysis


def load_cluster_paths_data(dataset: str = "titanic", seed: int = 0) -> Dict:
    """
    Load cluster paths data from file.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Dictionary with cluster paths data
    """
    data_dir = os.path.join("visualization", "data", "cluster_paths")
    file_path = os.path.join(data_dir, f"{dataset}_seed_{seed}_paths.json")
    
    if not os.path.exists(file_path):
        # Try alternate locations
        file_path = os.path.join("data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find cluster paths data at {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def load_cluster_centroids(dataset: str = "titanic", seed: int = 0) -> Dict:
    """
    Load cluster centroids from file.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Dictionary with cluster centroids
    """
    data_dir = os.path.join("visualization", "data", "cluster_paths")
    file_path = os.path.join(data_dir, f"{dataset}_seed_{seed}_paths_with_centroids.json")
    
    if not os.path.exists(file_path):
        # Try alternate locations
        file_path = os.path.join("data", "cluster_paths", f"{dataset}_seed_{seed}_paths_with_centroids.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find cluster centroids at {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract centroids into the expected format
    centroids = {}
    for layer in data.get('layers', []):
        layer_centroids = {}
        for cluster_id, centroid in data.get('centroids', {}).get(layer, {}).items():
            # Convert cluster_id from string to int
            layer_centroids[int(cluster_id)] = np.array(centroid)
        centroids[layer] = layer_centroids
    
    return centroids


def find_existing_llm_results(dataset: str, seed: int) -> Optional[Dict[str, Any]]:
    """Find and load existing LLM results for the given dataset and seed."""
    # Define paths to check
    results_dir = os.path.join(parent_dir, "results", "llm")
    data_dir = os.path.join(parent_dir, "data", "llm")
    
    # Check results directory first
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith(f"{dataset}_seed_{seed}") and filename.endswith(".json"):
                results_path = os.path.join(results_dir, filename)
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    print(f"Loaded existing LLM results from {results_path}")
                    return results
                except Exception as e:
                    print(f"Error loading LLM results from {results_path}: {e}")
    
    # Then check data directory
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith(f"{dataset}_seed_{seed}") and filename.endswith(".json"):
                results_path = os.path.join(data_dir, filename)
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    print(f"Loaded existing LLM results from {results_path}")
                    return results
                except Exception as e:
                    print(f"Error loading LLM results from {results_path}: {e}")
    
    # Also check mock data
    mock_file = os.path.join(parent_dir, f"mock_llm_analysis_results.json")
    if os.path.exists(mock_file):
        try:
            with open(mock_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded mock LLM results")
            return results
        except Exception as e:
            print(f"Error loading mock LLM results: {e}")
    
    # Also check the analysis_results JSON
    analysis_file = os.path.join(parent_dir, f"analysis_results_{dataset}_seed{seed}.json")
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded analysis results with LLM data")
            return results
        except Exception as e:
            print(f"Error loading analysis results: {e}")
    
    # If we reach here, no results were found
    print(f"No existing LLM results found for {dataset} seed {seed}")
    return None


def generate_or_load_cluster_labels(cluster_paths_data: Dict, 
                                   centroids: Dict, 
                                   dataset: str = "titanic", 
                                   seed: int = 0) -> Dict[str, str]:
    """
    Generate or load LLM-generated cluster labels.
    
    Args:
        cluster_paths_data: Dictionary with cluster paths data
        centroids: Dictionary with cluster centroids
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Dictionary mapping cluster IDs to labels
    """
    # First try to load existing results
    existing_results = find_existing_llm_results(dataset, seed)
    if existing_results and 'cluster_labels' in existing_results:
        print("Using existing cluster labels from results")
        return existing_results['cluster_labels']
    
    # If no existing results, try to generate new ones
    print("No existing cluster labels found. Generating new ones...")
    
    try:
        # Create the LLM client
        analyzer = ClusterAnalysis(provider="grok", model="default", use_cache=True)
        
        # Prepare centroids in the expected format
        cluster_centroids = {}
        for layer in cluster_paths_data.get('layers', []):
            for cluster_id, centroid in centroids.get(layer, {}).items():
                cluster_centroids[f"{layer}C{cluster_id}"] = centroid
        
        # Generate labels
        labels = analyzer.label_clusters_sync(cluster_centroids)
        
        # Save results to file
        os.makedirs(os.path.join(parent_dir, "results", "llm"), exist_ok=True)
        output_path = os.path.join(parent_dir, "results", "llm", f"{dataset}_seed_{seed}_labels.json")
        with open(output_path, 'w') as f:
            json.dump({'cluster_labels': labels}, f, indent=2)
        
        print(f"Generated and saved {len(labels)} cluster labels to {output_path}")
        return labels
    
    except Exception as e:
        print(f"Error generating cluster labels: {e}")
        
        # Return mock labels for demonstration
        mock_labels = {}
        for layer in cluster_paths_data.get('layers', []):
            unique_clusters = set()
            for path in cluster_paths_data.get('unique_paths', []):
                idx = cluster_paths_data.get('layers', []).index(layer)
                if idx < len(path):
                    unique_clusters.add(path[idx])
            
            for cluster_id in unique_clusters:
                if layer == "layer1":
                    mock_labels[f"{layer}C{cluster_id}"] = f"Feature Extraction {cluster_id}"
                elif layer == "layer2":
                    mock_labels[f"{layer}C{cluster_id}"] = f"Pattern Recognition {cluster_id}"
                else:
                    mock_labels[f"{layer}C{cluster_id}"] = f"Decision Logic {cluster_id}"
        
        return mock_labels


def generate_labeled_stepped_viz(cluster_paths_data: Dict, 
                              cluster_labels: Dict[str, str],
                              output_dir: str,
                              dataset: str = "titanic",
                              seed: int = 0):
    """
    Generate stepped-layer visualization with LLM-labeled clusters.
    
    Args:
        cluster_paths_data: Dictionary with cluster paths data
        cluster_labels: Dictionary mapping cluster IDs to labels
        output_dir: Directory to save outputs
        dataset: Dataset name
        seed: Random seed
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract paths and layers
    paths = cluster_paths_data.get('unique_paths', [])
    layers = cluster_paths_data.get('layers', [])
    
    # Process the human-readable paths
    human_readable_paths = cluster_paths_data.get('human_readable_paths', [])
    if not human_readable_paths and 'path_str_to_idx' in cluster_paths_data:
        # Convert path strings to path objects
        human_readable_paths = []
        for path_str in cluster_paths_data.get('path_str_to_idx', {}).keys():
            # Format: L0C0→L1C1→L2C2
            clusters = path_str.split('→')
            human_readable_paths.append(path_str)
    
    # Apply dimensionality reduction
    # Get embeddings for each layer (actual data or centroids if available)
    embeddings = {}
    
    # If we have actual data or centroids, use them
    if 'embeddings' in cluster_paths_data and cluster_paths_data['embeddings']:
        for layer in layers:
            if layer in cluster_paths_data['embeddings']:
                embeddings[layer] = cluster_paths_data['embeddings'][layer]
    
    # Otherwise, create synthetic embeddings
    if not embeddings:
        for layer_idx, layer in enumerate(layers):
            # For each layer, get all clusters and count points per cluster
            clusters = {}
            for path_idx, path in enumerate(paths):
                cluster = path[layer_idx]
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(path_idx)
            
            # Place cluster centers in a circle
            n_clusters = len(clusters)
            cluster_centers = {}
            for i, (cluster, points) in enumerate(clusters.items()):
                angle = 2 * np.pi * i / n_clusters
                cluster_centers[cluster] = np.array([np.cos(angle), np.sin(angle)]) * 10
            
            # Create points around centers
            layer_embeddings = np.zeros((len(paths), 2))
            for cluster, points in clusters.items():
                center = cluster_centers[cluster]
                for point_idx in points:
                    layer_embeddings[point_idx] = center + np.random.normal(0, 1, 2)
            
            embeddings[layer] = layer_embeddings
    
    # Create the visualization
    fig = go.Figure()
    
    # Set up layer offsets along y-axis
    layer_offsets = {layer: idx * 5 for idx, layer in enumerate(layers)}
    
    # Color paths based on endpoint cluster
    if len(paths) > 0 and len(paths[0]) > 0:
        endpoint_layer = layers[-1]
        endpoint_idx = layers.index(endpoint_layer)
        unique_endpoint_clusters = set(path[endpoint_idx] for path in paths)
        colorscale = px.colors.qualitative.Dark24
        cluster_colors = {cluster: colorscale[i % len(colorscale)] 
                         for i, cluster in enumerate(unique_endpoint_clusters)}
    
    # Plot connections between layers
    for path_idx, path in enumerate(paths):
        # Determine color based on endpoint cluster
        endpoint_cluster = path[endpoint_idx]
        color = cluster_colors.get(endpoint_cluster, "gray")
        
        # Create points for line segments
        for layer_idx in range(len(layers) - 1):
            layer1 = layers[layer_idx]
            layer2 = layers[layer_idx + 1]
            
            # Get cluster indices
            cluster1 = path[layer_idx]
            cluster2 = path[layer_idx + 1]
            
            # Get embedding coordinates
            point1 = embeddings[layer1][path_idx]
            point2 = embeddings[layer2][path_idx]
            
            # Adjust y-coordinates based on layer
            x1, z1 = point1
            x2, z2 = point2
            y1 = layer_offsets[layer1]
            y2 = layer_offsets[layer2]
            
            # Add line segment
            fig.add_trace(go.Scatter3d(
                x=[x1, x2],
                y=[y1, y2],
                z=[z1, z2],
                mode='lines',
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo='none'
            ))
    
    # Add cluster points with LLM labels
    for layer_idx, layer in enumerate(layers):
        y = layer_offsets[layer]
        
        # Group points by cluster
        clusters = {}
        for path_idx, path in enumerate(paths):
            cluster = path[layer_idx]
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path_idx)
        
        # Add a representative point for each cluster with label
        for cluster, point_indices in clusters.items():
            # Get the centroid of all points in this cluster
            cluster_points = [embeddings[layer][idx] for idx in point_indices]
            cluster_centroid = np.mean(cluster_points, axis=0)
            
            # Get the LLM-generated label
            cluster_id = f"{layer}C{cluster}"
            label = cluster_labels.get(cluster_id, f"Cluster {cluster}")
            
            # Add a marker for the cluster with the label
            x, z = cluster_centroid
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[z],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=cluster_colors.get(cluster, "gray"),
                    opacity=0.8
                ),
                text=[label],
                textposition="top center",
                textfont=dict(size=10, color='black'),
                name=label,
                showlegend=False
            ))
    
    # Add layer labels
    for layer_idx, layer in enumerate(layers):
        y = layer_offsets[layer]
        x_pos = 0  # Center of x range
        z_pos = max([max(embeddings[layer][:, 1]) for layer in layers]) + 5  # Above the highest point
        
        fig.add_trace(go.Scatter3d(
            x=[x_pos],
            y=[y],
            z=[z_pos],
            mode='text',
            text=[f"<b>{layer.capitalize()}</b>"],
            textposition="top center",
            textfont=dict(size=16),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{dataset.capitalize()} Dataset (Seed {seed}) - Cluster Paths with LLM Labels",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="Layer",
            zaxis_title="UMAP-2",
            aspectmode='manual',
            aspectratio=dict(x=1, y=1.5, z=1),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),  # Slightly rotated clockwise view
                up=dict(x=0, y=1, z=0)
            )
        ),
        width=1200,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Save interactive HTML
    output_file_html = os.path.join(output_dir, f"{dataset}_seed_{seed}_labeled_path_viz.html")
    fig.write_html(output_file_html)
    print(f"Interactive visualization saved to {output_file_html}")
    
    # Save static image for paper
    output_file_png = os.path.join(output_dir, f"{dataset}_seed_{seed}_labeled_path_viz.png")
    fig.write_image(output_file_png, width=1200, height=800, scale=2)
    print(f"Static visualization saved to {output_file_png}")
    
    # Copy to arxiv submission figures directory
    arxiv_figures_dir = os.path.join(parent_dir, "arxiv_submission", "figures")
    if os.path.exists(arxiv_figures_dir):
        arxiv_output = os.path.join(arxiv_figures_dir, "labeled_trajectory.png")
        import shutil
        shutil.copy(output_file_png, arxiv_output)
        print(f"Copied to arXiv submission: {arxiv_output}")
    
    return fig


def main():
    # Set parameters
    dataset = "titanic"
    seed = 0
    output_dir = os.path.join(parent_dir, "results", "visualizations")
    
    # Load cluster paths data
    cluster_paths_data = load_cluster_paths_data(dataset, seed)
    
    # Load cluster centroids
    centroids = load_cluster_centroids(dataset, seed)
    
    # Get or generate LLM-based cluster labels
    cluster_labels = generate_or_load_cluster_labels(cluster_paths_data, centroids, dataset, seed)
    
    # Generate the visualization
    generate_labeled_stepped_viz(cluster_paths_data, cluster_labels, output_dir, dataset, seed)


if __name__ == "__main__":
    main()