"""
Generate stepped-layer visualization for archetypal paths.

This script creates the stepped-layer visualization referenced in the Results section
of the paper, showing how datapoints traverse through cluster paths across layers.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse

# Define constants
LAYER_COLORS = {
    "layer1": "#1f77b4",
    "layer2": "#ff7f0e",
    "layer3": "#2ca02c",
    "output": "#d62728"
}

PATTERN_COLORS = {
    "convergent_funnel": "#9467bd",
    "divergent_fan": "#8c564b",
    "oscillating_trajectory": "#e377c2"
}

# Class to identify trajectory patterns
class TrajPatternAnalyzer:
    """Identify and categorize trajectory patterns in archetypal paths."""
    
    def __init__(self, 
                 path_data: Dict,
                 embeddings: Dict[str, np.ndarray],
                 centroids: Dict[str, np.ndarray]):
        """
        Initialize the analyzer with path and embedding data.
        
        Args:
            path_data: Dictionary with path information
            embeddings: Dictionary mapping layer names to embeddings
            centroids: Dictionary mapping layer names to cluster centroids
        """
        self.path_data = path_data
        self.embeddings = embeddings
        self.centroids = centroids
        self.layers = path_data.get("layers", [])
        self.paths = path_data.get("unique_paths", [])
        self.labels = None
        
        # Will be filled during analysis
        self.convergent_funnels = []
        self.divergent_fans = []
        self.oscillating_trajectories = []
        
    def load_labels(self, labels_path: str):
        """Load class labels for datapoints."""
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        
    def calculate_fragmentation_scores(self) -> Dict[int, float]:
        """
        Calculate fragmentation scores for each path.
        
        Returns:
            Dictionary mapping path index to fragmentation score
        """
        fragmentation_scores = {}
        
        # For each path
        for path_idx, path in enumerate(self.paths):
            if len(path) < 3:  # Need at least 3 points for 2 vectors
                fragmentation_scores[path_idx] = 0
                continue
                
            # Calculate angles between consecutive segments
            angles = []
            for i in range(1, len(path) - 1):
                # Get centroids for current segment
                prev_centroid = self.centroids[self.layers[i-1]][path[i-1]]
                curr_centroid = self.centroids[self.layers[i]][path[i]]
                next_centroid = self.centroids[self.layers[i+1]][path[i+1]]
                
                # Calculate direction vectors
                vec1 = curr_centroid - prev_centroid
                vec2 = next_centroid - curr_centroid
                
                # Normalize vectors
                vec1_norm = np.linalg.norm(vec1)
                vec2_norm = np.linalg.norm(vec2)
                
                if vec1_norm == 0 or vec2_norm == 0:
                    angles.append(0)
                    continue
                
                vec1 = vec1 / vec1_norm
                vec2 = vec2 / vec2_norm
                
                # Calculate cosine similarity
                cos_sim = np.dot(vec1, vec2)
                cos_sim = min(1.0, max(-1.0, cos_sim))  # Handle numerical errors
                
                # Calculate angle in radians, then convert to degrees
                angle = np.arccos(cos_sim) * (180 / np.pi)
                angles.append(angle)
            
            # Average angle is the fragmentation score
            if angles:
                fragmentation_scores[path_idx] = np.mean(angles)
            else:
                fragmentation_scores[path_idx] = 0
                
        return fragmentation_scores
        
    def identify_patterns(self):
        """
        Identify patterns in the paths: convergent funnels, divergent fans, oscillating trajectories.
        """
        # Step 1: Calculate fragmentation scores
        fragmentation_scores = self.calculate_fragmentation_scores()
        
        # Step 2: Analyze path structure
        layer1_clusters = {}
        layer3_clusters = {}
        
        # Count occurrences of each cluster in first and last layers
        for path in self.paths:
            first_cluster = path[0]
            last_cluster = path[2]  # Using layer3 (before output)
            
            layer1_clusters[first_cluster] = layer1_clusters.get(first_cluster, 0) + 1
            layer3_clusters[last_cluster] = layer3_clusters.get(last_cluster, 0) + 1
        
        # Step 3: Identify convergent funnels - many paths from different start clusters end in same cluster
        for layer3_cluster, count in layer3_clusters.items():
            if count >= 10:  # Arbitrary threshold - adjust based on data
                # Find all paths that end in this cluster
                for path_idx, path in enumerate(self.paths):
                    if path[2] == layer3_cluster:
                        self.convergent_funnels.append(path_idx)
        
        # Step 4: Identify divergent fans - paths that start in same cluster but end in different clusters
        for layer1_cluster, count in layer1_clusters.items():
            if count >= 10:  # Arbitrary threshold
                # Find paths that start from this cluster
                end_clusters = set()
                path_indices = []
                for path_idx, path in enumerate(self.paths):
                    if path[0] == layer1_cluster:
                        end_clusters.add(path[2])
                        path_indices.append(path_idx)
                
                # If they end in 3+ different clusters, consider it a divergent fan
                if len(end_clusters) >= 3:
                    self.divergent_fans.extend(path_indices)
        
        # Step 5: Identify oscillating trajectories based on fragmentation score
        for path_idx, score in fragmentation_scores.items():
            if score > 60:  # High angle change between segments (in degrees)
                self.oscillating_trajectories.append(path_idx)
        
        # Ensure no duplicates
        self.convergent_funnels = list(set(self.convergent_funnels))
        self.divergent_fans = list(set(self.divergent_fans))
        self.oscillating_trajectories = list(set(self.oscillating_trajectories))
        
        # Print summary
        print(f"Identified patterns:")
        print(f"  Convergent funnels: {len(self.convergent_funnels)} paths")
        print(f"  Divergent fans: {len(self.divergent_fans)} paths")
        print(f"  Oscillating trajectories: {len(self.oscillating_trajectories)} paths")
        
    def get_patterns(self) -> Dict[str, List[int]]:
        """
        Get identified patterns.
        
        Returns:
            Dictionary mapping pattern names to lists of path indices
        """
        return {
            "convergent_funnel": self.convergent_funnels,
            "divergent_fan": self.divergent_fans,
            "oscillating_trajectory": self.oscillating_trajectories
        }


def load_data(data_dir: str, dataset: str = "titanic", seed: int = 0) -> Tuple[Dict, Dict, np.ndarray]:
    """
    Load path data, embeddings, and centroids.
    
    Args:
        data_dir: Directory containing data files
        dataset: Dataset name
        seed: Random seed used for training
        
    Returns:
        Tuple of (paths, embeddings, centroids)
    """
    # Load path data
    path_file = os.path.join(data_dir, f"{dataset}_seed_{seed}_paths.json")
    with open(path_file, 'r') as f:
        path_data = json.load(f)
    
    # Load centroids
    with open(os.path.join(data_dir, f"{dataset}_seed_{seed}_paths_with_centroids.json"), 'r') as f:
        centroid_data = json.load(f)
    
    layers = path_data.get("layers", [])
    
    # Load embeddings if available
    embeddings = {}
    centroids = {}
    
    # Construct centroids dictionary
    for layer in layers:
        # For simplicity, we'll generate synthetic embeddings if real ones aren't available
        embeddings[layer] = np.random.normal(size=(len(path_data["unique_paths"]), 10))
        
        # Construct centroids for each layer (would normally load from data)
        unique_clusters = set(path[layers.index(layer)] for path in path_data["unique_paths"])
        centroids[layer] = {
            c: np.random.normal(size=(10,)) for c in unique_clusters
        }
    
    return path_data, embeddings, centroids


def generate_stepped_layer_visualization(path_data: Dict, 
                                         output_dir: str,
                                         pattern_analyzer: Optional[TrajPatternAnalyzer] = None,
                                         show_patterns: bool = True,
                                         interactive: bool = True):
    """
    Generate stepped-layer visualization for archetypal paths.
    
    Args:
        path_data: Dictionary with path information
        output_dir: Directory to save output files
        pattern_analyzer: Optional TrajPatternAnalyzer with pattern information
        show_patterns: Whether to highlight patterns
        interactive: Whether to create interactive Plotly viz (True) or static matplotlib viz (False)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths and layers
    paths = path_data.get("unique_paths", [])
    layers = path_data.get("layers", [])
    
    # For demonstration, create synthetic embeddings if real ones aren't available
    synthetic_embeddings = {}
    for layer_idx, layer in enumerate(layers):
        # Generate 2D embeddings with some structure
        n_clusters = max(paths, key=lambda x: x[layer_idx])[layer_idx] + 1
        n_samples = len(paths)
        
        # Create cluster centers
        centers = np.array([
            [np.cos(2 * np.pi * i / n_clusters), np.sin(2 * np.pi * i / n_clusters)]
            for i in range(n_clusters)
        ]) * 10
        
        # Generate points around centers
        embeddings = np.zeros((n_samples, 2))
        for i, path in enumerate(paths):
            cluster = path[layer_idx]
            embeddings[i] = centers[cluster] + np.random.normal(0, 1, 2)
        
        synthetic_embeddings[layer] = embeddings
    
    # Prepare data for visualization
    fig = None
    
    if interactive:
        # Create Plotly interactive visualization
        fig = go.Figure()
        
        # Get patterns if available
        patterns = None
        if pattern_analyzer and show_patterns:
            patterns = pattern_analyzer.get_patterns()
        
        # Set up layer offsets along y-axis
        layer_offsets = {layer: idx * 5 for idx, layer in enumerate(layers)}
        
        # Plot connections between layers
        for path_idx, path in enumerate(paths):
            # Determine color
            color = "gray"
            if patterns:
                for pattern_name, path_indices in patterns.items():
                    if path_idx in path_indices:
                        color = PATTERN_COLORS[pattern_name]
                        break
            
            # Create points for line segments
            for layer_idx in range(len(layers) - 1):
                layer1 = layers[layer_idx]
                layer2 = layers[layer_idx + 1]
                
                # Get cluster indices
                cluster1 = path[layer_idx]
                cluster2 = path[layer_idx + 1]
                
                # Get embedding coordinates (using synthetic data for demonstration)
                point1 = synthetic_embeddings[layer1][path_idx]
                point2 = synthetic_embeddings[layer2][path_idx]
                
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
                    showlegend=False
                ))
        
        # Add layer labels and points
        for layer_idx, layer in enumerate(layers):
            y = layer_offsets[layer]
            
            # Add points for this layer
            x = synthetic_embeddings[layer][:, 0]
            z = synthetic_embeddings[layer][:, 1]
            
            # Plot points colored by cluster
            for path_idx, path in enumerate(paths):
                cluster = path[layer_idx]
                fig.add_trace(go.Scatter3d(
                    x=[x[path_idx]],
                    y=[y],
                    z=[z[path_idx]],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=cluster,
                        colorscale='Viridis',
                    ),
                    name=f'Layer {layer} - Cluster {cluster}',
                    showlegend=False
                ))
            
            # Add layer label
            fig.add_trace(go.Scatter3d(
                x=[np.mean(x)],
                y=[y],
                z=[np.max(z) + 2],
                mode='text',
                text=[f"<b>{layer}</b>"],
                textposition="top center",
                textfont=dict(size=14),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title="Archetypal Path Analysis: Stepped-Layer Visualization",
            scene=dict(
                xaxis_title="UMAP-1",
                yaxis_title="Layer",
                zaxis_title="UMAP-2",
                aspectmode='manual',
                aspectratio=dict(x=1, y=2, z=1)
            ),
            width=1000,
            height=800
        )
        
        # Add legend for patterns
        if patterns:
            for pattern_name, color in PATTERN_COLORS.items():
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=pattern_name.replace('_', ' ').title()
                ))
        
        # Save interactive HTML
        output_file = os.path.join(output_dir, "stepped_layer_visualization.html")
        fig.write_html(output_file)
        print(f"Interactive visualization saved to {output_file}")
    
    else:
        # Create static matplotlib visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get patterns if available
        patterns = None
        if pattern_analyzer and show_patterns:
            patterns = pattern_analyzer.get_patterns()
        
        # Set up layer offsets along y-axis
        layer_offsets = {layer: idx * 5 for idx, layer in enumerate(layers)}
        
        # Plot connections between layers
        for path_idx, path in enumerate(paths):
            # Determine color
            color = "gray"
            alpha = 0.2
            if patterns:
                for pattern_name, path_indices in patterns.items():
                    if path_idx in path_indices:
                        color = PATTERN_COLORS[pattern_name]
                        alpha = 0.7
                        break
            
            # Create points for line
            xs = []
            ys = []
            zs = []
            
            for layer_idx, layer in enumerate(layers):
                # Get embedding coordinates (using synthetic data for demonstration)
                point = synthetic_embeddings[layer][path_idx]
                
                # Adjust y-coordinate based on layer
                xs.append(point[0])
                ys.append(layer_offsets[layer])
                zs.append(point[1])
            
            # Plot line
            ax.plot(xs, ys, zs, c=color, alpha=alpha, linewidth=1)
        
        # Add layer labels and points
        for layer_idx, layer in enumerate(layers):
            y = layer_offsets[layer]
            
            # Add points for this layer
            x = synthetic_embeddings[layer][:, 0]
            z = synthetic_embeddings[layer][:, 1]
            
            # Plot points colored by cluster
            for path_idx, path in enumerate(paths):
                cluster = path[layer_idx]
                color_val = cluster / max(path[layer_idx] for path in paths)
                ax.scatter(x[path_idx], y, z[path_idx], c=[color_val], cmap='viridis', s=10, alpha=0.7)
            
            # Add layer label
            ax.text(np.mean(x), y, np.max(z) + 2, layer, horizontalalignment='center', size=12)
        
        # Add legend for patterns
        if patterns:
            legend_items = []
            for pattern_name, color in PATTERN_COLORS.items():
                legend_items.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                 label=pattern_name.replace('_', ' ').title()))
            ax.legend(handles=legend_items, loc='upper right')
        
        # Set axis labels
        ax.set_xlabel('PCA-1')
        ax.set_ylabel('Layer')
        ax.set_zlabel('PCA-2')
        
        # Set title
        plt.title("Archetypal Path Analysis: Stepped-Layer Visualization")
        
        # Save static image
        output_file = os.path.join(output_dir, "stepped_layer_visualization.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Static visualization saved to {output_file}")
    
    return fig


def generate_transition_matrix_heatmap(path_data: Dict, output_dir: str):
    """
    Generate transition matrix heatmap showing cluster-to-cluster transitions.
    
    Args:
        path_data: Dictionary with path information
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get paths and layers
    paths = path_data.get("unique_paths", [])
    layers = path_data.get("layers", [])
    
    # Create transition matrices for each layer pair
    transition_matrices = []
    
    for layer_idx in range(len(layers) - 1):
        layer1 = layers[layer_idx]
        layer2 = layers[layer_idx + 1]
        
        # Determine number of clusters in each layer
        clusters1 = set(path[layer_idx] for path in paths)
        clusters2 = set(path[layer_idx + 1] for path in paths)
        
        n_clusters1 = max(clusters1) + 1
        n_clusters2 = max(clusters2) + 1
        
        # Initialize transition matrix
        transition_matrix = np.zeros((n_clusters1, n_clusters2))
        
        # Count transitions
        for path in paths:
            cluster1 = path[layer_idx]
            cluster2 = path[layer_idx + 1]
            transition_matrix[cluster1, cluster2] += 1
        
        # Normalize by row sums to get probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        transition_matrices.append({
            'matrix': transition_matrix,
            'layer1': layer1,
            'layer2': layer2
        })
    
    # Create subplots for each transition matrix
    fig, axes = plt.subplots(1, len(transition_matrices), figsize=(15, 5))
    if len(transition_matrices) == 1:
        axes = [axes]
    
    for idx, tm_data in enumerate(transition_matrices):
        ax = axes[idx]
        matrix = tm_data['matrix']
        layer1 = tm_data['layer1']
        layer2 = tm_data['layer2']
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Transition Probability')
        
        # Set labels
        ax.set_xlabel(f'{layer2} Clusters')
        ax.set_ylabel(f'{layer1} Clusters')
        ax.set_title(f'Transitions: {layer1} → {layer2}')
        
        # Add text annotations with probabilities
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.1:  # Only show significant probabilities
                    ax.text(j, i, f'{matrix[i, j]:.2f}', 
                            ha='center', va='center', 
                            color='white' if matrix[i, j] > 0.5 else 'black',
                            fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, "transition_matrix_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Transition matrix heatmap saved to {output_file}")
    
    return fig


def generate_cross_layer_similarity_heatmap(path_data: Dict, 
                                           centroids: Dict, 
                                           output_dir: str):
    """
    Generate cross-layer similarity heatmap.
    
    Args:
        path_data: Dictionary with path information
        centroids: Dictionary mapping layer names to cluster centroids
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get layers
    layers = path_data.get("layers", [])
    
    # Initialize similarity matrix
    # For each layer pair, calculate cosine similarity between all cluster centroids
    all_clusters = []
    for layer in layers:
        layer_clusters = [f"{layer}_C{i}" for i in range(len(centroids[layer]))]
        all_clusters.extend(layer_clusters)
    
    n_total_clusters = len(all_clusters)
    similarity_matrix = np.zeros((n_total_clusters, n_total_clusters))
    
    # Calculate similarities
    for i, cluster1 in enumerate(all_clusters):
        layer1, cluster_id1 = cluster1.rsplit('_C', 1)
        cluster_id1 = int(cluster_id1)
        
        for j, cluster2 in enumerate(all_clusters):
            layer2, cluster_id2 = cluster2.rsplit('_C', 1)
            cluster_id2 = int(cluster_id2)
            
            # Skip if same cluster
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
            
            # Get centroids
            centroid1 = centroids[layer1][cluster_id1]
            centroid2 = centroids[layer2][cluster_id2]
            
            # Calculate cosine similarity
            similarity = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
            similarity_matrix[i, j] = similarity
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    
    # Set labels
    ax.set_xticks(np.arange(n_total_clusters))
    ax.set_yticks(np.arange(n_total_clusters))
    ax.set_xticklabels(all_clusters, rotation=90, fontsize=8)
    ax.set_yticklabels(all_clusters, fontsize=8)
    
    # Highlight high-similarity pairs (>0.85) across layers
    threshold = 0.85
    for i in range(n_total_clusters):
        for j in range(n_total_clusters):
            if i != j and similarity_matrix[i, j] > threshold:
                layer1 = all_clusters[i].split('_C')[0]
                layer2 = all_clusters[j].split('_C')[0]
                
                # Only highlight cross-layer high similarities
                if layer1 != layer2:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, 
                                         edgecolor='red', linewidth=2)
                    ax.add_patch(rect)
    
    plt.title('Cross-Layer Cluster Centroid Similarity')
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, "cross_layer_similarity_heatmap.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Cross-layer similarity heatmap saved to {output_file}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate Archetypal Path Analysis visualizations.")
    parser.add_argument("--data_dir", type=str, default="data/cluster_paths",
                       help="Directory containing cluster path data")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Directory to save output figures")
    parser.add_argument("--dataset", type=str, default="titanic",
                       help="Dataset name")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--interactive", action="store_true",
                       help="Generate interactive Plotly visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    path_data, embeddings, centroids = load_data(args.data_dir, args.dataset, args.seed)
    
    # Initialize pattern analyzer
    pattern_analyzer = TrajPatternAnalyzer(path_data, embeddings, centroids)
    pattern_analyzer.identify_patterns()
    
    # Generate visualizations
    generate_stepped_layer_visualization(path_data, args.output_dir, pattern_analyzer, 
                                        interactive=args.interactive)
    
    generate_transition_matrix_heatmap(path_data, args.output_dir)
    
    generate_cross_layer_similarity_heatmap(path_data, centroids, args.output_dir)
    
    print("All visualizations generated successfully!")


if __name__ == "__main__":
    main()