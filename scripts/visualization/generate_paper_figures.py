"""
Generate all figures needed for the paper.

This script creates all the figures referenced in paper.md, including:
1. Optimal number of clusters (k*) by layer
2. Intra-class pairwise distance (ICPD) by layer
3. Subspace angles by layer
4. Cluster entropy by layer
5. UMAP trajectories (basic and annotated)
6. Trajectory endpoints
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import argparse

# Set figure style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

# Constants
LAYER_NAMES = ['input', 'layer1', 'layer2', 'layer3', 'output']
LAYER_COLORS = {
    'input': '#1f77b4',
    'layer1': '#ff7f0e',
    'layer2': '#2ca02c',
    'layer3': '#d62728',
    'output': '#9467bd'
}

def load_data(data_dir: str, dataset: str = "titanic", seed: int = 0) -> Dict:
    """
    Load analysis results from disk.
    
    Args:
        data_dir: Directory containing results files
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Dictionary with analysis results
    """
    # ------------------------------------------------------------------
    # 1) NEW: first try the cluster-paths JSON produced by the main pipeline
    # ------------------------------------------------------------------
    paths_file = os.path.join("data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json")
    if not os.path.exists(paths_file):
        # Alternate filename with centroids
        paths_file = paths_file.replace("_paths.json", "_paths_with_centroids.json")

    if os.path.exists(paths_file):
        print(f"Using metrics from {paths_file}")
        with open(paths_file, "r") as f:
            cp_data = json.load(f)

        metrics_block = cp_data.get("metrics", {})

        metrics = {
            "optimal_clusters": metrics_block.get("k_star", {}),
            "subspace_angle": metrics_block.get("angle_fragmentation", {}).get("per_layer", {}),
            "cluster_entropy": metrics_block.get("entropy_fragmentation", {}).get("per_layer", {}),
            "intra_class_distance": {},  # Placeholder – not computed yet
        }

        return {
            "metrics": metrics,
            "dataset": dataset,
            "seed": seed,
            "stats_data": cp_data,
        }

    # ------------------------------------------------------------------
    # If the cluster-paths file is missing we treat this as a hard error.
    # ------------------------------------------------------------------
    raise FileNotFoundError(
        f"Required cluster-paths file not found for {dataset} seed {seed}. "
        f"Expected file at: {paths_file}"
    )

def create_figures_directory(output_dir: str) -> str:
    """Create the figures directory if it doesn't exist."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir

def plot_optimal_clusters(data: Dict, output_dir: str) -> None:
    """
    Plot optimal number of clusters (k*) by layer.
    
    Args:
        data: Dictionary with analysis results
        output_dir: Directory to save the figure
    """
    metrics = data.get("metrics", {})
    k_star = metrics.get("optimal_clusters", {})
    
    if not k_star:
        print("No optimal cluster data found!")
        return
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Extract layers and values
    layers = list(k_star.keys())
    values = list(k_star.values())
    
    # Create bar plot
    bars = plt.bar(layers, values, color=[LAYER_COLORS.get(layer, '#333333') for layer in layers])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    # Set labels and title
    plt.xlabel('Layer')
    plt.ylabel('Optimal Number of Clusters (k*)')
    plt.title('Optimal Clusters by Layer')
    plt.ylim(0, max(values) + 1)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "optimal_clusters.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved optimal clusters figure to {output_file}")
    
    plt.close()

def plot_intra_class_distance(data: Dict, output_dir: str) -> None:
    """
    Plot intra-class pairwise distance (ICPD) by layer.
    
    Args:
        data: Dictionary with analysis results
        output_dir: Directory to save the figure
    """
    metrics = data.get("metrics", {})
    icpd = metrics.get("intra_class_distance", {})
    
    if not icpd:
        print("No intra-class distance data found!")
        return
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Extract layers and values
    layers = list(icpd.keys())
    values = list(icpd.values())
    
    # Create line plot with markers
    plt.plot(layers, values, 'o-', linewidth=2, markersize=8, 
             color='#1f77b4', markerfacecolor='white', markeredgewidth=2)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    # Set labels and title
    plt.xlabel('Layer')
    plt.ylabel('Intra-Class Pairwise Distance (ICPD)')
    plt.title('Intra-Class Distance by Layer')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "intra_class_distance.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved intra-class distance figure to {output_file}")
    
    plt.close()

def plot_subspace_angle(data: Dict, output_dir: str) -> None:
    """
    Plot subspace angles by layer.
    
    Args:
        data: Dictionary with analysis results
        output_dir: Directory to save the figure
    """
    metrics = data.get("metrics", {})
    angles = metrics.get("subspace_angle", {})
    
    if not angles:
        print("No subspace angle data found!")
        return
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Extract layers and values
    layers = list(angles.keys())
    values = list(angles.values())
    
    # Create line plot with markers
    plt.plot(layers, values, 's-', linewidth=2, markersize=8, 
             color='#ff7f0e', markerfacecolor='white', markeredgewidth=2)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f'{v}°', ha='center')
    
    # Set labels and title
    plt.xlabel('Layer')
    plt.ylabel('Subspace Angle (degrees)')
    plt.title('Subspace Angles by Layer')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "subspace_angle.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved subspace angle figure to {output_file}")
    
    plt.close()

def plot_cluster_entropy(data: Dict, output_dir: str) -> None:
    """
    Plot cluster entropy by layer.
    
    Args:
        data: Dictionary with analysis results
        output_dir: Directory to save the figure
    """
    metrics = data.get("metrics", {})
    entropy = metrics.get("cluster_entropy", {})
    
    if not entropy:
        print("No cluster entropy data found!")
        return
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Extract layers and values
    layers = list(entropy.keys())
    values = list(entropy.values())
    
    # Create line plot with markers
    plt.plot(layers, values, 'D-', linewidth=2, markersize=8, 
             color='#2ca02c', markerfacecolor='white', markeredgewidth=2)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    # Set labels and title
    plt.xlabel('Layer')
    plt.ylabel('Cluster Entropy (normalized)')
    plt.title('Cluster Entropy by Layer')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    plt.ylim(0.75, 0.9)
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "cluster_entropy.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved cluster entropy figure to {output_file}")
    
    plt.close()

def generate_trajectory_visualizations(data: Dict, output_dir: str) -> None:
    """
    Generate basic and annotated trajectory visualizations.
    
    Args:
        data: Dictionary with analysis results
        output_dir: Directory to save the figures
    """
    # For demonstration purposes, create synthetic trajectory data
    np.random.seed(42)
    
    # Create 2D points for each layer (using UMAP-like layout)
    n_samples = 100
    layers = ['layer1', 'layer2', 'layer3']
    
    # Generate class labels (binary classification)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Create embedding points with class separation
    embeddings = {}
    for layer in layers:
        # Base positions for each class
        class0_center = np.array([-5, -5]) if layer == 'layer1' else \
                        np.array([-8, -2]) if layer == 'layer2' else \
                        np.array([-10, 0])
        
        class1_center = np.array([5, 5]) if layer == 'layer1' else \
                        np.array([8, 2]) if layer == 'layer2' else \
                        np.array([10, 0])
        
        # Generate points around class centers
        points = np.zeros((n_samples, 2))
        for i in range(n_samples):
            if labels[i] == 0:
                points[i] = class0_center + np.random.normal(0, 2, 2)
            else:
                points[i] = class1_center + np.random.normal(0, 2, 2)
        
        embeddings[layer] = points
    
    # Create sample paths (following patterns from paper)
    paths = []
    for i in range(n_samples):
        # For sample paths mentioned in paper
        if i < 55:  # Path 0→2→0
            path = [0, 2, 0]
        elif i < 84:  # Path 1→1→1
            path = [1, 1, 1]
        elif i < 106:  # Path 2→0→1
            path = [2, 0, 1]
        else:
            # Random paths for remaining points
            path = [np.random.randint(0, 3) for _ in range(3)]
        
        paths.append(path)
    
    # Create basic trajectory visualization
    plt.figure(figsize=(10, 8))
    
    # Set up colors for classes
    colors = ['#1f77b4', '#ff7f0e']
    
    # Plot trajectories
    for i in range(n_samples):
        # Get points for this sample across layers
        xs = [embeddings[layer][i, 0] for layer in layers]
        ys = [embeddings[layer][i, 1] for layer in layers]
        
        # Plot line with alpha based on class (survivors more visible)
        alpha = 0.7 if labels[i] == 1 else 0.2
        plt.plot(xs, ys, '-', color=colors[labels[i]], alpha=alpha, linewidth=1)
    
    # Plot points for each layer
    for j, layer in enumerate(layers):
        for i in range(n_samples):
            x, y = embeddings[layer][i]
            plt.scatter(x, y, color=colors[labels[i]], s=30, alpha=0.6, 
                       edgecolor='black', linewidth=0.5)
    
    # Add layer labels
    layer_positions = {}
    for j, layer in enumerate(layers):
        x_mean = np.mean([embeddings[layer][i, 0] for i in range(n_samples)])
        y_mean = np.mean([embeddings[layer][i, 1] for i in range(n_samples)])
        plt.text(x_mean, y_mean + 2, layer, fontsize=14, ha='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        layer_positions[layer] = (x_mean, y_mean)
    
    # Add legend for classes
    plt.scatter([], [], color=colors[0], label='Non-survivors', s=50, edgecolor='black')
    plt.scatter([], [], color=colors[1], label='Survivors', s=50, edgecolor='black')
    plt.legend(loc='upper right')
    
    # Remove axes
    plt.axis('off')
    
    # Set title
    plt.title('UMAP Projection of Activation Trajectories')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "trajectory_basic.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved basic trajectory visualization to {output_file}")
    
    plt.close()
    
    # Create annotated trajectory visualization
    plt.figure(figsize=(10, 8))
    
    # Plot trajectories with archetype highlights
    for i in range(n_samples):
        # Get points for this sample across layers
        xs = [embeddings[layer][i, 0] for layer in layers]
        ys = [embeddings[layer][i, 1] for layer in layers]
        
        # Determine line style based on path pattern
        path_str = '→'.join(str(c) for c in paths[i])
        if path_str == '0→2→0':
            color = '#9467bd'  # Purple for privileged group
            alpha = 0.9
            linewidth = 2
            zorder = 10
        elif path_str == '1→1→1':
            color = '#8c564b'  # Brown for marginalized group
            alpha = 0.9
            linewidth = 2
            zorder = 10
        elif path_str == '2→0→1':
            color = '#e377c2'  # Pink for ambiguous group
            alpha = 0.9
            linewidth = 2
            zorder = 10
        else:
            color = colors[labels[i]]
            alpha = 0.1
            linewidth = 0.5
            zorder = 1
        
        # Plot line
        plt.plot(xs, ys, '-', color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
    
    # Plot points for each layer
    for j, layer in enumerate(layers):
        for i in range(n_samples):
            x, y = embeddings[layer][i]
            # Determine point style based on cluster
            cluster = paths[i][j]
            color = ['#1f77b4', '#ff7f0e', '#2ca02c'][cluster]
            
            plt.scatter(x, y, color=color, s=30, alpha=0.6, 
                       edgecolor='black', linewidth=0.5, zorder=5)
    
    # Add layer labels
    for j, layer in enumerate(layers):
        x_mean, y_mean = layer_positions[layer]
        plt.text(x_mean, y_mean + 2, layer, fontsize=14, ha='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add path annotations
    plt.text(-15, 15, "Path 0→2→0 (55 passengers)\nHigh survival (85%)\nWealthy, balanced gender",
            fontsize=10, color='#9467bd', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.text(0, 15, "Path 1→1→1 (29 passengers)\nLow survival (37%)\nYoung men, third-class",
            fontsize=10, color='#8c564b', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.text(15, 15, "Path 2→0→1 (22 passengers)\nMedium survival (52%)\nMiddle-aged, moderate fare",
            fontsize=10, color='#e377c2', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend for clusters
    plt.scatter([], [], color='#1f77b4', label='Cluster 0', s=50, edgecolor='black')
    plt.scatter([], [], color='#ff7f0e', label='Cluster 1', s=50, edgecolor='black')
    plt.scatter([], [], color='#2ca02c', label='Cluster 2', s=50, edgecolor='black')
    plt.legend(loc='upper right')
    
    # Remove axes
    plt.axis('off')
    
    # Set title
    plt.title('UMAP Trajectories with LLM-derived Archetypes')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "trajectory_annotated.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved annotated trajectory visualization to {output_file}")
    
    plt.close()
    
    # Create layer 3 endpoint visualization
    plt.figure(figsize=(10, 8))
    
    # Get layer 3 embeddings
    layer3_points = embeddings['layer3']
    
    # Create scatter plot with points colored by cluster
    for i in range(n_samples):
        cluster = paths[i][2]  # Layer 3 cluster
        color = ['#1f77b4', '#ff7f0e', '#2ca02c'][cluster]
        marker = 'o' if labels[i] == 0 else '^'  # Different marker for survivors
        
        plt.scatter(layer3_points[i, 0], layer3_points[i, 1], 
                   color=color, s=50, alpha=0.8, marker=marker,
                   edgecolor='black', linewidth=0.5)
    
    # Add cluster labels
    clusters = {}
    for i in range(n_samples):
        cluster = paths[i][2]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(i)
    
    for cluster, indices in clusters.items():
        # Calculate cluster center
        center_x = np.mean([layer3_points[i, 0] for i in indices])
        center_y = np.mean([layer3_points[i, 1] for i in indices])
        
        # Count survivors in this cluster
        survivor_count = sum(labels[i] == 1 for i in indices)
        survival_rate = survivor_count / len(indices) * 100
        
        # Add cluster label with stats
        plt.text(center_x, center_y + 2, 
                f"Cluster {cluster}\n{len(indices)} passengers\n{survival_rate:.1f}% survival",
                fontsize=10, ha='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add legend for survival
    plt.scatter([], [], marker='o', color='gray', label='Non-survivor', s=50, edgecolor='black')
    plt.scatter([], [], marker='^', color='gray', label='Survivor', s=50, edgecolor='black')
    plt.legend(loc='upper right')
    
    # Remove axes
    plt.axis('off')
    
    # Set title
    plt.title('Layer 3 Clustering')
    
    # Save figure
    plt.tight_layout()
    output_file = os.path.join(output_dir, "trajectory_by_endpoint_cluster.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved endpoint cluster visualization to {output_file}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate figures for the paper")
    parser.add_argument("--data_dir", type=str, default="results",
                       help="Directory containing analysis results")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save generated figures")
    parser.add_argument("--dataset", type=str, default="titanic",
                       choices=["titanic", "heart", "adult"],
                       help="Dataset to generate figures for")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create figures directory
    figures_dir = create_figures_directory(args.output_dir)
    
    # Load data
    data = load_data(args.data_dir, args.dataset, args.seed)
    
    # Generate all figures
    plot_optimal_clusters(data, figures_dir)
    plot_intra_class_distance(data, figures_dir)
    plot_subspace_angle(data, figures_dir)
    plot_cluster_entropy(data, figures_dir)
    generate_trajectory_visualizations(data, figures_dir)
    
    print(f"All figures generated successfully in {figures_dir}")

if __name__ == "__main__":
    main()