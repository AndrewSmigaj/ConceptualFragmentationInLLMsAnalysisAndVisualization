"""
Example script that demonstrates Explainable Threshold Similarity (ETS) clustering
on activation data from a trained model.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering, compute_ets_statistics, explain_ets_similarity
)
from concept_fragmentation.metrics.cluster_entropy import compute_cluster_assignments
from concept_fragmentation.visualization.activations import plot_activations_2d, reduce_dimensions
from concept_fragmentation.data.loaders import get_dataset_loader
from concept_fragmentation.utils.helpers import set_seed
from concept_fragmentation.config import METRICS, RANDOM_SEED

def load_activations(dataset_name: str, layer_name: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load activation data from a trained model.
    
    Args:
        dataset_name: Name of the dataset
        layer_name: Name of the layer to load activations from
        seed: Random seed used for the model
        
    Returns:
        Tuple of (activations, labels)
    """
    # Base path for results
    results_dir = os.path.join("results", "baselines", dataset_name, f"baseline_seed{seed}")
    
    # Load activations
    activations_path = os.path.join(results_dir, f"{layer_name}_activations.npy")
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"Activations file not found: {activations_path}")
    
    activations = np.load(activations_path)
    
    # Load labels
    labels_path = os.path.join(results_dir, "y_test.npy")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    labels = np.load(labels_path)
    
    return activations, labels

def generate_plots(
    activations: np.ndarray,
    labels: np.ndarray,
    kmeans_labels: np.ndarray,
    ets_labels: np.ndarray,
    kmeans_silhouette: float,
    ets_silhouette: float,
    output_dir: str,
    prefix: str
):
    """
    Generate comparison plots for k-means and ETS clustering.
    
    Args:
        activations: Activation data
        labels: True class labels
        kmeans_labels: Cluster assignments from k-means
        ets_labels: Cluster assignments from ETS
        kmeans_silhouette: Silhouette score for k-means
        ets_silhouette: Silhouette score for ETS
        output_dir: Directory to save plots
        prefix: Prefix for plot filenames
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Reduce dimensions for visualization
    reduced_act = reduce_dimensions(activations, method='pca', n_components=2)
    
    # Plot 1: By true labels
    fig_true = plt.figure(figsize=(10, 8))
    plt.scatter(reduced_act[:, 0], reduced_act[:, 1], c=labels, cmap='tab10', alpha=0.7, s=50)
    plt.title("Activations Colored by True Labels")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Class")
    plt.savefig(os.path.join(output_dir, f"{prefix}_true_labels.png"), dpi=150)
    plt.close()
    
    # Plot 2: K-means clustering
    fig_kmeans = plt.figure(figsize=(10, 8))
    plt.scatter(reduced_act[:, 0], reduced_act[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.7, s=50)
    plt.title(f"K-means Clustering (k={len(np.unique(kmeans_labels))}, silhouette={kmeans_silhouette:.3f})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.savefig(os.path.join(output_dir, f"{prefix}_kmeans.png"), dpi=150)
    plt.close()
    
    # Plot 3: ETS clustering
    fig_ets = plt.figure(figsize=(10, 8))
    plt.scatter(reduced_act[:, 0], reduced_act[:, 1], c=ets_labels, cmap='tab10', alpha=0.7, s=50)
    plt.title(f"ETS Clustering (clusters={len(np.unique(ets_labels))}, silhouette={ets_silhouette:.3f})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.savefig(os.path.join(output_dir, f"{prefix}_ets.png"), dpi=150)
    plt.close()
    
    # Plot 4: Comparison of silhouette scores across threshold percentiles
    if prefix.endswith("_ets_comparison"):
        threshold_percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        silhouette_scores = []
        n_clusters = []
        
        for pct in threshold_percentiles:
            labels, _ = compute_ets_clustering(activations, threshold_percentile=pct)
            unique_clusters = len(np.unique(labels))
            
            if unique_clusters > 1:  # Silhouette requires at least 2 clusters
                score = silhouette_score(activations, labels)
            else:
                score = 0.0
                
            silhouette_scores.append(score)
            n_clusters.append(unique_clusters)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot silhouette scores
        ax1.set_xlabel('Threshold Percentile')
        ax1.set_ylabel('Silhouette Score', color='tab:blue')
        ax1.plot(threshold_percentiles, silhouette_scores, 'o-', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create second y-axis for number of clusters
        ax2 = ax1.twinx()
        ax2.set_ylabel('Number of Clusters', color='tab:red')
        ax2.plot(threshold_percentiles, n_clusters, 'o-', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        plt.title('ETS Performance vs Threshold Percentile')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_threshold_comparison.png"), dpi=150)
        plt.close()

def run_ets_comparison(
    dataset_name: str,
    layer_name: str = "layer2",
    seed: int = RANDOM_SEED,
    output_dir: str = "results/ets_example"
):
    """
    Run comparison between k-means and ETS clustering.
    
    Args:
        dataset_name: Name of the dataset to use
        layer_name: Network layer to analyze
        seed: Random seed
        output_dir: Directory to save results
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    print(f"Running ETS comparison on {dataset_name}, layer {layer_name}")
    
    # Load activations
    print("Loading activations...")
    activations, labels = load_activations(dataset_name, layer_name, seed)
    print(f"Loaded {len(activations)} samples with {activations.shape[1]} features")
    
    # Run k-means clustering
    print("Running k-means clustering...")
    k = min(8, max(2, activations.shape[0] // 30))  # Reasonable default k
    kmeans_labels, kmeans = compute_cluster_assignments(
        activations, k, 
        n_init=METRICS["cluster_entropy"]["n_init"],
        max_iter=METRICS["cluster_entropy"]["max_iter"],
        random_state=seed
    )
    
    # Compute silhouette score for k-means
    kmeans_silhouette = silhouette_score(activations, kmeans_labels)
    print(f"K-means: {len(np.unique(kmeans_labels))} clusters, silhouette={kmeans_silhouette:.3f}")
    
    # Run ETS clustering
    print("Running ETS clustering...")
    threshold_pct = METRICS["explainable_threshold_similarity"]["threshold_percentile"]
    ets_labels, thresholds = compute_ets_clustering(
        activations,
        threshold_percentile=threshold_pct,
        verbose=True
    )
    
    # Compute silhouette score for ETS
    if len(np.unique(ets_labels)) > 1:  # Silhouette requires at least 2 clusters
        ets_silhouette = silhouette_score(activations, ets_labels)
    else:
        ets_silhouette = 0.0
    
    print(f"ETS: {len(np.unique(ets_labels))} clusters, silhouette={ets_silhouette:.3f}")
    
    # Generate comparison plots
    print("Generating plots...")
    generate_plots(
        activations, labels, kmeans_labels, ets_labels,
        kmeans_silhouette, ets_silhouette,
        output_dir,
        f"{dataset_name}_{layer_name}_ets_comparison"
    )
    
    # Compute and print ETS statistics
    print("Computing ETS statistics...")
    stats = compute_ets_statistics(activations, ets_labels, thresholds)
    
    print("\nETS Clustering Statistics:")
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"Cluster sizes: min={stats['cluster_sizes']['min']}, "
          f"max={stats['cluster_sizes']['max']}, "
          f"mean={stats['cluster_sizes']['mean']:.1f}")
    print(f"Active dimensions per cluster: min={stats['active_dimensions']['min']}, "
          f"max={stats['active_dimensions']['max']}, "
          f"mean={stats['active_dimensions']['mean']:.1f}")
    
    # Generate example explanation
    if len(activations) >= 2:
        print("\nExample ETS explanation:")
        # Find two points from different clusters
        idx1 = 0
        idx2 = next((i for i in range(len(ets_labels)) if ets_labels[i] != ets_labels[0]), 1)
        
        explanation = explain_ets_similarity(activations[idx1], activations[idx2], thresholds)
        
        print(f"Points {idx1} and {idx2} are {'similar' if explanation['is_similar'] else 'different'}")
        
        if not explanation["is_similar"]:
            print(f"Distinguishing dimensions: {', '.join(explanation['distinguishing_dimensions'])}")
            
        print(f"Dimensions compared: {explanation['num_dimensions_compared']}")
        print(f"Dimensions within threshold: {explanation['num_dimensions_within_threshold']}")
    
    print("\nDone!")
    print(f"Results saved to {output_dir}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETS clustering example")
    parser.add_argument("--dataset", type=str, default="titanic", 
                        help="Dataset name (e.g., titanic, adult, heart)")
    parser.add_argument("--layer", type=str, default="layer2", 
                        help="Layer name (e.g., input, layer1, layer2, layer3, output)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, 
                        help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/ets_example", 
                        help="Output directory")
    
    args = parser.parse_args()
    
    run_ets_comparison(
        dataset_name=args.dataset,
        layer_name=args.layer,
        seed=args.seed,
        output_dir=args.output_dir
    )