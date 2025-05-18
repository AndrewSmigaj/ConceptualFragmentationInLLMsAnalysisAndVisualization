"""
Test ETS clustering on synthetic data.

This script generates synthetic activation data with known clusters
and runs Explainable Threshold Similarity (ETS) clustering on it.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import sys
import os
# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering, compute_ets_statistics, explain_ets_similarity
)

def generate_synthetic_data(n_samples=300, n_clusters=3, n_features=10, 
                           cluster_std=0.5, random_state=42):
    """
    Generate synthetic data with known clusters.
    
    Args:
        n_samples: Number of samples to generate
        n_clusters: Number of clusters
        n_features: Number of features (dimensions)
        cluster_std: Standard deviation of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (activations, true_labels)
    """
    np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.uniform(-10, 10, (n_clusters, n_features))
    
    # Assign points to clusters
    labels = np.random.randint(0, n_clusters, n_samples)
    
    # Generate points around cluster centers
    activations = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        cluster_idx = labels[i]
        activations[i] = centers[cluster_idx] + np.random.normal(0, cluster_std, n_features)
    
    return activations, labels

def run_comparison(activations, true_labels, threshold_percentile=0.1, output_dir=None):
    """
    Run comparison between ETS clustering and k-means.
    
    Args:
        activations: Activation data matrix
        true_labels: Ground truth cluster labels
        threshold_percentile: Percentile to use for ETS thresholding
        output_dir: Directory to save plots (if None, plots are displayed)
    """
    print("Running ETS clustering...")
    
    # Run ETS clustering
    ets_labels, thresholds = compute_ets_clustering(
        activations, 
        threshold_percentile=threshold_percentile,
        verbose=True
    )
    
    n_clusters_detected = len(np.unique(ets_labels))
    print(f"Detected {n_clusters_detected} clusters with ETS clustering")
    
    # Run k-means with the same number of clusters as in the ground truth
    n_clusters_true = len(np.unique(true_labels))
    print(f"Running K-means with {n_clusters_true} clusters...")
    kmeans = KMeans(n_clusters=n_clusters_true, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(activations)
    
    # Compute evaluation metrics
    if n_clusters_detected > 1 and n_clusters_true > 1:
        ets_silhouette = silhouette_score(activations, ets_labels)
        kmeans_silhouette = silhouette_score(activations, kmeans_labels)
        
        ets_ari = adjusted_rand_score(true_labels, ets_labels)
        kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
        
        print(f"ETS clustering: silhouette={ets_silhouette:.3f}, ARI={ets_ari:.3f}")
        print(f"K-means clustering: silhouette={kmeans_silhouette:.3f}, ARI={kmeans_ari:.3f}")
    else:
        print("Cannot compute silhouette score with only one cluster")
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(activations)
    
    # Plot results
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Ground truth
    plt.subplot(1, 3, 1)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=true_labels, cmap='tab10', alpha=0.7, s=50)
    plt.title("Ground Truth Labels")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Plot 2: K-means clustering
    plt.subplot(1, 3, 2)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.7, s=50)
    plt.title(f"K-means Clustering (k={n_clusters_true})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # Plot 3: ETS clustering
    plt.subplot(1, 3, 3)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=ets_labels, cmap='tab10', alpha=0.7, s=50)
    plt.title(f"ETS Clustering (clusters={n_clusters_detected})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "ets_comparison.png"), dpi=150)
        plt.close()
    else:
        plt.show()
    
    # Print ETS statistics
    print("\nETS Clustering Statistics:")
    stats = compute_ets_statistics(activations, ets_labels, thresholds)
    
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"Cluster sizes: min={stats['cluster_sizes']['min']}, "
          f"max={stats['cluster_sizes']['max']}, "
          f"mean={stats['cluster_sizes']['mean']:.1f}")
    print(f"Active dimensions per cluster: min={stats['active_dimensions']['min']}, "
          f"max={stats['active_dimensions']['max']}, "
          f"mean={stats['active_dimensions']['mean']:.1f}")
    
    # Show top dimensions by importance
    sorted_dims = sorted(stats['dimension_importance'].items(), 
                         key=lambda x: float(x[1]), reverse=True)
    print("\nTop dimensions by importance:")
    for dim, importance in sorted_dims[:5]:  # Show top 5
        print(f"Dimension {dim}: {float(importance):.3f}")
    
    # Generate an example explanation
    print("\nExample ETS explanation:")
    # Find two points from different clusters
    if len(np.unique(ets_labels)) > 1:
        cluster0_idx = np.where(ets_labels == 0)[0][0]
        cluster1_idx = np.where(ets_labels != 0)[0][0]
        
        explanation = explain_ets_similarity(
            activations[cluster0_idx], 
            activations[cluster1_idx], 
            thresholds
        )
        
        print(f"Points from clusters {ets_labels[cluster0_idx]} and {ets_labels[cluster1_idx]}")
        print(f"Are similar: {explanation['is_similar']}")
        
        if not explanation["is_similar"]:
            print(f"Distinguishing dimensions: {', '.join(explanation['distinguishing_dimensions'])}")
        
        print(f"Dimensions compared: {explanation['num_dimensions_compared']}")
        print(f"Dimensions within threshold: {explanation['num_dimensions_within_threshold']}")

if __name__ == "__main__":
    print("Generating synthetic data...")
    activations, true_labels = generate_synthetic_data(
        n_samples=300,
        n_clusters=5,
        n_features=20,
        cluster_std=0.5
    )
    
    print(f"Generated {len(activations)} samples with {activations.shape[1]} features")
    print(f"True number of clusters: {len(np.unique(true_labels))}")
    
    # Run comparison with different threshold percentiles
    for percentile in [0.05, 0.1, 0.2, 0.3]:
        print(f"\nRunning with threshold_percentile={percentile}")
        run_comparison(
            activations,
            true_labels,
            threshold_percentile=percentile,
            output_dir="results/ets_synthetic"
        )