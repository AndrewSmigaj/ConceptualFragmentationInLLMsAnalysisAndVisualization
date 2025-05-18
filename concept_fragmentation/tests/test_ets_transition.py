"""
Testing ETS integration with transition matrix analysis for archetypal paths.

This script tests how ETS clustering works when used to create transition 
matrices and archetypal paths across multiple network layers.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import sys
import pandas as pd

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.metrics.explainable_threshold_similarity import compute_ets_clustering
from concept_fragmentation.analysis.transition_matrix import (
    compute_transition_matrix, 
    compute_transition_entropy
)

def generate_multi_layer_data(
    n_samples: int = 200,
    n_features_per_layer: list = [5, 8, 10, 6],
    n_clusters_per_layer: list = [3, 5, 4, 3],
    random_state: int = 42
) -> dict:
    """
    Generate synthetic activation data for multiple network layers with
    controlled transition patterns between layers.
    
    Args:
        n_samples: Number of samples
        n_features_per_layer: List of feature dimensions for each layer
        n_clusters_per_layer: List of clusters in each layer
        random_state: Random seed
        
    Returns:
        Dictionary mapping layer names to activation matrices
    """
    np.random.seed(random_state)
    
    n_layers = len(n_features_per_layer)
    layers = {f"layer{i+1}": None for i in range(n_layers)}
    ground_truth = {f"layer{i+1}": None for i in range(n_layers)}
    
    # Generate first layer with clean clusters
    layer1_clusters = np.repeat(np.arange(n_clusters_per_layer[0]), 
                               n_samples // n_clusters_per_layer[0] + 1)[:n_samples]
    np.random.shuffle(layer1_clusters)
    
    for i, layer_name in enumerate(layers.keys()):
        n_features = n_features_per_layer[i]
        n_clusters = n_clusters_per_layer[i]
        
        if i == 0:
            # First layer - generate clean clusters
            activations = np.zeros((n_samples, n_features))
            cluster_centers = np.random.uniform(-10, 10, (n_clusters, n_features))
            
            # Ensure centers are well-separated
            for j in range(n_clusters):
                for k in range(j+1, n_clusters):
                    while np.linalg.norm(cluster_centers[j] - cluster_centers[k]) < 7.0:
                        cluster_centers[k] = np.random.uniform(-10, 10, n_features)
            
            # Generate data for each cluster
            for j in range(n_samples):
                cluster_idx = layer1_clusters[j]
                activations[j] = cluster_centers[cluster_idx] + np.random.normal(0, 0.5, n_features)
            
            # Store data and ground truth
            layers[layer_name] = activations
            ground_truth[layer_name] = layer1_clusters
            
            # Previous layer clusters for transition pattern
            prev_clusters = layer1_clusters
        else:
            # Subsequent layers - create controlled transitions
            activations = np.zeros((n_samples, n_features))
            cluster_centers = np.random.uniform(-10, 10, (n_clusters, n_features))
            
            # Ensure centers are well-separated
            for j in range(n_clusters):
                for k in range(j+1, n_clusters):
                    while np.linalg.norm(cluster_centers[j] - cluster_centers[k]) < 7.0:
                        cluster_centers[k] = np.random.uniform(-10, 10, n_features)
            
            # Create patterns based on previous layer
            # Each cluster in this layer can get input from multiple previous clusters
            # Transition matrix will be designed to have specific patterns
            
            # Create fixed transition patterns (e.g., split, merge, preserve)
            prev_n_clusters = n_clusters_per_layer[i-1]
            transition_pattern = np.zeros((prev_n_clusters, n_clusters))
            
            if prev_n_clusters < n_clusters:
                # Splitting pattern (1-to-many)
                splits_per_cluster = n_clusters // prev_n_clusters
                remainder = n_clusters % prev_n_clusters
                
                for j in range(prev_n_clusters):
                    num_splits = splits_per_cluster + (1 if j < remainder else 0)
                    start_idx = j * splits_per_cluster + min(j, remainder)
                    for k in range(num_splits):
                        transition_pattern[j, start_idx + k] = 1.0
                        
            elif prev_n_clusters > n_clusters:
                # Merging pattern (many-to-1)
                clusters_per_merge = prev_n_clusters // n_clusters
                remainder = prev_n_clusters % n_clusters
                
                for j in range(n_clusters):
                    num_merged = clusters_per_merge + (1 if j < remainder else 0)
                    start_idx = j * clusters_per_merge + min(j, remainder)
                    for k in range(num_merged):
                        transition_pattern[start_idx + k, j] = 1.0
            else:
                # One-to-one mapping with some noise
                for j in range(n_clusters):
                    transition_pattern[j, j] = 0.8
                    other_targets = [(j+1) % n_clusters, (j-1) % n_clusters]
                    for other in other_targets:
                        transition_pattern[j, other] = 0.1
            
            # Normalize transition pattern into probabilities
            transition_pattern = transition_pattern / np.sum(transition_pattern, axis=1, keepdims=True)
            
            # Apply the transition pattern to assign clusters
            current_layer_clusters = np.zeros(n_samples, dtype=int)
            for j in range(n_samples):
                prev_cluster = prev_clusters[j]
                probs = transition_pattern[prev_cluster]
                current_layer_clusters[j] = np.random.choice(n_clusters, p=probs)
            
            # Generate data for each assigned cluster
            for j in range(n_samples):
                cluster_idx = current_layer_clusters[j]
                activations[j] = cluster_centers[cluster_idx] + np.random.normal(0, 0.5, n_features)
            
            # Store data and ground truth
            layers[layer_name] = activations
            ground_truth[layer_name] = current_layer_clusters
            
            # Update previous clusters for next layer
            prev_clusters = current_layer_clusters
    
    return layers, ground_truth

def extract_paths(layer_clusters: dict) -> np.ndarray:
    """
    Extract paths from layer clusters.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster labels
        
    Returns:
        Array of paths (sequences of cluster assignments)
    """
    # Get ordered layer names
    layer_names = sorted(layer_clusters.keys())
    
    # Get number of samples
    n_samples = len(layer_clusters[layer_names[0]])
    
    # Create paths array
    n_layers = len(layer_names)
    paths = np.zeros((n_samples, n_layers), dtype=int)
    
    # Fill in paths
    for i, layer in enumerate(layer_names):
        paths[:, i] = layer_clusters[layer]
    
    return paths

def plot_transition_matrices(transition_matrices: dict, output_dir: str):
    """
    Plot transition matrices as heatmaps.
    
    Args:
        transition_matrices: Dictionary mapping transitions to matrices
        output_dir: Directory to save plots
    """
    for transition_name, matrix in transition_matrices.items():
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Transition Probability')
        plt.title(f"Transition Matrix: {transition_name}")
        plt.xlabel("Target Cluster")
        plt.ylabel("Source Cluster")
        
        # Add text annotations
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] > 0.1:  # Only show significant transitions
                    plt.text(j, i, f"{matrix[i, j]:.2f}", 
                            ha="center", va="center", 
                            color="white" if matrix[i, j] > 0.5 else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"transition_{transition_name}.png"))
        plt.close()

def visualize_layer_clusters(activations: dict, clusters: dict, output_dir: str, method_name: str):
    """
    Visualize clusters in each layer using PCA.
    
    Args:
        activations: Dictionary mapping layer names to activation matrices
        clusters: Dictionary mapping layer names to cluster labels
        output_dir: Directory to save plots
        method_name: Name of clustering method for plot titles
    """
    for layer_name in activations.keys():
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(activations[layer_name])
        
        # Plot clusters
        plt.figure(figsize=(10, 8))
        unique_clusters = np.unique(clusters[layer_name])
        
        for cluster_id in unique_clusters:
            mask = (clusters[layer_name] == cluster_id)
            plt.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.7
            )
        
        plt.title(f"{method_name} Clustering - {layer_name}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{method_name.lower()}_{layer_name}.png"))
        plt.close()

class TestETSTransition(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create output directory for plots
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_ets_transition_matrices(self):
        """Test ETS clustering with transition matrix analysis."""
        print("\n=== ETS Transition Analysis Test ===")
        
        # Generate multi-layer activation data
        layers, ground_truth = generate_multi_layer_data(
            n_samples=200,
            n_features_per_layer=[5, 8, 10, 6],
            n_clusters_per_layer=[3, 5, 4, 3],
            random_state=42
        )
        
        print(f"Generated data for {len(layers)} layers")
        
        # Apply ETS clustering to each layer
        ets_layer_clusters = {}
        threshold_percentile = 0.1
        
        for layer_name, activations in layers.items():
            print(f"Running ETS clustering for {layer_name}...")
            ets_labels, thresholds = compute_ets_clustering(
                activations, 
                threshold_percentile=threshold_percentile
            )
            ets_layer_clusters[layer_name] = ets_labels
            
            n_clusters = len(np.unique(ets_labels))
            expected_clusters = len(np.unique(ground_truth[layer_name]))
            print(f"  ETS found {n_clusters} clusters, expected {expected_clusters}")
        
        # For comparison, apply k-means clustering
        kmeans_layer_clusters = {}
        
        for layer_name, activations in layers.items():
            print(f"Running k-means clustering for {layer_name}...")
            expected_clusters = len(np.unique(ground_truth[layer_name]))
            kmeans = KMeans(n_clusters=expected_clusters, random_state=42)
            kmeans_labels = kmeans.fit_predict(activations)
            kmeans_layer_clusters[layer_name] = kmeans_labels
        
        # Compute transition matrices for ETS clusters
        ets_transitions = {}
        layer_names = sorted(layers.keys())
        
        for i in range(len(layer_names) - 1):
            source_layer = layer_names[i]
            target_layer = layer_names[i + 1]
            transition_name = f"{source_layer}_to_{target_layer}"
            
            print(f"Computing ETS transition matrix: {transition_name}")
            transition_matrix = compute_transition_matrix(
                ets_layer_clusters[source_layer],
                ets_layer_clusters[target_layer]
            )
            
            ets_transitions[transition_name] = transition_matrix
        
        # Compute transition matrices for k-means clusters
        kmeans_transitions = {}
        
        for i in range(len(layer_names) - 1):
            source_layer = layer_names[i]
            target_layer = layer_names[i + 1]
            transition_name = f"{source_layer}_to_{target_layer}"
            
            transition_matrix = compute_transition_matrix(
                kmeans_layer_clusters[source_layer],
                kmeans_layer_clusters[target_layer]
            )
            
            kmeans_transitions[transition_name] = transition_matrix
        
        # Compute transition entropy for both methods
        ets_entropy = {}
        kmeans_entropy = {}
        
        for transition_name, matrix in ets_transitions.items():
            ets_entropy[transition_name] = compute_transition_entropy(matrix)
        
        for transition_name, matrix in kmeans_transitions.items():
            kmeans_entropy[transition_name] = compute_transition_entropy(matrix)
        
        # Extract paths for both methods
        ets_paths = extract_paths(ets_layer_clusters)
        kmeans_paths = extract_paths(kmeans_layer_clusters)
        
        # Count unique paths
        unique_ets_paths, ets_path_counts = np.unique(ets_paths, axis=0, return_counts=True)
        unique_kmeans_paths, kmeans_path_counts = np.unique(kmeans_paths, axis=0, return_counts=True)
        
        print(f"ETS found {len(unique_ets_paths)} unique paths")
        print(f"K-means found {len(unique_kmeans_paths)} unique paths")
        
        # Print top paths for both methods
        n_top_paths = min(5, len(unique_ets_paths), len(unique_kmeans_paths))
        
        print("\nTop ETS paths:")
        ets_top_indices = np.argsort(-ets_path_counts)[:n_top_paths]
        for i in ets_top_indices:
            path = unique_ets_paths[i]
            count = ets_path_counts[i]
            print(f"  Path {path} - {count} samples ({count/len(ets_paths)*100:.1f}%)")
        
        print("\nTop K-means paths:")
        kmeans_top_indices = np.argsort(-kmeans_path_counts)[:n_top_paths]
        for i in kmeans_top_indices:
            path = unique_kmeans_paths[i]
            count = kmeans_path_counts[i]
            print(f"  Path {path} - {count} samples ({count/len(kmeans_paths)*100:.1f}%)")
        
        # Plot transition matrices
        print("\nPlotting ETS transition matrices...")
        plot_transition_matrices(ets_transitions, self.output_dir)
        
        print("Plotting K-means transition matrices...")
        plot_transition_matrices(kmeans_transitions, self.output_dir)
        
        # Visualize clusters in each layer
        print("\nVisualizing ETS clusters...")
        visualize_layer_clusters(layers, ets_layer_clusters, self.output_dir, "ETS")
        
        print("Visualizing K-means clusters...")
        visualize_layer_clusters(layers, kmeans_layer_clusters, self.output_dir, "KMeans")
        
        # Compare entropy metrics
        print("\nTransition Entropy Comparison:")
        all_entropy_data = []
        
        for transition_name in ets_transitions.keys():
            ets_values = ets_entropy[transition_name]
            kmeans_values = kmeans_entropy[transition_name]
            
            print(f"\n{transition_name}:")
            print(f"  ETS: mean_entropy={ets_values['mean_entropy']:.3f}, "
                 f"normalized_entropy={ets_values['normalized_entropy']:.3f}, "
                 f"sparsity={ets_values['sparsity']:.3f}")
            
            print(f"  K-means: mean_entropy={kmeans_values['mean_entropy']:.3f}, "
                 f"normalized_entropy={kmeans_values['normalized_entropy']:.3f}, "
                 f"sparsity={kmeans_values['sparsity']:.3f}")
            
            # Store for plotting
            all_entropy_data.append({
                'transition': transition_name,
                'ets_mean_entropy': ets_values['mean_entropy'],
                'kmeans_mean_entropy': kmeans_values['mean_entropy'],
                'ets_normalized_entropy': ets_values['normalized_entropy'],
                'kmeans_normalized_entropy': kmeans_values['normalized_entropy'],
                'ets_sparsity': ets_values['sparsity'],
                'kmeans_sparsity': kmeans_values['sparsity']
            })
        
        # Plot comparison of entropy metrics
        df = pd.DataFrame(all_entropy_data)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        x = np.arange(len(df['transition']))
        width = 0.35
        plt.bar(x - width/2, df['ets_mean_entropy'], width, label='ETS')
        plt.bar(x + width/2, df['kmeans_mean_entropy'], width, label='K-means')
        plt.xlabel('Transition')
        plt.ylabel('Mean Entropy')
        plt.title('Mean Entropy Comparison')
        plt.xticks(x, df['transition'], rotation=45, ha='right')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.bar(x - width/2, df['ets_normalized_entropy'], width, label='ETS')
        plt.bar(x + width/2, df['kmeans_normalized_entropy'], width, label='K-means')
        plt.xlabel('Transition')
        plt.ylabel('Normalized Entropy')
        plt.title('Normalized Entropy Comparison')
        plt.xticks(x, df['transition'], rotation=45, ha='right')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.bar(x - width/2, df['ets_sparsity'], width, label='ETS')
        plt.bar(x + width/2, df['kmeans_sparsity'], width, label='K-means')
        plt.xlabel('Transition')
        plt.ylabel('Sparsity')
        plt.title('Sparsity Comparison')
        plt.xticks(x, df['transition'], rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "entropy_comparison.png"))
        plt.close()
        
        # Plot path distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(n_top_paths), ets_path_counts[ets_top_indices])
        plt.title("ETS Top Path Distribution")
        plt.xlabel("Path Rank")
        plt.ylabel("Count")
        
        plt.subplot(1, 2, 2)
        plt.bar(range(n_top_paths), kmeans_path_counts[kmeans_top_indices])
        plt.title("K-means Top Path Distribution")
        plt.xlabel("Path Rank")
        plt.ylabel("Count")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "path_distribution.png"))
        plt.close()
        
        # Assertions for validation
        
        # ETS should find appropriate number of clusters
        for layer_name in layers.keys():
            ets_n_clusters = len(np.unique(ets_layer_clusters[layer_name]))
            expected_n_clusters = len(np.unique(ground_truth[layer_name]))
            
            # We expect ETS to be within a reasonable range of the expected clusters
            # This may differ slightly from k-means which is explicitly told the number
            self.assertLessEqual(abs(ets_n_clusters - expected_n_clusters), 2,
                              f"ETS clusters for {layer_name} significantly different than expected")
        
        # Sparsity should be reasonable (not too dense, not too sparse)
        for transition_name, entropy in ets_entropy.items():
            self.assertGreater(entropy['sparsity'], 0.3,
                               f"ETS transition matrix for {transition_name} is too dense")
            self.assertLess(entropy['sparsity'], 0.9,
                            f"ETS transition matrix for {transition_name} is too sparse")
        
        # Paths should be reasonably distributed (not all in one path)
        ets_path_entropy = -np.sum((ets_path_counts / len(ets_paths)) * 
                                 np.log2(ets_path_counts / len(ets_paths)))
        self.assertGreater(ets_path_entropy, 1.0,
                          "ETS paths are too concentrated in a few paths")
        
        print("\nCompleted ETS transition integration test")

if __name__ == '__main__':
    unittest.main()