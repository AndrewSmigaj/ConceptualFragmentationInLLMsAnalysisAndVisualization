"""
Performance testing for Explainable Threshold Similarity (ETS) clustering.

This script tests the ETS clustering algorithm's performance on datasets
of varying sizes to evaluate scaling behavior, efficiency of batched processing,
and consistency of results.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score
from memory_profiler import memory_usage
import unittest

# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_dimension_thresholds,
    compute_similarity_matrix
)

def generate_scalable_data(
    n_samples: int, 
    n_features: int, 
    n_clusters: int = 5,
    cluster_std: float = 0.5,
    random_state: int = 42
) -> tuple:
    """
    Generate synthetic dataset that scales to large sizes.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        n_clusters: Number of clusters
        cluster_std: Standard deviation within clusters
        random_state: Random seed
        
    Returns:
        Tuple of (data, labels)
    """
    np.random.seed(random_state)
    
    # Ensure clusters are well separated to allow for consistent comparisons
    means = np.random.uniform(-10, 10, size=(n_clusters, n_features))
    
    # Ensure separation between cluster centers
    min_distance = 5.0
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            while np.linalg.norm(means[i] - means[j]) < min_distance:
                means[j] = np.random.uniform(-10, 10, size=n_features)
    
    # Calculate samples per cluster
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    # Generate data
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    idx = 0
    for i in range(n_clusters):
        # Add one extra sample to early clusters if there's a remainder
        cluster_size = samples_per_cluster + (1 if i < remainder else 0)
        end_idx = idx + cluster_size
        
        # Generate samples for this cluster
        X[idx:end_idx] = means[i] + np.random.normal(0, cluster_std, size=(cluster_size, n_features))
        y[idx:end_idx] = i
        
        idx = end_idx
    
    return X, y

class TestETSPerformance(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create output directory for plots
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def test_sample_size_scaling(self):
        """Test ETS performance scaling with increasing sample size."""
        print("\n=== Sample Size Scaling Test ===")
        
        # Fixed parameters
        n_features = 10
        n_clusters = 5
        threshold_percentile = 0.1
        
        # Range of sample sizes to test
        sample_sizes = [100, 500, 1000, 2000, 5000]
        
        # Initialize results storage
        results = []
        
        for n_samples in sample_sizes:
            print(f"\nTesting with {n_samples} samples...")
            
            # Generate data
            X, y_true = generate_scalable_data(
                n_samples=n_samples,
                n_features=n_features,
                n_clusters=n_clusters
            )
            
            # Measure time for threshold computation
            start_time = time.time()
            thresholds = compute_dimension_thresholds(X, threshold_percentile)
            threshold_time = time.time() - start_time
            
            # Measure memory and time for default batch size
            default_batch_size = 1000
            
            start_time = time.time()
            mem_usage = max(memory_usage((compute_ets_clustering, (X, None, threshold_percentile, default_batch_size))))
            ets_labels, _ = compute_ets_clustering(X, None, threshold_percentile, default_batch_size)
            total_time = time.time() - start_time
            
            # Calculate silhouette if enough clusters are found
            n_clusters_found = len(np.unique(ets_labels))
            if n_clusters_found > 1:
                silhouette = silhouette_score(X, ets_labels)
            else:
                silhouette = 0
            
            # Record results
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'batch_size': default_batch_size,
                'threshold_time': threshold_time,
                'total_time': total_time,
                'memory_mb': mem_usage,
                'n_clusters_found': n_clusters_found,
                'silhouette': silhouette
            })
            
            print(f"  Time: {total_time:.2f}s, Memory: {mem_usage:.2f}MB")
            print(f"  Clusters found: {n_clusters_found}, Expected: {n_clusters}")
            print(f"  Silhouette score: {silhouette:.4f}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Plot scaling results
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Time vs sample size
        axs[0, 0].plot(df['n_samples'], df['total_time'], 'o-')
        axs[0, 0].set_title('Execution Time vs Sample Size')
        axs[0, 0].set_xlabel('Number of Samples')
        axs[0, 0].set_ylabel('Time (seconds)')
        
        # Memory vs sample size
        axs[0, 1].plot(df['n_samples'], df['memory_mb'], 'o-')
        axs[0, 1].set_title('Memory Usage vs Sample Size')
        axs[0, 1].set_xlabel('Number of Samples')
        axs[0, 1].set_ylabel('Memory (MB)')
        
        # Clusters found vs sample size
        axs[1, 0].plot(df['n_samples'], df['n_clusters_found'], 'o-')
        axs[1, 0].set_title('Clusters Found vs Sample Size')
        axs[1, 0].set_xlabel('Number of Samples')
        axs[1, 0].set_ylabel('Number of Clusters')
        axs[1, 0].axhline(y=n_clusters, color='r', linestyle='--', label='Expected')
        axs[1, 0].legend()
        
        # Silhouette vs sample size
        axs[1, 1].plot(df['n_samples'], df['silhouette'], 'o-')
        axs[1, 1].set_title('Silhouette Score vs Sample Size')
        axs[1, 1].set_xlabel('Number of Samples')
        axs[1, 1].set_ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ets_sample_scaling.png'))
        
        # Print summary
        print("\nSample Size Scaling Summary:")
        print(df.to_string(index=False))
        
        # Assertions
        self.assertTrue(all(df['n_clusters_found'] == n_clusters), 
                        "ETS should find the same number of clusters regardless of sample size")
        
        # Check that execution time scales reasonably (not exponentially)
        # We expect roughly quadratic scaling due to pairwise comparisons
        if len(sample_sizes) >= 3:
            ratio1 = df.iloc[-1]['total_time'] / df.iloc[-2]['total_time']
            ratio2 = df.iloc[-2]['total_time'] / df.iloc[-3]['total_time']
            sample_ratio1 = df.iloc[-1]['n_samples'] / df.iloc[-2]['n_samples']
            sample_ratio2 = df.iloc[-2]['n_samples'] / df.iloc[-3]['n_samples']
            
            print(f"Time scaling ratio (last step): {ratio1:.2f}")
            print(f"Sample size ratio (last step): {sample_ratio1:.2f}")
            print(f"Expected quadratic scaling: {sample_ratio1**2:.2f}")
            
            self.assertLess(ratio1, sample_ratio1**3, 
                           "Execution time should scale better than cubic with sample size")
    
    def test_dimension_scaling(self):
        """Test ETS performance scaling with increasing dimensions."""
        print("\n=== Dimension Scaling Test ===")
        
        # Fixed parameters
        n_samples = 500
        n_clusters = 5
        threshold_percentile = 0.1
        batch_size = 1000
        
        # Range of feature dimensions to test
        feature_dims = [5, 10, 50, 100, 200]
        
        # Initialize results storage
        results = []
        
        for n_features in feature_dims:
            print(f"\nTesting with {n_features} dimensions...")
            
            # Generate data
            X, y_true = generate_scalable_data(
                n_samples=n_samples,
                n_features=n_features,
                n_clusters=n_clusters
            )
            
            # Measure time for threshold computation
            start_time = time.time()
            thresholds = compute_dimension_thresholds(X, threshold_percentile)
            threshold_time = time.time() - start_time
            
            # Measure memory and time
            start_time = time.time()
            mem_usage = max(memory_usage((compute_ets_clustering, (X, None, threshold_percentile, batch_size))))
            ets_labels, _ = compute_ets_clustering(X, None, threshold_percentile, batch_size)
            total_time = time.time() - start_time
            
            # Calculate silhouette if enough clusters are found
            n_clusters_found = len(np.unique(ets_labels))
            if n_clusters_found > 1:
                silhouette = silhouette_score(X, ets_labels)
            else:
                silhouette = 0
            
            # Record results
            results.append({
                'n_samples': n_samples,
                'n_features': n_features,
                'threshold_time': threshold_time,
                'total_time': total_time,
                'memory_mb': mem_usage,
                'n_clusters_found': n_clusters_found,
                'silhouette': silhouette
            })
            
            print(f"  Time: {total_time:.2f}s, Memory: {mem_usage:.2f}MB")
            print(f"  Clusters found: {n_clusters_found}, Expected: {n_clusters}")
            print(f"  Silhouette score: {silhouette:.4f}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Plot scaling results
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Time vs dimensions
        axs[0, 0].plot(df['n_features'], df['total_time'], 'o-')
        axs[0, 0].set_title('Execution Time vs Dimensions')
        axs[0, 0].set_xlabel('Number of Dimensions')
        axs[0, 0].set_ylabel('Time (seconds)')
        
        # Memory vs dimensions
        axs[0, 1].plot(df['n_features'], df['memory_mb'], 'o-')
        axs[0, 1].set_title('Memory Usage vs Dimensions')
        axs[0, 1].set_xlabel('Number of Dimensions')
        axs[0, 1].set_ylabel('Memory (MB)')
        
        # Clusters found vs dimensions
        axs[1, 0].plot(df['n_features'], df['n_clusters_found'], 'o-')
        axs[1, 0].set_title('Clusters Found vs Dimensions')
        axs[1, 0].set_xlabel('Number of Dimensions')
        axs[1, 0].set_ylabel('Number of Clusters')
        axs[1, 0].axhline(y=n_clusters, color='r', linestyle='--', label='Expected')
        axs[1, 0].legend()
        
        # Silhouette vs dimensions
        axs[1, 1].plot(df['n_features'], df['silhouette'], 'o-')
        axs[1, 1].set_title('Silhouette Score vs Dimensions')
        axs[1, 1].set_xlabel('Number of Dimensions')
        axs[1, 1].set_ylabel('Silhouette Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ets_dimension_scaling.png'))
        
        # Print summary
        print("\nDimension Scaling Summary:")
        print(df.to_string(index=False))
        
        # Assertions
        self.assertTrue(all(df['n_clusters_found'] == n_clusters), 
                       "ETS should find the same number of clusters regardless of dimensions")
        
        # Check that execution time scales roughly linearly with dimensions
        # (the main bottleneck should be the number of samples, not dimensions)
        if len(feature_dims) >= 3:
            ratio1 = df.iloc[-1]['total_time'] / df.iloc[-2]['total_time']
            dims_ratio1 = df.iloc[-1]['n_features'] / df.iloc[-2]['n_features']
            
            print(f"Time scaling ratio (last step): {ratio1:.2f}")
            print(f"Dimension ratio (last step): {dims_ratio1:.2f}")
            
            self.assertLess(ratio1, dims_ratio1**2, 
                           "Execution time should scale better than quadratic with dimensions")
    
    def test_batch_size_impact(self):
        """Test the impact of different batch sizes on performance."""
        print("\n=== Batch Size Impact Test ===")
        
        # Fixed parameters
        n_samples = 2000
        n_features = 20
        n_clusters = 5
        threshold_percentile = 0.1
        
        # Generate data
        X, y_true = generate_scalable_data(
            n_samples=n_samples,
            n_features=n_features,
            n_clusters=n_clusters
        )
        
        # Compute thresholds once for all tests
        thresholds = compute_dimension_thresholds(X, threshold_percentile)
        
        # Range of batch sizes to test
        batch_sizes = [100, 250, 500, 1000, 2000, None]
        
        # Initialize results storage
        results = []
        
        for batch_size in batch_sizes:
            batch_name = str(batch_size) if batch_size is not None else "None"
            print(f"\nTesting with batch size {batch_name}...")
            
            # Measure similarity matrix computation time
            start_time = time.time()
            similarity_matrix = compute_similarity_matrix(X, thresholds, batch_size if batch_size is not None else n_samples)
            similarity_time = time.time() - start_time
            
            # Measure full clustering time
            start_time = time.time()
            mem_usage = max(memory_usage((compute_ets_clustering, (X, thresholds, None, batch_size))))
            ets_labels, _ = compute_ets_clustering(X, thresholds, None, batch_size)
            total_time = time.time() - start_time
            
            # Check cluster result consistency
            n_clusters_found = len(np.unique(ets_labels))
            
            # Record results
            results.append({
                'batch_size': batch_name,
                'similarity_time': similarity_time,
                'total_time': total_time,
                'memory_mb': mem_usage,
                'n_clusters_found': n_clusters_found
            })
            
            print(f"  Similarity matrix time: {similarity_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s, Memory: {mem_usage:.2f}MB")
            print(f"  Clusters found: {n_clusters_found}, Expected: {n_clusters}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Make batch size numeric for plotting (except None)
        numeric_batch = []
        for bs in df['batch_size']:
            try:
                numeric_batch.append(int(bs))
            except ValueError:
                numeric_batch.append(n_samples)  # Use n_samples for "None"
                
        df['numeric_batch'] = numeric_batch
        
        # Plot batch size results
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # Time vs batch size
        axs[0].plot(df['numeric_batch'], df['similarity_time'], 'o-', label='Similarity Matrix')
        axs[0].plot(df['numeric_batch'], df['total_time'], 'o-', label='Total')
        axs[0].set_title('Execution Time vs Batch Size')
        axs[0].set_xlabel('Batch Size')
        axs[0].set_ylabel('Time (seconds)')
        axs[0].set_xscale('log')
        axs[0].legend()
        
        # Memory vs batch size
        axs[1].plot(df['numeric_batch'], df['memory_mb'], 'o-')
        axs[1].set_title('Memory Usage vs Batch Size')
        axs[1].set_xlabel('Batch Size')
        axs[1].set_ylabel('Memory (MB)')
        axs[1].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ets_batch_size_impact.png'))
        
        # Print summary
        print("\nBatch Size Impact Summary:")
        print(df.to_string(index=False))
        
        # Assertions
        self.assertTrue(df['n_clusters_found'].nunique() == 1, 
                       "ETS should find the same number of clusters regardless of batch size")
        
        # Check if batch processing is more memory efficient than no batching
        no_batch_memory = df[df['batch_size'] == 'None']['memory_mb'].values[0]
        batch_memory = df[df['batch_size'].astype(str) == '1000']['memory_mb'].values[0]
        
        print(f"Memory with no batching: {no_batch_memory:.2f}MB")
        print(f"Memory with 1000-size batching: {batch_memory:.2f}MB")
        
        self.assertLessEqual(batch_memory, no_batch_memory * 1.2, 
                           "Batched processing should not use significantly more memory")
        
    def test_result_consistency(self):
        """Test that ETS produces consistent results across different runs."""
        print("\n=== Result Consistency Test ===")
        
        # Fixed parameters
        n_samples = 1000
        n_features = 20
        n_clusters = 5
        threshold_percentile = 0.1
        batch_size = 1000
        
        # Generate data
        X, y_true = generate_scalable_data(
            n_samples=n_samples,
            n_features=n_features,
            n_clusters=n_clusters
        )
        
        # Run ETS multiple times and compare results
        runs = 3
        all_labels = []
        all_thresholds = []
        
        for i in range(runs):
            print(f"\nRun {i+1}...")
            
            # Run ETS clustering
            ets_labels, thresholds = compute_ets_clustering(
                X, None, threshold_percentile, batch_size
            )
            
            all_labels.append(ets_labels)
            all_thresholds.append(thresholds)
            
            print(f"  Clusters found: {len(np.unique(ets_labels))}")
        
        # Check threshold consistency
        threshold_diffs = []
        for i in range(runs-1):
            for j in range(i+1, runs):
                diff = np.abs(all_thresholds[i] - all_thresholds[j])
                rel_diff = diff / (all_thresholds[i] + 1e-10)  # Avoid division by zero
                threshold_diffs.append(np.mean(rel_diff))
        
        avg_threshold_diff = np.mean(threshold_diffs)
        print(f"Average relative threshold difference: {avg_threshold_diff:.6f}")
        
        # Check label consistency
        from sklearn.metrics import adjusted_rand_score
        label_aris = []
        for i in range(runs-1):
            for j in range(i+1, runs):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                label_aris.append(ari)
        
        avg_label_ari = np.mean(label_aris)
        print(f"Average ARI between runs: {avg_label_ari:.4f}")
        
        # Assertions
        self.assertLess(avg_threshold_diff, 1e-10, 
                       "Thresholds should be identical across runs")
        
        self.assertGreaterEqual(avg_label_ari, 0.99, 
                              "Cluster assignments should be consistent across runs")

if __name__ == '__main__':
    unittest.main()