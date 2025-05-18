"""
Test script for validating Explainable Threshold Similarity (ETS) clustering.

This script uses synthetic datasets to test various aspects of the ETS
clustering algorithm, including:
- Basic clustering functionality
- Handling of mixed-scale features 
- Dimension-wise thresholding
- Explainability and transparency
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

import sys
import os
# Add parent directory to path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_dimension_thresholds,
    explain_ets_similarity,
    compute_ets_statistics
)

# Import synthetic data generators
from test_data.ets_synthetic import (
    generate_synthetic_ets_data,
    generate_mixed_scale_dataset, 
    generate_dimension_threshold_dataset,
    generate_semantic_dimension_dataset,
    generate_explanation_test_cases,
    verify_explanation_correctness,
    evaluate_clustering_metrics,
    visualize_clusters,
    visualize_thresholds
)

class TestETSValidation(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        # Create output directory for plots
        self.output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_basic_clustering(self):
        """Test basic clustering functionality with well-separated clusters."""
        # Generate synthetic data with clear clusters
        data, true_labels = generate_synthetic_ets_data(
            n_clusters=3,
            n_features=2,
            n_samples_per_cluster=50,
            cluster_std=0.3,
            random_state=42
        )
        
        # Run ETS clustering
        ets_labels, thresholds = compute_ets_clustering(
            data,
            threshold_percentile=0.1
        )
        
        # Compare with true labels
        metrics = evaluate_clustering_metrics(data, true_labels, ets_labels)
        
        # Get number of clusters
        n_clusters_ets = len(np.unique(ets_labels))
        n_clusters_true = len(np.unique(true_labels))
        
        # Assertions
        self.assertEqual(n_clusters_ets, n_clusters_true, 
                         f"ETS detected {n_clusters_ets} clusters but expected {n_clusters_true}")
        self.assertGreater(metrics["adjusted_rand_index"], 0.9,
                          f"ARI score should be high for well-separated clusters, got {metrics['adjusted_rand_index']}")
        
        # Compare with k-means for reference
        kmeans = KMeans(n_clusters=n_clusters_true, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        
        kmeans_metrics = evaluate_clustering_metrics(data, true_labels, kmeans_labels)
        
        # Print comparison
        print("\n=== Basic Clustering Test ===")
        print(f"ETS detected {n_clusters_ets} clusters, expected {n_clusters_true}")
        print(f"ETS ARI: {metrics['adjusted_rand_index']:.4f}")
        print(f"K-means ARI: {kmeans_metrics['adjusted_rand_index']:.4f}")
        
        # Visualize clusters (ETS vs true)
        fig = visualize_clusters(data, true_labels, "True Clusters")
        plt.savefig(os.path.join(self.output_dir, "basic_true_clusters.png"))
        
        fig = visualize_clusters(data, ets_labels, "ETS Clusters")
        plt.savefig(os.path.join(self.output_dir, "basic_ets_clusters.png"))
        
        # Visualize thresholds
        fig = visualize_thresholds(data, thresholds)
        plt.savefig(os.path.join(self.output_dir, "basic_thresholds.png"))
    
    def test_mixed_scale_features(self):
        """Test ETS with features having vastly different scales."""
        # Generate dataset with mixed feature scales
        data, true_labels, feature_scales = generate_mixed_scale_dataset(
            n_clusters=4,
            n_samples_per_cluster=50,
            random_state=42
        )
        
        # Run ETS clustering
        ets_labels, thresholds = compute_ets_clustering(
            data,
            threshold_percentile=0.1
        )
        
        # Compute metrics
        metrics = evaluate_clustering_metrics(data, true_labels, ets_labels)
        
        # Run k-means for comparison
        n_clusters_true = len(np.unique(true_labels))
        kmeans = KMeans(n_clusters=n_clusters_true, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)
        
        kmeans_metrics = evaluate_clustering_metrics(data, true_labels, kmeans_labels)
        
        # Print results
        print("\n=== Mixed Scale Features Test ===")
        print(f"Feature scales: {feature_scales}")
        print(f"ETS thresholds: {thresholds}")
        print(f"ETS detected {len(np.unique(ets_labels))} clusters, expected {n_clusters_true}")
        print(f"ETS ARI: {metrics['adjusted_rand_index']:.4f}")
        print(f"K-means ARI: {kmeans_metrics['adjusted_rand_index']:.4f}")
        
        # Verify thresholds scale appropriately with feature scales
        threshold_scale_ratios = []
        for i in range(len(feature_scales)-1):
            if feature_scales[i] > 0 and feature_scales[i+1] > 0:
                scale_ratio = feature_scales[i] / feature_scales[i+1]
                threshold_ratio = thresholds[i] / thresholds[i+1]
                threshold_scale_ratios.append((scale_ratio, threshold_ratio))
        
        # Check correlation between feature scale ratios and threshold ratios
        for scale_ratio, threshold_ratio in threshold_scale_ratios:
            print(f"Scale ratio: {scale_ratio:.2f}, Threshold ratio: {threshold_ratio:.2f}")
        
        # Assertions - ETS should adapt to feature scales
        self.assertGreater(metrics["adjusted_rand_index"], 0.8,
                          "ETS should handle mixed scales well with high ARI")
        
        # Since ETS adapts thresholds per dimension, it should handle mixed scales better than k-means
        self.assertGreaterEqual(metrics["adjusted_rand_index"], kmeans_metrics["adjusted_rand_index"] * 0.9,
                              "ETS should perform at least as well as k-means on mixed scale data")
        
        # Visualize clusters using first two dimensions
        fig = visualize_clusters(data, true_labels, "True Clusters (Mixed Scales)", dimensions=(0, 1))
        plt.savefig(os.path.join(self.output_dir, "mixed_scale_true_clusters.png"))
        
        fig = visualize_clusters(data, ets_labels, "ETS Clusters (Mixed Scales)", dimensions=(0, 1))
        plt.savefig(os.path.join(self.output_dir, "mixed_scale_ets_clusters.png"))
        
        # Visualize thresholds
        fig = visualize_thresholds(data, thresholds)
        plt.savefig(os.path.join(self.output_dir, "mixed_scale_thresholds.png"))
    
    def test_dimension_thresholds(self):
        """Test that ETS correctly identifies clusters based on specific dimension thresholds."""
        # Generate dataset where clusters are defined by specific dimension thresholds
        data, true_labels, distinguishing_dimensions = generate_dimension_threshold_dataset(
            n_samples=300,
            random_state=42
        )
        
        # Run ETS clustering
        ets_labels, thresholds = compute_ets_clustering(
            data,
            threshold_percentile=0.1
        )
        
        # Compute metrics
        metrics = evaluate_clustering_metrics(data, true_labels, ets_labels)
        
        # Print results
        print("\n=== Dimension Threshold Test ===")
        print(f"Distinguishing dimensions: {distinguishing_dimensions}")
        print(f"ETS thresholds: {thresholds}")
        print(f"ETS detected {len(np.unique(ets_labels))} clusters, expected {len(np.unique(true_labels))}")
        print(f"ETS ARI: {metrics['adjusted_rand_index']:.4f}")
        
        # Test points from different clusters to verify they differ in expected dimensions
        for cluster_pair, dims in distinguishing_dimensions.items():
            c1, c2 = map(int, cluster_pair.split('_vs_'))
            
            # Get a sample point from each cluster
            idx1 = np.where(true_labels == c1)[0][0]
            idx2 = np.where(true_labels == c2)[0][0]
            
            point1 = data[idx1]
            point2 = data[idx2]
            
            # Generate explanation
            explanation = explain_ets_similarity(point1, point2, thresholds)
            
            # Extract distinguishing dimensions from explanation
            explanation_dims = []
            for dim_name in explanation["distinguishing_dimensions"]:
                if "Dimension" in dim_name:
                    try:
                        dim_idx = int(dim_name.split()[-1])
                        explanation_dims.append(dim_idx)
                    except ValueError:
                        pass
            
            # Validate explanation
            is_correct, error_msg = verify_explanation_correctness(point1, point2, thresholds, explanation)
            
            print(f"Cluster pair {cluster_pair}:")
            print(f"  Expected distinguishing dimensions: {dims}")
            print(f"  Explanation distinguishing dimensions: {explanation_dims}")
            print(f"  Explanation correct: {is_correct}")
            if not is_correct:
                print(f"  Error: {error_msg}")
            
            # The distinguishing dimensions in the explanation should be a subset of the expected dimensions
            # Note: The thresholds might be different from those used to generate the data
            self.assertTrue(set(explanation_dims).issubset(set(dims)) or set(dims).issubset(set(explanation_dims)),
                           f"Explanation dimensions should match or be subset of expected dimensions")
        
        # Visualize clusters using dimensions that should distinguish clusters
        key_dimensions = [0, 2]  # Dimensions that should be most discriminative
        fig = visualize_clusters(data, true_labels, "True Clusters (Dimension Thresholds)", dimensions=key_dimensions)
        plt.savefig(os.path.join(self.output_dir, "dimension_threshold_true_clusters.png"))
        
        fig = visualize_clusters(data, ets_labels, "ETS Clusters (Dimension Thresholds)", dimensions=key_dimensions)
        plt.savefig(os.path.join(self.output_dir, "dimension_threshold_ets_clusters.png"))
    
    def test_explanation_functionality(self):
        """Test the explanation functionality of ETS."""
        # Use the explanation test cases
        test_cases = generate_explanation_test_cases()
        
        for i, case in enumerate(test_cases):
            # Extract case details
            point1 = case["point1"]
            point2 = case["point2"]
            thresholds = case["thresholds"]
            expected_similar = case["expected_similar"]
            expected_distinguishing_dims = case["expected_distinguishing_dims"]
            
            # Get feature names if available
            feature_names = case.get("feature_names")
            
            # Generate explanation
            explanation = explain_ets_similarity(
                point1, point2, thresholds, 
                feature_names=feature_names
            )
            
            # Verify explanation correctness
            is_correct, error_msg = verify_explanation_correctness(point1, point2, thresholds, explanation)
            
            print(f"\nTest case {i+1}:")
            print(f"  Points similar: {explanation['is_similar']}, expected: {expected_similar}")
            print(f"  Distinguishing dimensions: {explanation['distinguishing_dimensions']}")
            print(f"  Expected distinguishing dimensions: {expected_distinguishing_dims}")
            print(f"  Explanation correct: {is_correct}")
            if not is_correct:
                print(f"  Error: {error_msg}")
            
            # Assertions
            self.assertEqual(explanation["is_similar"], expected_similar,
                           f"is_similar={explanation['is_similar']}, expected {expected_similar}")
            
            # Verify that the explanation identifies the correct distinguishing dimensions
            # Note: Need to handle the way dimensions are named in explanations
            if not expected_similar:
                # Extract dimension indices from names
                explanation_dims = []
                for dim_name in explanation["distinguishing_dimensions"]:
                    if "Dimension" in dim_name:
                        try:
                            dim_idx = int(dim_name.split()[-1])
                            explanation_dims.append(dim_idx)
                        except ValueError:
                            pass
                    elif feature_names is not None:
                        # Handle named features
                        try:
                            dim_idx = feature_names.index(dim_name)
                            explanation_dims.append(dim_idx)
                        except ValueError:
                            pass
                
                # Sort both lists for comparison
                expected_distinguishing_dims.sort()
                explanation_dims.sort()
                
                self.assertEqual(explanation_dims, expected_distinguishing_dims,
                               f"distinguishing_dimensions={explanation_dims}, expected {expected_distinguishing_dims}")
    
    def test_semantic_dimensions(self):
        """Test ETS with semantically meaningful dimensions."""
        # Generate dataset with semantic dimensions
        data, true_labels, feature_names, explanations = generate_semantic_dimension_dataset(
            n_samples=200,
            random_state=42
        )
        
        # Run ETS clustering
        ets_labels, thresholds = compute_ets_clustering(
            data,
            threshold_percentile=0.1
        )
        
        # Compute metrics
        metrics = evaluate_clustering_metrics(data, true_labels, ets_labels)
        
        # Print results
        print("\n=== Semantic Dimensions Test ===")
        print(f"Feature names: {feature_names}")
        print(f"ETS detected {len(np.unique(ets_labels))} clusters, expected {len(np.unique(true_labels))}")
        print(f"ETS ARI: {metrics['adjusted_rand_index']:.4f}")
        
        # Test explanation with semantic dimensions
        # Get sample points from different clusters
        clusters = np.unique(true_labels)
        if len(clusters) >= 2:
            idx1 = np.where(true_labels == clusters[0])[0][0]
            idx2 = np.where(true_labels == clusters[1])[0][0]
            
            point1 = data[idx1]
            point2 = data[idx2]
            
            # Generate explanation
            explanation = explain_ets_similarity(point1, point2, thresholds, feature_names=feature_names)
            
            print("\nSemantic dimension explanation:")
            print(f"  Clusters compared: {true_labels[idx1]} vs {true_labels[idx2]}")
            print(f"  Similar: {explanation['is_similar']}")
            print(f"  Distinguishing dimensions: {explanation['distinguishing_dimensions']}")
            
            # Visualize the explanation
            from test_data.ets_synthetic import visualize_explanation
            fig = visualize_explanation(point1, point2, thresholds, feature_names=feature_names)
            plt.savefig(os.path.join(self.output_dir, "semantic_explanation.png"))
            
            # Assertions
            self.assertIsNotNone(explanation["distinguishing_dimensions"],
                               "Explanation should include distinguishing dimensions")
        
        # Compute ETS statistics
        stats = compute_ets_statistics(data, ets_labels, thresholds)
        
        print("\nETS Statistics:")
        print(f"  Number of clusters: {stats['n_clusters']}")
        print(f"  Cluster sizes (min/max/mean): {stats['cluster_sizes']['min']}/{stats['cluster_sizes']['max']}/{stats['cluster_sizes']['mean']:.1f}")
        print(f"  Active dimensions per cluster (min/max/mean): {stats['active_dimensions']['min']}/{stats['active_dimensions']['max']}/{stats['active_dimensions']['mean']:.1f}")
        
        # Print top dimensions by importance
        importance = sorted([(int(k), v) for k, v in stats['dimension_importance'].items()], key=lambda x: x[1], reverse=True)
        print("\nTop dimensions by importance:")
        for dim, score in importance[:3]:
            dim_name = feature_names[dim] if dim < len(feature_names) else f"Dimension {dim}"
            print(f"  {dim_name}: {score:.4f}")
        
        # Assertions on statistics
        self.assertGreaterEqual(stats['n_clusters'], 1, "Should detect at least one cluster")
        self.assertGreater(stats['active_dimensions']['mean'], 0, "Should have active dimensions")
    
    def test_threshold_sensitivity(self):
        """Test ETS sensitivity to different threshold percentiles."""
        # Generate synthetic data
        data, true_labels = generate_synthetic_ets_data(
            n_clusters=4,
            n_features=5,
            n_samples_per_cluster=30,
            cluster_std=0.3,
            random_state=42
        )
        
        # Test different threshold percentiles
        threshold_percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
        results = []
        
        for percentile in threshold_percentiles:
            # Run ETS clustering
            ets_labels, thresholds = compute_ets_clustering(
                data,
                threshold_percentile=percentile
            )
            
            # Compute metrics
            metrics = evaluate_clustering_metrics(data, true_labels, ets_labels)
            
            # Add to results
            results.append({
                "percentile": percentile,
                "n_clusters": len(np.unique(ets_labels)),
                "ari": metrics["adjusted_rand_index"],
                "silhouette": metrics["silhouette_score"],
                "mean_threshold": np.mean(thresholds)
            })
        
        # Print results
        print("\n=== Threshold Sensitivity Test ===")
        print("Percentile | Clusters | ARI | Silhouette | Mean Threshold")
        print("-" * 60)
        for result in results:
            print(f"{result['percentile']:.2f} | {result['n_clusters']} | {result['ari']:.4f} | {result['silhouette']:.4f} | {result['mean_threshold']:.4f}")
        
        # Plot metrics vs threshold percentile
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot([r['percentile'] for r in results], [r['n_clusters'] for r in results], 'o-')
        plt.title('Number of Clusters vs Threshold Percentile')
        plt.xlabel('Threshold Percentile')
        plt.ylabel('Number of Clusters')
        
        plt.subplot(2, 2, 2)
        plt.plot([r['percentile'] for r in results], [r['ari'] for r in results], 'o-')
        plt.title('ARI vs Threshold Percentile')
        plt.xlabel('Threshold Percentile')
        plt.ylabel('Adjusted Rand Index')
        
        plt.subplot(2, 2, 3)
        plt.plot([r['percentile'] for r in results], [r['silhouette'] for r in results], 'o-')
        plt.title('Silhouette Score vs Threshold Percentile')
        plt.xlabel('Threshold Percentile')
        plt.ylabel('Silhouette Score')
        
        plt.subplot(2, 2, 4)
        plt.plot([r['percentile'] for r in results], [r['mean_threshold'] for r in results], 'o-')
        plt.title('Mean Threshold vs Threshold Percentile')
        plt.xlabel('Threshold Percentile')
        plt.ylabel('Mean Threshold Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "threshold_sensitivity.png"))
        
        # Find optimal threshold percentile based on ARI
        best_percentile = max(results, key=lambda x: x['ari'])['percentile']
        print(f"\nBest threshold percentile based on ARI: {best_percentile}")
        
        # Assertions - lower percentiles should generally give more clusters
        self.assertGreaterEqual(
            results[0]['n_clusters'], 
            results[-1]['n_clusters'],
            "Lower percentiles should generally give more clusters"
        )

if __name__ == '__main__':
    unittest.main()