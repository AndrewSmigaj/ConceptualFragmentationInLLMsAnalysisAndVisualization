"""
Tests for Explainable Threshold Similarity (ETS) clustering.
"""

import unittest
import numpy as np
from ..metrics.explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_dimension_thresholds,
    explain_ets_similarity,
    compute_ets_statistics
)

class TestExplainableThresholdSimilarity(unittest.TestCase):
    
    def test_basic_clustering(self):
        # Create synthetic data with clear clusters
        X = np.array([
            [1.0, 1.0],  # Cluster 1
            [1.1, 0.9],  # Cluster 1
            [5.0, 5.0],  # Cluster 2
            [5.1, 4.9],  # Cluster 2
            [10.0, 0.0], # Cluster 3
            [10.1, 0.1]  # Cluster 3
        ])
        
        # Set explicit thresholds
        thresholds = np.array([0.5, 0.5])
        
        # Compute ETS clustering
        labels, returned_thresholds = compute_ets_clustering(X, thresholds)
        
        # Check that we get 3 clusters
        self.assertEqual(len(np.unique(labels)), 3)
        
        # Check that points in the same cluster have the same label
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[4], labels[5])
        
        # Check that points in different clusters have different labels
        self.assertNotEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[4])
        self.assertNotEqual(labels[2], labels[4])
        
        # Check that returned thresholds match input
        np.testing.assert_array_equal(thresholds, returned_thresholds)
    
    def test_automatic_threshold(self):
        # Create synthetic data with clear clusters but more dispersed
        X = np.array([
            [1.0, 1.0], [1.1, 0.9], [1.2, 1.1],  # Cluster 1
            [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],  # Cluster 2
            [10.0, 0.0], [10.1, 0.1], [9.9, -0.1]  # Cluster 3
        ])
        
        # Compute ETS clustering with automatic threshold with a very small percentile
        # to ensure stricter thresholds that should separate the groups
        labels, thresholds = compute_ets_clustering(X, threshold_percentile=0.01)
        
        # Check we get reasonable number of clusters (exact number may vary based on auto-threshold)
        self.assertGreaterEqual(len(np.unique(labels)), 3)
        
        # Thresholds should be positive
        self.assertTrue(np.all(thresholds > 0))
    
    def test_dimension_thresholds(self):
        # Create data with different scales in each dimension
        X = np.array([
            [1.0, 100.0],
            [1.1, 110.0],
            [2.0, 200.0],
            [2.1, 210.0]
        ])
        
        # Compute thresholds
        thresholds = compute_dimension_thresholds(X, threshold_percentile=0.1)
        
        # Check that thresholds adapt to dimension scale
        self.assertLess(thresholds[0], thresholds[1])
        self.assertGreaterEqual(thresholds[1] / thresholds[0], 50)  # Roughly 100x scale difference
    
    def test_empty_and_small_datasets(self):
        # Empty dataset
        X_empty = np.array([]).reshape(0, 5)
        labels_empty, thresholds_empty = compute_ets_clustering(X_empty)
        self.assertEqual(len(labels_empty), 0)
        self.assertEqual(len(thresholds_empty), 5)
        
        # Single datapoint (should form its own cluster)
        X_single = np.array([[1.0, 2.0, 3.0]])
        labels_single, thresholds_single = compute_ets_clustering(X_single)
        self.assertEqual(len(labels_single), 1)
        self.assertEqual(labels_single[0], 0)
        self.assertEqual(len(thresholds_single), 3)
    
    def test_identical_points(self):
        # Create dataset with identical points
        X = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        
        # Compute ETS clustering
        labels, _ = compute_ets_clustering(X)
        
        # All points should be in the same cluster
        self.assertEqual(len(np.unique(labels)), 1)
    
    def test_explanation(self):
        # Create two sample points
        point1 = np.array([1.0, 5.0, 10.0])
        point2 = np.array([1.2, 5.5, 9.0])
        thresholds = np.array([0.5, 1.0, 2.0])
        
        # These points should be similar based on the thresholds
        explanation = explain_ets_similarity(point1, point2, thresholds)
        
        # Check explanation structure
        self.assertTrue(explanation["is_similar"])
        self.assertEqual(len(explanation["dimensions"]), 3)
        self.assertEqual(explanation["num_dimensions_compared"], 3)
        self.assertEqual(explanation["num_dimensions_within_threshold"], 3)
        
        # Test with points that are not similar
        point3 = np.array([2.0, 5.0, 10.0])  # First dimension exceeds threshold
        explanation2 = explain_ets_similarity(point1, point3, thresholds)
        
        self.assertFalse(explanation2["is_similar"])
        self.assertEqual(explanation2["num_dimensions_within_threshold"], 2)
        self.assertEqual(len(explanation2["distinguishing_dimensions"]), 1)
    
    def test_statistics(self):
        # Create synthetic data with clear clusters
        X = np.array([
            [1.0, 1.0], [1.1, 0.9], [1.2, 0.8],  # Cluster 1
            [5.0, 5.0], [5.1, 4.9], [4.9, 5.1],  # Cluster 2
            [10.0, 0.0], [10.1, 0.1], [9.9, -0.1]  # Cluster 3
        ])
        
        # Set thresholds
        thresholds = np.array([0.5, 0.5])
        
        # Compute ETS clustering
        labels, _ = compute_ets_clustering(X, thresholds)
        
        # Compute statistics
        stats = compute_ets_statistics(X, labels, thresholds)
        
        # Basic checks
        self.assertEqual(stats["n_clusters"], 3)
        self.assertEqual(stats["cluster_sizes"]["min"], 3)
        self.assertEqual(stats["cluster_sizes"]["max"], 3)
        
        # Check threshold stats
        self.assertEqual(stats["threshold_stats"]["min"], 0.5)
        self.assertEqual(stats["threshold_stats"]["max"], 0.5)

if __name__ == '__main__':
    unittest.main()