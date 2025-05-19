"""
Unit tests for the similarity metrics module.
"""

import unittest
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import our module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.analysis.similarity_metrics import (
    compute_centroid_similarity,
    normalize_similarity_matrix,
    compute_layer_similarity_matrix,
    get_top_similar_clusters,
    serialize_similarity_matrix,
    deserialize_similarity_matrix
)


class TestSimilarityMetrics(unittest.TestCase):
    """Test suite for the similarity metrics implementation."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic layer clusters with known centroids
        self.layer_clusters = {
            "layer0": {
                "labels": np.array([0, 0, 1, 1, 2]),
                "centers": np.array([
                    [1.0, 0.0],  # Cluster 0: aligned with x-axis
                    [0.0, 1.0],  # Cluster 1: aligned with y-axis
                    [1.0, 1.0]   # Cluster 2: diagonal
                ])
            },
            "layer1": {
                "labels": np.array([0, 1, 0, 1, 2]),
                "centers": np.array([
                    [1.0, 0.1],  # Cluster 0: similar to layer0 cluster 0
                    [0.1, 1.0],  # Cluster 1: similar to layer0 cluster 1
                    [2.0, 2.0]   # Cluster 2: different from layer0 clusters
                ])
            }
        }
        
        # Create id_to_layer_cluster mapping
        self.id_to_layer_cluster = {
            0: ("layer0", 0, 0),  # Layer 0, Cluster 0
            1: ("layer0", 1, 0),  # Layer 0, Cluster 1
            2: ("layer0", 2, 0),  # Layer 0, Cluster 2
            3: ("layer1", 0, 1),  # Layer 1, Cluster 0
            4: ("layer1", 1, 1),  # Layer 1, Cluster 1
            5: ("layer1", 2, 1)   # Layer 1, Cluster 2
        }

    def test_compute_centroid_similarity_cosine(self):
        """Test computing centroid similarity matrix with cosine similarity."""
        sim_matrix = compute_centroid_similarity(
            self.layer_clusters, self.id_to_layer_cluster, metric='cosine'
        )
        
        # Verify that we have the expected pairs
        # Layer0-Cluster0 vs Layer1-Cluster0
        self.assertIn((0, 3), sim_matrix)
        self.assertIn((3, 0), sim_matrix)
        
        # Layer0-Cluster1 vs Layer1-Cluster1
        self.assertIn((1, 4), sim_matrix)
        self.assertIn((4, 1), sim_matrix)
        
        # Check that similar clusters have high similarity scores
        self.assertGreater(sim_matrix[(0, 3)], 0.9)  # Almost parallel vectors
        self.assertGreater(sim_matrix[(1, 4)], 0.9)  # Almost parallel vectors
        
        # Check that orthogonal vectors have low similarity
        # Layer0-Cluster0 (x-axis) vs Layer1-Cluster1 (y-axis)
        self.assertIn((0, 4), sim_matrix)
        self.assertLess(sim_matrix[(0, 4)], 0.2)  # Nearly orthogonal
        
        # Verify symmetry
        for (id1, id2) in list(sim_matrix.keys()):
            self.assertAlmostEqual(sim_matrix[(id1, id2)], sim_matrix[(id2, id1)])

    def test_compute_centroid_similarity_euclidean(self):
        """Test computing centroid similarity matrix with euclidean similarity."""
        sim_matrix = compute_centroid_similarity(
            self.layer_clusters, self.id_to_layer_cluster, metric='euclidean'
        )
        
        # Verify that we have the expected pairs
        self.assertIn((0, 3), sim_matrix)
        self.assertIn((1, 4), sim_matrix)
        
        # Similar clusters should have high similarity (low distance)
        self.assertGreater(sim_matrix[(0, 3)], 0.8)
        self.assertGreater(sim_matrix[(1, 4)], 0.8)
        
        # Different clusters should have lower similarity (higher distance)
        # Layer0-Cluster2 vs Layer1-Cluster2
        self.assertIn((2, 5), sim_matrix)
        self.assertLess(sim_matrix[(2, 5)], 0.8)

    def test_normalize_similarity_matrix(self):
        """Test normalizing similarity matrix."""
        # Create a sample similarity matrix
        sim_matrix = {
            (0, 3): 0.95,
            (3, 0): 0.95,
            (1, 4): 0.90,
            (4, 1): 0.90,
            (2, 5): 0.50,
            (5, 2): 0.50
        }
        
        # Normalize
        norm_matrix = normalize_similarity_matrix(sim_matrix)
        
        # Check normalization
        self.assertAlmostEqual(norm_matrix[(0, 3)], 1.0)  # Max similarity
        self.assertAlmostEqual(norm_matrix[(2, 5)], 0.0)  # Min similarity
        
        # Check intermediate values
        self.assertAlmostEqual(norm_matrix[(1, 4)], 0.8889, places=4)  # (0.90 - 0.50) / (0.95 - 0.50)
        
        # Check symmetry
        for (id1, id2) in list(norm_matrix.keys()):
            self.assertAlmostEqual(norm_matrix[(id1, id2)], norm_matrix[(id2, id1)])

    def test_compute_layer_similarity_matrix(self):
        """Test computing layer similarity matrix."""
        # Create a sample similarity matrix
        sim_matrix = {
            (0, 3): 0.95, (3, 0): 0.95,
            (0, 4): 0.10, (4, 0): 0.10,
            (1, 3): 0.20, (3, 1): 0.20,
            (1, 4): 0.90, (4, 1): 0.90,
            (2, 5): 0.50, (5, 2): 0.50
        }
        
        # Compute layer similarity
        layer_sim = compute_layer_similarity_matrix(sim_matrix, self.id_to_layer_cluster)
        
        # There's only one layer pair (0, 1)
        self.assertIn((0, 1), layer_sim)
        
        # Check statistics
        stats = layer_sim[(0, 1)]
        self.assertAlmostEqual(stats["mean"], 0.53, places=2)
        self.assertAlmostEqual(stats["median"], 0.50, places=2)
        self.assertAlmostEqual(stats["max"], 0.95, places=2)
        self.assertAlmostEqual(stats["min"], 0.10, places=2)
        self.assertEqual(stats["count"], 5)  # 5 unique pairs between layer 0 and 1

    def test_get_top_similar_clusters(self):
        """Test getting top similar clusters."""
        # Create a sample similarity matrix
        sim_matrix = {
            (0, 3): 0.95, (3, 0): 0.95,
            (0, 4): 0.20, (4, 0): 0.20,
            (0, 5): 0.10, (5, 0): 0.10,
            (1, 3): 0.30, (3, 1): 0.30,
            (1, 4): 0.90, (4, 1): 0.90,
            (1, 5): 0.20, (5, 1): 0.20,
            (2, 3): 0.40, (3, 2): 0.40,
            (2, 4): 0.30, (4, 2): 0.30,
            (2, 5): 0.60, (5, 2): 0.60
        }
        
        # Get top similar clusters for layer 0
        top_similar = get_top_similar_clusters(
            sim_matrix, self.id_to_layer_cluster, layer_idx=0, top_k=2, min_similarity=0.1
        )
        
        # Check results
        self.assertEqual(len(top_similar), 3)  # 3 clusters in layer 0
        
        # Cluster 0 should have 3 and 4 as most similar
        top_0 = top_similar[0]
        self.assertEqual(len(top_0), 2)  # Asked for top 2
        self.assertEqual(top_0[0][0], 3)  # Most similar is cluster 3
        self.assertAlmostEqual(top_0[0][1], 0.95)  # With similarity 0.95
        
        # Cluster 1 should have 4 and 3 as most similar
        top_1 = top_similar[1]
        self.assertEqual(len(top_1), 2)  # Asked for top 2
        self.assertEqual(top_1[0][0], 4)  # Most similar is cluster 4
        self.assertAlmostEqual(top_1[0][1], 0.90)  # With similarity 0.90
        
        # With high min_similarity threshold, we should get fewer results
        top_similar_high = get_top_similar_clusters(
            sim_matrix, self.id_to_layer_cluster, layer_idx=0, top_k=2, min_similarity=0.5
        )
        
        # Only clusters 0 and 1 should have results above 0.5
        self.assertEqual(len(top_similar_high[0]), 1)  # Only cluster 3 above 0.5
        self.assertEqual(len(top_similar_high[1]), 1)  # Only cluster 4 above 0.5
        self.assertEqual(len(top_similar_high[2]), 1)  # Only cluster 5 above 0.5

    def test_serialization(self):
        """Test serializing and deserializing similarity matrix."""
        # Create a sample similarity matrix
        sim_matrix = {
            (0, 3): 0.95,
            (3, 0): 0.95,
            (1, 4): 0.93,
            (4, 1): 0.93,
            (2, 5): 0.42,
            (5, 2): 0.42
        }
        
        # Serialize
        serialized = serialize_similarity_matrix(sim_matrix)
        
        # Check serialized format
        self.assertEqual(serialized["0,3"], 0.95)
        self.assertEqual(serialized["1,4"], 0.93)
        self.assertEqual(serialized["2,5"], 0.42)
        
        # Deserialize
        deserialized = deserialize_similarity_matrix(serialized)
        
        # Check deserialized format
        self.assertEqual(deserialized[(0, 3)], 0.95)
        self.assertEqual(deserialized[(1, 4)], 0.93)
        self.assertEqual(deserialized[(2, 5)], 0.42)
        
        # Test round-trip preservation
        for pair, value in sim_matrix.items():
            self.assertEqual(deserialized[pair], value)


if __name__ == "__main__":
    unittest.main()