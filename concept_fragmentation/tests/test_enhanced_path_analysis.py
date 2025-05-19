"""
Unit tests for the enhanced path analysis functionality in the cluster_paths module.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys
import shutil
import tempfile
from typing import Dict, List, Tuple, Any

# Add the parent directory to the path so we can import our module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.analysis.cluster_paths import (
    assign_unique_cluster_ids,
    compute_cluster_paths,
    compute_fragmentation_score,
    identify_similarity_convergent_paths,
    write_cluster_paths
)
from concept_fragmentation.analysis.similarity_metrics import (
    compute_centroid_similarity,
    normalize_similarity_matrix
)


class TestEnhancedPathAnalysis(unittest.TestCase):
    """Test suite for the enhanced path analysis functionality."""

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
                ]),
                "activations": np.array([
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 1.0]
                ])
            },
            "layer1": {
                "labels": np.array([0, 1, 0, 1, 2]),
                "centers": np.array([
                    [1.0, 0.1],  # Cluster 0: similar to layer0 cluster 0
                    [0.1, 1.0],  # Cluster 1: similar to layer0 cluster 1
                    [2.0, 2.0]   # Cluster 2: different from layer0 clusters
                ]),
                "activations": np.array([
                    [1.0, 0.1],
                    [0.1, 1.0],
                    [1.0, 0.1],
                    [0.1, 1.0],
                    [2.0, 2.0]
                ])
            },
            "layer2": {
                "labels": np.array([0, 0, 1, 1, 2]),
                "centers": np.array([
                    [0.9, 0.0],  # Cluster 0: similar to layer0 cluster 0
                    [0.0, 0.9],  # Cluster 1: similar to layer0 cluster 1
                    [1.0, 0.0]   # Cluster 2: also similar to layer0 cluster 0 (convergence)
                ]),
                "activations": np.array([
                    [0.9, 0.0],
                    [0.9, 0.0],
                    [0.0, 0.9],
                    [0.0, 0.9],
                    [1.0, 0.0]
                ])
            }
        }
        
        # Create a test DataFrame
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "target": [0, 1, 0, 1, 0]
        })
        
        # Create temp directory for outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_assign_unique_cluster_ids(self):
        """Test assigning unique cluster IDs across layers."""
        updated_clusters, id_to_layer_cluster, cluster_to_unique_id = assign_unique_cluster_ids(self.layer_clusters)
        
        # Check that all layers have unique_labels
        for layer_name in self.layer_clusters:
            self.assertIn("unique_labels", updated_clusters[layer_name])
            self.assertIn("id_mapping", updated_clusters[layer_name])
        
        # Check that unique IDs are properly assigned
        # There should be 9 unique clusters (3 in each of the 3 layers)
        self.assertEqual(len(id_to_layer_cluster), 9)
        
        # Check mapping consistency
        for unique_id, (layer_name, original_id, _) in id_to_layer_cluster.items():
            self.assertEqual(cluster_to_unique_id[layer_name][original_id], unique_id)
    
    def test_compute_fragmentation_score(self):
        """Test computing fragmentation score for a path."""
        # Assign unique IDs
        updated_clusters, id_to_layer_cluster, _ = assign_unique_cluster_ids(self.layer_clusters)
        
        # Create a similarity matrix
        similarity_matrix = compute_centroid_similarity(
            updated_clusters, id_to_layer_cluster, metric='cosine', min_similarity=0.0
        )
        
        # Create paths with different fragmentation levels
        # Path 1: Coherent path (low fragmentation)
        # Sample follows clusters that remain similar across layers
        coherent_path = np.array([0, 3, 6])  # Layer0-Cluster0 → Layer1-Cluster0 → Layer2-Cluster0
        
        # Path 2: Incoherent path (high fragmentation)
        # Sample moves between very different clusters
        incoherent_path = np.array([0, 4, 7])  # Layer0-Cluster0 → Layer1-Cluster1 → Layer2-Cluster1
        
        # Compute fragmentation scores
        coherent_score = compute_fragmentation_score(coherent_path, similarity_matrix, id_to_layer_cluster)
        incoherent_score = compute_fragmentation_score(incoherent_path, similarity_matrix, id_to_layer_cluster)
        
        # Check that the coherent path has lower fragmentation
        self.assertLess(coherent_score, incoherent_score)
        self.assertLess(coherent_score, 0.3)  # Coherent path should have low fragmentation
        self.assertGreater(incoherent_score, 0.7)  # Incoherent path should have high fragmentation
    
    def test_identify_similarity_convergent_paths(self):
        """Test identifying similarity-convergent paths."""
        # Assign unique IDs
        updated_clusters, id_to_layer_cluster, _ = assign_unique_cluster_ids(self.layer_clusters)
        
        # Create a similarity matrix
        similarity_matrix = compute_centroid_similarity(
            updated_clusters, id_to_layer_cluster, metric='cosine', min_similarity=0.0
        )
        normalized_matrix = normalize_similarity_matrix(similarity_matrix)
        
        # Create test paths
        # Path 1: Contains a convergence (layer2-cluster2 is similar to layer0-cluster0)
        convergent_path = np.array([0, 4, 8])  # Layer0-Cluster0 → Layer1-Cluster1 → Layer2-Cluster2
        
        # Path 2: No notable convergence
        non_convergent_path = np.array([1, 4, 7])  # Layer0-Cluster1 → Layer1-Cluster1 → Layer2-Cluster1
        
        # Create paths array
        paths = np.vstack([convergent_path, non_convergent_path])
        
        # Set high similarity between Layer0-Cluster0 and Layer2-Cluster2
        # This creates an artificial convergence
        similarity_matrix[(0, 8)] = 0.95
        similarity_matrix[(8, 0)] = 0.95
        normalized_matrix = normalize_similarity_matrix(similarity_matrix)
        
        # Identify convergent paths
        convergent_paths = identify_similarity_convergent_paths(
            paths, normalized_matrix, id_to_layer_cluster, min_similarity=0.8
        )
        
        # Check that we identified the convergent path
        self.assertIn(0, convergent_paths)  # The first path should be identified
        self.assertNotIn(1, convergent_paths)  # The second path should not be identified
        
        # Check convergence details
        convergences = convergent_paths[0]
        self.assertEqual(len(convergences), 1)  # One convergence in this path
        self.assertEqual(convergences[0]["early_cluster"], 0)  # Layer0-Cluster0
        self.assertEqual(convergences[0]["late_cluster"], 8)  # Layer2-Cluster2
    
    def test_write_cluster_paths_with_similarity(self):
        """Test writing cluster paths with similarity metrics."""
        # Assign unique IDs
        updated_clusters, _, _ = assign_unique_cluster_ids(self.layer_clusters)
        
        # Write cluster paths
        output_path = write_cluster_paths(
            dataset_name="test_dataset",
            seed=42,
            layer_clusters=updated_clusters,
            df=self.df,
            target_column="target",
            demographic_columns=["feature1", "feature2"],
            output_dir=self.temp_dir,
            top_k=2,
            max_members=10,
            compute_similarity=True,
            similarity_metric="cosine",
            min_similarity=0.3
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)
        
        # Load the JSON to verify its structure
        import json
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Check the output data structure
        self.assertIn("dataset", data)
        self.assertIn("seed", data)
        self.assertIn("layers", data)
        self.assertIn("unique_paths", data)
        self.assertIn("original_paths", data)
        self.assertIn("human_readable_paths", data)
        self.assertIn("path_archetypes", data)
        self.assertIn("similarity", data)
        
        # Check similarity data
        similarity_data = data["similarity"]
        self.assertIn("raw_similarity", similarity_data)
        self.assertIn("normalized_similarity", similarity_data)
        self.assertIn("layer_similarity", similarity_data)
        self.assertIn("top_similar_clusters", similarity_data)
        self.assertIn("convergent_paths", similarity_data)
        self.assertIn("fragmentation_scores", similarity_data)
        
        # Check fragmentation scores
        frag_scores = similarity_data["fragmentation_scores"]
        self.assertIn("scores", frag_scores)
        self.assertIn("high_fragmentation_paths", frag_scores)
        self.assertIn("low_fragmentation_paths", frag_scores)
        self.assertIn("mean", frag_scores)
        self.assertIn("median", frag_scores)


if __name__ == "__main__":
    unittest.main()