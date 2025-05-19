import unittest
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import our module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.analysis.cluster_paths import (
    assign_unique_cluster_ids,
    get_human_readable_path,
    compute_cluster_paths,
    compute_path_archetypes
)


class TestUniqueClusterIDs(unittest.TestCase):
    """Test suite for the unique cluster ID implementation."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic layer clusters
        self.layer_clusters = {
            "layer0": {
                "labels": np.array([0, 0, 1, 1, 2]),
                "centers": np.array([
                    [1.0, 1.0],  # Center for cluster 0
                    [5.0, 5.0],  # Center for cluster 1
                    [10.0, 10.0]  # Center for cluster 2
                ]),
                "k": 3
            },
            "layer1": {
                "labels": np.array([0, 1, 0, 1, 2]),
                "centers": np.array([
                    [2.0, 2.0],  # Center for cluster 0
                    [6.0, 6.0],  # Center for cluster 1
                    [12.0, 12.0]  # Center for cluster 2
                ]),
                "k": 3
            },
            "layer2": {
                "labels": np.array([0, 0, 1, 1, 2]),
                "centers": np.array([
                    [3.0, 3.0],  # Center for cluster 0
                    [7.0, 7.0],  # Center for cluster 1
                    [14.0, 14.0]  # Center for cluster 2
                ]),
                "k": 3
            }
        }
        
        # Create sample dataframe
        self.sample_df = {
            "id": np.array([0, 1, 2, 3, 4]),
            "feature1": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
            "feature2": np.array(["A", "B", "A", "B", "C"]),
            "target": np.array([0, 1, 0, 1, 1])
        }
        import pandas as pd
        self.df = pd.DataFrame(self.sample_df)

    def test_assign_unique_cluster_ids(self):
        """Test assigning unique IDs to clusters."""
        # Assign unique IDs
        updated_clusters, id_to_layer_cluster, cluster_to_unique_id = assign_unique_cluster_ids(
            self.layer_clusters.copy()
        )
        
        # Check that all layers have unique_labels
        for layer_name, layer_info in updated_clusters.items():
            self.assertIn("unique_labels", layer_info)
            self.assertEqual(len(layer_info["unique_labels"]), len(layer_info["labels"]))
        
        # Check that we have the correct number of unique IDs
        # 3 clusters per layer, 3 layers = 9 unique IDs
        self.assertEqual(len(id_to_layer_cluster), 9)
        
        # Check that cluster_to_unique_id has the right structure
        self.assertEqual(len(cluster_to_unique_id), 3)  # 3 layers
        for layer_name, mapping in cluster_to_unique_id.items():
            self.assertIn(layer_name, self.layer_clusters)
            # 3 clusters per layer
            self.assertEqual(len(mapping), 3)

    def test_human_readable_path(self):
        """Test generating human-readable paths."""
        # Create sample ID mapping
        id_to_layer_cluster = {
            0: ("layer0", 0, 0),
            1: ("layer0", 1, 0),
            2: ("layer0", 2, 0),
            3: ("layer1", 0, 1),
            4: ("layer1", 1, 1),
            5: ("layer1", 2, 1),
            6: ("layer2", 0, 2),
            7: ("layer2", 1, 2),
            8: ("layer2", 2, 2)
        }
        
        # Test single path
        path = np.array([0, 4, 7])
        readable = get_human_readable_path(path, id_to_layer_cluster)
        self.assertEqual(readable, "L0C0→L1C1→L2C1")
        
        # Test another path
        path = np.array([2, 3, 8])
        readable = get_human_readable_path(path, id_to_layer_cluster)
        self.assertEqual(readable, "L0C2→L1C0→L2C2")
        
        # Test with unknown ID
        path = np.array([0, 99, 7])
        readable = get_human_readable_path(path, id_to_layer_cluster)
        self.assertEqual(readable, "L0C0→Unknown(99)→L2C1")

    def test_compute_cluster_paths(self):
        """Test computing paths with unique IDs."""
        # Compute paths
        unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths = compute_cluster_paths(
            self.layer_clusters.copy()
        )
        
        # Check dimensions
        self.assertEqual(unique_paths.shape, (5, 3))  # 5 samples, 3 layers
        self.assertEqual(original_paths.shape, (5, 3))
        self.assertEqual(len(human_readable_paths), 5)
        self.assertEqual(len(layer_names), 3)
        
        # Check that ID mapping makes sense
        for unique_id, (layer_name, original_id, layer_idx) in id_to_layer_cluster.items():
            self.assertIn(layer_name, self.layer_clusters)
            self.assertIn(original_id, np.unique(self.layer_clusters[layer_name]["labels"]))
            self.assertEqual(layer_idx, layer_names.index(layer_name))
        
        # Verify paths are consistent
        for i in range(len(unique_paths)):
            path = unique_paths[i]
            original = original_paths[i]
            human_readable = human_readable_paths[i]
            
            # Check original path matches original cluster IDs
            for j, layer_name in enumerate(layer_names):
                self.assertEqual(original[j], self.layer_clusters[layer_name]["labels"][i])
            
            # Check human readable path
            expected_readable = "→".join([f"L{j}C{original[j]}" for j in range(len(original))])
            self.assertEqual(human_readable, expected_readable)

    def test_compute_path_archetypes(self):
        """Test computing archetypes with unique IDs."""
        # First compute paths
        unique_paths, layer_names, id_to_layer_cluster, _, human_readable_paths = compute_cluster_paths(
            self.layer_clusters.copy()
        )
        
        # Compute archetypes
        archetypes = compute_path_archetypes(
            unique_paths,
            layer_names,
            self.df,
            id_to_layer_cluster=id_to_layer_cluster,
            human_readable_paths=human_readable_paths,
            target_column="target"
        )
        
        # Check we got archetypes
        self.assertGreater(len(archetypes), 0)
        
        # Check archetype structure
        for archetype in archetypes:
            self.assertIn("path", archetype)
            self.assertIn("numeric_path", archetype)
            self.assertIn("count", archetype)
            self.assertIn("percentage", archetype)
            self.assertIn("target_rate", archetype)
            self.assertIn("demo_stats", archetype)
            
            # Check path format has layer-specific labels (L0C0→L1C1→...)
            path_str = archetype["path"]
            self.assertTrue("L0C" in path_str)
            self.assertTrue("→" in path_str)
            
            # Check numeric path is a list of IDs
            self.assertTrue(isinstance(archetype["numeric_path"], list))


if __name__ == "__main__":
    unittest.main()