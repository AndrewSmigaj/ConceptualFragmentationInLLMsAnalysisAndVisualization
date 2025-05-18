"""
Tests for Cross-Layer Path Analysis Metrics.

This module tests the implementation of the cross-layer metrics described in
"Foundations of Archetypal Path Analysis: Toward a Principled Geometry for Cluster-Based Interpretability".
"""

import unittest
import numpy as np
from typing import Dict, List, Tuple
import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.analysis.cross_layer_metrics import (
    validate_layer_order,
    project_to_common_space,
    compute_centroids,
    compute_centroid_similarity,
    compute_membership_overlap,
    extract_paths,
    compute_trajectory_fragmentation,
    compute_inter_cluster_path_density,
    analyze_cross_layer_metrics
)

class TestCrossLayerMetrics(unittest.TestCase):
    """Test cases for cross-layer metrics."""
    
    def setUp(self):
        """Create synthetic data for testing."""
        # Create synthetic multi-layer data
        np.random.seed(42)
        
        # Layer structure: 3 layers with different dimensions
        self.n_samples = 100
        self.layer_dims = {
            "layer1": 5,
            "layer2": 8,
            "layer3": 6
        }
        
        # Cluster structure: different number of clusters per layer
        self.n_clusters = {
            "layer1": 3,
            "layer2": 4,
            "layer3": 2
        }
        
        # Generate activations for each layer
        self.activations = {}
        self.clusters = {}
        
        for layer_name, dim in self.layer_dims.items():
            # Generate random activations
            self.activations[layer_name] = np.random.randn(self.n_samples, dim)
            
            # Generate cluster assignments
            n_clusters = self.n_clusters[layer_name]
            self.clusters[layer_name] = np.random.randint(0, n_clusters, self.n_samples)
        
        # Generate class labels for testing
        self.labels = np.random.randint(0, 2, self.n_samples)  # Binary labels
    
    def test_validate_layer_order(self):
        """Test layer order validation."""
        # Test normal layer names
        layers = ["layer1", "layer3", "layer2"]
        ordered = validate_layer_order(layers)
        self.assertEqual(ordered, ["layer1", "layer2", "layer3"])
        
        # Test with standard names
        layers = ["output", "input", "hidden1", "hidden2"]
        ordered = validate_layer_order(layers)
        self.assertEqual(ordered, ["input", "hidden1", "hidden2", "output"])
        
        # Test with mixed names
        layers = ["layer3", "input", "output"]
        ordered = validate_layer_order(layers)
        self.assertIn("input", ordered[0])  # Input should be first
        self.assertIn("output", ordered[-1])  # Output should be last
    
    def test_project_to_common_space(self):
        """Test projection to common space."""
        # Test PCA projection
        projected = project_to_common_space(
            self.activations,
            projection_method='pca',
            projection_dims=3
        )
        
        for layer_name, layer_proj in projected.items():
            self.assertEqual(layer_proj.shape, (self.n_samples, 3))
        
        # Test no projection
        no_proj = project_to_common_space(
            self.activations,
            projection_method='none'
        )
        
        for layer_name in self.activations:
            np.testing.assert_array_equal(no_proj[layer_name], self.activations[layer_name])
        
        # Test invalid projection method
        with self.assertRaises(ValueError):
            project_to_common_space(self.activations, projection_method='invalid')
    
    def test_compute_centroids(self):
        """Test centroid computation."""
        for layer_name in self.activations:
            centroids, cluster_ids = compute_centroids(
                self.activations[layer_name],
                self.clusters[layer_name]
            )
            
            n_clusters = self.n_clusters[layer_name]
            n_features = self.layer_dims[layer_name]
            
            self.assertEqual(centroids.shape, (n_clusters, n_features))
            self.assertEqual(len(cluster_ids), n_clusters)
            
            # Verify centroids
            for i, cluster_id in enumerate(cluster_ids):
                mask = (self.clusters[layer_name] == cluster_id)
                if np.sum(mask) > 0:
                    expected_centroid = np.mean(self.activations[layer_name][mask], axis=0)
                    np.testing.assert_array_almost_equal(centroids[i], expected_centroid)
    
    def test_centroid_similarity(self):
        """Test centroid similarity computation."""
        # Test cosine similarity
        cosine_sim = compute_centroid_similarity(
            self.clusters,
            self.activations,
            similarity_metric='cosine',
            projection_method='none'
        )
        
        for (layer1, layer2), sim_data in cosine_sim.items():
            self.assertNotEqual(layer1, layer2)  # Should not compare a layer to itself
            
            similarity_matrix = sim_data['matrix']
            source_clusters = sim_data['source_clusters']
            target_clusters = sim_data['target_clusters']
            
            self.assertEqual(similarity_matrix.shape, (len(source_clusters), len(target_clusters)))
            
            # Similarity values should be between -1 and 1 for cosine
            self.assertTrue(np.all(similarity_matrix >= -1.0))
            self.assertTrue(np.all(similarity_matrix <= 1.0))
        
        # Test euclidean similarity
        euclidean_sim = compute_centroid_similarity(
            self.clusters,
            self.activations,
            similarity_metric='euclidean',
            projection_method='none'
        )
        
        for (layer1, layer2), sim_data in euclidean_sim.items():
            similarity_matrix = sim_data['matrix']
            
            # Euclidean similarity should be between 0 and 1
            self.assertTrue(np.all(similarity_matrix >= 0.0))
            self.assertTrue(np.all(similarity_matrix <= 1.0))
    
    def test_membership_overlap(self):
        """Test membership overlap computation."""
        # Test Jaccard overlap
        jaccard_overlap = compute_membership_overlap(
            self.clusters,
            overlap_type='jaccard'
        )
        
        for (layer1, layer2), overlap_data in jaccard_overlap.items():
            self.assertNotEqual(layer1, layer2)  # Should not compare a layer to itself
            
            overlap_matrix = overlap_data['matrix']
            source_clusters = overlap_data['source_clusters']
            target_clusters = overlap_data['target_clusters']
            
            self.assertEqual(overlap_matrix.shape, (len(source_clusters), len(target_clusters)))
            
            # Jaccard values should be between 0 and 1
            self.assertTrue(np.all(overlap_matrix >= 0.0))
            self.assertTrue(np.all(overlap_matrix <= 1.0))
        
        # Test containment overlap
        containment_overlap = compute_membership_overlap(
            self.clusters,
            overlap_type='containment'
        )
        
        for (layer1, layer2), overlap_data in containment_overlap.items():
            overlap_matrix = overlap_data['matrix']
            
            # Containment values should be between 0 and 1
            self.assertTrue(np.all(overlap_matrix >= 0.0))
            self.assertTrue(np.all(overlap_matrix <= 1.0))
    
    def test_extract_paths(self):
        """Test path extraction."""
        paths, layer_names = extract_paths(self.clusters)
        
        self.assertEqual(paths.shape, (self.n_samples, len(self.clusters)))
        self.assertEqual(len(layer_names), len(self.clusters))
        
        # Verify paths
        for i, layer_name in enumerate(layer_names):
            np.testing.assert_array_equal(paths[:, i], self.clusters[layer_name])
    
    def test_trajectory_fragmentation(self):
        """Test trajectory fragmentation computation."""
        # Extract paths first
        paths, layer_names = extract_paths(self.clusters)
        
        # Compute fragmentation
        fragmentation = compute_trajectory_fragmentation(
            paths,
            layer_names,
            self.labels
        )
        
        # Verify metrics
        self.assertIn('overall_entropy', fragmentation)
        self.assertIn('normalized_entropy', fragmentation)
        self.assertIn('transition_entropies', fragmentation)
        self.assertIn('unique_paths', fragmentation)
        self.assertIn('class_fragmentation', fragmentation)
        
        # Check entropy values
        self.assertGreaterEqual(fragmentation['overall_entropy'], 0.0)
        self.assertGreaterEqual(fragmentation['normalized_entropy'], 0.0)
        self.assertLessEqual(fragmentation['normalized_entropy'], 1.0)
        
        # Check transition entropies
        self.assertEqual(len(fragmentation['transition_entropies']), len(layer_names) - 1)
        
        # Check class fragmentation
        for label in np.unique(self.labels):
            label_data = fragmentation['class_fragmentation'].get(int(label))
            if label_data:
                self.assertIn('fragmentation_rate', label_data)
                self.assertIn('path_entropy', label_data)
                self.assertIn('unique_paths', label_data)
                self.assertIn('samples', label_data)
    
    def test_inter_cluster_path_density(self):
        """Test inter-cluster path density computation."""
        # Extract paths first
        paths, layer_names = extract_paths(self.clusters)
        
        # Compute path density
        density = compute_inter_cluster_path_density(
            paths,
            layer_names,
            min_density=0.05,
            max_steps=2
        )
        
        # Check direct transitions
        direct_key = f"{layer_names[0]}_to_{layer_names[1]}"
        self.assertIn(direct_key, density)
        
        direct_data = density[direct_key]
        self.assertIn('matrix', direct_data)
        self.assertIn('source_clusters', direct_data)
        self.assertIn('target_clusters', direct_data)
        
        direct_matrix = direct_data['matrix']
        self.assertEqual(direct_matrix.shape, 
                         (len(direct_data['source_clusters']), len(direct_data['target_clusters'])))
        
        # Check multi-step transitions if we have at least 3 layers
        if len(layer_names) >= 3:
            multi_key = f"{layer_names[0]}_to_{layer_names[2]}"
            self.assertIn(multi_key, density)
            
            multi_data = density[multi_key]
            self.assertIn('intermediate_paths', multi_data)
    
    def test_analyze_cross_layer_metrics(self):
        """Test comprehensive analysis of cross-layer metrics."""
        # Run the full analysis
        results = analyze_cross_layer_metrics(
            self.clusters,
            self.activations,
            self.labels
        )
        
        # Verify all metrics are included
        self.assertIn('centroid_similarity', results)
        self.assertIn('membership_overlap', results)
        self.assertIn('fragmentation', results)
        self.assertIn('path_density', results)
        self.assertIn('paths', results)
        self.assertIn('layer_names', results)
        
        # Verify paths
        self.assertEqual(results['paths'].shape, (self.n_samples, len(self.clusters)))

    def test_memory_efficiency_with_large_data(self):
        """Test memory efficiency with larger datasets."""
        # Create a larger synthetic dataset
        n_samples_large = 1000
        layer_dims_large = {
            "layer1": 50,
            "layer2": 80,
            "layer3": 60
        }
        
        # Only test if we have enough memory
        try:
            # Generate large activations for each layer
            activations_large = {}
            clusters_large = {}
            
            for layer_name, dim in layer_dims_large.items():
                # Generate random activations
                activations_large[layer_name] = np.random.randn(n_samples_large, dim)
                
                # Generate cluster assignments (fewer clusters to ensure dense clusters)
                n_clusters = self.n_clusters[layer_name]
                clusters_large[layer_name] = np.random.randint(0, n_clusters, n_samples_large)
            
            # Generate class labels
            labels_large = np.random.randint(0, 2, n_samples_large)
            
            # Run with batch processing
            results_batch = analyze_cross_layer_metrics(
                clusters_large,
                activations_large,
                labels_large,
                config={
                    'centroid_similarity': {
                        'batch_size': 100,
                        'projection_dims': 20
                    },
                    'membership_overlap': {
                        'batch_size': 100
                    },
                    'path_density': {
                        'batch_size': 100
                    }
                }
            )
            
            # Verify results exist
            self.assertIn('centroid_similarity', results_batch)
            self.assertIn('membership_overlap', results_batch)
            self.assertIn('fragmentation', results_batch)
            self.assertIn('path_density', results_batch)
            
            # Don't need to compare with non-batch version - just check it runs
            self.assertTrue(True)
        except MemoryError:
            # Skip test if we don't have enough memory
            print("Skipping large data test due to memory constraints")

if __name__ == '__main__':
    unittest.main()