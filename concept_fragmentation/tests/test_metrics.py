import unittest
import torch
import numpy as np
from sklearn.datasets import make_blobs
from ..metrics.cluster_entropy import compute_cluster_entropy as cluster_entropy
from ..metrics.subspace_angle import compute_subspace_angle as subspace_angle, compute_principal_angles as principal_angles

class TestClusterEntropy(unittest.TestCase):
    """Test cases for the cluster entropy metric."""
    
    def setUp(self):
        """Set up test data with controlled clustering properties."""
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Case 1: Perfectly separated clusters (zero entropy)
        X1, y1 = make_blobs(
            n_samples=100, 
            n_features=10, 
            centers=2, 
            cluster_std=0.1,
            random_state=42
        )
        self.separated_activations = torch.tensor(X1, dtype=torch.float32)
        self.separated_labels = torch.tensor(y1, dtype=torch.long)
        
        # Case 2: Overlapping clusters (high entropy)
        X2, y2 = make_blobs(
            n_samples=100, 
            n_features=10, 
            centers=2, 
            cluster_std=10.0,
            random_state=42
        )
        self.overlapping_activations = torch.tensor(X2, dtype=torch.float32)
        self.overlapping_labels = torch.tensor(y2, dtype=torch.long)
        
        # Case 3: Multi-class with different entropies
        X3, y3 = make_blobs(
            n_samples=150, 
            n_features=10, 
            centers=3, 
            cluster_std=[0.5, 2.0, 5.0],
            random_state=42
        )
        self.multi_class_activations = torch.tensor(X3, dtype=torch.float32)
        self.multi_class_labels = torch.tensor(y3, dtype=torch.long)
        
        # Case 4: Dict input format
        self.activations_dict = {
            'layer1': self.separated_activations,
            'layer2': self.overlapping_activations
        }

    def test_separated_clusters(self):
        """Test entropy calculation with well-separated clusters."""
        result = cluster_entropy(
            self.separated_activations, 
            self.separated_labels
        )
        
        # Well-separated clusters should have very low entropy
        self.assertIn('class_entropies', result)
        for class_idx, entropy in result['class_entropies'].items():
            self.assertLessEqual(entropy, 0.35, 
                                f"Expected low entropy for well-separated clusters, got {entropy}")
        
        # Check aggregate metrics and that k was chosen
        self.assertIn('mean_entropy', result)
        self.assertIn('chosen_k', result)
        self.assertLessEqual(result['mean_entropy'], 0.35)
        
    def test_overlapping_clusters(self):
        """Test entropy calculation with overlapping clusters."""
        result = cluster_entropy(
            self.overlapping_activations, 
            self.overlapping_labels
        )
        
        # Overlapping clusters should have higher entropy
        self.assertIn('class_entropies', result)
        for class_idx, entropy in result['class_entropies'].items():
            self.assertGreaterEqual(entropy, 0.4, 
                                   f"Expected high entropy for overlapping clusters, got {entropy}")
        
    def test_multi_class(self):
        """Test entropy calculation with multiple classes of varying separation."""
        result = cluster_entropy(
            self.multi_class_activations, 
            self.multi_class_labels
        )
        
        # Each class should have different entropy values
        self.assertIn('class_entropies', result)
        self.assertEqual(len(result['class_entropies']), 3, 
                         f"Expected 3 classes, got {len(result['class_entropies'])}")
        
        # Class 0 (tight cluster) should have lower entropy than class 2 (dispersed)
        if 0 in result['class_entropies'] and 2 in result['class_entropies']:
            # Check with a small tolerance to handle the case where both might be 0
            self.assertLessEqual(
                result['class_entropies'][0],
                result['class_entropies'][2] + 1e-5,
                "Expected class 0 (tight) to have no higher entropy than class 2 (dispersed)"
            )
            
    def test_dict_input(self):
        """Test using dictionary input format with layer name."""
        result = cluster_entropy(
            self.activations_dict, 
            self.separated_labels,
            layer_name='layer1'
        )
        
        # Results should match the direct tensor input for the same layer
        direct_result = cluster_entropy(
            self.separated_activations, 
            self.separated_labels
        )
        
        self.assertAlmostEqual(result['mean_entropy'], direct_result['mean_entropy'], places=5)
        
    def test_return_clusters(self):
        """Test returning cluster assignments."""
        result = cluster_entropy(
            self.separated_activations, 
            self.separated_labels,
            return_clusters=True
        )
        
        # Should include cluster assignments
        self.assertIn('cluster_assignments', result)
        
        # Cluster assignments should have the right format
        clusters = result['cluster_assignments']
        self.assertIsInstance(clusters, dict)
        
        # Each class should have assignments
        for class_idx in np.unique(self.separated_labels.numpy()):
            self.assertIn(int(class_idx), clusters)
            
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Single sample per class
        single_sample_act = torch.randn(2, 10)
        single_sample_labels = torch.tensor([0, 1])
        
        result = cluster_entropy(
            single_sample_act, 
            single_sample_labels
        )
        
        # Should handle without errors, chosen_k should be 1, entropy should be 0
        self.assertEqual(result['chosen_k'], 1)
        self.assertEqual(result['class_entropies'][0], 0.0)
        self.assertEqual(result['class_entropies'][1], 0.0)
        
        # Dictionary input without layer name should raise error
        with self.assertRaises(ValueError):
            cluster_entropy(
                self.activations_dict, 
                self.separated_labels
            )
            
    def test_fixed_k_selection(self):
        """Test using fixed k selection instead of auto."""
        result = cluster_entropy(
            self.separated_activations, 
            self.separated_labels,
            n_clusters=2,  # Use smaller k for more stable results
            k_selection='fixed'
        )
        
        # With fixed k, chosen_k should not be in results
        self.assertNotIn('chosen_k', result)
        
        # Should still have entropy results
        self.assertIn('class_entropies', result)
        for class_idx, entropy in result['class_entropies'].items():
            self.assertLessEqual(entropy, 0.35)


class TestSubspaceAngle(unittest.TestCase):
    """Test cases for the subspace angle metric."""
    
    def setUp(self):
        """Set up test data with controlled subspace properties."""
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Case 1: Orthogonal subspaces (90 degree angles)
        basis1 = np.zeros((10, 3))
        basis1[0:3, 0] = 1.0  # Span first 3 dimensions only
        basis2 = np.zeros((10, 3))
        basis2[3:6, 0] = 1.0  # Span dimensions 3-5 only
        
        # Generate data in these subspaces
        n_samples = 50
        self.orthogonal_activations = torch.zeros((n_samples * 2, 10))
        
        # First class - only first 3 dimensions have values
        class1 = torch.randn(n_samples, 3) @ torch.tensor(basis1[0:3, :].T, dtype=torch.float32)
        self.orthogonal_activations[:n_samples, 0:3] = class1
        
        # Second class - only dimensions 3-5 have values
        class2 = torch.randn(n_samples, 3) @ torch.tensor(basis2[3:6, :].T, dtype=torch.float32)
        self.orthogonal_activations[n_samples:, 3:6] = class2
        
        self.orthogonal_labels = torch.cat([
            torch.zeros(n_samples, dtype=torch.long),
            torch.ones(n_samples, dtype=torch.long)
        ])
        
        # Case 2: Parallel subspaces (0 degree angles)
        self.parallel_activations = torch.zeros((n_samples * 2, 10))
        
        # Both classes operate in the same subspace
        class1 = torch.randn(n_samples, 3) @ torch.tensor(basis1[0:3, :].T, dtype=torch.float32)
        class2 = torch.randn(n_samples, 3) @ torch.tensor(basis1[0:3, :].T, dtype=torch.float32)
        
        self.parallel_activations[:n_samples, 0:3] = class1
        self.parallel_activations[n_samples:, 0:3] = class2
        
        self.parallel_labels = torch.cat([
            torch.zeros(n_samples, dtype=torch.long),
            torch.ones(n_samples, dtype=torch.long)
        ])
        
        # Case 3: Dict input format
        self.activations_dict = {
            'layer1': self.orthogonal_activations,
            'layer2': self.parallel_activations
        }
        
    def test_principal_angles(self):
        """Test principal angles calculation directly."""
        # Create orthogonal bases
        basis1 = np.eye(5, 3)  # 5x3 matrix with orthonormal columns
        basis2 = np.zeros((5, 2))
        basis2[3:5, :] = np.eye(2)  # 5x2 matrix with orthonormal columns
        
        angles = principal_angles(basis1, basis2)
        
        # All angles should be 90 degrees
        self.assertEqual(len(angles), min(basis1.shape[1], basis2.shape[1]))
        for angle in angles:
            self.assertGreaterEqual(angle, 80.0, f"Expected angle close to 90 degrees, got {angle}")
            
        # Create parallel bases
        basis3 = np.eye(5, 2)  # First two standard basis vectors
        basis4 = np.eye(5, 2)  # Same subspace
        
        angles = principal_angles(basis3, basis4)
        
        # All angles should be close to 0 degrees
        for angle in angles:
            self.assertLessEqual(angle, 10.0, f"Expected angle close to 0 degrees, got {angle}")
            
    def test_orthogonal_subspaces(self):
        """Test with classes having orthogonal subspaces."""
        result = subspace_angle(
            self.orthogonal_activations, 
            self.orthogonal_labels,
            n_components=3
        )
        
        # Check results structure
        self.assertIn('mean_angle', result)
        self.assertIn('class_angles', result)
        
        # Angles should be close to 90 degrees for orthogonal subspaces
        self.assertGreaterEqual(result['mean_angle'], 80.0, 
                              f"Expected angle close to 90° for orthogonal subspaces, got {result['mean_angle']}°")

    def test_parallel_subspaces(self):
        """Test with classes having parallel subspaces."""
        result = subspace_angle(
            self.parallel_activations, 
            self.parallel_labels,
            n_components=3
        )
        
        # Check results structure
        self.assertIn('mean_angle', result)
        self.assertIn('class_angles', result)
        
        # Angles should be close to 0 degrees for parallel subspaces
        self.assertLessEqual(result['mean_angle'], 10.0, 
                           f"Expected angle close to 0° for parallel subspaces, got {result['mean_angle']}°")

    def test_dict_input(self):
        """Test using dictionary input format with layer name."""
        result = subspace_angle(
            self.activations_dict, 
            self.orthogonal_labels, 
            layer_name='layer1',
            n_components=3
        )
        
        # Results should match the direct tensor input for the same layer
        direct_result = subspace_angle(
            self.orthogonal_activations, 
            self.orthogonal_labels,
            n_components=3
        )
        
        self.assertAlmostEqual(result['mean_angle'], direct_result['mean_angle'], places=1)
        
    def test_bootstrap_statistics(self):
        """Test bootstrap statistics."""
        result = subspace_angle(
            self.orthogonal_activations, 
            self.orthogonal_labels,
            n_components=3,
            bootstrap_samples=5
        )
        
        # Check that confidence intervals exist
        for class_label, stats in result['class_angles'].items():
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('ci_lower', stats)
            self.assertIn('ci_upper', stats)
            
            # Lower bound should be less than mean
            self.assertLessEqual(stats['ci_lower'], stats['mean'])
            # Upper bound should be greater than mean
            self.assertGreaterEqual(stats['ci_upper'], stats['mean'])
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Dictionary input without layer name should raise error
        with self.assertRaises(ValueError):
            subspace_angle(
                self.activations_dict, 
                self.orthogonal_labels,
                n_components=3
            )
            
        # Too few samples per class
        few_samples_act = torch.randn(4, 10)
        few_samples_labels = torch.tensor([0, 0, 1, 1])
        
        result = subspace_angle(
            few_samples_act, 
            few_samples_labels,
            n_components=1
        )
        
        # Should handle gracefully with NaN values or reasonable values
        self.assertTrue(np.isnan(result['mean_angle']) or result['mean_angle'] < 30.0,
                        "Expected NaN or reasonable angle value for too few samples")

    def test_variance_threshold(self):
        """Test using variance threshold for component selection."""
        # Create dataset with clear variance distribution
        n_samples = 100
        n_features = 20
        
        # Generate data with decreasing variance in each dimension
        X = np.zeros((n_samples, n_features))
        for i in range(n_features):
            X[:, i] = np.random.normal(0, 10 / (i + 1), n_samples)
        
        # Create activations with two classes
        activations = torch.tensor(X, dtype=torch.float32)
        activations_class1 = activations[:n_samples//2].clone()
        activations_class2 = activations[n_samples//2:].clone()
        
        # Add class-specific signal to first few dimensions
        activations_class1[:, 0:3] += 5.0
        activations_class2[:, 0:3] -= 5.0
        
        # Combined activations
        combined_activations = torch.cat([activations_class1, activations_class2])
        labels = torch.cat([
            torch.zeros(n_samples//2, dtype=torch.long),
            torch.ones(n_samples//2, dtype=torch.long)
        ])
        
        # Test with high variance threshold (should select fewer components)
        result_high = subspace_angle(
            combined_activations, 
            labels,
            var_threshold=0.95
        )
        
        # Test with low variance threshold (should select more components)
        result_low = subspace_angle(
            combined_activations, 
            labels,
            var_threshold=0.7
        )
        
        # Both should return valid results
        self.assertIn('mean_angle', result_high)
        self.assertIn('mean_angle', result_low)
        
        # Test backward compatibility with n_components
        result_legacy = subspace_angle(
            combined_activations, 
            labels,
            n_components=5
        )
        
        self.assertIn('mean_angle', result_legacy)

if __name__ == '__main__':
    unittest.main()
