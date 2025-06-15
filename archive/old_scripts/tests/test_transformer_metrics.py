"""
Tests for transformer-specific metrics.

This module contains unit tests for the transformer-specific metrics
implemented in the concept_fragmentation.metrics package, ensuring that
they work correctly and produce expected results.
"""

import unittest
import numpy as np
import torch
from typing import Dict, List, Any

# Import metrics modules
from concept_fragmentation.metrics.transformer_metrics import (
    calculate_attention_entropy,
    calculate_attention_sparsity,
    calculate_cross_head_agreement,
    analyze_attention_patterns
)

from concept_fragmentation.metrics.token_path_metrics import (
    calculate_token_path_coherence,
    calculate_token_path_divergence,
    calculate_semantic_stability,
    analyze_token_paths
)

from concept_fragmentation.metrics.concept_purity_metrics import (
    calculate_intra_cluster_coherence,
    calculate_cross_layer_stability,
    calculate_concept_separability,
    analyze_concept_purity
)

from concept_fragmentation.analysis.transformer_metrics_integration import (
    GPT2MetricsAnalyzer,
    GPT2AnalysisResult
)


class TestAttentionMetrics(unittest.TestCase):
    """Test cases for attention metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample attention matrices
        # Shape: [batch_size, n_heads, seq_len, seq_len]
        batch_size = 2
        n_heads = 4
        seq_len = 8
        
        # Create uniform attention
        uniform_attention = np.ones((batch_size, n_heads, seq_len, seq_len))
        # Normalize along last dimension
        uniform_attention = uniform_attention / seq_len
        
        # Create focused attention (attending to first token)
        focused_attention = np.zeros((batch_size, n_heads, seq_len, seq_len))
        focused_attention[:, :, :, 0] = 1.0
        
        # Create semi-focused attention
        semi_focused = np.zeros((batch_size, n_heads, seq_len, seq_len))
        # First half of heads attend to first token
        semi_focused[:, :n_heads//2, :, 0] = 1.0
        # Second half of heads have uniform attention
        for h in range(n_heads//2, n_heads):
            semi_focused[:, h, :, :] = 1.0 / seq_len
        
        self.uniform_attention = uniform_attention
        self.focused_attention = focused_attention
        self.semi_focused_attention = semi_focused
        
        # Create attention data dictionary
        self.attention_data = {
            "layer_0": uniform_attention,
            "layer_1": semi_focused_attention,
            "layer_2": focused_attention
        }
        
        # Create token mask
        self.token_mask = np.ones((batch_size, seq_len))
        # Mask some tokens
        self.token_mask[0, -2:] = 0
        self.token_mask[1, -1] = 0
    
    def test_attention_entropy(self):
        """Test attention entropy calculation."""
        # Uniform attention should have maximum entropy
        uniform_entropy = calculate_attention_entropy(self.uniform_attention)
        
        # Focused attention should have minimum entropy
        focused_entropy = calculate_attention_entropy(self.focused_attention)
        
        # Semi-focused should be in between
        semi_entropy = calculate_attention_entropy(self.semi_focused_attention)
        
        # Check shapes
        self.assertEqual(uniform_entropy.shape, (2, 4))
        self.assertEqual(focused_entropy.shape, (2, 4))
        self.assertEqual(semi_entropy.shape, (2, 4))
        
        # Check values
        self.assertTrue(np.all(uniform_entropy > semi_entropy))
        self.assertTrue(np.all(semi_entropy > focused_entropy))
    
    def test_attention_sparsity(self):
        """Test attention sparsity calculation."""
        # Uniform attention should have minimum sparsity
        uniform_sparsity = calculate_attention_sparsity(self.uniform_attention)
        
        # Focused attention should have maximum sparsity
        focused_sparsity = calculate_attention_sparsity(self.focused_attention)
        
        # Semi-focused should be in between
        semi_sparsity = calculate_attention_sparsity(self.semi_focused_attention)
        
        # Check shapes
        self.assertEqual(uniform_sparsity.shape, (2, 4))
        self.assertEqual(focused_sparsity.shape, (2, 4))
        self.assertEqual(semi_sparsity.shape, (2, 4))
        
        # Check values
        self.assertTrue(np.all(focused_sparsity > semi_sparsity))
        self.assertTrue(np.all(semi_sparsity > uniform_sparsity))
    
    def test_cross_head_agreement(self):
        """Test cross-head agreement calculation."""
        # In uniform attention, all heads should agree
        uniform_agreement = calculate_cross_head_agreement(self.uniform_attention)
        
        # In focused attention, all heads should agree
        focused_agreement = calculate_cross_head_agreement(self.focused_attention)
        
        # In semi-focused, heads should partially agree
        semi_agreement = calculate_cross_head_agreement(self.semi_focused_attention)
        
        # Check structure
        self.assertEqual(set(uniform_agreement.keys()), {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)})
        
        # Check values
        for pair, agreement in uniform_agreement.items():
            self.assertAlmostEqual(agreement, 1.0, places=5)
        
        for pair, agreement in focused_agreement.items():
            self.assertAlmostEqual(agreement, 1.0, places=5)
        
        # For semi-focused, we expect lower agreement between different types of heads
        for pair, agreement in semi_agreement.items():
            h1, h2 = pair
            if h1 < 2 and h2 < 2:  # Both focused heads
                self.assertAlmostEqual(agreement, 1.0, places=5)
            elif h1 >= 2 and h2 >= 2:  # Both uniform heads
                self.assertAlmostEqual(agreement, 1.0, places=5)
            else:  # Mixed head types
                self.assertLess(agreement, 0.5)
    
    def test_analyze_attention_patterns(self):
        """Test the combined attention pattern analysis."""
        result = analyze_attention_patterns(
            attention_data=self.attention_data,
            token_mask=self.token_mask,
            include_cross_head_agreement=True
        )
        
        # Check that result has all expected attributes
        self.assertIsNotNone(result.entropy)
        self.assertIsNotNone(result.sparsity)
        self.assertIsNotNone(result.layer_entropy)
        self.assertIsNotNone(result.layer_sparsity)
        self.assertIsNotNone(result.head_entropy)
        self.assertIsNotNone(result.head_sparsity)
        self.assertIsNotNone(result.cross_head_agreement)
        
        # Check that all layers were analyzed
        self.assertEqual(set(result.layers_analyzed), {"layer_0", "layer_1", "layer_2"})
        
        # Check layer ordering in metrics
        self.assertEqual(list(result.layer_entropy.keys()), ["layer_0", "layer_1", "layer_2"])
        
        # Check that cross-head agreement was calculated for each layer
        self.assertEqual(set(result.cross_head_agreement.keys()), {"layer_0", "layer_1", "layer_2"})


class TestTokenPathMetrics(unittest.TestCase):
    """Test cases for token path metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample token representations across layers
        # Shape: [batch_size, seq_len, hidden_dim]
        batch_size = 2
        seq_len = 6
        hidden_dim = 10
        
        # Create sequence of layer representations with gradual changes
        np.random.seed(42)  # For reproducibility
        
        # Layer 0: Initial representation
        layer0 = np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Layer 1: Small change from layer 0
        layer1 = layer0 + 0.1 * np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Layer 2: Small change from layer 1
        layer2 = layer1 + 0.1 * np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Layer 3: Big change for first half of tokens, small change for second half
        layer3 = layer2.copy()
        layer3[:, :seq_len//2, :] += np.random.randn(batch_size, seq_len//2, hidden_dim)
        layer3[:, seq_len//2:, :] += 0.1 * np.random.randn(batch_size, seq_len//2, hidden_dim)
        
        # Layer 4: Big change for all tokens
        layer4 = layer3 + np.random.randn(batch_size, seq_len, hidden_dim)
        
        # Store representations
        self.layer_representations = {
            "layer_0": layer0,
            "layer_1": layer1,
            "layer_2": layer2,
            "layer_3": layer3,
            "layer_4": layer4
        }
        
        # Create token mask
        self.token_mask = np.ones((batch_size, seq_len))
        # Mask some tokens
        self.token_mask[0, -1] = 0
        self.token_mask[1, -1] = 0
        
        # Create token strings
        self.token_strings = [f"token_{i}" for i in range(seq_len)]
    
    def test_token_path_coherence(self):
        """Test token path coherence calculation."""
        coherence = calculate_token_path_coherence(
            layer_representations=self.layer_representations,
            token_mask=self.token_mask,
            metric="cosine"
        )
        
        # Check that we have coherence values for each token
        self.assertEqual(len(coherence), 5)  # 6 tokens - 1 masked = 5
        
        # Coherence values should be between 0 and 1
        for token, value in coherence.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_token_path_divergence(self):
        """Test token path divergence calculation."""
        divergence = calculate_token_path_divergence(
            layer_representations=self.layer_representations,
            token_mask=self.token_mask,
            metric="cosine",
            token_strings=self.token_strings
        )
        
        # Check that we have divergence values for each layer transition
        self.assertEqual(len(divergence), 4)  # 5 layers = 4 transitions
        
        # Check structure of results
        for layer_pair, info in divergence.items():
            self.assertIn("mean", info)
            self.assertIn("max", info)
            self.assertIn("min", info)
            self.assertIn("top_tokens", info)
        
        # Layer 2-3 and 3-4 should have higher divergence than 0-1 and 1-2
        # due to the larger changes in those transitions
        l01_div = divergence[("layer_0", "layer_1")]["mean"]
        l12_div = divergence[("layer_1", "layer_2")]["mean"]
        l23_div = divergence[("layer_2", "layer_3")]["mean"]
        l34_div = divergence[("layer_3", "layer_4")]["mean"]
        
        self.assertLess(l01_div, l23_div)
        self.assertLess(l12_div, l23_div)
        self.assertLess(l01_div, l34_div)
        self.assertLess(l12_div, l34_div)
    
    def test_semantic_stability(self):
        """Test semantic stability calculation."""
        stability = calculate_semantic_stability(
            layer_representations=self.layer_representations,
            token_mask=self.token_mask,
            metric="cosine",
            layer_windows=2
        )
        
        # Check that we have stability values for each token
        self.assertEqual(len(stability), 5)  # 6 tokens - 1 masked = 5
        
        # Stability values should be between 0 and 1
        for token, value in stability.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_analyze_token_paths(self):
        """Test the combined token path analysis."""
        result = analyze_token_paths(
            layer_representations=self.layer_representations,
            token_mask=self.token_mask,
            token_strings=self.token_strings,
            metric="cosine"
        )
        
        # Check that result has all expected attributes
        self.assertIsNotNone(result.path_coherence)
        self.assertIsNotNone(result.path_divergence)
        self.assertIsNotNone(result.semantic_stability)
        self.assertIsNotNone(result.neighborhood_preservation)
        self.assertIsNotNone(result.token_influence)
        self.assertIsNotNone(result.token_specialization)
        self.assertIsNotNone(result.aggregated_metrics)
        
        # Check that all metrics are aggregated
        self.assertIn("average_coherence", result.aggregated_metrics)
        self.assertIn("average_divergence", result.aggregated_metrics)
        self.assertIn("average_stability", result.aggregated_metrics)
        self.assertIn("average_preservation", result.aggregated_metrics)
        self.assertIn("average_influence", result.aggregated_metrics)
        self.assertIn("average_specialization", result.aggregated_metrics)
        
        # Check that all layers were analyzed
        self.assertEqual(set(result.layers_analyzed), 
                         {"layer_0", "layer_1", "layer_2", "layer_3", "layer_4"})


class TestConceptPurityMetrics(unittest.TestCase):
    """Test cases for concept purity metrics."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample activations across layers
        # Shape: [n_samples, n_features]
        n_samples = 100
        n_features = 20
        
        np.random.seed(42)  # For reproducibility
        
        # Create synthetic cluster data
        # We'll create 4 clusters in 2D space, then add noise in higher dimensions
        
        # First, create cluster centers in 2D
        centers = np.array([
            [1, 1],    # Cluster 0
            [-1, 1],   # Cluster 1
            [-1, -1],  # Cluster 2
            [1, -1]    # Cluster 3
        ])
        
        # Generate points around cluster centers
        points_2d = []
        labels = []
        
        for i in range(n_samples):
            # Randomly assign to a cluster
            cluster_id = i % 4
            center = centers[cluster_id]
            
            # Add Gaussian noise
            point = center + 0.2 * np.random.randn(2)
            
            points_2d.append(point)
            labels.append(cluster_id)
        
        points_2d = np.array(points_2d)
        labels = np.array(labels)
        
        # Extend to higher dimensions with noise
        layer0 = np.zeros((n_samples, n_features))
        layer0[:, :2] = points_2d
        layer0[:, 2:] = 0.1 * np.random.randn(n_samples, n_features - 2)
        
        # Create layer1 by adding small noise to layer0
        layer1 = layer0 + 0.05 * np.random.randn(n_samples, n_features)
        
        # Create layer2 with more noise, making clusters less distinct
        layer2 = layer0 + 0.2 * np.random.randn(n_samples, n_features)
        
        # Create layer3 with even more noise, making clusters overlap
        layer3 = layer0 + 0.5 * np.random.randn(n_samples, n_features)
        
        # Store activations
        self.layer_activations = {
            "layer_0": layer0,
            "layer_1": layer1,
            "layer_2": layer2,
            "layer_3": layer3
        }
        
        # Store ground truth labels
        self.ground_truth = labels
        
        # Create cluster labels for each layer
        # We'll make layer0 match ground truth perfectly
        # Then gradually introduce errors in later layers
        from sklearn.cluster import KMeans
        
        self.cluster_labels = {}
        self.cluster_labels["layer_0"] = labels.copy()
        
        # Layer 1: 90% accuracy
        layer1_labels = labels.copy()
        error_indices = np.random.choice(n_samples, size=n_samples//10, replace=False)
        for idx in error_indices:
            layer1_labels[idx] = (labels[idx] + 1) % 4
        self.cluster_labels["layer_1"] = layer1_labels
        
        # Layer 2: Using KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.cluster_labels["layer_2"] = kmeans.fit_predict(layer2)
        
        # Layer 3: Using KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.cluster_labels["layer_3"] = kmeans.fit_predict(layer3)
    
    def test_intra_cluster_coherence(self):
        """Test intra-cluster coherence calculation."""
        coherence = calculate_intra_cluster_coherence(
            layer_activations=self.layer_activations,
            cluster_labels=self.cluster_labels,
            metric="cosine"
        )
        
        # Check that we have coherence values for each layer
        self.assertEqual(len(coherence), 4)
        
        # Coherence should decrease in later layers as clusters become less distinct
        self.assertGreater(coherence["layer_0"], coherence["layer_2"])
        self.assertGreater(coherence["layer_1"], coherence["layer_3"])
    
    def test_cross_layer_stability(self):
        """Test cross-layer stability calculation."""
        stability = calculate_cross_layer_stability(
            cluster_labels=self.cluster_labels
        )
        
        # Check that we have stability values for each layer transition
        self.assertEqual(len(stability), 3)  # 4 layers = 3 transitions
        
        # Stability should decrease for later transitions as errors accumulate
        l01_stability = stability[("layer_0", "layer_1")]
        l12_stability = stability[("layer_1", "layer_2")]
        l23_stability = stability[("layer_2", "layer_3")]
        
        self.assertGreater(l01_stability, l12_stability)
    
    def test_concept_separability(self):
        """Test concept separability calculation."""
        separability = calculate_concept_separability(
            layer_activations=self.layer_activations,
            cluster_labels=self.cluster_labels,
            metric="cosine"
        )
        
        # Check that we have separability values for each layer
        self.assertEqual(len(separability), 4)
        
        # Check structure of results
        for layer, metrics in separability.items():
            self.assertIn("calinski_harabasz", metrics)
            self.assertIn("davies_bouldin", metrics)
            self.assertIn("between_within_ratio", metrics)
        
        # Separability should decrease in later layers as clusters become less distinct
        ratio0 = separability["layer_0"]["between_within_ratio"]
        ratio3 = separability["layer_3"]["between_within_ratio"]
        
        self.assertGreater(ratio0, ratio3)
    
    def test_analyze_concept_purity(self):
        """Test the combined concept purity analysis."""
        result = analyze_concept_purity(
            layer_activations=self.layer_activations,
            cluster_labels=self.cluster_labels,
            ground_truth_labels=self.ground_truth,
            metric="cosine"
        )
        
        # Check that result has all expected attributes
        self.assertIsNotNone(result.intra_cluster_coherence)
        self.assertIsNotNone(result.cross_layer_stability)
        self.assertIsNotNone(result.concept_separability)
        self.assertIsNotNone(result.concept_entropy)
        self.assertIsNotNone(result.layer_concept_purity)
        self.assertIsNotNone(result.concept_contamination)
        self.assertIsNotNone(result.concept_overlap)
        self.assertIsNotNone(result.aggregated_metrics)
        
        # Check that all metrics are aggregated
        self.assertIn("average_coherence", result.aggregated_metrics)
        self.assertIn("average_stability", result.aggregated_metrics)
        self.assertIn("average_separability", result.aggregated_metrics)
        self.assertIn("average_purity", result.aggregated_metrics)
        
        # Check that all layers were analyzed
        self.assertEqual(set(result.layers_analyzed), 
                         {"layer_0", "layer_1", "layer_2", "layer_3"})
        
        # Check that concept metrics were calculated
        self.assertEqual(len(result.concept_contamination), 4)  # 4 clusters
        self.assertGreaterEqual(len(result.concept_overlap), 6)  # 4C2 = 6 pairs


if __name__ == "__main__":
    unittest.main()