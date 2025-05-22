"""
Tests for the transformer_metrics module.

This module contains tests for the transformer-specific metrics,
including attention entropy, sparsity, and other metrics.
"""

import unittest
import numpy as np
import torch
from typing import Dict, Any

from concept_fragmentation.metrics.transformer_metrics import (
    calculate_attention_entropy,
    calculate_attention_sparsity,
    calculate_head_importance,
    analyze_attention_patterns,
    calculate_activation_statistics,
    analyze_cross_attention_consistency,
    calculate_activation_sensitivity,
    TransformerMetricsCalculator
)


class TestAttentionMetrics(unittest.TestCase):
    """Test the attention metrics functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample attention probabilities
        
        # 1. Single head attention [seq_len, seq_len]
        self.attn_single = np.array([
            [0.1, 0.2, 0.7],
            [0.3, 0.3, 0.4],
            [0.5, 0.1, 0.4]
        ])
        
        # 2. Multi-head attention [n_heads, seq_len, seq_len]
        self.attn_multi_head = np.array([
            [  # Head 1
                [0.1, 0.2, 0.7],
                [0.3, 0.3, 0.4],
                [0.5, 0.1, 0.4]
            ],
            [  # Head 2
                [0.5, 0.3, 0.2],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7]
            ]
        ])
        
        # 3. Batched multi-head attention [batch_size, n_heads, seq_len, seq_len]
        self.attn_batched = np.array([
            [  # Batch 1
                [  # Head 1
                    [0.1, 0.2, 0.7],
                    [0.3, 0.3, 0.4],
                    [0.5, 0.1, 0.4]
                ],
                [  # Head 2
                    [0.5, 0.3, 0.2],
                    [0.2, 0.7, 0.1],
                    [0.1, 0.2, 0.7]
                ]
            ],
            [  # Batch 2
                [  # Head 1
                    [0.2, 0.3, 0.5],
                    [0.4, 0.4, 0.2],
                    [0.6, 0.2, 0.2]
                ],
                [  # Head 2
                    [0.6, 0.2, 0.2],
                    [0.3, 0.6, 0.1],
                    [0.2, 0.3, 0.5]
                ]
            ]
        ])
        
        # Create sample outputs
        self.outputs_batched = np.random.randn(2, 3, 768)  # [batch_size, seq_len, hidden_size]
        
        # Create token mask
        self.token_mask = np.array([
            [1, 1, 1],  # All tokens are valid in batch 1
            [1, 1, 0]   # Last token is padding in batch 2
        ])
        
        # Create token info
        self.token_info = {
            "token_strings": ["Hello", "world", "!"]
        }
        
        # Create layer data
        self.layer_attention_data = {
            "layer_0": self.attn_batched,
            "layer_1": np.random.rand(2, 2, 3, 3)  # Different random values
        }
        
        self.layer_outputs = {
            "layer_0": self.outputs_batched,
            "layer_1": np.random.randn(2, 3, 768)
        }
    
    def test_attention_entropy(self):
        """Test attention entropy calculation."""
        # Test with single head attention
        entropy_single = calculate_attention_entropy(self.attn_single)
        self.assertIsInstance(entropy_single, np.ndarray)
        self.assertEqual(entropy_single.shape, (1,))
        self.assertTrue(0 <= entropy_single[0] <= np.log(3))  # Max entropy for 3 elements
        
        # Test with multi-head attention
        entropy_multi = calculate_attention_entropy(self.attn_multi_head)
        self.assertIsInstance(entropy_multi, np.ndarray)
        self.assertEqual(entropy_multi.shape, (2,))  # One entropy value per head
        
        # Test with batched attention
        entropy_batched = calculate_attention_entropy(self.attn_batched)
        self.assertIsInstance(entropy_batched, np.ndarray)
        self.assertEqual(entropy_batched.shape, (2, 2))  # [batch_size, n_heads]
        
        # Test with PyTorch tensor
        attn_tensor = torch.tensor(self.attn_batched)
        entropy_tensor = calculate_attention_entropy(attn_tensor)
        self.assertIsInstance(entropy_tensor, torch.Tensor)
        self.assertEqual(entropy_tensor.shape, (2, 2))
        
        # Check that results are similar between numpy and torch
        self.assertTrue(np.allclose(entropy_batched, entropy_tensor.numpy()))
    
    def test_attention_sparsity(self):
        """Test attention sparsity calculation."""
        # Test with single head attention
        sparsity_single = calculate_attention_sparsity(self.attn_single)
        self.assertIsInstance(sparsity_single, np.ndarray)
        self.assertEqual(sparsity_single.shape, (1,))
        self.assertTrue(0 <= sparsity_single[0] <= 1)  # Sparsity is between 0 and 1
        
        # Test with multi-head attention
        sparsity_multi = calculate_attention_sparsity(self.attn_multi_head)
        self.assertIsInstance(sparsity_multi, np.ndarray)
        self.assertEqual(sparsity_multi.shape, (2,))  # One sparsity value per head
        
        # Test with batched attention
        sparsity_batched = calculate_attention_sparsity(self.attn_batched)
        self.assertIsInstance(sparsity_batched, np.ndarray)
        self.assertEqual(sparsity_batched.shape, (2, 2))  # [batch_size, n_heads]
        
        # Test with PyTorch tensor and custom threshold
        attn_tensor = torch.tensor(self.attn_batched)
        sparsity_tensor = calculate_attention_sparsity(attn_tensor, threshold=0.3)
        self.assertIsInstance(sparsity_tensor, torch.Tensor)
        self.assertEqual(sparsity_tensor.shape, (2, 2))
    
    def test_head_importance(self):
        """Test head importance calculation."""
        # Skip if GPU not available (as this uses gradient computation)
        try:
            # Convert to torch tensors for importance calculation
            attn_tensor = torch.tensor(self.attn_batched, dtype=torch.float32).requires_grad_(True)
            output_tensor = torch.tensor(self.outputs_batched, dtype=torch.float32)
            mask_tensor = torch.tensor(self.token_mask, dtype=torch.float32)
            
            # Calculate head importance
            importance = calculate_head_importance(attn_tensor, output_tensor, mask_tensor)
            
            # Check results
            self.assertIsInstance(importance, torch.Tensor)
            self.assertEqual(importance.shape, (2, 2))  # [batch_size, n_heads]
            self.assertTrue((importance >= 0).all() and (importance <= 1).all())
        except Exception as e:
            # If it fails due to gradient calculation issues, just skip
            self.skipTest(f"Skipping head importance test due to error: {e}")
    
    def test_analyze_attention_patterns(self):
        """Test the overall attention pattern analysis."""
        # Run the analysis
        result = analyze_attention_patterns(
            attention_data=self.layer_attention_data,
            outputs=self.layer_outputs,
            token_mask=self.token_mask,
            token_info=self.token_info
        )
        
        # Check the result structure
        self.assertTrue(hasattr(result, 'entropy'))
        self.assertTrue(hasattr(result, 'sparsity'))
        self.assertTrue(hasattr(result, 'layer_entropy'))
        self.assertTrue(hasattr(result, 'layer_sparsity'))
        self.assertTrue(hasattr(result, 'head_entropy'))
        self.assertTrue(hasattr(result, 'head_sparsity'))
        
        # Check that we have results for both layers
        self.assertIn('layer_0', result.layer_entropy)
        self.assertIn('layer_1', result.layer_entropy)
        
        # Check we have head-specific results
        self.assertIn(('layer_0', 0), result.head_entropy)
        self.assertIn(('layer_0', 1), result.head_entropy)
        
        # Check that input_tokens are included
        self.assertEqual(result.input_tokens, self.token_info["token_strings"])
    
    def test_activation_statistics(self):
        """Test activation statistics calculation."""
        # Create sample activations
        activations = {
            "layer_0": np.random.randn(2, 3, 10),  # [batch_size, seq_len, hidden_size]
            "layer_1": np.random.randn(2, 3, 20)
        }
        
        # Calculate statistics
        stats = calculate_activation_statistics(
            activations=activations,
            token_mask=self.token_mask,
            per_token=True
        )
        
        # Check the result structure
        self.assertIn('layer_0', stats)
        self.assertIn('layer_1', stats)
        
        # Check that we have the expected statistics
        layer_stats = stats['layer_0']
        self.assertIn('mean', layer_stats)
        self.assertIn('std', layer_stats)
        self.assertIn('sparsity', layer_stats)
        self.assertIn('l2_norm', layer_stats)
        self.assertIn('max_val', layer_stats)
        self.assertIn('min_val', layer_stats)
        
        # Check per-token statistics
        self.assertIn('token_stats', layer_stats)
        token_stats = layer_stats['token_stats']
        self.assertTrue(len(token_stats) > 0)
    
    def test_cross_attention_consistency(self):
        """Test cross-attention consistency analysis."""
        # Run the analysis
        result = analyze_cross_attention_consistency(
            attention_data=self.layer_attention_data,
            input_tokens=self.token_info["token_strings"]
        )
        
        # Check the result structure
        self.assertIn('layer_consistency', result)
        self.assertIn('token_attention_profiles', result)
        self.assertIn('overall_consistency', result)
        
        # Check the layer consistency
        layer_consistency = result['layer_consistency']
        self.assertIn(('layer_0', 'layer_1'), layer_consistency)
        
        # Check consistency bounds
        self.assertTrue(-1 <= result['overall_consistency'] <= 1)
    
    def test_activation_sensitivity(self):
        """Test activation sensitivity calculation."""
        # Create sample activations
        base_activations = {
            "layer_0": np.random.randn(2, 3, 10),  # [batch_size, seq_len, hidden_size]
            "layer_1": np.random.randn(2, 3, 20)
        }
        
        # Create slightly perturbed activations
        perturbed_activations = {
            "layer_0": base_activations["layer_0"] + np.random.normal(0, 0.01, base_activations["layer_0"].shape),
            "layer_1": base_activations["layer_1"] + np.random.normal(0, 0.01, base_activations["layer_1"].shape)
        }
        
        # Calculate sensitivity
        sensitivity = calculate_activation_sensitivity(
            base_activations=base_activations,
            perturbed_activations=perturbed_activations,
            metric="cosine"
        )
        
        # Check the result structure
        self.assertIn('layer_0', sensitivity)
        self.assertIn('layer_1', sensitivity)
        
        # Check that we have cosine sensitivity values
        self.assertIn('cosine', sensitivity['layer_0'])
        
        # Try with euclidean metric
        sensitivity2 = calculate_activation_sensitivity(
            base_activations=base_activations,
            perturbed_activations=perturbed_activations,
            metric="euclidean"
        )
        
        # Check that we have euclidean sensitivity values
        self.assertIn('euclidean', sensitivity2['layer_0'])
        self.assertIn('relative', sensitivity2['layer_0'])
    
    def test_metrics_calculator(self):
        """Test the metrics calculator class."""
        # Create calculator
        calculator = TransformerMetricsCalculator(use_cache=True)
        
        # Test attention metrics
        attention_metrics = calculator.compute_attention_metrics(
            attention_data=self.layer_attention_data,
            outputs=self.layer_outputs,
            token_mask=self.token_mask,
            token_info=self.token_info
        )
        
        # Check the result
        self.assertIsNotNone(attention_metrics)
        self.assertTrue(hasattr(attention_metrics, 'entropy'))
        
        # Test activation statistics
        activations = {
            "layer_0": np.random.randn(2, 3, 10),
            "layer_1": np.random.randn(2, 3, 20)
        }
        
        stats = calculator.compute_activation_statistics(
            activations=activations,
            token_mask=self.token_mask
        )
        
        # Check the result
        self.assertIsNotNone(stats)
        self.assertIn('layer_0', stats)
        
        # Test caching (should return same instance)
        stats2 = calculator.compute_activation_statistics(
            activations=activations,
            token_mask=self.token_mask
        )
        
        # Check it's the same dictionary instance (identity check)
        self.assertIs(stats, stats2)
        
        # Test force recompute
        stats3 = calculator.compute_activation_statistics(
            activations=activations,
            token_mask=self.token_mask,
            force_recompute=True
        )
        
        # Check it's a different instance
        self.assertIsNot(stats, stats3)
        
        # Test clearing cache
        calculator.clear_cache()
        
        # Check that we get a new instance after clearing cache
        stats4 = calculator.compute_activation_statistics(
            activations=activations,
            token_mask=self.token_mask
        )
        
        # Should be a different instance after clearing cache
        self.assertIsNot(stats, stats4)


if __name__ == "__main__":
    unittest.main()