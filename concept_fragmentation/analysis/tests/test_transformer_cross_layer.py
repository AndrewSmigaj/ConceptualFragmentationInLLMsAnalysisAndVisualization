"""
Tests for the transformer_cross_layer module.

This module contains tests for the TransformerCrossLayerAnalyzer class
and related functionality for cross-layer analysis in transformer models.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import os
import networkx as nx

# Import the module to test
from concept_fragmentation.analysis.transformer_cross_layer import (
    TransformerCrossLayerAnalyzer,
    CrossLayerTransformerAnalysisResult
)

# Import related modules
from concept_fragmentation.analysis.transformer_dimensionality import (
    TransformerDimensionalityReducer
)
from concept_fragmentation.metrics.transformer_metrics import (
    TransformerMetricsCalculator
)


class TestTransformerCrossLayerAnalyzer(unittest.TestCase):
    """Test the TransformerCrossLayerAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Create analyzer instance
        self.analyzer = TransformerCrossLayerAnalyzer(
            cache_dir=self.temp_dir,
            random_state=42,
            use_cache=True
        )
        
        # Generate test data with different shapes
        
        # 1. Layer activations for a small transformer
        self.layer_activations = {
            "layer0": np.random.normal(size=(4, 16, 128)),  # [batch, seq_len, hidden_dim]
            "layer1": np.random.normal(size=(4, 16, 128)),
            "layer2": np.random.normal(size=(4, 16, 128))
        }
        
        # 2. Attention matrices
        self.attention_data = {
            "layer0": np.random.normal(size=(4, 8, 16, 16)),  # [batch, n_heads, seq_len, seq_len]
            "layer1": np.random.normal(size=(4, 8, 16, 16)),
            "layer2": np.random.normal(size=(4, 8, 16, 16))
        }
        
        # Normalize attention probabilities (must sum to 1)
        for layer, attn in self.attention_data.items():
            # Apply softmax over the last dimension
            exp_attn = np.exp(attn)
            self.attention_data[layer] = exp_attn / np.sum(exp_attn, axis=-1, keepdims=True)
        
        # 3. Token IDs and strings
        self.token_ids = np.random.randint(0, 1000, size=(4, 16))
        self.token_strings = [f"token_{i}" for i in range(16)]
        
        # 4. Class labels for a toy classification task
        self.class_labels = np.array([0, 1, 0, 1])
        
        # 5. PyTorch tensor versions
        self.torch_layer_activations = {
            layer: torch.tensor(data, dtype=torch.float32)
            for layer, data in self.layer_activations.items()
        }
        
        self.torch_attention_data = {
            layer: torch.tensor(data, dtype=torch.float32)
            for layer, data in self.attention_data.items()
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_analyze_layer_relationships(self):
        """Test analysis of relationships between layers."""
        # Analyze layer relationships
        relationships = self.analyzer.analyze_layer_relationships(
            layer_activations=self.layer_activations,
            similarity_metric="cosine",
            dimensionality_reduction=True,
            n_components=32
        )
        
        # Check the result
        self.assertIn("similarity_matrix", relationships)
        self.assertIn("graph", relationships)
        self.assertIn("layers", relationships)
        self.assertIn("activation_stats", relationships)
        
        # Check similarity matrix
        self.assertEqual(len(relationships["similarity_matrix"]), 3)  # 3 pairs for 3 layers
        
        # Check that graph is valid
        self.assertIsInstance(relationships["graph"], nx.Graph)
        self.assertEqual(len(relationships["graph"].nodes), 3)
        self.assertEqual(len(relationships["graph"].edges), 3)
        
        # Test with different similarity metric
        relationships_cka = self.analyzer.analyze_layer_relationships(
            layer_activations=self.layer_activations,
            similarity_metric="cka",
            dimensionality_reduction=False
        )
        
        self.assertEqual(relationships_cka["similarity_metric"], "cka")
        
        # Test with PyTorch tensors
        relationships_torch = self.analyzer.analyze_layer_relationships(
            layer_activations=self.torch_layer_activations,
            similarity_metric="cosine",
            dimensionality_reduction=True,
            n_components=32
        )
        
        self.assertIn("similarity_matrix", relationships_torch)
    
    def test_analyze_attention_flow(self):
        """Test analysis of attention flow across layers."""
        # Analyze attention flow
        flow = self.analyzer.analyze_attention_flow(
            attention_data=self.attention_data,
            layer_order=["layer0", "layer1", "layer2"]
        )
        
        # Check the result
        self.assertIn("layer_entropy", flow)
        self.assertIn("layer_entropy_differences", flow)
        self.assertIn("attention_flow_direction", flow)
        self.assertIn("attention_pattern_consistency", flow)
        self.assertIn("flow_graph", flow)
        
        # Check layer entropy
        self.assertEqual(len(flow["layer_entropy"]), 3)
        
        # Check entropy differences
        self.assertEqual(len(flow["layer_entropy_differences"]), 2)  # 2 transitions for 3 layers
        
        # Check flow graph
        self.assertIsInstance(flow["flow_graph"], nx.DiGraph)
        self.assertEqual(len(flow["flow_graph"].nodes), 3)
        self.assertEqual(len(flow["flow_graph"].edges), 2)
        
        # Test with PyTorch tensors
        flow_torch = self.analyzer.analyze_attention_flow(
            attention_data=self.torch_attention_data
        )
        
        self.assertIn("layer_entropy", flow_torch)
    
    def test_analyze_token_trajectory(self):
        """Test analysis of token trajectory across layers."""
        # Analyze token trajectory
        trajectory = self.analyzer.analyze_token_trajectory(
            layer_activations=self.layer_activations,
            token_ids=self.token_ids,
            token_strings=self.token_strings,
            focus_tokens=["token_1", "token_5"],
            layer_order=["layer0", "layer1", "layer2"]
        )
        
        # Check the result
        self.assertIn("token_stability", trajectory)
        self.assertIn("semantic_shift", trajectory)
        self.assertIn("avg_token_stability", trajectory)
        self.assertIn("avg_semantic_shift", trajectory)
        self.assertIn("trajectory_graph", trajectory)
        self.assertIn("focused_tokens", trajectory)
        
        # Check token stability
        self.assertGreaterEqual(len(trajectory["token_stability"]), 1)
        
        # Check that the focused tokens are included
        token_keys = list(trajectory["token_stability"].keys())
        self.assertTrue(any("token_1" in key for key in token_keys) or 
                       any("token_5" in key for key in token_keys))
        
        # Check trajectory graph
        self.assertIsInstance(trajectory["trajectory_graph"], nx.DiGraph)
        
        # Test with PyTorch tensors
        trajectory_torch = self.analyzer.analyze_token_trajectory(
            layer_activations=self.torch_layer_activations,
            token_ids=self.token_ids,
            token_strings=self.token_strings
        )
        
        self.assertIn("token_stability", trajectory_torch)
    
    def test_analyze_representation_evolution(self):
        """Test analysis of representation evolution across layers."""
        # Analyze representation evolution
        evolution = self.analyzer.analyze_representation_evolution(
            layer_activations=self.layer_activations,
            class_labels=self.class_labels,
            layer_order=["layer0", "layer1", "layer2"]
        )
        
        # Check the result
        self.assertIn("representation_divergence", evolution)
        self.assertIn("class_separation", evolution)
        self.assertIn("sparsity", evolution)
        self.assertIn("compactness", evolution)
        
        # Check representation divergence
        self.assertEqual(len(evolution["representation_divergence"]), 2)  # 2 transitions
        
        # Check class separation
        self.assertEqual(len(evolution["class_separation"]), 3)  # 3 layers
        
        # Test with PyTorch tensors
        evolution_torch = self.analyzer.analyze_representation_evolution(
            layer_activations=self.torch_layer_activations,
            class_labels=self.class_labels
        )
        
        self.assertIn("representation_divergence", evolution_torch)
    
    def test_analyze_embedding_comparisons(self):
        """Test analysis of embedding space comparisons."""
        # Analyze embedding comparisons
        embeddings = self.analyzer.analyze_embedding_comparisons(
            layer_activations=self.layer_activations,
            pca_components=3,
            layer_order=["layer0", "layer1", "layer2"]
        )
        
        # Check the result
        self.assertIn("embeddings", embeddings)
        self.assertIn("embedding_similarities", embeddings)
        self.assertEqual(embeddings["pca_components"], 3)
        
        # Check embeddings
        for layer in ["layer0", "layer1", "layer2"]:
            self.assertIn(layer, embeddings["embeddings"])
            self.assertEqual(embeddings["embeddings"][layer].shape, (4, 3))  # [batch, pca_components]
        
        # Check embedding similarities
        self.assertEqual(len(embeddings["embedding_similarities"]), 3)  # 3 pairs
        
        # Test with PyTorch tensors
        embeddings_torch = self.analyzer.analyze_embedding_comparisons(
            layer_activations=self.torch_layer_activations,
            pca_components=2
        )
        
        self.assertEqual(embeddings_torch["pca_components"], 2)
    
    def test_analyze_transformer_cross_layer(self):
        """Test comprehensive cross-layer analysis."""
        # Perform comprehensive analysis
        result = self.analyzer.analyze_transformer_cross_layer(
            layer_activations=self.layer_activations,
            attention_data=self.attention_data,
            token_ids=self.token_ids,
            token_strings=self.token_strings,
            class_labels=self.class_labels,
            focus_tokens=["token_1", "token_5"],
            layer_order=["layer0", "layer1", "layer2"],
            config={
                "similarity_metric": "cosine",
                "dimensionality_reduction": True,
                "n_components": 32,
                "pca_components": 3
            }
        )
        
        # Check the result type
        self.assertIsInstance(result, CrossLayerTransformerAnalysisResult)
        
        # Check result attributes
        self.assertIsNotNone(result.layer_relationships)
        self.assertIsNotNone(result.attention_flow)
        self.assertIsNotNone(result.token_trajectory)
        self.assertIsNotNone(result.representation_evolution)
        self.assertIsNotNone(result.embedding_comparisons)
        self.assertIsNotNone(result.plot_data)
        
        # Check layers analyzed
        self.assertEqual(len(result.layers_analyzed), 3)
        self.assertEqual(set(result.layers_analyzed), {"layer0", "layer1", "layer2"})
        
        # Check heads analyzed
        self.assertEqual(len(result.attention_heads_analyzed), 3)
        for layer in ["layer0", "layer1", "layer2"]:
            self.assertIn(layer, result.attention_heads_analyzed)
            self.assertEqual(len(result.attention_heads_analyzed[layer]), 8)  # 8 heads per layer
        
        # Test result summary
        summary = result.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("mean_layer_similarity", summary)
        self.assertIn("mean_entropy_change", summary)
        self.assertIn("mean_token_stability", summary)
        self.assertIn("mean_semantic_shift", summary)
        self.assertIn("layers_analyzed", summary)
        self.assertIn("total_attention_heads", summary)
        
        # Test with PyTorch tensors
        result_torch = self.analyzer.analyze_transformer_cross_layer(
            layer_activations=self.torch_layer_activations,
            attention_data=self.torch_attention_data,
            token_ids=self.token_ids,
            token_strings=self.token_strings,
            class_labels=self.class_labels
        )
        
        self.assertIsInstance(result_torch, CrossLayerTransformerAnalysisResult)
    
    def test_cache_clearing(self):
        """Test that cache clearing works."""
        # Perform analysis to populate cache
        self.analyzer.analyze_layer_relationships(
            layer_activations=self.layer_activations,
            similarity_metric="cosine"
        )
        
        # Clear cache
        self.analyzer.clear_cache()
        
        # Check that internal cache is empty
        self.assertEqual(len(self.analyzer._cache), 0)


class TestCrossLayerTransformerAnalysisResult(unittest.TestCase):
    """Test the CrossLayerTransformerAnalysisResult class."""
    
    def test_result_summary(self):
        """Test the get_summary method."""
        # Create a sample result
        result = CrossLayerTransformerAnalysisResult(
            layer_relationships={
                "similarity_matrix": {
                    ("layer0", "layer1"): 0.8,
                    ("layer0", "layer2"): 0.6,
                    ("layer1", "layer2"): 0.7
                },
                "graph": nx.Graph()
            },
            attention_flow={
                "layer_entropy_differences": {
                    ("layer0", "layer1"): 0.1,
                    ("layer1", "layer2"): -0.2
                },
                "attention_pattern_consistency": {
                    ("layer0", "layer1"): 0.75,
                    ("layer1", "layer2"): 0.65
                }
            },
            token_trajectory={
                "token_stability": {
                    "token_1": {("layer0", "layer1"): 0.9, ("layer1", "layer2"): 0.8},
                    "token_2": {("layer0", "layer1"): 0.85, ("layer1", "layer2"): 0.75}
                },
                "focused_tokens": ["token_1", "token_2"]
            },
            representation_evolution={
                "semantic_shift": {
                    ("layer0", "layer1"): 0.2,
                    ("layer1", "layer2"): 0.3
                }
            },
            embedding_comparisons={},
            layers_analyzed=["layer0", "layer1", "layer2"],
            attention_heads_analyzed={
                "layer0": [0, 1, 2, 3],
                "layer1": [0, 1, 2, 3],
                "layer2": [0, 1, 2, 3]
            },
            plot_data={}
        )
        
        # Get summary
        summary = result.get_summary()
        
        # Check summary values
        self.assertIn("mean_layer_similarity", summary)
        self.assertAlmostEqual(summary["mean_layer_similarity"], 0.7, places=1)
        
        self.assertIn("mean_entropy_change", summary)
        self.assertAlmostEqual(summary["mean_entropy_change"], -0.05, places=2)
        
        self.assertIn("mean_attention_consistency", summary)
        self.assertAlmostEqual(summary["mean_attention_consistency"], 0.7, places=1)
        
        self.assertIn("mean_token_stability", summary)
        self.assertAlmostEqual(summary["mean_token_stability"], 0.825, places=3)
        
        self.assertIn("focused_tokens", summary)
        self.assertEqual(summary["focused_tokens"], ["token_1", "token_2"])
        
        self.assertIn("mean_semantic_shift", summary)
        self.assertAlmostEqual(summary["mean_semantic_shift"], 0.25, places=2)
        
        self.assertIn("layers_analyzed", summary)
        self.assertEqual(summary["layers_analyzed"], 3)
        
        self.assertIn("total_attention_heads", summary)
        self.assertEqual(summary["total_attention_heads"], 12)


if __name__ == "__main__":
    unittest.main()