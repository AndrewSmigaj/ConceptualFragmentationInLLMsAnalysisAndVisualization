"""
Tests for the transformer_dimensionality module.

This module contains tests for the TransformerDimensionalityReducer class
and related functionality for handling high-dimensional transformer spaces.
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import os

# Import the module to test
from concept_fragmentation.analysis.transformer_dimensionality import (
    TransformerDimensionalityReducer,
    DimensionalityReductionResult,
    DimensionalityReductionPipelineStage
)


class TestTransformerDimensionalityReducer(unittest.TestCase):
    """Test the TransformerDimensionalityReducer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Create reducer instance
        self.reducer = TransformerDimensionalityReducer(
            cache_dir=self.temp_dir,
            random_state=42,
            use_cache=True,
            verbose=False
        )
        
        # Generate test data with different dimensions and shapes
        
        # 1. Low-dimensional data
        self.low_dim_data = np.random.normal(size=(100, 20))
        
        # 2. High-dimensional data (simulates transformer hidden states)
        self.high_dim_data = np.random.normal(size=(100, 1024))
        
        # 3. Transformer sequence data with batch dimension
        self.sequence_data = np.random.normal(size=(8, 32, 768))  # [batch, seq_len, hidden_dim]
        
        # 4. PyTorch tensor data
        self.torch_data = torch.randn(16, 24, 512)  # [batch, seq_len, hidden_dim]
        
        # 5. Very high-dimensional data to test progressive reduction
        self.very_high_dim_data = np.random.normal(size=(50, 4096))
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_basic_reduction(self):
        """Test basic dimensionality reduction."""
        # Reduce dimensionality of high-dim data
        result = self.reducer.reduce_dimensionality(
            activations=self.high_dim_data,
            n_components=50,
            method="pca"
        )
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.reduced_activations.shape, (100, 50))
        self.assertEqual(result.method, "pca")
        self.assertEqual(result.original_dim, 1024)
        self.assertEqual(result.reduced_dim, 50)
        self.assertIsNotNone(result.explained_variance)
    
    def test_skip_low_dimensional_data(self):
        """Test that low-dimensional data isn't reduced further."""
        # Try to reduce already low-dim data
        result = self.reducer.reduce_dimensionality(
            activations=self.low_dim_data,
            n_components=50,
            method="pca"
        )
        
        # Check that no reduction was performed
        self.assertTrue(result.success)
        self.assertEqual(result.reduced_activations.shape, (100, 20))
        self.assertEqual(result.method, "identity")
        self.assertEqual(result.original_dim, 20)
        self.assertEqual(result.reduced_dim, 20)
    
    def test_reduce_sequence_data(self):
        """Test reduction of sequence data with batch dimension."""
        # Reduce dimensionality of sequence data
        result = self.reducer.reduce_dimensionality(
            activations=self.sequence_data,
            n_components=64,
            method="pca"
        )
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.reduced_activations.shape, (8, 32, 64))
        self.assertEqual(result.original_dim, 768)
        self.assertEqual(result.reduced_dim, 64)
    
    def test_reduce_torch_tensor(self):
        """Test reduction of PyTorch tensor data."""
        # Reduce dimensionality of torch tensor
        result = self.reducer.reduce_dimensionality(
            activations=self.torch_data,
            n_components=32,
            method="pca"
        )
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.reduced_activations.shape, (16, 24, 32))
        self.assertEqual(result.original_dim, 512)
        self.assertEqual(result.reduced_dim, 32)
    
    def test_random_projection(self):
        """Test random projection method."""
        # Reduce with random projection
        result = self.reducer.reduce_dimensionality(
            activations=self.high_dim_data,
            n_components=50,
            method="random_projection"
        )
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.reduced_activations.shape, (100, 50))
        self.assertTrue(
            result.method in [
                "gaussian_random_projection",
                "sparse_random_projection",
                "numpy_random_projection"
            ]
        )
    
    def test_caching(self):
        """Test that caching works."""
        # Perform reduction twice with same parameters
        result1 = self.reducer.reduce_dimensionality(
            activations=self.high_dim_data,
            n_components=50,
            method="pca"
        )
        
        # Should use cache for second reduction
        result2 = self.reducer.reduce_dimensionality(
            activations=self.high_dim_data,
            n_components=50,
            method="pca"
        )
        
        # Check that results are the same
        self.assertTrue(result1.success and result2.success)
        self.assertTrue(np.allclose(result1.reduced_activations, result2.reduced_activations))
        
        # Try with force_recompute
        result3 = self.reducer.reduce_dimensionality(
            activations=self.high_dim_data,
            n_components=50,
            method="pca",
            force_recompute=True
        )
        
        # Check that force_recompute still gives same results
        self.assertTrue(result3.success)
        self.assertTrue(np.allclose(result1.reduced_activations, result3.reduced_activations))
    
    def test_progressive_reduction(self):
        """Test progressive dimensionality reduction."""
        # Reduce very high-dimensional data progressively
        result = self.reducer.progressive_dimensionality_reduction(
            activations=self.very_high_dim_data,
            target_dim=10,
            initial_method="pca",
            secondary_method="random_projection"
        )
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.reduced_activations.shape, (50, 10))
        self.assertTrue("+" in result.method)  # Should be "pca+random_projection" or similar
        self.assertEqual(result.original_dim, 4096)
        self.assertEqual(result.reduced_dim, 10)


class TestDimensionalityReductionPipelineStage(unittest.TestCase):
    """Test the DimensionalityReductionPipelineStage class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Create pipeline stage
        self.pipeline_stage = DimensionalityReductionPipelineStage(
            n_components=32,
            method="auto",
            progressive=True,
            use_cache=True,
            cache_dir=self.temp_dir,
            random_state=42
        )
        
        # Create test data
        self.test_data = {
            "activations": {
                "layer1": np.random.normal(size=(10, 128)),
                "layer2": np.random.normal(size=(10, 256)),
                "layer3": np.random.normal(size=(10, 512)),
                "metadata": {"some": "metadata"}  # Non-tensor data
            }
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_stage_basic(self):
        """Test basic functionality of pipeline stage."""
        # Process data through pipeline stage
        result = self.pipeline_stage.process(self.test_data)
        
        # Check the result
        self.assertIn("reduced_activations", result)
        self.assertIn("reduction_metadata", result)
        
        # Check that all layers were processed
        self.assertEqual(len(result["reduced_activations"]), 3)  # Excluding metadata
        
        # Check shapes of reduced activations
        self.assertEqual(result["reduced_activations"]["layer1"].shape, (10, 32))
        self.assertEqual(result["reduced_activations"]["layer2"].shape, (10, 32))
        self.assertEqual(result["reduced_activations"]["layer3"].shape, (10, 32))
        
        # Check metadata
        for layer in ["layer1", "layer2", "layer3"]:
            self.assertIn(layer, result["reduction_metadata"])
            self.assertEqual(result["reduction_metadata"][layer]["reduced_dim"], 32)
            self.assertTrue(result["reduction_metadata"][layer]["success"])
    
    def test_pipeline_stage_layer_filtering(self):
        """Test layer filtering in pipeline stage."""
        # Create pipeline stage with filter
        filtered_stage = DimensionalityReductionPipelineStage(
            n_components=32,
            method="auto",
            filter_layers="layer[12]",  # Only process layer1 and layer2
            progressive=True,
            use_cache=True,
            cache_dir=self.temp_dir,
            random_state=42
        )
        
        # Process data through filtered pipeline stage
        result = filtered_stage.process(self.test_data)
        
        # Check that only filtered layers were processed
        self.assertEqual(len(result["reduced_activations"]), 2)
        self.assertIn("layer1", result["reduced_activations"])
        self.assertIn("layer2", result["reduced_activations"])
        self.assertNotIn("layer3", result["reduced_activations"])


if __name__ == "__main__":
    unittest.main()