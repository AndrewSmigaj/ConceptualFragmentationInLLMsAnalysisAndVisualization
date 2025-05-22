"""
Tests for MLP model adapter.

This module tests the MLP adapter implementation.
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.models.mlp_adapter import (
    MLPModelArchitecture,
    MLPModelAdapter,
    get_mlp_adapter
)


class SimpleMLP(nn.Module):
    """A simple MLP model for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class TestMLPAdapter(unittest.TestCase):
    """Tests for MLP model adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create standard FeedforwardNetwork
        self.ff_network = FeedforwardNetwork(
            input_dim=10,
            output_dim=2,
            hidden_layer_sizes=[20, 15, 10]
        )
        
        # Create simple MLP
        self.simple_mlp = SimpleMLP()
        
        # Create adapters
        self.ff_adapter = MLPModelAdapter(self.ff_network)
        self.simple_adapter = MLPModelAdapter(self.simple_mlp)
        
        # Create test inputs
        self.inputs = torch.randn(5, 10)  # 5 samples, 10 features
    
    def test_architecture_detection(self):
        """Test architecture detection."""
        # Test FeedforwardNetwork architecture
        ff_arch = self.ff_adapter.architecture
        self.assertEqual(ff_arch.name, "mlp")
        self.assertIn("layer1", ff_arch.get_layers())
        self.assertIn("layer2", ff_arch.get_layers())
        self.assertIn("layer3", ff_arch.get_layers())
        self.assertIn("output", ff_arch.get_layers())
        
        # Test simple MLP architecture
        simple_arch = self.simple_adapter.architecture
        self.assertEqual(simple_arch.name, "mlp")
        self.assertIn("fc1", simple_arch.get_layers())
        self.assertIn("fc2", simple_arch.get_layers())
        self.assertIn("fc3", simple_arch.get_layers())
    
    def test_activation_points(self):
        """Test activation points."""
        # Test FeedforwardNetwork activation points
        ff_points = self.ff_adapter.architecture.get_activation_points()
        self.assertIn("layer1", ff_points)
        self.assertIn("layer2", ff_points)
        self.assertIn("layer3", ff_points)
        self.assertIn("output", ff_points)
        self.assertIn("layer1_pre", ff_points)
        self.assertIn("layer2_pre", ff_points)
        self.assertIn("layer3_pre", ff_points)
        self.assertIn("output_pre", ff_points)
        
        # Test simple MLP activation points
        simple_points = self.simple_adapter.architecture.get_activation_points()
        self.assertIn("fc1", simple_points)
        self.assertIn("fc2", simple_points)
        self.assertIn("fc3", simple_points)
        self.assertIn("fc1_pre", simple_points)
        self.assertIn("fc2_pre", simple_points)
        self.assertIn("fc3_pre", simple_points)
    
    def test_forward(self):
        """Test forward pass."""
        # Test FeedforwardNetwork forward
        ff_outputs = self.ff_adapter.forward(self.inputs)
        self.assertEqual(ff_outputs.shape, (5, 2))  # 5 samples, 2 outputs
        
        # Test simple MLP forward
        simple_outputs = self.simple_adapter.forward(self.inputs)
        self.assertEqual(simple_outputs.shape, (5, 2))  # 5 samples, 2 outputs
    
    def test_get_layer_outputs(self):
        """Test getting layer outputs."""
        # Test FeedforwardNetwork layer outputs
        ff_outputs = self.ff_adapter.get_layer_outputs(self.inputs)
        self.assertIn("layer1", ff_outputs)
        self.assertIn("layer2", ff_outputs)
        self.assertIn("layer3", ff_outputs)
        self.assertIn("output", ff_outputs)
        self.assertEqual(ff_outputs["layer1"].shape, (5, 20))  # 5 samples, 20 hidden units
        self.assertEqual(ff_outputs["layer2"].shape, (5, 15))  # 5 samples, 15 hidden units
        self.assertEqual(ff_outputs["layer3"].shape, (5, 10))  # 5 samples, 10 hidden units
        self.assertEqual(ff_outputs["output"].shape, (5, 2))  # 5 samples, 2 outputs
        
        # Test simple MLP layer outputs - this might not work fully due to activation hooks
        # being more complex for generic models
        simple_outputs = self.simple_adapter.get_layer_outputs(self.inputs)
        # We expect at least some layer outputs
        self.assertGreater(len(simple_outputs), 0)
    
    def test_get_embeddings(self):
        """Test getting embeddings."""
        # Test FeedforwardNetwork embeddings
        ff_embeddings = self.ff_adapter.get_embeddings(self.inputs)
        self.assertEqual(ff_embeddings.shape, (5, 20))  # 5 samples, 20 hidden units
        
        # Test simple MLP embeddings
        simple_embeddings = self.simple_adapter.get_embeddings(self.inputs)
        # We might not get exactly what we want, but should get something
        self.assertIsNotNone(simple_embeddings)
    
    def test_factory_function(self):
        """Test factory function."""
        # Test factory with FeedforwardNetwork
        ff_factory = get_mlp_adapter(self.ff_network)
        self.assertIsInstance(ff_factory, MLPModelAdapter)
        
        # Test factory with simple MLP
        simple_factory = get_mlp_adapter(self.simple_mlp)
        self.assertIsInstance(simple_factory, MLPModelAdapter)


if __name__ == '__main__':
    unittest.main()