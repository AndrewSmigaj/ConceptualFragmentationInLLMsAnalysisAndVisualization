"""
Tests for model interfaces module.

This module tests the model interfaces defined in model_interfaces.py.
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional

from concept_fragmentation.models.model_interfaces import (
    ModelInterface, 
    SequenceModelInterface,
    AttentionModelInterface,
    ModelArchitecture,
    ActivationPoint,
    BaseModelAdapter,
    BaseModelArchitecture,
    LayerInfo,
    ActivationPointImpl
)


class SimpleTestModel(nn.Module):
    """A simple model for testing the model interfaces."""
    
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of the sequence
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TestModelArchitecture(BaseModelArchitecture):
    """Test implementation of ModelArchitecture."""
    
    def _initialize(self):
        """Initialize layers and activation points."""
        # Add layers
        for name, module in self.model.named_modules():
            if name == '':
                continue
                
            if isinstance(module, nn.Embedding):
                layer_type = 'embedding'
            elif isinstance(module, nn.LSTM):
                layer_type = 'lstm'
            elif isinstance(module, nn.Linear):
                layer_type = 'linear'
            else:
                layer_type = 'other'
                
            self._layers[name] = LayerInfo(
                name=name,
                module=module,
                type=layer_type
            )
            
        # Add activation points
        for name, layer in self._layers.items():
            self._activation_points[f"{name}_output"] = ActivationPointImpl(
                name=f"{name}_output",
                layer_name=name,
                module=layer.module,
                type='output'
            )


class TestModelAdapter(BaseModelAdapter):
    """Test implementation of ModelInterface."""
    
    def __init__(self, model: nn.Module):
        """Initialize with model."""
        architecture = TestModelArchitecture(model, "test_architecture")
        super().__init__(model, architecture)
        
    def get_layer_outputs(
        self, 
        inputs: Any, 
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Get outputs from specific layers."""
        outputs = {}
        
        # Get all layer names if none specified
        if layer_names is None:
            layer_names = list(self.architecture.get_layers().keys())
            
        # Create hooks for each layer
        hooks = {}
        
        def hook_fn(layer_name):
            def hook(module, input, output):
                outputs[layer_name] = output
            return hook
            
        # Register hooks
        for layer_name in layer_names:
            if layer_name in self.architecture.get_layers():
                layer = self.architecture.get_layers()[layer_name]
                hooks[layer_name] = layer.register_forward_hook(hook_fn(layer_name))
                
        # Forward pass
        _ = self.model(inputs)
        
        # Remove hooks
        for hook in hooks.values():
            hook.remove()
            
        return outputs
    
    def get_embeddings(self, inputs: Any) -> torch.Tensor:
        """Get embedding layer outputs."""
        outputs = self.get_layer_outputs(inputs, ['embedding'])
        return outputs.get('embedding', torch.tensor([]))


class TestModelInterfaces(unittest.TestCase):
    """Tests for model interfaces."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleTestModel()
        self.adapter = TestModelAdapter(self.model)
        self.inputs = torch.randint(0, 100, (2, 10))  # Batch size 2, sequence length 10
        
    def test_architecture_initialization(self):
        """Test architecture initialization."""
        architecture = self.adapter.architecture
        self.assertEqual(architecture.name, "test_architecture")
        self.assertIn("embedding", architecture.get_layers())
        self.assertIn("lstm", architecture.get_layers())
        self.assertIn("fc1", architecture.get_layers())
        self.assertIn("fc2", architecture.get_layers())
        
        # Test layer types
        self.assertTrue(architecture.is_layer_type("embedding", "embedding"))
        self.assertTrue(architecture.is_layer_type("lstm", "lstm"))
        self.assertTrue(architecture.is_layer_type("fc1", "linear"))
        self.assertTrue(architecture.is_layer_type("fc2", "linear"))
        
        # Test activation points
        activation_points = architecture.get_activation_points()
        self.assertIn("embedding_output", activation_points)
        self.assertIn("lstm_output", activation_points)
        self.assertIn("fc1_output", activation_points)
        self.assertIn("fc2_output", activation_points)
        
    def test_model_adapter(self):
        """Test model adapter."""
        # Test forward
        outputs = self.adapter.forward(self.inputs)
        self.assertEqual(outputs.shape, (2, 10))  # Batch size 2, output size 10
        
        # Test get_layer_outputs
        layer_outputs = self.adapter.get_layer_outputs(self.inputs)
        self.assertIn("fc1", layer_outputs)
        self.assertEqual(layer_outputs["fc1"].shape, (2, 32))  # Batch size 2, hidden size 32
        
        # Test get_embeddings
        embeddings = self.adapter.get_embeddings(self.inputs)
        self.assertEqual(embeddings.shape, (2, 10, 32))  # Batch size 2, sequence length 10, embedding dim 32


if __name__ == '__main__':
    unittest.main()