import unittest
import torch
import torch.nn as nn
import numpy as np
from ..hooks.activation_hooks import (
    ActivationHook, 
    get_activation_hooks,
    capture_activations,
    get_neuron_importance
)

# Create a simple test model
class SimpleTestModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=2):
        super(SimpleTestModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class TestActivationHooks(unittest.TestCase):
    """Test cases for the activation hooks module."""
    
    def setUp(self):
        """Set up test model and data."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a simple model
        self.model = SimpleTestModel()
        
        # Create test input
        self.test_input = torch.randn(32, 5)  # Batch of 32, 5 features
        
        # Get expected output shape
        with torch.no_grad():
            self.expected_output_shape = self.model(self.test_input).shape
        
    def test_activation_hook(self):
        """Test the ActivationHook class."""
        hook = ActivationHook(self.model)
        
        # Store should be empty initially
        self.assertEqual(len(hook.activations), 0)
        
        # Register hooks and run forward pass
        hook.register_hooks()
        with torch.no_grad():
            _ = self.model(self.test_input)
        
        # Check that activations were captured
        self.assertGreater(len(hook.activations), 0)
        
        # Check layer access
        for name in hook.activations.keys():
            self.assertIsNotNone(hook.activations[name])
        
        # Clear and check
        hook.clear_activations()
        self.assertEqual(len(hook.activations), 0)
        
        # Clean up
        hook.remove_hooks()
        
    def test_register_hooks(self):
        """Test registering hooks on model layers."""
        # Register hooks on all layers
        hook = get_activation_hooks(self.model)
        
        # Run a forward pass
        with torch.no_grad():
            _ = self.model(self.test_input)
        
        # Check that activations were captured
        self.assertGreater(len(hook.activations), 0)
        
        # Check that we have the right layers
        layer_names = set(dict(self.model.named_modules()).keys())
        for name in layer_names:
            if name != '':  # Skip root module
                self.assertIn(name, hook.activations)
        
        # Check activation shapes
        for name, activation in hook.activations.items():
            self.assertEqual(activation.shape[0], self.test_input.shape[0])  # Batch dimension
        
        # Clean up
        hook.remove_hooks()
        
    def test_register_hooks_specific_layers(self):
        """Test registering hooks on specific layers."""
        # Register hooks only on fc1 and fc3
        layer_names = ['fc1', 'fc3']
        hook = get_activation_hooks(self.model, layer_names=layer_names)
        
        # Run a forward pass
        with torch.no_grad():
            _ = self.model(self.test_input)
        
        # Check that only specified layers were captured
        self.assertEqual(len(hook.activations), 2)
        self.assertIn('fc1', hook.activations)
        self.assertIn('fc3', hook.activations)
        self.assertNotIn('fc2', hook.activations)
        
        # Clean up
        hook.remove_hooks()
        
    def test_context_manager(self):
        """Test the capture_activations context manager."""
        # Use context manager
        with capture_activations(self.model) as activations:
            with torch.no_grad():
                _ = self.model(self.test_input)
            
            # Check that activations were captured
            self.assertGreater(len(activations), 0)
            
        # After context, no hooks should be active
        # Create a new hook to see if activations are cleared
        hook = ActivationHook(self.model)
        hook.register_hooks()
        with torch.no_grad():
            _ = self.model(self.test_input)
        
        # Now test the behavior after removing hooks
        hook.remove_hooks()
        hook.clear_activations()
        with torch.no_grad():
            _ = self.model(self.test_input)
        
        # Should be empty because hooks were removed
        self.assertEqual(len(hook.activations), 0)
        
    def test_get_neuron_importance(self):
        """Test neuron importance calculation."""
        # Capture activations
        with capture_activations(self.model) as activations:
            with torch.no_grad():
                _ = self.model(self.test_input)
            
            # Calculate importance with different methods
            importance_mean_abs = get_neuron_importance(activations, method='mean_abs')
            importance_max_abs = get_neuron_importance(activations, method='max_abs')
            importance_variance = get_neuron_importance(activations, method='variance')
            
            # Check that results have the right format
            for layer_name, scores in importance_mean_abs.items():
                # Get corresponding activation
                activation = activations[layer_name]
                
                # If activation is 2D (batch, features), scores should have length=features
                if len(activation.shape) == 2:
                    self.assertEqual(len(scores), activation.shape[1])
                
            # Check invalid method raises error
            with self.assertRaises(ValueError):
                get_neuron_importance(activations, method='invalid_method')

    def test_model_output_unchanged(self):
        """Test that adding hooks doesn't change model output."""
        # Get output without hooks
        with torch.no_grad():
            output_no_hooks = self.model(self.test_input)
        
        # Get output with hooks
        hook = get_activation_hooks(self.model)
        with torch.no_grad():
            output_with_hooks = self.model(self.test_input)
        
        # Cleanup
        hook.remove_hooks()
        
        # Outputs should be identical
        torch.testing.assert_close(output_no_hooks, output_with_hooks)
        
    def test_cleanup_after_exception(self):
        """Test that hooks are removed even if an exception occurs."""
        try:
            with capture_activations(self.model) as activations:
                # Cause an exception
                raise RuntimeError("Test exception")
        except RuntimeError:
            pass
        
        # Create a new hook to see if activations are cleared
        hook = ActivationHook(self.model)
        hook.register_hooks()
        with torch.no_grad():
            _ = self.model(self.test_input)
        
        # Clean up
        hook.remove_hooks()

if __name__ == '__main__':
    unittest.main()
