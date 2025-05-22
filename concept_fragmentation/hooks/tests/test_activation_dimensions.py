"""
Tests for dimension handling in activation hooks.

This module tests the dimension validation and handling functionality in the 
activation_hooks module, particularly for scenarios with dimension mismatches
between train and test data.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from concept_fragmentation.hooks.activation_hooks import (
    ActivationHook,
    get_activation_hooks,
    track_train_test_dimensions,
    set_dimension_logging,
    set_dimension_mismatch_strategy,
    get_dimension_mismatch_strategy
)


class SimpleModel(nn.Module):
    """
    A simple model for testing activation hooks with different layer dimensions.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(32, output_dim)
    
    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)
        return x


class TestActivationDimensions(unittest.TestCase):
    """
    Test cases for dimension handling in activation hooks.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        # Create models with different input dimensions
        self.model_small = SimpleModel(2, 1)
        self.model_large = SimpleModel(64, 1)
        
        # Create test data with different dimensions
        self.data_small = torch.rand(10, 2)
        self.data_large = torch.rand(10, 64)
        
        # Enable dimension logging for tests
        set_dimension_logging(True)

    def test_dimension_mismatch_strategy_setting(self):
        """
        Test setting and getting dimension mismatch strategy.
        """
        original_strategy = get_dimension_mismatch_strategy()
        
        # Test valid strategies
        for strategy in ['warn', 'error', 'truncate', 'pad']:
            set_dimension_mismatch_strategy(strategy)
            self.assertEqual(get_dimension_mismatch_strategy(), strategy)
        
        # Test invalid strategy
        with self.assertRaises(ValueError):
            set_dimension_mismatch_strategy('invalid_strategy')
        
        # Restore original strategy
        set_dimension_mismatch_strategy(original_strategy)
    
    def test_track_dimensions_warn_strategy(self):
        """
        Test track_train_test_dimensions with 'warn' strategy (default).
        """
        # Set strategy to 'warn'
        set_dimension_mismatch_strategy('warn')
        
        # Create train and test activations with dimension mismatch
        train_activations = {
            'layer1': torch.rand(10, 64),
            'layer2': torch.rand(10, 32),
            'layer3': torch.rand(10, 1)
        }
        
        test_activations = {
            'layer1': torch.rand(5, 32),  # Different feature dimension
            'layer2': torch.rand(5, 32),  # Same feature dimension
            'layer3': torch.rand(5, 1)    # Same feature dimension
        }
        
        # Track dimensions
        results = track_train_test_dimensions(
            train_activations, 
            test_activations,
            phase="testing",
            dataset="test_dataset"
        )
        
        # Check results
        self.assertIn('layers', results)
        self.assertIn('layer1', results['layers'])
        self.assertFalse(results['layers']['layer1']['shape_match'])
        self.assertTrue(results['layers']['layer2']['shape_match'])
        
        # With 'warn' strategy, no adjustments should be made
        self.assertNotIn('adjusted_train', results['layers']['layer1'])
        self.assertNotIn('adjusted_test', results['layers']['layer1'])
    
    def test_track_dimensions_truncate_strategy(self):
        """
        Test track_train_test_dimensions with 'truncate' strategy.
        """
        # Set strategy to 'truncate'
        set_dimension_mismatch_strategy('truncate')
        
        # Create train and test activations with dimension mismatch
        train_activations = {
            'layer1': torch.rand(10, 64),  # Larger feature dimension
            'layer2': torch.rand(10, 32)
        }
        
        test_activations = {
            'layer1': torch.rand(5, 32),  # Smaller feature dimension
            'layer2': torch.rand(5, 32)
        }
        
        # Track dimensions with truncation
        results = track_train_test_dimensions(
            train_activations, 
            test_activations,
            phase="testing",
            dataset="test_dataset"
        )
        
        # Check results
        self.assertIn('adjusted_train', results['layers']['layer1'])
        self.assertEqual(results['layers']['layer1']['adjusted_train'].shape[1], 32)
        self.assertNotIn('adjusted_test', results['layers']['layer1'])  # No need to adjust test
    
    def test_track_dimensions_pad_strategy(self):
        """
        Test track_train_test_dimensions with 'pad' strategy.
        """
        # Set strategy to 'pad'
        set_dimension_mismatch_strategy('pad')
        
        # Create train and test activations with dimension mismatch
        train_activations = {
            'layer1': torch.rand(10, 32),  # Smaller feature dimension
            'layer2': torch.rand(10, 32)
        }
        
        test_activations = {
            'layer1': torch.rand(5, 64),  # Larger feature dimension
            'layer2': torch.rand(5, 32)
        }
        
        # Track dimensions with padding
        results = track_train_test_dimensions(
            train_activations, 
            test_activations,
            phase="testing",
            dataset="test_dataset"
        )
        
        # Check results
        self.assertIn('adjusted_train', results['layers']['layer1'])
        self.assertEqual(results['layers']['layer1']['adjusted_train'].shape[1], 64)
        self.assertNotIn('adjusted_test', results['layers']['layer1'])  # No need to adjust test
    
    def test_track_dimensions_error_strategy(self):
        """
        Test track_train_test_dimensions with 'error' strategy.
        """
        # Set strategy to 'error'
        set_dimension_mismatch_strategy('error')
        
        # Create train and test activations with dimension mismatch
        train_activations = {
            'layer1': torch.rand(10, 64),
            'layer2': torch.rand(10, 32)
        }
        
        test_activations = {
            'layer1': torch.rand(5, 32),  # Different feature dimension
            'layer2': torch.rand(5, 32)
        }
        
        # Track dimensions should raise an error
        with self.assertRaises(ValueError):
            track_train_test_dimensions(
                train_activations, 
                test_activations,
                phase="testing",
                dataset="test_dataset"
            )
    
    def test_activation_hook_dimension_logging(self):
        """
        Test activation hook with dimension logging enabled.
        """
        # Enable dimension logging
        set_dimension_logging(True)
        
        # Create activation hook
        hook = ActivationHook(self.model_small)
        hook.register_hooks()
        
        # Forward pass
        output = self.model_small(self.data_small)
        
        # Check activations
        activations = hook.layer_activations
        self.assertIn('layer1', activations)
        self.assertIn('layer2', activations)
        self.assertIn('layer3', activations)
        
        # Convert to numpy and check dimensions
        numpy_activations = hook.numpy_activations()
        self.assertEqual(numpy_activations['layer1'].shape, (10, 64))
        self.assertEqual(numpy_activations['layer2'].shape, (10, 32))
        self.assertEqual(numpy_activations['layer3'].shape, (10, 1))
        
        # Clean up
        hook.remove_hooks()


if __name__ == '__main__':
    unittest.main()