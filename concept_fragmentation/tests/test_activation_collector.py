"""
Tests for the enhanced ActivationCollector implementation.

This module contains tests for the streaming activation collection
functionality, focusing on memory efficiency and consistent outputs.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

from concept_fragmentation.activation import (
    ActivationCollector, ActivationFormat, 
    collect_activations, CollectionConfig
)


class SimpleModel(nn.Module):
    """Simple model for testing activation collection."""
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(20, 15)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(15, 5)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        return x


class TestActivationCollector(unittest.TestCase):
    """Test cases for ActivationCollector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.inputs = torch.randn(32, 10)  # Batch of 32 samples
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        config = CollectionConfig(device='cpu')
        collector = ActivationCollector(config)
        self.assertEqual(collector.config.device, 'cpu')
        self.assertEqual(collector.config.format, ActivationFormat.NUMPY)
    
    def test_basic_collection(self):
        """Test basic activation collection."""
        collector = ActivationCollector()
        collector.register_model(self.model)
        
        activations = collector.collect(self.model, self.inputs)
        
        # Check that we get activations for each layer
        self.assertIn('layer1', activations)
        self.assertIn('relu1', activations)
        self.assertIn('layer2', activations)
        self.assertIn('relu2', activations)
        self.assertIn('layer3', activations)
        
        # Check activation shapes
        self.assertEqual(activations['layer1'].shape, (32, 20))
        self.assertEqual(activations['layer2'].shape, (32, 15))
        self.assertEqual(activations['layer3'].shape, (32, 5))
    
    def test_streaming_collection(self):
        """Test streaming activation collection."""
        collector = ActivationCollector()
        collector.register_model(self.model)
        
        # Split inputs into smaller batches
        batches = torch.split(self.inputs, 8)  # 4 batches of 8 samples
        
        # Collect streaming activations
        batch_generator = collector.collect(
            self.model, 
            batches,  # Passing an iterable of batches
            streaming=True
        )
        
        # Process each batch
        batch_count = 0
        all_activations = {}
        
        for batch in batch_generator:
            batch_count += 1
            batch_activations = batch['activations']
            
            # Check that we get activations for each layer
            self.assertIn('layer1', batch_activations)
            self.assertIn('layer2', batch_activations)
            self.assertIn('layer3', batch_activations)
            
            # Accumulate activations
            for layer, activation in batch_activations.items():
                if layer not in all_activations:
                    all_activations[layer] = []
                all_activations[layer].append(activation)
        
        # Check that we got 4 batches
        self.assertEqual(batch_count, 4)
        
        # Concatenate activations from all batches
        for layer, activations in all_activations.items():
            all_activations[layer] = np.concatenate(activations, axis=0)
        
        # Check shapes
        self.assertEqual(all_activations['layer1'].shape, (32, 20))
        self.assertEqual(all_activations['layer2'].shape, (32, 15))
        self.assertEqual(all_activations['layer3'].shape, (32, 5))
    
    def test_collect_and_store(self):
        """Test collecting and storing activations."""
        collector = ActivationCollector()
        
        # Save to a file
        output_path = os.path.join(self.temp_dir, 'activations.pkl')
        saved_path = collector.collect_and_store(
            self.model,
            self.inputs,
            output_path=output_path
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(saved_path))
        
        # Load the activations
        loaded = ActivationCollector.load_activations(saved_path)
        
        # Check that we got all layers
        self.assertIn('activations', loaded)
        self.assertIn('layer1', loaded['activations'])
        self.assertIn('layer2', loaded['activations'])
        self.assertIn('layer3', loaded['activations'])
        
        # Check shapes
        self.assertEqual(loaded['activations']['layer1'].shape, (32, 20))
        self.assertEqual(loaded['activations']['layer2'].shape, (32, 15))
        self.assertEqual(loaded['activations']['layer3'].shape, (32, 5))
    
    def test_streaming_storage(self):
        """Test collecting and storing streaming activations."""
        collector = ActivationCollector()
        
        # Split inputs into smaller batches
        batches = torch.split(self.inputs, 8)  # 4 batches of 8 samples
        
        # Save to a file
        output_path = os.path.join(self.temp_dir, 'streaming_activations.pkl')
        saved_path = collector.collect_and_store(
            self.model,
            batches,
            output_path=output_path,
            streaming=True
        )
        
        # Check that the file was created
        self.assertTrue(os.path.exists(saved_path))
        
        # Load the activations
        loaded = ActivationCollector.load_activations(saved_path, concat_batches=True)
        
        # Check that we got all layers
        self.assertIn('activations', loaded)
        self.assertIn('layer1', loaded['activations'])
        self.assertIn('layer2', loaded['activations'])
        self.assertIn('layer3', loaded['activations'])
        
        # Check shapes
        self.assertEqual(loaded['activations']['layer1'].shape, (32, 20))
        self.assertEqual(loaded['activations']['layer2'].shape, (32, 15))
        self.assertEqual(loaded['activations']['layer3'].shape, (32, 5))
    
    def test_collect_train_test(self):
        """Test collecting both training and test activations."""
        collector = ActivationCollector()
        
        # Create train and test data
        train_data = torch.randn(24, 10)
        test_data = torch.randn(8, 10)
        
        # Save to directory
        results = collector.collect_train_test(
            self.model,
            train_data,
            test_data,
            output_dir=self.temp_dir
        )
        
        # Check that files were created
        self.assertIn('train', results)
        self.assertIn('test', results)
        self.assertTrue(os.path.exists(results['train']))
        self.assertTrue(os.path.exists(results['test']))
        
        # Load the activations
        train_loaded = ActivationCollector.load_activations(results['train'])
        test_loaded = ActivationCollector.load_activations(results['test'])
        
        # Check shapes
        self.assertEqual(train_loaded['activations']['layer1'].shape, (24, 20))
        self.assertEqual(test_loaded['activations']['layer1'].shape, (8, 20))
    
    def test_context_manager_api(self):
        """Test the context manager API for activation collection."""
        # Test with default parameters
        with collect_activations(self.model) as get_activations:
            # Forward pass
            _ = self.model(self.inputs)
            
            # Get activations
            activations = get_activations()
            
            # Check that we got all layers
            self.assertIn('layer1', activations)
            self.assertIn('layer2', activations)
            self.assertIn('layer3', activations)
            
            # Check that activations are tensors
            self.assertIsInstance(activations['layer1'], torch.Tensor)
        
        # Test with numpy conversion
        with collect_activations(self.model, to_numpy=True) as get_activations:
            # Forward pass
            _ = self.model(self.inputs)
            
            # Get activations
            activations = get_activations()
            
            # Check that activations are numpy arrays
            self.assertIsInstance(activations['layer1'], np.ndarray)
        
        # Test with specific layers
        with collect_activations(self.model, layer_names=['layer1', 'layer3']) as get_activations:
            # Forward pass
            _ = self.model(self.inputs)
            
            # Get activations
            activations = get_activations()
            
            # Check that we only got specified layers
            self.assertIn('layer1', activations)
            self.assertNotIn('layer2', activations)
            self.assertIn('layer3', activations)


if __name__ == '__main__':
    unittest.main()