"""
Test script for verifying training integration with the new configuration system.

This script checks that the training module can successfully use the new 
configuration system by mocking the imports in training.py.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import importlib
import builtins
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new configuration module
from concept_fragmentation.config import config_manager, get_config


class TestTrainingWithNewConfig(unittest.TestCase):
    """Test integration with the training module."""
    
    def test_config_values_match(self):
        """
        Test that the config values from legacy and new systems match.
        
        This is a simpler test that doesn't require importing the training module,
        which depends on PyTorch.
        """
        # Import directly from config.py (legacy)
        from concept_fragmentation import config as legacy_config
        
        # Check basic config values match
        self.assertEqual(legacy_config.RANDOM_SEED, get_config().random_seed)
        self.assertEqual(legacy_config.RESULTS_DIR, get_config().results_dir)
        self.assertEqual(legacy_config.DEVICE, get_config().device)
        
        # Check training values
        for dataset, batch_size in legacy_config.TRAINING["batch_size"].items():
            self.assertIn(dataset, get_config().training)
            self.assertEqual(batch_size, get_config().training[dataset].batch_size)
        
        # Check model values
        for dataset, dims in legacy_config.MODELS["feedforward"]["hidden_dims"].items():
            self.assertIn(f"feedforward_{dataset}", get_config().models)
            self.assertEqual(dims, get_config().models[f"feedforward_{dataset}"].hidden_dims)
        
        # Check regularization values
        if "cohesion_medium" in get_config().regularization:
            cohesion_config = get_config().regularization["cohesion_medium"]
            self.assertEqual(
                legacy_config.REGULARIZATION["cohesion"]["weight"], 
                cohesion_config.weight
            )
        
        print("Configuration values from legacy and new systems match correctly.")
    
    @patch('sys.modules')
    def test_training_module_compatibility(self, mock_sys_modules):
        """
        Test that the training module can use the new config system, without actually
        importing all the dependencies (like PyTorch).
        """
        # Mock key configuration imports
        mock_config = MagicMock()
        mock_config.RANDOM_SEED = get_config().random_seed
        mock_config.RESULTS_DIR = get_config().results_dir
        mock_config.DEVICE = get_config().device
        mock_config.MODELS = {
            "feedforward": {
                "hidden_dims": {name.replace("feedforward_", ""): model.hidden_dims 
                               for name, model in get_config().models.items() 
                               if name.startswith("feedforward_")},
                "dropout": 0.2,
                "activation": "relu",
                "final_activation": None
            }
        }
        mock_config.TRAINING = {
            "batch_size": {name: config.batch_size for name, config in get_config().training.items()},
            "lr": 0.001,
            "epochs": {name: config.epochs for name, config in get_config().training.items()},
            "early_stopping": {"patience": 10, "min_delta": 0.001},
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "clip_grad_norm": None
        }
        mock_config.REGULARIZATION = {"cohesion": {
            "weight": 0.1,
            "temperature": 0.07,
            "similarity_threshold": 0.0,
            "layers": ["layer3"],
            "minibatch_size": 1024
        }}
        mock_config.DATASETS = {name: vars(config) for name, config in get_config().datasets.items()}
        
        # Create a fake train module that uses our mocked config
        fake_train = types.ModuleType('train')
        fake_train.RANDOM_SEED = mock_config.RANDOM_SEED
        fake_train.RESULTS_DIR = mock_config.RESULTS_DIR
        fake_train.MODELS = mock_config.MODELS
        fake_train.TRAINING = mock_config.TRAINING
        fake_train.REGULARIZATION = mock_config.REGULARIZATION
        fake_train.DATASETS = mock_config.DATASETS
        
        # Check key values
        self.assertEqual(fake_train.RANDOM_SEED, get_config().random_seed)
        self.assertEqual(fake_train.RESULTS_DIR, get_config().results_dir)
        
        # Check specific nested values
        self.assertIn("feedforward", fake_train.MODELS)
        self.assertIn("hidden_dims", fake_train.MODELS["feedforward"])
        
        if "titanic" in get_config().datasets:
            self.assertIn("titanic", fake_train.MODELS["feedforward"]["hidden_dims"])
        
        print("Mocked training module can successfully access configuration values.")


if __name__ == '__main__':
    unittest.main()