"""
Unit tests for ConfigManager.

This module tests the ConfigManager class, including singleton behavior,
loading, overriding, and validation.
"""

import unittest
import json
import yaml
import tempfile
import os
from pathlib import Path

from concept_fragmentation.config.config_manager import ConfigManager
from concept_fragmentation.config.config_classes import Config, DatasetConfig, ModelConfig


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a new ConfigManager instance
        self.cm = ConfigManager()
        
        # Create a sample config
        self.test_config = Config(
            random_seed=123,
            results_dir="test_results",
            datasets={
                "test": DatasetConfig(
                    path="data/test.csv",
                    test_size=0.2,
                    val_size=0.1
                )
            },
            models={
                "test_model": ModelConfig(
                    hidden_dims=[64, 32],
                    dropout=0.3
                )
            }
        )
    
    def test_singleton(self):
        """Test that ConfigManager behaves as a singleton."""
        cm1 = ConfigManager()
        cm2 = ConfigManager()
        
        # Both instances should be the same object
        self.assertIs(cm1, cm2)
        
        # Set config in one instance and check in the other
        cm1.set_config(self.test_config)
        self.assertEqual(cm2.get_config().random_seed, 123)
    
    def test_get_config(self):
        """Test getting the current configuration."""
        self.cm.set_config(self.test_config)
        config = self.cm.get_config()
        
        self.assertEqual(config.random_seed, 123)
        self.assertEqual(config.results_dir, "test_results")
        self.assertIn("test", config.datasets)
    
    def test_set_config(self):
        """Test setting a new configuration."""
        # Set the test config
        self.cm.set_config(self.test_config)
        
        # Check that it was set correctly
        config = self.cm.get_config()
        self.assertEqual(config.random_seed, 123)
        
        # Set a different config
        other_config = Config(random_seed=456)
        self.cm.set_config(other_config)
        
        # Check that it was updated
        config = self.cm.get_config()
        self.assertEqual(config.random_seed, 456)
    
    def test_load_from_dict(self):
        """Test loading configuration from a dictionary."""
        config_dict = {
            "random_seed": 789,
            "results_dir": "dict_results",
            "datasets": {
                "dict_test": {
                    "path": "data/dict_test.csv",
                    "test_size": 0.3
                }
            }
        }
        
        self.cm.load_from_dict(config_dict)
        
        # Check that it was loaded correctly
        config = self.cm.get_config()
        self.assertEqual(config.random_seed, 789)
        self.assertEqual(config.results_dir, "dict_results")
        self.assertIn("dict_test", config.datasets)
        self.assertEqual(config.datasets["dict_test"].path, "data/dict_test.csv")
    
    def test_load_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            json_file = temp.name
            
            # Write test config to file
            self.test_config.to_json(json_file)
        
        try:
            # Load from the file
            self.cm.load_from_file(json_file)
            
            # Check that it was loaded correctly
            config = self.cm.get_config()
            self.assertEqual(config.random_seed, 123)
            self.assertEqual(config.results_dir, "test_results")
            self.assertIn("test", config.datasets)
        finally:
            os.unlink(json_file)
    
    def test_override(self):
        """Test overriding specific configuration values."""
        # Set the test config
        self.cm.set_config(self.test_config)
        
        # Override some values
        self.cm.override({
            "random_seed": 999,
            "results_dir": "override_results",
            "datasets.test.test_size": 0.4
        })
        
        # Check that values were overridden
        config = self.cm.get_config()
        self.assertEqual(config.random_seed, 999)
        self.assertEqual(config.results_dir, "override_results")
        self.assertEqual(config.datasets["test"].test_size, 0.4)
        
        # Check that non-overridden values are preserved
        self.assertEqual(config.datasets["test"].val_size, 0.1)
        self.assertIn("test_model", config.models)
    
    def test_get_dataset_config(self):
        """Test getting configuration for a specific dataset."""
        # Set the test config
        self.cm.set_config(self.test_config)
        
        # Get dataset config
        dataset_config = self.cm.get_dataset_config("test")
        
        # Check that it's the right config
        self.assertIsNotNone(dataset_config)
        self.assertEqual(dataset_config.path, "data/test.csv")
        self.assertEqual(dataset_config.test_size, 0.2)
        
        # Check that non-existent dataset returns None
        self.assertIsNone(self.cm.get_dataset_config("non_existent"))
    
    def test_get_model_config(self):
        """Test getting configuration for a specific model."""
        # Set the test config
        self.cm.set_config(self.test_config)
        
        # Get model config
        model_config = self.cm.get_model_config("test_model")
        
        # Check that it's the right config
        self.assertIsNotNone(model_config)
        self.assertEqual(model_config.hidden_dims, [64, 32])
        self.assertEqual(model_config.dropout, 0.3)
        
        # Check that non-existent model returns None
        self.assertIsNone(self.cm.get_model_config("non_existent"))
    
    def test_get_experiment_config(self):
        """Test getting configuration for a specific experiment."""
        # Set the test config
        self.cm.set_config(self.test_config)
        
        # Get experiment config
        exp_config = self.cm.get_experiment_config("test", "exp001")
        
        # Check that it's based on the test config but with updated results_dir
        self.assertIsNotNone(exp_config)
        self.assertEqual(exp_config.random_seed, 123)
        self.assertEqual(exp_config.results_dir, os.path.join("test_results", "test", "exp001"))
        
        # Check that datasets were preserved
        self.assertIn("test", exp_config.datasets)
    
    def test_save_config(self):
        """Test saving configuration to a file."""
        # Set the test config
        self.cm.set_config(self.test_config)
        
        # JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            json_file = temp.name
        
        try:
            # Save to JSON
            self.cm.save_config(json_file, format='json')
            
            # Check that file exists and contains the right data
            self.assertTrue(os.path.exists(json_file))
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            self.assertEqual(data["random_seed"], 123)
            self.assertEqual(data["results_dir"], "test_results")
        finally:
            os.unlink(json_file)
        
        # YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
            yaml_file = temp.name
        
        try:
            # Save to YAML
            self.cm.save_config(yaml_file, format='yaml')
            
            # Check that file exists and contains the right data
            self.assertTrue(os.path.exists(yaml_file))
            
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
            
            self.assertEqual(data["random_seed"], 123)
            self.assertEqual(data["results_dir"], "test_results")
        finally:
            os.unlink(yaml_file)
    
    def test_validate(self):
        """Test validating the current configuration."""
        # Set a valid config
        self.cm.set_config(self.test_config)
        
        # Should validate successfully
        self.assertTrue(self.cm.validate())
        
        # Create invalid config by directly modifying the config object
        config = self.cm.get_config()
        config.models["invalid_model"] = ModelConfig(hidden_dims="not_a_list")
        
        # Should fail validation
        self.assertFalse(self.cm.validate())


if __name__ == "__main__":
    unittest.main()