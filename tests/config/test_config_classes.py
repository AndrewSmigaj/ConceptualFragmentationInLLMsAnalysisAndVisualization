"""
Unit tests for configuration classes.

This module tests the configuration classes in the config module,
including serialization, deserialization, and validation.
"""

import unittest
import json
import yaml
import tempfile
import os
import copy
from pathlib import Path

from concept_fragmentation.config.config_classes import (
    Config,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    RegularizationConfig,
    MetricsConfig
)


class TestDatasetConfig(unittest.TestCase):
    """Test DatasetConfig class."""
    
    def test_init(self):
        """Test initialization with required parameters."""
        config = DatasetConfig(path="data/test.csv")
        self.assertEqual(config.path, "data/test.csv")
        self.assertEqual(config.test_size, 0.2)  # Default value
    
    def test_post_init(self):
        """Test post-initialization processing."""
        config = DatasetConfig(path="data/test.csv")
        self.assertIsNotNone(config.impute_strategy)
        self.assertIn("categorical", config.impute_strategy)
        self.assertIn("numerical", config.impute_strategy)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig class."""
    
    def test_init(self):
        """Test initialization with required parameters."""
        config = ModelConfig(hidden_dims=[64, 32])
        self.assertEqual(config.hidden_dims, [64, 32])
        self.assertEqual(config.dropout, 0.2)  # Default value
    
    def test_validation(self):
        """Test validation of parameters."""
        # Invalid hidden_dims
        with self.assertRaises(ValueError):
            ModelConfig(hidden_dims="invalid")
        
        # Invalid activation
        with self.assertRaises(ValueError):
            ModelConfig(hidden_dims=[64], activation="invalid")
        
        # Invalid final_activation
        with self.assertRaises(ValueError):
            ModelConfig(hidden_dims=[64], final_activation="invalid")


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig class."""
    
    def test_init(self):
        """Test initialization with required parameters."""
        config = TrainingConfig(batch_size=32)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.lr, 0.001)  # Default value
    
    def test_validation(self):
        """Test validation of parameters."""
        # Invalid batch_size
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=0)
        
        # Invalid learning rate
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=32, lr=0)
        
        # Invalid optimizer
        with self.assertRaises(ValueError):
            TrainingConfig(batch_size=32, optimizer="invalid")


class TestRegularizationConfig(unittest.TestCase):
    """Test RegularizationConfig class."""
    
    def test_init(self):
        """Test initialization with required parameters."""
        config = RegularizationConfig()
        self.assertEqual(config.weight, 0.0)  # Default value
    
    def test_validation(self):
        """Test validation of parameters."""
        # Invalid weight
        with self.assertRaises(ValueError):
            RegularizationConfig(weight=-1)
        
        # Invalid temperature
        with self.assertRaises(ValueError):
            RegularizationConfig(temperature=0)


class TestConfig(unittest.TestCase):
    """Test Config class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config(
            random_seed=123,
            results_dir="test_results",
            datasets={
                "test": DatasetConfig(
                    path="data/test.csv",
                    test_size=0.2,
                    val_size=0.1,
                    categorical_features=["A", "B"],
                    numerical_features=["X", "Y"],
                    target="target"
                )
            },
            models={
                "test_model": ModelConfig(
                    hidden_dims=[64, 32],
                    dropout=0.3,
                    activation="relu"
                )
            },
            training={
                "test": TrainingConfig(
                    batch_size=64,
                    lr=0.01,
                    epochs=100
                )
            }
        )
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config_dict = self.config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["random_seed"], 123)
        self.assertEqual(config_dict["results_dir"], "test_results")
        self.assertIn("datasets", config_dict)
        self.assertIn("test", config_dict["datasets"])
    
    def test_to_json(self):
        """Test conversion to JSON."""
        # Test without file
        json_str = self.config.to_json()
        self.assertIsInstance(json_str, str)
        
        # Test with file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            json_file = temp.name
        
        try:
            self.config.to_json(json_file)
            self.assertTrue(os.path.exists(json_file))
            
            with open(json_file, 'r') as f:
                content = f.read()
            
            self.assertIsInstance(content, str)
            json_dict = json.loads(content)
            self.assertEqual(json_dict["random_seed"], 123)
        finally:
            os.unlink(json_file)
    
    def test_to_yaml(self):
        """Test conversion to YAML."""
        # Test without file
        yaml_str = self.config.to_yaml()
        self.assertIsInstance(yaml_str, str)
        
        # Test with file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            yaml_file = temp.name
        
        try:
            self.config.to_yaml(yaml_file)
            self.assertTrue(os.path.exists(yaml_file))
            
            with open(yaml_file, 'r') as f:
                content = f.read()
            
            self.assertIsInstance(content, str)
            yaml_dict = yaml.safe_load(content)
            self.assertEqual(yaml_dict["random_seed"], 123)
        finally:
            os.unlink(yaml_file)
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = self.config.to_dict()
        new_config = Config.from_dict(config_dict)
        
        self.assertEqual(new_config.random_seed, self.config.random_seed)
        self.assertEqual(new_config.results_dir, self.config.results_dir)
        self.assertIn("test", new_config.datasets)
        self.assertEqual(new_config.datasets["test"].path, "data/test.csv")
    
    def test_from_json(self):
        """Test creation from JSON."""
        json_str = self.config.to_json()
        new_config = Config.from_json(json_str)
        
        self.assertEqual(new_config.random_seed, self.config.random_seed)
        self.assertEqual(new_config.results_dir, self.config.results_dir)
        self.assertIn("test", new_config.datasets)
    
    def test_from_yaml(self):
        """Test creation from YAML."""
        yaml_str = self.config.to_yaml()
        new_config = Config.from_yaml(yaml_str)
        
        self.assertEqual(new_config.random_seed, self.config.random_seed)
        self.assertEqual(new_config.results_dir, self.config.results_dir)
        self.assertIn("test", new_config.datasets)
    
    def test_from_file(self):
        """Test creation from file."""
        # JSON file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            json_file = temp.name
        
        try:
            self.config.to_json(json_file)
            json_config = Config.from_file(json_file)
            
            self.assertEqual(json_config.random_seed, self.config.random_seed)
            self.assertEqual(json_config.results_dir, self.config.results_dir)
        finally:
            os.unlink(json_file)
        
        # YAML file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
            yaml_file = temp.name
        
        try:
            self.config.to_yaml(yaml_file)
            yaml_config = Config.from_file(yaml_file)
            
            self.assertEqual(yaml_config.random_seed, self.config.random_seed)
            self.assertEqual(yaml_config.results_dir, self.config.results_dir)
        finally:
            os.unlink(yaml_file)
    
    def test_merge(self):
        """Test merging two configurations."""
        other_config = Config(
            random_seed=456,
            models={
                "new_model": ModelConfig(
                    hidden_dims=[128, 64],
                    dropout=0.4
                )
            }
        )
        
        merged_config = self.config.merge(other_config)
        
        # Check that values from other_config take precedence
        self.assertEqual(merged_config.random_seed, 456)
        
        # Check that dictionaries are merged
        self.assertIn("test_model", merged_config.models)
        self.assertIn("new_model", merged_config.models)
        
        # Check that original configs are not modified
        self.assertEqual(self.config.random_seed, 123)
        self.assertNotIn("new_model", self.config.models)
    
    def test_update(self):
        """Test updating specific configuration values."""
        updated_config = self.config.update(
            random_seed=789,
            results_dir="updated_results",
            "models.test_model.dropout": 0.5
        )
        
        # Check that values are updated
        self.assertEqual(updated_config.random_seed, 789)
        self.assertEqual(updated_config.results_dir, "updated_results")
        self.assertEqual(updated_config.models["test_model"].dropout, 0.5)
        
        # Check that original config is not modified
        self.assertEqual(self.config.random_seed, 123)
        self.assertEqual(self.config.results_dir, "test_results")
        self.assertEqual(self.config.models["test_model"].dropout, 0.3)


if __name__ == "__main__":
    unittest.main()