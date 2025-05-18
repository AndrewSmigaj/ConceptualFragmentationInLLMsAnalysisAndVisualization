"""
Integration test for the new configuration system.

This test ensures that the new configuration system correctly loads and maintains
backward compatibility with the legacy configuration.
"""

import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import both the legacy config directly and the new config module
from concept_fragmentation import config as legacy_config
from concept_fragmentation.config import config_manager, get_config


class TestConfigIntegration(unittest.TestCase):
    """Test the integration between the legacy and new configuration systems."""
    
    def test_legacy_config_loaded(self):
        """Test that the legacy configuration is correctly loaded by the ConfigManager."""
        # Load the config from ConfigManager
        config = get_config()
        
        # Check that basic settings are loaded correctly
        self.assertEqual(config.random_seed, legacy_config.RANDOM_SEED)
        self.assertEqual(config.results_dir, legacy_config.RESULTS_DIR)
        self.assertEqual(config.device, legacy_config.DEVICE)
        
        # Check dataset configuration
        for dataset_name, dataset_config in legacy_config.DATASETS.items():
            self.assertIn(dataset_name, config.datasets)
            new_config = config.datasets[dataset_name]
            
            # Check key attributes
            if 'path' in dataset_config:
                self.assertEqual(new_config.path, dataset_config['path'])
            if 'test_size' in dataset_config:
                self.assertEqual(new_config.test_size, dataset_config['test_size'])
            if 'categorical_features' in dataset_config:
                self.assertEqual(set(new_config.categorical_features), set(dataset_config['categorical_features']))
        
        # Check model configuration
        for dataset_name, dims in legacy_config.MODELS['feedforward']['hidden_dims'].items():
            self.assertIn(f"feedforward_{dataset_name}", config.models)
            self.assertEqual(config.models[f"feedforward_{dataset_name}"].hidden_dims, dims)
        
        # Verify at least one regularization config is loaded
        self.assertTrue(len(config.regularization) > 0)
        
        # Check metrics configuration
        self.assertEqual(config.metrics.cluster_entropy.default_k, 
                         legacy_config.METRICS['cluster_entropy']['default_k'])
        
        # Check training configuration
        for dataset_name, batch_size in legacy_config.TRAINING['batch_size'].items():
            self.assertIn(dataset_name, config.training)
            self.assertEqual(config.training[dataset_name].batch_size, batch_size)
        
        # Check visualization configuration
        self.assertEqual(config.visualization.umap.n_neighbors, 
                         legacy_config.VISUALIZATION['umap']['n_neighbors'])
        
        print("Legacy config successfully loaded and matches with new config system.")
        
    def test_config_serialization(self):
        """Test that the configuration can be serialized and deserialized."""
        config = get_config()
        
        # Test serialization to JSON
        json_str = config.to_json()
        loaded_config = config_manager.load_from_dict(json.loads(json_str))
        
        # Check that key attributes match
        self.assertEqual(config.random_seed, loaded_config.random_seed)
        self.assertEqual(config.results_dir, loaded_config.results_dir)
        self.assertEqual(config.device, loaded_config.device)
        
        # Check that datasets match
        for dataset_name, dataset_config in config.datasets.items():
            self.assertIn(dataset_name, loaded_config.datasets)
            
        print("Configuration serialization/deserialization works correctly.")


if __name__ == '__main__':
    unittest.main()