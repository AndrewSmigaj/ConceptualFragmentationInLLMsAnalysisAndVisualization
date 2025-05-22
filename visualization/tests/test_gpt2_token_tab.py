"""
Tests for GPT-2 token path tab for the visualization dashboard.
"""

import unittest
import sys
import os
import tempfile
import json
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import module to test
from visualization.gpt2_token_tab import (
    find_gpt2_analysis_results,
    load_window_data,
    load_apa_results,
    create_gpt2_token_tab
)


class TestGPT2TokenTab(unittest.TestCase):
    """Test the GPT-2 token path visualization tab functions."""
    
    def setUp(self):
        """Set up test data and environment."""
        # Create a temporary directory for mock data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create results directory structure
        self.results_dir = os.path.join(self.temp_dir.name, "results", "gpt2", "test_analysis")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create window directory
        self.window_dir = os.path.join(self.results_dir, "window_0_2")
        os.makedirs(self.window_dir, exist_ok=True)
        
        # Create APA results directory
        self.apa_dir = os.path.join(self.results_dir, "results", "window_0_2")
        os.makedirs(self.apa_dir, exist_ok=True)
        
        # Create mock metadata file
        self.metadata = {
            "model_type": "gpt2",
            "layer_files": {
                "window_0_2": {
                    "layer_0": os.path.join(self.window_dir, "gpt2_activations_window_0_2_layer_0.npy"),
                    "layer_1": os.path.join(self.window_dir, "gpt2_activations_window_0_2_layer_1.npy"),
                    "layer_2": os.path.join(self.window_dir, "gpt2_activations_window_0_2_layer_2.npy")
                }
            },
            "config": {
                "context_window": 512,
                "include_lm_head": True
            }
        }
        
        self.metadata_file = os.path.join(self.results_dir, "gpt2_activations_metadata.json")
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
        
        # Create mock window metadata file
        self.window_metadata = {
            "metadata": {
                "tokens": [
                    ["The", "quick", "brown", "fox"]
                ],
                "token_ids": [[464, 2068, 6282, 4796]],
                "attention_mask": [[1, 1, 1, 1]]
            },
            "window_layers": [0, 1, 2]
        }
        
        self.window_metadata_file = os.path.join(self.window_dir, "window_0_2_metadata.json")
        with open(self.window_metadata_file, 'w') as f:
            json.dump(self.window_metadata, f)
        
        # Create mock activation files
        for layer in range(3):
            layer_file = os.path.join(self.window_dir, f"gpt2_activations_window_0_2_layer_{layer}.npy")
            # Create a small activation array: batch=1, seq_len=4, features=10
            np.save(layer_file, np.random.random((1, 4, 10)))
        
        # Create mock cluster label files
        for layer in range(3):
            label_file = os.path.join(self.apa_dir, f"layer_{layer}_labels.npy")
            # Create cluster labels for 4 tokens
            np.save(label_file, np.array([0, 1, 2, 1]))
        
        # Add GPT2_RESULTS_DIR patch
        self.results_dir_patcher = patch('visualization.gpt2_token_tab.GPT2_RESULTS_DIR', self.temp_dir.name)
        self.results_dir_patcher.start()
    
    def tearDown(self):
        """Clean up temporary directories and patches."""
        self.temp_dir.cleanup()
        self.results_dir_patcher.stop()
    
    def test_find_gpt2_analysis_results(self):
        """Test finding GPT-2 analysis results."""
        results = find_gpt2_analysis_results()
        
        # Check that our mock result was found
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["model_type"], "gpt2")
        self.assertEqual(results[0]["metadata_file"], self.metadata_file)
        self.assertIn("window_0_2", results[0]["windows"])
    
    def test_load_window_data(self):
        """Test loading window data."""
        window_data = load_window_data(self.metadata_file, "window_0_2")
        
        # Check structure
        self.assertIn("activations", window_data)
        self.assertIn("metadata", window_data)
        self.assertIn("window_layers", window_data)
        
        # Check activations
        self.assertIn("layer_0", window_data["activations"])
        self.assertIn("layer_1", window_data["activations"])
        self.assertIn("layer_2", window_data["activations"])
        
        # Check shape
        self.assertEqual(window_data["activations"]["layer_0"].shape, (1, 4, 10))
        
        # Check metadata
        self.assertIn("tokens", window_data["metadata"])
        self.assertIn("token_ids", window_data["metadata"])
        self.assertIn("attention_mask", window_data["metadata"])
    
    def test_load_apa_results(self):
        """Test loading APA results."""
        apa_results = load_apa_results(self.metadata_file, "window_0_2")
        
        # Check structure
        self.assertIn("clusters", apa_results)
        
        # Check clusters
        self.assertIn("layer_0", apa_results["clusters"])
        self.assertIn("layer_1", apa_results["clusters"])
        self.assertIn("layer_2", apa_results["clusters"])
        
        # Check labels
        self.assertIn("labels", apa_results["clusters"]["layer_0"])
        self.assertEqual(len(apa_results["clusters"]["layer_0"]["labels"]), 4)
    
    def test_create_gpt2_token_tab(self):
        """Test creating the GPT-2 token path tab layout."""
        tab = create_gpt2_token_tab()
        
        # Check that the tab was created
        self.assertEqual(tab.label, "GPT-2 Token Paths")
        
        # In a real test, we would check more components, but that's complex
        # with Dash components. Here we just verify it creates something.
        self.assertIsNotNone(tab)


if __name__ == "__main__":
    unittest.main()