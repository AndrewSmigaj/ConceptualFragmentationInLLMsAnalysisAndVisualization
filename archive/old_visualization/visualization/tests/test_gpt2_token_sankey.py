"""
Tests for GPT-2 token Sankey diagram visualization.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import module to test
from visualization.gpt2_token_sankey import (
    extract_token_paths,
    prepare_token_sankey_data,
    generate_token_sankey_diagram,
    get_token_path_stats,
    create_token_path_comparison,
    create_3layer_window_sankey
)


class TestGPT2TokenSankey(unittest.TestCase):
    """Test the GPT-2 token Sankey diagram visualization tools."""
    
    def setUp(self):
        """Set up test data."""
        # Create mock activations
        self.activations = {
            "layer_0": np.random.random((2, 5, 10)),  # 2 batch, 5 tokens, 10 features
            "layer_1": np.random.random((2, 5, 10)),
            "layer_2": np.random.random((2, 5, 10))
        }
        
        # Create mock token metadata
        self.token_metadata = {
            "tokens": [
                ["The", "quick", "brown", "fox", "jumps"],
                ["A", "lazy", "dog", "sleeps", "now"]
            ],
            "token_ids": np.array([
                [464, 2068, 6282, 4796, 5246],
                [257, 4395, 3290, 11241, 1110]
            ]),
            "attention_mask": np.ones((2, 5), dtype=int)
        }
        
        # Create mock cluster labels
        self.cluster_labels = {
            "layer_0": np.array([0, 1, 2, 1, 0, 2, 0, 1, 0, 1]),  # Flattened: batch*seq_len
            "layer_1": np.array([1, 0, 0, 2, 1, 0, 2, 1, 2, 0]),
            "layer_2": np.array([0, 2, 1, 0, 2, 1, 0, 2, 1, 0])
        }
        
        # Create mock window data
        self.window_data = {
            "activations": self.activations,
            "metadata": self.token_metadata,
            "window_layers": ["layer_0", "layer_1", "layer_2"]
        }
        
        # Create mock analysis result
        self.analysis_result = {
            "clusters": {
                "layer_0": {"labels": self.cluster_labels["layer_0"]},
                "layer_1": {"labels": self.cluster_labels["layer_1"]},
                "layer_2": {"labels": self.cluster_labels["layer_2"]}
            }
        }
    
    def test_extract_token_paths(self):
        """Test extracting token paths from activations."""
        token_paths = extract_token_paths(
            self.activations,
            self.token_metadata,
            self.cluster_labels
        )
        
        # Check basic structure
        self.assertIn("layers", token_paths)
        self.assertIn("tokens", token_paths)
        self.assertIn("token_ids", token_paths)
        self.assertIn("paths_by_token", token_paths)
        self.assertIn("paths_by_token_position", token_paths)
        
        # Check layers
        self.assertEqual(token_paths["layers"], ["layer_0", "layer_1", "layer_2"])
        
        # Check token paths
        self.assertEqual(len(token_paths["paths_by_token_position"]), 10)  # 2 batches * 5 tokens
        
        # Check a specific token
        token_key = "batch0_pos0"  # "The"
        self.assertIn(token_key, token_paths["paths_by_token_position"])
        self.assertEqual(token_paths["paths_by_token_position"][token_key]["token_text"], "The")
        self.assertEqual(len(token_paths["paths_by_token_position"][token_key]["cluster_path"]), 3)
    
    def test_prepare_token_sankey_data(self):
        """Test preparing Sankey data for token paths."""
        # First extract the token paths
        token_paths = extract_token_paths(
            self.activations,
            self.token_metadata,
            self.cluster_labels
        )
        
        # Test without highlight tokens
        sankey_data = prepare_token_sankey_data(token_paths)
        
        # Check basic structure
        self.assertIn("nodes", sankey_data)
        self.assertIn("node_labels", sankey_data)
        self.assertIn("node_colors", sankey_data)
        self.assertIn("sources", sankey_data)
        self.assertIn("targets", sankey_data)
        self.assertIn("values", sankey_data)
        self.assertIn("link_colors", sankey_data)
        self.assertIn("link_hovers", sankey_data)
        
        # Test with highlight tokens
        highlight_tokens = ["batch0_pos0", "batch1_pos0"]  # "The", "A"
        sankey_data = prepare_token_sankey_data(token_paths, highlight_tokens=highlight_tokens)
        
        # Check that we have links
        self.assertGreater(len(sankey_data["sources"]), 0)
        self.assertGreater(len(sankey_data["targets"]), 0)
        self.assertGreater(len(sankey_data["values"]), 0)
    
    def test_generate_token_sankey_diagram(self):
        """Test generating a Sankey diagram for token paths."""
        # First extract the token paths
        token_paths = extract_token_paths(
            self.activations,
            self.token_metadata,
            self.cluster_labels
        )
        
        # Generate diagram
        fig = generate_token_sankey_diagram(token_paths)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))
        self.assertEqual(len(fig.data), 1)  # One trace (Sankey)
        self.assertEqual(fig.data[0].type, 'sankey')
    
    def test_get_token_path_stats(self):
        """Test calculating token path statistics."""
        # First extract the token paths
        token_paths = extract_token_paths(
            self.activations,
            self.token_metadata,
            self.cluster_labels
        )
        
        # Get statistics
        stats = get_token_path_stats(token_paths)
        
        # Check basic structure
        self.assertIn("total_tokens", stats)
        self.assertIn("unique_tokens", stats)
        self.assertIn("paths", stats)
        self.assertIn("tokens_per_cluster", stats)
        self.assertIn("path_diversity", stats)
        self.assertIn("most_common_paths", stats)
        self.assertIn("most_fragmented_tokens", stats)
        
        # Check token counts
        self.assertEqual(stats["total_tokens"], 10)  # 2 batches * 5 tokens
        self.assertEqual(stats["unique_tokens"], 10)  # All unique in this test
    
    def test_create_token_path_comparison(self):
        """Test creating a token path comparison visualization."""
        # First extract the token paths
        token_paths = extract_token_paths(
            self.activations,
            self.token_metadata,
            self.cluster_labels
        )
        
        # Create comparison
        tokens_to_compare = ["The", "quick"]
        fig = create_token_path_comparison(token_paths, tokens_to_compare)
        
        # Check that the figure was created
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'data'))
    
    def test_create_3layer_window_sankey(self):
        """Test creating a 3-layer window Sankey visualization."""
        # Create a temporary directory for output
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create 3-layer window Sankey
            results = create_3layer_window_sankey(
                self.window_data,
                self.analysis_result,
                output_dir=temp_dir,
                save_html=True
            )
            
            # Check that the results contain the expected figures
            self.assertIn("sankey", results)
            self.assertIn("comparison", results)
            self.assertIn("stats", results)
            
            # Check that files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "token_sankey_diagram.html")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "token_path_comparison.html")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "token_path_stats.json")))


if __name__ == "__main__":
    unittest.main()