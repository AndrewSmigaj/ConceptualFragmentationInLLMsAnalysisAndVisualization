"""
Tests for the GPT-2 Archetypal Path Analysis command-line tool.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module to test
from concept_fragmentation.analysis.gpt2_cluster_paths import (
    main,
    extract_gpt2_activations,
    analyze_window_activations,
    visualize_window_results,
    write_results
)


class TestGPT2AnalysisCLI(unittest.TestCase):
    """Test the GPT-2 Analysis command-line interface."""
    
    def test_imports(self):
        """Test that imports work correctly."""
        # This test passes if the imports above succeed
        self.assertTrue(True)
    
    @patch('concept_fragmentation.analysis.gpt2_cluster_paths.extract_gpt2_activations')
    @patch('concept_fragmentation.analysis.gpt2_cluster_paths.analyze_window_activations')
    @patch('concept_fragmentation.analysis.gpt2_cluster_paths.visualize_window_results')
    @patch('concept_fragmentation.analysis.gpt2_cluster_paths.write_results')
    @patch('concept_fragmentation.analysis.gpt2_cluster_paths.setup_output_dirs')
    @patch('argparse.ArgumentParser.parse_args')
    def test_pipeline_flow(self, mock_parse_args, mock_setup_dirs, mock_write, mock_visualize, mock_analyze, mock_extract):
        """Test the main pipeline flow."""
        # Set up mock returns
        mock_parse_args.return_value = MagicMock(
            input_file=None,
            text="Test text",
            sample=False,
            model="gpt2",
            device="cpu",
            context_window=128,
            window_size=3,
            stride=1,
            n_clusters=5,
            seed=42,
            visualize=True,
            highlight_tokens=None,
            min_path_count=1,
            output_dir="./test_output",
            timestamp=None
        )
        mock_setup_dirs.return_value = {
            "main": "./test_output/gpt2_123",
            "activations": "./test_output/gpt2_123/activations",
            "clusters": "./test_output/gpt2_123/clusters",
            "results": "./test_output/gpt2_123/results",
            "visualizations": "./test_output/gpt2_123/visualizations"
        }
        mock_extract.return_value = {
            "window_0_2": {"activations": {}, "metadata": {}}
        }
        mock_analyze.return_value = {"window_name": "window_0_2", "clusters": {}}
        mock_visualize.return_value = {}
        mock_write.return_value = "./test_output/gpt2_123/results.json"
        
        # Call the main function with mocked command-line args
        with patch('sys.argv', ['run_gpt2_analysis.py', '--text', 'Test text']):
            try:
                main()
                # If we get here, the test passes
                self.assertTrue(True)
            except SystemExit:
                # main() might call sys.exit() which raises SystemExit
                pass
        
        # Verify pipeline flow
        mock_extract.assert_called_once()
        mock_analyze.assert_called_once()
        mock_visualize.assert_called_once()
        mock_write.assert_called_once()


if __name__ == '__main__':
    unittest.main()