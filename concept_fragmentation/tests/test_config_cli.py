"""
Tests for the configuration CLI module.

This module tests the command-line interface functions in the configuration
package.
"""

import unittest
import sys
import os
import tempfile
import json
import yaml
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concept_fragmentation.config import config_manager, get_config, Config
from concept_fragmentation.config.cli import main, view_command, export_command, import_command, validate_command, set_command


class Args:
    """Mock class for CLI arguments."""
    pass


class TestConfigCLI(unittest.TestCase):
    """Test the configuration CLI interface."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Save the original config
        self.original_config = config_manager.get_config()
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore the original config
        config_manager.set_config(self.original_config)
        
        # Clean up temporary directory
        self.temp_dir.cleanup()
    
    def test_view_command(self):
        """Test the view command."""
        # Test viewing entire config
        args = Args()
        args.format = "json"
        args.output = None
        args.section = None
        
        with patch('builtins.print') as mock_print:
            exit_code = view_command(args)
            self.assertEqual(exit_code, 0)
            mock_print.assert_called_once()
        
        # Test viewing a specific section
        args = Args()
        args.format = "python"
        args.output = None
        args.section = "metrics.cluster_entropy"
        
        with patch('builtins.print') as mock_print:
            exit_code = view_command(args)
            self.assertEqual(exit_code, 0)
            mock_print.assert_called_once()
        
        # Test saving to file
        output_file = self.temp_path / "config.json"
        args = Args()
        args.format = "json"
        args.output = str(output_file)
        args.section = None
        
        with patch('builtins.print') as mock_print:
            exit_code = view_command(args)
            self.assertEqual(exit_code, 0)
            mock_print.assert_called_once()
            self.assertTrue(output_file.exists())
    
    def test_export_command(self):
        """Test the export command."""
        # Test exporting entire config
        output_file = self.temp_path / "exported_config.json"
        args = Args()
        args.format = "json"
        args.output = str(output_file)
        args.dataset = None
        args.experiment = None
        
        with patch('builtins.print') as mock_print:
            exit_code = export_command(args)
            self.assertEqual(exit_code, 0)
            mock_print.assert_called_once()
            self.assertTrue(output_file.exists())
        
        # Test exporting in YAML format
        output_file = self.temp_path / "exported_config.yaml"
        args = Args()
        args.format = "yaml"
        args.output = str(output_file)
        args.dataset = None
        args.experiment = None
        
        with patch('builtins.print') as mock_print:
            exit_code = export_command(args)
            self.assertEqual(exit_code, 0)
            mock_print.assert_called_once()
            self.assertTrue(output_file.exists())
    
    def test_import_command(self):
        """Test the import command."""
        # Create a test config file
        test_config = {
            "random_seed": 12345,
            "results_dir": "/tmp/test_results",
            "device": "cpu",
            "log_level": 20
        }
        config_file = self.temp_path / "test_config.json"
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Test importing config
        args = Args()
        args.input = str(config_file)
        args.validate = True
        args.merge = False
        args.save = False
        
        with patch('builtins.print') as mock_print:
            exit_code = import_command(args)
            self.assertEqual(exit_code, 0)
            
            # Check that config was updated
            new_config = config_manager.get_config()
            self.assertEqual(new_config.random_seed, 12345)
            self.assertEqual(new_config.results_dir, "/tmp/test_results")
    
    def test_validate_command(self):
        """Test the validate command."""
        # Create a valid test config file
        valid_config = {
            "random_seed": 12345,
            "results_dir": "/tmp/test_results",
            "device": "cpu",
            "log_level": 20
        }
        valid_file = self.temp_path / "valid_config.json"
        with open(valid_file, 'w') as f:
            json.dump(valid_config, f)
        
        # Test validating a file
        args = Args()
        args.input = str(valid_file)
        
        with patch('builtins.print') as mock_print:
            exit_code = validate_command(args)
            self.assertEqual(exit_code, 0)
        
        # Create an invalid test config file
        invalid_config = {
            "random_seed": "not an integer",  # This should cause validation to fail
            "results_dir": "/tmp/test_results",
            "device": "cpu"
        }
        invalid_file = self.temp_path / "invalid_config.json"
        with open(invalid_file, 'w') as f:
            json.dump(invalid_config, f)
        
        # Test validating current config
        args = Args()
        args.input = None
        
        with patch('builtins.print') as mock_print:
            exit_code = validate_command(args)
            self.assertEqual(exit_code, 0)  # Default config should be valid
    
    def test_set_command(self):
        """Test the set command."""
        # Test setting an integer value
        args = Args()
        args.key = "random_seed"
        args.value = "54321"
        args.save = False
        
        with patch('builtins.print') as mock_print:
            exit_code = set_command(args)
            self.assertEqual(exit_code, 0)
            
            # Check that value was set
            new_config = config_manager.get_config()
            self.assertEqual(new_config.random_seed, 54321)
        
        # Test setting a boolean value
        args = Args()
        args.key = "metrics.subspace_angle.random_state"
        args.value = "100"
        args.save = False
        
        with patch('builtins.print') as mock_print:
            exit_code = set_command(args)
            self.assertEqual(exit_code, 0)
            
            # Check that value was set
            new_config = config_manager.get_config()
            self.assertEqual(new_config.metrics.subspace_angle.random_state, 100)
    
    def test_main(self):
        """Test the main CLI function."""
        # Test the view command
        with patch('sys.argv', ['cli.py', 'view']), \
             patch('concept_fragmentation.config.cli.view_command') as mock_view_command:
            mock_view_command.return_value = 0
            exit_code = main()
            self.assertEqual(exit_code, 0)
            mock_view_command.assert_called_once()
        
        # Test the export command
        with patch('sys.argv', ['cli.py', 'export', 'test.json']), \
             patch('concept_fragmentation.config.cli.export_command') as mock_export_command:
            mock_export_command.return_value = 0
            exit_code = main()
            self.assertEqual(exit_code, 0)
            mock_export_command.assert_called_once()
        
        # Test with no command (should show help)
        with patch('sys.argv', ['cli.py']), \
             patch('argparse.ArgumentParser.print_help') as mock_print_help:
            exit_code = main()
            self.assertEqual(exit_code, 1)
            mock_print_help.assert_called_once()


if __name__ == '__main__':
    unittest.main()