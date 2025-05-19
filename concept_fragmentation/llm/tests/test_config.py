"""
Tests for environment variable configuration for LLM clients.
"""

import unittest
from unittest.mock import patch
import os
import tempfile
from typing import Dict, Any, Optional

from ..config import (
    load_api_key_from_env,
    get_env_var_name,
    load_config_from_env,
    load_config_from_file,
    ENV_VAR_MAPPINGS
)


class TestConfig(unittest.TestCase):
    """Test the configuration utilities for LLM clients."""
    
    def test_get_env_var_name(self):
        """Test getting the correct environment variable name for each provider."""
        # Test default mappings
        self.assertEqual(get_env_var_name("openai"), "OPENAI_API_KEY")
        self.assertEqual(get_env_var_name("gpt"), "OPENAI_API_KEY")  # Alias
        self.assertEqual(get_env_var_name("anthropic"), "ANTHROPIC_API_KEY")
        self.assertEqual(get_env_var_name("claude"), "ANTHROPIC_API_KEY")  # Alias
        self.assertEqual(get_env_var_name("grok"), "GROK_API_KEY")
        self.assertEqual(get_env_var_name("xai"), "GROK_API_KEY")  # Alias
        
        # Test unknown provider
        with self.assertRaises(ValueError):
            get_env_var_name("unknown_provider")
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"})
    def test_load_api_key_from_env(self):
        """Test loading API key from environment variables."""
        # Test loading existing key
        key = load_api_key_from_env("openai")
        self.assertEqual(key, "test-openai-key")
        
        # Test alias
        key = load_api_key_from_env("gpt")
        self.assertEqual(key, "test-openai-key")
        
        # Test missing key
        with self.assertRaises(ValueError):
            load_api_key_from_env("anthropic")  # Not in mock environment
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_BASE_URL": "https://custom-openai.example.com",
        "OPENAI_TIMEOUT": "30",
        "OPENAI_MAX_RETRIES": "5",
        "OPENAI_DEFAULT_MODEL": "gpt-4-turbo",
        "ANTHROPIC_API_KEY": "test-anthropic-key"
    })
    def test_load_config_from_env(self):
        """Test loading configuration from environment variables."""
        # Test OpenAI configuration
        config = load_config_from_env("openai")
        
        self.assertEqual(config["api_key"], "test-openai-key")
        self.assertEqual(config["base_url"], "https://custom-openai.example.com")
        self.assertEqual(config["timeout"], 30)  # Converted to int
        self.assertEqual(config["max_retries"], 5)  # Converted to int
        self.assertEqual(config["model"], "gpt-4-turbo")
        
        # Test Anthropic with minimal configuration
        config = load_config_from_env("anthropic")
        
        self.assertEqual(config["api_key"], "test-anthropic-key")
        self.assertNotIn("base_url", config)  # Not set in env
        self.assertNotIn("timeout", config)  # Not set in env
        
        # Test provider with no configuration
        with self.assertRaises(ValueError):
            load_config_from_env("grok")  # Not in mock environment
    
    def test_load_config_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.write("""
            {
                "openai": {
                    "api_key": "file-openai-key",
                    "base_url": "https://file-api.example.com",
                    "timeout": 45,
                    "max_retries": 4,
                    "default_model": "gpt-4-file"
                },
                "anthropic": {
                    "api_key": "file-anthropic-key"
                }
            }
            """)
            temp_file_path = temp_file.name
        
        try:
            # Test loading OpenAI config
            config = load_config_from_file(temp_file_path, "openai")
            
            self.assertEqual(config["api_key"], "file-openai-key")
            self.assertEqual(config["base_url"], "https://file-api.example.com")
            self.assertEqual(config["timeout"], 45)
            self.assertEqual(config["max_retries"], 4)
            self.assertEqual(config["model"], "gpt-4-file")
            
            # Test loading Anthropic config with minimal settings
            config = load_config_from_file(temp_file_path, "anthropic")
            
            self.assertEqual(config["api_key"], "file-anthropic-key")
            self.assertNotIn("base_url", config)
            
            # Test loading missing provider
            with self.assertRaises(ValueError):
                load_config_from_file(temp_file_path, "grok")
            
            # Test loading from non-existent file
            with self.assertRaises(FileNotFoundError):
                load_config_from_file("/nonexistent/path.json", "openai")
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "env-openai-key",
        "OPENAI_BASE_URL": "https://env-api.example.com"
    })
    def test_env_precedence_over_file(self):
        """Test that environment variables take precedence over file configuration."""
        # Create a temporary config file with different values
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file.write("""
            {
                "openai": {
                    "api_key": "file-openai-key",
                    "base_url": "https://file-api.example.com",
                    "timeout": 45
                }
            }
            """)
            temp_file_path = temp_file.name
        
        try:
            # First load from file
            file_config = load_config_from_file(temp_file_path, "openai")
            
            # Then merge with env (env should override)
            env_config = load_config_from_env("openai", raise_on_missing_key=False)
            
            # Merge configs (env takes precedence)
            merged_config = {**file_config, **env_config}
            
            # Check that environment values took precedence
            self.assertEqual(merged_config["api_key"], "env-openai-key")
            self.assertEqual(merged_config["base_url"], "https://env-api.example.com")
            
            # Check that file-only values are preserved
            self.assertEqual(merged_config["timeout"], 45)
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)


if __name__ == '__main__':
    unittest.main()