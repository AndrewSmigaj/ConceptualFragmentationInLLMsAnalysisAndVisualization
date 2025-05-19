"""
Tests for the LLM client factory.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
from typing import Dict, Any, Optional

from ..factory import LLMClientFactory
from ..client import BaseLLMClient
from ..responses import LLMResponse


# Mock implementations of provider-specific clients
class MockGrokClient(BaseLLMClient):
    """Mock implementation of GrokClient for testing."""
    
    provider_name = "grok"
    
    async def generate(self, prompt: str, temperature: float = 0.7,
                       max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
        """Mocked generate method."""
        return LLMResponse(
            text="Grok response",
            model=self.model,
            provider=self.provider_name,
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
            raw_response={}
        )


class MockClaudeClient(BaseLLMClient):
    """Mock implementation of ClaudeClient for testing."""
    
    provider_name = "claude"
    
    async def generate(self, prompt: str, temperature: float = 0.7,
                       max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
        """Mocked generate method."""
        return LLMResponse(
            text="Claude response",
            model=self.model,
            provider=self.provider_name,
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
            raw_response={}
        )


class MockOpenAIClient(BaseLLMClient):
    """Mock implementation of OpenAIClient for testing."""
    
    provider_name = "openai"
    
    async def generate(self, prompt: str, temperature: float = 0.7,
                       max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
        """Mocked generate method."""
        return LLMResponse(
            text="OpenAI response",
            model=self.model,
            provider=self.provider_name,
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
            raw_response={}
        )


class TestLLMClientFactory(unittest.TestCase):
    """Test the LLMClientFactory class."""
    
    def setUp(self):
        """Set up mocks and patches for each test."""
        # Add mock provider map
        self.original_provider_map = LLMClientFactory.PROVIDER_MAP.copy()
        LLMClientFactory.PROVIDER_MAP = {
            "grok": MockGrokClient,
            "xai": MockGrokClient,  # Alias for Grok
            "claude": MockClaudeClient,
            "anthropic": MockClaudeClient,  # Alias for Claude
            "openai": MockOpenAIClient,
            "gpt": MockOpenAIClient,  # Alias for OpenAI
        }
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            "GROK_API_KEY": "mock-grok-key",
            "ANTHROPIC_API_KEY": "mock-claude-key",
            "OPENAI_API_KEY": "mock-openai-key"
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up after each test."""
        # Restore provider map
        LLMClientFactory.PROVIDER_MAP = self.original_provider_map
        
        # Stop environment variable patch
        self.env_patcher.stop()
    
    def test_create_client_with_provider(self):
        """Test creating clients for each provider."""
        # Create a client for each provider
        grok_client = LLMClientFactory.create_client("grok")
        claude_client = LLMClientFactory.create_client("claude")
        openai_client = LLMClientFactory.create_client("openai")
        
        # Check client types
        self.assertIsInstance(grok_client, MockGrokClient)
        self.assertIsInstance(claude_client, MockClaudeClient)
        self.assertIsInstance(openai_client, MockOpenAIClient)
    
    def test_create_client_with_alias(self):
        """Test creating clients using provider aliases."""
        # Create clients using aliases
        xai_client = LLMClientFactory.create_client("xai")
        anthropic_client = LLMClientFactory.create_client("anthropic")
        gpt_client = LLMClientFactory.create_client("gpt")
        
        # Check client types
        self.assertIsInstance(xai_client, MockGrokClient)
        self.assertIsInstance(anthropic_client, MockClaudeClient)
        self.assertIsInstance(gpt_client, MockOpenAIClient)
    
    def test_create_client_with_explicit_api_key(self):
        """Test creating a client with an explicitly provided API key."""
        client = LLMClientFactory.create_client("grok", api_key="custom-key")
        self.assertEqual(client.api_key, "custom-key")
    
    def test_create_client_with_env_api_key(self):
        """Test creating a client with an API key from environment variables."""
        # Create clients without explicit API keys
        grok_client = LLMClientFactory.create_client("grok")
        claude_client = LLMClientFactory.create_client("claude")
        openai_client = LLMClientFactory.create_client("openai")
        
        # Check that API keys were loaded from environment variables
        self.assertEqual(grok_client.api_key, "mock-grok-key")
        self.assertEqual(claude_client.api_key, "mock-claude-key")
        self.assertEqual(openai_client.api_key, "mock-openai-key")
    
    def test_create_client_with_model(self):
        """Test creating a client with a specified model."""
        client = LLMClientFactory.create_client("grok", model="custom-model")
        self.assertEqual(client.model, "custom-model")
    
    def test_create_client_with_config(self):
        """Test creating a client with additional configuration."""
        config = {
            "timeout": 120,
            "max_retries": 5,
            "base_url": "https://custom-api.example.com"
        }
        client = LLMClientFactory.create_client("grok", config=config)
        
        self.assertEqual(client.timeout, 120)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.base_url, "https://custom-api.example.com")
    
    def test_create_client_unknown_provider(self):
        """Test that creating a client with an unknown provider raises a ValueError."""
        with self.assertRaises(ValueError):
            LLMClientFactory.create_client("unknown-provider")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_create_client_missing_api_key(self):
        """Test handling of missing API key when not provided explicitly and not in environment."""
        # Patch os.environ to be empty
        with self.assertRaises(ValueError):
            LLMClientFactory.create_client("grok")
    
    def test_get_available_providers(self):
        """Test getting the list of available providers."""
        providers = LLMClientFactory.get_available_providers()
        
        # Check that the main providers are included
        self.assertIn("grok", providers)
        self.assertIn("claude", providers)
        self.assertIn("openai", providers)
        
        # Check that aliases are included
        self.assertIn("xai", providers)
        self.assertIn("anthropic", providers)
        self.assertIn("gpt", providers)


if __name__ == '__main__':
    unittest.main()