"""
Tests for the base LLM client implementation.
"""

import unittest
import abc
import asyncio
from unittest.mock import patch, MagicMock
from typing import Optional, Dict, Any, Awaitable

from ..client import BaseLLMClient
from ..responses import LLMResponse


class MockBaseLLMClient(BaseLLMClient):
    """Mock implementation of BaseLLMClient for testing."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "mock-model",
        timeout: int = 60,
        max_retries: int = 3,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
            **kwargs
        )
        self.mock_response = LLMResponse(
            text="This is a mock response",
            model=model,
            provider="mock",
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
            raw_response={"mock": True}
        )
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using the mock implementation."""
        # Store the last prompt for testing
        self.last_prompt = prompt
        self.last_temperature = temperature
        self.last_max_tokens = max_tokens
        self.last_kwargs = kwargs
        
        return self.mock_response


class TestBaseLLMClient(unittest.TestCase):
    """Test the BaseLLMClient abstract base class."""
    
    def test_abstract_class(self):
        """Test that BaseLLMClient is an abstract class that can't be instantiated directly."""
        with self.assertRaises(TypeError):
            client = BaseLLMClient(api_key="test-key")
    
    def test_initialization(self):
        """Test that a concrete subclass can be instantiated with the expected attributes."""
        client = MockBaseLLMClient(
            api_key="test-key",
            model="test-model",
            timeout=30,
            max_retries=5,
            base_url="https://api.example.com"
        )
        
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.model, "test-model")
        self.assertEqual(client.timeout, 30)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.base_url, "https://api.example.com")
    
    def test_initialization_defaults(self):
        """Test default values in initialization."""
        client = MockBaseLLMClient(api_key="test-key")
        
        self.assertEqual(client.model, "mock-model")
        self.assertEqual(client.timeout, 60)
        self.assertEqual(client.max_retries, 3)
        self.assertIsNone(client.base_url)
    
    def test_generate_method_parameters(self):
        """Test that the generate method captures parameters correctly."""
        client = MockBaseLLMClient(api_key="test-key")
        
        # Run the generate method
        response = asyncio.run(client.generate(
            prompt="Hello, world!",
            temperature=0.5,
            max_tokens=100,
            custom_param="custom-value"
        ))
        
        # Check that parameters were captured correctly
        self.assertEqual(client.last_prompt, "Hello, world!")
        self.assertEqual(client.last_temperature, 0.5)
        self.assertEqual(client.last_max_tokens, 100)
        self.assertEqual(client.last_kwargs.get("custom_param"), "custom-value")
    
    def test_generate_sync(self):
        """Test the synchronous wrapper for generate."""
        client = MockBaseLLMClient(api_key="test-key")
        
        # Test the synchronous wrapper
        response = client.generate_sync(
            prompt="Hello, sync world!",
            temperature=0.3
        )
        
        # Verify the response
        self.assertEqual(response.text, "This is a mock response")
        self.assertEqual(response.model, "mock-model")
        self.assertEqual(response.provider, "mock")
        
        # Verify that parameters were passed correctly
        self.assertEqual(client.last_prompt, "Hello, sync world!")
        self.assertEqual(client.last_temperature, 0.3)
    
    @patch('asyncio.get_event_loop')
    def test_generate_sync_event_loop_creation(self, mock_get_loop):
        """Test that generate_sync creates a new event loop if needed."""
        # Mock the get_event_loop to raise a RuntimeError (simulating no running loop)
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        mock_loop = MagicMock()
        mock_loop.run_until_complete = MagicMock()
        
        # Mock the new_event_loop method
        with patch('asyncio.new_event_loop', return_value=mock_loop):
            with patch('asyncio.set_event_loop'):
                client = MockBaseLLMClient(api_key="test-key")
                response = client.generate_sync(prompt="Hello, no loop!")
                
                # Check that a new loop was created and run_until_complete was called
                mock_loop.run_until_complete.assert_called_once()
    
    @patch.object(BaseLLMClient, '_get_default_model')
    def test_get_default_model(self, mock_get_default):
        """Test that _get_default_model is called when model is 'default'."""
        mock_get_default.return_value = "default-test-model"
        
        # Initialize with model="default" to trigger _get_default_model
        client = MockBaseLLMClient(
            api_key="test-key",
            model="default"
        )
        
        mock_get_default.assert_called_once()
        self.assertEqual(client.model, "default-test-model")
    
    def test_api_key_validation(self):
        """Test validation of API key (should be implemented by subclasses)."""
        # We expect this to pass since MockBaseLLMClient doesn't implement validation
        client = MockBaseLLMClient(api_key=None)
        self.assertIsNone(client.api_key)


if __name__ == '__main__':
    unittest.main()