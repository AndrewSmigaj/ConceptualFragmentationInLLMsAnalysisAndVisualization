"""
Tests for provider-specific LLM client implementations.
"""

import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import aiohttp
import asyncio
from typing import Dict, Any, Optional, List

from ..client import BaseLLMClient
from ..responses import LLMResponse

# Import the provider-specific clients
# Note: These imports would be updated once the actual implementations exist
try:
    from ..grok import GrokClient
    from ..claude import ClaudeClient
    from ..openai_client import OpenAIClient
    CLIENTS_IMPORTED = True
except ImportError:
    # Mock implementations for testing if the actual classes don't exist yet
    class GrokClient(BaseLLMClient):
        """Mock implementation of GrokClient for testing."""
        
        provider_name = "grok"
        
        async def generate(self, prompt: str, temperature: float = 0.7,
                        max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
            """Mock generate method."""
            return LLMResponse(
                text="Grok response",
                model=self.model,
                provider=self.provider_name,
                tokens_used=10,
                raw_response={}
            )

    class ClaudeClient(BaseLLMClient):
        """Mock implementation of ClaudeClient for testing."""
        
        provider_name = "claude"
        
        async def generate(self, prompt: str, temperature: float = 0.7,
                        max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
            """Mock generate method."""
            return LLMResponse(
                text="Claude response",
                model=self.model,
                provider=self.provider_name,
                tokens_used=10,
                raw_response={}
            )

    class OpenAIClient(BaseLLMClient):
        """Mock implementation of OpenAIClient for testing."""
        
        provider_name = "openai"
        
        async def generate(self, prompt: str, temperature: float = 0.7,
                        max_tokens: Optional[int] = None, **kwargs) -> LLMResponse:
            """Mock generate method."""
            return LLMResponse(
                text="OpenAI response",
                model=self.model,
                provider=self.provider_name,
                tokens_used=10,
                raw_response={}
            )
    
    CLIENTS_IMPORTED = False


class TestGrokClient(unittest.IsolatedAsyncioTestCase):
    """Test the GrokClient implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-grok-key"
        self.model = "grok-1"
    
    @patch('aiohttp.ClientSession.post')
    async def test_generate(self, mock_post):
        """Test the generate method of GrokClient."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "completion": "This is a response from Grok AI.",
            "tokens_consumed": 10,
            "tokens_generated": 15,
            "model": "grok-1"
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = GrokClient(api_key=self.api_key, model=self.model)
        response = await client.generate(
            prompt="What is AI?",
            temperature=0.7,
            max_tokens=100
        )
        
        # Check the response
        self.assertEqual(response.text, "This is a response from Grok AI.")
        self.assertEqual(response.model, "grok-1")
        self.assertEqual(response.provider, "grok")
        self.assertEqual(response.tokens_used, 25)  # 10 + 15
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 15)
        
        # Check the API call
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("json", call_kwargs)
        self.assertIn("headers", call_kwargs)
        
        # Check headers
        headers = call_kwargs["headers"]
        self.assertEqual(headers.get("x-api-key"), self.api_key)
        
        # Check request body
        body = call_kwargs["json"]
        self.assertEqual(body.get("prompt"), "What is AI?")
        self.assertEqual(body.get("temperature"), 0.7)
        self.assertEqual(body.get("max_tokens"), 100)
    
    @patch('aiohttp.ClientSession.post')
    async def test_error_handling(self, mock_post):
        """Test error handling in GrokClient."""
        # Mock an API error response
        mock_response = MagicMock()
        mock_response.status = 429  # Too Many Requests
        mock_response.text = AsyncMock(return_value="Rate limit exceeded")
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = GrokClient(api_key=self.api_key, model=self.model)
        
        # Test that the appropriate exception is raised
        with self.assertRaises(Exception) as context:
            await client.generate(prompt="What is AI?")
        
        # Check the exception message
        self.assertIn("429", str(context.exception))
        self.assertIn("Rate limit", str(context.exception))
    
    def test_default_model(self):
        """Test the default model selection for GrokClient."""
        # Create a client without specifying a model
        client = GrokClient(api_key=self.api_key)
        
        # Check that a default model was selected
        self.assertIsNotNone(client.model)
        self.assertNotEqual(client.model, "default")


class TestClaudeClient(unittest.IsolatedAsyncioTestCase):
    """Test the ClaudeClient implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-claude-key"
        self.model = "claude-3-opus-20240229"
    
    @patch('aiohttp.ClientSession.post')
    async def test_generate(self, mock_post):
        """Test the generate method of ClaudeClient."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "content": [
                {"type": "text", "text": "This is a response from Claude."}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 18
            },
            "model": "claude-3-opus-20240229"
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = ClaudeClient(api_key=self.api_key, model=self.model)
        response = await client.generate(
            prompt="What is AI?",
            temperature=0.5,
            max_tokens=200,
            system_prompt="You are a helpful AI assistant."
        )
        
        # Check the response
        self.assertEqual(response.text, "This is a response from Claude.")
        self.assertEqual(response.model, "claude-3-opus-20240229")
        self.assertEqual(response.provider, "claude")
        self.assertEqual(response.tokens_used, 30)  # 12 + 18
        self.assertEqual(response.prompt_tokens, 12)
        self.assertEqual(response.completion_tokens, 18)
        self.assertEqual(response.finish_reason, "end_turn")
        
        # Check the API call
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("json", call_kwargs)
        self.assertIn("headers", call_kwargs)
        
        # Check headers
        headers = call_kwargs["headers"]
        self.assertEqual(headers.get("x-api-key") or headers.get("anthropic-api-key"), self.api_key)
        
        # Check request body
        body = call_kwargs["json"]
        self.assertEqual(body.get("model"), self.model)
        self.assertEqual(body.get("temperature"), 0.5)
        self.assertEqual(body.get("max_tokens"), 200)
        
        # Check system prompt and user message
        messages = body.get("messages", [])
        self.assertGreaterEqual(len(messages), 1)  # At least one message
        
        # If using system prompt, it should be included
        if "system" in body:
            self.assertEqual(body["system"], "You are a helpful AI assistant.")
    
    @patch('aiohttp.ClientSession.post')
    async def test_error_handling(self, mock_post):
        """Test error handling in ClaudeClient."""
        # Mock an API error response
        mock_response = MagicMock()
        mock_response.status = 400  # Bad Request
        mock_response.text = AsyncMock(return_value='{"error": {"message": "Invalid request"}}')
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = ClaudeClient(api_key=self.api_key, model=self.model)
        
        # Test that the appropriate exception is raised
        with self.assertRaises(Exception) as context:
            await client.generate(prompt="What is AI?")
        
        # Check the exception message
        self.assertIn("400", str(context.exception))
        self.assertIn("Invalid request", str(context.exception))
    
    def test_default_model(self):
        """Test the default model selection for ClaudeClient."""
        # Create a client without specifying a model
        client = ClaudeClient(api_key=self.api_key)
        
        # Check that a default model was selected
        self.assertIsNotNone(client.model)
        self.assertNotEqual(client.model, "default")


class TestOpenAIClient(unittest.IsolatedAsyncioTestCase):
    """Test the OpenAIClient implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-openai-key"
        self.model = "gpt-4"
    
    @patch('aiohttp.ClientSession.post')
    async def test_generate(self, mock_post):
        """Test the generate method of OpenAIClient."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "content": "This is a response from GPT-4."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35
            },
            "model": "gpt-4"
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = OpenAIClient(api_key=self.api_key, model=self.model)
        response = await client.generate(
            prompt="What is AI?",
            temperature=0.3,
            max_tokens=150,
            system_prompt="You are a helpful AI assistant."
        )
        
        # Check the response
        self.assertEqual(response.text, "This is a response from GPT-4.")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.provider, "openai")
        self.assertEqual(response.tokens_used, 35)
        self.assertEqual(response.prompt_tokens, 15)
        self.assertEqual(response.completion_tokens, 20)
        self.assertEqual(response.finish_reason, "stop")
        
        # Check the API call
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        self.assertIn("json", call_kwargs)
        self.assertIn("headers", call_kwargs)
        
        # Check headers
        headers = call_kwargs["headers"]
        self.assertEqual(headers.get("Authorization"), f"Bearer {self.api_key}")
        
        # Check request body
        body = call_kwargs["json"]
        self.assertEqual(body.get("model"), self.model)
        self.assertEqual(body.get("temperature"), 0.3)
        self.assertEqual(body.get("max_tokens"), 150)
        
        # Check messages format
        messages = body.get("messages", [])
        self.assertGreaterEqual(len(messages), 1)  # At least one message
        
        # Check system message if it exists
        if len(messages) > 1 and messages[0].get("role") == "system":
            self.assertEqual(messages[0].get("content"), "You are a helpful AI assistant.")
        
        # Check user message
        user_message = messages[-1] if messages[-1].get("role") == "user" else None
        self.assertIsNotNone(user_message)
        self.assertEqual(user_message.get("content"), "What is AI?")
    
    @patch('aiohttp.ClientSession.post')
    async def test_error_handling(self, mock_post):
        """Test error handling in OpenAIClient."""
        # Mock an API error response
        mock_response = MagicMock()
        mock_response.status = 401  # Unauthorized
        mock_response.text = AsyncMock(return_value='{"error": {"message": "Invalid API key"}}')
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = OpenAIClient(api_key=self.api_key, model=self.model)
        
        # Test that the appropriate exception is raised
        with self.assertRaises(Exception) as context:
            await client.generate(prompt="What is AI?")
        
        # Check the exception message
        self.assertIn("401", str(context.exception))
        self.assertIn("Invalid API key", str(context.exception))
    
    def test_default_model(self):
        """Test the default model selection for OpenAIClient."""
        # Create a client without specifying a model
        client = OpenAIClient(api_key=self.api_key)
        
        # Check that a default model was selected
        self.assertIsNotNone(client.model)
        self.assertNotEqual(client.model, "default")
    
    @patch('aiohttp.ClientSession.post')
    async def test_chat_vs_completion_endpoint(self, mock_post):
        """Test that the client uses the correct endpoint based on the model."""
        # This is a simplified test since we're mocking the response
        # In a real implementation, the endpoint URL would be different for chat vs completion
        
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [
                {
                    "message": {
                        "content": "This is a response from GPT-4."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35
            },
            "model": "gpt-4"
        })
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Create the client and call generate
        client = OpenAIClient(api_key=self.api_key, model=self.model)
        response = await client.generate(prompt="What is AI?")
        
        # We're mainly checking that the function completes successfully
        self.assertEqual(response.text, "This is a response from GPT-4.")


if __name__ == '__main__':
    unittest.main()