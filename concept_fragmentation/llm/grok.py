"""
Grok/xAI API client implementation.

This client works with the Grok/xAI API using OpenAI-compatible SDK.
"""

import aiohttp
import json
import os
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple

from .client import BaseLLMClient
from .responses import LLMResponse, ResponseParser

# Check if openai is available
try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class GrokClient(BaseLLMClient):
    """Client for Grok/xAI API using OpenAI-compatible SDK."""
    
    provider_name = "grok"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "default",
        timeout: int = 60,
        max_retries: int = 3,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Grok client.
        
        Args:
            api_key: The API key for Grok
            model: The model to use (default: grok-beta)
            timeout: Timeout in seconds for API requests
            max_retries: Maximum number of retries for failed requests
            base_url: Optional base URL for the API (default: https://api.x.ai/v1)
            **kwargs: Additional parameters
        """
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url or "https://api.x.ai/v1",
            **kwargs
        )
        
        # Check if OpenAI SDK is available
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK is required for Grok API integration. "
                "Please install it with: pip install openai"
            )
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout
        )
    
    def _get_default_model(self) -> str:
        """Get the default model for Grok."""
        return "grok-beta"
    
    def _validate_api_key(self) -> None:
        """Validate the Grok API key."""
        if not self.api_key:
            raise ValueError("Grok API key is required")
        if not self.api_key.startswith("xai-"):
            raise ValueError("Grok API key should start with 'xai-'")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using the Grok API via OpenAI SDK.
        
        Args:
            prompt: The prompt to send to the API
            temperature: Controls randomness (0.0 = deterministic, 1.0 = maximum randomness)
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            An LLMResponse object containing the generated text and metadata
            
        Raises:
            Exception: If the API request fails
        """
        # Prepare messages list
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Build completion parameters
        completion_params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            completion_params["max_tokens"] = max_tokens
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in completion_params:
                completion_params[key] = value
        
        # Define the async request function
        async def make_request():
            """Make the API request using OpenAI SDK."""
            try:
                # Make the API request
                response = await self.client.chat.completions.create(**completion_params)
                
                # Extract the response content
                content = response.choices[0].message.content
                
                # Create response object
                return LLMResponse(
                    text=content,
                    model=self.model,
                    provider=self.provider_name,
                    tokens_used=response.usage.total_tokens,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    finish_reason=response.choices[0].finish_reason,
                    raw_response=response.model_dump()
                )
            except Exception as e:
                raise Exception(f"Grok API request failed: {str(e)}")
        
        # Make the request with retries
        return await self._retry_request(make_request)