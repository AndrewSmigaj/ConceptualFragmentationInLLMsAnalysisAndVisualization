"""
Grok/xAI API client implementation.

This client works with the Grok/xAI API.
"""

import aiohttp
import json
from typing import Dict, Any, Optional, List, Union, Tuple

from .client import BaseLLMClient
from .responses import LLMResponse, ResponseParser


class GrokClient(BaseLLMClient):
    """Client for Grok/xAI API."""
    
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
            model: The model to use (default: grok-1)
            timeout: Timeout in seconds for API requests
            max_retries: Maximum number of retries for failed requests
            base_url: Optional base URL for the API (default: https://api.xai.com/v1)
            **kwargs: Additional parameters
        """
        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url or "https://api.grok.meta.com/v1",
            **kwargs
        )
    
    def _get_default_model(self) -> str:
        """Get the default model for Grok."""
        return "grok-1"
    
    def _validate_api_key(self) -> None:
        """Validate the Grok API key."""
        if not self.api_key:
            raise ValueError("Grok API key is required")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using the Grok API.
        
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
        # Build API request data
        request_data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
        }
        
        # Add max_tokens if specified
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        
        # Add system prompt if specified and supported by the API
        if system_prompt:
            request_data["system_prompt"] = system_prompt
        
        # Add any additional parameters
        for key, value in kwargs.items():
            if key not in request_data:
                request_data[key] = value
        
        # Determine the endpoint URL
        endpoint = f"{self.base_url}/completion"
        
        # Define the async request function
        async def make_request():
            """Make the API request."""
            async with aiohttp.ClientSession() as session:
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json"
                }
                
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=request_data,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        error_message = f"API request failed with status {response.status}: {error_text}"
                        raise Exception(error_message)
                    
                    # Parse the response
                    response_data = await response.json()
                    return ResponseParser.parse_grok_response(
                        response_data,
                        self.model,
                        self.provider_name
                    )
        
        # Make the request with retries
        return await self._retry_request(make_request)