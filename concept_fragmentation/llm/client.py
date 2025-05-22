"""
Base client class for LLM API interactions.

This module provides an abstract base class for LLM clients,
defining a common interface for different providers.
"""

import abc
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Union

from .responses import LLMResponse


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients."""
    
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
        Initialize the LLM client.
        
        Args:
            api_key: The API key for the LLM provider
            model: The model to use (default: "default" which will be replaced with provider default)
            timeout: Timeout in seconds for API requests
            max_retries: Maximum number of retries for failed requests
            base_url: Optional custom base URL for the API
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.kwargs = kwargs
        
        # Set the model, using default if specified
        if model == "default":
            self.model = self._get_default_model()
        else:
            self.model = model
        
        # Validate API key (can be implemented by subclasses)
        self._validate_api_key()
    
    def _validate_api_key(self) -> None:
        """
        Validate the API key.
        
        Raises:
            ValueError: If the API key is invalid or missing when required
        """
        # Base implementation does minimal validation
        # Subclasses can implement stricter validation
        pass
    
    def _get_default_model(self) -> str:
        """
        Get the default model for this provider.
        
        Returns:
            The default model name
        """
        # Subclasses should override this with provider-specific defaults
        return "default-model"
    
    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (0.0 = deterministic, 1.0 = maximum randomness)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An LLMResponse object containing the generated text and metadata
            
        Raises:
            Exception: If the API request fails
        """
        pass
    
    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Synchronous wrapper for the generate method.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Controls randomness (0.0 = deterministic, 1.0 = maximum randomness)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An LLMResponse object containing the generated text and metadata
            
        Raises:
            Exception: If the API request fails
        """
        try:
            # Try to get the event loop
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there is no event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async method and return the result
        return loop.run_until_complete(self.generate(prompt, temperature, max_tokens, **kwargs))
    
    async def _retry_request(
        self,
        request_func,
        max_retries: int = None,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs
    ):
        """
        Retry a request with exponential backoff.
        
        Args:
            request_func: Async function to execute the request
            max_retries: Maximum number of retries (defaults to self.max_retries)
            initial_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay after each retry
            **kwargs: Additional parameters to pass to request_func
            
        Returns:
            The result of the successful request
            
        Raises:
            Exception: If all retries fail
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        retries = 0
        delay = initial_delay
        last_exception = None
        
        while retries <= max_retries:
            try:
                return await request_func(**kwargs)
            except Exception as e:
                last_exception = e
                retries += 1
                
                # Check if we should retry
                if retries > max_retries:
                    break
                
                # Wait with exponential backoff
                await asyncio.sleep(delay)
                delay *= backoff_factor
        
        # If we get here, all retries failed
        raise last_exception or Exception("Request failed after all retries")

# Create an alias for backward compatibility
LLMClient = BaseLLMClient