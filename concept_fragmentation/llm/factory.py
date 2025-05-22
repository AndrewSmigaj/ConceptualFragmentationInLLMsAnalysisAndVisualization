"""
Factory for creating appropriate LLM clients.

This module provides a factory for creating LLM clients based on the specified provider.
"""

import os
from typing import Dict, Any, Optional, Type, List

from .client import BaseLLMClient

# Import API keys if available
try:
    from .api_keys import (
        OPENAI_KEY,
        OPENAI_API_BASE,
        XAI_API_KEY,
        GEMINI_API_KEY,
        ANTHROPIC_API_KEY
    )
    API_KEYS_AVAILABLE = True
except ImportError:
    API_KEYS_AVAILABLE = False
    OPENAI_KEY = None
    OPENAI_API_BASE = None
    XAI_API_KEY = None
    GEMINI_API_KEY = None
    ANTHROPIC_API_KEY = None


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    # Map of provider names to client classes
    # This will be populated once all client implementations are available
    PROVIDER_MAP: Dict[str, Type[BaseLLMClient]] = {}
    
    # Map of environment variable names for API keys by provider
    ENV_API_KEY_MAP = {
        "grok": ["GROK_API_KEY", "XAI_API_KEY"],
        "xai": ["XAI_API_KEY", "GROK_API_KEY"],
        "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "gpt": ["OPENAI_API_KEY"],
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    }
    
    # Map of default API keys from api_keys.py
    DEFAULT_API_KEYS = {
        "grok": XAI_API_KEY,
        "xai": XAI_API_KEY,
        "openai": OPENAI_KEY,
        "gpt": OPENAI_KEY,
        "gemini": GEMINI_API_KEY,
        "claude": ANTHROPIC_API_KEY,
        "anthropic": ANTHROPIC_API_KEY
    }
    
    @classmethod
    def create_client(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        model: str = "default",
        config: Optional[Dict[str, Any]] = None
    ) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.
        
        Args:
            provider: The name of the provider (e.g., "openai", "claude", "grok")
            api_key: The API key to use (if None, will try to get from environment)
            model: The model to use (default: provider's default model)
            config: Additional configuration for the client
            
        Returns:
            An instance of the appropriate LLM client
            
        Raises:
            ValueError: If the provider is unknown or no API key is available
        """
        # Get the client class for the provider
        client_class = cls._get_client_class(provider)
        
        # Get the API key if not provided
        if api_key is None:
            api_key = cls._get_api_key(provider)
        
        # Initialize configuration dictionary
        config_dict = config or {}
        
        # Add provider-specific configurations
        if provider in ["openai", "gpt"] and OPENAI_API_BASE and "base_url" not in config_dict:
            config_dict["base_url"] = OPENAI_API_BASE
        
        # Create and return the client
        return client_class(
            api_key=api_key,
            model=model,
            **config_dict
        )
    
    @classmethod
    def _get_client_class(cls, provider: str) -> Type[BaseLLMClient]:
        """
        Get the client class for the specified provider.
        
        Args:
            provider: The name of the provider
            
        Returns:
            The client class
            
        Raises:
            ValueError: If the provider is unknown
        """
        # Normalize provider name
        provider = provider.lower().strip()
        
        # Check if provider is supported
        if provider not in cls.PROVIDER_MAP:
            available = ", ".join(cls.get_available_providers())
            raise ValueError(f"Unknown provider: {provider}. Available providers: {available}")
        
        return cls.PROVIDER_MAP[provider]
    
    @classmethod
    def _get_api_key(cls, provider: str) -> str:
        """
        Get the API key for the specified provider.
        
        Args:
            provider: The name of the provider
            
        Returns:
            The API key
            
        Raises:
            ValueError: If no API key is available for the provider
        """
        # Try to get from imported api_keys.py first
        if API_KEYS_AVAILABLE and provider in cls.DEFAULT_API_KEYS:
            api_key = cls.DEFAULT_API_KEYS.get(provider)
            if api_key:
                return api_key
        
        # Try to get from environment variables
        env_vars = cls.ENV_API_KEY_MAP.get(provider, [])
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if api_key:
                return api_key
        
        # If no API key found, raise an error
        env_var_names = ", ".join(cls.ENV_API_KEY_MAP.get(provider, []))
        raise ValueError(
            f"No API key found for provider '{provider}'. "
            f"Please provide an API key or set one of these environment variables: {env_var_names}"
        )
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get a list of available provider names.
        
        Returns:
            A list of available provider names
        """
        return list(cls.PROVIDER_MAP.keys())


# Add a convenience function for backward compatibility
def create_llm_client(
    provider: str,
    api_key: Optional[str] = None,
    model: str = "default",
    config: Optional[Dict[str, Any]] = None
) -> BaseLLMClient:
    """
    Create an LLM client for the specified provider.
    
    Args:
        provider: The name of the provider (e.g., "openai", "claude", "grok")
        api_key: The API key to use (if None, will try to get from environment)
        model: The model to use (default: provider's default model)
        config: Additional configuration for the client
        
    Returns:
        An instance of the appropriate LLM client
        
    Raises:
        ValueError: If the provider is unknown or no API key is available
    """
    return LLMClientFactory.create_client(provider, api_key, model, config)


# Import the provider-specific clients and register them
# This prevents circular imports
def register_clients():
    """Register all available LLM clients with the factory."""
    # Try to import each client implementation and register it if available
    print("Registering LLM clients...")
    
    try:
        from .openai_client import OpenAIClient
        LLMClientFactory.PROVIDER_MAP["openai"] = OpenAIClient
        LLMClientFactory.PROVIDER_MAP["gpt"] = OpenAIClient
        print("[+] Registered OpenAI/GPT client")
    except ImportError as e:
        print(f"[-] Failed to register OpenAI client: {e}")
    
    try:
        from .claude import ClaudeClient
        LLMClientFactory.PROVIDER_MAP["claude"] = ClaudeClient
        LLMClientFactory.PROVIDER_MAP["anthropic"] = ClaudeClient
        print("[+] Registered Claude/Anthropic client")
    except ImportError as e:
        print(f"[-] Failed to register Claude client: {e}")
    
    try:
        from .grok import GrokClient
        LLMClientFactory.PROVIDER_MAP["grok"] = GrokClient
        LLMClientFactory.PROVIDER_MAP["xai"] = GrokClient
        print("[+] Registered Grok/xAI client")
    except ImportError as e:
        print(f"[-] Failed to register Grok client: {e}")
    
    try:
        from .gemini import GeminiClient
        LLMClientFactory.PROVIDER_MAP["gemini"] = GeminiClient
        print("[+] Registered Gemini client")
    except ImportError as e:
        print(f"[-] Failed to register Gemini client: {e}")
        
    print(f"Available providers: {', '.join(LLMClientFactory.PROVIDER_MAP.keys())}")


# Register clients when the module is imported
register_clients()