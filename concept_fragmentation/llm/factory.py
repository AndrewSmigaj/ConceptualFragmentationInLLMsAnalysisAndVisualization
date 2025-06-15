"""
Factory for creating appropriate LLM clients.

This module provides a factory for creating LLM clients based on the specified provider.
"""

import os
import logging
from typing import Dict, Any, Optional, Type, List

from .client import BaseLLMClient

# Try to use Concept MRI debug logger if available, otherwise use module logger
try:
    logger = logging.getLogger('concept_mri.debug')
except:
    logger = logging.getLogger(__name__)

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

# Also try to import from local_config if available
try:
    import sys
    from pathlib import Path
    # Add parent directories to path to find local_config
    root_dir = Path(__file__).parent.parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    from local_config import OPENAI_KEY as LOCAL_OPENAI_KEY, XAI_API_KEY as LOCAL_XAI_API_KEY, GEMINI_API_KEY as LOCAL_GEMINI_API_KEY
    
    # Override empty keys with local_config values
    if not OPENAI_KEY or OPENAI_KEY == "":
        OPENAI_KEY = LOCAL_OPENAI_KEY
    if not XAI_API_KEY or XAI_API_KEY == "":
        XAI_API_KEY = LOCAL_XAI_API_KEY
    if not GEMINI_API_KEY or GEMINI_API_KEY == "":
        GEMINI_API_KEY = LOCAL_GEMINI_API_KEY
    # Note: ANTHROPIC_API_KEY not in local_config, keep from api_keys.py
    
    logger.info(f"Successfully loaded API keys from local_config - XAI_API_KEY: {'Set' if XAI_API_KEY else 'Not set'}")
except ImportError as e:
    logger.warning(f"Could not import API keys from local_config: {e}")


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
    def update_default_keys(cls):
        """Update DEFAULT_API_KEYS with values loaded from local_config"""
        cls.DEFAULT_API_KEYS = {
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
        logger.info(f"Creating LLM client for provider: {provider}, model: {model}")
        
        # Get the client class for the provider
        client_class = cls._get_client_class(provider)
        
        # Get the API key if not provided
        if api_key is None:
            logger.info(f"No API key provided, looking up key for provider: {provider}")
            api_key = cls._get_api_key(provider)
        else:
            logger.info(f"Using provided API key for provider: {provider}")
        
        # Initialize configuration dictionary
        config_dict = config or {}
        
        # Add provider-specific configurations
        if provider in ["openai", "gpt"] and OPENAI_API_BASE and "base_url" not in config_dict:
            config_dict["base_url"] = OPENAI_API_BASE
            logger.info(f"Using custom base URL for OpenAI: {OPENAI_API_BASE}")
        
        # Create and return the client
        logger.info(f"Creating {client_class.__name__} instance")
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
        logger.info(f"Looking up API key for provider: {provider}")
        
        # Try to get from imported api_keys.py first
        if API_KEYS_AVAILABLE and provider in cls.DEFAULT_API_KEYS:
            api_key = cls.DEFAULT_API_KEYS.get(provider)
            logger.info(f"Checking DEFAULT_API_KEYS[{provider}]: {'Found' if api_key else 'Empty/None'}")
            if api_key:
                logger.info(f"Found API key for {provider} in DEFAULT_API_KEYS")
                return api_key
        
        # Try to get from environment variables
        env_vars = cls.ENV_API_KEY_MAP.get(provider, [])
        logger.info(f"Checking environment variables for {provider}: {env_vars}")
        for env_var in env_vars:
            api_key = os.environ.get(env_var)
            if api_key:
                logger.info(f"Found API key for {provider} in environment variable: {env_var}")
                return api_key
        
        # If no API key found, raise an error
        env_var_names = ", ".join(cls.ENV_API_KEY_MAP.get(provider, []))
        logger.error(f"No API key found for provider '{provider}'")
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
    logger.debug("Registering LLM clients...")
    
    try:
        from .openai_client import OpenAIClient
        LLMClientFactory.PROVIDER_MAP["openai"] = OpenAIClient
        LLMClientFactory.PROVIDER_MAP["gpt"] = OpenAIClient
        logger.debug("[+] Registered OpenAI/GPT client")
    except ImportError as e:
        logger.debug(f"[-] Failed to register OpenAI client: {e}")
    
    try:
        from .claude import ClaudeClient
        LLMClientFactory.PROVIDER_MAP["claude"] = ClaudeClient
        LLMClientFactory.PROVIDER_MAP["anthropic"] = ClaudeClient
        logger.debug("[+] Registered Claude/Anthropic client")
    except ImportError as e:
        logger.debug(f"[-] Failed to register Claude client: {e}")
    
    try:
        from .grok import GrokClient
        LLMClientFactory.PROVIDER_MAP["grok"] = GrokClient
        LLMClientFactory.PROVIDER_MAP["xai"] = GrokClient
        logger.debug("[+] Registered Grok/xAI client")
    except ImportError as e:
        logger.debug(f"[-] Failed to register Grok client: {e}")
    
    try:
        from .gemini import GeminiClient
        LLMClientFactory.PROVIDER_MAP["gemini"] = GeminiClient
        logger.debug("[+] Registered Gemini client")
    except ImportError as e:
        logger.debug(f"[-] Failed to register Gemini client: {e}")
        
    logger.info(f"Available LLM providers: {', '.join(LLMClientFactory.PROVIDER_MAP.keys())}")


# Register clients when the module is imported
register_clients()

# Update default keys after imports
LLMClientFactory.update_default_keys()