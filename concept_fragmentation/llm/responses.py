"""
Standardized response classes for LLM clients.

This module provides a unified response format for different LLM providers,
normalizing the different response structures into a common interface.
"""

from typing import Dict, Any, Optional, List, Union


class LLMResponse:
    """Standardized response object from an LLM provider."""
    
    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        tokens_used: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        finish_reason: Optional[str] = None,
        raw_response: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an LLM response.
        
        Args:
            text: The generated text response
            model: The model used to generate the response
            provider: The provider name (e.g., "openai", "claude", "grok")
            tokens_used: Total number of tokens used (prompt + completion)
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            finish_reason: Why the model stopped generating (e.g., "stop", "length")
            raw_response: The original, unmodified response from the provider
        """
        self.text = text
        self.model = model
        self.provider = provider
        self.tokens_used = tokens_used
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.finish_reason = finish_reason
        self.raw_response = raw_response or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the response to a dictionary for serialization."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "tokens_used": self.tokens_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
            "raw_response": self.raw_response
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """Create an LLMResponse from a dictionary."""
        return cls(
            text=data.get("text", ""),
            model=data.get("model", "unknown"),
            provider=data.get("provider", "unknown"),
            tokens_used=data.get("tokens_used", 0),
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            finish_reason=data.get("finish_reason"),
            raw_response=data.get("raw_response", {})
        )
    
    def __str__(self) -> str:
        """String representation of the response."""
        return f"LLMResponse(text='{self.text[:50]}...', model='{self.model}', provider='{self.provider}')"
    
    def __repr__(self) -> str:
        """Detailed representation of the response."""
        return (f"LLMResponse(text='{self.text[:50]}...', model='{self.model}', "
                f"provider='{self.provider}', tokens_used={self.tokens_used})")


class ResponseParser:
    """Static methods for parsing provider-specific responses into LLMResponse objects."""
    
    @staticmethod
    def parse_openai_response(
        response: Dict[str, Any],
        model: str,
        provider: str = "openai"
    ) -> LLMResponse:
        """
        Parse an OpenAI API response into an LLMResponse.
        
        Args:
            response: The raw API response from OpenAI
            model: The model name
            provider: The provider name (default: "openai")
            
        Returns:
            A standardized LLMResponse object
        """
        # Extract text from the response
        text = ""
        if "choices" in response and response["choices"]:
            # Chat completions format
            if "message" in response["choices"][0]:
                text = response["choices"][0]["message"].get("content", "")
            # Legacy completions format
            elif "text" in response["choices"][0]:
                text = response["choices"][0].get("text", "")
        
        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        tokens_used = usage.get("total_tokens", prompt_tokens + completion_tokens)
        
        # Extract finish reason
        finish_reason = None
        if "choices" in response and response["choices"]:
            finish_reason = response["choices"][0].get("finish_reason")
        
        # Create standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            raw_response=response
        )
    
    @staticmethod
    def parse_claude_response(
        response: Dict[str, Any],
        model: str,
        provider: str = "claude"
    ) -> LLMResponse:
        """
        Parse an Anthropic/Claude API response into an LLMResponse.
        
        Args:
            response: The raw API response from Claude
            model: The model name
            provider: The provider name (default: "claude")
            
        Returns:
            A standardized LLMResponse object
        """
        # Extract text from the response
        text = ""
        if "content" in response and response["content"]:
            # Concatenate all text content blocks
            for content_block in response["content"]:
                if content_block.get("type") == "text":
                    text += content_block.get("text", "")
        
        # Extract token counts
        usage = response.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        tokens_used = prompt_tokens + completion_tokens
        
        # Extract finish reason
        finish_reason = response.get("stop_reason")
        
        # Create standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            raw_response=response
        )
    
    @staticmethod
    def parse_grok_response(
        response: Dict[str, Any],
        model: str,
        provider: str = "grok"
    ) -> LLMResponse:
        """
        Parse a Grok/xAI API response into an LLMResponse.
        
        Args:
            response: The raw API response from Grok
            model: The model name
            provider: The provider name (default: "grok")
            
        Returns:
            A standardized LLMResponse object
        """
        # Extract text from the response (presuming Grok API format)
        text = response.get("completion", "")
        
        # Extract token counts
        prompt_tokens = response.get("tokens_consumed", 0)
        completion_tokens = response.get("tokens_generated", 0)
        tokens_used = prompt_tokens + completion_tokens
        
        # Finish reason not always provided by Grok
        finish_reason = response.get("finish_reason")
        
        # Create standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            raw_response=response
        )
    
    @staticmethod
    def parse_gemini_response(
        response: Dict[str, Any],
        model: str,
        provider: str = "gemini"
    ) -> LLMResponse:
        """
        Parse a Google Gemini API response into an LLMResponse.
        
        Args:
            response: The raw API response from Gemini
            model: The model name
            provider: The provider name (default: "gemini")
            
        Returns:
            A standardized LLMResponse object
        """
        # Extract text from the response
        text = ""
        if "candidates" in response and response["candidates"]:
            candidate = response["candidates"][0]
            if "content" in candidate and candidate["content"]:
                for part in candidate["content"].get("parts", []):
                    if "text" in part:
                        text += part["text"]
        
        # Extract token counts if available
        usage = response.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        tokens_used = prompt_tokens + completion_tokens
        
        # Extract finish reason
        finish_reason = None
        if "candidates" in response and response["candidates"]:
            finish_reason = response["candidates"][0].get("finishReason")
        
        # Create standardized response
        return LLMResponse(
            text=text,
            model=model,
            provider=provider,
            tokens_used=tokens_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            raw_response=response
        )