"""
Tests for LLM response parsing and standardization.
"""

import unittest
from unittest.mock import patch
from typing import Dict, Any, Optional

from ..responses import LLMResponse, ResponseParser


class TestLLMResponse(unittest.TestCase):
    """Test the LLMResponse class."""
    
    def test_basic_initialization(self):
        """Test basic initialization of LLMResponse."""
        response = LLMResponse(
            text="This is a test response",
            model="test-model",
            provider="test-provider",
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
            raw_response={"response": "data"}
        )
        
        self.assertEqual(response.text, "This is a test response")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.provider, "test-provider")
        self.assertEqual(response.tokens_used, 10)
        self.assertEqual(response.prompt_tokens, 5)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.raw_response, {"response": "data"})
    
    def test_default_values(self):
        """Test default values for optional parameters."""
        response = LLMResponse(
            text="Minimal response",
            model="test-model",
            provider="test-provider"
        )
        
        self.assertEqual(response.text, "Minimal response")
        self.assertIsNone(response.tokens_used)
        self.assertIsNone(response.prompt_tokens)
        self.assertIsNone(response.completion_tokens)
        self.assertIsNone(response.finish_reason)
        self.assertEqual(response.raw_response, {})
    
    def test_automatic_token_calculation(self):
        """Test automatic calculation of tokens_used when prompt_tokens and completion_tokens are provided."""
        response = LLMResponse(
            text="Token test",
            model="test-model",
            provider="test-provider",
            prompt_tokens=10,
            completion_tokens=20,
        )
        
        self.assertEqual(response.tokens_used, 30)
    
    def test_token_calculation_precedence(self):
        """Test that explicit tokens_used takes precedence over calculated value."""
        response = LLMResponse(
            text="Token precedence test",
            model="test-model",
            provider="test-provider",
            tokens_used=50,  # Explicit value
            prompt_tokens=10,
            completion_tokens=20,
        )
        
        # Should use the explicit value, not the calculated one
        self.assertEqual(response.tokens_used, 50)
    
    def test_str_representation(self):
        """Test string representation of LLMResponse."""
        response = LLMResponse(
            text="String test",
            model="test-model",
            provider="test-provider",
            tokens_used=15
        )
        
        # Check that the string representation contains the text and model
        str_rep = str(response)
        self.assertIn("String test", str_rep)
        self.assertIn("test-model", str_rep)
        self.assertIn("test-provider", str_rep)
    
    def test_repr_representation(self):
        """Test the repr representation of LLMResponse."""
        response = LLMResponse(
            text="Repr test",
            model="test-model",
            provider="test-provider",
            tokens_used=15
        )
        
        # Check that repr contains all the essential attributes
        repr_str = repr(response)
        self.assertIn("Repr test", repr_str)
        self.assertIn("test-model", repr_str)
        self.assertIn("test-provider", repr_str)
        self.assertIn("tokens_used=15", repr_str)


class TestResponseParser(unittest.TestCase):
    """Test the ResponseParser class."""
    
    def test_parse_openai_response(self):
        """Test parsing an OpenAI-style response."""
        openai_response = {
            "choices": [
                {
                    "message": {
                        "content": "This is an OpenAI response"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            },
            "model": "gpt-4"
        }
        
        response = ResponseParser.parse(
            response_data=openai_response,
            provider="openai",
            model="gpt-4",
            prompt="Test prompt"
        )
        
        self.assertEqual(response.text, "This is an OpenAI response")
        self.assertEqual(response.model, "gpt-4")
        self.assertEqual(response.provider, "openai")
        self.assertEqual(response.tokens_used, 25)
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 15)
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.raw_response, openai_response)
    
    def test_parse_claude_response(self):
        """Test parsing a Claude-style response."""
        claude_response = {
            "content": [
                {"type": "text", "text": "This is a Claude response"}
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 18
            },
            "model": "claude-3-opus-20240229"
        }
        
        response = ResponseParser.parse(
            response_data=claude_response,
            provider="anthropic",
            model="claude-3-opus-20240229",
            prompt="Test prompt"
        )
        
        self.assertEqual(response.text, "This is a Claude response")
        self.assertEqual(response.model, "claude-3-opus-20240229")
        self.assertEqual(response.provider, "anthropic")
        self.assertEqual(response.tokens_used, 30)
        self.assertEqual(response.prompt_tokens, 12)
        self.assertEqual(response.completion_tokens, 18)
        self.assertEqual(response.finish_reason, "end_turn")
        self.assertEqual(response.raw_response, claude_response)
    
    def test_parse_grok_response(self):
        """Test parsing a Grok-style response."""
        grok_response = {
            "completion": "This is a Grok response",
            "tokens_consumed": 8,
            "tokens_generated": 12,
            "model": "grok-1"
        }
        
        response = ResponseParser.parse(
            response_data=grok_response,
            provider="grok",
            model="grok-1",
            prompt="Test prompt"
        )
        
        self.assertEqual(response.text, "This is a Grok response")
        self.assertEqual(response.model, "grok-1")
        self.assertEqual(response.provider, "grok")
        self.assertEqual(response.tokens_used, 20)  # sum of consumed and generated
        self.assertEqual(response.prompt_tokens, 8)
        self.assertEqual(response.completion_tokens, 12)
        self.assertEqual(response.finish_reason, None)  # Grok doesn't provide this
        self.assertEqual(response.raw_response, grok_response)
    
    def test_parse_fallback(self):
        """Test falling back to generic parsing for unknown providers."""
        unknown_response = {
            "result": "This is from an unknown provider",
            "metadata": {
                "tokens": 15
            }
        }
        
        response = ResponseParser.parse(
            response_data=unknown_response,
            provider="unknown",
            model="unknown-model",
            prompt="Test prompt"
        )
        
        # Should extract text using best-effort approach
        self.assertEqual(response.text, "This is from an unknown provider")
        self.assertEqual(response.model, "unknown-model")
        self.assertEqual(response.provider, "unknown")
        self.assertIsNone(response.tokens_used)  # Parser couldn't determine this
        self.assertEqual(response.raw_response, unknown_response)
    
    def test_handle_empty_response(self):
        """Test handling an empty response."""
        empty_response = {}
        
        response = ResponseParser.parse(
            response_data=empty_response,
            provider="test",
            model="test-model",
            prompt="Test prompt"
        )
        
        # Should create a response with empty text
        self.assertEqual(response.text, "")
        self.assertEqual(response.model, "test-model")
        self.assertEqual(response.provider, "test")
        self.assertEqual(response.raw_response, empty_response)


if __name__ == '__main__':
    unittest.main()