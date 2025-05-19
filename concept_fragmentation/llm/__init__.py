"""
LLM integration module for the Concept Fragmentation project.
"""

try:
    from .api_keys import (
        OPENAI_KEY, 
        OPENAI_API_BASE, 
        XAI_API_KEY, 
        GEMINI_API_KEY
    )
    API_KEYS_AVAILABLE = True
except ImportError:
    API_KEYS_AVAILABLE = False
    print("Warning: API keys not found. LLM features will be disabled.")