"""
Simple script to check if an LLM provider can be loaded and generate text.

Usage:
    python -m concept_fragmentation.llm.check_provider --provider <name> [--model <model>]

Example:
    python -m concept_fragmentation.llm.check_provider --provider grok
"""

import os
import sys
import argparse
from concept_fragmentation.llm.factory import LLMClientFactory

def main():
    """Main entry point for checking provider availability."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check LLM provider availability")
    parser.add_argument(
        "--provider", type=str, required=True,
        help="LLM provider name (e.g., 'grok', 'claude', 'openai', 'gemini')"
    )
    parser.add_argument(
        "--model", type=str, default="default",
        help="Model to use (default: provider's default model)"
    )
    args = parser.parse_args()
    
    print(f"Checking availability of provider: {args.provider}")
    print(f"Available providers: {', '.join(LLMClientFactory.get_available_providers())}")
    
    try:
        # Try to create the client
        client = LLMClientFactory.create_client(
            provider=args.provider,
            model=args.model
        )
        
        print(f"Successfully created client for provider: {args.provider}")
        print(f"Using model: {client.model}")
        
        # Check if the client has necessary attributes and methods
        if hasattr(client, 'generate_sync'):
            print("[+] Client has generate_sync method")
        else:
            print("[-] Client is missing generate_sync method")
            
        # Try a simple generation (uncomment if needed)
        # response = client.generate_sync(
        #     prompt="Test prompt: what is 2+2?",
        #     temperature=0.7,
        #     max_tokens=50
        # )
        # print(f"Generated response: {response.text}")
        
        return True
    except Exception as e:
        print(f"Failed to create or use client: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)