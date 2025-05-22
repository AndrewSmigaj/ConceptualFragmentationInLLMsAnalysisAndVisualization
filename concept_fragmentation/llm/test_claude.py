"""
Simple test script for the Claude API integration.
"""

import asyncio
from factory import LLMClientFactory

async def main():
    """Test the Claude client."""
    print("Testing Claude client...")
    
    # Create a Claude client
    client = LLMClientFactory.create_client("claude")
    
    # Test with a simple prompt
    prompt = "Explain the concept of conceptual fragmentation in neural networks in one paragraph."
    
    print(f"Sending prompt to Claude: {prompt}")
    
    # Generate response
    response = await client.generate(
        prompt=prompt,
        temperature=0.7,
    )
    
    # Print the response
    print("\nResponse from Claude:")
    print(f"Model: {response.model}")
    print(f"Provider: {response.provider}")
    print(f"Content: {response.content}")
    print(f"Tokens: {response.token_count}")

if __name__ == "__main__":
    asyncio.run(main())