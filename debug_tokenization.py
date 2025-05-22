"""
Debug script to understand GPT-2 tokenization of our sentences.
"""

def debug_tokenization():
    """Debug GPT-2 tokenization to understand the mismatch."""
    
    # Test sentences
    test_sentences = [
        "good but bad",
        "great but awful", 
        "happy but sad"
    ]
    
    print("=== TOKENIZATION DEBUG ===")
    
    try:
        import sys
        sys.path.append('./venv311/Lib/site-packages')
        from transformers import GPT2Tokenizer
        
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        for sentence in test_sentences:
            print(f"\nSentence: '{sentence}'")
            print(f"Word split: {sentence.split()}")
            
            # Tokenize using GPT-2
            tokens = tokenizer.tokenize(sentence)
            token_ids = tokenizer.encode(sentence, return_tensors="pt")
            decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids[0]]
            
            print(f"GPT-2 tokens: {tokens}")
            print(f"Token IDs shape: {token_ids.shape}")
            print(f"Token IDs: {token_ids[0].tolist()}")
            print(f"Decoded tokens: {decoded_tokens}")
            
    except ImportError as e:
        print(f"Cannot import transformers: {e}")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tokenization()