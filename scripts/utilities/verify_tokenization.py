"""
Verify tokenization consistency for GPT-2 APA Pivot Experiment.

Check that all sentences tokenize to exactly 3 tokens with "but" at index 1.
"""

from transformers import GPT2Tokenizer

def verify_tokenization():
    """Verify all sentences tokenize correctly."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load sentences
    with open("gpt2_pivot_contrast_sentences.txt", "r") as f:
        contrast_sentences = [line.strip() for line in f if line.strip()]
    
    with open("gpt2_pivot_consistent_sentences.txt", "r") as f:
        consistent_sentences = [line.strip() for line in f if line.strip()]
    
    all_sentences = contrast_sentences + consistent_sentences
    
    print(f"Verifying tokenization for {len(all_sentences)} sentences...")
    
    issues = []
    
    for i, sentence in enumerate(all_sentences):
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence)
        
        # Check token count
        if len(tokens) != 3:
            issues.append(f"Sentence {i}: '{sentence}' has {len(tokens)} tokens instead of 3: {tokens}")
            continue
            
        # Check if "but" is at index 1
        if tokens[1] not in [" but", "but"]:
            issues.append(f"Sentence {i}: '{sentence}' doesn't have 'but' at index 1: {tokens}")
            continue
            
        # Print first few for verification
        if i < 5:
            print(f"✓ '{sentence}' -> {tokens}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} tokenization issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more issues")
        return False
    else:
        print(f"\n✅ All {len(all_sentences)} sentences tokenize correctly!")
        print("   - All have exactly 3 tokens")
        print("   - All have 'but' at index 1")
        return True

if __name__ == "__main__":
    success = verify_tokenization()