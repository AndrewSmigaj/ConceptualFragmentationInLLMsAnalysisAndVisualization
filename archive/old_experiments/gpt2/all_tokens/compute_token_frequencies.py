#!/usr/bin/env python3
"""
Compute GPT-2 token frequencies from Brown corpus.
Quick and simple - just count how often each token appears.
"""

import json
from collections import Counter
from pathlib import Path
import nltk
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Download Brown corpus if needed
try:
    nltk.corpus.brown.words()
except:
    print("Downloading Brown corpus...")
    nltk.download('brown')

def compute_token_frequencies():
    # Initialize GPT-2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Get Brown corpus text
    print("Loading Brown corpus...")
    from nltk.corpus import brown
    
    # Process in chunks to show progress
    token_counter = Counter()
    sentences = brown.sents()
    
    print(f"Processing {len(sentences)} sentences...")
    for sent in tqdm(sentences):
        # Join words and tokenize
        text = ' '.join(sent)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counter.update(tokens)
    
    print(f"\nTotal unique tokens: {len(token_counter)}")
    print(f"Total token occurrences: {sum(token_counter.values())}")
    
    # Get top 10k most frequent
    top_10k = token_counter.most_common(10000)
    
    # Save results
    output_dir = Path(__file__).parent
    
    # Save frequency data
    freq_data = {
        'token_frequencies': dict(top_10k),
        'total_tokens': sum(token_counter.values()),
        'unique_tokens': len(token_counter)
    }
    
    with open(output_dir / 'gpt2_token_frequencies_brown.json', 'w') as f:
        json.dump(freq_data, f)
    
    # Save just the token IDs in order
    top_10k_ids = [token_id for token_id, count in top_10k]
    with open(output_dir / 'top_10k_frequent_token_ids.json', 'w') as f:
        json.dump(top_10k_ids, f)
    
    # Print sample
    print("\nTop 20 most frequent tokens:")
    for i, (token_id, count) in enumerate(top_10k[:20]):
        token_str = tokenizer.decode([token_id])
        print(f"{i+1}. Token {token_id}: '{token_str}' (count: {count:,})")
    
    print(f"\nSaved to: {output_dir}")
    return top_10k_ids


if __name__ == "__main__":
    compute_token_frequencies()