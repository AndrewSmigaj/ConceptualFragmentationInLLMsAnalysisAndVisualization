#!/usr/bin/env python3
"""
Extract top 10k most common tokens from GPT-2 vocabulary for more manageable analysis.
"""

import json
import numpy as np
from pathlib import Path
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_top_10k_tokens():
    base_dir = Path(__file__).parent
    
    # Load full token analysis
    logging.info("Loading full token analysis...")
    with open(base_dir / "full_token_analysis.json", 'r', encoding='utf-8') as f:
        all_tokens = json.load(f)
    
    logging.info(f"Total tokens: {len(all_tokens)}")
    
    # Score tokens by importance (lower token_id = more common in training)
    # GPT-2 vocabulary is roughly ordered by frequency
    # First ~10k tokens contain most common words, punctuation, and important subwords
    
    # Sort by token_id to get most common tokens
    sorted_tokens = sorted(all_tokens, key=lambda x: x['token_id'])
    top_10k = sorted_tokens[:10000]
    
    # Analyze composition
    type_counts = Counter()
    morph_counts = Counter()
    has_space_count = 0
    
    for token in top_10k:
        type_counts[token['token_type']] += 1
        if token.get('has_leading_space', False):
            has_space_count += 1
        
        # Count morphological patterns
        token_str = token['token_str'].strip()
        if token_str.endswith('ing'):
            morph_counts['suffix_ing'] += 1
        elif token_str.endswith('ed'):
            morph_counts['suffix_ed'] += 1
        elif token_str.endswith('ly'):
            morph_counts['suffix_ly'] += 1
        elif token_str.endswith('er'):
            morph_counts['suffix_er'] += 1
        elif token_str.endswith('s') and len(token_str) > 2:
            morph_counts['suffix_plural'] += 1
    
    # Print analysis
    logging.info("\nTop 10k tokens composition:")
    for token_type, count in type_counts.most_common():
        logging.info(f"  {token_type}: {count} ({count/100:.1f}%)")
    
    logging.info(f"\nTokens with leading space: {has_space_count} ({has_space_count/100:.1f}%)")
    
    logging.info("\nMorphological patterns:")
    for pattern, count in morph_counts.most_common(10):
        logging.info(f"  {pattern}: {count}")
    
    # Extract just the token IDs for activation filtering
    top_10k_ids = [t['token_id'] for t in top_10k]
    
    # Save the token IDs
    output_path = base_dir / "top_10k_token_ids.json"
    with open(output_path, 'w') as f:
        json.dump(top_10k_ids, f)
    
    logging.info(f"\nSaved top 10k token IDs to {output_path}")
    
    # Also save full token info for analysis
    with open(base_dir / "top_10k_tokens_full.json", 'w') as f:
        json.dump(top_10k, f, indent=2)
    
    # Extract activations for just these tokens
    logging.info("\nExtracting activations for top 10k tokens...")
    
    # Load all activation chunks and filter
    activations_dir = base_dir / "activations"
    top_10k_activations = []
    
    # Load each chunk
    chunk_files = sorted(activations_dir.glob("activations_chunk_*.npy"))
    
    tokens_processed = 0
    for chunk_file in chunk_files:
        chunk = np.load(chunk_file)
        
        # Determine token range for this chunk
        chunk_size = chunk.shape[0]
        start_idx = tokens_processed
        end_idx = tokens_processed + chunk_size
        
        # Find which top 10k tokens are in this range
        tokens_in_chunk = [tid for tid in top_10k_ids if start_idx <= tid < end_idx]
        
        if tokens_in_chunk:
            # Extract activations for these tokens
            local_indices = [tid - start_idx for tid in tokens_in_chunk]
            chunk_activations = chunk[local_indices]
            top_10k_activations.append(chunk_activations)
        
        tokens_processed = end_idx
    
    # Concatenate all activations
    if top_10k_activations:
        final_activations = np.vstack(top_10k_activations)
        logging.info(f"Final activations shape: {final_activations.shape}")
        
        # Save as single file
        np.save(base_dir / "top_10k_activations.npy", final_activations)
        logging.info(f"Saved top 10k activations to top_10k_activations.npy")
    else:
        logging.error("No activations found for top 10k tokens!")
    
    return top_10k_ids


if __name__ == "__main__":
    extract_top_10k_tokens()
