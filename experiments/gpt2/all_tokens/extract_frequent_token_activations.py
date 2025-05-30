#!/usr/bin/env python3
"""
Extract activations for the most frequent GPT-2 tokens (based on actual usage).
"""

import numpy as np
import json
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_frequent_token_activations():
    base_dir = Path(__file__).parent
    
    # Load the frequent token IDs
    logging.info("Loading frequent token IDs...")
    with open(base_dir / "top_10k_frequent_token_ids.json", 'r') as f:
        frequent_token_ids = json.load(f)
    
    logging.info(f"Loaded {len(frequent_token_ids)} frequent token IDs")
    
    # Load metadata to understand activation structure
    metadata_path = base_dir / "activations" / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    total_tokens = metadata['vocab_size']
    num_layers = metadata['n_layers']
    hidden_size = metadata['hidden_dim']
    
    logging.info(f"Total tokens in vocabulary: {total_tokens}")
    logging.info(f"Extracting activations for top {len(frequent_token_ids)} most frequent tokens")
    
    # Create mapping from token ID to position in frequent list
    token_id_to_freq_idx = {tid: idx for idx, tid in enumerate(frequent_token_ids)}
    
    # Initialize array for frequent token activations
    frequent_activations = np.zeros((len(frequent_token_ids), num_layers, hidden_size), dtype=np.float32)
    
    # Load activations chunk by chunk
    activations_dir = base_dir / "activations"
    chunk_files = sorted(activations_dir.glob("activations_chunk_*.npy"))
    
    tokens_processed = 0
    tokens_found = 0
    
    for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
        # Load chunk
        chunk = np.load(chunk_file)
        chunk_size = chunk.shape[0]
        
        # Check which frequent tokens are in this chunk
        for local_idx in range(chunk_size):
            global_token_id = tokens_processed + local_idx
            
            if global_token_id in token_id_to_freq_idx:
                freq_idx = token_id_to_freq_idx[global_token_id]
                frequent_activations[freq_idx] = chunk[local_idx]
                tokens_found += 1
        
        tokens_processed += chunk_size
        
        # Early exit if we found all tokens
        if tokens_found == len(frequent_token_ids):
            logging.info("Found all frequent tokens!")
            break
    
    logging.info(f"Successfully extracted activations for {tokens_found}/{len(frequent_token_ids)} tokens")
    
    # Save the activations
    output_path = base_dir / "frequent_token_activations.npy"
    np.save(output_path, frequent_activations)
    logging.info(f"Saved activations to {output_path}")
    
    # Also save token information for these frequent tokens
    logging.info("Creating token information for frequent tokens...")
    
    # Load original token info
    with open(base_dir / "full_token_analysis.json", 'r', encoding='utf-8') as f:
        all_tokens = json.load(f)
    
    # Create lookup by token_id
    token_lookup = {t['token_id']: t for t in all_tokens}
    
    # Extract info for frequent tokens
    frequent_token_info = []
    for tid in frequent_token_ids:
        if tid in token_lookup:
            frequent_token_info.append(token_lookup[tid])
        else:
            logging.warning(f"Token ID {tid} not found in token analysis")
    
    # Save frequent token info
    with open(base_dir / "frequent_tokens_full.json", 'w', encoding='utf-8') as f:
        json.dump(frequent_token_info, f, indent=2)
    
    logging.info(f"Saved token information for {len(frequent_token_info)} tokens")
    
    # Print sample
    print("\nSample of frequent tokens:")
    for i in range(20):
        if i < len(frequent_token_info):
            token = frequent_token_info[i]
            print(f"{i+1}. Token {token['token_id']}: '{token['token_str']}' (type: {token['token_type']})")


if __name__ == "__main__":
    extract_frequent_token_activations()