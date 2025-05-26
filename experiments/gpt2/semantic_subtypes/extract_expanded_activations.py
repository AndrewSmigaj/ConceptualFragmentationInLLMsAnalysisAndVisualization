"""
Extract GPT-2 activations for the expanded word list.
"""

import os
import sys
import json
import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List
from transformers import GPT2Model, GPT2TokenizerFast
from tqdm import tqdm

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def extract_gpt2_activations(curated_data_path: str, output_dir: str):
    """Extract GPT-2 activations for all curated words."""
    
    # Load curated data
    with open(curated_data_path, 'r') as f:
        data = json.load(f)
    
    curated_words = data['curated_words']
    
    # Initialize model and tokenizer
    print("Loading GPT-2 model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = GPT2Model.from_pretrained('gpt2')
    model = model.to(device)
    model.eval()
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all words with metadata
    all_words = []
    word_to_category = {}
    
    for category, words in curated_words.items():
        for word in words:
            all_words.append(word)
            word_to_category[word] = category
    
    print(f"\nExtracting activations for {len(all_words)} words...")
    
    # Extract activations
    activations_by_layer = {i: [] for i in range(12)}
    word_list = []
    categories = []
    
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_words), batch_size)):
            batch_words = all_words[i:i+batch_size]
            
            # Tokenize with space prefix (GPT-2 convention)
            inputs = tokenizer([' ' + word for word in batch_words], 
                             return_tensors='pt', 
                             padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get all hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of 13 tensors (embeddings + 12 layers)
            
            # Extract activations for each layer
            for layer_idx in range(12):
                # Get activations from layer (layer_idx + 1 because index 0 is embeddings)
                layer_acts = hidden_states[layer_idx + 1]  # [batch, seq_len, hidden_dim]
                
                # Take first token activation (the actual word token)
                first_token_acts = layer_acts[:, 0, :].cpu().numpy()
                
                activations_by_layer[layer_idx].extend(first_token_acts)
            
            # Track words and categories
            word_list.extend(batch_words)
            categories.extend([word_to_category[w] for w in batch_words])
    
    # Convert to numpy arrays
    for layer_idx in range(12):
        activations_by_layer[layer_idx] = np.array(activations_by_layer[layer_idx])
    
    # Save results
    results = {
        'activations_by_layer': activations_by_layer,
        'words': word_list,
        'categories': categories,
        'word_to_category': word_to_category,
        'metadata': {
            'total_words': len(word_list),
            'n_layers': 12,
            'hidden_dim': 768,
            'model': 'gpt2',
            'curated_data_path': curated_data_path
        }
    }
    
    # Save as pickle
    output_file = output_path / 'gpt2_activations_expanded.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nActivations saved to: {output_file}")
    
    # Save summary
    summary = {
        'total_words': len(word_list),
        'categories': list(curated_words.keys()),
        'words_per_category': {cat: len(words) for cat, words in curated_words.items()},
        'activation_shape': f"{len(word_list)} words x 12 layers x 768 dimensions",
        'output_file': str(output_file)
    }
    
    summary_file = output_path / 'activation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results

def main():
    """Run activation extraction for expanded dataset."""
    
    # Paths
    curated_data_path = "data/gpt2_semantic_subtypes_curated_expanded.json"
    output_dir = "results/expanded_activations"
    
    # Check if curated data exists
    if not Path(curated_data_path).exists():
        print(f"ERROR: Curated data not found at {curated_data_path}")
        return
    
    # Extract activations
    extract_gpt2_activations(curated_data_path, output_dir)
    
    print("\nActivation extraction complete!")

if __name__ == "__main__":
    main()