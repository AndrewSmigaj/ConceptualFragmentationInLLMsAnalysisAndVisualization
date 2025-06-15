#!/usr/bin/env python3
"""
Run clustering on frequent tokens using existing infrastructure.
Simply points the existing clustering code to use frequent_token_activations.npy
"""

import shutil
from pathlib import Path
import json

# Import the existing clustering code
from cluster_top_10k_k3 import FixedKClusterer

def main():
    base_dir = Path(__file__).parent
    
    # Temporarily copy frequent token activations to where the existing code expects them
    frequent_path = base_dir / "frequent_token_activations.npy"
    expected_path = base_dir / "top_10k_activations.npy"
    
    # Backup existing file if it exists
    if expected_path.exists():
        shutil.move(expected_path, base_dir / "top_10k_activations_old.npy")
    
    # Copy frequent token activations
    shutil.copy2(frequent_path, expected_path)
    
    # Also update the token info files
    shutil.copy2(base_dir / "frequent_tokens_full.json", base_dir / "top_10k_tokens_full.json")
    shutil.copy2(base_dir / "top_10k_frequent_token_ids.json", base_dir / "top_10k_token_ids.json")
    
    try:
        # Run clustering with k=5 using existing code
        clusterer = FixedKClusterer(base_dir, k=5)
        clusterer.run_clustering()
        
        # Add note to results about using frequent tokens
        results_dir = base_dir / "clustering_results_k5"
        note_path = results_dir / "NOTE_FREQUENT_TOKENS.txt"
        with open(note_path, 'w') as f:
            f.write("This clustering was performed on the 10,000 most FREQUENT GPT-2 tokens\n")
            f.write("based on Brown corpus analysis, NOT the first 10k tokens by ID.\n")
            f.write("This gives us meaningful common words instead of just punctuation.\n")
    
    finally:
        # Restore original files
        if (base_dir / "top_10k_activations_old.npy").exists():
            shutil.move(base_dir / "top_10k_activations_old.npy", expected_path)

if __name__ == "__main__":
    main()