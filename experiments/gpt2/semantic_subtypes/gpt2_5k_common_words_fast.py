"""
Fast version of GPT-2 5000 Common Words Experiment with WordNet

Optimized for speed:
- Uses pre-downloaded word list
- Limits Brown corpus processing
- Adds progress indicators
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime
import requests

# Add parent directory to path
import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# NLP libraries
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

# Import existing infrastructure
from experiments.gpt2.shared.gpt2_token_validator import GPT2TokenValidator

# Import the data structures from the original
from gpt2_5k_common_words_experiment import EnrichedWord, CommonWordExtractor, analyze_organization_patterns


class FastCommonWordExtractor(CommonWordExtractor):
    """Faster version with progress indicators and optimizations"""
    
    def get_frequency_list(self, n: int = 5000) -> List[Tuple[str, int]]:
        """Get n most common words - optimized version"""
        
        # First try a local cache
        cache_file = Path("experiments/gpt2/semantic_subtypes/5k_common_words/word_frequency_cache.json")
        if cache_file.exists():
            print("Loading word frequency from cache...")
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                return [(w, i) for i, w in enumerate(cached[:n*2], 1)]
        
        # Try online list
        print("Fetching common words list...")
        try:
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                words = [w.strip() for w in response.text.split('\n') if w.strip() and w.strip().isalpha()]
                print(f"Got {len(words)} words from online list")
                
                # Cache for next time
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(words[:10000], f)
                
                return [(word, i) for i, word in enumerate(words[:n*2], 1)]
        except:
            pass
        
        # Use a minimal Brown corpus sample
        print("Using Brown corpus (limited sample)...")
        word_freq = Counter()
        
        # Only process first 100k words for speed
        for i, word in enumerate(brown.words()):
            if i > 100000:
                break
            word_lower = word.lower()
            if word_lower.isalpha() and len(word_lower) > 1:
                word_freq[word_lower] += 1
        
        return word_freq.most_common(n * 2)
    
    def extract_common_words(self, n: int = 5000) -> List[EnrichedWord]:
        """Extract with better progress tracking"""
        print(f"\n=== Extracting {n} Most Common Single-Token Words ===\n")
        
        # Get frequency list
        freq_list = self.get_frequency_list(n)
        print(f"Testing {len(freq_list)} candidate words...\n")
        
        enriched_words = []
        tested = 0
        
        for word, rank in freq_list:
            tested += 1
            
            # Progress every 100 words tested
            if tested % 100 == 0:
                print(f"Progress: Tested {tested}/{len(freq_list)} candidates, "
                      f"found {len(enriched_words)}/{n} valid words", end='\r')
            
            if len(enriched_words) >= n:
                break
            
            enriched = self.enrich_word(word, len(enriched_words) + 1)
            if enriched:
                enriched_words.append(enriched)
        
        print(f"\n\nSuccessfully extracted {len(enriched_words)} single-token words!")
        self._print_statistics(enriched_words)
        
        return enriched_words


def run_fast_5k_experiment():
    """Run optimized version of the experiment"""
    
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract common words
    print("Starting fast 5k common words experiment...")
    extractor = FastCommonWordExtractor()
    
    # Start with smaller number for testing
    target_words = 1000  # Start smaller, can increase later
    enriched_words = extractor.extract_common_words(target_words)
    
    # Save enriched words
    print(f"\nSaving {len(enriched_words)} enriched words...")
    with open(output_dir / "enriched_words.json", 'w') as f:
        json.dump([w.to_dict() for w in enriched_words], f, indent=2)
    
    # Extract activations
    print("\n=== Extracting GPT-2 Activations ===\n")
    import torch
    from transformers import GPT2Model, GPT2Tokenizer
    
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    # Extract activations
    all_activations = []
    batch_size = 50
    
    with torch.no_grad():
        for i in range(0, len(enriched_words), batch_size):
            batch = enriched_words[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(enriched_words) + batch_size - 1)//batch_size}", end='\r')
            
            for word in batch:
                inputs = tokenizer(word.word, return_tensors='pt').to(device)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Extract layer activations
                word_acts = []
                for layer_idx in range(1, 13):
                    layer_act = outputs.hidden_states[layer_idx][0, 0, :].cpu().numpy()
                    word_acts.append(layer_act)
                
                all_activations.append(np.array(word_acts))
    
    activations = np.array(all_activations)
    print(f"\n\nExtracted activations shape: {activations.shape}")
    
    # Save activations
    np.save(output_dir / "activations.npy", activations)
    
    # Basic clustering analysis
    print("\n=== Running Basic Clustering Analysis ===\n")
    from sklearn.cluster import KMeans
    
    # Just do k=5 for speed
    k = 5
    cluster_labels = {}
    
    for layer_idx in range(12):
        print(f"Clustering layer {layer_idx}...", end='\r')
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)  # Fewer inits for speed
        labels = kmeans.fit_predict(activations[:, layer_idx, :])
        cluster_labels[f'layer_{layer_idx}'] = labels
    
    # Extract paths
    print("\n\nExtracting paths...")
    paths = []
    for word_idx in range(len(enriched_words)):
        path = []
        for layer_idx in range(12):
            cluster_id = cluster_labels[f'layer_{layer_idx}'][word_idx]
            path.append(f'L{layer_idx}_C{cluster_id}')
        paths.append(path)
    
    # Save results
    results = {
        'n_words': len(enriched_words),
        'n_layers': 12,
        'k': k,
        'paths': paths,
        'cluster_labels': {k: v.tolist() for k, v in cluster_labels.items()}
    }
    
    with open(output_dir / "clustering_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Quick organization analysis
    print("\n=== Analyzing Organization Patterns ===\n")
    
    # Group by POS
    pos_groups = defaultdict(list)
    for i, word in enumerate(enriched_words):
        if word.primary_pos:
            pos_groups[word.primary_pos].append(i)
    
    print("POS-based path convergence:")
    for pos, indices in pos_groups.items():
        if len(indices) >= 10:
            # Check if they converge in final layer
            final_clusters = [paths[i][-1] for i in indices]
            unique_final = len(set(final_clusters))
            convergence = 1.0 - (unique_final - 1) / len(indices)
            print(f"  {pos}: {len(indices)} words, convergence={convergence:.3f}")
    
    print(f"\nExperiment complete! Results saved to {output_dir}")


if __name__ == "__main__":
    run_fast_5k_experiment()