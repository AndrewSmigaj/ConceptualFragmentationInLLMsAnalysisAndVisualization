#!/usr/bin/env python3
"""
Extract and save WordNet features for top 10k GPT-2 tokens.
This creates the wordnet_features.json file needed for trajectory analysis.
"""

import json
from pathlib import Path
import logging
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet

# Download WordNet if not available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_wordnet_features(base_dir: Path):
    """Extract WordNet features for top 10k tokens."""
    # Load top 10k token information
    logging.info("Loading top 10k token information...")
    with open(base_dir / "top_10k_tokens_full.json", 'r', encoding='utf-8') as f:
        token_list = json.load(f)
    
    logging.info(f"Loaded {len(token_list)} tokens")
    
    # Extract WordNet features
    wordnet_features = {}
    wordnet_count = 0
    
    for token_data in tqdm(token_list, desc="Extracting WordNet features"):
        token_id = token_data['token_id']
        token_str = token_data['token_str'].strip()
        
        # Only process alphabetic tokens that might be complete words
        if (token_data.get('is_alphabetic', False) and 
            not token_data.get('is_subword', False) and 
            len(token_str) > 2):
            
            # Get synsets for the token
            synsets = wordnet.synsets(token_str.lower())
            
            if synsets:
                # Get primary synset
                primary_synset = synsets[0]
                
                # Extract hypernyms
                hypernyms = []
                for hypernym in primary_synset.hypernyms()[:3]:  # Top 3 hypernyms
                    hypernyms.append(hypernym.lemmas()[0].name())
                
                # Extract synonyms from all synsets
                synonyms = []
                for synset in synsets[:3]:  # Consider top 3 synsets
                    for lemma in synset.lemmas()[:3]:  # Top 3 lemmas per synset
                        if lemma.name().lower() != token_str.lower():
                            synonyms.append(lemma.name())
                synonyms = list(set(synonyms))[:5]  # Unique, max 5
                
                # Get semantic category based on hypernyms
                semantic_category = get_semantic_category(primary_synset)
                
                wordnet_features[token_id] = {
                    'token': token_str,
                    'pos': primary_synset.pos(),
                    'definition': primary_synset.definition(),
                    'hypernyms': hypernyms,
                    'synonyms': synonyms,
                    'synset_count': len(synsets),
                    'semantic_category': semantic_category,
                    'lexname': primary_synset.lexname() if hasattr(primary_synset, 'lexname') else None
                }
                wordnet_count += 1
    
    logging.info(f"Extracted WordNet features for {wordnet_count} tokens out of {len(token_list)}")
    
    # Save results
    output_dir = base_dir / "clustering_results_per_layer"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "wordnet_features.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(wordnet_features, f, indent=2)
    
    logging.info(f"Saved WordNet features to {output_path}")
    
    # Also save to k=3 directory for consistency
    k3_dir = base_dir / "clustering_results_k3"
    if k3_dir.exists():
        k3_path = k3_dir / "wordnet_features.json"
        with open(k3_path, 'w', encoding='utf-8') as f:
            json.dump(wordnet_features, f, indent=2)
        logging.info(f"Also saved to {k3_path}")
    
    # Print summary statistics
    print("\n=== WordNet Feature Extraction Summary ===")
    print(f"Total tokens: {len(token_list)}")
    print(f"Tokens with WordNet data: {wordnet_count} ({wordnet_count/len(token_list)*100:.1f}%)")
    
    # POS distribution
    pos_counts = {}
    for features in wordnet_features.values():
        pos = features['pos']
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    print("\nPOS Distribution:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True):
        pos_name = {
            'n': 'Noun',
            'v': 'Verb', 
            'a': 'Adjective',
            's': 'Adjective Satellite',
            'r': 'Adverb'
        }.get(pos, pos)
        print(f"  {pos_name}: {count} ({count/wordnet_count*100:.1f}%)")
    
    # Semantic category distribution
    cat_counts = {}
    for features in wordnet_features.values():
        cat = features['semantic_category']
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    print("\nTop Semantic Categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat}: {count} ({count/wordnet_count*100:.1f}%)")


def get_semantic_category(synset):
    """Get high-level semantic category from synset."""
    # Get the highest level hypernym
    hypernym_paths = synset.hypernym_paths()
    if hypernym_paths:
        # Get the second level (first is always 'entity.n.01')
        if len(hypernym_paths[0]) > 1:
            category_synset = hypernym_paths[0][1]
            return category_synset.lemmas()[0].name()
    
    # Fallback to lexname
    if hasattr(synset, 'lexname'):
        lexname = synset.lexname()
        if lexname:
            return lexname.replace('.', '_')
    
    return 'unknown'


def main():
    base_dir = Path(__file__).parent
    
    # Check if top 10k tokens exist
    if not (base_dir / "top_10k_tokens_full.json").exists():
        logging.error("top_10k_tokens_full.json not found. Run extract_top_10k_tokens.py first.")
        return
    
    extract_wordnet_features(base_dir)


if __name__ == "__main__":
    main()