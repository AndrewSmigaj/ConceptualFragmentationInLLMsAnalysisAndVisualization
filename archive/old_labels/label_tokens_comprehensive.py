#!/usr/bin/env python3
"""
Comprehensive token labeling for GPT-2 tokens.
Labels each token with:
- Semantic features (WordNet-based when available)
- Grammatical features (POS, morphology, syntactic patterns)
- Hypernym hierarchy (semantic ancestry)
- Token structure features (prefix, suffix, subword patterns)
"""

import json
from pathlib import Path
import logging
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from collections import defaultdict
import re

# Download required NLTK data
for resource in ['wordnet', 'averaged_perceptron_tagger', 'universal_tagset']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ComprehensiveTokenLabeler:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "token_labels"
        self.results_dir.mkdir(exist_ok=True)
        
        # Common linguistic patterns
        self.suffixes = {
            # Verb suffixes
            'ing': {'pos': 'VERB', 'aspect': 'progressive'},
            'ed': {'pos': 'VERB', 'tense': 'past'},
            's': {'pos': 'VERB', 'person': '3rd_singular'},
            'ize': {'pos': 'VERB', 'derivation': 'verb_forming'},
            'ify': {'pos': 'VERB', 'derivation': 'verb_forming'},
            
            # Noun suffixes
            'tion': {'pos': 'NOUN', 'derivation': 'nominalization'},
            'ment': {'pos': 'NOUN', 'derivation': 'nominalization'},
            'ness': {'pos': 'NOUN', 'derivation': 'quality_state'},
            'ity': {'pos': 'NOUN', 'derivation': 'quality_state'},
            'er': {'pos': 'NOUN', 'derivation': 'agent_comparative'},
            'or': {'pos': 'NOUN', 'derivation': 'agent'},
            'ist': {'pos': 'NOUN', 'derivation': 'agent_believer'},
            'ism': {'pos': 'NOUN', 'derivation': 'belief_practice'},
            
            # Adjective suffixes
            'able': {'pos': 'ADJ', 'derivation': 'capability'},
            'ible': {'pos': 'ADJ', 'derivation': 'capability'},
            'ful': {'pos': 'ADJ', 'derivation': 'characterized_by'},
            'less': {'pos': 'ADJ', 'derivation': 'without'},
            'ous': {'pos': 'ADJ', 'derivation': 'characterized_by'},
            'al': {'pos': 'ADJ', 'derivation': 'relating_to'},
            'ic': {'pos': 'ADJ', 'derivation': 'relating_to'},
            'ive': {'pos': 'ADJ', 'derivation': 'having_quality'},
            
            # Adverb suffix
            'ly': {'pos': 'ADV', 'derivation': 'manner'}
        }
        
        self.prefixes = {
            'un': {'semantic': 'negation'},
            'dis': {'semantic': 'negation'},
            'non': {'semantic': 'negation'},
            'anti': {'semantic': 'opposition'},
            'pre': {'semantic': 'before'},
            'post': {'semantic': 'after'},
            're': {'semantic': 'again'},
            'over': {'semantic': 'excess'},
            'under': {'semantic': 'insufficient'},
            'sub': {'semantic': 'below'},
            'super': {'semantic': 'above'},
            'inter': {'semantic': 'between'},
            'trans': {'semantic': 'across'},
            'mis': {'semantic': 'wrong'},
            'co': {'semantic': 'together'}
        }
        
    def label_tokens(self, tokens_path: Path):
        """Label all tokens with comprehensive linguistic features."""
        # Load tokens
        logging.info(f"Loading tokens from {tokens_path}")
        with open(tokens_path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        
        logging.info(f"Labeling {len(tokens)} tokens...")
        
        labeled_tokens = {}
        stats = defaultdict(int)
        
        for token_data in tqdm(tokens, desc="Labeling tokens"):
            token_id = token_data['token_id']
            token_str = token_data['token_str']
            token_clean = token_str.strip()
            
            # Initialize label structure
            label = {
                'token_id': token_id,
                'token': token_str,
                'token_type': token_data['token_type'],
                'is_subword': token_data.get('is_subword', False),
                'grammatical': {},
                'semantic': {},
                'morphological': {},
                'structural': {},
                'hypernyms': []
            }
            
            # Skip if punctuation or too short
            if token_data.get('is_punctuation', False):
                label['grammatical']['pos'] = 'PUNCT'
                label['semantic']['category'] = 'punctuation'
                stats['punctuation'] += 1
            elif token_data.get('is_numeric', False):
                label['grammatical']['pos'] = 'NUM'
                label['semantic']['category'] = 'numeric'
                stats['numeric'] += 1
            elif token_data.get('is_alphabetic', False):
                # Process alphabetic tokens
                self._label_alphabetic_token(token_clean, label, token_data)
                stats['alphabetic'] += 1
            else:
                label['semantic']['category'] = 'mixed/special'
                stats['other'] += 1
            
            labeled_tokens[token_id] = label
        
        # Save results
        output_path = self.results_dir / "comprehensive_token_labels.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_tokens, f, indent=2)
        
        logging.info(f"Saved labeled tokens to {output_path}")
        
        # Print statistics
        self._print_statistics(labeled_tokens, stats)
        
        return labeled_tokens
    
    def _label_alphabetic_token(self, token: str, label: dict, token_data: dict):
        """Label alphabetic tokens with linguistic features."""
        token_lower = token.lower()
        
        # Check morphological patterns
        self._check_morphology(token, label)
        
        # For complete words, get detailed analysis
        if not token_data.get('is_subword', False) and len(token) > 2:
            # Try WordNet first
            synsets = wordnet.synsets(token_lower)
            if synsets:
                self._add_wordnet_features(synsets, label, token_lower)
            else:
                # Try removing common suffixes and retry
                base_form = self._get_base_form(token_lower)
                if base_form != token_lower:
                    synsets = wordnet.synsets(base_form)
                    if synsets:
                        self._add_wordnet_features(synsets, label, base_form)
                        label['morphological']['derived_from'] = base_form
            
            # Get POS tag using NLTK
            if not label['grammatical'].get('pos'):
                try:
                    # Use universal tagset for consistency
                    pos_tags = pos_tag([token], tagset='universal')
                    if pos_tags:
                        label['grammatical']['pos'] = pos_tags[0][1]
                except:
                    pass
        
        # Structural features
        label['structural']['length'] = len(token)
        label['structural']['has_uppercase'] = any(c.isupper() for c in token)
        label['structural']['is_capitalized'] = token[0].isupper() if token else False
        
        # Set default category if not set
        if not label['semantic'].get('category'):
            if token_data.get('is_subword', False):
                label['semantic']['category'] = 'subword'
            else:
                label['semantic']['category'] = 'word_unknown'
    
    def _check_morphology(self, token: str, label: dict):
        """Check morphological patterns."""
        token_lower = token.lower()
        
        # Check suffixes
        for suffix, features in self.suffixes.items():
            if token_lower.endswith(suffix) and len(token_lower) > len(suffix) + 2:
                label['morphological']['suffix'] = suffix
                label['morphological'].update(features)
                if 'pos' in features and not label['grammatical'].get('pos'):
                    label['grammatical']['pos'] = features['pos']
                break
        
        # Check prefixes
        for prefix, features in self.prefixes.items():
            if token_lower.startswith(prefix) and len(token_lower) > len(prefix) + 2:
                label['morphological']['prefix'] = prefix
                label['morphological'].update(features)
                break
    
    def _get_base_form(self, token: str):
        """Try to get base form by removing common suffixes."""
        # Simple stemming rules
        if token.endswith('ies'):
            return token[:-3] + 'y'
        elif token.endswith('es'):
            return token[:-2]
        elif token.endswith('s') and not token.endswith('ss'):
            return token[:-1]
        elif token.endswith('ed'):
            if token.endswith('ied'):
                return token[:-3] + 'y'
            else:
                return token[:-2]
        elif token.endswith('ing'):
            if token.endswith('ying'):
                return token[:-4] + 'y'
            else:
                return token[:-3]
        return token
    
    def _add_wordnet_features(self, synsets, label: dict, word: str):
        """Add WordNet-based features."""
        primary_synset = synsets[0]
        
        # Basic WordNet features
        label['semantic']['wordnet_pos'] = primary_synset.pos()
        label['semantic']['definition'] = primary_synset.definition()
        label['semantic']['synset_count'] = len(synsets)
        
        # Map WordNet POS to universal POS
        pos_map = {
            'n': 'NOUN',
            'v': 'VERB',
            'a': 'ADJ',
            's': 'ADJ',  # satellite adjective
            'r': 'ADV'
        }
        if not label['grammatical'].get('pos'):
            label['grammatical']['pos'] = pos_map.get(primary_synset.pos(), 'X')
        
        # Get lexical category
        if hasattr(primary_synset, 'lexname'):
            label['semantic']['lexname'] = primary_synset.lexname()
        
        # Extract hypernym hierarchy
        hypernyms = []
        current = primary_synset
        depth = 0
        while current and depth < 5:  # Limit depth
            for hypernym in current.hypernyms():
                hypernym_name = hypernym.lemmas()[0].name()
                hypernyms.append({
                    'level': depth,
                    'hypernym': hypernym_name,
                    'definition': hypernym.definition()
                })
                current = hypernym
                break
            else:
                break
            depth += 1
        
        label['hypernyms'] = hypernyms
        
        # Semantic category from top-level hypernym
        if hypernyms:
            # Use second level if available (first is often 'entity')
            if len(hypernyms) > 1:
                label['semantic']['category'] = hypernyms[1]['hypernym']
            else:
                label['semantic']['category'] = hypernyms[0]['hypernym']
        else:
            label['semantic']['category'] = primary_synset.lexname() or 'unknown'
        
        # Get related words
        synonyms = []
        for synset in synsets[:3]:  # Top 3 meanings
            for lemma in synset.lemmas()[:3]:
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        if synonyms:
            label['semantic']['synonyms'] = list(set(synonyms))[:5]
    
    def _print_statistics(self, labeled_tokens: dict, stats: dict):
        """Print labeling statistics."""
        print("\n=== Token Labeling Statistics ===")
        print(f"Total tokens labeled: {len(labeled_tokens)}")
        
        # Basic type distribution
        print("\nToken types:")
        for type_name, count in stats.items():
            print(f"  {type_name}: {count} ({count/len(labeled_tokens)*100:.1f}%)")
        
        # POS distribution
        pos_counts = defaultdict(int)
        for label in labeled_tokens.values():
            pos = label['grammatical'].get('pos', 'UNKNOWN')
            pos_counts[pos] += 1
        
        print("\nPOS Distribution:")
        for pos, count in sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {pos}: {count} ({count/len(labeled_tokens)*100:.1f}%)")
        
        # Tokens with WordNet data
        wordnet_count = sum(1 for label in labeled_tokens.values() 
                           if 'wordnet_pos' in label.get('semantic', {}))
        print(f"\nTokens with WordNet data: {wordnet_count} ({wordnet_count/len(labeled_tokens)*100:.1f}%)")
        
        # Morphological patterns
        suffix_counts = defaultdict(int)
        prefix_counts = defaultdict(int)
        for label in labeled_tokens.values():
            morph = label.get('morphological', {})
            if 'suffix' in morph:
                suffix_counts[morph['suffix']] += 1
            if 'prefix' in morph:
                prefix_counts[morph['prefix']] += 1
        
        print("\nTop suffixes:")
        for suffix, count in sorted(suffix_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  -{suffix}: {count}")
        
        print("\nTop prefixes:")
        for prefix, count in sorted(prefix_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {prefix}-: {count}")
        
        # Semantic categories
        category_counts = defaultdict(int)
        for label in labeled_tokens.values():
            category = label['semantic'].get('category', 'unknown')
            category_counts[category] += 1
        
        print("\nTop semantic categories:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {category}: {count} ({count/len(labeled_tokens)*100:.1f}%)")
        
        # Save detailed statistics
        stats_output = {
            'total_tokens': len(labeled_tokens),
            'token_types': dict(stats),
            'pos_distribution': dict(pos_counts),
            'wordnet_coverage': wordnet_count,
            'suffix_distribution': dict(suffix_counts),
            'prefix_distribution': dict(prefix_counts),
            'semantic_categories': dict(category_counts)
        }
        
        stats_path = self.results_dir / "labeling_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_output, f, indent=2)
        print(f"\nDetailed statistics saved to {stats_path}")


def main():
    base_dir = Path(__file__).parent
    labeler = ComprehensiveTokenLabeler(base_dir)
    
    # Label the top 10k tokens
    tokens_path = base_dir / "top_10k_tokens_full.json"
    if not tokens_path.exists():
        logging.error(f"Token file not found: {tokens_path}")
        return
    
    labeled_tokens = labeler.label_tokens(tokens_path)
    
    print(f"\nLabeling complete! Results saved in {labeler.results_dir}")


if __name__ == "__main__":
    main()