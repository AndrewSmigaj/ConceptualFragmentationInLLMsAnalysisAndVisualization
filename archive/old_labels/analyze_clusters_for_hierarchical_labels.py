#!/usr/bin/env python3
"""
Analyze actual cluster contents to create hierarchical labels.
This script examines the real tokens in each cluster to generate appropriate labels.
"""

import json
from pathlib import Path
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_token_characteristics(tokens):
    """Analyze a list of tokens to determine their characteristics."""
    characteristics = {
        'has_articles': False,
        'has_prepositions': False,
        'has_conjunctions': False,
        'has_pronouns': False,
        'has_auxiliaries': False,
        'has_punctuation': False,
        'has_numbers': False,
        'has_nouns': False,
        'has_verbs': False,
        'has_adjectives': False,
        'has_adverbs': False,
        'has_prefixes': False,
        'has_contractions': False,
        'mostly_capitalized': False,
        'mostly_content': False,
        'mostly_function': False
    }
    
    # Common word lists for categorization
    articles = {'a', 'an', 'the', 'A', 'An', 'The'}
    prepositions = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'under', 'over', 'through'}
    conjunctions = {'and', 'or', 'but', 'so', 'yet', 'for', 'nor', 'And', 'Or', 'But'}
    pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
                'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
                'I', 'You', 'He', 'She', 'It', 'We', 'They'}
    auxiliaries = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                   'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must'}
    
    # Common endings for parts of speech
    noun_endings = ['tion', 'ment', 'ness', 'ity', 'er', 'or', 'ist', 'ism']
    verb_endings = ['ing', 'ed', 'en']
    adj_endings = ['ful', 'less', 'able', 'ible', 'ous', 'ive', 'al', 'ic']
    adv_endings = ['ly']
    
    capitalized_count = 0
    content_indicators = 0
    function_indicators = 0
    
    for token in tokens[:30]:  # Analyze top 30 tokens
        # Clean token (remove leading space if present)
        clean_token = token.strip().lower()
        original_token = token.strip()
        
        # Check categories
        if clean_token in articles:
            characteristics['has_articles'] = True
            function_indicators += 1
        if clean_token in prepositions:
            characteristics['has_prepositions'] = True
            function_indicators += 1
        if clean_token in conjunctions:
            characteristics['has_conjunctions'] = True
            function_indicators += 1
        if clean_token in pronouns:
            characteristics['has_pronouns'] = True
            function_indicators += 1
        if clean_token in auxiliaries:
            characteristics['has_auxiliaries'] = True
            function_indicators += 1
        
        # Check punctuation
        if any(c in token for c in '.,!?;:-"\''''):
            characteristics['has_punctuation'] = True
        
        # Check numbers
        if any(c.isdigit() for c in token):
            characteristics['has_numbers'] = True
        
        # Check capitalization
        if original_token and original_token[0].isupper():
            capitalized_count += 1
        
        # Check word endings for content words
        for ending in noun_endings:
            if clean_token.endswith(ending):
                characteristics['has_nouns'] = True
                content_indicators += 1
        for ending in verb_endings:
            if clean_token.endswith(ending):
                characteristics['has_verbs'] = True
                content_indicators += 1
        for ending in adj_endings:
            if clean_token.endswith(ending):
                characteristics['has_adjectives'] = True
                content_indicators += 1
        for ending in adv_endings:
            if clean_token.endswith(ending):
                characteristics['has_adverbs'] = True
                content_indicators += 1
        
        # Check prefixes
        if any(clean_token.startswith(prefix) for prefix in ['un', 're', 'dis', 'pre', 'non', 'anti', 'de', 'over', 'under']):
            characteristics['has_prefixes'] = True
        
        # Check contractions
        if "'" in token or "'" in token:
            characteristics['has_contractions'] = True
    
    # Determine overall characteristics
    characteristics['mostly_capitalized'] = capitalized_count > len(tokens[:30]) * 0.5
    characteristics['mostly_function'] = function_indicators > content_indicators
    characteristics['mostly_content'] = content_indicators > function_indicators
    
    return characteristics


def determine_primary_category(characteristics, tokens):
    """Determine the primary category based on token characteristics."""
    # Count different types
    function_count = sum([
        characteristics['has_articles'],
        characteristics['has_prepositions'],
        characteristics['has_conjunctions'],
        characteristics['has_auxiliaries']
    ])
    
    content_count = sum([
        characteristics['has_nouns'],
        characteristics['has_verbs'],
        characteristics['has_adjectives'],
        characteristics['has_adverbs']
    ])
    
    # Decision logic
    if characteristics['has_punctuation'] and len([t for t in tokens[:10] if any(c in t for c in '.,!?;:-')]) > 5:
        return "Punctuation"
    elif characteristics['has_pronouns'] and len([t for t in tokens[:10] if t.strip().lower() in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her']]) > 3:
        return "Pronouns"
    elif characteristics['has_prefixes'] or characteristics['has_contractions']:
        return "Morphological"
    elif characteristics['mostly_capitalized']:
        return "Discourse"
    elif function_count >= 2 and characteristics['mostly_function']:
        return "Function Words"
    elif content_count >= 2 and characteristics['mostly_content']:
        return "Content Words"
    elif function_count >= content_count:
        return "Grammatical Markers"
    else:
        return "Content Words"


def analyze_clusters():
    """Analyze all clusters and generate hierarchical labels."""
    base_dir = Path(__file__).parent
    
    # Load cluster data
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    analysis_results = {}
    
    for layer in range(12):
        layer_key = f"layer_{layer}"
        analysis_results[layer_key] = {}
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                tokens = cluster_info["common_tokens"]
                
                # Analyze token characteristics
                characteristics = analyze_token_characteristics(tokens)
                
                # Determine primary category
                primary = determine_primary_category(characteristics, tokens)
                
                # Store analysis
                analysis_results[layer_key][cluster_key] = {
                    "tokens": tokens[:10],
                    "characteristics": characteristics,
                    "primary_category": primary,
                    "size": cluster_info["size"],
                    "percentage": cluster_info["percentage"]
                }
    
    # Save analysis results
    output_path = base_dir / "llm_labels_k10" / "cluster_analysis_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)
    
    logging.info(f"Saved cluster analysis to {output_path}")
    
    # Print summary
    print("\nCluster Analysis Summary:")
    print("=" * 60)
    
    category_counts = defaultdict(int)
    for layer_data in analysis_results.values():
        for cluster_data in layer_data.values():
            category_counts[cluster_data["primary_category"]] += 1
    
    print("\nPrimary Category Distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} clusters ({count/120*100:.1f}%)")
    
    # Show examples from each layer
    print("\nExamples by Layer:")
    for layer in range(12):
        layer_key = f"layer_{layer}"
        print(f"\nLayer {layer}:")
        for cluster_idx in range(min(3, 10)):  # Show first 3 clusters
            cluster_key = f"L{layer}_C{cluster_idx}"
            if cluster_key in analysis_results[layer_key]:
                data = analysis_results[layer_key][cluster_key]
                tokens_preview = ", ".join(data["tokens"][:5])
                print(f"  {cluster_key}: {data['primary_category']} - [{tokens_preview}]")


if __name__ == "__main__":
    analyze_clusters()