#!/usr/bin/env python3
"""Analyze cluster structure and create consistent hierarchical labels."""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def load_labeling_data(file_path: str) -> dict:
    """Load the labeling data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_token_patterns(tokens: List[str]) -> Dict[str, any]:
    """Analyze patterns in a list of tokens."""
    patterns = {
        'has_space': [],
        'is_punctuation': [],
        'is_function_word': [],
        'is_pronoun': [],
        'is_number': [],
        'is_proper_noun': [],
        'is_prefix': [],
        'is_suffix': [],
        'is_content_word': []
    }
    
    # Common function words
    function_words = {
        'the', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
        'and', 'or', 'but', 'not', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
        'those', 'it', 'its', 'as', 'from', 'into', 'through', 'during',
        'including', 'until', 'against', 'among', 'throughout', 'despite',
        'towards', 'upon', 'concerning', 'after', 'before', 'above', 'below',
        'between', 'under', 'over', 'up', 'down', 'out', 'off', 'if', 'then',
        'else', 'when', 'where', 'why', 'how', 'all', 'some', 'no', 'nor',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'll',
        've', 're', 'd', 'm'
    }
    
    pronouns = {
        'i', 'me', 'my', 'mine', 'myself',
        'you', 'your', 'yours', 'yourself',
        'he', 'him', 'his', 'himself',
        'she', 'her', 'hers', 'herself',
        'it', 'its', 'itself',
        'we', 'us', 'our', 'ours', 'ourselves',
        'they', 'them', 'their', 'theirs', 'themselves',
        'who', 'whom', 'whose', 'which', 'what',
        'this', 'that', 'these', 'those',
        'one', 'ones', 'someone', 'anyone', 'everyone', 'nobody',
        'something', 'anything', 'everything', 'nothing'
    }
    
    for token in tokens:
        clean_token = token.strip().lower()
        
        # Check for space prefix
        if token.startswith(' '):
            patterns['has_space'].append(token)
            
        # Check for punctuation
        if any(c in token for c in '.,!?;:()[]{}"\'-/@#$%^&*+=<>|\\~`'):
            patterns['is_punctuation'].append(token)
            
        # Check for numbers
        if any(c.isdigit() for c in token):
            patterns['is_number'].append(token)
            
        # Check for pronouns
        if clean_token in pronouns:
            patterns['is_pronoun'].append(token)
            
        # Check for function words
        if clean_token in function_words:
            patterns['is_function_word'].append(token)
            
        # Check for proper nouns (capitalized)
        if token.strip() and token.strip()[0].isupper():
            patterns['is_proper_noun'].append(token)
            
        # Check for common prefixes
        if any(clean_token.startswith(prefix) for prefix in ['un', 'dis', 'pre', 'post', 'anti', 'sub', 'super', 'inter', 'trans', 'over', 'under']):
            patterns['is_prefix'].append(token)
            
        # Check for suffixes (tokens without space that are common endings)
        if not token.startswith(' ') and any(token.endswith(suffix) for suffix in ['ing', 'ed', 'er', 'est', 'ly', 'ness', 'ment', 'tion', 'sion', 'ity', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'ize', 'ise']):
            patterns['is_suffix'].append(token)
            
        # If not any of the above, likely a content word
        if not any([
            clean_token in function_words,
            clean_token in pronouns,
            any(c in token for c in '.,!?;:()[]{}"\'-/@#$%^&*+=<>|\\~`'),
            any(c.isdigit() for c in token)
        ]):
            patterns['is_content_word'].append(token)
    
    return patterns

def categorize_cluster(cluster_data: dict) -> Tuple[str, str, str]:
    """Categorize a cluster based on its token patterns."""
    tokens = cluster_data.get('common_tokens', [])
    patterns = analyze_token_patterns(tokens)
    
    # Count pattern occurrences
    pattern_counts = {k: len(v) for k, v in patterns.items()}
    total_tokens = len(tokens)
    
    # Determine primary category based on dominant patterns
    if pattern_counts['is_punctuation'] > total_tokens * 0.3:
        primary = "Punctuation"
        if any('.' in t or ',' in t for t in patterns['is_punctuation']):
            subcategory = "Sentence Markers"
        elif any('(' in t or ')' in t or '[' in t or ']' in t for t in patterns['is_punctuation']):
            subcategory = "Brackets"
        else:
            subcategory = "Other Punctuation"
    
    elif pattern_counts['is_pronoun'] > total_tokens * 0.3:
        primary = "Pronouns"
        personal = [t for t in patterns['is_pronoun'] if t.strip().lower() in ['i', 'me', 'you', 'he', 'she', 'it', 'we', 'they']]
        if len(personal) > len(patterns['is_pronoun']) * 0.5:
            subcategory = "Personal Pronouns"
        else:
            subcategory = "Other Pronouns"
    
    elif pattern_counts['is_function_word'] > total_tokens * 0.4:
        primary = "Function Words"
        # Further categorize function words
        prepositions = [t for t in patterns['is_function_word'] if t.strip().lower() in ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of']]
        articles = [t for t in patterns['is_function_word'] if t.strip().lower() in ['the', 'a', 'an']]
        conjunctions = [t for t in patterns['is_function_word'] if t.strip().lower() in ['and', 'or', 'but', 'nor']]
        
        if len(prepositions) > len(patterns['is_function_word']) * 0.3:
            subcategory = "Prepositions"
        elif len(articles) > len(patterns['is_function_word']) * 0.3:
            subcategory = "Articles"
        elif len(conjunctions) > len(patterns['is_function_word']) * 0.3:
            subcategory = "Conjunctions"
        else:
            subcategory = "Mixed Function Words"
    
    elif pattern_counts['is_suffix'] > total_tokens * 0.3:
        primary = "Morphological"
        subcategory = "Suffixes"
    
    elif pattern_counts['is_prefix'] > total_tokens * 0.2:
        primary = "Morphological"
        subcategory = "Prefixes"
    
    elif pattern_counts['is_number'] > total_tokens * 0.3:
        primary = "Content Words"
        subcategory = "Numbers"
    
    elif pattern_counts['is_proper_noun'] > total_tokens * 0.3:
        primary = "Content Words"
        subcategory = "Proper Nouns"
    
    else:
        primary = "Content Words"
        # Analyze content words more carefully
        if pattern_counts['is_content_word'] > 0:
            # Check for semantic patterns
            body_parts = [t for t in patterns['is_content_word'] if any(part in t.lower() for part in ['head', 'hand', 'eye', 'face', 'body', 'feet', 'heart', 'hair'])]
            time_words = [t for t in patterns['is_content_word'] if any(time in t.lower() for time in ['time', 'day', 'year', 'hour', 'moment', 'night', 'morning'])]
            place_words = [t for t in patterns['is_content_word'] if any(place in t.lower() for place in ['place', 'home', 'house', 'room', 'street', 'road', 'building', 'land'])]
            
            if len(body_parts) > len(patterns['is_content_word']) * 0.2:
                subcategory = "Body Parts"
            elif len(time_words) > len(patterns['is_content_word']) * 0.2:
                subcategory = "Temporal Nouns"
            elif len(place_words) > len(patterns['is_content_word']) * 0.2:
                subcategory = "Spatial Nouns"
            else:
                subcategory = "Common Nouns"
        else:
            subcategory = "Mixed Content"
    
    # Create specific description
    description = f"{subcategory} with {pattern_counts['has_space']} space-prefixed tokens"
    
    return primary, subcategory, description

def create_consistency_id(primary: str, subcategory: str, layer: int) -> str:
    """Create a consistency ID for tracking similar clusters across layers."""
    # Create abbreviated forms
    primary_abbrev = {
        "Function Words": "FW",
        "Content Words": "CW",
        "Grammatical Markers": "GM",
        "Pronouns": "PR",
        "Punctuation": "PU",
        "Morphological": "MO",
        "Discourse": "DC"
    }.get(primary, "XX")
    
    subcategory_abbrev = subcategory.replace(" ", "_").upper()[:4]
    
    return f"{primary_abbrev}_{subcategory_abbrev}"

def analyze_all_clusters(data: dict) -> dict:
    """Analyze all clusters and create hierarchical labels."""
    clusters = data.get('clusters', {})
    results = {}
    consistency_tracking = defaultdict(list)
    
    for cluster_id, cluster_data in clusters.items():
        # Extract layer number from cluster ID (e.g., "L0_C0" -> 0)
        layer = int(cluster_id.split('_')[0][1:])
        
        primary, subcategory, description = categorize_cluster(cluster_data)
        consistency_id = create_consistency_id(primary, subcategory, layer)
        
        # Track which clusters have similar consistency IDs
        consistency_tracking[consistency_id].append(cluster_id)
        
        results[cluster_id] = {
            'primary_category': primary,
            'subcategory': subcategory,
            'description': description,
            'consistency_id': consistency_id,
            'layer': layer,
            'size': cluster_data['size'],
            'sample_tokens': cluster_data['common_tokens'][:5]
        }
    
    # Add consistency information
    for cluster_id, info in results.items():
        info['related_clusters'] = consistency_tracking[info['consistency_id']]
    
    return results

def main():
    """Main analysis function."""
    # Load data
    file_path = '/mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization/experiments/gpt2/all_tokens/llm_labels_k10/llm_labeling_data.json'
    data = load_labeling_data(file_path)
    
    # Analyze all clusters
    results = analyze_all_clusters(data)
    
    # Create summary statistics
    summary = {
        'total_clusters': len(results),
        'layers': 12,
        'clusters_per_layer': 10,
        'primary_categories': Counter([r['primary_category'] for r in results.values()]),
        'subcategories': Counter([r['subcategory'] for r in results.values()]),
        'consistency_groups': len(set(r['consistency_id'] for r in results.values()))
    }
    
    # Save results
    output = {
        'metadata': {
            'analysis_type': 'hierarchical_labeling',
            'consistency_tracking': True,
            'categories': {
                'primary': ["Function Words", "Content Words", "Grammatical Markers", "Pronouns", "Punctuation", "Morphological", "Discourse"],
                'description': 'Hierarchical categorization of GPT-2 token clusters'
            }
        },
        'summary': summary,
        'cluster_labels': results
    }
    
    output_path = '/mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization/experiments/gpt2/all_tokens/llm_labels_k10/hierarchical_cluster_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print("Hierarchical Cluster Analysis Complete")
    print(f"Total clusters analyzed: {summary['total_clusters']}")
    print(f"\nPrimary categories:")
    for cat, count in summary['primary_categories'].items():
        print(f"  {cat}: {count} clusters")
    print(f"\nUnique consistency groups: {summary['consistency_groups']}")
    
    # Print sample results for first few clusters in each layer
    print("\nSample cluster labels:")
    for layer in range(3):  # Show first 3 layers
        print(f"\n  Layer {layer}:")
        for cluster in range(3):  # Show first 3 clusters
            cluster_id = f"L{layer}_C{cluster}"
            if cluster_id in results:
                r = results[cluster_id]
                print(f"    {cluster_id}: {r['primary_category']} > {r['subcategory']} ({r['consistency_id']})")
                print(f"      Tokens: {', '.join(r['sample_tokens'])}")

if __name__ == "__main__":
    main()