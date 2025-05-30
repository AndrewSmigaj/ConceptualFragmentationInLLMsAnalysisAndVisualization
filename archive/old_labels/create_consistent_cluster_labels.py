#!/usr/bin/env python3
"""Create consistent hierarchical labels for GPT-2 token clusters."""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

def load_labeling_data(file_path: str) -> dict:
    """Load the labeling data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_token_type(token: str) -> str:
    """Determine the linguistic type of a token."""
    clean = token.strip().lower()
    
    # Punctuation
    if any(c in token for c in '.,!?;:()[]{}"\'-/@#$%^&*+=<>|\\~`'):
        return "punctuation"
    
    # Numbers
    if any(c.isdigit() for c in token):
        return "numeric"
    
    # Function words lists
    articles = {'the', 'a', 'an'}
    prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'into', 
                    'through', 'during', 'until', 'against', 'among', 'throughout', 
                    'upon', 'after', 'before', 'above', 'below', 'between', 'under', 'over'}
    conjunctions = {'and', 'or', 'but', 'nor', 'yet', 'so', 'for', 'if', 'then', 'else', 
                    'when', 'where', 'while', 'because', 'since', 'although', 'though'}
    auxiliaries = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                   'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
                   'may', 'might', 'must', 'can', 'shall', "'ll", "'d", "'ve", "'re", "'s"}
    pronouns = {'i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself',
                'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves',
                'they', 'them', 'their', 'theirs', 'themselves', 'who', 'whom',
                'whose', 'which', 'what', 'this', 'that', 'these', 'those'}
    determiners = {'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 
                   'its', 'our', 'their', 'some', 'any', 'no', 'all', 'both', 
                   'either', 'neither', 'each', 'every', 'few', 'many', 'much', 
                   'several', 'such'}
    
    # Morphological markers
    if not token.startswith(' ') and len(token) <= 4:
        if token in ['ing', 'ed', 'er', 'est', 'ly', 's', 'es', "'s", "'t", 'n\'t']:
            return "suffix"
        if token in ['un', 'dis', 'pre', 'post', 'anti', 'sub', 'super', 'inter', 'trans']:
            return "prefix"
    
    # Check against function word categories
    if clean in articles:
        return "article"
    if clean in prepositions:
        return "preposition"
    if clean in conjunctions:
        return "conjunction"
    if clean in auxiliaries:
        return "auxiliary"
    if clean in pronouns:
        return "pronoun"
    if clean in determiners:
        return "determiner"
    
    # Content words by semantic field
    body_parts = ['head', 'hand', 'eye', 'face', 'body', 'feet', 'heart', 'hair', 
                  'arm', 'leg', 'finger', 'mouth', 'nose', 'ear', 'tooth', 'teeth']
    time_words = ['time', 'day', 'year', 'hour', 'moment', 'night', 'morning', 
                  'evening', 'week', 'month', 'minute', 'second', 'today', 'tomorrow']
    place_words = ['place', 'home', 'house', 'room', 'street', 'road', 'building', 
                   'land', 'city', 'country', 'world', 'area', 'space', 'location']
    people_words = ['man', 'woman', 'person', 'people', 'child', 'children', 'boy', 
                    'girl', 'father', 'mother', 'friend', 'family', 'group']
    
    if any(part in clean for part in body_parts):
        return "body_part"
    if any(time in clean for time in time_words):
        return "temporal"
    if any(place in clean for place in place_words):
        return "spatial"
    if any(person in clean for person in people_words):
        return "person"
    
    # Default to content word
    return "content"

def analyze_cluster_composition(tokens: List[str]) -> Dict[str, float]:
    """Analyze the composition of token types in a cluster."""
    type_counts = Counter(get_token_type(token) for token in tokens)
    total = len(tokens)
    
    return {token_type: count / total for token_type, count in type_counts.items()}

def create_cluster_label(cluster_id: str, cluster_data: dict, all_clusters: dict) -> dict:
    """Create a consistent label for a cluster based on its composition."""
    tokens = cluster_data.get('common_tokens', [])
    composition = analyze_cluster_composition(tokens)
    
    # Find dominant type (at least 30% of tokens)
    dominant_types = [(t, p) for t, p in composition.items() if p >= 0.3]
    dominant_types.sort(key=lambda x: x[1], reverse=True)
    
    if not dominant_types:
        # No clear dominant type, analyze further
        dominant_types = [(t, p) for t, p in composition.items() if p >= 0.2]
        dominant_types.sort(key=lambda x: x[1], reverse=True)
    
    # Determine primary and subcategory
    if dominant_types:
        primary_type = dominant_types[0][0]
        
        # Map to hierarchical categories
        if primary_type in ['article', 'preposition', 'conjunction', 'auxiliary', 'determiner']:
            primary = "Function Words"
            subcategory = primary_type.title() + "s"
        elif primary_type == 'pronoun':
            primary = "Pronouns"
            subcategory = "Personal" if any(p in [t.strip().lower() for t in tokens[:10]] 
                                           for p in ['i', 'you', 'he', 'she', 'it', 'we', 'they']) else "Other"
        elif primary_type == 'punctuation':
            primary = "Punctuation"
            if any(p in tokens for p in ['.', '!', '?']):
                subcategory = "Sentence-final"
            elif any(p in tokens for p in [',', ';', ':']):
                subcategory = "Clause-separating"
            else:
                subcategory = "Other"
        elif primary_type in ['suffix', 'prefix']:
            primary = "Morphological"
            subcategory = primary_type.title() + "es"
        elif primary_type == 'numeric':
            primary = "Content Words"
            subcategory = "Numbers"
        elif primary_type in ['body_part', 'temporal', 'spatial', 'person']:
            primary = "Content Words"
            subcategory = primary_type.replace('_', ' ').title() + " Nouns"
        else:
            primary = "Content Words"
            # Check if mostly proper nouns (capitalized)
            capitalized = [t for t in tokens if t.strip() and t.strip()[0].isupper()]
            if len(capitalized) / len(tokens) > 0.3:
                subcategory = "Proper Nouns"
            else:
                subcategory = "Common Nouns"
    else:
        # Mixed cluster
        primary = "Mixed"
        subcategory = "Heterogeneous"
    
    # Create consistency ID based on content similarity across layers
    layer = int(cluster_id.split('_')[0][1:])
    
    # For early layers (0-3), focus on surface features
    # For middle layers (4-7), focus on grammatical function
    # For late layers (8-11), focus on semantic content
    
    if layer <= 3:
        consistency_base = f"{primary[:2]}_{subcategory[:4]}_SURF"
    elif layer <= 7:
        consistency_base = f"{primary[:2]}_{subcategory[:4]}_GRAM"
    else:
        consistency_base = f"{primary[:2]}_{subcategory[:4]}_SEM"
    
    # Add distinguisher if needed based on token overlap with other clusters
    distinguisher = ""
    token_set = set(tokens[:10])  # Use top 10 tokens for comparison
    
    # Check for similar clusters in the same layer
    for other_id, other_data in all_clusters.items():
        if other_id != cluster_id and other_id.startswith(f"L{layer}_"):
            other_tokens = set(other_data.get('common_tokens', [])[:10])
            overlap = len(token_set & other_tokens) / min(len(token_set), len(other_tokens))
            if overlap > 0.5:
                distinguisher = f"_{cluster_id.split('_')[1]}"
                break
    
    consistency_id = consistency_base + distinguisher
    
    # Create description
    token_stats = cluster_data.get('pattern_percentages', {})
    has_space = token_stats.get('has_space', 0)
    is_alpha = token_stats.get('is_alphabetic', 0)
    
    description = f"{subcategory} ({has_space:.0f}% space-prefixed, {is_alpha:.0f}% alphabetic)"
    
    return {
        'primary_category': primary,
        'subcategory': subcategory,
        'description': description,
        'consistency_id': consistency_id,
        'layer': layer,
        'composition': composition,
        'dominant_type': primary_type if dominant_types else 'mixed'
    }

def main():
    """Create consistent labels for all clusters."""
    # Load data
    file_path = '/mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization/experiments/gpt2/all_tokens/llm_labels_k10/llm_labeling_data.json'
    data = load_labeling_data(file_path)
    
    clusters = data.get('clusters', {})
    
    # Create labels for all clusters
    labels = {}
    for cluster_id, cluster_data in clusters.items():
        label = create_cluster_label(cluster_id, cluster_data, clusters)
        label['size'] = cluster_data['size']
        label['sample_tokens'] = cluster_data['common_tokens'][:10]
        labels[cluster_id] = label
    
    # Track consistency groups
    consistency_groups = defaultdict(list)
    for cluster_id, label in labels.items():
        consistency_groups[label['consistency_id']].append(cluster_id)
    
    # Add related clusters to each label
    for cluster_id, label in labels.items():
        label['related_clusters'] = consistency_groups[label['consistency_id']]
    
    # Create summary
    primary_counts = Counter(l['primary_category'] for l in labels.values())
    subcategory_counts = Counter(l['subcategory'] for l in labels.values())
    
    # Output results
    output = {
        'metadata': {
            'analysis_type': 'consistent_hierarchical_labeling',
            'total_clusters': len(labels),
            'layers': 12,
            'clusters_per_layer': 10,
            'labeling_strategy': {
                'early_layers': 'Surface features (0-3)',
                'middle_layers': 'Grammatical function (4-7)',
                'late_layers': 'Semantic content (8-11)'
            }
        },
        'summary': {
            'primary_categories': dict(primary_counts),
            'subcategories': dict(subcategory_counts),
            'consistency_groups': len(consistency_groups),
            'largest_consistency_group': max(len(v) for v in consistency_groups.values())
        },
        'cluster_labels': labels,
        'consistency_tracking': {
            cid: {
                'clusters': clusters,
                'layers': sorted(set(int(c.split('_')[0][1:]) for c in clusters)),
                'description': labels[clusters[0]]['primary_category'] + ' > ' + labels[clusters[0]]['subcategory']
            }
            for cid, clusters in consistency_groups.items()
        }
    }
    
    # Save results
    output_path = '/mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization/experiments/gpt2/all_tokens/llm_labels_k10/consistent_cluster_labels.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print analysis summary
    print("Consistent Cluster Labeling Complete\n")
    print(f"Total clusters: {len(labels)}")
    print(f"Consistency groups: {len(consistency_groups)}")
    print(f"Largest consistency group: {output['summary']['largest_consistency_group']} clusters\n")
    
    print("Primary Categories:")
    for cat, count in primary_counts.most_common():
        print(f"  {cat}: {count} clusters ({count/len(labels)*100:.1f}%)")
    
    print("\nTop Subcategories:")
    for subcat, count in subcategory_counts.most_common(10):
        print(f"  {subcat}: {count} clusters")
    
    print("\nExample labels by layer:")
    for layer in [0, 4, 8, 11]:
        print(f"\nLayer {layer}:")
        for i in range(3):
            cluster_id = f"L{layer}_C{i}"
            if cluster_id in labels:
                l = labels[cluster_id]
                print(f"  {cluster_id}: {l['primary_category']} > {l['subcategory']} [{l['consistency_id']}]")
                print(f"    Tokens: {', '.join(l['sample_tokens'][:5])}")

if __name__ == "__main__":
    main()