#!/usr/bin/env python3
"""
Create consistent and differentiated semantic labels for k=10 clusters.
Uses more sophisticated linguistic analysis to distinguish clusters.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import re
import os

def load_cluster_data(filepath: str) -> Dict:
    """Load the cluster data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_all_tokens(cluster_data: Dict) -> List[str]:
    """Extract all tokens from a cluster, not just common ones."""
    # For now, use common_tokens as we don't have access to all tokens
    return cluster_data.get('common_tokens', [])

def calculate_token_overlap(tokens1: List[str], tokens2: List[str]) -> float:
    """Calculate overlap between two token lists."""
    if not tokens1 or not tokens2:
        return 0.0
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0.0

def analyze_detailed_characteristics(tokens: List[str], cluster_data: Dict) -> Dict:
    """Perform detailed linguistic analysis of tokens."""
    
    # Core linguistic categories
    determiners = {' the', ' a', ' an', ' this', ' that', ' these', ' those', ' my', ' your', ' his', ' her', ' its', ' our', ' their'}
    prepositions = {' in', ' on', ' at', ' to', ' for', ' of', ' with', ' by', ' from', ' about', ' into', ' through', ' over', ' under'}
    conjunctions = {' and', ' or', ' but', ' nor', ' yet', ' so', ' for', ' as', ' if', ' when', ' while', ' because', ' since'}
    pronouns = {' he', ' she', ' it', ' they', ' we', ' you', ' I', ' me', ' him', ' her', ' them', ' us'}
    aux_verbs = {' is', ' are', ' was', ' were', ' have', ' has', ' had', ' do', ' does', ' did', ' will', ' would', ' could', ' should', ' may', ' might', ' must', ' shall'}
    modal_verbs = {' can', ' could', ' may', ' might', ' must', ' shall', ' should', ' will', ' would', ' ought'}
    
    # Common verb forms
    common_verbs = {' be', ' have', ' do', ' say', ' get', ' make', ' go', ' know', ' take', ' see', ' come', ' think', ' look', ' want', ' give', ' use', ' find', ' tell', ' ask', ' work'}
    
    # Semantic categories
    time_words = {' time', ' day', ' year', ' week', ' month', ' hour', ' minute', ' today', ' tomorrow', ' yesterday', ' now', ' then', ' when', ' soon', ' later', ' early', ' late'}
    place_words = {' place', ' home', ' house', ' room', ' street', ' city', ' country', ' world', ' area', ' region', ' location', ' building', ' office', ' school'}
    person_words = {' man', ' woman', ' person', ' people', ' child', ' boy', ' girl', ' Mr', ' Mrs', ' Miss', ' Dr', ' President'}
    body_parts = {' hand', ' head', ' eye', ' face', ' body', ' heart', ' mind', ' foot', ' arm', ' leg', ' back', ' hair'}
    
    # Initialize counts
    chars = {
        'determiners': 0,
        'prepositions': 0,
        'conjunctions': 0,
        'pronouns': 0,
        'auxiliary_verbs': 0,
        'modal_verbs': 0,
        'common_verbs': 0,
        'time_words': 0,
        'place_words': 0,
        'person_words': 0,
        'body_parts': 0,
        'punctuation': 0,
        'numeric': 0,
        'capitalized': 0,
        'space_prefixed': 0,
        'subword': 0,
        'suffix_ing': 0,
        'suffix_ed': 0,
        'suffix_ly': 0,
        'suffix_tion': 0,
        'suffix_ment': 0,
        'suffix_ness': 0,
        'suffix_er': 0,
        'suffix_est': 0,
        'suffix_plural': 0,
        'contractions': 0,
        'possessive': 0
    }
    
    for token in tokens:
        clean_token = token.strip()
        
        # Check categories
        if token in determiners:
            chars['determiners'] += 1
        if token in prepositions:
            chars['prepositions'] += 1
        if token in conjunctions:
            chars['conjunctions'] += 1
        if token in pronouns:
            chars['pronouns'] += 1
        if token in aux_verbs:
            chars['auxiliary_verbs'] += 1
        if token in modal_verbs:
            chars['modal_verbs'] += 1
        if token in common_verbs:
            chars['common_verbs'] += 1
        if token in time_words:
            chars['time_words'] += 1
        if token in place_words:
            chars['place_words'] += 1
        if token in person_words:
            chars['person_words'] += 1
        if token in body_parts:
            chars['body_parts'] += 1
            
        # Morphological patterns
        if re.match(r'^[^\w\s]+$', clean_token):
            chars['punctuation'] += 1
        if re.match(r'^\d+$', clean_token):
            chars['numeric'] += 1
        if clean_token and clean_token[0].isupper():
            chars['capitalized'] += 1
        if token.startswith(' '):
            chars['space_prefixed'] += 1
        if not token.startswith(' ') and len(clean_token) < 5:
            chars['subword'] += 1
            
        # Suffixes
        if clean_token.endswith('ing'):
            chars['suffix_ing'] += 1
        if clean_token.endswith('ed'):
            chars['suffix_ed'] += 1
        if clean_token.endswith('ly'):
            chars['suffix_ly'] += 1
        if clean_token.endswith('tion') or clean_token.endswith('sion'):
            chars['suffix_tion'] += 1
        if clean_token.endswith('ment'):
            chars['suffix_ment'] += 1
        if clean_token.endswith('ness'):
            chars['suffix_ness'] += 1
        if clean_token.endswith('er') and len(clean_token) > 3:
            chars['suffix_er'] += 1
        if clean_token.endswith('est'):
            chars['suffix_est'] += 1
        if clean_token.endswith('s') and len(clean_token) > 2:
            chars['suffix_plural'] += 1
            
        # Special patterns
        if "'" in clean_token:
            if clean_token.endswith("'s"):
                chars['possessive'] += 1
            elif "'t" in clean_token or "'ll" in clean_token or "'ve" in clean_token:
                chars['contractions'] += 1
    
    # Add pattern percentages from cluster data
    if 'pattern_percentages' in cluster_data:
        chars['has_space_pct'] = cluster_data['pattern_percentages'].get('has_space', 0)
        chars['is_alphabetic_pct'] = cluster_data['pattern_percentages'].get('is_alphabetic', 0)
        chars['is_punctuation_pct'] = cluster_data['pattern_percentages'].get('is_punctuation', 0)
        chars['is_numeric_pct'] = cluster_data['pattern_percentages'].get('is_numeric', 0)
        chars['is_uppercase_pct'] = cluster_data['pattern_percentages'].get('is_uppercase', 0)
    
    return chars

def create_differentiated_label(chars: Dict, cluster_data: Dict, existing_labels: Set[str]) -> str:
    """Create a distinctive label based on detailed characteristics."""
    
    # Check for dominant patterns
    total_tokens = len(extract_all_tokens(cluster_data))
    
    # Priority 1: Punctuation
    if chars.get('is_punctuation_pct', 0) > 50 or chars['punctuation'] / total_tokens > 0.5:
        return "Punctuation Marks"
    
    # Priority 2: Numeric
    if chars.get('is_numeric_pct', 0) > 30 or chars['numeric'] / total_tokens > 0.3:
        return "Numeric Tokens"
    
    # Priority 3: Function word categories
    function_total = (chars['determiners'] + chars['prepositions'] + 
                     chars['conjunctions'] + chars['auxiliary_verbs'])
    
    if function_total / total_tokens > 0.4:
        # Determine specific type
        if chars['determiners'] > chars['prepositions'] and chars['determiners'] > chars['conjunctions']:
            return "Determiners & Articles"
        elif chars['prepositions'] > chars['conjunctions']:
            return "Prepositions & Spatial Terms"
        elif chars['conjunctions'] > 0.1 * total_tokens:
            return "Conjunctions & Connectives"
        elif chars['auxiliary_verbs'] > 0.2 * total_tokens:
            if chars['modal_verbs'] > 0.1 * total_tokens:
                return "Modal & Auxiliary Verbs"
            else:
                return "Auxiliary Verbs"
        else:
            return "Mixed Function Words"
    
    # Priority 4: Pronouns
    if chars['pronouns'] / total_tokens > 0.3:
        return "Pronouns & References"
    
    # Priority 5: Morphological patterns
    suffix_total = (chars['suffix_ing'] + chars['suffix_ed'] + chars['suffix_ly'] + 
                   chars['suffix_tion'] + chars['suffix_ment'] + chars['suffix_ness'])
    
    if suffix_total / total_tokens > 0.4:
        # Determine dominant suffix
        if chars['suffix_ing'] > 0.2 * total_tokens:
            return "Progressive Verb Forms (-ing)"
        elif chars['suffix_ed'] > 0.2 * total_tokens:
            return "Past Tense Verbs (-ed)"
        elif chars['suffix_ly'] > 0.15 * total_tokens:
            return "Adverbs (-ly)"
        elif chars['suffix_tion'] > 0.1 * total_tokens:
            return "Nominalized Forms (-tion)"
        elif chars['suffix_plural'] > 0.3 * total_tokens:
            return "Plural Nouns"
        else:
            return "Derived Forms (Mixed Suffixes)"
    
    # Priority 6: Semantic categories
    semantic_total = (chars['time_words'] + chars['place_words'] + 
                     chars['person_words'] + chars['body_parts'])
    
    if semantic_total / total_tokens > 0.2:
        if chars['time_words'] > chars['place_words'] and chars['time_words'] > chars['person_words']:
            return "Temporal Expressions"
        elif chars['place_words'] > chars['person_words']:
            return "Spatial & Location Terms"
        elif chars['person_words'] > 0.1 * total_tokens:
            return "Person References & Titles"
        elif chars['body_parts'] > 0.1 * total_tokens:
            return "Body Parts & Physical Terms"
    
    # Priority 7: Capitalization patterns
    if chars['capitalized'] / total_tokens > 0.3:
        if chars['person_words'] > 0.1 * total_tokens:
            return "Proper Names & Titles"
        else:
            return "Capitalized Terms"
    
    # Priority 8: Contractions and possessives
    if (chars['contractions'] + chars['possessive']) / total_tokens > 0.2:
        if chars['contractions'] > chars['possessive']:
            return "Contractions"
        else:
            return "Possessive Forms"
    
    # Priority 9: Common verbs
    if chars['common_verbs'] / total_tokens > 0.3:
        return "Common Action Verbs"
    
    # Priority 10: Subwords
    if chars['subword'] / total_tokens > 0.4 and not chars['space_prefixed']:
        return "Subword Units"
    
    # Default: Try to be more specific based on layer
    layer = int(cluster_data.get('layer', 0))
    if layer <= 2:
        return "Early Layer Mixed Tokens"
    elif layer <= 6:
        return "Middle Layer Content Words"
    else:
        return "Late Layer Abstract Terms"

def find_cluster_groups_improved(data: Dict, similarity_threshold: float = 0.5) -> List[List[str]]:
    """Find groups of similar clusters with improved similarity calculation."""
    clusters = data['clusters']
    cluster_ids = list(clusters.keys())
    
    # Calculate similarity matrix
    n = len(cluster_ids)
    similarity_matrix = np.zeros((n, n))
    
    for i, id1 in enumerate(cluster_ids):
        tokens1 = extract_all_tokens(clusters[id1])
        for j, id2 in enumerate(cluster_ids):
            if i != j:
                tokens2 = extract_all_tokens(clusters[id2])
                similarity_matrix[i, j] = calculate_token_overlap(tokens1, tokens2)
    
    # Find groups using hierarchical clustering
    groups = []
    used = set()
    
    # Sort by layer to prefer grouping across layers
    cluster_ids_by_layer = sorted(cluster_ids, key=lambda x: int(x.split('_')[0][1:]))
    
    for id1 in cluster_ids_by_layer:
        if id1 in used:
            continue
            
        i = cluster_ids.index(id1)
        group = [id1]
        used.add(id1)
        
        # Find similar clusters, preferring different layers
        candidates = []
        for j, id2 in enumerate(cluster_ids):
            if id2 not in used and similarity_matrix[i, j] >= similarity_threshold:
                layer1 = int(id1.split('_')[0][1:])
                layer2 = int(id2.split('_')[0][1:])
                # Prefer clusters from different layers
                priority = abs(layer2 - layer1)
                candidates.append((similarity_matrix[i, j], priority, id2))
        
        # Sort by similarity and layer difference
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        for _, _, id2 in candidates:
            group.append(id2)
            used.add(id2)
        
        groups.append(group)
    
    return groups

def assign_differentiated_labels(data: Dict, groups: List[List[str]]) -> Dict[str, str]:
    """Assign differentiated labels to cluster groups."""
    clusters = data['clusters']
    labels = {}
    used_labels = set()
    
    # First pass: Assign labels to groups
    group_labels = {}
    
    for group_idx, group in enumerate(groups):
        # Analyze combined characteristics
        all_chars = defaultdict(float)
        total_weight = 0
        
        for cluster_id in group:
            tokens = extract_all_tokens(clusters[cluster_id])
            chars = analyze_detailed_characteristics(tokens, clusters[cluster_id])
            
            # Weight by cluster size
            weight = clusters[cluster_id]['size']
            for key, value in chars.items():
                all_chars[key] += value * weight
            total_weight += weight
        
        # Normalize
        for key in all_chars:
            all_chars[key] /= total_weight
        
        # Add layer information
        layers = [int(cid.split('_')[0][1:]) for cid in group]
        all_chars['layer'] = sum(layers) / len(layers)
        
        # Create label
        label = create_differentiated_label(dict(all_chars), clusters[group[0]], used_labels)
        
        # Make unique if necessary
        if label in used_labels:
            # Add layer range info
            min_layer = min(layers)
            max_layer = max(layers)
            if min_layer != max_layer:
                label = f"{label} (L{min_layer}-L{max_layer})"
            else:
                label = f"{label} (L{min_layer})"
        
        used_labels.add(label)
        group_labels[group_idx] = label
        
        # Assign to all clusters in group
        for cluster_id in group:
            labels[cluster_id] = label
    
    return labels

def main():
    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "llm_labels_k10", "llm_labeling_data.json")
    output_path = os.path.join(base_dir, "llm_labels_k10", "cluster_labels_k10_differentiated.json")
    
    print("Loading cluster data...")
    data = load_cluster_data(data_path)
    
    # Find similar cluster groups
    print("Finding similar cluster groups...")
    groups = find_cluster_groups_improved(data, similarity_threshold=0.5)
    
    print(f"Found {len(groups)} cluster groups")
    
    # Assign differentiated labels
    print("\nAssigning differentiated labels...")
    labels = assign_differentiated_labels(data, groups)
    
    # Add layer and cluster info
    formatted_labels = {}
    for cluster_id, label in labels.items():
        layer = int(cluster_id.split('_')[0][1:])
        cluster_num = int(cluster_id.split('_')[1][1:])
        formatted_labels[cluster_id] = {
            "label": label,
            "layer": layer,
            "cluster": cluster_num,
            "full_label": f"L{layer}: {label}"
        }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(formatted_labels, f, indent=2)
    
    print(f"\nLabels saved to {output_path}")
    
    # Print label distribution
    label_counts = Counter(label_data['label'] for label_data in formatted_labels.values())
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} clusters")
    
    # Create detailed tracking report
    tracking_path = os.path.join(base_dir, "llm_labels_k10", "differentiated_label_tracking.txt")
    with open(tracking_path, 'w') as f:
        f.write("Differentiated Label Tracking Report\n")
        f.write("===================================\n\n")
        
        # Group by label
        label_groups = defaultdict(list)
        for cluster_id, label_data in formatted_labels.items():
            label_groups[label_data['label']].append(cluster_id)
        
        for label, cluster_ids in sorted(label_groups.items()):
            f.write(f"Label: {label}\n")
            f.write(f"Clusters: {', '.join(sorted(cluster_ids))}\n")
            
            # Show sample tokens from first cluster
            first_cluster = data['clusters'][cluster_ids[0]]
            tokens = extract_all_tokens(first_cluster)[:10]
            f.write(f"Sample tokens: {tokens}\n")
            f.write(f"Cluster sizes: {[data['clusters'][cid]['size'] for cid in cluster_ids]}\n")
            f.write("\n")
    
    print(f"Tracking report saved to {tracking_path}")

if __name__ == "__main__":
    main()