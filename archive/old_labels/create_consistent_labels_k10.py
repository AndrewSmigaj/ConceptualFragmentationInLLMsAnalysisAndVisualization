#!/usr/bin/env python3
"""
Create consistent semantic labels for k=10 clusters across GPT-2 layers.
Identifies clusters with high token overlap and assigns them consistent labels.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import re

def load_cluster_data(filepath: str) -> Dict:
    """Load the cluster data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_tokens(cluster_data: Dict) -> Set[str]:
    """Extract all tokens from a cluster."""
    tokens = set()
    # Common tokens are the most representative
    tokens.update(cluster_data.get('common_tokens', []))
    return tokens

def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets of tokens."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def analyze_token_characteristics(tokens: List[str]) -> Dict[str, float]:
    """Analyze linguistic characteristics of a token set."""
    characteristics = {
        'function_words': 0,
        'pronouns': 0,
        'auxiliary_verbs': 0,
        'punctuation': 0,
        'content_words': 0,
        'suffixed_words': 0,
        'space_prefixed': 0
    }
    
    # Define linguistic categories
    function_words = {' the', ' a', ' an', ' and', ' or', ' but', ' in', ' on', ' at', ' to', ' for',
                     ' of', ' with', ' by', ' from', ' as', ' that', ' this', ' which', ' what'}
    pronouns = {' he', ' she', ' it', ' they', ' we', ' you', ' I', ' me', ' him', ' her', ' them',
               ' us', ' my', ' your', ' his', ' their', ' our', ' myself', ' himself', ' herself'}
    auxiliary_verbs = {' is', ' are', ' was', ' were', ' have', ' has', ' had', ' do', ' does', ' did',
                      ' will', ' would', ' could', ' should', ' may', ' might', ' must'}
    
    for token in tokens:
        if token in function_words:
            characteristics['function_words'] += 1
        if token in pronouns:
            characteristics['pronouns'] += 1
        if token in auxiliary_verbs:
            characteristics['auxiliary_verbs'] += 1
        if re.match(r'^[^\w\s]+$', token.strip()):
            characteristics['punctuation'] += 1
        if token.startswith(' '):
            characteristics['space_prefixed'] += 1
        if re.search(r'(ing|ed|ly|ness|ment|tion|er|est)$', token):
            characteristics['suffixed_words'] += 1
            
    # Normalize by total tokens
    total = len(tokens) if tokens else 1
    for key in characteristics:
        characteristics[key] /= total
        
    # Content words are those that aren't function words or punctuation
    characteristics['content_words'] = 1.0 - (characteristics['function_words'] + 
                                              characteristics['punctuation'])
    
    return characteristics

def create_label_from_characteristics(characteristics: Dict[str, float], 
                                    cluster_data: Dict) -> str:
    """Create a meaningful label based on token characteristics."""
    # Priority-based labeling
    if characteristics['punctuation'] > 0.5:
        return "Punctuation"
    
    if characteristics['function_words'] > 0.3:
        if characteristics['pronouns'] > 0.2:
            return "Function Words (Pronouns)"
        elif characteristics['auxiliary_verbs'] > 0.2:
            return "Function Words (Auxiliaries)"
        else:
            return "Core Function Words"
    
    # Check token types from cluster data
    token_types = cluster_data.get('token_types', {})
    pattern_counts = cluster_data.get('pattern_counts', {})
    
    if pattern_counts.get('is_numeric', 0) / cluster_data.get('size', 1) > 0.3:
        return "Numeric Tokens"
    
    # Check subword patterns
    subword = cluster_data.get('subword_patterns', {})
    total_subword = sum(subword.values()) - subword.get('null', 0) - subword.get('prefix_space', 0)
    
    if total_subword / cluster_data.get('size', 1) > 0.3:
        # Determine dominant suffix
        suffix_counts = {k: v for k, v in subword.items() 
                        if k not in ['null', 'prefix_space']}
        if suffix_counts:
            dominant_suffix = max(suffix_counts, key=suffix_counts.get)
            if dominant_suffix == 'suffix_ing':
                return "Verbal Forms (-ing)"
            elif dominant_suffix == 'suffix_ed':
                return "Past Tense Forms (-ed)"
            elif dominant_suffix == 'suffix_plural':
                return "Plural Forms"
            elif dominant_suffix in ['suffix_er', 'suffix_est']:
                return "Comparative/Superlative Forms"
            else:
                return "Morphological Variants"
    
    # Default to content words with more specificity
    if characteristics['content_words'] > 0.6:
        # Try to identify semantic category from common tokens
        common_tokens = cluster_data.get('common_tokens', [])[:10]
        
        # Check for semantic patterns
        body_parts = [' hand', ' head', ' eyes', ' feet', ' body', ' heart', ' hair']
        locations = [' home', ' street', ' road', ' building', ' land', ' floor', ' wall']
        temporal = [' time', ' day', ' year', ' moment', ' hours', ' night', ' morning']
        
        body_count = sum(1 for t in common_tokens if t in body_parts)
        location_count = sum(1 for t in common_tokens if t in locations)
        temporal_count = sum(1 for t in common_tokens if t in temporal)
        
        if body_count >= 3:
            return "Content Words (Body/Physical)"
        elif location_count >= 3:
            return "Content Words (Spatial/Location)"
        elif temporal_count >= 3:
            return "Content Words (Temporal)"
        else:
            return "Content Words (General Nouns)"
    
    return "Mixed Tokens"

def find_cluster_groups(data: Dict, similarity_threshold: float = 0.5) -> List[List[str]]:
    """Find groups of similar clusters across layers."""
    clusters = data['clusters']
    cluster_ids = list(clusters.keys())
    
    # Calculate similarity matrix
    n = len(cluster_ids)
    similarity_matrix = np.zeros((n, n))
    
    for i, id1 in enumerate(cluster_ids):
        tokens1 = extract_tokens(clusters[id1])
        for j, id2 in enumerate(cluster_ids):
            if i != j:
                tokens2 = extract_tokens(clusters[id2])
                similarity_matrix[i, j] = calculate_jaccard_similarity(tokens1, tokens2)
    
    # Find groups using greedy clustering
    groups = []
    used = set()
    
    for i in range(n):
        if cluster_ids[i] in used:
            continue
            
        group = [cluster_ids[i]]
        used.add(cluster_ids[i])
        
        # Find all clusters similar to this one
        for j in range(n):
            if cluster_ids[j] not in used and similarity_matrix[i, j] >= similarity_threshold:
                group.append(cluster_ids[j])
                used.add(cluster_ids[j])
        
        groups.append(group)
    
    return groups

def assign_consistent_labels(data: Dict, groups: List[List[str]]) -> Dict[str, str]:
    """Assign consistent labels to cluster groups."""
    clusters = data['clusters']
    labels = {}
    
    for group in groups:
        # Combine tokens from all clusters in the group
        all_tokens = []
        combined_characteristics = defaultdict(float)
        
        for cluster_id in group:
            tokens = list(extract_tokens(clusters[cluster_id]))
            all_tokens.extend(tokens)
            
            # Get characteristics
            chars = analyze_token_characteristics(tokens)
            for key, value in chars.items():
                combined_characteristics[key] += value
        
        # Average characteristics
        for key in combined_characteristics:
            combined_characteristics[key] /= len(group)
        
        # Use the first cluster's data for additional info
        first_cluster_data = clusters[group[0]]
        
        # Create label for the group
        label = create_label_from_characteristics(dict(combined_characteristics), 
                                                 first_cluster_data)
        
        # Assign same label to all clusters in group
        for cluster_id in group:
            labels[cluster_id] = label
    
    return labels

def main():
    # Load data
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "llm_labels_k10", "llm_labeling_data.json")
    output_path = os.path.join(base_dir, "llm_labels_k10", "cluster_labels_k10_consistent.json")
    
    print("Loading cluster data...")
    data = load_cluster_data(data_path)
    
    # Find similar cluster groups
    print("Finding similar cluster groups...")
    groups = find_cluster_groups(data, similarity_threshold=0.5)
    
    print(f"Found {len(groups)} cluster groups")
    
    # Show some group examples
    print("\nExample cluster groups:")
    for i, group in enumerate(groups[:5]):
        if len(group) > 1:
            print(f"Group {i+1}: {group}")
            # Show common tokens for first cluster in group
            tokens = list(extract_tokens(data['clusters'][group[0]]))[:10]
            print(f"  Sample tokens: {tokens}")
    
    # Assign consistent labels
    print("\nAssigning consistent labels...")
    labels = assign_consistent_labels(data, groups)
    
    # Add layer and cluster number for better visualization
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
    
    # Create tracking report
    tracking_path = os.path.join(base_dir, "llm_labels_k10", "consistent_label_tracking.txt")
    with open(tracking_path, 'w') as f:
        f.write("Consistent Label Tracking Report\n")
        f.write("================================\n\n")
        
        for i, group in enumerate(groups):
            if len(group) > 1:
                f.write(f"Group {i+1} - Label: {labels[group[0]]}\n")
                f.write(f"Clusters: {', '.join(group)}\n")
                
                # Show common tokens across group
                common_tokens = set(extract_tokens(data['clusters'][group[0]]))
                for cluster_id in group[1:]:
                    common_tokens &= extract_tokens(data['clusters'][cluster_id])
                
                f.write(f"Shared tokens ({len(common_tokens)}): {list(common_tokens)[:20]}\n")
                f.write("\n")
    
    print(f"Tracking report saved to {tracking_path}")

if __name__ == "__main__":
    main()