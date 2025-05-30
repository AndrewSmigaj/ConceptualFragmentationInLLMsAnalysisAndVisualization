#!/usr/bin/env python3
"""
Create linguistic labels for k=10 clusters using GPT-2 tokenizer to decode tokens.
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import numpy as np
from transformers import GPT2Tokenizer

# Linguistic categories
FUNCTION_WORDS = {
    'determiners': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their'],
    'prepositions': ['in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about', 'into', 'through', 'over', 'under'],
    'conjunctions': ['and', 'or', 'but', 'nor', 'yet', 'so', 'for', 'as', 'if', 'when', 'while', 'because', 'since'],
    'pronouns': ['he', 'she', 'it', 'they', 'we', 'you', 'I', 'me', 'him', 'her', 'them', 'us'],
    'auxiliaries': ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'been', 'be']
}

def load_data(base_dir: str) -> Tuple[Dict, List[int], Dict]:
    """Load clustering results and token IDs."""
    # Load cluster assignments
    with open(os.path.join(base_dir, "clustering_results_k10", "all_labels_k10.json"), 'r') as f:
        all_labels = json.load(f)
    
    # Load token IDs (assuming top 10k tokens)
    token_ids_path = os.path.join(base_dir, "top_10k_token_ids.json")
    if os.path.exists(token_ids_path):
        with open(token_ids_path, 'r') as f:
            token_ids = json.load(f)
    else:
        # Use range if file doesn't exist
        token_ids = list(range(10000))
    
    # Load cluster metadata
    with open(os.path.join(base_dir, "llm_labels_k10", "llm_labeling_data.json"), 'r') as f:
        metadata = json.load(f)
    
    return all_labels, token_ids, metadata

def get_tokens_by_cluster(all_labels: Dict, token_ids: List[int], tokenizer) -> Dict[str, List[str]]:
    """Group decoded tokens by cluster for each layer."""
    tokens_by_cluster = defaultdict(list)
    
    for layer_str, labels in all_labels.items():
        layer = int(layer_str)
        for idx, cluster in enumerate(labels):
            if idx < len(token_ids):
                token_id = token_ids[idx]
                token = tokenizer.decode([token_id])
                cluster_id = f"L{layer}_C{cluster}"
                tokens_by_cluster[cluster_id].append(token)
    
    return tokens_by_cluster

def analyze_cluster_tokens(tokens: List[str]) -> Dict[str, float]:
    """Analyze linguistic properties of tokens in a cluster."""
    if not tokens:
        return {}
    
    total = len(tokens)
    
    # Clean tokens for analysis
    clean_tokens = [t.strip().lower() for t in tokens]
    
    analysis = {
        'total_tokens': total,
        'unique_tokens': len(set(tokens)),
        
        # Function word analysis
        'determiners': sum(1 for t in clean_tokens if t in FUNCTION_WORDS['determiners']) / total,
        'prepositions': sum(1 for t in clean_tokens if t in FUNCTION_WORDS['prepositions']) / total,
        'conjunctions': sum(1 for t in clean_tokens if t in FUNCTION_WORDS['conjunctions']) / total,
        'pronouns': sum(1 for t in clean_tokens if t in FUNCTION_WORDS['pronouns']) / total,
        'auxiliaries': sum(1 for t in clean_tokens if t in FUNCTION_WORDS['auxiliaries']) / total,
        
        # Morphological patterns
        'has_space': sum(1 for t in tokens if t.startswith(' ') or t.startswith('Ġ')) / total,
        'capitalized': sum(1 for t in tokens if t and t[0].isupper()) / total,
        'punctuation': sum(1 for t in tokens if not any(c.isalnum() for c in t)) / total,
        'numeric': sum(1 for t in tokens if any(c.isdigit() for c in t)) / total,
        
        # Suffix patterns
        'ing_suffix': sum(1 for t in clean_tokens if t.endswith('ing')) / total,
        'ed_suffix': sum(1 for t in clean_tokens if t.endswith('ed')) / total,
        'ly_suffix': sum(1 for t in clean_tokens if t.endswith('ly')) / total,
        's_suffix': sum(1 for t in clean_tokens if t.endswith('s') and not t.endswith('ss')) / total,
        'er_suffix': sum(1 for t in clean_tokens if t.endswith('er')) / total,
        
        # Length patterns
        'short_tokens': sum(1 for t in clean_tokens if len(t) <= 3) / total,
        'long_tokens': sum(1 for t in clean_tokens if len(t) >= 8) / total,
    }
    
    # Calculate composite scores
    analysis['function_words'] = sum([
        analysis['determiners'], analysis['prepositions'], 
        analysis['conjunctions'], analysis['auxiliaries']
    ])
    
    analysis['morphological'] = sum([
        analysis['ing_suffix'], analysis['ed_suffix'], 
        analysis['ly_suffix'], analysis['s_suffix']
    ])
    
    return analysis

def create_cluster_label(analysis: Dict[str, float], sample_tokens: List[str]) -> str:
    """Create a descriptive label based on cluster analysis."""
    
    # High-level categories
    if analysis.get('punctuation', 0) > 0.7:
        return "Punctuation Marks"
    
    if analysis.get('numeric', 0) > 0.5:
        return "Numeric Tokens"
    
    # Function words
    if analysis.get('function_words', 0) > 0.5:
        # Find dominant function word type
        func_types = [
            ('determiners', "Determiners & Articles"),
            ('prepositions', "Prepositions & Spatial Terms"),
            ('conjunctions', "Conjunctions & Connectives"),
            ('auxiliaries', "Auxiliary & Modal Verbs"),
            ('pronouns', "Personal Pronouns")
        ]
        
        max_type = max(func_types, key=lambda x: analysis.get(x[0], 0))
        if analysis.get(max_type[0], 0) > 0.2:
            return max_type[1]
        else:
            return "Mixed Function Words"
    
    # Morphological patterns
    if analysis.get('morphological', 0) > 0.4:
        morph_types = [
            ('ing_suffix', "Progressive Forms (-ing)"),
            ('ed_suffix', "Past Tense Forms (-ed)"),
            ('ly_suffix', "Adverbial Forms (-ly)"),
            ('s_suffix', "Plural Forms"),
            ('er_suffix', "Comparative Forms (-er)")
        ]
        
        max_morph = max(morph_types, key=lambda x: analysis.get(x[0], 0))
        if analysis.get(max_morph[0], 0) > 0.2:
            return max_morph[1]
    
    # Special patterns
    if analysis.get('capitalized', 0) > 0.4:
        # Check sample for proper nouns
        cap_samples = [t for t in sample_tokens[:20] if t and t[0].isupper()]
        if any(t in ['Mr', 'Mrs', 'Dr', 'President'] for t in cap_samples):
            return "Titles & Proper Names"
        else:
            return "Capitalized Words"
    
    if analysis.get('has_space', 0) < 0.1:
        return "Subword Units"
    
    if analysis.get('short_tokens', 0) > 0.6:
        return "Short Common Words"
    
    if analysis.get('long_tokens', 0) > 0.3:
        return "Complex/Technical Terms"
    
    # Default: analyze sample tokens
    clean_samples = [t.strip().lower() for t in sample_tokens[:20]]
    
    # Check for semantic patterns
    if any(word in clean_samples for word in ['time', 'day', 'year', 'hour', 'minute']):
        return "Temporal Expressions"
    elif any(word in clean_samples for word in ['place', 'home', 'house', 'city', 'country']):
        return "Spatial/Location Terms"
    elif any(word in clean_samples for word in ['man', 'woman', 'person', 'people', 'child']):
        return "Human/Person References"
    elif any(word in clean_samples for word in ['think', 'know', 'believe', 'feel', 'understand']):
        return "Cognitive/Mental Terms"
    else:
        return "General Content Words"

def find_consistent_groups(tokens_by_cluster: Dict[str, List[str]], threshold: float = 0.5) -> List[Set[str]]:
    """Find groups of clusters with similar tokens across layers."""
    cluster_ids = list(tokens_by_cluster.keys())
    groups = []
    used = set()
    
    for id1 in cluster_ids:
        if id1 in used:
            continue
            
        group = {id1}
        used.add(id1)
        tokens1 = set(tokens_by_cluster[id1])
        
        for id2 in cluster_ids:
            if id2 not in used and id2 != id1:
                tokens2 = set(tokens_by_cluster[id2])
                
                # Calculate Jaccard similarity
                if tokens1 and tokens2:
                    similarity = len(tokens1 & tokens2) / len(tokens1 | tokens2)
                    if similarity >= threshold:
                        group.add(id2)
                        used.add(id2)
        
        groups.append(group)
    
    return groups

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print("Loading cluster data...")
    all_labels, token_ids, metadata = load_data(base_dir)
    
    print("Decoding tokens by cluster...")
    tokens_by_cluster = get_tokens_by_cluster(all_labels, token_ids, tokenizer)
    
    print("Finding consistent cluster groups...")
    groups = find_consistent_groups(tokens_by_cluster, threshold=0.3)
    
    print(f"Found {len(groups)} cluster groups")
    
    # Create labels for each group
    labels = {}
    label_counts = Counter()
    
    for group in groups:
        # Combine tokens from all clusters in group
        combined_tokens = []
        for cluster_id in group:
            combined_tokens.extend(tokens_by_cluster[cluster_id])
        
        # Analyze combined tokens
        analysis = analyze_cluster_tokens(combined_tokens)
        
        # Create label
        sample_tokens = combined_tokens[:30]
        base_label = create_cluster_label(analysis, sample_tokens)
        
        # Make unique if needed
        if label_counts[base_label] > 0:
            layers = sorted(set(int(cid.split('_')[0][1:]) for cid in group))
            if len(layers) > 1:
                base_label = f"{base_label} (L{min(layers)}-{max(layers)})"
            else:
                base_label = f"{base_label} (L{layers[0]})"
        
        label_counts[base_label] += 1
        
        # Assign to all clusters
        for cluster_id in group:
            labels[cluster_id] = base_label
    
    # Format output
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
    output_path = os.path.join(base_dir, "llm_labels_k10", "cluster_labels_k10_llm.json")
    with open(output_path, 'w') as f:
        json.dump(formatted_labels, f, indent=2)
    
    print(f"\nLabels saved to {output_path}")
    
    # Create detailed report
    report_path = os.path.join(base_dir, "llm_labels_k10", "llm_labeling_report.txt")
    with open(report_path, 'w') as f:
        f.write("GPT-2 Cluster Labeling Report\n")
        f.write("=============================\n\n")
        
        # Group by label
        label_to_clusters = defaultdict(list)
        for cluster_id, label_data in formatted_labels.items():
            label_to_clusters[label_data['label']].append(cluster_id)
        
        for label, cluster_ids in sorted(label_to_clusters.items()):
            f.write(f"\nLabel: {label}\n")
            f.write(f"Clusters: {', '.join(sorted(cluster_ids))}\n")
            
            # Sample tokens
            first_cluster = cluster_ids[0]
            samples = tokens_by_cluster[first_cluster][:20]
            f.write(f"Sample tokens: {samples}\n")
            
            # Analysis
            analysis = analyze_cluster_tokens(tokens_by_cluster[first_cluster])
            f.write(f"Function words: {analysis.get('function_words', 0):.2%}\n")
            f.write(f"Morphological: {analysis.get('morphological', 0):.2%}\n")
            f.write(f"Has space: {analysis.get('has_space', 0):.2%}\n")
    
    print(f"Report saved to {report_path}")
    
    # Print summary
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} groups")

if __name__ == "__main__":
    main()