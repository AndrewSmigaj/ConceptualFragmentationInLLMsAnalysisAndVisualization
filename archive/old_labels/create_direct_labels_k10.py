#!/usr/bin/env python3
"""
Create labels for k=10 clusters through direct analysis.
This script generates the labels based on analyzing the token patterns.
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


def analyze_and_label_clusters():
    """Analyze clusters and assign consistent semantic labels."""
    base_dir = Path(__file__).parent
    
    # Load analysis data
    with open(base_dir / "llm_labels_k10" / "cluster_analysis_k10.json", 'r') as f:
        analysis_data = json.load(f)
    
    # Load full cluster data for detailed analysis
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Define consistent labels based on token patterns
    # I'll analyze the tokens to identify consistent groups across layers
    
    # First, let's identify cluster signatures (unique token patterns)
    cluster_signatures = {}
    for cluster_key, data in analysis_data["clusters"].items():
        # Create a signature based on top tokens
        top_tokens = data["top_50_tokens"][:20]
        # Normalize tokens for comparison
        normalized = [t.strip().lower() for t in top_tokens]
        cluster_signatures[cluster_key] = {
            'tokens': set(normalized),
            'raw_tokens': top_tokens,
            'layer': data['layer'],
            'size': data['size']
        }
    
    # Find groups of similar clusters across layers
    cluster_groups = []
    processed = set()
    
    for key1, sig1 in cluster_signatures.items():
        if key1 in processed:
            continue
            
        group = [key1]
        processed.add(key1)
        
        for key2, sig2 in cluster_signatures.items():
            if key2 in processed or key2 == key1:
                continue
                
            # Calculate overlap
            overlap = len(sig1['tokens'] & sig2['tokens'])
            if overlap >= 10:  # At least 10 common tokens
                group.append(key2)
                processed.add(key2)
        
        if len(group) > 1:
            cluster_groups.append(group)
    
    # Assign labels based on analysis
    labels = {}
    
    # Analyze each cluster and assign appropriate label
    for cluster_key, data in cluster_data["clusters"].items():
        tokens = data["common_tokens"][:50]
        
        # Detailed token analysis
        label = analyze_tokens_for_label(tokens, cluster_key, analysis_data["clusters"][cluster_key])
        labels[cluster_key] = label
    
    # Ensure consistency for similar clusters
    for group in cluster_groups:
        # Find the most appropriate label for the group
        group_tokens = []
        for cluster_key in group:
            group_tokens.extend(cluster_data["clusters"][cluster_key]["common_tokens"][:20])
        
        # Determine group label
        group_label = determine_group_label(group_tokens, group)
        
        # Apply to all clusters in group
        for cluster_key in group:
            labels[cluster_key] = group_label
    
    return labels


def analyze_tokens_for_label(tokens, cluster_key, analysis_data):
    """Analyze tokens to determine appropriate semantic label."""
    
    # Common word categories
    function_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'no', 'just', 'him', 'know', 'take'
    }
    
    pronouns = {'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 
                'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
                'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves'}
    
    prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 
                    'through', 'over', 'under', 'between', 'among', 'into', 'onto', 'upon'}
    
    auxiliaries = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
                   'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
                   'shall', 'should', 'may', 'might', 'can', 'could', 'must'}
    
    determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your',
                   'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every',
                   'all', 'both', 'half', 'either', 'neither', 'each', 'much', 'many'}
    
    conjunctions = {'and', 'or', 'but', 'nor', 'for', 'yet', 'so', 'because', 'since',
                    'unless', 'although', 'while', 'if', 'when', 'where', 'that', 'whether'}
    
    # Clean tokens for analysis
    clean_tokens = [t.strip().lower() for t in tokens[:30]]
    
    # Count categories
    counts = defaultdict(int)
    
    for token in clean_tokens:
        if token in pronouns:
            counts['pronouns'] += 1
        if token in prepositions:
            counts['prepositions'] += 1
        if token in auxiliaries:
            counts['auxiliaries'] += 1
        if token in determiners:
            counts['determiners'] += 1
        if token in conjunctions:
            counts['conjunctions'] += 1
        if token in function_words:
            counts['function_words'] += 1
        
        # Check for punctuation
        if len(token) == 1 and not token.isalnum():
            counts['punctuation'] += 1
        
        # Check for content words (nouns/verbs/adjectives)
        if token not in function_words and token.isalpha() and len(token) > 2:
            counts['content_words'] += 1
    
    # Determine label based on dominant patterns
    if counts['punctuation'] >= 5:
        return "Punctuation Marks"
    elif counts['pronouns'] >= 5:
        return "Personal Pronouns"
    elif counts['prepositions'] >= 5:
        return "Prepositions"
    elif counts['auxiliaries'] >= 5:
        return "Auxiliary Verbs"
    elif counts['determiners'] >= 5:
        return "Determiners"
    elif counts['conjunctions'] >= 3:
        return "Conjunctions"
    elif counts['function_words'] >= 10:
        return "Core Function Words"
    elif counts['content_words'] >= 15:
        # Further analyze content words
        if any(t.endswith('ing') for t in clean_tokens[:10]):
            return "Progressive Forms"
        elif any(t.endswith('ed') for t in clean_tokens[:10]):
            return "Past Forms"
        elif any(t.endswith('ly') for t in clean_tokens[:10]):
            return "Adverbs"
        else:
            return "Content Words"
    else:
        # Check for specific patterns
        if any(t.startswith('Ġ') for t in tokens[:10]):
            return "Space-Prefixed Tokens"
        elif sum(1 for t in tokens[:10] if len(t) <= 2) >= 5:
            return "Short Tokens"
        elif sum(1 for t in clean_tokens[:10] if t.isdigit()) >= 3:
            return "Numeric Tokens"
        else:
            return "Mixed Tokens"


def determine_group_label(group_tokens, cluster_keys):
    """Determine the best label for a group of similar clusters."""
    
    # Get unique tokens
    token_counts = defaultdict(int)
    for token in group_tokens:
        token_counts[token.strip().lower()] += 1
    
    # Get most common tokens
    common_tokens = [t for t, c in sorted(token_counts.items(), key=lambda x: x[1], reverse=True) if c >= 2][:30]
    
    # Use the same analysis logic
    return analyze_tokens_for_label(common_tokens, cluster_keys[0], {})


def main():
    """Generate and save direct analysis labels."""
    base_dir = Path(__file__).parent
    
    # Generate labels
    labels = analyze_and_label_clusters()
    
    # Create results structure
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "direct_analysis",
            "k": 10,
            "total_clusters": 120,
            "method": "direct_linguistic_analysis",
            "description": "Labels generated through direct linguistic analysis of token patterns"
        },
        "labels": {}
    }
    
    # Load cluster data for metadata
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Organize by layer
    for layer in range(12):
        layer_key = f"layer_{layer}"
        results["labels"][layer_key] = {}
        
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in labels and cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                results["labels"][layer_key][cluster_key] = {
                    "label": labels[cluster_key],
                    "description": f"Cluster containing tokens like: {', '.join(repr(t) for t in cluster_info['common_tokens'][:5])}",
                    "size": cluster_info["size"],
                    "percentage": cluster_info["percentage"]
                }
    
    # Save results
    output_path = base_dir / "llm_labels_k10" / "cluster_labels_k10.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved labels to {output_path}")
    
    # Generate summary report
    label_counts = defaultdict(int)
    for layer_data in results["labels"].values():
        for cluster_data in layer_data.values():
            label_counts[cluster_data["label"]] += 1
    
    print("\nLabel Distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} clusters")
    
    print(f"\nTotal unique labels: {len(label_counts)}")


if __name__ == "__main__":
    main()