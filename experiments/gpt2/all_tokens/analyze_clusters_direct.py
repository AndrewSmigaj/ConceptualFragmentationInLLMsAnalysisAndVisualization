#!/usr/bin/env python3
"""
Direct analysis of k=10 clusters.
This script prepares the cluster data for direct analysis.
"""

import json
from pathlib import Path
from collections import defaultdict


def analyze_token_types(tokens):
    """Categorize tokens into linguistic types."""
    types = defaultdict(int)
    
    # Common function words
    function_words = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
        'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
        'me', 'when', 'make', 'can', 'like', 'no', 'just', 'him', 'know', 'take',
        'into', 'your', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
        'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
        'after', 'use', 'two', 'how', 'our', 'well', 'way', 'even', 'new', 'want',
        'because', 'any', 'these', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
        'were', 'said', 'did', 'get', 'may', 'am', 'de', 'en', 'un'
    }
    
    # Prepositions
    prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about', 'through', 'over', 'under', 'between', 'among'}
    
    # Pronouns
    pronouns = {'I', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
    
    # Modal verbs
    modals = {'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}
    
    for token in tokens[:100]:  # Analyze first 100 tokens
        t = token.strip().lower()
        
        if t in pronouns:
            types['pronouns'] += 1
        elif t in prepositions:
            types['prepositions'] += 1
        elif t in modals:
            types['modal_verbs'] += 1
        elif t in function_words:
            types['function_words'] += 1
        elif len(t) == 1 and not t.isalnum():
            types['punctuation'] += 1
        elif t.startswith('Ġ'):  # GPT-2 space token
            types['space_prefixed'] += 1
        elif t.isdigit():
            types['numbers'] += 1
        elif t.endswith('ing'):
            types['ing_forms'] += 1
        elif t.endswith('ed'):
            types['ed_forms'] += 1
        elif t.endswith('ly'):
            types['adverbs'] += 1
        elif t.endswith('s') and len(t) > 1:
            types['plural_forms'] += 1
        else:
            types['other'] += 1
    
    return dict(types)


def main():
    base_dir = Path(__file__).parent
    
    # Load cluster data
    with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
        cluster_data = json.load(f)
    
    # Prepare analysis data
    analysis_data = {
        "clusters": {},
        "summary": {
            "total_clusters": 0,
            "layers": 12,
            "k": 10
        }
    }
    
    # Analyze each cluster
    for layer in range(12):
        for cluster_idx in range(10):
            cluster_key = f"L{layer}_C{cluster_idx}"
            
            if cluster_key in cluster_data["clusters"]:
                cluster_info = cluster_data["clusters"][cluster_key]
                tokens = cluster_info["common_tokens"]
                
                # Analyze token types
                token_types = analyze_token_types(tokens)
                
                # Find dominant type
                if token_types:
                    dominant_type = max(token_types.items(), key=lambda x: x[1])[0]
                else:
                    dominant_type = "mixed"
                
                analysis_data["clusters"][cluster_key] = {
                    "layer": layer,
                    "cluster_idx": cluster_idx,
                    "size": cluster_info["size"],
                    "percentage": cluster_info["percentage"],
                    "top_50_tokens": tokens[:50],
                    "token_types": token_types,
                    "dominant_type": dominant_type
                }
                
                analysis_data["summary"]["total_clusters"] += 1
    
    # Save analysis data
    output_path = base_dir / "llm_labels_k10" / "cluster_analysis_k10.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2)
    
    print(f"Saved cluster analysis to {output_path}")
    
    # Print summary for direct analysis
    print("\nCLUSTER ANALYSIS SUMMARY FOR DIRECT LLM ANALYSIS")
    print("=" * 60)
    print("\nPlease analyze these clusters and provide consistent semantic labels.")
    print("\nKey patterns to look for:")
    print("1. Clusters with similar token sets across layers should have the same label")
    print("2. Focus on linguistic function (e.g., 'Function Words', 'Content Words')")
    print("3. Consider grammatical categories (e.g., 'Pronouns', 'Prepositions')")
    print("\nExample clusters for analysis:")
    
    # Show a few examples
    for i, (cluster_key, data) in enumerate(analysis_data["clusters"].items()):
        if i >= 5:
            break
        print(f"\n{cluster_key}:")
        print(f"  Size: {data['size']} tokens ({data['percentage']:.1f}%)")
        print(f"  Dominant type: {data['dominant_type']}")
        print(f"  Top tokens: {', '.join(repr(t) for t in data['top_50_tokens'][:10])}")
        
        # Show token type distribution
        if data['token_types']:
            print("  Token types:")
            for ttype, count in sorted(data['token_types'].items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    - {ttype}: {count}")


if __name__ == "__main__":
    main()