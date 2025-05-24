#!/usr/bin/env python3
"""
Analyze cluster contents for the optimal clustering experiment.
Shows which words are in each cluster and their semantic properties.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

def load_results():
    """Load clustering results and curated words."""
    # Load clustering results
    result_dir = Path("semantic_subtypes_optimal_experiment_20250523_182344")
    
    with open(result_dir / "semantic_subtypes_kmeans_optimal.pkl", 'rb') as f:
        kmeans_results = pickle.load(f)
    
    with open(result_dir / "semantic_subtypes_ets_optimal.pkl", 'rb') as f:
        ets_results = pickle.load(f)
    
    # Load curated words
    with open("data/gpt2_semantic_subtypes_curated.json", 'r') as f:
        curated_data = json.load(f)
    
    return kmeans_results, ets_results, curated_data

def analyze_cluster_contents(clustering_results: Dict[str, Any], curated_data: Dict[str, Any], method_name: str):
    """Analyze which words are in each cluster."""
    
    sentences = clustering_results['sentences']
    
    # Create word to subtype mapping
    word_to_subtype = {}
    for subtype, words in curated_data['curated_words'].items():
        for word in words:
            word_to_subtype[word] = subtype
    
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} CLUSTER CONTENTS ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze each layer
    for layer_idx in range(13):
        layer_key = f"layer_{layer_idx}"
        if layer_key not in clustering_results['layer_results']:
            continue
            
        layer_data = clustering_results['layer_results'][layer_key]
        
        print(f"\n{'-'*60}")
        print(f"LAYER {layer_idx}")
        print(f"{'-'*60}")
        
        # Get cluster assignments
        cluster_assignments = defaultdict(list)
        
        for sent_idx, clusters in layer_data['cluster_labels'].items():
            if 0 in clusters:  # Token position 0
                cluster_id = clusters[0]
                word = sentences[sent_idx]
                subtype = word_to_subtype.get(word, 'unknown')
                cluster_assignments[cluster_id].append((word, subtype))
        
        # Analyze each cluster
        for cluster_id in sorted(cluster_assignments.keys()):
            words_in_cluster = cluster_assignments[cluster_id]
            
            print(f"\nCluster {cluster_id} ({len(words_in_cluster)} words):")
            
            # Count subtypes in this cluster
            subtype_counts = Counter(subtype for _, subtype in words_in_cluster)
            
            # Print subtype distribution
            print("  Subtype distribution:")
            for subtype, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
                percentage = (count / len(words_in_cluster)) * 100
                print(f"    {subtype}: {count} ({percentage:.1f}%)")
            
            # Show sample words by subtype
            print("  Sample words by subtype:")
            words_by_subtype = defaultdict(list)
            for word, subtype in words_in_cluster:
                words_by_subtype[subtype].append(word)
            
            for subtype in sorted(words_by_subtype.keys()):
                sample = sorted(words_by_subtype[subtype])[:10]
                if len(words_by_subtype[subtype]) > 10:
                    print(f"    {subtype}: {', '.join(sample)}... ({len(words_by_subtype[subtype])} total)")
                else:
                    print(f"    {subtype}: {', '.join(sample)}")
        
        # Calculate cluster purity
        if cluster_assignments:
            total_words = sum(len(words) for words in cluster_assignments.values())
            max_subtype_words = sum(max(Counter(s for _, s in words).values()) for words in cluster_assignments.values())
            purity = max_subtype_words / total_words
            print(f"\nLayer {layer_idx} purity: {purity:.3f}")

def compare_methods(kmeans_results, ets_results, curated_data):
    """Compare cluster assignments between methods."""
    
    print(f"\n{'='*80}")
    print("METHOD COMPARISON")
    print(f"{'='*80}")
    
    sentences = kmeans_results['sentences']
    
    # Create word to subtype mapping
    word_to_subtype = {}
    for subtype, words in curated_data['curated_words'].items():
        for word in words:
            word_to_subtype[word] = subtype
    
    # Compare specific interesting layers
    interesting_layers = [0, 5, 10]  # Early, middle, late
    
    for layer_idx in interesting_layers:
        layer_key = f"layer_{layer_idx}"
        
        print(f"\n{'-'*40}")
        print(f"Layer {layer_idx} Comparison")
        print(f"{'-'*40}")
        
        # Get assignments from both methods
        kmeans_layer = kmeans_results['layer_results'][layer_key]
        ets_layer = ets_results['layer_results'][layer_key]
        
        # Build assignment mappings
        kmeans_assignments = {}
        ets_assignments = {}
        
        for sent_idx in kmeans_layer['cluster_labels']:
            if 0 in kmeans_layer['cluster_labels'][sent_idx]:
                kmeans_assignments[sent_idx] = kmeans_layer['cluster_labels'][sent_idx][0]
            if sent_idx in ets_layer['cluster_labels'] and 0 in ets_layer['cluster_labels'][sent_idx]:
                ets_assignments[sent_idx] = ets_layer['cluster_labels'][sent_idx][0]
        
        # Find words that are clustered differently
        differently_clustered = []
        for sent_idx in kmeans_assignments:
            if sent_idx in ets_assignments:
                if kmeans_assignments[sent_idx] != ets_assignments[sent_idx]:
                    word = sentences[sent_idx]
                    subtype = word_to_subtype.get(word, 'unknown')
                    differently_clustered.append({
                        'word': word,
                        'subtype': subtype,
                        'kmeans_cluster': kmeans_assignments[sent_idx],
                        'ets_cluster': ets_assignments[sent_idx]
                    })
        
        print(f"Words clustered differently: {len(differently_clustered)}")
        
        # Show examples
        if differently_clustered:
            print("\nExamples of different clustering:")
            for item in differently_clustered[:10]:
                print(f"  '{item['word']}' ({item['subtype']}): "
                      f"K-means C{item['kmeans_cluster']} vs ETS C{item['ets_cluster']}")

def main():
    """Run cluster content analysis."""
    
    print("Loading results...")
    kmeans_results, ets_results, curated_data = load_results()
    
    # Analyze K-means clusters
    analyze_cluster_contents(kmeans_results, curated_data, "K-means")
    
    # Analyze ETS clusters
    analyze_cluster_contents(ets_results, curated_data, "ETS")
    
    # Compare methods
    compare_methods(kmeans_results, ets_results, curated_data)
    
    # Save analysis for LLM
    print("\nGenerating LLM analysis data...")
    
    # Focus on key layers for detailed analysis
    key_layers = [0, 3, 5, 7, 10, 12]  # Sample across network depth
    
    llm_data = {
        'experiment': 'semantic_subtypes_optimal_clustering',
        'total_words': len(kmeans_results['sentences']),
        'semantic_subtypes': list(curated_data['curated_words'].keys()),
        'layer_analysis': {}
    }
    
    for layer_idx in key_layers:
        layer_key = f"layer_{layer_idx}"
        
        llm_data['layer_analysis'][layer_key] = {
            'kmeans': extract_layer_summary(kmeans_results, curated_data, layer_idx),
            'ets': extract_layer_summary(ets_results, curated_data, layer_idx)
        }
    
    # Save for LLM analysis
    with open("cluster_contents_for_llm.json", 'w') as f:
        json.dump(llm_data, f, indent=2)
    
    print("\nLLM analysis data saved to: cluster_contents_for_llm.json")

def extract_layer_summary(clustering_results, curated_data, layer_idx):
    """Extract summary for a specific layer."""
    
    layer_key = f"layer_{layer_idx}"
    layer_data = clustering_results['layer_results'][layer_key]
    sentences = clustering_results['sentences']
    
    # Create word to subtype mapping
    word_to_subtype = {}
    for subtype, words in curated_data['curated_words'].items():
        for word in words:
            word_to_subtype[word] = subtype
    
    # Get cluster assignments
    cluster_summaries = {}
    
    for sent_idx, clusters in layer_data['cluster_labels'].items():
        if 0 in clusters:
            cluster_id = clusters[0]
            if cluster_id not in cluster_summaries:
                cluster_summaries[cluster_id] = {
                    'words': [],
                    'subtype_counts': defaultdict(int)
                }
            
            word = sentences[sent_idx]
            subtype = word_to_subtype.get(word, 'unknown')
            cluster_summaries[cluster_id]['words'].append(word)
            cluster_summaries[cluster_id]['subtype_counts'][subtype] += 1
    
    # Format summaries
    formatted_summaries = {}
    for cluster_id, summary in cluster_summaries.items():
        formatted_summaries[f"cluster_{cluster_id}"] = {
            'size': len(summary['words']),
            'subtype_distribution': dict(summary['subtype_counts']),
            'sample_words': sorted(summary['words'])[:20]  # First 20 alphabetically
        }
    
    return formatted_summaries

if __name__ == "__main__":
    main()