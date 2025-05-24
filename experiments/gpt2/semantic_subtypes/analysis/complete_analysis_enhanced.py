#!/usr/bin/env python3
"""
Enhanced version of complete_analysis.py that includes cluster contents.
This adds which sentences are in each cluster while keeping all existing functionality.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict, Counter
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

def load_results(results_dir: str):
    """Load the clustering results."""
    results_path = Path(results_dir)
    
    # Load activations
    with open(results_path / "semantic_subtypes_activations.pkl", 'rb') as f:
        activations_data = pickle.load(f)
    
    # Load clustering results
    with open(results_path / "semantic_subtypes_kmeans_clustering.pkl", 'rb') as f:
        kmeans_results = pickle.load(f)
        
    with open(results_path / "semantic_subtypes_ets_clustering.pkl", 'rb') as f:
        ets_results = pickle.load(f)
    
    return activations_data, kmeans_results, ets_results

def get_cluster_contents(clustering_results: Dict[str, Any]) -> Dict[str, Dict[int, List[str]]]:
    """
    Extract which sentences are in each cluster at each layer.
    
    Returns:
        Dict[layer_key][cluster_id] = [list of sentences]
    """
    sentences = clustering_results['sentences']
    layer_results = clustering_results['layer_results']
    
    cluster_contents = {}
    
    for layer_key, layer_data in layer_results.items():
        if 'cluster_labels' not in layer_data:
            continue
            
        clusters = defaultdict(list)
        
        # cluster_labels is dict: {sentence_idx: {token_idx: cluster_id}}
        for sent_idx, token_clusters in layer_data['cluster_labels'].items():
            if sent_idx < len(sentences):
                sentence = sentences[sent_idx]
                # For single-token words, use token_idx 0
                cluster_id = token_clusters.get(0, -1)
                if cluster_id != -1:
                    clusters[cluster_id].append(sentence)
        
        cluster_contents[layer_key] = dict(sorted(clusters.items()))
    
    return cluster_contents

def get_words_for_paths(clustering_results: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Get which words follow each archetypal path.
    
    Returns:
        Dict[path_string] = [list of words following this path]
    """
    sentences = clustering_results['sentences']
    token_paths = clustering_results.get('token_paths', {})
    
    path_to_words = defaultdict(list)
    
    for sent_idx, token_data in token_paths.items():
        if sent_idx < len(sentences) and 0 in token_data and token_data[0]:
            sentence = sentences[sent_idx]
            path_str = " -> ".join(token_data[0])
            path_to_words[path_str].append(sentence)
    
    return dict(path_to_words)

def compute_basic_apa_metrics(clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute basic APA metrics without requiring labels."""
    metrics = {
        'num_unique_paths': len(set(str(path) for paths in clustering_results['token_paths'].values() 
                                   for token_paths in paths.values() 
                                   for path in token_paths if path)),
        'layer_metrics': {}
    }
    
    # Compute per-layer metrics
    for layer_key, layer_data in clustering_results['layer_results'].items():
        if 'silhouette_score' in layer_data:
            metrics['layer_metrics'][layer_key] = {
                'num_clusters': layer_data.get('optimal_k', layer_data.get('num_clusters', 'N/A')),
                'silhouette_score': layer_data['silhouette_score']
            }
    
    return metrics

def analyze_semantic_organization(activations_data: Dict[str, Any], 
                                clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze semantic organization patterns."""
    # Load the curated words to map sentences to subtypes
    curated_file = Path("data/gpt2_semantic_subtypes_curated.json")
    if not curated_file.exists():
        return {'subtype_paths': {}, 'coherence_scores': {}}
    
    with open(curated_file, 'r') as f:
        curated_data = json.load(f)
    
    # Map sentences to subtypes
    sentence_to_subtype = {}
    curated_words = curated_data.get('curated_words', {})
    for subtype, words in curated_words.items():
        for word in words:
            sentence_to_subtype[word] = subtype
    
    # Analyze paths by subtype
    subtype_paths = defaultdict(list)
    for sent_idx, token_paths in clustering_results['token_paths'].items():
        if sent_idx < len(clustering_results['sentences']):
            sentence = clustering_results['sentences'][sent_idx]
            if sentence in sentence_to_subtype:
                subtype = sentence_to_subtype[sentence]
                if 0 in token_paths and token_paths[0]:
                    path = " -> ".join(token_paths[0])
                    subtype_paths[subtype].append(path)
    
    # Calculate coherence scores
    coherence_scores = {}
    for subtype, paths in subtype_paths.items():
        if paths:
            unique_paths = len(set(paths))
            coherence = 1.0 - (unique_paths - 1) / len(paths)
            coherence_scores[subtype] = {
                'coherence': coherence,
                'num_words': len(paths),
                'num_unique_paths': unique_paths
            }
    
    return {
        'subtype_paths': dict(subtype_paths),
        'coherence_scores': coherence_scores
    }

def prepare_llm_analysis_data(kmeans_metrics: Dict[str, Any],
                            ets_metrics: Dict[str, Any],
                            kmeans_results: Dict[str, Any], 
                            ets_results: Dict[str, Any],
                            semantic_analysis: Dict[str, Any],
                            output_file: str):
    """Prepare enhanced data for user's LLM analysis."""
    
    # Get cluster contents for both methods
    kmeans_clusters = get_cluster_contents(kmeans_results)
    ets_clusters = get_cluster_contents(ets_results)
    
    # Get words for paths
    kmeans_path_words = get_words_for_paths(kmeans_results)
    
    with open(output_file, 'w') as f:
        f.write("# GPT-2 Semantic Subtypes Experiment Results (Enhanced)\n\n")
        f.write("## Experiment Overview\n")
        f.write("- 774 single-token words across 8 semantic subtypes\n")
        f.write("- 13 GPT-2 layers (embedding + 12 transformer)\n")
        f.write("- Two clustering methods: K-means and ETS\n\n")
        
        # K-means results with cluster contents
        f.write("## K-means Clustering Results\n")
        f.write(f"- Total unique paths: {kmeans_metrics['num_unique_paths']}\n")
        f.write("- Layer-wise silhouette scores:\n")
        for layer, metrics in sorted(kmeans_metrics['layer_metrics'].items()):
            f.write(f"  - {layer}: {metrics['silhouette_score']:.3f} ({metrics['num_clusters']} clusters)\n")
        
        # Add cluster contents for key layers
        f.write("\n### Cluster Contents (Key Layers)\n")
        for layer_idx in [0, 6, 11]:
            layer_key = f"layer_{layer_idx}"
            if layer_key in kmeans_clusters:
                f.write(f"\n#### Layer {layer_idx}\n")
                for cluster_id, sentences in kmeans_clusters[layer_key].items():
                    f.write(f"\n**Cluster {cluster_id}** ({len(sentences)} sentences):\n")
                    # Show first 30 sentences
                    sample = sentences[:30]
                    f.write(f"{', '.join(sample)}")
                    if len(sentences) > 30:
                        f.write(f" ... and {len(sentences) - 30} more")
                    f.write("\n")
        
        # Semantic organization with actual words
        f.write("\n## Semantic Organization Analysis\n")
        f.write("### Within-Subtype Coherence\n")
        for subtype, scores in sorted(semantic_analysis['coherence_scores'].items()):
            f.write(f"- {subtype}: {scores['coherence']:.3f} coherence ")
            f.write(f"({scores['num_unique_paths']} paths for {scores['num_words']} words)\n")
        
        # Enhanced paths with actual words
        f.write("\n### Example Archetypal Paths by Subtype (with words)\n")
        for subtype in sorted(semantic_analysis['subtype_paths'].keys())[:4]:
            paths = semantic_analysis['subtype_paths'][subtype]
            if paths:
                path_counts = Counter(paths)
                f.write(f"\n**{subtype}** (top 3 paths):\n")
                for path, count in path_counts.most_common(3):
                    f.write(f"\n- Path: {path}\n")
                    f.write(f"  Count: {count} words\n")
                    # Show which words follow this path
                    if path in kmeans_path_words:
                        path_words = [w for w in kmeans_path_words[path] 
                                     if w in semantic_analysis['coherence_scores'].get(subtype, {}).get('words', [])]
                        if not path_words:  # If filtering didn't work, show all
                            path_words = kmeans_path_words[path]
                        sample = path_words[:10]
                        f.write(f"  Words: {', '.join(sample)}")
                        if len(path_words) > 10:
                            f.write(f" ... and {len(path_words) - 10} more")
                        f.write("\n")
        
        # ETS results if different from K-means
        if ets_results != kmeans_results:
            f.write("\n## ETS Clustering Results\n")
            f.write(f"- Total unique paths: {ets_metrics['num_unique_paths']}\n")
            
            # Show ETS cluster contents for comparison
            f.write("\n### ETS Cluster Contents (Layer 6 example)\n")
            layer_key = "layer_6"
            if layer_key in ets_clusters:
                # Show first 5 clusters as example
                for cluster_id, sentences in list(ets_clusters[layer_key].items())[:5]:
                    f.write(f"\n**Cluster {cluster_id}** ({len(sentences)} sentences):\n")
                    sample = sentences[:20]
                    f.write(f"{', '.join(sample)}")
                    if len(sentences) > 20:
                        f.write(f" ... and {len(sentences) - 20} more")
                    f.write("\n")
                
                if len(ets_clusters[layer_key]) > 5:
                    f.write(f"\n... and {len(ets_clusters[layer_key]) - 5} more clusters\n")
        
        # Key questions (without K-means vs ETS comparison)
        f.write("\n## Key Questions for Analysis\n")
        
        f.write("\n### Cluster Interpretation\n")
        f.write("1. Looking at the sentences in each cluster, what semantic or grammatical themes emerge?\n")
        f.write("2. Can you suggest descriptive labels for the clusters in layers 0, 6, and 12?\n")
        
        f.write("\n### Archetypal Path Analysis\n")
        f.write("3. What do the archetypal paths represent semantically?\n")
        f.write("4. Why might certain words follow the same path through the layers?\n")
        f.write("5. Do words within the same subtype follow similar paths?\n")
        
        f.write("\n### Layer Evolution\n")
        f.write("6. How does the clustering evolve from early to late layers?\n")
        f.write("7. Do early layers capture more syntactic features while later layers capture semantics?\n")
        
        f.write("\n### Semantic Organization Insights\n")
        f.write("8. How does GPT-2 organize semantic knowledge across layers?\n")
        f.write("9. Do grammatical categories (nouns, verbs, etc.) cluster together?\n")
        f.write("10. What do the coherence scores reveal about GPT-2's internal organization?\n")

def main():
    # Use the most recent complete results
    results_dir = "semantic_subtypes_experiment_20250523_111112"
    
    print(f"Loading results from {results_dir}...")
    activations_data, kmeans_results, ets_results = load_results(results_dir)
    
    print("Computing basic APA metrics...")
    kmeans_metrics = compute_basic_apa_metrics(kmeans_results)
    ets_metrics = compute_basic_apa_metrics(ets_results)
    
    print("Analyzing semantic organization...")
    semantic_analysis = analyze_semantic_organization(activations_data, kmeans_results)
    
    print("Preparing enhanced LLM analysis data...")
    output_file = Path(results_dir) / "llm_analysis_data_enhanced.md"
    prepare_llm_analysis_data(kmeans_metrics, ets_metrics, kmeans_results, ets_results, semantic_analysis, str(output_file))
    
    print(f"\nAnalysis complete! Results saved to:")
    print(f"  {output_file}")
    print("\nThe enhanced analysis now includes:")
    print("  - Which sentences are in each cluster")
    print("  - Which words follow each archetypal path")
    print("  - Ready for your manual LLM analysis")

if __name__ == "__main__":
    main()