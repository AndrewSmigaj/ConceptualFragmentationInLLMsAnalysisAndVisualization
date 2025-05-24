#!/usr/bin/env python3
"""
Complete the semantic subtypes analysis using existing results.
This script bypasses the label-based metrics that don't apply to our experiment.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
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
    # Get word to subtype mapping
    word_to_subtype = activations_data.get('word_to_subtype', {})
    
    # Analyze paths by subtype
    subtype_paths = {}
    for sent_idx, token_paths in clustering_results['token_paths'].items():
        word = activations_data['sentences'][sent_idx]
        subtype = word_to_subtype.get(word, 'unknown')
        
        if subtype not in subtype_paths:
            subtype_paths[subtype] = []
        
        # Get the path for token 0 (single words)
        if 0 in token_paths and token_paths[0]:
            path_str = ' -> '.join(token_paths[0])
            subtype_paths[subtype].append(path_str)
    
    # Compute within-subtype coherence
    coherence_scores = {}
    for subtype, paths in subtype_paths.items():
        if len(paths) > 1:
            # Count how many unique paths exist for this subtype
            unique_paths = len(set(paths))
            coherence = 1.0 - (unique_paths - 1) / len(paths)
            coherence_scores[subtype] = {
                'coherence': coherence,
                'num_words': len(paths),
                'num_unique_paths': unique_paths
            }
    
    return {
        'subtype_paths': subtype_paths,
        'coherence_scores': coherence_scores
    }

def prepare_llm_analysis_data(kmeans_results: Dict[str, Any], 
                            ets_results: Dict[str, Any],
                            semantic_analysis: Dict[str, Any],
                            output_file: str):
    """Prepare data for user's LLM analysis."""
    with open(output_file, 'w') as f:
        f.write("# GPT-2 Semantic Subtypes Experiment Results\\n\\n")
        f.write("## Experiment Overview\\n")
        f.write("- 774 single-token words across 8 semantic subtypes\\n")
        f.write("- 13 GPT-2 layers (embedding + 12 transformer)\\n")
        f.write("- Two clustering methods: K-means and ETS (fell back to K-means)\\n\\n")
        
        f.write("## K-means Clustering Results\\n")
        f.write(f"- Total unique paths: {kmeans_results['num_unique_paths']}\\n")
        f.write("- Layer-wise silhouette scores:\\n")
        for layer, metrics in sorted(kmeans_results['layer_metrics'].items()):
            f.write(f"  - {layer}: {metrics['silhouette_score']:.3f} ({metrics['num_clusters']} clusters)\\n")
        
        f.write("\\n## Semantic Organization Analysis\\n")
        f.write("### Within-Subtype Coherence\\n")
        for subtype, scores in sorted(semantic_analysis['coherence_scores'].items()):
            f.write(f"- {subtype}: {scores['coherence']:.3f} coherence ")
            f.write(f"({scores['num_unique_paths']} paths for {scores['num_words']} words)\\n")
        
        # Add example paths for each subtype
        f.write("\\n### Example Archetypal Paths by Subtype\\n")
        for subtype in sorted(semantic_analysis['subtype_paths'].keys())[:4]:  # Show first 4 subtypes
            paths = semantic_analysis['subtype_paths'][subtype]
            if paths:
                # Count path frequencies
                from collections import Counter
                path_counts = Counter(paths)
                f.write(f"\\n**{subtype}** (top 3 paths):\\n")
                for path, count in path_counts.most_common(3):
                    f.write(f"- {path} ({count} words)\\n")
        
        f.write("\\n## Key Questions for Analysis\\n")
        f.write("\\n### Archetypal Path Analysis\\n")
        f.write("1. What archetypal paths emerge for each semantic subtype?\\n")
        f.write("2. Do words within the same subtype follow similar paths (within-subtype coherence)?\\n")
        f.write("3. Do different subtypes follow distinct paths (between-subtype differentiation)?\\n")
        f.write("4. At what layers do semantic distinctions become most pronounced?\\n")
        f.write("5. Which layers show the highest fragmentation scores?\\n")
        f.write("6. What is the path purity for each semantic subtype?\\n")
        
        f.write("\\n### Clustering Method Comparison\\n")
        f.write("7. Do K-means and ETS reveal similar archetypal paths?\\n")
        f.write("8. Does ETS's threshold-based approach improve semantic clustering?\\n")
        f.write("9. Which method better separates semantic subtypes?\\n")
        f.write("10. Which method produces more interpretable cluster assignments?\\n")
        
        f.write("\\n### Semantic Organization Insights\\n")
        f.write("11. How does GPT-2 organize semantic knowledge across layers?\\n")
        f.write("12. Do grammatical categories (nouns, verbs, etc.) cluster together?\\n")
        f.write("13. Within grammatical categories, how are semantic distinctions encoded?\\n")
        f.write("14. What do the cross-layer metrics reveal about representation stability?\\n")

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
    
    print("Preparing LLM analysis data...")
    output_file = Path(results_dir) / "llm_analysis_data.md"
    prepare_llm_analysis_data(kmeans_metrics, ets_metrics, semantic_analysis, str(output_file))
    
    print(f"\\nAnalysis complete! Results saved to:")
    print(f"  {output_file}")
    print("\\nYou can now copy the contents of llm_analysis_data.md for your manual LLM analysis.")

if __name__ == "__main__":
    main()