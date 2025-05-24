#!/usr/bin/env python3
"""
Extract and display cluster contents from existing GPT-2 semantic subtypes results.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from experiments.gpt2.semantic_subtypes.gpt2_semantic_subtypes_wordlists import (
    CONCRETE_NOUNS, ABSTRACT_NOUNS, PHYSICAL_ADJECTIVES, EMOTIVE_ADJECTIVES,
    MANNER_ADVERBS, DEGREE_ADVERBS, ACTION_VERBS, STATIVE_VERBS
)

# Construct semantic subtypes dictionary
SEMANTIC_SUBTYPES = {
    'concrete_nouns': CONCRETE_NOUNS,
    'abstract_nouns': ABSTRACT_NOUNS,
    'physical_adjectives': PHYSICAL_ADJECTIVES,
    'emotive_adjectives': EMOTIVE_ADJECTIVES,
    'manner_adverbs': MANNER_ADVERBS,
    'degree_adverbs': DEGREE_ADVERBS,
    'action_verbs': ACTION_VERBS,
    'stative_verbs': STATIVE_VERBS
}


def load_results(experiment_path: Path):
    """Load experiment results."""
    results_file = experiment_path / "results" / "semantic_subtypes_results.json"
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, "r") as f:
        return json.load(f)


def extract_cluster_contents(results):
    """Extract and organize words by cluster for both K-means and ETS."""
    
    # Get word list from results
    word_list = results['word_list']
    
    # Process both clustering methods
    cluster_contents = {}
    
    for method in ['kmeans', 'ets']:
        if method not in results['clustering_results']:
            continue
            
        method_results = results['clustering_results'][method]
        cluster_contents[method] = {}
        
        # For each layer
        for layer_idx in range(13):  # GPT-2 has 13 layers (0-12)
            layer_key = f"layer_{layer_idx}"
            if layer_key not in method_results:
                continue
                
            layer_data = method_results[layer_key]
            cluster_labels = layer_data.get('cluster_labels', {})
            
            # Group words by cluster
            clusters_by_id = defaultdict(list)
            
            # cluster_labels is dict: {sentence_idx: {token_idx: cluster_id}}
            for sent_idx_str, token_clusters in cluster_labels.items():
                sent_idx = int(sent_idx_str)
                if sent_idx < len(word_list):
                    word = word_list[sent_idx]
                    # Assuming token_idx 0 for single-token words
                    cluster_id = token_clusters.get('0', -1)
                    if cluster_id != -1:
                        clusters_by_id[cluster_id].append(word)
            
            # Sort clusters by ID and store
            cluster_contents[method][layer_key] = {
                'num_clusters': layer_data.get('optimal_k', 0),
                'silhouette_score': layer_data.get('silhouette_score', 0.0),
                'clusters': dict(sorted(clusters_by_id.items()))
            }
    
    return cluster_contents


def generate_cluster_report(cluster_contents, output_path: Path):
    """Generate a detailed report of cluster contents."""
    
    report_lines = [
        "# GPT-2 Semantic Subtypes - Cluster Contents Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Overview",
        f"This report shows the actual words grouped in each cluster for both K-means and ETS clustering methods.",
        "\n---"
    ]
    
    # Add results for each method
    for method in ['kmeans', 'ets']:
        if method not in cluster_contents:
            continue
            
        method_name = "K-means" if method == 'kmeans' else "ETS"
        report_lines.append(f"\n## {method_name} Clustering Results")
        
        # Focus on key layers (early, middle, late)
        key_layers = [0, 6, 12]
        
        for layer_idx in key_layers:
            layer_key = f"layer_{layer_idx}"
            if layer_key not in cluster_contents[method]:
                continue
                
            layer_data = cluster_contents[method][layer_key]
            num_clusters = layer_data['num_clusters']
            silhouette = layer_data['silhouette_score']
            clusters = layer_data['clusters']
            
            report_lines.append(f"\n### Layer {layer_idx}")
            report_lines.append(f"- **Number of clusters**: {num_clusters}")
            report_lines.append(f"- **Silhouette score**: {silhouette:.3f}")
            report_lines.append(f"- **Cluster details**:")
            
            # Show all clusters with their contents
            for cluster_id, words in clusters.items():
                report_lines.append(f"\n#### Cluster {cluster_id} ({len(words)} words)")
                
                # Group words by semantic subtype for analysis
                words_by_type = defaultdict(list)
                for word in words:
                    for subtype, subtype_words in SEMANTIC_SUBTYPES.items():
                        if word in subtype_words:
                            words_by_type[subtype].append(word)
                            break
                
                # Show words grouped by type
                if words_by_type:
                    report_lines.append("\n**Words by semantic type:**")
                    for subtype, type_words in sorted(words_by_type.items()):
                        if type_words:
                            report_lines.append(f"- *{subtype}*: {', '.join(sorted(type_words))}")
                
                # Show all words in cluster
                report_lines.append(f"\n**All words**: {', '.join(sorted(words))}")
    
    # Add analysis questions
    report_lines.extend([
        "\n---",
        "\n## Key Questions for Analysis",
        "\n### 1. Cluster Labeling",
        "What semantic themes or patterns characterize each cluster? Consider:",
        "- Grammatical categories (nouns, verbs, adjectives, adverbs)",
        "- Semantic fields (concrete/abstract, emotional/physical, etc.)",
        "- Functional roles in language",
        "",
        "### 2. Method Comparison", 
        "How do K-means and ETS clustering differ in their groupings?",
        "- Does ETS create more granular distinctions?",
        "- Which method better preserves semantic subtypes?",
        "- Are there systematic differences in how they group words?",
        "",
        "### 3. Layer Evolution",
        "How do word groupings change across layers (0 → 6 → 12)?",
        "- Do clusters become more or less semantically coherent?",
        "- Is there evidence of hierarchical processing?",
        "- Do certain word types stabilize earlier/later?",
        "",
        "### 4. Semantic Coherence",
        "Are clusters semantically coherent or mixed?",
        "- Which clusters show clear semantic themes?",
        "- Which clusters seem arbitrary or mixed?",
        "- Is there a pattern to the mixing?",
        "",
        "### 5. Theoretical Implications",
        "What do these groupings tell us about GPT-2's internal representations?",
        "- Evidence for/against distributed vs. localized representations",
        "- Relationship between syntax and semantics in the model",
        "- Implications for interpretability and control"
    ])
    
    # Save report
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"Cluster contents report saved to: {output_path}")
    
    # Also save raw cluster contents as JSON
    json_path = output_path.parent / "cluster_contents.json"
    with open(json_path, "w") as f:
        json.dump(cluster_contents, f, indent=2)
    
    print(f"Raw cluster contents saved to: {json_path}")


def main():
    """Extract cluster contents from existing results."""
    
    # Find the most recent experiment directory
    experiment_dirs = list(Path(".").glob("semantic_subtypes_experiment_*"))
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    # Use the most recent one
    experiment_path = sorted(experiment_dirs)[-1]
    print(f"Using experiment directory: {experiment_path}")
    
    # Load results
    results = load_results(experiment_path)
    if not results:
        return
    
    # Extract cluster contents
    cluster_contents = extract_cluster_contents(results)
    
    # Generate report
    report_path = experiment_path / "cluster_contents_report.md"
    generate_cluster_report(cluster_contents, report_path)
    
    # Print summary
    print("\nSummary of clusters per layer:")
    for method in ['kmeans', 'ets']:
        if method in cluster_contents:
            print(f"\n{method.upper()}:")
            for layer_idx in [0, 6, 12]:
                layer_key = f"layer_{layer_idx}"
                if layer_key in cluster_contents[method]:
                    num_clusters = cluster_contents[method][layer_key]['num_clusters']
                    print(f"  Layer {layer_idx}: {num_clusters} clusters")


if __name__ == "__main__":
    main()