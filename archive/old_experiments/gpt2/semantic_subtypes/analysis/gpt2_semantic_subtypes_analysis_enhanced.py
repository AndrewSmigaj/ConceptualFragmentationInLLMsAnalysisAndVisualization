#!/usr/bin/env python3
"""
Enhanced GPT-2 semantic subtypes analysis with cluster content extraction.
This script runs the complete analysis and includes words in each cluster.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging

# Add parent directories to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from experiments.gpt2.semantic_subtypes.gpt2_semantic_subtypes_experiment import SemanticSubtypesExperiment
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

logger = logging.getLogger(__name__)


def extract_cluster_contents(experiment_path: Path):
    """Extract and organize words by cluster for both K-means and ETS."""
    
    # Load experiment results
    results_file = experiment_path / "results" / "semantic_subtypes_results.json"
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, "r") as f:
        results = json.load(f)
    
    # Load word list
    all_words = []
    for subtype, words in SEMANTIC_SUBTYPES.items():
        all_words.extend(words)
    
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
                if sent_idx < len(all_words):
                    word = all_words[sent_idx]
                    # Assuming token_idx 0 for single-token words
                    cluster_id = token_clusters.get('0', -1)
                    if cluster_id != -1:
                        clusters_by_id[cluster_id].append(word)
            
            # Sort clusters by ID and store
            cluster_contents[method][layer_key] = {
                'num_clusters': layer_data.get('optimal_k', 0),
                'clusters': dict(sorted(clusters_by_id.items()))
            }
    
    return cluster_contents


def generate_enhanced_analysis(experiment_path: Path, optimal_ets_threshold: float = 0.992):
    """Generate enhanced analysis with cluster contents and optimal ETS threshold."""
    
    # First, re-run experiment with optimal ETS threshold
    logger.info(f"Re-running experiment with optimal ETS threshold: {optimal_ets_threshold}")
    
    experiment = SemanticSubtypesExperiment(
        output_dir=str(experiment_path),
        clustering_methods=['kmeans', 'ets'],
        ets_threshold_percentile=optimal_ets_threshold
    )
    
    # Run the experiment
    results = experiment.run()
    
    # Extract cluster contents
    cluster_contents = extract_cluster_contents(experiment_path)
    
    if not cluster_contents:
        logger.error("Failed to extract cluster contents")
        return
    
    # Generate enhanced analysis report
    report_lines = [
        "# GPT-2 Semantic Subtypes Analysis - Enhanced Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\nOptimal ETS threshold (from elbow analysis): {optimal_ets_threshold}",
        "\n## Overview",
        f"- Total words analyzed: {len(results['word_list'])}",
        f"- Semantic subtypes: {len(SEMANTIC_SUBTYPES)}",
        "\n## Clustering Results with Word Contents",
    ]
    
    # Add results for each method
    for method in ['kmeans', 'ets']:
        if method not in cluster_contents:
            continue
            
        method_name = "K-means" if method == 'kmeans' else f"ETS (threshold={optimal_ets_threshold})"
        report_lines.append(f"\n### {method_name} Clustering")
        
        # Focus on key layers (early, middle, late)
        key_layers = [0, 6, 12]
        
        for layer_idx in key_layers:
            layer_key = f"layer_{layer_idx}"
            if layer_key not in cluster_contents[method]:
                continue
                
            layer_data = cluster_contents[method][layer_key]
            num_clusters = layer_data['num_clusters']
            clusters = layer_data['clusters']
            
            report_lines.append(f"\n#### Layer {layer_idx} ({num_clusters} clusters)")
            
            # Show up to 10 clusters with their contents
            for cluster_id, words in list(clusters.items())[:10]:
                # Show first 10 words per cluster
                sample_words = words[:10]
                if len(words) > 10:
                    sample_words.append(f"... ({len(words)-10} more)")
                
                report_lines.append(f"\n**Cluster {cluster_id}** ({len(words)} words):")
                report_lines.append(f"  {', '.join(sample_words)}")
            
            if len(clusters) > 10:
                report_lines.append(f"\n... and {len(clusters)-10} more clusters")
    
    # Add LLM analysis prompts
    report_lines.extend([
        "\n## Key Questions for LLM Analysis",
        "\n1. **Cluster Labeling**: What semantic themes or patterns characterize each cluster?",
        "2. **Method Comparison**: How do K-means and ETS clustering differ in their groupings?",
        "3. **Layer Evolution**: How do word groupings change across layers?",
        "4. **Semantic Coherence**: Are clusters semantically coherent or mixed?",
        "5. **Subtype Preservation**: Do original semantic subtypes remain grouped together?",
    ])
    
    # Save enhanced report
    report_path = experiment_path / "enhanced_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Enhanced analysis report saved to: {report_path}")
    
    # Also save cluster contents as JSON for further analysis
    cluster_contents_path = experiment_path / "cluster_contents.json"
    with open(cluster_contents_path, "w") as f:
        json.dump(cluster_contents, f, indent=2)
    
    logger.info(f"Cluster contents saved to: {cluster_contents_path}")
    
    return cluster_contents


def main():
    """Run enhanced analysis with optimal ETS threshold from elbow analysis."""
    
    # Use the existing experiment directory
    experiment_path = Path("semantic_subtypes_experiment_20250523_111112")
    
    if not experiment_path.exists():
        logger.error(f"Experiment directory not found: {experiment_path}")
        return
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run enhanced analysis with optimal threshold from elbow analysis
    optimal_threshold = 0.992  # From elbow analysis
    generate_enhanced_analysis(experiment_path, optimal_threshold)


if __name__ == "__main__":
    main()