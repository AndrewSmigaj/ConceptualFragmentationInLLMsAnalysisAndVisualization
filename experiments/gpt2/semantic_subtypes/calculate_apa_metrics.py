#!/usr/bin/env python3
"""
Calculate APA (Archetypal Path Analysis) metrics for the optimal clustering results.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from experiments.gpt2.shared.gpt2_apa_metrics import GPT2APAMetrics

def main():
    """Calculate APA metrics for both clustering methods."""
    
    print("Loading clustering results...")
    
    # Load results
    result_dir = Path("semantic_subtypes_optimal_experiment_20250523_182344")
    
    with open(result_dir / "semantic_subtypes_kmeans_optimal.pkl", 'rb') as f:
        kmeans_results = pickle.load(f)
    
    with open(result_dir / "semantic_subtypes_ets_optimal.pkl", 'rb') as f:
        ets_results = pickle.load(f)
    
    # Create output directory
    output_dir = result_dir / "apa_metrics"
    output_dir.mkdir(exist_ok=True)
    
    # Calculate metrics for K-means
    print("\nCalculating APA metrics for K-means...")
    kmeans_metrics = GPT2APAMetrics()
    kmeans_analysis = kmeans_metrics.calculate_all_metrics(kmeans_results)
    
    # Save K-means metrics
    kmeans_output = output_dir / "kmeans_apa_metrics.json"
    with open(kmeans_output, 'w') as f:
        json.dump(kmeans_analysis, f, indent=2)
    print(f"K-means metrics saved to: {kmeans_output}")
    
    # Calculate metrics for ETS
    print("\nCalculating APA metrics for ETS...")
    ets_metrics = GPT2APAMetrics()
    ets_analysis = ets_metrics.calculate_all_metrics(ets_results)
    
    # Save ETS metrics
    ets_output = output_dir / "ets_apa_metrics.json"
    with open(ets_output, 'w') as f:
        json.dump(ets_analysis, f, indent=2)
    print(f"ETS metrics saved to: {ets_output}")
    
    # Generate comparison summary
    print("\n" + "="*60)
    print("APA METRICS COMPARISON")
    print("="*60)
    
    # Path metrics comparison
    print("\nPath Metrics:")
    print(f"{'Metric':<30} {'K-means':<15} {'ETS':<15}")
    print("-"*60)
    
    k_paths = kmeans_analysis['path_metrics']
    e_paths = ets_analysis['path_metrics']
    
    print(f"{'Total unique paths':<30} {k_paths['total_unique_paths']:<15} {e_paths['total_unique_paths']:<15}")
    print(f"{'Avg path diversity':<30} {k_paths['average_path_diversity']:<15.3f} {e_paths['average_path_diversity']:<15.3f}")
    print(f"{'Path coherence':<30} {k_paths['path_coherence']:<15.3f} {e_paths['path_coherence']:<15.3f}")
    
    # Fragmentation comparison
    print("\nFragmentation Metrics:")
    k_frag = kmeans_analysis['fragmentation_summary']
    e_frag = ets_analysis['fragmentation_summary']
    
    print(f"{'Avg fragmentation':<30} {k_frag['average_fragmentation']:<15.3f} {e_frag['average_fragmentation']:<15.3f}")
    print(f"{'Max fragmentation':<30} {k_frag['max_fragmentation']:<15.3f} {e_frag['max_fragmentation']:<15.3f}")
    
    # Semantic coherence comparison
    print("\nSemantic Coherence:")
    k_sem = kmeans_analysis.get('semantic_coherence', {})
    e_sem = ets_analysis.get('semantic_coherence', {})
    
    if k_sem and e_sem:
        print(f"{'Avg coherence':<30} {k_sem.get('average_coherence', 0):<15.3f} {e_sem.get('average_coherence', 0):<15.3f}")
    
    # Save comparison
    comparison = {
        'kmeans_summary': {
            'total_paths': k_paths['total_unique_paths'],
            'avg_diversity': k_paths['average_path_diversity'],
            'path_coherence': k_paths['path_coherence'],
            'avg_fragmentation': k_frag['average_fragmentation']
        },
        'ets_summary': {
            'total_paths': e_paths['total_unique_paths'],
            'avg_diversity': e_paths['average_path_diversity'],
            'path_coherence': e_paths['path_coherence'],
            'avg_fragmentation': e_frag['average_fragmentation']
        }
    }
    
    comparison_file = output_dir / "apa_metrics_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {comparison_file}")

if __name__ == "__main__":
    main()