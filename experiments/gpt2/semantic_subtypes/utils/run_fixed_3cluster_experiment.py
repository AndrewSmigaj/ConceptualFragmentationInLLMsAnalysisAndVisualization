#!/usr/bin/env python3
"""
Run semantic subtypes experiment with fixed 3 clusters for both K-means and ETS.
- K-means: k=3 for all layers
- ETS: threshold=0.997 for all layers
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from experiments.gpt2.pivot.gpt2_pivot_clusterer import GPT2PivotClusterer

def run_fixed_clustering(activations_file: str, output_dir: Path):
    """Run clustering with fixed 3 clusters."""
    
    print(f"Loading activations from {activations_file}...")
    with open(activations_file, 'rb') as f:
        activations_data = pickle.load(f)
    
    # Run K-means with k=3
    print("\n" + "="*60)
    print("Running K-means clustering with k=3 for all layers")
    print("="*60)
    
    kmeans_clusterer = GPT2PivotClusterer(
        k_range=(3, 3),  # Fixed k=3
        random_state=42,
        clustering_method='kmeans'
    )
    
    kmeans_results = kmeans_clusterer.cluster_all_layers(activations_data)
    
    # Save K-means results
    kmeans_file = output_dir / "semantic_subtypes_kmeans_k3_clustering.pkl"
    kmeans_clusterer.save_results(kmeans_results, str(kmeans_file))
    print(f"K-means results saved to: {kmeans_file}")
    
    # Run ETS with threshold=0.997
    print("\n" + "="*60)
    print("Running ETS clustering with threshold=0.997 for all layers")
    print("="*60)
    
    # Set the threshold percentile for ETS
    ets_clusterer = GPT2PivotClusterer(
        k_range=(2, 15),  # Not used for ETS
        random_state=42,
        clustering_method='ets',
        threshold_percentile=0.997  # Fixed threshold
    )
    
    ets_results = ets_clusterer.cluster_all_layers(activations_data)
    
    # Save ETS results
    ets_file = output_dir / "semantic_subtypes_ets_997_clustering.pkl"
    ets_clusterer.save_results(ets_results, str(ets_file))
    print(f"ETS results saved to: {ets_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    
    # K-means summary
    print("\nK-means (k=3):")
    for layer_idx in range(13):  # GPT-2 has 13 layers (0-12)
        layer_key = f"layer_{layer_idx}"
        if layer_key in kmeans_results['layer_results']:
            layer_data = kmeans_results['layer_results'][layer_key]
            k = layer_data.get('optimal_k', 'N/A')
            score = layer_data.get('silhouette_score', 0.0)
            print(f"  Layer {layer_idx}: k={k}, silhouette={score:.3f}")
    
    # ETS summary
    print("\nETS (threshold=0.997):")
    for layer_idx in range(13):
        layer_key = f"layer_{layer_idx}"
        if layer_key in ets_results['layer_results']:
            layer_data = ets_results['layer_results'][layer_key]
            # Count actual clusters
            if 'cluster_labels' in layer_data:
                all_clusters = set()
                for sent_clusters in layer_data['cluster_labels'].values():
                    for cluster_id in sent_clusters.values():
                        all_clusters.add(cluster_id)
                n_clusters = len(all_clusters)
                score = layer_data.get('silhouette_score', 0.0)
                print(f"  Layer {layer_idx}: {n_clusters} clusters, silhouette={score:.3f}")
    
    return kmeans_results, ets_results

def main():
    """Run the fixed 3-cluster experiment."""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"semantic_subtypes_3cluster_experiment_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Use existing activations
    activations_file = "semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl"
    
    # Run clustering
    kmeans_results, ets_results = run_fixed_clustering(activations_file, output_dir)
    
    # Save experiment config
    config = {
        'experiment': 'semantic_subtypes_3cluster',
        'timestamp': timestamp,
        'parameters': {
            'kmeans': {'k': 3, 'method': 'fixed'},
            'ets': {'threshold_percentile': 0.997, 'method': 'fixed'},
            'random_state': 42
        },
        'activations_file': activations_file,
        'output_files': {
            'kmeans': str(output_dir / "semantic_subtypes_kmeans_k3_clustering.pkl"),
            'ets': str(output_dir / "semantic_subtypes_ets_997_clustering.pkl")
        }
    }
    
    config_file = output_dir / "experiment_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nExperiment config saved to: {config_file}")
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()