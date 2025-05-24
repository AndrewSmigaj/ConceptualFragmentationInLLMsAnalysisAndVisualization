#!/usr/bin/env python3
"""
Run semantic subtypes experiment with optimal clustering configurations per layer.
Uses elbow method results for K-means and matching ETS thresholds.
"""

import json
import pickle
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from experiments.gpt2.pivot.gpt2_pivot_clusterer import GPT2PivotClusterer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_dimension_thresholds,
    compute_similarity_matrix,
    compute_ets_clustering
)

# Layer-specific configurations from elbow analysis
LAYER_CONFIGS = {
    0: {'k': 4, 'ets_threshold': 0.997150},
    1: {'k': 3, 'ets_threshold': 0.996400},
    2: {'k': 3, 'ets_threshold': 0.996700},  # Note: ETS gives 4 clusters
    3: {'k': 3, 'ets_threshold': 0.997150},
    4: {'k': 3, 'ets_threshold': 0.997300},
    5: {'k': 3, 'ets_threshold': 0.996700},
    6: {'k': 3, 'ets_threshold': 0.996700},
    7: {'k': 3, 'ets_threshold': 0.996700},
    8: {'k': 3, 'ets_threshold': 0.996700},
    9: {'k': 3, 'ets_threshold': 0.996700},
    10: {'k': 3, 'ets_threshold': 0.996700},
    11: {'k': 3, 'ets_threshold': 0.996100},
    12: {'k': 3, 'ets_threshold': 0.996550}
}

def cluster_layer_kmeans(activations, k):
    """Cluster a layer using K-means with specified k."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(activations)
    centers = kmeans.cluster_centers_
    
    # Calculate silhouette score if valid
    if k > 1 and k < len(activations):
        silhouette = silhouette_score(activations, labels)
    else:
        silhouette = 0.0
    
    return labels, centers, silhouette

def cluster_layer_ets(activations, threshold):
    """Cluster a layer using ETS with specified threshold."""
    cluster_labels, dimension_thresholds = compute_ets_clustering(
        activations,
        threshold_percentile=threshold,
        min_threshold=1e-5,
        verbose=False,
        batch_size=100
    )
    
    # Calculate cluster centers
    unique_labels = sorted(set(cluster_labels))
    centers = []
    for label in unique_labels:
        mask = cluster_labels == label
        center = np.mean(activations[mask], axis=0)
        centers.append(center)
    
    # Calculate silhouette score if valid
    n_clusters = len(unique_labels)
    if n_clusters > 1 and n_clusters < len(activations):
        silhouette = silhouette_score(activations, cluster_labels)
    else:
        silhouette = 0.0
    
    return cluster_labels, centers, silhouette, n_clusters

def run_optimal_clustering(activations_file: str, output_dir: Path):
    """Run clustering with optimal configurations per layer."""
    
    print(f"Loading activations from {activations_file}...")
    with open(activations_file, 'rb') as f:
        activations_data = pickle.load(f)
    
    activations = activations_data['activations']
    sentences = activations_data['sentences']
    
    # Initialize results structures
    kmeans_results = {
        'sentences': sentences,
        'tokens': activations_data.get('tokens', []),
        'metadata': activations_data.get('metadata', {}),
        'layer_results': {}
    }
    
    ets_results = {
        'sentences': sentences,
        'tokens': activations_data.get('tokens', []),
        'metadata': activations_data.get('metadata', {}),
        'layer_results': {}
    }
    
    print("\n" + "="*60)
    print("Running optimal clustering for each layer")
    print("="*60)
    
    for layer_idx in range(13):
        print(f"\nLayer {layer_idx}:")
        config = LAYER_CONFIGS[layer_idx]
        
        # Extract layer activations
        layer_activations = []
        activation_indices = []  # Track sentence indices
        
        for sent_idx in sorted(activations.keys()):
            if isinstance(sent_idx, int):
                sent_data = activations[sent_idx]
                if 0 in sent_data:  # Token position 0
                    token_data = sent_data[0]
                    if layer_idx in token_data:
                        layer_activations.append(token_data[layer_idx])
                        activation_indices.append(sent_idx)
        
        layer_activations = np.array(layer_activations)
        
        # K-means clustering
        print(f"  K-means with k={config['k']}...", end='', flush=True)
        start_time = time.time()
        kmeans_labels, kmeans_centers, kmeans_silhouette = cluster_layer_kmeans(
            layer_activations, config['k']
        )
        kmeans_time = time.time() - start_time
        print(f" done ({kmeans_time:.2f}s, silhouette={kmeans_silhouette:.3f})")
        
        # Store K-means results
        kmeans_layer_result = {
            'optimal_k': config['k'],
            'silhouette_score': float(kmeans_silhouette),
            'cluster_centers': [center.tolist() for center in kmeans_centers],
            'cluster_labels': {},
            'layer_idx': layer_idx
        }
        
        # Map labels back to sentence indices
        for i, sent_idx in enumerate(activation_indices):
            if sent_idx not in kmeans_layer_result['cluster_labels']:
                kmeans_layer_result['cluster_labels'][sent_idx] = {}
            kmeans_layer_result['cluster_labels'][sent_idx][0] = int(kmeans_labels[i])
        
        kmeans_results['layer_results'][f'layer_{layer_idx}'] = kmeans_layer_result
        
        # ETS clustering
        print(f"  ETS with threshold={config['ets_threshold']:.6f}...", end='', flush=True)
        start_time = time.time()
        ets_labels, ets_centers, ets_silhouette, ets_n_clusters = cluster_layer_ets(
            layer_activations, config['ets_threshold']
        )
        ets_time = time.time() - start_time
        print(f" done ({ets_time:.2f}s, {ets_n_clusters} clusters, silhouette={ets_silhouette:.3f})")
        
        # Store ETS results
        ets_layer_result = {
            'threshold': config['ets_threshold'],
            'num_clusters': ets_n_clusters,
            'silhouette_score': float(ets_silhouette),
            'cluster_centers': [center.tolist() for center in ets_centers],
            'cluster_labels': {},
            'layer_idx': layer_idx
        }
        
        # Map labels back to sentence indices
        for i, sent_idx in enumerate(activation_indices):
            if sent_idx not in ets_layer_result['cluster_labels']:
                ets_layer_result['cluster_labels'][sent_idx] = {}
            ets_layer_result['cluster_labels'][sent_idx][0] = int(ets_labels[i])
        
        ets_results['layer_results'][f'layer_{layer_idx}'] = ets_layer_result
    
    # Save results
    kmeans_file = output_dir / "semantic_subtypes_kmeans_optimal.pkl"
    with open(kmeans_file, 'wb') as f:
        pickle.dump(kmeans_results, f)
    print(f"\nK-means results saved to: {kmeans_file}")
    
    ets_file = output_dir / "semantic_subtypes_ets_optimal.pkl"
    with open(ets_file, 'wb') as f:
        pickle.dump(ets_results, f)
    print(f"ETS results saved to: {ets_file}")
    
    return kmeans_results, ets_results

def generate_summary_report(kmeans_results, ets_results, output_dir):
    """Generate a summary report of the clustering results."""
    
    print("\nGenerating summary report...")
    
    report = {
        'experiment': 'semantic_subtypes_optimal_clustering',
        'timestamp': datetime.now().isoformat(),
        'configuration': LAYER_CONFIGS,
        'results': {
            'kmeans': {},
            'ets': {}
        }
    }
    
    # Summarize K-means results
    for layer_idx in range(13):
        layer_key = f'layer_{layer_idx}'
        if layer_key in kmeans_results['layer_results']:
            layer_data = kmeans_results['layer_results'][layer_key]
            report['results']['kmeans'][layer_key] = {
                'k': layer_data['optimal_k'],
                'silhouette': layer_data['silhouette_score']
            }
    
    # Summarize ETS results
    for layer_idx in range(13):
        layer_key = f'layer_{layer_idx}'
        if layer_key in ets_results['layer_results']:
            layer_data = ets_results['layer_results'][layer_key]
            report['results']['ets'][layer_key] = {
                'threshold': layer_data['threshold'],
                'num_clusters': layer_data['num_clusters'],
                'silhouette': layer_data['silhouette_score']
            }
    
    # Save report
    report_file = output_dir / "clustering_summary_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Summary report saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING SUMMARY")
    print("="*60)
    print(f"{'Layer':<8} {'K-means':<20} {'ETS':<30}")
    print(f"{'':8} {'k':>3} {'silhouette':>12} {'threshold':>10} {'clusters':>8} {'silhouette':>10}")
    print("-"*60)
    
    for layer_idx in range(13):
        layer_key = f'layer_{layer_idx}'
        k_data = report['results']['kmeans'].get(layer_key, {})
        e_data = report['results']['ets'].get(layer_key, {})
        
        print(f"{layer_idx:<8} {k_data.get('k', '-'):>3} "
              f"{k_data.get('silhouette', 0):>12.3f} "
              f"{e_data.get('threshold', 0):>10.6f} "
              f"{e_data.get('num_clusters', '-'):>8} "
              f"{e_data.get('silhouette', 0):>10.3f}")

def main():
    """Run the optimal clustering experiment."""
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"semantic_subtypes_optimal_experiment_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Use existing activations
    activations_file = "semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl"
    
    # Run clustering
    kmeans_results, ets_results = run_optimal_clustering(activations_file, output_dir)
    
    # Generate summary report
    generate_summary_report(kmeans_results, ets_results, output_dir)
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()