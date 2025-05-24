#!/usr/bin/env python3
"""
Refine search around 0.99 percentile to find optimal ETS threshold.
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def test_threshold_detailed(percentile, layer_11_activations, words, word_to_subtype):
    """Test threshold and return detailed metrics."""
    from gpt2_pivot_clusterer import GPT2PivotClusterer
    from collections import Counter
    
    clusterer = GPT2PivotClusterer(
        clustering_method='ets',
        threshold_percentile=percentile,
        random_state=42
    )
    
    if clusterer._setup_sklearn() and clusterer.ets_available:
        try:
            labels, centers, n_clusters, silhouette = clusterer._cluster_with_ets(layer_11_activations)
            
            # Calculate semantic coherence
            cluster_to_subtypes = {}
            for label, word in zip(labels, words):
                if label not in cluster_to_subtypes:
                    cluster_to_subtypes[label] = []
                subtype = word_to_subtype.get(word, 'unknown')
                cluster_to_subtypes[label].append(subtype)
            
            # Average purity
            total_purity = 0
            for subtypes in cluster_to_subtypes.values():
                if subtypes:
                    most_common = Counter(subtypes).most_common(1)[0][1]
                    purity = most_common / len(subtypes)
                    total_purity += purity
            
            avg_purity = total_purity / len(cluster_to_subtypes) if cluster_to_subtypes else 0
            
            return {
                'n_clusters': n_clusters,
                'silhouette': silhouette,
                'purity': avg_purity,
                'quality': 0.4 * silhouette + 0.6 * avg_purity
            }
        except Exception as e:
            print(f"Error at {percentile}: {str(e)[:40]}")
            return None
    return None

def refine_search():
    """Refine search around 0.99 percentile."""
    # Load data
    results_dir = Path("semantic_subtypes_experiment_20250523_111112")
    with open(results_dir / "semantic_subtypes_activations.pkl", 'rb') as f:
        activations_data = pickle.load(f)
    
    word_to_subtype = activations_data.get('word_to_subtype', {})
    
    # Extract layer 11 activations and words
    layer_11_activations = []
    words = []
    for sent_idx in sorted(activations_data['activations'].keys()):
        if 0 in activations_data['activations'][sent_idx]:
            if 11 in activations_data['activations'][sent_idx][0]:
                layer_11_activations.append(
                    activations_data['activations'][sent_idx][0][11]
                )
                if sent_idx < len(activations_data['sentences']):
                    words.append(activations_data['sentences'][sent_idx])
    
    print(f"Refined search around 0.99 percentile ({len(layer_11_activations)} samples)")
    print("\nPercentile | Clusters | Silhouette | Purity | Quality")
    print("-" * 60)
    
    # Test range from 0.985 to 0.994
    test_points = np.arange(0.985, 0.9945, 0.001)
    
    results = []
    
    for percentile in test_points:
        print(f"  {percentile:.3f}    | ", end="", flush=True)
        metrics = test_threshold_detailed(percentile, layer_11_activations, words, word_to_subtype)
        
        if metrics:
            print(f"{metrics['n_clusters']:4d}     |   {metrics['silhouette']:.3f}    | "
                  f"{metrics['purity']:.3f}  | {metrics['quality']:.3f}")
            results.append((percentile, metrics))
        else:
            print("Failed")
    
    # Find best in reasonable range (20-80 clusters)
    good_results = [(p, m) for p, m in results if 20 <= m['n_clusters'] <= 80]
    
    if good_results:
        # Sort by quality
        best = max(good_results, key=lambda x: x[1]['quality'])
        print(f"\n🎯 OPTIMAL THRESHOLD: {best[0]:.3f}")
        print(f"   Clusters: {best[1]['n_clusters']}")
        print(f"   Silhouette: {best[1]['silhouette']:.3f}")
        print(f"   Semantic Purity: {best[1]['purity']:.3f}")
        print(f"   Overall Quality: {best[1]['quality']:.3f}")
        
        return best[0]
    
    return None

if __name__ == "__main__":
    optimal = refine_search()