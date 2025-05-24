#!/usr/bin/env python3
"""
ETS elbow analysis per layer to find optimal thresholds for each layer.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from pathlib import Path
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.metrics.explainable_threshold_similarity import compute_ets_clustering

def extract_layer_activations(activations_data, layer_idx):
    """Extract activations for a specific layer."""
    layer_activations = []
    for sent_idx in range(len(activations_data['sentences'])):
        if sent_idx in activations_data['activations']:
            if 0 in activations_data['activations'][sent_idx]:
                if layer_idx in activations_data['activations'][sent_idx][0]:
                    layer_activations.append(activations_data['activations'][sent_idx][0][layer_idx])
    
    return np.array(layer_activations)

def find_elbow_point(thresholds, clusters):
    """Find elbow using maximum distance from line method."""
    if len(thresholds) < 3:
        return 0, thresholds[0] if thresholds else 0.1
    
    # Normalize the data
    x = np.array(thresholds)
    y = np.array(clusters)
    
    # Calculate distances from line connecting first and last point
    distances = []
    for i in range(1, len(x)-1):
        # Vector from first to last point
        line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
        # Vector from first to current point
        point_vec = np.array([x[i] - x[0], y[i] - y[0]])
        
        # Distance from point to line (cross product method)
        if np.linalg.norm(line_vec) > 0:
            distance = abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
        else:
            distance = 0
        distances.append(distance)
    
    # Find maximum distance (elbow point)
    if distances:
        elbow_idx = np.argmax(distances) + 1
        return elbow_idx, thresholds[elbow_idx]
    else:
        return 0, thresholds[0]

def analyze_layer_ets_thresholds():
    """Analyze optimal ETS thresholds per layer."""
    
    print("=== ETS Per-Layer Threshold Analysis ===\n")
    
    # Load activations
    print("Loading activations...")
    with open("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_activations.pkl", 'rb') as f:
        activations_data = pickle.load(f)
    
    print(f"Loaded {len(activations_data['sentences'])} sentences")
    
    # Test thresholds - focus on the range where clusters actually drop
    test_thresholds = [0.95, 0.98, 0.985, 0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999]
    
    # Results storage
    layer_results = {}
    optimal_thresholds = {}
    
    # Analyze each layer
    for layer_idx in range(12):  # Layers 0-11
        print(f"\n{'='*50}")
        print(f"LAYER {layer_idx}")
        print('='*50)
        
        # Extract layer activations
        print(f"Extracting layer {layer_idx} activations...")
        layer_data = extract_layer_activations(activations_data, layer_idx)
        print(f"  Shape: {layer_data.shape}")
        print(f"  Value range: [{layer_data.min():.3f}, {layer_data.max():.3f}]")
        
        # Test different thresholds
        layer_clusters = []
        layer_thresholds = []
        
        print(f"\nTesting thresholds on layer {layer_idx}:")
        for i, threshold in enumerate(test_thresholds):
            print(f"  [{i+1}/{len(test_thresholds)}] Testing {threshold}... ", end='', flush=True)
            
            try:
                start_time = time.time()
                cluster_labels, _ = compute_ets_clustering(
                    layer_data,
                    threshold_percentile=threshold,
                    min_threshold=1e-5,
                    verbose=False
                )
                elapsed = time.time() - start_time
                
                n_clusters = len(set(cluster_labels))
                layer_clusters.append(n_clusters)
                layer_thresholds.append(threshold)
                
                print(f"{n_clusters} clusters ({elapsed:.2f}s)")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        # Find elbow point for this layer
        if len(layer_thresholds) >= 3:
            elbow_idx, optimal_threshold = find_elbow_point(layer_thresholds, layer_clusters)
            optimal_clusters = layer_clusters[elbow_idx]
            
            print(f"\n  LAYER {layer_idx} OPTIMAL: threshold={optimal_threshold:.3f}, clusters={optimal_clusters}")
            
            optimal_thresholds[layer_idx] = {
                'threshold': optimal_threshold,
                'clusters': optimal_clusters,
                'all_thresholds': layer_thresholds,
                'all_clusters': layer_clusters
            }
        else:
            print(f"\n  LAYER {layer_idx}: Not enough data points for elbow analysis")
            optimal_thresholds[layer_idx] = {
                'threshold': 0.99,  # Default
                'clusters': layer_clusters[0] if layer_clusters else 2,
                'all_thresholds': layer_thresholds,
                'all_clusters': layer_clusters
            }
        
        layer_results[layer_idx] = {
            'thresholds': layer_thresholds,
            'clusters': layer_clusters
        }
    
    # Generate summary
    print(f"\n{'='*60}")
    print("SUMMARY: OPTIMAL THRESHOLDS PER LAYER")
    print('='*60)
    
    for layer_idx in sorted(optimal_thresholds.keys()):
        opt = optimal_thresholds[layer_idx]
        print(f"Layer {layer_idx:2d}: threshold={opt['threshold']:.3f} → {opt['clusters']:3d} clusters")
    
    # Save results
    results_file = "ets_per_layer_analysis.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump({
            'layer_results': layer_results,
            'optimal_thresholds': optimal_thresholds,
            'test_thresholds': test_thresholds
        }, f)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate visualization
    print("\nGenerating visualization...")
    try:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.flatten()
        
        for layer_idx in range(12):
            ax = axes[layer_idx]
            
            if layer_idx in layer_results:
                thresholds = layer_results[layer_idx]['thresholds']
                clusters = layer_results[layer_idx]['clusters']
                
                ax.plot(thresholds, clusters, 'b-o', markersize=4)
                
                # Mark optimal point
                if layer_idx in optimal_thresholds:
                    opt_thresh = optimal_thresholds[layer_idx]['threshold']
                    opt_clusters = optimal_thresholds[layer_idx]['clusters']
                    ax.plot(opt_thresh, opt_clusters, 'ro', markersize=8, label=f'Optimal: {opt_thresh:.3f}')
                    ax.legend(fontsize=8)
                
                ax.set_title(f'Layer {layer_idx}', fontsize=10)
                ax.set_xlabel('Threshold', fontsize=8)
                ax.set_ylabel('Clusters', fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Layer {layer_idx}', fontsize=10)
        
        plt.tight_layout()
        viz_file = "ets_per_layer_elbow_analysis.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {viz_file}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Analysis insights
    print(f"\n{'='*60}")
    print("ANALYSIS INSIGHTS")
    print('='*60)
    
    thresholds_list = [optimal_thresholds[i]['threshold'] for i in range(12) if i in optimal_thresholds]
    clusters_list = [optimal_thresholds[i]['clusters'] for i in range(12) if i in optimal_thresholds]
    
    if thresholds_list:
        print(f"Threshold range: {min(thresholds_list):.3f} to {max(thresholds_list):.3f}")
        print(f"Cluster range: {min(clusters_list)} to {max(clusters_list)}")
        print(f"Mean threshold: {np.mean(thresholds_list):.3f}")
        print(f"Mean clusters: {np.mean(clusters_list):.1f}")
        
        # Show trend across layers
        print(f"\nLayer progression:")
        for i in range(0, 12, 2):
            if i in optimal_thresholds and i+1 in optimal_thresholds:
                print(f"  Layers {i}-{i+1}: thresholds {optimal_thresholds[i]['threshold']:.3f}-{optimal_thresholds[i+1]['threshold']:.3f}, "
                      f"clusters {optimal_thresholds[i]['clusters']}-{optimal_thresholds[i+1]['clusters']}")
    
    return optimal_thresholds

if __name__ == "__main__":
    optimal_thresholds = analyze_layer_ets_thresholds()
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("Next step: Run experiment with layer-specific optimal thresholds")
    print('='*60)