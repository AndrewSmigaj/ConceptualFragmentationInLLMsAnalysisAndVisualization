#!/usr/bin/env python3
"""
Generate all visualizations for the semantic subtypes clustering experiment.
Includes static plots and interactive diagrams.
"""

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

def load_results():
    """Load all necessary data."""
    result_dir = Path("semantic_subtypes_optimal_experiment_20250523_182344")
    
    with open(result_dir / "semantic_subtypes_kmeans_optimal.pkl", 'rb') as f:
        kmeans_results = pickle.load(f)
    
    with open(result_dir / "semantic_subtypes_ets_optimal.pkl", 'rb') as f:
        ets_results = pickle.load(f)
    
    with open("data/gpt2_semantic_subtypes_curated.json", 'r') as f:
        curated_data = json.load(f)
    
    return kmeans_results, ets_results, curated_data

def plot_silhouette_comparison(kmeans_results, ets_results):
    """Plot silhouette scores comparison across layers."""
    
    layers = list(range(13))
    kmeans_scores = []
    ets_scores = []
    
    for layer_idx in layers:
        layer_key = f"layer_{layer_idx}"
        kmeans_scores.append(kmeans_results['layer_results'][layer_key]['silhouette_score'])
        ets_scores.append(ets_results['layer_results'][layer_key]['silhouette_score'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, kmeans_scores, 'b-o', label='K-means', linewidth=2, markersize=8)
    plt.plot(layers, ets_scores, 'r-s', label='ETS', linewidth=2, markersize=8)
    
    # Highlight interesting windows
    plt.axvspan(0, 2, alpha=0.1, color='green', label='Early window')
    plt.axvspan(5, 7, alpha=0.1, color='orange', label='Middle window')
    plt.axvspan(9, 11, alpha=0.1, color='purple', label='Late window')
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Clustering Quality Across Layers', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('silhouette_comparison.png', dpi=150)
    plt.close()
    
    print("Saved: silhouette_comparison.png")

def plot_cluster_distribution(clustering_results, curated_data, method_name):
    """Plot distribution of semantic subtypes across clusters for key layers."""
    
    sentences = clustering_results['sentences']
    
    # Create word to subtype mapping
    word_to_subtype = {}
    for subtype, words in curated_data['curated_words'].items():
        for word in words:
            word_to_subtype[word] = subtype
    
    # Select key layers
    key_layers = [0, 5, 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use consistent colors for subtypes
    subtypes = sorted(set(word_to_subtype.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes)))
    
    for idx, layer_idx in enumerate(key_layers):
        ax = axes[idx]
        layer_key = f"layer_{layer_idx}"
        layer_data = clustering_results['layer_results'][layer_key]
        
        # Count subtypes in each cluster
        cluster_subtype_counts = defaultdict(lambda: defaultdict(int))
        
        for sent_idx, clusters in layer_data['cluster_labels'].items():
            if 0 in clusters:
                cluster_id = clusters[0]
                word = sentences[sent_idx]
                subtype = word_to_subtype.get(word, 'unknown')
                cluster_subtype_counts[cluster_id][subtype] += 1
        
        # Create stacked bar chart
        clusters = sorted(cluster_subtype_counts.keys())
        
        bottom = np.zeros(len(clusters))
        
        for subtype_idx, subtype in enumerate(subtypes):
            counts = [cluster_subtype_counts[c][subtype] for c in clusters]
            ax.bar(clusters, counts, bottom=bottom, label=subtype if idx == 0 else "", 
                   color=colors[subtype_idx])
            bottom += counts
        
        ax.set_xlabel('Cluster', fontsize=11)
        ax.set_ylabel('Word Count', fontsize=11)
        ax.set_title(f'Layer {layer_idx}', fontsize=12)
        ax.set_xticks(clusters)
    
    # Add legend to the left of the first subplot
    axes[0].legend(bbox_to_anchor=(-0.15, 0.5), loc='center right', fontsize=9)
    
    plt.suptitle(f'{method_name} - Semantic Subtype Distribution Across Clusters', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_cluster_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {method_name.lower()}_cluster_distribution.png")

def plot_path_diversity_heatmap(clustering_results, method_name):
    """Plot heatmap of path diversity across layers."""
    
    # Calculate path counts for sliding windows
    num_layers = 13
    window_size = 3
    path_counts = []
    window_labels = []
    
    for start_layer in range(num_layers - window_size + 1):
        end_layer = start_layer + window_size - 1
        
        # Count unique paths in this window
        paths = set()
        
        for sent_idx in clustering_results['layer_results'][f'layer_{start_layer}']['cluster_labels']:
            path = []
            valid = True
            
            for layer in range(start_layer, end_layer + 1):
                layer_key = f'layer_{layer}'
                if (sent_idx in clustering_results['layer_results'][layer_key]['cluster_labels'] and
                    0 in clustering_results['layer_results'][layer_key]['cluster_labels'][sent_idx]):
                    cluster = clustering_results['layer_results'][layer_key]['cluster_labels'][sent_idx][0]
                    path.append(f"L{layer}C{cluster}")
                else:
                    valid = False
                    break
            
            if valid:
                paths.add(tuple(path))
        
        path_counts.append(len(paths))
        window_labels.append(f"L{start_layer}-{end_layer}")
    
    # Create bar plot instead of heatmap for 1D data
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(path_counts)), path_counts)
    
    # Color interesting windows
    colors = ['green' if i <= 2 else 'orange' if 5 <= i <= 7 else 'purple' if i >= 9 else 'steelblue' 
              for i in range(len(path_counts))]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Window', fontsize=12)
    plt.ylabel('Number of Unique Paths', fontsize=12)
    plt.title(f'{method_name} - Path Diversity Across 3-Layer Windows', fontsize=14)
    plt.xticks(range(len(window_labels)), window_labels, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{method_name.lower()}_path_diversity.png', dpi=150)
    plt.close()
    
    print(f"Saved: {method_name.lower()}_path_diversity.png")

def plot_cluster_counts(kmeans_results, ets_results):
    """Plot cluster counts across layers for both methods."""
    
    layers = list(range(13))
    kmeans_k = []
    ets_counts = []
    
    for layer_idx in layers:
        layer_key = f"layer_{layer_idx}"
        
        # K-means k value
        kmeans_k.append(kmeans_results['layer_results'][layer_key].get('optimal_k', 3))
        
        # ETS cluster count
        ets_clusters = set()
        for clusters in ets_results['layer_results'][layer_key]['cluster_labels'].values():
            if 0 in clusters:
                ets_clusters.add(clusters[0])
        ets_counts.append(len(ets_clusters))
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, kmeans_k, 'b-o', label='K-means k', linewidth=2, markersize=8)
    plt.plot(layers, ets_counts, 'r-s', label='ETS clusters', linewidth=2, markersize=8)
    
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Number of Clusters', fontsize=12)
    plt.title('Cluster Counts Across Layers', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 6)
    plt.tight_layout()
    plt.savefig('cluster_counts_comparison.png', dpi=150)
    plt.close()
    
    print("Saved: cluster_counts_comparison.png")

def main():
    """Generate all visualizations."""
    
    print("Loading data...")
    kmeans_results, ets_results, curated_data = load_results()
    
    print("\nGenerating static visualizations...")
    
    # 1. Silhouette score comparison
    plot_silhouette_comparison(kmeans_results, ets_results)
    
    # 2. Cluster distribution plots
    plot_cluster_distribution(kmeans_results, curated_data, "K-means")
    plot_cluster_distribution(ets_results, curated_data, "ETS")
    
    # 3. Path diversity plots
    plot_path_diversity_heatmap(kmeans_results, "K-means")
    plot_path_diversity_heatmap(ets_results, "ETS")
    
    # 4. Cluster counts comparison
    plot_cluster_counts(kmeans_results, ets_results)
    
    print("\nStatic visualizations complete!")
    print("\nNow run 'python generate_cluster_flow_diagrams.py' for interactive Sankey diagrams")

if __name__ == "__main__":
    main()