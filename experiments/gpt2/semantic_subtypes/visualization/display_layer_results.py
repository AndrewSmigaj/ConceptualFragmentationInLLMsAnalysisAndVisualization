#!/usr/bin/env python3
"""
Display the optimal clustering results per layer.
"""

# Based on the output we saw:
results = {
    'layer_0': {'optimal_k': 4, 'silhouette': 0.005, 'ets_threshold': 0.997150, 'ets_n_clusters': 4},
    'layer_1': {'optimal_k': 3, 'silhouette': 0.039, 'ets_threshold': 0.996400, 'ets_n_clusters': 3},
    'layer_2': {'optimal_k': 3, 'silhouette': 0.056, 'ets_threshold': 0.996700, 'ets_n_clusters': 4},
    'layer_3': {'optimal_k': 3, 'silhouette': 0.103, 'ets_threshold': 0.997150, 'ets_n_clusters': 3},
    'layer_4': {'optimal_k': 3, 'silhouette': 0.165, 'ets_threshold': 0.997300, 'ets_n_clusters': 3},
    'layer_5': {'optimal_k': 3, 'silhouette': 0.189, 'ets_threshold': 0.996700, 'ets_n_clusters': 3},
    'layer_6': {'optimal_k': 3, 'silhouette': 0.207, 'ets_threshold': 0.996700, 'ets_n_clusters': 3},
    'layer_7': {'optimal_k': 3, 'silhouette': 0.223, 'ets_threshold': 0.996700, 'ets_n_clusters': 3},
    'layer_8': {'optimal_k': 3, 'silhouette': 0.228, 'ets_threshold': 0.996700, 'ets_n_clusters': 3},
    'layer_9': {'optimal_k': 3, 'silhouette': 0.231, 'ets_threshold': 0.996700, 'ets_n_clusters': 3},
    'layer_10': {'optimal_k': 3, 'silhouette': 0.232, 'ets_threshold': 0.996700, 'ets_n_clusters': 3},
    'layer_11': {'optimal_k': 3, 'silhouette': 0.231, 'ets_threshold': 0.996100, 'ets_n_clusters': 3},
    'layer_12': {'optimal_k': 3, 'silhouette': 0.176, 'ets_threshold': 0.996550, 'ets_n_clusters': 3}
}

print("="*80)
print("OPTIMAL CLUSTERING CONFIGURATION PER LAYER")
print("="*80)
print()
print(f"{'Layer':<8} {'K-means k':<12} {'Silhouette':<12} {'ETS Threshold':<15} {'ETS Clusters':<12}")
print("-"*70)

for layer_idx in range(13):
    layer_key = f'layer_{layer_idx}'
    config = results[layer_key]
    print(f"{layer_idx:<8} {config['optimal_k']:<12} {config['silhouette']:<12.3f} "
          f"{config['ets_threshold']:<15.6f} {config['ets_n_clusters']:<12}")

print()
print("KEY OBSERVATIONS:")
print("-"*40)
print("1. K-means elbow method suggests k=3 for almost all layers (except layer 0 with k=4)")
print("2. Silhouette scores increase from early to middle layers, peak around layers 9-10")
print("3. Most common ETS threshold is 0.996700 (used in layers 5-10)")
print("4. Layer 2 has a mismatch: K-means suggests 3 but ETS gives 4 clusters")

# Save as JSON
import json
with open("layer_clustering_config.json", 'w') as f:
    json.dump(results, f, indent=2)
print("\nConfiguration saved to: layer_clustering_config.json")