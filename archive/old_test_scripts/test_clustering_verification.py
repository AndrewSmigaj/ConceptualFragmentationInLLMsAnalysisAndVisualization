#!/usr/bin/env python3
"""Verify the k=2 clustering results are legitimate."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load activations
print("Loading activations...")
activations = np.load('top_10k_activations.npy')
print(f'Activations shape: {activations.shape}')
print(f'Data type: {activations.dtype}')
print(f'Value range: [{activations.min():.4f}, {activations.max():.4f}]')

# Analyze data distribution
print("\nPer-layer statistics:")
for layer_idx in range(12):
    layer_acts = activations[:, layer_idx, :]
    print(f'Layer {layer_idx}: mean={layer_acts.mean():.4f}, std={layer_acts.std():.4f}, '
          f'min={layer_acts.min():.4f}, max={layer_acts.max():.4f}')

# Test multiple k values on a few layers
print("\nManual clustering verification:")
test_layers = [0, 5, 11]  # Early, middle, late

for layer_idx in test_layers:
    print(f"\nLayer {layer_idx}:")
    layer_acts = activations[:, layer_idx, :]
    
    # Test different k values
    for k in [2, 3, 4, 5, 10, 20]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(layer_acts)
        score = silhouette_score(layer_acts, labels, sample_size=min(5000, len(labels)))
        print(f'  k={k}: silhouette={score:.4f}')
    
    # Visualize with PCA for k=2
    print(f"\n  Analyzing k=2 clustering for layer {layer_idx}:")
    kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_2 = kmeans_2.fit_predict(layer_acts)
    
    # Count cluster sizes
    unique, counts = np.unique(labels_2, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"    Cluster {cluster}: {count} tokens ({count/len(labels_2)*100:.1f}%)")
    
    # Check if clusters are balanced or severely imbalanced
    imbalance_ratio = max(counts) / min(counts)
    print(f"    Imbalance ratio: {imbalance_ratio:.2f}")

# Additional check: are the activations degenerate?
print("\nChecking for degenerate activations:")
for layer_idx in range(12):
    layer_acts = activations[:, layer_idx, :]
    
    # Check variance along each dimension
    dim_vars = np.var(layer_acts, axis=0)
    zero_var_dims = np.sum(dim_vars < 1e-10)
    low_var_dims = np.sum(dim_vars < 1e-5)
    
    print(f"Layer {layer_idx}: {zero_var_dims} zero-variance dims, "
          f"{low_var_dims} low-variance dims out of {layer_acts.shape[1]}")

print("\nConclusion: The k=2 results appear to be", end=" ")
if imbalance_ratio < 3:
    print("LEGITIMATE - clusters are reasonably balanced")
else:
    print("SUSPICIOUS - clusters are highly imbalanced")