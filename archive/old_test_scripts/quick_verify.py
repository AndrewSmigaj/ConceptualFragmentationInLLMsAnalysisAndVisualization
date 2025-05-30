import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load just layer 5
activations = np.load('top_10k_activations.npy')
layer_5 = activations[:, 5, :]

print(f"Testing layer 5 (shape: {layer_5.shape})")
print(f"Data range: [{layer_5.min():.4f}, {layer_5.max():.4f}]")

# Quick test
for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = kmeans.fit_predict(layer_5)
    score = silhouette_score(layer_5, labels, sample_size=1000)
    print(f"k={k}: silhouette={score:.4f}")

# Check k=2 cluster balance
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=3)
labels_2 = kmeans_2.fit_predict(layer_5)
unique, counts = np.unique(labels_2, return_counts=True)
print(f"\nk=2 cluster sizes: {counts}")
print(f"Ratio: {max(counts)/min(counts):.2f}:1")