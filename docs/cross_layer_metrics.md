# Cross-Layer Metrics for Neural Networks

The cross-layer metrics module provides tools for analyzing relationships between cluster representations across different layers of a neural network. These metrics help understand how concepts evolve and transform as they propagate through the network.

## Overview

Traditional neural network analysis often focuses on individual layers in isolation. Cross-layer metrics provide a complementary perspective by examining the relationships between representations at different depths of the network. This approach can reveal:

- How concepts are transformed between layers
- Which clusters are related across layers
- How coherent or fragmented the representation paths are
- The flow of information through the network

## Available Metrics

### Centroid Similarity (ρᶜ)

Measures the similarity between cluster centroids across different layers, revealing which clusters in different layers might represent related concepts.

```python
from concept_fragmentation.metrics.cross_layer_metrics import compute_centroid_similarity

# layer_clusters is a dictionary mapping layer names to cluster info dictionaries
similarity_matrices = compute_centroid_similarity(
    layer_clusters, 
    similarity_metric="cosine"  # Options: "cosine", "euclidean", "correlation"
)

# similarity_matrices maps layer pairs to similarity matrices
# e.g., similarity_matrices[("layer1", "layer2")] is a matrix where
# element [i,j] is the similarity between cluster i in layer1 and cluster j in layer2
```

### Membership Overlap (J)

Quantifies how datapoints from one cluster are distributed across clusters in another layer, showing the consistency of cluster assignments.

```python
from concept_fragmentation.metrics.cross_layer_metrics import compute_membership_overlap

# layer_clusters is a dictionary mapping layer names to cluster info dictionaries
overlap_matrices = compute_membership_overlap(
    layer_clusters, 
    normalize=True  # Whether to normalize by cluster sizes
)

# overlap_matrices maps layer pairs to overlap matrices
# e.g., overlap_matrices[("layer1", "layer2")] is a matrix where
# element [i,j] is the proportion of points in cluster i of layer1 
# that are also in cluster j of layer2
```

### Trajectory Fragmentation (F)

Measures how coherent or dispersed a datapoint's path is through the network. Lower values indicate more coherent paths.

```python
from concept_fragmentation.metrics.cross_layer_metrics import compute_trajectory_fragmentation

# layer_clusters is a dictionary mapping layer names to cluster info dictionaries
# class_labels is a numpy array of ground truth class labels
fragmentation_scores = compute_trajectory_fragmentation(
    layer_clusters, 
    class_labels
)

# fragmentation_scores maps layer names to fragmentation scores
# Higher scores indicate more fragmentation (class members scattered across clusters)
```

### Inter-Cluster Path Density

Analyzes the connectivity patterns between clusters in adjacent layers, creating a graph representation of concept flow.

```python
from concept_fragmentation.metrics.cross_layer_metrics import compute_path_density

# layer_clusters is a dictionary mapping layer names to cluster info dictionaries
density_scores, path_graph = compute_path_density(
    layer_clusters, 
    min_overlap=0.1  # Minimum overlap required to consider clusters connected
)

# density_scores maps layer pairs to density scores (higher means more connections)
# path_graph is a NetworkX graph representing the cluster connectivity
```

## Comprehensive Analysis

You can compute all metrics at once using the `analyze_cross_layer_metrics` function:

```python
from concept_fragmentation.metrics.cross_layer_metrics import analyze_cross_layer_metrics

# layer_clusters is a dictionary mapping layer names to cluster info dictionaries
# class_labels is a numpy array of ground truth class labels
results = analyze_cross_layer_metrics(
    layer_clusters,
    class_labels=class_labels,
    config={
        "similarity_metric": "cosine",
        "min_overlap": 0.1
    }
)

# results contains all metrics in a single dictionary
```

## Visualization

The framework includes visualization tools for all cross-layer metrics in the `visualization.cross_layer_viz` module:

```python
from visualization.cross_layer_viz import (
    plot_centroid_similarity_heatmap,
    plot_membership_overlap_sankey,
    plot_trajectory_fragmentation_bars,
    plot_path_density_network,
    create_cross_layer_dashboard
)

# Create heatmap of centroid similarity
heatmap_fig = plot_centroid_similarity_heatmap(
    results["centroid_similarity"],
    colorscale="Viridis"
)

# Create Sankey diagram of membership overlap
sankey_fig = plot_membership_overlap_sankey(
    results["membership_overlap"],
    layer_clusters,
    min_overlap=0.1
)

# Create bar chart of trajectory fragmentation
bar_fig = plot_trajectory_fragmentation_bars(
    results["trajectory_fragmentation"]
)

# Create network graph of path density
network_fig = plot_path_density_network(
    results["path_graph"],
    layout="multipartite"
)

# Create a comprehensive dashboard with all visualizations
dashboard_figures = create_cross_layer_dashboard(
    results,
    layer_clusters,
    min_overlap=0.1
)
```

## Interactive Dashboard

The cross-layer metrics are integrated into the main visualization dashboard, which can be launched using:

```bash
python visualization/main.py
```

In the dashboard:
1. Select the "Cross-Layer Metrics" tab
2. Choose a dataset and configuration
3. Select which metrics to display
4. Adjust visualization parameters 
5. Click "Update Cross-Layer Metrics" to generate the visualizations

## Interpretation Guidelines

When analyzing cross-layer metrics:

- **High centroid similarity** between clusters in different layers suggests they represent similar concepts
- **High membership overlap** indicates consistent cluster assignment across layers
- **Low trajectory fragmentation** suggests coherent concept representation throughout the network
- **Dense paths** between specific clusters suggest strong conceptual relationships

Use these metrics together to build a comprehensive understanding of how concepts are transformed and propagated through the network.

## References

For more details on the mathematical foundations of these metrics, see the `foundations.md` document in the project root.