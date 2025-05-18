# Implementation Plan for Archetypal Path Analysis

This document outlines the plan to implement the Archetypal Path Analysis (APA) framework described in "Foundations of Archetypal Path Analysis: Toward a Principled Geometry for Cluster-Based Interpretability."

## 1. Core Components to Implement

### 1.1 Explainable Threshold Similarity (ETS)

```python
# concept_fragmentation/metrics/explainable_threshold_similarity.py

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

def compute_ets_clustering(
    activations: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    threshold_percentile: float = 0.1,
    batch_size: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster activations using Explainable Threshold Similarity (ETS).
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        thresholds: Optional array of thresholds per dimension
        threshold_percentile: Percentile to use for automatic threshold calculation
        batch_size: Batch size for processing large datasets
        
    Returns:
        Tuple of (cluster_labels, thresholds)
    """
    n_samples, n_features = activations.shape
    
    # Compute thresholds if not provided
    if thresholds is None:
        # For each dimension, compute threshold as a percentile of pairwise differences
        thresholds = np.zeros(n_features)
        for j in range(n_features):
            # Get all pairwise differences for this dimension
            col = activations[:, j].reshape(-1, 1)
            diffs = np.abs(col - col.T)
            # Set threshold as percentile of non-zero differences
            non_zero_diffs = diffs[diffs > 0]
            if len(non_zero_diffs) > 0:
                thresholds[j] = np.percentile(non_zero_diffs, threshold_percentile * 100)
            else:
                thresholds[j] = 0.0
    
    # Process similarity matrix in batches for memory efficiency
    similarity_matrix = np.zeros((n_samples, n_samples), dtype=bool)
    
    for i in range(0, n_samples, batch_size):
        i_end = min(i + batch_size, n_samples)
        chunk_i = activations[i:i_end]
        
        for j in range(i, n_samples, batch_size):
            j_end = min(j + batch_size, n_samples)
            chunk_j = activations[j:j_end]
            
            # Compute pairwise differences for this batch
            for idx_i in range(i_end - i):
                for idx_j in range(j_end - j):
                    if i + idx_i <= j + idx_j:  # Only compute upper triangle
                        # Two points are similar if all dimension-wise differences are below thresholds
                        diffs = np.abs(chunk_i[idx_i] - chunk_j[idx_j])
                        is_similar = np.all(diffs <= thresholds)
                        
                        global_i = i + idx_i
                        global_j = j + idx_j
                        
                        similarity_matrix[global_i, global_j] = is_similar
                        similarity_matrix[global_j, global_i] = is_similar
    
    # Use connected components to form clusters
    from scipy.sparse.csgraph import connected_components
    n_clusters, cluster_labels = connected_components(similarity_matrix, directed=False)
    
    return cluster_labels, thresholds
```

### 1.2 Cluster Stability Metrics

```python
# concept_fragmentation/metrics/cluster_stability.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import adjusted_rand_score
from scipy.stats import wasserstein_distance

def compute_cluster_stability(
    cluster_assignments1: np.ndarray,
    cluster_assignments2: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute stability metrics between two clustering results.
    
    Args:
        cluster_assignments1: First set of cluster assignments
        cluster_assignments2: Second set of cluster assignments
        sample_weights: Optional sample weights for EMD calculation
        
    Returns:
        Dictionary of stability metrics (ARI, EMD)
    """
    # Compute Adjusted Rand Index
    ari = adjusted_rand_score(cluster_assignments1, cluster_assignments2)
    
    # Convert cluster labels to distribution for EMD
    unique_clusters1 = np.unique(cluster_assignments1)
    unique_clusters2 = np.unique(cluster_assignments2)
    
    dist1 = np.zeros(len(unique_clusters1))
    dist2 = np.zeros(len(unique_clusters2))
    
    for i, c in enumerate(unique_clusters1):
        if sample_weights is not None:
            dist1[i] = np.sum(sample_weights[cluster_assignments1 == c])
        else:
            dist1[i] = np.sum(cluster_assignments1 == c)
    
    for i, c in enumerate(unique_clusters2):
        if sample_weights is not None:
            dist2[i] = np.sum(sample_weights[cluster_assignments2 == c])
        else:
            dist2[i] = np.sum(cluster_assignments2 == c)
    
    # Normalize distributions
    dist1 = dist1 / np.sum(dist1)
    dist2 = dist2 / np.sum(dist2)
    
    # Extend shorter distribution if needed
    if len(dist1) < len(dist2):
        dist1 = np.pad(dist1, (0, len(dist2) - len(dist1)))
    elif len(dist2) < len(dist1):
        dist2 = np.pad(dist2, (0, len(dist1) - len(dist2)))
    
    # Compute Earth Mover's Distance
    emd = wasserstein_distance(np.arange(len(dist1)), np.arange(len(dist2)), dist1, dist2)
    
    return {
        "adjusted_rand_index": float(ari),
        "earth_movers_distance": float(emd)
    }
```

### 1.3 Transition Matrix Analysis

```python
# concept_fragmentation/analysis/transition_matrix.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import scipy.stats as stats

def compute_transition_matrix(
    prev_clusters: np.ndarray,
    next_clusters: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute transition matrix between clusters in consecutive layers.
    
    Args:
        prev_clusters: Cluster assignments in previous layer
        next_clusters: Cluster assignments in next layer
        normalize: Whether to normalize by row to get transition probabilities
        
    Returns:
        Transition matrix where T[i,j] = count/probability of moving from cluster i to j
    """
    n_prev_clusters = np.max(prev_clusters) + 1
    n_next_clusters = np.max(next_clusters) + 1
    
    # Initialize transition matrix
    transition_matrix = np.zeros((n_prev_clusters, n_next_clusters))
    
    # Count transitions
    for i in range(len(prev_clusters)):
        transition_matrix[prev_clusters[i], next_clusters[i]] += 1
    
    # Normalize if requested
    if normalize:
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
    
    return transition_matrix

def compute_transition_entropy(transition_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute entropy and sparsity metrics for transition matrix.
    
    Args:
        transition_matrix: Normalized transition matrix
        
    Returns:
        Dictionary of transition metrics (entropy, sparsity)
    """
    # Compute row-wise entropy
    entropy = np.zeros(transition_matrix.shape[0])
    for i in range(transition_matrix.shape[0]):
        row = transition_matrix[i]
        # Skip empty rows
        if np.sum(row) > 0:
            # Get non-zero probabilities
            probs = row[row > 0]
            entropy[i] = -np.sum(probs * np.log2(probs))
    
    # Compute sparsity (percentage of zero entries)
    sparsity = np.sum(transition_matrix == 0) / transition_matrix.size
    
    # Compute metrics
    mean_entropy = float(np.mean(entropy))
    max_entropy = float(np.max(entropy))
    
    # Normalize entropy by log2(n_cols)
    normalized_entropy = mean_entropy / np.log2(transition_matrix.shape[1])
    
    return {
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
        "sparsity": float(sparsity)
    }

def compute_multi_step_transitions(
    layer_clusters: Dict[str, np.ndarray],
    layer_names: List[str],
    max_steps: int = 3
) -> Dict[str, np.ndarray]:
    """
    Compute multi-step transition matrices across layers.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        layer_names: Ordered list of layer names
        max_steps: Maximum number of steps to analyze
        
    Returns:
        Dictionary mapping transition paths to transition matrices
    """
    transitions = {}
    
    for step in range(1, max_steps + 1):
        for i in range(len(layer_names) - step):
            source_layer = layer_names[i]
            target_layer = layer_names[i + step]
            
            transitions[f"{source_layer}_to_{target_layer}"] = compute_transition_matrix(
                layer_clusters[source_layer],
                layer_clusters[target_layer]
            )
    
    return transitions
```

## 2. Dash Visualization Integration

### 2.1 ETS Clustering Integration

```python
# Add to visualization/dash_app.py (or appropriate module)

# Add ETS as a clustering option
CLUSTERING_METHODS = [
    {"label": "K-Means", "value": "kmeans"},
    {"label": "DBSCAN", "value": "dbscan"},
    {"label": "ETS (Explainable Threshold)", "value": "ets"}  # Add this option
]

# Add threshold controls to the UI
html.Div([
    html.Label("ETS Threshold Percentile:"),
    dcc.Slider(
        id="ets-threshold-slider",
        min=0.01, max=0.5, step=0.01, value=0.1,
        marks={0.01: "0.01", 0.1: "0.1", 0.2: "0.2", 0.5: "0.5"}
    ),
], id="ets-controls", style={"display": "none"}),

# Update the clustering callback
@app.callback(
    # Existing outputs...
    [Input("clustering-method", "value"),
     Input("ets-threshold-slider", "value"),
     # Other existing inputs...
    ]
)
def perform_clustering(method, threshold_percentile, ...):
    """Perform clustering on the activations"""
    # Display ETS controls only when ETS is selected
    if method == "ets":
        from ..metrics.explainable_threshold_similarity import compute_ets_clustering
        
        # Show progress indicator for long computations
        # ... (progress indicator code)
        
        # Compute ETS clustering
        labels, thresholds = compute_ets_clustering(
            activations, threshold_percentile=threshold_percentile
        )
        
        # Store results
        clustering_results = {
            "labels": labels,
            "thresholds": thresholds,
            "method": "ets",
            "n_clusters": len(np.unique(labels))
        }
        
        return clustering_results, {"display": "block"}  # Show ETS controls
    else:
        # Existing clustering code for other methods
        # ...
        return clustering_results, {"display": "none"}  # Hide ETS controls
```

### 2.2 Transition Matrix Visualization

```python
# Add to visualization/dash_app.py

# Add transition matrix tab
dcc.Tab(label="Transition Matrix", children=[
    html.Div([
        html.H4("Cluster Transition Analysis"),
        
        # Layer selection
        html.Div([
            html.Div([
                html.Label("Source Layer:"),
                dcc.Dropdown(
                    id="transition-source-layer",
                    options=[],  # Will be populated dynamically
                    placeholder="Select source layer..."
                ),
            ], className="six columns"),
            
            html.Div([
                html.Label("Target Layer:"),
                dcc.Dropdown(
                    id="transition-target-layer",
                    options=[],  # Will be populated dynamically
                    placeholder="Select target layer..."
                ),
            ], className="six columns"),
        ], className="row"),
        
        # Transition matrix visualization
        dcc.Graph(id="transition-matrix-plot"),
        
        # Metrics display
        html.Div(id="transition-metrics-display")
    ])
]),

# Transition matrix callback
@app.callback(
    [Output("transition-matrix-plot", "figure"),
     Output("transition-metrics-display", "children")],
    [Input("transition-source-layer", "value"),
     Input("transition-target-layer", "value"),
     Input("clustering-results", "data")]
)
def update_transition_matrix(source_layer, target_layer, clustering_results):
    """Generate transition matrix visualization between selected layers"""
    if not source_layer or not target_layer or source_layer == target_layer:
        return empty_figure("Select two different layers"), ""
    
    if not clustering_results or "layer_clusters" not in clustering_results:
        return empty_figure("Perform clustering first"), ""
    
    # Get cluster assignments
    layer_clusters = clustering_results["layer_clusters"]
    if source_layer not in layer_clusters or target_layer not in layer_clusters:
        return empty_figure("Missing clusters for selected layers"), ""
    
    source_clusters = layer_clusters[source_layer]["labels"]
    target_clusters = layer_clusters[target_layer]["labels"]
    
    # Compute transition matrix
    from ..analysis.transition_matrix import compute_transition_matrix, compute_transition_entropy
    trans_matrix = compute_transition_matrix(source_clusters, target_clusters)
    metrics = compute_transition_entropy(trans_matrix)
    
    # Create heatmap figure using plotly
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=trans_matrix,
        colorscale='Viridis',
        text=[[f"{val:.2f}" for val in row] for row in trans_matrix],
        texttemplate="%{text}",
    ))
    
    fig.update_layout(
        title=f"Cluster Transitions: {source_layer} → {target_layer}",
        xaxis_title=f"{target_layer} Clusters",
        yaxis_title=f"{source_layer} Clusters"
    )
    
    # Create metrics display
    metrics_display = html.Div([
        html.H5("Transition Metrics"),
        html.Ul([
            html.Li(f"Mean Entropy: {metrics['mean_entropy']:.3f}"),
            html.Li(f"Normalized Entropy: {metrics['normalized_entropy']:.3f}"),
            html.Li(f"Sparsity: {metrics['sparsity']:.3f}"),
        ])
    ])
    
    return fig, metrics_display
```

### 2.3 Dimension-wise Explainability View

```python
# Add to visualization/dash_app.py

# Add ETS explainability tab
dcc.Tab(label="ETS Explainability", children=[
    html.Div([
        html.H4("Dimension-wise Cluster Explanation"),
        
        html.Div([
            html.P("This view explains why points are grouped together using dimension-specific thresholds."),
            html.P("Select two points to compare their dimension values against thresholds."),
        ]),
        
        # Point selection
        html.Div([
            html.Div([
                html.Label("Point 1:"),
                dcc.Dropdown(
                    id="ets-point1-selector",
                    options=[],  # Will be populated dynamically
                    placeholder="Select first point..."
                ),
            ], className="six columns"),
            
            html.Div([
                html.Label("Point 2:"),
                dcc.Dropdown(
                    id="ets-point2-selector",
                    options=[],  # Will be populated dynamically
                    placeholder="Select second point..."
                ),
            ], className="six columns"),
        ], className="row"),
        
        # Similarity explanation
        html.Div(id="dimension-similarity-explanation"),
        
        # Dimension thresholds table
        html.Div([
            html.H5("Dimension Thresholds"),
            html.Div(id="dimension-thresholds-table")
        ])
    ])
]),

# Dimension explanation callback
@app.callback(
    Output("dimension-similarity-explanation", "children"),
    [Input("ets-point1-selector", "value"),
     Input("ets-point2-selector", "value"),
     Input("active-layer", "value"),
     Input("clustering-results", "data")]
)
def explain_dimension_similarity(point1_idx, point2_idx, active_layer, clustering_results):
    """Explain similarity between two points based on dimension-wise thresholds"""
    if not point1_idx or not point2_idx or not active_layer or not clustering_results:
        return html.P("Select two points and ensure ETS clustering is performed.")
    
    if clustering_results.get("method") != "ets" or "thresholds" not in clustering_results:
        return html.P("This view requires ETS clustering. Please select ETS as the clustering method.")
    
    # Get activations and thresholds
    activations = get_layer_activations(active_layer)
    thresholds = clustering_results["thresholds"]
    labels = clustering_results["labels"]
    
    # Convert to integers if needed
    point1_idx = int(point1_idx)
    point2_idx = int(point2_idx)
    
    # Check if points are in the same cluster
    same_cluster = labels[point1_idx] == labels[point2_idx]
    
    # Compute dimension-wise differences
    differences = np.abs(activations[point1_idx] - activations[point2_idx])
    
    # Generate explanation table
    table_rows = []
    for dim in range(len(differences)):
        if differences[dim] > thresholds[dim]:
            status = "✗ Exceeds threshold"
            row_class = "exceeds-threshold"
        else:
            status = "✓ Within threshold"
            row_class = "within-threshold"
            
        table_rows.append(html.Tr([
            html.Td(f"Dimension {dim}"),
            html.Td(f"{activations[point1_idx][dim]:.3f}"),
            html.Td(f"{activations[point2_idx][dim]:.3f}"),
            html.Td(f"{differences[dim]:.3f}"),
            html.Td(f"{thresholds[dim]:.3f}"),
            html.Td(status)
        ], className=row_class))
    
    # Summary message
    if same_cluster:
        summary = html.Div([
            html.H5("Points are in the same cluster", style={"color": "green"}),
            html.P("All dimension-wise differences are within their respective thresholds.")
        ])
    else:
        # Find dimensions that cause separation
        separating_dims = [i for i in range(len(differences)) if differences[i] > thresholds[i]]
        summary = html.Div([
            html.H5("Points are in different clusters", style={"color": "red"}),
            html.P(f"Separation caused by {len(separating_dims)} dimensions: {', '.join([str(i) for i in separating_dims])}")
        ])
    
    # Full explanation
    explanation = html.Div([
        summary,
        html.Table([
            html.Thead(html.Tr([
                html.Th("Dimension"), 
                html.Th("Value 1"),
                html.Th("Value 2"),
                html.Th("Difference"),
                html.Th("Threshold"),
                html.Th("Status")
            ])),
            html.Tbody(table_rows)
        ], className="dimension-table")
    ])
    
    return explanation
```

### 2.4 Archetype Path Analysis Tab

```python
# Add to visualization/dash_app.py

# Add archetype path analysis tab
dcc.Tab(label="Archetype Paths", children=[
    html.Div([
        html.H4("Cluster Path Analysis"),
        
        # Path selection
        html.Div([
            html.Label("Select Path:"),
            dcc.Dropdown(
                id="path-selector",
                options=[],  # Will be populated dynamically
                placeholder="Select a path pattern..."
            ),
        ]),
        
        # Path statistics
        html.Div(id="path-statistics"),
        
        # Sample points with this path
        html.Div([
            html.H5("Path Samples"),
            dash_table.DataTable(
                id="path-samples-table",
                columns=[],  # Will be populated based on dataset
                style_table={"overflowX": "auto"}
            )
        ]),
        
        # Path interpretation
        html.Div([
            html.H5("Path Interpretation"),
            html.Button("Generate Interpretation", id="generate-interpretation-btn"),
            html.Div(id="path-interpretation-output")
        ])
    ])
]),

# Path analysis callbacks
@app.callback(
    [Output("path-selector", "options"),
     Output("path-statistics", "children")],
    [Input("clustering-results", "data"),
     Input("dataset-info", "data")]
)
def update_path_analysis(clustering_results, dataset_info):
    """Update path analysis based on clustering results"""
    if not clustering_results or "layer_clusters" not in clustering_results:
        return [], html.P("Perform clustering first")
    
    # Compute paths
    layer_clusters = clustering_results["layer_clusters"]
    layer_names = sorted(layer_clusters.keys())
    
    # Get cluster assignments for each layer
    cluster_assignments = {
        layer: layer_clusters[layer]["labels"] for layer in layer_names
    }
    
    # Compute paths using the cluster_paths module
    from ..analysis.cluster_paths import compute_cluster_paths, compute_path_archetypes
    paths, _ = compute_cluster_paths(cluster_assignments)
    
    # Convert paths to strings for display
    path_strings = []
    for path in paths:
        path_str = "→".join(str(cluster_id) for cluster_id in path)
        path_strings.append(path_str)
    
    # Count path frequencies
    from collections import Counter
    path_counts = Counter(path_strings)
    
    # Get top paths
    top_paths = path_counts.most_common(10)
    
    # Create options for dropdown
    path_options = [
        {"label": f"{path} ({count} samples, {count/len(paths)*100:.1f}%)", "value": path}
        for path, count in top_paths
    ]
    
    # Create path statistics overview
    path_stats = html.Div([
        html.H5("Path Distribution"),
        html.P(f"Total unique paths: {len(path_counts)}"),
        html.P(f"Most common path: {top_paths[0][0]} ({top_paths[0][1]} samples)"),
        
        # Path distribution as bar chart
        dcc.Graph(
            figure={
                "data": [
                    {
                        "x": [p[0] for p in top_paths],
                        "y": [p[1] for p in top_paths],
                        "type": "bar",
                        "name": "Path count"
                    }
                ],
                "layout": {
                    "title": "Top Paths by Frequency",
                    "xaxis": {"title": "Path"},
                    "yaxis": {"title": "Count"}
                }
            }
        )
    ])
    
    return path_options, path_stats
```

## 3. Experiment Scripts

### 3.1 ETS Comparison Experiment

```python
# concept_fragmentation/experiments/run_ets_comparison.py

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from typing import Dict, Tuple

from ..metrics.cluster_entropy import compute_cluster_assignments
from ..metrics.explainable_threshold_similarity import compute_ets_clustering
from ..visualization.activations import plot_activations_2d
from ..utils.helpers import set_seed
from ..data.loaders import get_dataset_loader
from ..hooks.activation_hooks import get_activations

def compare_clustering_methods(
    dataset_name: str,
    layer_name: str,
    seed: int = 42,
    output_dir: str = "results/ets_comparison"
) -> Tuple[Dict, Dict]:
    """
    Compare k-means and ETS clustering on the same dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        layer_name: Network layer to analyze
        seed: Random seed
        output_dir: Directory to save results
        
    Returns:
        Tuple of (kmeans_results, ets_results)
    """
    # Setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and model
    dataset_loader = get_dataset_loader(dataset_name)
    train_df, test_df, X_train, X_test, y_train, y_test = dataset_loader.load_data_and_split()
    
    # Get activations (assuming model is already trained)
    results_dir = os.path.join("results", "baselines", dataset_name, f"baseline_seed{seed}")
    activations = get_activations(results_dir, layer_name)
    
    # Standard k-means clustering
    k = 5  # Example value, should be determined automatically
    kmeans_labels, kmeans = compute_cluster_assignments(
        activations, k, n_init=10, max_iter=300, random_state=seed
    )
    
    # ETS clustering
    ets_labels, thresholds = compute_ets_clustering(
        activations, threshold_percentile=0.1
    )
    
    # Compute silhouette scores for comparison
    kmeans_silhouette = silhouette_score(activations, kmeans_labels)
    ets_silhouette = silhouette_score(activations, ets_labels)
    
    # Plot both clustering results
    fig_kmeans = plot_activations_2d(
        activations, y_test, 
        title=f"K-means Clustering (k={k}, silhouette={kmeans_silhouette:.3f})",
        cluster_labels=kmeans_labels
    )
    
    fig_ets = plot_activations_2d(
        activations, y_test,
        title=f"ETS Clustering (clusters={len(np.unique(ets_labels))}, silhouette={ets_silhouette:.3f})",
        cluster_labels=ets_labels
    )
    
    # Save figures
    fig_kmeans.savefig(os.path.join(output_dir, f"{dataset_name}_{layer_name}_kmeans.png"))
    fig_ets.savefig(os.path.join(output_dir, f"{dataset_name}_{layer_name}_ets.png"))
    
    # Return results
    kmeans_results = {
        "labels": kmeans_labels,
        "n_clusters": k,
        "silhouette": kmeans_silhouette
    }
    
    ets_results = {
        "labels": ets_labels,
        "n_clusters": len(np.unique(ets_labels)),
        "silhouette": ets_silhouette,
        "thresholds": thresholds
    }
    
    return kmeans_results, ets_results
```

### 3.2 Full Archetypal Path Analysis Experiment

```python
# concept_fragmentation/experiments/run_archetypal_path_analysis.py

import argparse
import os
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Optional, Union

from ..utils.helpers import set_seed
from ..data.loaders import get_dataset_loader
from ..analysis.cluster_paths import compute_cluster_paths, compute_path_archetypes
from ..analysis.transition_matrix import compute_transition_matrix, compute_transition_entropy
from ..analysis.llm_prompts import generate_path_analysis_prompt
from ..metrics.explainable_threshold_similarity import compute_ets_clustering
from ..hooks.activation_hooks import get_activations

# Normalized column mappings for different datasets
DATASET_MAPPINGS = {
    "titanic": {
        "demographic_columns": {
            "age": "Age",
            "gender": "Sex",
            "class": "Pclass",
            "fare": "Fare",
            "embarkation": "Embarked"
        },
        "target_column": "Survived"
    },
    "adult": {
        "demographic_columns": {
            "age": "age",
            "gender": "sex",
            "education": "education",
            "occupation": "occupation",
            "race": "race"
        },
        "target_column": "income"
    },
    "heart": {
        "demographic_columns": {
            "age": "age",
            "gender": "sex",
            "chestpain": "cp",
            "bloodpressure": "trestbps",
            "cholesterol": "chol"
        },
        "target_column": "target"
    }
}

def get_normalized_columns(dataset_name: str) -> Tuple[Dict[str, str], str]:
    """Get standardized column mappings for a dataset"""
    mapping = DATASET_MAPPINGS.get(dataset_name.lower(), {})
    return mapping.get("demographic_columns", {}), mapping.get("target_column")

def run_apa_experiment(
    dataset_name: str,
    seed: int = 42,
    use_ets: bool = True,
    output_dir: str = "results/archetypal_path_analysis",
    demographic_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None
) -> Dict:
    """
    Run a complete Archetypal Path Analysis experiment.
    
    Args:
        dataset_name: Name of the dataset to use
        seed: Random seed
        use_ets: Whether to use ETS instead of k-means
        output_dir: Directory to save results
        demographic_columns: Columns to include in demographic statistics
        target_column: Target column name
        
    Returns:
        Dictionary of results
    """
    # Setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    experiment_id = f"{dataset_name}_{'ets' if use_ets else 'kmeans'}_seed{seed}"
    
    # Load data
    dataset_loader = get_dataset_loader(dataset_name)
    train_df, test_df, X_train, X_test, y_train, y_test = dataset_loader.load_data_and_split()
    
    # Get normalized column names
    column_mappings, default_target = get_normalized_columns(dataset_name)
    
    # Set default columns based on dataset if not specified
    if demographic_columns is None:
        demographic_columns = list(column_mappings.values())
    
    if target_column is None:
        target_column = default_target
    
    # Load model and compute activations
    results_dir = os.path.join("results", "baselines", dataset_name, f"baseline_seed{seed}")
    
    # Get activations for each layer
    layer_names = ["input", "layer1", "layer2", "layer3", "output"]
    layer_activations = {}
    
    for layer in layer_names:
        layer_activations[layer] = get_activations(results_dir, layer)
    
    # Compute clusters using either ETS or k-means
    layer_clusters = {}
    
    for layer in layer_names:
        activations = layer_activations[layer]
        
        if use_ets:
            # Use ETS clustering
            print(f"Computing ETS clustering for {layer}...")
            labels, thresholds = compute_ets_clustering(
                activations,
                threshold_percentile=0.1
            )
            layer_clusters[layer] = {
                "labels": labels,
                "thresholds": thresholds,
                "method": "ets",
                "n_clusters": len(np.unique(labels))
            }
        else:
            # Use k-means
            from ..metrics.optimal_num_clusters import _find_optimal_k
            from ..metrics.cluster_entropy import compute_cluster_assignments
            
            print(f"Computing k-means clustering for {layer}...")
            k, _ = _find_optimal_k(activations, k_range=(2, 10))
            labels, kmeans = compute_cluster_assignments(
                activations, k, n_init=10, max_iter=300, random_state=seed
            )
            
            layer_clusters[layer] = {
                "labels": labels,
                "centers": kmeans.cluster_centers_,
                "method": "kmeans",
                "n_clusters": k
            }
    
    # Compute cluster paths
    print("Computing cluster paths...")
    paths, path_layer_names = compute_cluster_paths({
        layer: layer_clusters[layer]["labels"] for layer in layer_names
    })
    
    # Compute path archetypes
    print("Computing path archetypes...")
    archetypes = compute_path_archetypes(
        paths, layer_names, test_df, 
        target_column=target_column,
        demographic_columns=demographic_columns,
        top_k=5
    )
    
    # Compute transition matrices
    print("Computing transition matrices...")
    transition_matrices = {}
    transition_metrics = {}
    
    for i in range(len(layer_names) - 1):
        prev_layer, next_layer = layer_names[i], layer_names[i+1]
        prev_clusters = layer_clusters[prev_layer]["labels"]
        next_clusters = layer_clusters[next_layer]["labels"]
        
        # Compute transition matrix
        trans_matrix = compute_transition_matrix(prev_clusters, next_clusters)
        transition_matrices[f"{prev_layer}_to_{next_layer}"] = trans_matrix
        
        # Compute entropy and sparsity
        metrics = compute_transition_entropy(trans_matrix)
        transition_metrics[f"{prev_layer}_to_{next_layer}"] = metrics
    
    # Generate LLM prompt for path analysis
    prompt = generate_path_analysis_prompt({
        "layers": layer_names,
        "path_archetypes": archetypes
    }, dataset_name, demographic_columns, target_column)
    
    # Save prompt to file
    with open(os.path.join(output_dir, f"{experiment_id}_llm_prompt.txt"), 'w') as f:
        f.write(prompt)
    
    # Prepare results
    results = {
        "dataset": dataset_name,
        "seed": seed,
        "clustering_method": "ets" if use_ets else "kmeans",
        "layers": layer_names,
        "paths": paths.tolist(),
        "path_archetypes": archetypes,
        "transition_metrics": transition_metrics,
        "layer_cluster_counts": {
            layer: layer_clusters[layer]["n_clusters"] for layer in layer_names
        }
    }
    
    # Save results to JSON
    with open(os.path.join(output_dir, f"{experiment_id}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir}/{experiment_id}_results.json")
    return results
```

### 3.3 Metrics Comparison

```python
# concept_fragmentation/experiments/run_metrics_comparison.py

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Optional, Tuple

from ..utils.helpers import set_seed
from ..data.loaders import get_dataset_loader
from ..hooks.activation_hooks import get_activations
from ..metrics.cluster_entropy import compute_cluster_entropy
from ..metrics.subspace_angle import compute_subspace_angle
from ..metrics.intra_class_distance import compute_intra_class_distance
from ..metrics.optimal_num_clusters import compute_optimal_k
from ..metrics.explainable_threshold_similarity import compute_ets_clustering

def compare_metrics(
    dataset_name: str,
    seed: int = 42,
    output_dir: str = "results/metrics_comparison"
) -> Dict[str, Dict]:
    """
    Compare different fragmentation metrics on the same dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        seed: Random seed
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping metrics to their results
    """
    # Setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data and model
    dataset_loader = get_dataset_loader(dataset_name)
    train_df, test_df, X_train, X_test, y_train, y_test = dataset_loader.load_data_and_split()
    
    # Get layer activations
    results_dir = os.path.join("results", "baselines", dataset_name, f"baseline_seed{seed}")
    
    layer_names = ["input", "layer1", "layer2", "layer3", "output"]
    layer_activations = {}
    
    for layer in layer_names:
        layer_activations[layer] = get_activations(results_dir, layer)
    
    # Compute all metrics
    metrics_results = {}
    
    # 1. Cluster Entropy
    cluster_entropy_results = {}
    for layer in layer_names:
        activations = layer_activations[layer]
        result = compute_cluster_entropy(
            activations, y_test,
            k_selection='auto',
            return_clusters=True
        )
        cluster_entropy_results[layer] = result
    
    metrics_results["cluster_entropy"] = {
        "values": {layer: result["mean_entropy"] for layer, result in cluster_entropy_results.items()},
        "details": cluster_entropy_results
    }
    
    # 2. Subspace Angle
    subspace_angle_results = {}
    for layer in layer_names:
        activations = layer_activations[layer]
        result = compute_subspace_angle(
            activations, y_test,
            var_threshold=0.9
        )
        subspace_angle_results[layer] = result
    
    metrics_results["subspace_angle"] = {
        "values": {layer: result["mean_angle"] for layer, result in subspace_angle_results.items()},
        "details": subspace_angle_results
    }
    
    # 3. Intra-Class Distance
    icpd_results = {}
    for layer in layer_names:
        activations = layer_activations[layer]
        result = compute_intra_class_distance(
            activations, y_test,
            normalize_dim=True
        )
        icpd_results[layer] = result
    
    metrics_results["intra_class_distance"] = {
        "values": {layer: result["mean_distance"] for layer, result in icpd_results.items()},
        "details": icpd_results
    }
    
    # 4. Optimal Number of Clusters (k*)
    kstar_results = {}
    for layer in layer_names:
        activations = layer_activations[layer]
        result = compute_optimal_k(
            activations, y_test,
            k_range=(2, 10)
        )
        kstar_results[layer] = result
    
    metrics_results["optimal_k"] = {
        "values": {layer: result["mean_k"] for layer, result in kstar_results.items()},
        "details": kstar_results
    }
    
    # Plot metrics comparison
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Fragmentation Metrics Comparison - {dataset_name.capitalize()}", fontsize=16)
    
    # Plot each metric as a line chart
    metrics = [
        ("cluster_entropy", "Cluster Entropy", axs[0, 0]),
        ("subspace_angle", "Subspace Angle (degrees)", axs[0, 1]),
        ("intra_class_distance", "Intra-Class Distance", axs[1, 0]),
        ("optimal_k", "Optimal Clusters (k*)", axs[1, 1])
    ]
    
    for metric_key, metric_name, ax in metrics:
        values = metrics_results[metric_key]["values"]
        
        # Plot line
        ax.plot(list(range(len(layer_names))), [values[layer] for layer in layer_names], 
                marker='o', linewidth=2)
        
        # Set labels
        ax.set_title(metric_name)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_name)
        ax.set_xticks(list(range(len(layer_names))))
        ax.set_xticklabels(layer_names)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_metrics_comparison.png"), dpi=300)
    
    # Save results to JSON
    with open(os.path.join(output_dir, f"{dataset_name}_metrics_results.json"), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_compatible = {}
        for metric, result in metrics_results.items():
            json_compatible[metric] = {
                "values": {k: float(v) for k, v in result["values"].items()},
                "details": {}  # Simplified for JSON
            }
        json.dump(json_compatible, f, indent=2)
    
    return metrics_results
```

## 4. LLM Integration

```python
# concept_fragmentation/analysis/llm_prompts.py

def generate_path_analysis_prompt(
    path_data: Dict,
    dataset_name: str,
    demographic_keys: List[str],
    target_column: str
) -> str:
    """
    Generate a prompt for LLM-based analysis of archetype paths.
    
    Args:
        path_data: Dictionary of path data with statistics
        dataset_name: Name of the dataset
        demographic_keys: List of demographic columns to include
        target_column: Target column name
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an interpretability analyst narrating a neural network's internal decision process for {dataset_name} data.

I'll provide information about archetype paths - sequences of clusters that datapoints pass through as they move through the network's layers. Your task is to generate an insightful narrative that explains what these paths reveal about the model's decision-making process.

# Path Information

Layers: {path_data['layers']}

## Top Archetype Paths:
"""

    # Add information for each path
    for i, path in enumerate(path_data['path_archetypes']):
        prompt += f"\n### Path {i+1}: {path['path']}\n"
        prompt += f"- Count: {path['count']} datapoints ({(path['count'] / len(path_data.get('paths', [])) * 100):.2f}% of dataset)\n"
        
        # Include target information if available
        target_rate_key = f"{target_column}_rate"
        if target_rate_key in path:
            prompt += f"- {target_column.capitalize()} rate: {path[target_rate_key]:.2f}\n\n"
        
        prompt += "Demographic statistics:\n"
        if 'demo_stats' in path:
            for key in demographic_keys:
                if key in path['demo_stats']:
                    stats = path['demo_stats'][key]
                    if isinstance(stats, dict):
                        # Categorical
                        dist_str = ", ".join(f"{k}: {v:.2f}" for k, v in stats.items())
                        prompt += f"- {key}: {dist_str}\n"
                    else:
                        # Numerical - check if it's a dictionary with stats
                        if isinstance(stats, dict) and 'mean' in stats:
                            prompt += f"- {key}: mean={stats['mean']:.2f}, std={stats['std']:.2f}\n"
                        else:
                            prompt += f"- {key}: {stats}\n"
        
        prompt += "\n"
    
    prompt += """
# Your Analysis Task

For each archetype path, provide:

1. A descriptive name that captures the essence of this path (e.g., "Privileged Survivors", "Marginal Cases")
2. An explanation of what happens to datapoints along this path through the layers
3. Key demographic patterns or biases you notice
4. Insights about how the model makes decisions for this group
5. Any fairness concerns or implications

After analyzing each path, provide an overall summary of what these paths collectively reveal about the model's internal decision logic and potential biases.

Focus on constructing a coherent narrative that helps explain how the model processes information across its layers. Use vivid language but stay grounded in the data.
"""
    
    return prompt

def process_llm_path_analysis(prompt: str, model: str = "gpt-4") -> str:
    """
    Send prompt to an LLM for path analysis.
    This is a stub - implement with your preferred LLM API.
    
    Args:
        prompt: The formatted prompt
        model: Model identifier
        
    Returns:
        LLM response
    """
    # Example integration with an LLM API
    # In practice, implement this with your preferred LLM provider
    # import openai
    # response = openai.ChatCompletion.create(
    #    model=model,
    #    messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    
    # Stub implementation - replace with actual LLM integration
    return "LLM analysis would appear here. Implement the LLM API integration according to your preferred provider."
```

## 5. Configuration Updates

```python
# Update concept_fragmentation/config.py to include new configuration

# ETS Parameters
METRICS["explainable_threshold_similarity"] = {
    "threshold_percentile": 0.1,
    "min_dimension_variance": 0.01,
    "batch_size": 1000
}

# Stability Parameters
METRICS["cluster_stability"] = {
    "n_bootstrap_samples": 10,
    "min_cluster_size": 5
}

# Analysis Parameters
ANALYSIS = {
    "path_archetypes": {
        "top_k": 5,
        "max_members": 50
    },
    "transition_matrix": {
        "entropy_normalization": True
    }
}
```

## 6. Testing Plan

Create comprehensive unit tests for the new components:

```python
# concept_fragmentation/tests/test_ets.py

import unittest
import numpy as np
import torch
from ..metrics.explainable_threshold_similarity import compute_ets_clustering
from ..metrics.cluster_stability import compute_cluster_stability

class TestExplainableThresholdSimilarity(unittest.TestCase):
    def test_basic_clustering(self):
        # Create synthetic data with clear clusters
        X = np.array([
            [1.0, 1.0],  # Cluster 1
            [1.1, 0.9],  # Cluster 1
            [5.0, 5.0],  # Cluster 2
            [5.1, 4.9],  # Cluster 2
            [10.0, 0.0], # Cluster 3
            [10.1, 0.1]  # Cluster 3
        ])
        
        # Set explicit thresholds
        thresholds = np.array([0.5, 0.5])
        
        # Compute ETS clustering
        labels, returned_thresholds = compute_ets_clustering(X, thresholds)
        
        # Check that we get 3 clusters
        self.assertEqual(len(np.unique(labels)), 3)
        
        # Check that points in the same cluster have the same label
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertEqual(labels[4], labels[5])
        
        # Check that points in different clusters have different labels
        self.assertNotEqual(labels[0], labels[2])
        self.assertNotEqual(labels[0], labels[4])
        self.assertNotEqual(labels[2], labels[4])
        
        # Check that returned thresholds match input
        np.testing.assert_array_equal(thresholds, returned_thresholds)
    
    def test_automatic_threshold(self):
        # Create synthetic data
        X = np.array([
            [1.0, 1.0], [1.1, 0.9],  # Cluster 1
            [5.0, 5.0], [5.1, 4.9],  # Cluster 2
            [10.0, 0.0], [10.1, 0.1] # Cluster 3
        ])
        
        # Compute ETS clustering with automatic threshold
        labels, thresholds = compute_ets_clustering(X, threshold_percentile=0.1)
        
        # We should still get 3 clusters with automatic thresholds
        self.assertEqual(len(np.unique(labels)), 3)
        
        # Thresholds should be positive
        self.assertTrue(np.all(thresholds > 0))

class TestClusterStability(unittest.TestCase):
    def test_identical_clusters(self):
        # When clusters are identical, ARI should be 1 and EMD should be 0
        clusters1 = np.array([0, 0, 1, 1, 2, 2])
        clusters2 = np.array([0, 0, 1, 1, 2, 2])
        
        stability = compute_cluster_stability(clusters1, clusters2)
        
        self.assertAlmostEqual(stability["adjusted_rand_index"], 1.0)
        self.assertAlmostEqual(stability["earth_movers_distance"], 0.0)
    
    def test_different_clusters(self):
        # When clusters are completely different, ARI should be low
        clusters1 = np.array([0, 0, 1, 1, 2, 2])
        clusters2 = np.array([2, 2, 0, 0, 1, 1])
        
        stability = compute_cluster_stability(clusters1, clusters2)
        
        # Different labels but same structure - ARI should be high
        self.assertGreater(stability["adjusted_rand_index"], 0.5)
```

## 7. Implementation Timeline

1. **Week 1: Core Metrics**
   - Implement ETS clustering algorithm
   - Add cluster stability metrics
   - Create transition matrix analysis

2. **Week 2: Dash Integration**
   - Add ETS as a clustering method
   - Implement transition matrix visualization
   - Create dimension-wise explainability view

3. **Week 3: Path Analysis**
   - Implement path computation and archetype analysis
   - Add LLM prompt generation
   - Create path analysis tab in dashboard

4. **Week 4: Testing & Documentation**
   - Write unit tests for all new components
   - Create tutorial notebooks
   - Document usage in README

## 8. Optimization Considerations

- **Memory Usage**: Implement batch processing for large datasets
- **Caching**: Cache expensive computations (e.g., similarity matrices)
- **Progress Indicators**: Show progress for long-running operations
- **Error Handling**: Robust error handling for boundary cases (e.g., empty clusters)

## 9. Future Extensions

- **Topological Analysis**: Add persistent homology for cluster stability
- **Feature Attribution**: Identify input features responsible for transitions
- **Self-Interpretation**: Enable LLMs to analyze their own activation patterns
- **Interactive Exploration**: Add interactive path exploration in dashboard

## 10. Requirements

Add to requirements.txt:
```
scipy>=1.7.0
scikit-learn>=1.0.0
```

Optional LLM integration:
```
openai>=1.0.0  # Or alternative LLM API
```