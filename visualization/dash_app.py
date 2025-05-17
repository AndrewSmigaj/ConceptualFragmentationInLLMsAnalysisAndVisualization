"""
Dash web application for interactive exploration of neural network trajectories.

This module provides a web-based interface for exploring the 3D
visualizations of neural network layer trajectories.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import re

try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output, State
except ImportError:
    print("Dash not installed. Install with: pip install dash")
    print("Exiting...")
    sys.exit(1)

# Add parent directory to path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualization.data_interface import (
    load_stats, get_best_config, get_baseline_config, 
    select_samples, load_dataset_metadata,
    load_layer_class_metrics, load_activations, compute_layer_clusters,
    get_config_id, get_config_path
)
from visualization.reducers import Embedder, embed_layer_activations
from visualization.traj_plot import build_single_scene_figure, plot_dataset_trajectories, save_figure, LAYER_SEPARATION_OFFSET
from visualization.cluster_utils import compute_layer_clusters_embedded, get_embedded_clusters_cache_path
from concept_fragmentation.config import DATASETS as DATASET_CFG
from concept_fragmentation.analysis.cluster_stats import write_cluster_stats

# Define the datasets and seeds
DATASETS = ["titanic", "heart"]
SEEDS = [0, 1, 2]
MAX_SAMPLES = 500
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Initialize the Dash app
app = dash.Dash(__name__, title="Neural Network Trajectory Explorer", suppress_callback_exceptions=True)

# Placeholder for actual metrics. User should replace with actual metric names.
AVAILABLE_POINT_COLOR_METRICS = {
    "cluster_entropy": "Cluster Entropy",
    "subspace_angle": "Subspace Angle",
    "icpd": "Intra-Class Pairwise Distance",
    "kstar": "Optimal Number of Clusters (k*)",
    "stability": "Representation Stability (Δ-Norm)"
}

# Define color modes
COLOR_MODES = {
    "class": "Color by Class",
    "metric": "Color by Metric",
    "cluster": "Color by Cluster",
    "cluster_majority": "Color by Majority-Class of Cluster"
}

# Helper functions for JSON serialization
def numpy_to_list_recursive(data: Any) -> Any:
    """Recursively convert numpy arrays in a nested structure to lists."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: numpy_to_list_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [numpy_to_list_recursive(item) for item in data]
    else:
        return data

def list_to_numpy_recursive(data: Any) -> Any:
    """
    Recursively convert lists (that were previously numpy arrays and converted via .tolist())
    back to numpy arrays. This is a basic version and might need adjustment
    if the list structure is very complex or if non-array lists are present.
    """
    if isinstance(data, dict):
        return {k: list_to_numpy_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Try to convert to np.array. If it fails, assume it's a list of other structures.
        try:
            # Check if elements are themselves lists or dicts to recurse,
            # otherwise attempt conversion to np.array.
            # This handles cases like list of lists (2D array) or list of numbers (1D array).
            if data and (isinstance(data[0], (dict, list))):
                 # If elements are dicts or lists, recurse into them first
                 processed_list = [list_to_numpy_recursive(item) for item in data]
                 # Check if the processed list can now form a homogeneous NumPy array
                 try:
                     return np.array(processed_list)
                 except ValueError: # If still inhomogeneous (e.g. list of dicts), return as list of processed items
                     return processed_list
            else:
                 # Assuming it's a list of numbers/simple types that can form an array
                 return np.array(data)
        except Exception:
            # If conversion to np.array fails at any point, return as a list of recursively processed items
            return [list_to_numpy_recursive(item) for item in data] # Fallback, should ideally not happen for embeddings
    else:
        return data

# Define the layout
app.layout = html.Div([
    html.H1("Neural Network Trajectory Explorer", style={"textAlign": "center"}),
    
    html.Div([
        html.Div([
            html.Label("Dataset:"),
            dcc.Dropdown(
                id="dataset-dropdown",
                options=[{"label": d.title(), "value": d} for d in DATASETS],
                value=DATASETS[0] if DATASETS else None
            ),
        ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.Label("Seed:"),
            dcc.Dropdown(
                id="seed-dropdown",
                options=[{"label": f"Seed {s}", "value": s} for s in SEEDS],
                value=SEEDS[0] if SEEDS else None
            ),
        ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
        
        html.Div([
            html.Label("Number of Samples:"),
            dcc.Slider(
                id="samples-slider",
                min=20,
                max=MAX_SAMPLES,
                step=20,
                value=100,
                marks={i: str(i) for i in range(0, MAX_SAMPLES + 1, 100)},
            ),
        ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
    ]),
    
    html.Div([
        html.Div([
            html.Label("Visualization Controls:"),
            dcc.Checklist(
                id="visualization-controls",
                options=[
                    {"label": "Show Baseline", "value": "baseline"},
                    {"label": "Show Regularized", "value": "regularized"},
                    {"label": "Show Arrows", "value": "arrows"},
                    {"label": "Show Cluster Centers", "value": "show_centers"},
                    {"label": "Normalize Embeddings", "value": "normalize"}
                ],
                value=["baseline", "regularized", "arrows", "normalize"]
            ),
            html.Br(), # Added for spacing
            html.Label("Color Mode:"),
            dcc.Dropdown(
                id="color-mode-dropdown",
                options=[
                    {"label": name, "value": key}
                    for key, name in COLOR_MODES.items()
                ],
                value="class" # Default to color by class
            ),
            html.Br(),
            html.Label("Trajectory Point Color Metric:"),
            dcc.Dropdown(
                id="trajectory-point-metric-dropdown",
                options=[
                    {"label": name, "value": key}
                    for key, name in AVAILABLE_POINT_COLOR_METRICS.items()
                ],
                value=list(AVAILABLE_POINT_COLOR_METRICS.keys())[0] # Default to the first metric
            ),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
        
        html.Div([
            html.Label("Highlight Options:"),
            dcc.Checklist(
                id="highlight-controls",
                options=[
                    {"label": "Highlight High Fragmentation", "value": "high_frag"},
                    {"label": "Color by Class", "value": "class_color"}
                ],
                value=["high_frag"]
            ),
            html.Label("Number of Highlights:"),
            dcc.Slider(
                id="highlight-slider",
                min=0,
                max=50,
                step=5,
                value=20,
                marks={i: str(i) for i in range(0, 51, 10)},
            ),
            html.Br(),
            html.Label("Max Clusters (k):"),
            dcc.Slider(
                id="max-k-slider",
                min=4,
                max=15,
                step=1,
                value=10,
                marks={i: str(i) for i in range(4, 16, 2)},
            ),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
        
        html.Div([
            html.Button("Update Visualization", id="update-button", n_clicks=0, 
                        style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px"}),
            html.Button("Export HTML", id="export-button", n_clicks=0,
                        style={"backgroundColor": "#008CBA", "color": "white", "padding": "10px", "marginLeft": "10px"}),
            dcc.Download(id="download-html"),
        ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "20px"}),
        
        # Cache clearing button
        html.Button('Clear Cache', id='clear-cache-button', n_clicks=0),
        html.Div(id='cache-cleared-message'),
    ]),
    
    dcc.Loading(
        id="loading-visualization",
        type="circle",
        children=[
            dcc.Graph(
                id="trajectory-graph",
                style={"height": "700px"},
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "trajectory_plot",
                        "height": 700,
                        "width": 1200
                    }
                }
            )
        ]
    ),
    
    # DEBUG: We need to add this element for showing fracture scores
    dcc.Graph(id="layer-fracture-graph", style={"height": "250px", "marginTop": "15px"}),
    
    # Cluster statistics display
    html.Div(
        id="cluster-summary",
        style={
            "padding": "15px", 
            "marginTop": "10px", 
            "backgroundColor": "#f8f9fa", 
            "border": "1px solid #ddd",
            "borderRadius": "5px"
        },
        children="Click a cluster center to view statistics"
    ),
    
    html.Div(id="status-message", style={"padding": "10px", "color": "gray"}),
    
    # Hidden data storage
    dcc.Store(id="embeddings-store"),
    dcc.Store(id="layer-clusters-store"),
    dcc.Store(id="true-labels-store"),
    dcc.Store(id="dataset-data-store")
])

def load_and_embed_data(dataset, seed, max_k=10):
    """Load and embed data for the given dataset and seed."""
    try:
        # Get configurations
        best_config = get_best_config(dataset)
        baseline_config = get_baseline_config(dataset)
        
        # Create embedder
        embedder = Embedder(n_components=3, random_state=42, cache_dir=CACHE_DIR)
        
        # Embed baseline
        print(f"Embedding baseline data for {dataset}, seed {seed}")
        baseline_embeddings, baseline_true_labels = embed_layer_activations(
            dataset, baseline_config, seed, embedder=embedder, use_test_set=True)
        
        # Embed best config
        print(f"Embedding regularized data for {dataset}, seed {seed}")
        best_embeddings, best_true_labels = embed_layer_activations(
            dataset, best_config, seed, embedder=embedder, use_test_set=True)
        
        # Create dictionary for storage
        embeddings_dict = {
            "baseline": {seed: baseline_embeddings},
            "regularized": {seed: best_embeddings}
        }
        
        # Store true labels for each configuration
        true_labels_dict = {
            "baseline": baseline_true_labels,
            "regularized": best_true_labels
        }
        
        # Use the true labels that are available, preferring regularized if available
        actual_true_labels = None
        if best_true_labels is not None:
            actual_true_labels = best_true_labels
            print(f"Using true labels from regularized config with shape {actual_true_labels.shape}")
        elif baseline_true_labels is not None:
            actual_true_labels = baseline_true_labels
            print(f"Using true labels from baseline config with shape {actual_true_labels.shape}")
        
        # Compute clusters in the embedded space for each configuration
        print(f"Computing embedded clusters for {dataset}, seed {seed}, max_k={max_k}")
        
        # Get cache paths
        baseline_cache_dir = os.path.join(CACHE_DIR, "embedded_clusters")
        best_cache_dir = os.path.join(CACHE_DIR, "embedded_clusters")
        baseline_cache_path = get_embedded_clusters_cache_path(
            dataset, "baseline", seed, max_k, cache_dir=baseline_cache_dir
        )
        best_config_id = get_config_id(best_config)
        best_cache_path = get_embedded_clusters_cache_path(
            dataset, best_config_id, seed, max_k, cache_dir=best_cache_dir
        )
        
        # Compute clusters
        baseline_clusters = compute_layer_clusters_embedded(
            baseline_embeddings, max_k=max_k, random_state=42, cache_path=baseline_cache_path
        )
        best_clusters = compute_layer_clusters_embedded(
            best_embeddings, max_k=max_k, random_state=42, cache_path=best_cache_path
        )
        
        # Store clusters by configuration
        layer_clusters_by_config = {
            "baseline": baseline_clusters,
            "regularized": best_clusters
        }
        
        return embeddings_dict, layer_clusters_by_config, true_labels_dict, None
    
    except Exception as e:
        import traceback
        error_msg = f"Error loading embeddings: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, None, error_msg

@app.callback(
    [Output("embeddings-store", "data"),
     Output("layer-clusters-store", "data"),
     Output("true-labels-store", "data"),
     Output("dataset-data-store", "data"),
     Output("status-message", "children")],
    [Input("dataset-dropdown", "value"),
     Input("seed-dropdown", "value"),
     Input("max-k-slider", "value")]
)
def update_embeddings(dataset, seed, max_k):
    """Update embeddings and cluster data when dataset, seed, or max_k changes."""
    if not dataset or seed is None:
        return None, None, None, None, "Please select a dataset and seed."
    
    print(f"Loading data for {dataset}, seed {seed}, max_k {max_k}")
    embeddings_dict, layer_clusters, true_labels, error_msg = load_and_embed_data(dataset, seed, max_k)
    
    if error_msg:
        return None, None, None, None, error_msg
    
    # Convert embeddings to JSON-serializable format
    embeddings_json = numpy_to_list_recursive(embeddings_dict)
    
    # Convert clusters to JSON-serializable format
    clusters_json = numpy_to_list_recursive(layer_clusters)
    
    # Convert true labels to JSON-serializable format
    true_labels_json = numpy_to_list_recursive(true_labels)
    
    # Load dataset numeric features
    try:
        # Get dataset path from config
        dataset_path = DATASET_CFG[dataset]["path"]
        numeric_features = DATASET_CFG[dataset]["numerical_features"]
        
        # DEBUG: Print which numeric features are available
        print(f"DEBUG: Numeric features from config: {numeric_features}")
        print(f"DEBUG: All features in config: {DATASET_CFG[dataset]}")
        
        # Instead of direct CSV loading, use the data loaders
        from concept_fragmentation.data.loaders import get_dataset_loader
        
        # Get the appropriate dataset loader
        loader = get_dataset_loader(dataset)
        
        # Load data using the loader
        train_df, test_df = loader.load_data()
        
        # DEBUG: Show all columns in the loaded dataframe
        print(f"DEBUG: All columns in train_df: {train_df.columns.tolist()}")
        
        # Include both numeric features and the 'sex' column
        features_to_include = numeric_features.copy()
        if 'Sex' in train_df.columns:
            features_to_include.append('Sex')
        elif 'sex' in train_df.columns:
            features_to_include.append('sex')
            
        # DEBUG: Print which features we're including
        print(f"DEBUG: Features included in df_selected: {features_to_include}")
        
        # Filter to selected features
        df_selected = train_df[features_to_include]
        
        # DEBUG: Check final selected dataframe
        print(f"DEBUG: Columns in df_selected: {df_selected.columns.tolist()}")
        print(f"DEBUG: df_selected shape: {df_selected.shape}")
        print(f"DEBUG: df_selected types: {df_selected.dtypes}")
        
        # Convert to JSON-serializable format
        dataset_json = df_selected.to_json(orient="split")
        
        # Generate cluster statistics using baseline clusters
        if "baseline" in layer_clusters:
            # Use the simplified write_cluster_stats that works with pre-computed clusters
            json_path = write_cluster_stats(
                dataset_name=dataset,
                seed=seed,
                layer_clusters=layer_clusters["baseline"],
                df=df_selected
            )
            print(f"Wrote cluster statistics for {dataset} (seed {seed}) using baseline clusters to: {json_path}")
            # DEBUG: Check the clusters we have by layer
            for config_name, config_clusters in layer_clusters.items():
                print(f"DEBUG: Config {config_name} has layers: {list(config_clusters.keys())}")
                for layer, layer_info in config_clusters.items():
                    print(f"DEBUG: {config_name} {layer} keys: {list(layer_info.keys())}")
    except Exception as e:
        import traceback
        error_msg = f"Error loading dataset or writing cluster stats: {str(e)}\n{traceback.format_exc()}"
        print(f"DEBUG ERROR: {error_msg}")
        dataset_json = None
    
    return embeddings_json, clusters_json, true_labels_json, dataset_json, f"Loaded {dataset} data for seed {seed}"

@app.callback(
    [Output("trajectory-graph", "figure"),
     Output("layer-fracture-graph", "figure")],
    [Input("update-button", "n_clicks")],
    [State("embeddings-store", "data"),
     State("layer-clusters-store", "data"),
     State("true-labels-store", "data"),
     State("dataset-dropdown", "value"),
     State("samples-slider", "value"),
     State("highlight-slider", "value"),
     State("visualization-controls", "value"),
     State("highlight-controls", "value"),
     State("trajectory-point-metric-dropdown", "value"),
     State("color-mode-dropdown", "value"),
     State("seed-dropdown", "value")]
)
def update_visualization(n_clicks, stored_data, stored_clusters, stored_true_labels, dataset, n_samples, n_highlights, 
                        vis_controls, highlight_controls, selected_point_metric, color_mode, seed):
    """
    Update the visualization based on user selections.
    """
    if n_clicks is None or not stored_data or not dataset:
        # Initial load or no data
        empty_traj_fig = go.Figure().update_layout(
            title="Click 'Update Visualization' after selecting options",
            height=700
        )
        empty_fracture_fig = go.Figure().update_layout(
            title="Click 'Update Visualization' to see fragmentation scores",
            xaxis_title="Layer",
            yaxis_title="Fragmentation Score",
            height=250
        )
        return empty_traj_fig, empty_fracture_fig
    
    # Debug: Print received data types
    print("DEBUG: Starting update_visualization function")
    
    # Diagnostic check for each store object
    print(f"DEBUG: stored_data type: {type(stored_data)}")
    print(f"DEBUG: stored_clusters type: {type(stored_clusters)}")
    if stored_clusters:
        if isinstance(stored_clusters, dict) and len(stored_clusters) > 0:
            config = next(iter(stored_clusters.keys()))
            print(f"DEBUG: First config: {config}")
            if isinstance(stored_clusters[config], dict) and len(stored_clusters[config]) > 0:
                layer = next(iter(stored_clusters[config].keys()))
                print(f"DEBUG: First layer: {layer}")
                if isinstance(stored_clusters[config][layer], dict):
                    print(f"DEBUG: Layer keys: {list(stored_clusters[config][layer].keys())}")
    
    # Diagnostic check for options
    print(f"DEBUG: vis_controls: {vis_controls}")
    print(f"DEBUG: selected_point_metric: {selected_point_metric}")
    
    # Convert stored JSON data back to Python objects with NumPy arrays
    embeddings_dict = list_to_numpy_recursive(stored_data)
    
    # Convert stored clusters back
    layer_clusters = None
    if stored_clusters:
        layer_clusters = list_to_numpy_recursive(stored_clusters)
    
    # Convert stored true labels back
    true_labels_by_config = None
    if stored_true_labels:
        true_labels_by_config = list_to_numpy_recursive(stored_true_labels)
        print(f"DEBUG: true_labels_by_config type: {type(true_labels_by_config)}")
        print(f"DEBUG: true_labels_by_config keys: {true_labels_by_config.keys() if isinstance(true_labels_by_config, dict) else 'not a dict'}")
    
    # Create custom cluster name mappings
    # This is hardcoded for now, but could be loaded from a configuration file
    cluster_name_mapping = {
        # Layer 1
        "baseline-layer1-cluster0": "Older mixed-class males (likely to die)",
        "baseline-layer1-cluster1": "Younger low-fare males (likely survivors)",
        "baseline-layer1-cluster2": "Middle-aged moderate-fare group (likely to die)",
        "baseline-layer1-cluster3": "Young females, lower class (likely survivors)",

        # Layer 2
        "baseline-layer2-cluster0": "Middle-class passengers, moderate fare (died)",
        "baseline-layer2-cluster1": "Low-fare men, unrefined identity (died)",
        "baseline-layer2-cluster2": "High-fare older passengers (likely to die)",
        "baseline-layer2-cluster3": "Females and children, low-to-mid fare (survivors)",
        "baseline-layer2-cluster4": "Older moderate-income males (died)",
        "baseline-layer2-cluster5": "Surviving females, likely 1st class",

        # Layer 3
        "baseline-layer3-cluster0": "High-fare older survivors",
        "baseline-layer3-cluster1": "Lower-fare males (some saved)",
        "baseline-layer3-cluster2": "Young male non-survivors"
    }
    
    # Filter configurations based on user selection
    show_baseline = "baseline" in vis_controls
    show_regularized = "regularized" in vis_controls
    
    filtered_embeddings = {}
    filtered_layer_clusters = {}
    config_for_samples = None
    config_details = None
    clusters_for_samples = None
    true_labels_for_samples = None

    if show_baseline and "baseline" in embeddings_dict:
        filtered_embeddings["baseline"] = embeddings_dict["baseline"]
        if layer_clusters and "baseline" in layer_clusters:
             filtered_layer_clusters["baseline"] = layer_clusters["baseline"]
             config_for_samples = "baseline"
             config_details = get_baseline_config(dataset)
             clusters_for_samples = layer_clusters["baseline"]
             if true_labels_by_config and "baseline" in true_labels_by_config:
                 true_labels_for_samples = true_labels_by_config["baseline"]
                 print(f"DEBUG: Using baseline true_labels_for_samples, type: {type(true_labels_for_samples)}")
                 if isinstance(true_labels_for_samples, np.ndarray):
                     print(f"DEBUG: true_labels_for_samples shape: {true_labels_for_samples.shape}")
                 else:
                     print(f"DEBUG: true_labels_for_samples is not an array, it's a {type(true_labels_for_samples)}")

    if show_regularized and "regularized" in embeddings_dict:
        filtered_embeddings["regularized"] = embeddings_dict["regularized"]
        if layer_clusters and "regularized" in layer_clusters:
            filtered_layer_clusters["regularized"] = layer_clusters["regularized"]
            # Prefer regularized for sample selection if available
            config_for_samples = "regularized"
            config_details = get_best_config(dataset)
            clusters_for_samples = layer_clusters["regularized"]
            if true_labels_by_config and "regularized" in true_labels_by_config:
                true_labels_for_samples = true_labels_by_config["regularized"]
                print(f"DEBUG: Using regularized true_labels_for_samples, type: {type(true_labels_for_samples)}")
                if isinstance(true_labels_for_samples, np.ndarray):
                    print(f"DEBUG: true_labels_for_samples shape: {true_labels_for_samples.shape}")
                else:
                    print(f"DEBUG: true_labels_for_samples is not an array, it's a {type(true_labels_for_samples)}")
    
    if not filtered_embeddings:
        return go.Figure().update_layout(
            title="No configurations selected to display",
            height=700
        ), go.Figure().update_layout(
            title="No data available",
            xaxis_title="Layer",
            yaxis_title="Fragmentation Score",
            height=250
        )
    
    # Get metadata for this dataset
    try:
        metadata = load_dataset_metadata(dataset)
        class_labels = metadata.get("class_labels", [])
        print(f"DEBUG: class_labels from metadata, type: {type(class_labels)}")
        print(f"DEBUG: class_labels from metadata, length: {len(class_labels) if hasattr(class_labels, '__len__') else 'no length'}")
    except Exception as e:
        print(f"Error loading metadata: {e}")
        class_labels = []
    
    # Select samples with high/low fragmentation
    highlight_indices = []
    if "high_frag" in highlight_controls and n_highlights > 0:
        try:
            if config_for_samples and clusters_for_samples:
                print(f"DEBUG: Before select_samples, true_labels_for_samples type: {type(true_labels_for_samples)}")
                if isinstance(true_labels_for_samples, np.ndarray):
                    print(f"DEBUG: Before select_samples, true_labels_for_samples shape: {true_labels_for_samples.shape}")
                
                print(f"Using pre-computed embedded layer clusters from '{config_for_samples}' config for sample selection")
                highlight_indices = select_samples(
                    dataset, config_details, seed, 
                    k_frag=n_highlights, k_norm=0,
                    layer_clusters=clusters_for_samples,
                    actual_class_labels=true_labels_for_samples
                )
                print(f"DEBUG: After select_samples, highlight_indices: {highlight_indices[:5]}... (showing first 5)")
            else:
                 print("Skipping sample selection as no suitable config/clusters found.")

        except Exception as e:
            import traceback
            print(f"DEBUG: Error in select_samples: {e}")
            print(f"DEBUG: {traceback.format_exc()}")
    
    # Visualization options
    show_arrows = "arrows" in vis_controls
    normalize = "normalize" in vis_controls
    show_cluster_centers = "show_centers" in vis_controls
    
    # Color mode
    color_by = color_mode if color_mode else "class"
    
    # Sample color values for the "metric" color mode
    sample_color_values = None
    if color_by == "metric":
        # Load metric values (this would need to be implemented based on your data)
        # For now, just use a dummy placeholder
        pass
    
    # If we have true labels from embeddings, use them instead of metadata labels
    # This ensures alignment with the actual data we're visualizing
    print(f"DEBUG: Before class_labels check, true_labels_for_samples is {type(true_labels_for_samples)}")
    print(f"DEBUG: Before class_labels check, class_labels is {type(class_labels)}")
    
    # Fix the potential ambiguous truth value error by checking existence properly
    if true_labels_for_samples is not None:
        # Check properly depending on the type
        has_labels = False
        if isinstance(true_labels_for_samples, np.ndarray):
            has_labels = true_labels_for_samples.size > 0
        elif isinstance(true_labels_for_samples, list):
            has_labels = len(true_labels_for_samples) > 0
        else:
            has_labels = bool(true_labels_for_samples)
            
        if has_labels:
            print(f"DEBUG: Using true labels from embeddings")
            class_labels = np.array(true_labels_for_samples)
            print(f"DEBUG: class_labels now: {type(class_labels)} with shape {class_labels.shape if isinstance(class_labels, np.ndarray) else 'not array'}")
    elif class_labels:
        print(f"DEBUG: Using class labels from metadata: {class_labels}")
    
    # Build the figure
    title = f"{dataset.title()} Neural Network Layer Trajectories"
    try:
        print(f"DEBUG: Before build_single_scene_figure, class_labels type: {type(class_labels)}")
        if isinstance(class_labels, np.ndarray):
            print(f"DEBUG: class_labels shape: {class_labels.shape}")
        else:
            print(f"DEBUG: class_labels not an ndarray: {class_labels}")
            
        # Ensure we pass class_labels as a numpy array if it isn't already
        class_labels_arg = np.array(class_labels) if class_labels is not None and not isinstance(class_labels, np.ndarray) else class_labels
        
        # Build the trajectory figure
        fig = build_single_scene_figure(
            embeddings_dict=filtered_embeddings,
            samples_to_plot=None,  # Use all available samples up to max
            max_samples=n_samples,
            highlight_indices=highlight_indices,
            class_labels=class_labels_arg,
            sample_color_values=sample_color_values,
            color_value_name=selected_point_metric if color_by == "metric" else "",
            title=title,
            show_arrows=show_arrows,
            normalize=normalize,
            layer_clusters=filtered_layer_clusters,
            show_cluster_centers=show_cluster_centers,
            color_by=color_by,
            cluster_name_mapping=cluster_name_mapping  # Pass the custom cluster name mapping
        )
        
        # Now build the fracture graph using the fracture_graph function
        fracture_fig = create_fracture_graph(stored_data, stored_clusters, stored_true_labels, dataset, vis_controls, seed, selected_point_metric)
        
        return fig, fracture_fig
    except Exception as e:
        import traceback
        print(f"DEBUG: Error building figure: {e}")
        print(f"DEBUG: {traceback.format_exc()}")
        
        # Create error figures for both graphs
        traj_error_fig = go.Figure().update_layout(
            title=f"Error in trajectory graph: {str(e)}",
            height=700
        )
        
        # Try to at least generate the fracture graph even if the trajectory graph fails
        try:
            fracture_fig = create_fracture_graph(stored_data, stored_clusters, stored_true_labels, dataset, vis_controls, seed, selected_point_metric)
        except Exception as frac_e:
            print(f"DEBUG: Error building fracture figure: {frac_e}")
            fracture_fig = go.Figure().update_layout(
                title=f"Error in fracture graph: {str(frac_e)}",
                xaxis_title="Layer",
                yaxis_title="Fragmentation Score",
                height=250
            )
            
        return traj_error_fig, fracture_fig

@app.callback(
    Output("download-html", "data"),
    [Input("export-button", "n_clicks")],
    [State("trajectory-graph", "figure"),
     State("dataset-dropdown", "value")]
)
def export_html(n_clicks, figure, dataset):
    """Export the current visualization as an HTML file."""
    if n_clicks == 0 or figure is None:
        return None
    
    # Convert figure to HTML
    import plotly.io as pio
    html_str = pio.to_html(figure, full_html=True, include_plotlyjs="cdn")
    
    return dict(
        content=html_str,
        filename=f"{dataset}_trajectories.html"
    )

# Add cache clearing callback
@app.callback(
    Output("cache-cleared-message", "children"),
    [Input("clear-cache-button", "n_clicks")]
)
def clear_cache(n_clicks):
    if n_clicks > 0:
        try:
            # Delete all files in cache directory
            files_deleted = 0
            for f in os.listdir(CACHE_DIR):
                file_path = os.path.join(CACHE_DIR, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_deleted += 1
            return f"Cache cleared successfully! {files_deleted} files deleted."
        except Exception as e:
            return f"Error clearing cache: {str(e)}"
    return ""

@app.callback(
    Output("cluster-summary", "children"),
    [Input("trajectory-graph", "clickData")],
    [State("dataset-data-store", "data"),
     State("layer-clusters-store", "data"),
     State("dataset-dropdown", "value"),
     State("seed-dropdown", "value")]
)
def show_cluster_stats(click_data, dataset_json, clusters_json, dataset, seed):
    """Show descriptive statistics for the clicked cluster with enhanced demographic information."""
    if not click_data or not dataset_json or not clusters_json or not dataset:
        return "Click a cluster center to view statistics."
    
    try:
        # Extract the point data and ensure it's a cluster center (has customdata)
        points = click_data.get("points", [])
        if not points or "customdata" not in points[0]:
            return "Click a cluster center to view statistics."
        
        # Extract cluster details
        cluster_id, layer, config_name = points[0]["customdata"]
        
        # Convert clusters back from JSON
        clusters = list_to_numpy_recursive(clusters_json)
        
        # Get cluster labels for this layer
        if config_name not in clusters or layer not in clusters[config_name]:
            return f"No cluster data found for {config_name}, {layer}."
        
        cluster_info = clusters[config_name][layer]
        if "labels" not in cluster_info:
            return f"No cluster labels found for {config_name}, {layer}."
        
        # Get indices of samples in this cluster
        cluster_labels = cluster_info["labels"]
        cluster_mask = cluster_labels == cluster_id
        sample_indices = np.where(cluster_mask)[0]
        
        if len(sample_indices) == 0:
            return f"No samples found in Cluster {cluster_id} of {layer}."
        
        # Load cluster paths data for additional context if available
        cluster_paths_data = load_cluster_paths_data(dataset, seed)
        
        # Try to get enhanced metrics from the paths data
        target_info = None
        demographic_info = None
        cluster_description = f"Cluster {cluster_id}"
        
        if cluster_paths_data and "path_archetypes" in cluster_paths_data:
            # Find archetype path that matches this cluster at this layer
            layer_idx = None
            try:
                layer_idx = cluster_paths_data["layers"].index(layer)
            except ValueError:
                # Layer not found in paths, try to match by position
                if layer.startswith("layer") and len(cluster_paths_data["layers"]) > 1:
                    # Try to match numeric part if this is layerN
                    layer_match = re.match(r'layer(\d+)', layer)
                    if layer_match:
                        layer_num = int(layer_match.group(1))
                        # Adjust for 0-indexing vs 1-indexing
                        if 0 < layer_num <= len(cluster_paths_data["layers"]):
                            layer_idx = layer_num - 1
            
            if layer_idx is not None:
                # Find archetypes with this cluster in this layer position
                matching_archetypes = []
                for archetype in cluster_paths_data["path_archetypes"]:
                    path_clusters = archetype["path"].split("→")
                    if layer_idx < len(path_clusters) and path_clusters[layer_idx] == str(cluster_id):
                        matching_archetypes.append(archetype)
                
                if matching_archetypes:
                    # Use largest matching archetype for stats
                    largest_archetype = max(matching_archetypes, key=lambda x: x["count"])
                    
                    # Add target information if available (e.g., survival rate for Titanic)
                    if dataset == "titanic" and "survived_rate" in largest_archetype:
                        survival_rate = largest_archetype["survived_rate"]
                        target_info = f"Survival Rate: {survival_rate:.1%}"
                        
                        # Add descriptive label based on survival rate
                        if survival_rate > 0.7:
                            cluster_description = f"High Survival Cluster (Cluster {cluster_id})"
                        elif survival_rate < 0.3:
                            cluster_description = f"Low Survival Cluster (Cluster {cluster_id})"
                    
                    # Extract demographic highlights if available
                    if "demo_stats" in largest_archetype:
                        demo_stats = largest_archetype["demo_stats"]
                        demo_highlights = []
                        
                        # Process demographics by dataset
                        if dataset == "titanic":
                            # Process sex distribution if available
                            if "sex" in demo_stats:
                                sex_dist = demo_stats["sex"]
                                if isinstance(sex_dist, dict):
                                    if "male" in sex_dist and sex_dist["male"] > 0.7:
                                        demo_highlights.append("Predominantly Male")
                                    elif "male" in sex_dist and sex_dist["male"] < 0.3:
                                        demo_highlights.append("Predominantly Female")
                            
                            # Process class distribution
                            if "pclass" in demo_stats:
                                pclass_dist = demo_stats["pclass"]
                                if isinstance(pclass_dist, dict):
                                    class_desc = []
                                    if "1" in pclass_dist and pclass_dist["1"] > 0.5:
                                        class_desc.append("1st Class")
                                    if "2" in pclass_dist and pclass_dist["2"] > 0.5:
                                        class_desc.append("2nd Class") 
                                    if "3" in pclass_dist and pclass_dist["3"] > 0.5:
                                        class_desc.append("3rd Class")
                                    
                                    if class_desc:
                                        demo_highlights.append(" & ".join(class_desc) + " Passengers")
                            
                            # Process age information
                            if "age" in demo_stats:
                                age_stats = demo_stats["age"]
                                if isinstance(age_stats, dict) and "mean" in age_stats:
                                    mean_age = age_stats["mean"]
                                    if mean_age < 18:
                                        demo_highlights.append("Children/Youth")
                                    elif mean_age < 30:
                                        demo_highlights.append("Young Adults")
                                    elif mean_age > 50:
                                        demo_highlights.append("Older Adults")
                        
                        # Adult dataset
                        elif dataset == "adult":
                            # Process educational stats
                            if "education" in demo_stats:
                                edu_dist = demo_stats["education"]
                                top_edu = max(edu_dist.items(), key=lambda x: x[1]) if isinstance(edu_dist, dict) else None
                                if top_edu and top_edu[1] > 0.4:
                                    demo_highlights.append(f"{top_edu[0].title()} Education")
                            
                            # Process occupation stats
                            if "occupation" in demo_stats:
                                occ_dist = demo_stats["occupation"]
                                top_occ = max(occ_dist.items(), key=lambda x: x[1]) if isinstance(occ_dist, dict) else None
                                if top_occ and top_occ[1] > 0.3:
                                    demo_highlights.append(f"{top_occ[0].title()} Occupation")
                        
                        # Heart dataset 
                        elif dataset == "heart":
                            # Process chest pain type
                            if "cp" in demo_stats:
                                cp_dist = demo_stats["cp"]
                                top_cp = max(cp_dist.items(), key=lambda x: x[1]) if isinstance(cp_dist, dict) else None
                                if top_cp and top_cp[1] > 0.5:
                                    cp_types = {
                                        "0": "Typical Angina", 
                                        "1": "Atypical Angina",
                                        "2": "Non-anginal Pain",
                                        "3": "Asymptomatic"
                                    }
                                    cp_desc = cp_types.get(top_cp[0], f"CP Type {top_cp[0]}")
                                    demo_highlights.append(cp_desc)
                        
                        # Combine demographic highlights
                        if demo_highlights:
                            demographic_info = ", ".join(demo_highlights)
                            
                            # Update cluster description with demographic info
                            if len(demo_highlights) == 1:
                                cluster_description = f"{demo_highlights[0]} (Cluster {cluster_id})"
                            elif len(demo_highlights) > 1:
                                # Use first demographic feature if multiple exist
                                cluster_description = f"{demo_highlights[0]} (Cluster {cluster_id})"
        
        try:
            # Load and prepare feature dataframe
            if isinstance(dataset_json, str):
                df = pd.read_json(dataset_json, orient="split")
            else:
                df = pd.DataFrame(dataset_json["data"], columns=dataset_json["columns"])
            
            # Create a dataframe of available features for this cluster
            feature_df = pd.DataFrame()
            
            # Get the available features based on dataset
            if dataset == "titanic":
                available_features = ["Age", "Sex", "Pclass", "Fare", "Survived"]
            elif dataset == "adult":
                available_features = ["Age", "Education", "Occupation", "Race", "Sex", "Income"]
            elif dataset == "heart":
                available_features = ["Age", "Sex", "CP", "Trestbps", "Chol", "Target"]
            else:
                available_features = df.columns.tolist()
            
            for feature in available_features:
                if feature in df.columns:
                    feature_df[feature] = df[feature]
                elif feature.lower() in df.columns:
                    feature_df[feature] = df[feature.lower()]
            
            # Convert any string sex column to numeric
            if 'Sex' in feature_df.columns:
                feature_df['Sex'] = feature_df['Sex'].apply(lambda x: 1 if str(x).lower() in ['male', 'm'] else 0)
            elif 'sex' in feature_df.columns:
                feature_df['sex'] = feature_df['sex'].apply(lambda x: 1 if str(x).lower() in ['male', 'm'] else 0)
            
            # Select samples in this cluster
            if len(sample_indices) > feature_df.shape[0]:
                sample_indices = sample_indices[:feature_df.shape[0]]
            
            cluster_samples = feature_df.iloc[sample_indices]
            
            # Calculate statistics
            stats = cluster_samples.describe().round(2)
            
        except Exception as e:
            import traceback
            print(f"DEBUG ERROR getting features: {str(e)}\n{traceback.format_exc()}")
            
            # Fallback to original approach
            df = pd.read_json(dataset_json, orient="split")
            
            # Select samples in this cluster
            if len(sample_indices) > df.shape[0]:
                sample_indices = sample_indices[:df.shape[0]]
            
            cluster_samples = df.iloc[sample_indices]
            stats = cluster_samples.describe().round(2)
        
        # Create data table
        table = dash_table.DataTable(
            id='cluster-stats-table',
            columns=[
                {"name": "Metric", "id": "metric"},
                *[{"name": col, "id": col} for col in stats.columns]
            ],
            data=[
                {"metric": idx, **{col: stats.loc[idx, col] for col in stats.columns}}
                for idx in stats.index
            ],
            style_table={
                'overflowX': 'auto',
                'width': '100%'
            },
            style_cell={
                'textAlign': 'left',
                'padding': '8px',
                'font-family': 'sans-serif'
            },
            style_header={
                'backgroundColor': '#f4f4f4',
                'fontWeight': 'bold'
            }
        )
        
        # Create enhanced summary with demographic information
        summary_div = html.Div([
            html.H4(cluster_description),
            html.H5(f"Layer: {get_friendly_layer_name(layer)}, Configuration: {config_name.title()}"),
            html.P(f"Number of samples: {len(sample_indices)}"),
        ])
        
        # Add target info if available
        if target_info:
            summary_div.children.append(html.P(target_info, style={"fontWeight": "bold"}))
            
        # Add demographic info if available
        if demographic_info:
            summary_div.children.append(html.P(f"Demographics: {demographic_info}"))
            
        # Add data table
        summary_div.children.append(table)
        
        return summary_div
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating cluster statistics: {str(e)}\n{traceback.format_exc()}"
        print(f"DEBUG ERROR IN CLUSTER STATS: {error_msg}")
        return f"Error: {str(e)}"

# Diagnostic for the missing fracture line graph
print("DEBUG: Current app layout has these children:")
for i, child in enumerate(app.layout.children):
    try:
        child_id = child.id if hasattr(child, "id") else f"No id (type: {type(child)})"
        print(f"Layout child {i}: {child_id}")
    except:
        print(f"Layout child {i}: <error getting info>")

print("DEBUG: Fixed version - removed conflicting callbacks")

# Add this function if it doesn't exist (similar to the one in traj_plot.py)
def get_friendly_layer_name(layer_name: str) -> str:
    """
    Convert internal layer names to user-friendly display names.
    
    Args:
        layer_name: The internal layer name (e.g., 'layer1', 'input')
        
    Returns:
        A user-friendly layer name (e.g., 'Input Space', 'Layer 1')
    """
    if layer_name == "input":
        return "Input Space"
    elif layer_name == "output":
        return "Output Layer"
    elif layer_name.startswith("layer"):
        # Extract layer number if present
        match = re.match(r'layer(\d+)', layer_name)
        if match:
            layer_num = int(match.group(1))
            if layer_num == 1:
                return "Input Space"  # Hidden Layer 1 is now Input Space
            elif layer_num == 2:
                return "Layer 1"      # Hidden Layer 2 is now Layer 1
            elif layer_num == 3:
                return "Layer 2"      # Hidden Layer 3 is now Layer 2
            else:
                return f"Layer {layer_num-1}"  # Other layers shifted by 1
    
    # If no pattern matches, just capitalize and clean up the name
    return layer_name.replace('_', ' ').title()

def create_fracture_graph(stored_data, stored_clusters, stored_true_labels, dataset, vis_controls, seed, selected_metric):
    """
    Create the fracture metrics line graph showing the three key metrics:
    1. Intra-Class Pairwise Distance (ICPD)
    2. Optimal Number of Clusters (k*)
    3. Representation Stability (Δ-Norm)
    
    Additionally, shows the traditional metrics for comparison:
    - Cluster Entropy
    - Subspace Angle
    """
    # Initialize figure
    fig = go.Figure()
    
    if not stored_data or not stored_clusters or not stored_true_labels:
        fig.update_layout(
            title="No data available for fracture metrics",
            xaxis_title="Layer",
            yaxis_title="Metric Value",
            height=250
        )
        return fig
    
    # Select which config to use
    config_name = "baseline"
    
    # Determine which metrics to show based on selected_metric
    metrics_to_show = selected_metric
    
    # Convert data from JSON
    embeddings = list_to_numpy_recursive(stored_data)
    clusters = list_to_numpy_recursive(stored_clusters)
    true_labels = list_to_numpy_recursive(stored_true_labels)
    
    # Check if we have valid baseline data
    if config_name not in embeddings or config_name not in clusters:
        fig.update_layout(
            title=f"No {config_name} data available for fracture metrics",
            xaxis_title="Layer",
            yaxis_title="Metric Value",
            height=250
        )
        return fig
    
    # Get class labels
    class_labels = true_labels.get(config_name)
    if class_labels is None or len(class_labels) == 0:
        fig.update_layout(
            title="No class labels available for fracture metrics",
            xaxis_title="Layer",
            yaxis_title="Metric Value",
            height=250
        )
        return fig
    
    # Get layer clusters
    layer_clusters = clusters.get(config_name)
    if layer_clusters is None:
        fig.update_layout(
            title=f"No {config_name} cluster data available",
            xaxis_title="Layer",
            yaxis_title="Metric Value",
            height=250
        )
        return fig
    
    # Track all available layers
    all_layers = set()
    
    # We need to compute metrics for several layers
    # Import proper metrics from concept_fragmentation
    from concept_fragmentation.metrics.cluster_entropy import compute_cluster_entropy
    from concept_fragmentation.metrics.intra_class_distance import compute_intra_class_distance
    from concept_fragmentation.metrics.representation_stability import compute_representation_stability
    from concept_fragmentation.metrics.subspace_angle import compute_subspace_angle
    
    # Set up aliases for consistent function naming
    cluster_entropy = compute_cluster_entropy
    icpd_score = compute_intra_class_distance
    repstability_score = compute_representation_stability
    subspace_angle = compute_subspace_angle
    
    # Store metric values
    metric_values = {
        "cluster_entropy": [],
        "icpd": [],
        "kstar": [],
        "stability": [],
        "subspace_angle": []
    }
    layer_names = []
    
    # Calculate metrics for each layer
    for layer_name, cluster_info in layer_clusters.items():
        all_layers.add(layer_name)
        
        if "labels" not in cluster_info or "k" not in cluster_info:
            continue
            
        # Get cluster labels and k for this layer
        cluster_labels = cluster_info["labels"]
        k = cluster_info["k"]
        
        # Skip if k < 2 (not meaningful for most metrics)
        if k < 2 and "kstar" not in metrics_to_show:
            continue
            
        # Ensure class_labels and cluster_labels have compatible shapes
        if len(class_labels) != len(cluster_labels):
            if len(class_labels) > len(cluster_labels):
                class_labels = class_labels[:len(cluster_labels)]
            else:
                # Can't compute if we don't have enough class labels
                continue
        
        # Convert seed to string for dictionary lookup
        seed_key = str(seed)
        
        # Compute the metrics
        # Use proper embeddings access with string key
        if config_name in embeddings and seed_key in embeddings[config_name] and layer_name in embeddings[config_name][seed_key]:
            layer_embeddings = embeddings[config_name][seed_key][layer_name]
            # Compute cluster entropy using embeddings
            entropy_result = cluster_entropy(layer_embeddings, class_labels)
            entropy_val = entropy_result.get('mean_entropy', 0.0)
        else:
            # Fallback to zeros if embeddings not available
            entropy_val = 0.0
            
        # ICPD requires activations and class labels
        if config_name in embeddings and seed_key in embeddings[config_name] and layer_name in embeddings[config_name][seed_key]:
            layer_embeddings = embeddings[config_name][seed_key][layer_name]
            icpd_result = icpd_score(layer_embeddings, class_labels)
            icpd_val = icpd_result.get('mean_distance', 0.0)
        else:
            icpd_val = 0.0
        
        # k* is simply the optimal number of clusters
        kstar_val = float(k)
        
        # For stability, we need to compare with previous layer
        # For now, just use a placeholder
        stability_val = 0.0
        
        # Subspace angle requires embeddings
        angle_val = 0.0
        if config_name in embeddings and seed_key in embeddings[config_name] and layer_name in embeddings[config_name][seed_key]:
            layer_embeddings = embeddings[config_name][seed_key][layer_name]
            # Simple version: just use mean intra-class angle
            angle_result = subspace_angle(layer_embeddings, class_labels)
            angle_val = angle_result.get('mean_angle', 0.0)
            
        # Store values
        metric_values["cluster_entropy"].append(entropy_val)
        metric_values["icpd"].append(icpd_val)
        metric_values["kstar"].append(kstar_val)
        metric_values["stability"].append(stability_val)
        metric_values["subspace_angle"].append(angle_val)
        layer_names.append(layer_name)
    
    # Sort layers by natural order
    try:
        # Define a natural sort key function if not already present
        def _natural_layer_sort(layer_name):
            if layer_name == "input":
                return (0, 0)
            elif layer_name == "output":
                return (999, 0)
            elif layer_name.startswith("layer"):
                match = re.match(r'layer(\d+)', layer_name)
                if match:
                    return (1, int(match.group(1)))
            return (2, layer_name)
            
        indices = [i for i, _ in sorted(enumerate(layer_names), key=lambda x: _natural_layer_sort(x[1]))]
        sorted_layer_names = [layer_names[i] for i in indices]
    except Exception as e:
        print(f"Error sorting layers: {e}")
        sorted_layer_names = layer_names
    
    # Sort metric values by the same indices
    for metric in metric_values:
        try:
            metric_values[metric] = [metric_values[metric][i] for i in indices]
        except Exception as e:
            print(f"Error sorting metric values: {e}")
    
    # Get friendly layer names for the x-axis
    friendly_layer_names = [get_friendly_layer_name(layer) for layer in sorted_layer_names]
    
    # Plot the requested metric
    if metrics_to_show in metric_values:
        metric_label = AVAILABLE_POINT_COLOR_METRICS.get(metrics_to_show, metrics_to_show)
        fig.add_trace(go.Scatter(
            x=list(range(len(sorted_layer_names))),
            y=metric_values[metrics_to_show],
            mode='lines+markers',
            name=metric_label,
            line=dict(color='rgb(31, 119, 180)', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f"{metric_label} by Layer",
            xaxis=dict(
                title="Layer",
                tickvals=list(range(len(sorted_layer_names))),
                ticktext=friendly_layer_names
            ),
            yaxis=dict(title=metric_label),
            height=250,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        fig.update_layout(
            title=f"Metric '{metrics_to_show}' not available",
            xaxis_title="Layer",
            yaxis_title="Metric Value",
            height=250
        )
    
    return fig

# Add this function after the load_and_embed_data function
def load_cluster_paths_data(dataset: str, seed: int) -> Optional[Dict[str, Any]]:
    """Load cluster paths data for the given dataset and seed."""
    try:
        # Path to cluster paths JSON file
        paths_file = os.path.join("data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json")
        
        if not os.path.exists(paths_file):
            print(f"Cluster paths file not found: {paths_file}")
            return None
            
        # Load the JSON data
        with open(paths_file, 'r') as f:
            cluster_paths_data = json.load(f)
            
        print(f"Loaded cluster paths data for {dataset} (seed {seed})")
        return cluster_paths_data
    
    except Exception as e:
        import traceback
        error_msg = f"Error loading cluster paths data: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

if __name__ == "__main__":
    print("Starting Dash app...")
    print("Navigate to http://127.0.0.1:8050/ in your web browser.")
    app.run_server(debug=True) 