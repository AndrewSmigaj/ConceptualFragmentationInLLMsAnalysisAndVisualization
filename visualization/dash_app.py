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
import networkx as nx

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
from visualization.traj_plot import build_single_scene_figure, plot_dataset_trajectories, save_figure, LAYER_SEPARATION_OFFSET, get_friendly_layer_name
from visualization.cluster_utils import compute_layer_clusters_embedded, get_embedded_clusters_cache_path
from concept_fragmentation.config import DATASETS as DATASET_CFG
from concept_fragmentation.analysis.cluster_stats import write_cluster_stats

# Import cross-layer metrics
from concept_fragmentation.metrics.cross_layer_metrics import (
    compute_centroid_similarity, compute_membership_overlap,
    compute_trajectory_fragmentation, compute_path_density,
    analyze_cross_layer_metrics
)

# Import LLM tab
from visualization.llm_tab import create_llm_tab, register_llm_callbacks

# Import GPT-2 metrics tab
from visualization.gpt2_metrics_tab import create_gpt2_metrics_tab, register_gpt2_metrics_callbacks

# Import GPT-2 persistence
from concept_fragmentation.persistence import GPT2AnalysisPersistence

# Import path metrics tab
from visualization.path_metrics_tab import (
    create_path_metrics_tab,
    register_path_metrics_callbacks
)

from visualization.cross_layer_viz import (
    plot_centroid_similarity_heatmap, plot_membership_overlap_sankey,
    plot_trajectory_fragmentation_bars, plot_path_density_network
)
from visualization.cross_layer_utils import (
    serialize_cross_layer_metrics, deserialize_cross_layer_metrics,
    networkx_to_dict, dict_to_networkx
)

# Import similarity network visualization
from visualization.similarity_network_tab import (
    create_similarity_network_tab,
    register_similarity_callbacks
)

# Import path fragmentation visualization
from visualization.path_fragmentation_tab import (
    create_path_fragmentation_tab,
    register_path_fragmentation_callbacks
)

# Import LLM availability checker
from visualization.llm_tab import check_provider_availability

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
    
    # Main tabs
    dcc.Tabs([
        # Similarity Network Tab
        create_similarity_network_tab(),
        
        # Path Fragmentation Tab
        create_path_fragmentation_tab(),
        
        # Path Metrics Tab
        create_path_metrics_tab(),
        
        # GPT-2 Metrics Dashboard Tab
        create_gpt2_metrics_tab(),
        
        # LLM Integration Tab
        create_llm_tab(),
        
        # Tab 1: Trajectory Visualization
        dcc.Tab(label="Trajectory Visualization", children=[
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
                            {"label": "Normalize Embeddings", "value": "normalize"},
                            {"label": "Shape by Class", "value": "shape_by_class"}
                        ],
                        value=["baseline", "regularized", "arrows", "normalize", "shape_by_class"]
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
        ]), 
        
        # Tab 2: Cross-Layer Metrics
        dcc.Tab(label="Cross-Layer Metrics", children=[
            html.Div([
                html.Div([
                    html.Label("Dataset:"),
                    dcc.Dropdown(
                        id="cl-dataset-dropdown",
                        options=[{"label": d.title(), "value": d} for d in DATASETS],
                        value=DATASETS[0] if DATASETS else None
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Label("Seed:"),
                    dcc.Dropdown(
                        id="cl-seed-dropdown",
                        options=[{"label": f"Seed {s}", "value": s} for s in SEEDS],
                        value=SEEDS[0] if SEEDS else None
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Label("Configuration:"),
                    dcc.RadioItems(
                        id="cl-config-radio",
                        options=[
                            {"label": "Baseline", "value": "baseline"},
                            {"label": "Regularized", "value": "regularized"}
                        ],
                        value="baseline",
                        inline=True
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Metric Controls:"),
                    dcc.Checklist(
                        id="cl-metric-controls",
                        options=[
                            {"label": "Centroid Similarity", "value": "centroid_similarity"},
                            {"label": "Membership Overlap", "value": "membership_overlap"},
                            {"label": "Trajectory Fragmentation", "value": "trajectory_fragmentation"},
                            {"label": "Path Density", "value": "path_density"}
                        ],
                        value=["centroid_similarity", "membership_overlap", "trajectory_fragmentation", "path_density"]
                    ),
                ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
                
                html.Div([
                    html.Label("Visualization Parameters:"),
                    html.Br(),
                    html.Label("Min. Overlap Threshold:"),
                    dcc.Slider(
                        id="cl-overlap-slider",
                        min=0.05,
                        max=0.5,
                        step=0.05,
                        value=0.1,
                        marks={i/100: str(i/100) for i in range(5, 55, 10)},
                    ),
                ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
                
                html.Div([
                    html.Button("Update Cross-Layer Metrics", id="cl-update-button", n_clicks=0, 
                                style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px"}),
                ], style={"width": "20%", "display": "inline-block", "verticalAlign": "top", "padding": "20px"}),
            ]),
            
            # Cross-Layer Metrics Visualizations
            html.Div([
                # Loading indicator for cross-layer metrics
                dcc.Loading(
                    id="loading-cl-metrics",
                    type="circle",
                    children=[
                        # Centroid Similarity Section
                        html.Div([
                            html.H3("Centroid Similarity Heatmaps", style={"textAlign": "center"}),
                            dcc.Graph(id="centroid-similarity-graph")
                        ], id="centroid-similarity-section"),
                        
                        # Membership Overlap Section
                        html.Div([
                            html.H3("Membership Overlap Flow", style={"textAlign": "center"}),
                            dcc.Graph(id="membership-overlap-graph")
                        ], id="membership-overlap-section"),
                        
                        # Trajectory Fragmentation Section
                        html.Div([
                            html.H3("Trajectory Fragmentation by Layer", style={"textAlign": "center"}),
                            dcc.Graph(id="trajectory-fragmentation-graph")
                        ], id="trajectory-fragmentation-section"),
                        
                        # Path Density Section
                        html.Div([
                            html.H3("Inter-Cluster Path Density", style={"textAlign": "center"}),
                            dcc.Graph(id="path-density-graph")
                        ], id="path-density-section"),
                    ]
                )
            ])
        ])
    ]),
    
    html.Div(id="status-message", style={"padding": "10px", "color": "gray"}),
    
    # Hidden data storage
    dcc.Store(id="embeddings-store"),
    dcc.Store(id="layer-clusters-store"),
    dcc.Store(id="true-labels-store"),
    dcc.Store(id="dataset-data-store"),
    
    # Storage for cross-layer metrics
    dcc.Store(id="cl-metrics-store")
])

def load_and_embed_data(dataset, seed, max_k=10):
    """Load and embed data for the given dataset and seed."""
    try:
        # Get configurations
        best_config = get_best_config(dataset)
        baseline_config = get_baseline_config(dataset)
        
        print(f"DEBUG: Using dataset={dataset}, seed={seed}")
        print(f"DEBUG: Baseline config = {baseline_config}")
        print(f"DEBUG: Best config = {best_config}")
        
        # Create embedder
        embedder = Embedder(n_components=3, random_state=42, cache_dir=CACHE_DIR)
        
        # Embed baseline
        print(f"Embedding baseline data for {dataset}, seed {seed}")
        baseline_embeddings, baseline_true_labels = embed_layer_activations(
            dataset, baseline_config, seed, embedder=embedder, use_test_set=True)
        
        # Check what layers we got in baseline
        if baseline_embeddings:
            print(f"DEBUG: Baseline layers: {list(baseline_embeddings.keys())}")
            for layer, embedding in baseline_embeddings.items():
                print(f"  DEBUG: Baseline {layer} shape = {embedding.shape}")
        else:
            print("DEBUG: No baseline embeddings were returned")
        
        # Embed best config
        print(f"Embedding regularized data for {dataset}, seed {seed}")
        best_embeddings, best_true_labels = embed_layer_activations(
            dataset, best_config, seed, embedder=embedder, use_test_set=True)
        
        # Check what layers we got in regularized
        if best_embeddings:
            print(f"DEBUG: Regularized layers: {list(best_embeddings.keys())}")
            for layer, embedding in best_embeddings.items():
                print(f"  DEBUG: Regularized {layer} shape = {embedding.shape}")
        else:
            print("DEBUG: No regularized embeddings were returned")
        
        # Create dictionary for storage
        embeddings_dict = {
            "baseline": {seed: baseline_embeddings},
            "regularized": {seed: best_embeddings}
        }
        
        # Check that we have at least some layers for each config
        if not baseline_embeddings or len(baseline_embeddings) == 0:
            print("WARNING: No baseline embeddings available, will attempt to continue with just regularized")
        
        if not best_embeddings or len(best_embeddings) == 0:
            print("WARNING: No regularized embeddings available, will attempt to continue with just baseline")
        
        if (not baseline_embeddings or len(baseline_embeddings) == 0) and (not best_embeddings or len(best_embeddings) == 0):
            raise ValueError(f"No embeddings available for either baseline or regularized configurations")
        
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
        os.makedirs(baseline_cache_dir, exist_ok=True)
        os.makedirs(best_cache_dir, exist_ok=True)
        
        baseline_cache_path = get_embedded_clusters_cache_path(
            dataset, "baseline", seed, max_k, cache_dir=baseline_cache_dir
        )
        best_config_id = get_config_id(best_config)
        best_cache_path = get_embedded_clusters_cache_path(
            dataset, best_config_id, seed, max_k, cache_dir=best_cache_dir
        )
        
        # Compute clusters for each config
        layer_clusters_by_config = {}
        
        # Process baseline clusters if we have baseline embeddings
        if baseline_embeddings and len(baseline_embeddings) > 0:
            try:
                print(f"Computing baseline clusters from {len(baseline_embeddings)} embeddings")
                baseline_clusters = compute_layer_clusters_embedded(
                    baseline_embeddings, max_k=max_k, random_state=42, cache_path=baseline_cache_path
                )
                
                # Check what layers got clusters
                print(f"DEBUG: Baseline cluster layers: {list(baseline_clusters.keys())}")
                layer_clusters_by_config["baseline"] = baseline_clusters
            except Exception as e:
                print(f"WARNING: Failed to compute baseline clusters: {e}")
        
        # Process regularized clusters if we have regularized embeddings
        if best_embeddings and len(best_embeddings) > 0:
            try:
                print(f"Computing regularized clusters from {len(best_embeddings)} embeddings")
                best_clusters = compute_layer_clusters_embedded(
                    best_embeddings, max_k=max_k, random_state=42, cache_path=best_cache_path
                )
                
                # Check what layers got clusters
                print(f"DEBUG: Regularized cluster layers: {list(best_clusters.keys())}")
                layer_clusters_by_config["regularized"] = best_clusters
            except Exception as e:
                print(f"WARNING: Failed to compute regularized clusters: {e}")
        
        # Final summary
        if "baseline" in layer_clusters_by_config and "regularized" in layer_clusters_by_config:
            print(f"Successfully computed clusters for both configs")
        elif "baseline" in layer_clusters_by_config:
            print(f"Only computed clusters for baseline config")
        elif "regularized" in layer_clusters_by_config:
            print(f"Only computed clusters for regularized config")
        else:
            print(f"WARNING: Failed to compute clusters for any config, visualization will be limited")
        
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
        
        # Check if LLM results exist
        llm_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "llm")
        print(f"Checking for LLM results in: {llm_results_dir}")
        if os.path.exists(llm_results_dir):
            llm_files = [f for f in os.listdir(llm_results_dir) if f.startswith(f"{dataset}_seed_{seed}")]
            if llm_files:
                print(f"Found LLM results files: {llm_files}")
        
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
    
    # Detailed debug of stored data embeddings
    if stored_data and isinstance(stored_data, dict):
        for config_name, config_data in stored_data.items():
            print(f"DEBUG: Config: {config_name}")
            if isinstance(config_data, dict):
                for seed, seed_data in config_data.items():
                    print(f"DEBUG:   Seed: {seed}")
                    if isinstance(seed_data, dict):
                        print(f"DEBUG:     Available layers: {list(seed_data.keys())}")
                        for layer_name, layer_data in seed_data.items():
                            print(f"DEBUG:       Layer {layer_name}: type={type(layer_data)}")
                            if hasattr(layer_data, 'shape'):
                                print(f"DEBUG:       Layer {layer_name} shape: {layer_data.shape}")
    
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
    cluster_name_mapping = {}
    
    # Try to load LLM-generated labels first
    try:
        # Look for LLM results with the latest timestamp
        llm_results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "llm")
        if os.path.exists(llm_results_dir):
            llm_files = [f for f in os.listdir(llm_results_dir) if f.startswith(f"{dataset}_seed_{seed}") and f.endswith(".json")]
            if llm_files:
                # Sort by timestamp to get the most recent
                llm_files.sort(reverse=True)
                latest_file = os.path.join(llm_results_dir, llm_files[0])
                print(f"Loading cluster labels from {latest_file}")
                with open(latest_file, 'r') as f:
                    llm_results = json.load(f)
                    if "cluster_labels" in llm_results:
                        print(f"Found {len(llm_results['cluster_labels'])} cluster labels from LLM analysis")
                        cluster_name_mapping = llm_results["cluster_labels"]
    except Exception as e:
        print(f"Error loading LLM cluster labels: {e}")
    
    # Fall back to hardcoded labels if no LLM labels were found or loading failed
    if not cluster_name_mapping:
        print("Using hardcoded cluster labels as fallback")
        if dataset == "titanic":
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
        elif dataset == "heart":
            cluster_name_mapping = {
                # Layer 1
                "baseline-layer1-cluster0": "Low-risk middle-aged patients",
                "baseline-layer1-cluster1": "High-risk elderly patients with chest pain",
                "baseline-layer1-cluster2": "Middle-aged men with elevated cholesterol",
                "baseline-layer1-cluster3": "Younger patients with atypical symptoms",

                # Layer 2
                "baseline-layer2-cluster0": "Elderly patients with cardiac abnormalities",
                "baseline-layer2-cluster1": "Middle-aged women with typical angina",
                "baseline-layer2-cluster2": "High-risk males with multiple risk factors",
                "baseline-layer2-cluster3": "Younger patients with reversible defects",

                # Layer 3
                "baseline-layer3-cluster0": "Severe cardiac cases with chest pain",
                "baseline-layer3-cluster1": "Low-risk patients with good test results",
                "baseline-layer3-cluster2": "High-risk elderly with abnormal ECG"
            }
    
    # Ensure we have correct keys in the mapping (layer3 may be called 'hidden3' in some contexts)
    # Create additional entries with alternative keys to make sure all layers are covered
    updated_mapping = cluster_name_mapping.copy()
    for key, value in cluster_name_mapping.items():
        # Convert keys like "baseline-layer3-cluster0" to "baseline-hidden3-cluster0"
        if "layer3" in key:
            hidden_key = key.replace("layer3", "hidden3")
            updated_mapping[hidden_key] = value
        
        # Also handle layerX to hiddenX conversion
        if "layer1" in key:
            hidden_key = key.replace("layer1", "hidden1")
            updated_mapping[hidden_key] = value
        elif "layer2" in key:
            hidden_key = key.replace("layer2", "hidden2")
            updated_mapping[hidden_key] = value
    
    cluster_name_mapping = updated_mapping
    print(f"Final cluster name mapping contains {len(cluster_name_mapping)} entries")
    
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
        # Check if "shape_by_class" is in the visualization controls
        use_shape_by_class = "shape_by_class" in vis_controls
        
        # Debug: Print dataset for diagnostic purposes
        print(f"DEBUG: Using dataset={dataset} for visualization")
        
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
            cluster_name_mapping=cluster_name_mapping,  # Pass the custom cluster name mapping
            shape_by_class=use_shape_by_class,  # Use different shapes based on class
            dataset_name=dataset  # Pass dataset name for class labels
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

def load_cluster_paths_data(dataset: str, seed: int) -> Dict[str, Any]:
    """
    Load cluster paths data for the given dataset and seed.
    
    This function searches multiple locations for cluster paths JSON files and validates
    that they contain the required similarity data.
    
    Args:
        dataset: Name of the dataset (e.g., 'titanic')
        seed: Random seed used for the experiment
        
    Returns:
        Dictionary containing cluster paths data, or error information if not found
    """
    import glob
    
    try:
        # Get the project root directory (absolute path)
        # Try multiple approaches to find the project root
        possible_roots = [
            os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # From visualization dir
            os.path.abspath(os.getcwd()),  # Current directory
            os.path.abspath(os.path.dirname(os.getcwd())),  # Parent of current directory
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),  # Relative to this file
        ]
        
        # Define all possible path patterns to check
        possible_paths = []
        
        # For each possible root, add standard paths
        for root in possible_roots:
            # Results directories (most likely to have data with similarity matrices)
            possible_paths.extend([
                os.path.join(root, "results", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
                os.path.join(root, "concept_fragmentation", "results", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
            ])
            
            # Data directories (fallback locations)
            possible_paths.extend([
                os.path.join(root, "data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
                os.path.join(root, "concept_fragmentation", "data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
                os.path.join(root, "visualization", "data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
            ])
        
        # Add relative paths for backward compatibility
        possible_paths.extend([
            os.path.join("results", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
            os.path.join("data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
            os.path.join("visualization", "results", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
            os.path.join("visualization", "data", "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
        ])
        
        # Add recursive glob path to catch files in any subdirectory (more expensive, use last)
        for root in possible_roots:
            possible_paths.append(os.path.join(root, "**", f"{dataset}_seed_{seed}_paths.json"))
        
        # Remove duplicates while preserving order
        unique_paths = []
        seen = set()
        for path in possible_paths:
            normalized = os.path.normpath(path)
            if normalized not in seen:
                seen.add(normalized)
                unique_paths.append(path)
        
        # Try direct paths first (faster than glob)
        checked_paths = []
        direct_match_paths = [p for p in unique_paths if "**" not in p]
        
        for paths_file in direct_match_paths:
            checked_paths.append(paths_file)
            
            if os.path.exists(paths_file):
                with open(paths_file, 'r') as f:
                    try:
                        cluster_paths_data = json.load(f)
                        
                        # Check if it has similarity data
                        has_similarity = (
                            "similarity" in cluster_paths_data and
                            "normalized_similarity" in cluster_paths_data["similarity"] and
                            len(cluster_paths_data["similarity"]["normalized_similarity"]) > 0
                        )
                        
                        if has_similarity:
                            print(f"Successfully loaded cluster paths data with similarity matrix for {dataset} (seed {seed})")
                            similarity_count = len(cluster_paths_data["similarity"]["normalized_similarity"])
                            print(f"Found {similarity_count} similarity connections in {paths_file}")
                            return cluster_paths_data
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON in {paths_file}")
                    except Exception as e:
                        print(f"Warning: Error processing {paths_file}: {str(e)}")
        
        # If no direct matches, try glob patterns
        glob_paths = [p for p in unique_paths if "**" in p]
        for pattern in glob_paths:
            try:
                matching_files = glob.glob(pattern, recursive=True)
                for paths_file in matching_files:
                    checked_paths.append(paths_file)
                    
                    with open(paths_file, 'r') as f:
                        try:
                            cluster_paths_data = json.load(f)
                            
                            # Check if it has similarity data
                            has_similarity = (
                                "similarity" in cluster_paths_data and
                                "normalized_similarity" in cluster_paths_data["similarity"] and
                                len(cluster_paths_data["similarity"]["normalized_similarity"]) > 0
                            )
                            
                            if has_similarity:
                                print(f"Successfully loaded cluster paths data with similarity matrix for {dataset} (seed {seed})")
                                similarity_count = len(cluster_paths_data["similarity"]["normalized_similarity"])
                                print(f"Found {similarity_count} similarity connections in {paths_file}")
                                return cluster_paths_data
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON in {paths_file}")
                        except Exception as e:
                            print(f"Warning: Error processing {paths_file}: {str(e)}")
            except Exception as e:
                print(f"Warning: Error processing glob pattern {pattern}: {str(e)}")
        
        # If we get here, no valid data was found
        missing_data_error = {
            "status": "error",
            "error_type": "missing_data",
            "dataset": dataset,
            "seed": seed,
            "checked_paths": checked_paths,
            "message": f"No cluster paths file with similarity data found for dataset '{dataset}' with seed {seed}",
            "command": f"python -m concept_fragmentation.analysis.cluster_paths --compute_similarity --min_similarity 0.3 --dataset {dataset} --seed {seed}"
        }
        
        # Print helpful message to console
        print(f"\n*** MISSING DATA ERROR ***")
        print(f"No cluster paths file with similarity data found for dataset '{dataset}' with seed {seed}")
        print("\nChecked these locations:")
        for path in checked_paths[:10]:  # Limit to first 10 to avoid overwhelming output
            print(f"  - {path}")
        if len(checked_paths) > 10:
            print(f"  ... and {len(checked_paths) - 10} more locations")
        
        print("\nTo generate the required data, run:")
        print(f"  {missing_data_error['command']}")
        print("This will generate similarity data in the results directory.\n")
        
        return missing_data_error
    
    except Exception as e:
        import traceback
        error_msg = f"Error loading cluster paths data: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        return {
            "status": "error",
            "error_type": "exception",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Cross-Layer Metrics callbacks
@app.callback(
    Output("cl-metrics-store", "data"),
    [Input("cl-update-button", "n_clicks")],
    [State("cl-dataset-dropdown", "value"),
     State("cl-seed-dropdown", "value"),
     State("cl-config-radio", "value"),
     State("layer-clusters-store", "data"),
     State("true-labels-store", "data"),
     State("embeddings-store", "data"),
     State("cl-overlap-slider", "value")]
)
def update_cross_layer_metrics(n_clicks, dataset, seed, config, stored_clusters, stored_true_labels, stored_embeddings, min_overlap):
    """Compute cross-layer metrics and store them."""
    if n_clicks is None or not dataset or seed is None or not config or not stored_clusters:
        return None
    
    try:
        # Convert stored data from JSON
        clusters = list_to_numpy_recursive(stored_clusters)
        true_labels = list_to_numpy_recursive(stored_true_labels)
        embeddings = list_to_numpy_recursive(stored_embeddings)
        
        # Check if we have the selected configuration
        if config not in clusters:
            return {
                "error": f"No cluster data available for {config} configuration."
            }
        
        layer_clusters = clusters[config]
        
        # Get class labels for the selected configuration
        class_labels = true_labels.get(config) if true_labels else None
        
        # Get activations for the selected configuration (if available)
        activations = None
        if embeddings and config in embeddings and str(seed) in embeddings[config]:
            activations = embeddings[config][str(seed)]
        
        # Create configuration dictionary
        metrics_config = {
            "similarity_metric": "cosine",
            "min_overlap": min_overlap if min_overlap is not None else 0.1
        }
        
        # Use the analyze_cross_layer_metrics wrapper function to compute all metrics
        print(f"Computing cross-layer metrics for {dataset} {config} (seed {seed})")
        all_metrics = analyze_cross_layer_metrics(
            layer_clusters=layer_clusters,
            activations=activations,
            class_labels=class_labels,
            config=metrics_config
        )
        
        # Convert complex data structures to JSON-serializable format
        numpy_to_list = numpy_to_list_recursive  # Use existing function for numpy arrays
        
        # Handle the tuple keys in dictionaries (which JSON doesn't support)
        result = {}
        
        # Process centroid_similarity (has tuple keys)
        if "centroid_similarity" in all_metrics:
            centroid_sim = all_metrics["centroid_similarity"]
            # Convert tuple keys to strings and numpy arrays to lists
            result["centroid_similarity"] = {
                str(k): numpy_to_list(v) for k, v in centroid_sim.items()
            }
        
        # Process membership_overlap (has tuple keys)
        if "membership_overlap" in all_metrics:
            membership_overlap = all_metrics["membership_overlap"]
            # Convert tuple keys to strings and numpy arrays to lists
            result["membership_overlap"] = {
                str(k): numpy_to_list(v) for k, v in membership_overlap.items()
            }
        
        # Process trajectory_fragmentation (simple dict)
        if "trajectory_fragmentation" in all_metrics:
            result["trajectory_fragmentation"] = all_metrics["trajectory_fragmentation"]
        
        # Process path_density (has tuple keys)
        if "path_density" in all_metrics:
            path_density = all_metrics["path_density"]
            # Convert tuple keys to strings
            result["path_density"] = {
                str(k): v for k, v in path_density.items()
            }
        
        # Process path_graph (convert NetworkX graph to dict)
        if "path_graph" in all_metrics:
            G = all_metrics["path_graph"]
            # Convert graph to dictionary representation
            graph_dict = {
                "nodes": [],
                "edges": [],
                "directed": isinstance(G, nx.DiGraph)
            }
            
            # Add nodes with attributes
            for node in G.nodes():
                node_attrs = dict(G.nodes[node])
                graph_dict["nodes"].append({
                    "id": node,
                    "attributes": node_attrs
                })
            
            # Add edges with attributes
            for u, v in G.edges():
                edge_attrs = dict(G.edges[u, v])
                graph_dict["edges"].append({
                    "source": u,
                    "target": v,
                    "attributes": edge_attrs
                })
                
            result["path_graph"] = graph_dict
            
        # Add any error messages
        for key in all_metrics:
            if key.endswith("_error"):
                result[key] = str(all_metrics[key])
                
        return result
    
    except Exception as e:
        import traceback
        error_msg = f"Error computing cross-layer metrics: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {"error": str(e)}

@app.callback(
    [Output("centroid-similarity-graph", "figure"),
     Output("centroid-similarity-section", "style")],
    [Input("cl-metrics-store", "data"),
     Input("cl-metric-controls", "value")]
)
def update_centroid_similarity_graph(cl_metrics, metric_controls):
    """Update centroid similarity heatmap visualization."""
    if cl_metrics is None or "error" in cl_metrics:
        return go.Figure().update_layout(
            title="No centroid similarity data available",
            height=500
        ), {"display": "none"}
    
    # Hide if not selected
    if "centroid_similarity" not in metric_controls:
        return go.Figure(), {"display": "none"}
    
    try:
        # Check for centroid similarity error
        if "centroid_similarity_error" in cl_metrics:
            return go.Figure().update_layout(
                title=f"Error: {cl_metrics['centroid_similarity_error']}",
                height=500
            ), {"display": "block"}
        
        # Get centroid similarity data
        centroid_similarity_str = cl_metrics.get("centroid_similarity", {})
        
        # Convert string tuple keys back to actual tuples
        centroid_similarity = {}
        for k_str, v in centroid_similarity_str.items():
            # Convert string representation of tuple back to tuple
            if k_str.startswith('(') and k_str.endswith(')') and ',' in k_str:
                try:
                    # Safely evaluate the string as a tuple
                    k_tuple = eval(k_str)
                    # Ensure it's a proper tuple with two elements
                    if isinstance(k_tuple, tuple) and len(k_tuple) == 2:
                        # Convert values to numpy arrays if needed
                        if isinstance(v, list):
                            v_array = np.array(v)
                            centroid_similarity[k_tuple] = v_array
                        else:
                            centroid_similarity[k_tuple] = v
                except:
                    # Skip malformed keys
                    print(f"Warning: Skipping malformed key: {k_str}")
                    continue
        
        # Check if we have any valid data
        if not centroid_similarity:
            return go.Figure().update_layout(
                title="No valid centroid similarity data available",
                height=500
            ), {"display": "block"}
        
        # Create heatmap
        fig = plot_centroid_similarity_heatmap(
            centroid_similarity,
            colorscale="Viridis",
            height=500,
            width=1000,
            get_friendly_layer_name=get_friendly_layer_name
        )
        
        return fig, {"display": "block"}
    
    except Exception as e:
        import traceback
        error_msg = f"Error creating centroid similarity graph: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            height=500
        ), {"display": "block"}

@app.callback(
    [Output("membership-overlap-graph", "figure"),
     Output("membership-overlap-section", "style")],
    [Input("cl-metrics-store", "data"),
     Input("cl-metric-controls", "value"),
     Input("cl-overlap-slider", "value")]
)
def update_membership_overlap_graph(cl_metrics, metric_controls, min_overlap):
    """Update membership overlap Sankey diagram."""
    if cl_metrics is None or "error" in cl_metrics:
        return go.Figure().update_layout(
            title="No membership overlap data available",
            height=500
        ), {"display": "none"}
    
    # Hide if not selected
    if "membership_overlap" not in metric_controls:
        return go.Figure(), {"display": "none"}
    
    try:
        # Check for membership overlap error
        if "membership_overlap_error" in cl_metrics:
            return go.Figure().update_layout(
                title=f"Error: {cl_metrics['membership_overlap_error']}",
                height=500
            ), {"display": "block"}
        
        # Get membership overlap data with string keys
        membership_overlap_str = cl_metrics.get("membership_overlap", {})
        
        # Convert string tuple keys back to actual tuples
        membership_overlap = {}
        for k_str, v in membership_overlap_str.items():
            # Convert string representation of tuple back to tuple
            if k_str.startswith('(') and k_str.endswith(')') and ',' in k_str:
                try:
                    # Safely evaluate the string as a tuple
                    k_tuple = eval(k_str)
                    # Ensure it's a proper tuple with two elements
                    if isinstance(k_tuple, tuple) and len(k_tuple) == 2:
                        # Convert values to numpy arrays if needed
                        if isinstance(v, list):
                            v_array = np.array(v)
                            membership_overlap[k_tuple] = v_array
                        else:
                            membership_overlap[k_tuple] = v
                except:
                    # Skip malformed keys
                    print(f"Warning: Skipping malformed key: {k_str}")
                    continue
        
        # If empty, display an error
        if not membership_overlap:
            return go.Figure().update_layout(
                title="No valid membership overlap data available",
                height=500
            ), {"display": "block"}
        
        # Create dummy layer_clusters dict for Sankey function
        layer_clusters = {}
        
        # Process all layer pairs to extract unique layers
        for (layer1, layer2), overlap_matrix in membership_overlap.items():
            # Extract cluster counts from matrix dimensions
            n_clusters1, n_clusters2 = overlap_matrix.shape
            
            # Add to layer_clusters if not already there
            if layer1 not in layer_clusters:
                layer_clusters[layer1] = {
                    "labels": np.arange(n_clusters1),
                    "k": n_clusters1
                }
            
            if layer2 not in layer_clusters:
                layer_clusters[layer2] = {
                    "labels": np.arange(n_clusters2),
                    "k": n_clusters2
                }
        
        # Create Sankey diagram
        fig = plot_membership_overlap_sankey(
            membership_overlap,
            layer_clusters,
            min_overlap=min_overlap,
            height=600,
            width=1000,
            get_friendly_layer_name=get_friendly_layer_name
        )
        
        return fig, {"display": "block"}
    
    except Exception as e:
        import traceback
        error_msg = f"Error creating membership overlap graph: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            height=500
        ), {"display": "block"}

@app.callback(
    [Output("trajectory-fragmentation-graph", "figure"),
     Output("trajectory-fragmentation-section", "style")],
    [Input("cl-metrics-store", "data"),
     Input("cl-metric-controls", "value")]
)
def update_trajectory_fragmentation_graph(cl_metrics, metric_controls):
    """Update trajectory fragmentation bar chart."""
    if cl_metrics is None or "error" in cl_metrics:
        return go.Figure().update_layout(
            title="No trajectory fragmentation data available",
            height=400
        ), {"display": "none"}
    
    # Hide if not selected
    if "trajectory_fragmentation" not in metric_controls:
        return go.Figure(), {"display": "none"}
    
    try:
        # Check for trajectory fragmentation error
        if "trajectory_fragmentation_error" in cl_metrics:
            return go.Figure().update_layout(
                title=f"Error: {cl_metrics['trajectory_fragmentation_error']}",
                height=400
            ), {"display": "block"}
        
        # Get trajectory fragmentation data
        trajectory_fragmentation = cl_metrics.get("trajectory_fragmentation", {})
        
        # If empty, display an error
        if not trajectory_fragmentation:
            return go.Figure().update_layout(
                title="No class labels available for trajectory fragmentation analysis",
                height=400
            ), {"display": "block"}
        
        # Create bar chart
        fig = plot_trajectory_fragmentation_bars(
            trajectory_fragmentation,
            height=400,
            width=800,
            get_friendly_layer_name=get_friendly_layer_name
        )
        
        return fig, {"display": "block"}
    
    except Exception as e:
        import traceback
        error_msg = f"Error creating trajectory fragmentation graph: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            height=400
        ), {"display": "block"}

@app.callback(
    [Output("path-density-graph", "figure"),
     Output("path-density-section", "style")],
    [Input("cl-metrics-store", "data"),
     Input("cl-metric-controls", "value")]
)
def update_path_density_graph(cl_metrics, metric_controls):
    """Update path density network graph."""
    if cl_metrics is None or "error" in cl_metrics:
        return go.Figure().update_layout(
            title="No path density data available",
            height=600
        ), {"display": "none"}
    
    # Hide if not selected
    if "path_density" not in metric_controls:
        return go.Figure(), {"display": "none"}
    
    try:
        # Check for path density error
        if "path_density_error" in cl_metrics:
            return go.Figure().update_layout(
                title=f"Error: {cl_metrics['path_density_error']}",
                height=600
            ), {"display": "block"}
        
        # Get path graph data
        path_graph_dict = cl_metrics.get("path_graph", {})
        
        # Check if we have valid graph data
        if not path_graph_dict or "nodes" not in path_graph_dict or not path_graph_dict["nodes"]:
            return go.Figure().update_layout(
                title="No path density data available",
                height=600
            ), {"display": "block"}
        
        # Build NetworkX graph from dictionary
        G = nx.Graph() if not path_graph_dict.get("directed", False) else nx.DiGraph()
        
        # Add nodes with attributes
        for node_data in path_graph_dict.get("nodes", []):
            if "id" in node_data:
                node_id = node_data["id"]
                attrs = node_data.get("attributes", {})
                G.add_node(node_id, **attrs)
        
        # Add edges with attributes
        for edge_data in path_graph_dict.get("edges", []):
            if "source" in edge_data and "target" in edge_data:
                source = edge_data["source"]
                target = edge_data["target"]
                attrs = edge_data.get("attributes", {})
                G.add_edge(source, target, **attrs)
        
        # Create network visualization
        fig = plot_path_density_network(
            G,
            layout="multipartite",
            height=600,
            width=1000,
            get_friendly_layer_name=get_friendly_layer_name
        )
        
        return fig, {"display": "block"}
    
    except Exception as e:
        import traceback
        error_msg = f"Error creating path density graph: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            height=600
        ), {"display": "block"}

# Register the callbacks for the Similarity Network tab
register_similarity_callbacks(app)

# Register the callbacks for the Path Fragmentation tab
register_path_fragmentation_callbacks(app)

# Register the callbacks for the Path Metrics tab
register_path_metrics_callbacks(app)

# Register the callbacks for the GPT-2 metrics dashboard tab
register_gpt2_metrics_callbacks(app)

# Register the callbacks for the LLM Integration tab
register_llm_callbacks(app)

if __name__ == "__main__":
    print("Starting Dash app...")
    print("Navigate to http://127.0.0.1:8050/ in your web browser.")
    app.run_server(debug=True)