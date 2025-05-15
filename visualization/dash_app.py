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
    get_config_id
)
from visualization.reducers import Embedder, embed_layer_activations
from visualization.traj_plot import build_single_scene_figure, plot_dataset_trajectories, save_figure, LAYER_SEPARATION_OFFSET
from visualization.cluster_utils import compute_layer_clusters_embedded, get_embedded_clusters_cache_path
from concept_fragmentation.config import DATASETS as DATASET_CFG

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
    "subspace_angle": "Subspace Angle"
    # Add more metrics here as they become available, e.g. "custom_fracture_score"
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
        
        # Instead of direct CSV loading, use the data loaders
        from concept_fragmentation.data.loaders import get_dataset_loader
        
        # Get the appropriate dataset loader
        loader = get_dataset_loader(dataset)
        
        # Load data using the loader
        train_df, test_df = loader.load_data()
        
        # Include both numeric features and the 'sex' column
        features_to_include = numeric_features.copy()
        if 'Sex' in train_df.columns:
            features_to_include.append('Sex')
        elif 'sex' in train_df.columns:
            features_to_include.append('sex')
        
        # Filter to selected features
        df_selected = train_df[features_to_include]
        
        # Convert to JSON-serializable format
        dataset_json = df_selected.to_json(orient="split")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dataset_json = None
    
    return embeddings_json, clusters_json, true_labels_json, dataset_json, f"Loaded {dataset} data for seed {seed}"

@app.callback(
    Output("trajectory-graph", "figure"),
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
        return go.Figure().update_layout(
            title="Click 'Update Visualization' after selecting options",
            height=700
        )
    
    # Debug: Print received data types
    print("DEBUG: Starting update_visualization function")
    
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
            color_by=color_by
        )
        return fig
    except Exception as e:
        import traceback
        print(f"DEBUG: Error building figure: {e}")
        print(f"DEBUG: {traceback.format_exc()}")
        return go.Figure().update_layout(
            title=f"Error: {str(e)}",
            height=700
        )

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
     State("dataset-dropdown", "value")]
)
def show_cluster_stats(click_data, dataset_json, clusters_json, dataset):
    """Show descriptive statistics for the clicked cluster."""
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
        
        # Load dataset features
        df = pd.read_json(dataset_json, orient="split")
        print("DEBUG: DataFrame columns after loading:", df.columns)

        # Convert 'sex' column to 0 and 1 if needed
        if 'sex' in df.columns:
            df['sex'] = df['sex'].apply(lambda x: 1 if x.lower() in ['male', 'm'] else 0)
            df['sex'] = df['sex'].astype(int)  # Ensure 'sex' is treated as a numeric column
            print("DEBUG: 'sex' column after conversion:", df['sex'].head())

        # Select samples in this cluster
        if len(sample_indices) > df.shape[0]:
            sample_indices = sample_indices[:df.shape[0]]
        
        cluster_samples = df.iloc[sample_indices]
        
        # Calculate statistics
        print("DEBUG: DataFrame dtypes before describe:", df.dtypes)  # Debug statement to check dtypes
        stats = cluster_samples.describe().round(2)
        print("DEBUG: Statistics DataFrame columns:", stats.columns)
        print("DEBUG: Statistics DataFrame head:", stats.head())
        
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
        
        # Create summary
        summary_div = html.Div([
            html.H4(f"Cluster {cluster_id} Statistics ({layer}, {config_name})"),
            html.P(f"Number of samples: {len(sample_indices)}"),
            table
        ])
        
        return summary_div
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating cluster statistics: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Starting Dash app...")
    print("Navigate to http://127.0.0.1:8050/ in your web browser.")
    app.run_server(debug=True) 