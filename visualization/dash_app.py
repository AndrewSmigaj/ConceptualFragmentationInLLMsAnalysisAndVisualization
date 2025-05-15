"""
Dash web application for interactive exploration of neural network trajectories.

This module provides a web-based interface for exploring the 3D
visualizations of neural network layer trajectories.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.graph_objects as go

try:
    import dash
    from dash import dcc, html
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
    load_layer_class_metrics, load_activations  # Add this import
)
from visualization.reducers import Embedder, embed_layer_activations
from visualization.traj_plot import build_single_scene_figure, plot_dataset_trajectories, save_figure, LAYER_SEPARATION_OFFSET

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
                    {"label": "Normalize Embeddings", "value": "normalize"}
                ],
                value=["baseline", "regularized", "arrows", "normalize"]
            ),
            html.Br(), # Added for spacing
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
    
    html.Div(id="status-message", style={"padding": "10px", "color": "gray"}),
    
    # Hidden data storage
    dcc.Store(id="embeddings-store")
])

def load_and_embed_data(dataset, seed):
    """Load and embed data for the given dataset and seed."""
    try:
        # Get configurations
        best_config = get_best_config(dataset)
        baseline_config = get_baseline_config(dataset)
        
        # Create embedder
        embedder = Embedder(n_components=3, random_state=42, cache_dir=CACHE_DIR)
        
        # Embed baseline
        print(f"Embedding baseline data for {dataset}, seed {seed}")
        baseline_embeddings = embed_layer_activations(
            dataset, baseline_config, seed, embedder=embedder, use_test_set=True)
        
        # Embed best config
        print(f"Embedding regularized data for {dataset}, seed {seed}")
        best_embeddings = embed_layer_activations(
            dataset, best_config, seed, embedder=embedder, use_test_set=True)
        
        # Create dictionary for storage
        embeddings_dict = {
            "baseline": {seed: baseline_embeddings},
            "regularized": {seed: best_embeddings}
        }
        
        return embeddings_dict, None
    
    except Exception as e:
        import traceback
        error_msg = f"Error loading embeddings: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

@app.callback(
    [Output("embeddings-store", "data"),
     Output("status-message", "children")],
    [Input("dataset-dropdown", "value"),
     Input("seed-dropdown", "value")]
)
def update_embeddings(dataset, seed):
    """Update embeddings when dataset or seed changes."""
    print(f"DEBUG: dash_app.py - update_embeddings callback triggered for dataset: {dataset}, seed: {seed}")
    status = f"Loading embeddings for {dataset.title()}, seed {seed}..."
    
    print(f"DEBUG: dash_app.py - update_embeddings: Calling load_and_embed_data for {dataset}, {seed}")
    embeddings_dict, error = load_and_embed_data(dataset, seed)
    print(f"DEBUG: dash_app.py - update_embeddings: Returned from load_and_embed_data for {dataset}, {seed}")

    if error:
        print(f"DEBUG: dash_app.py - update_embeddings: Error occurred: {error}")
        return None, f"Error: {error}"
    
    serializable_embeddings_dict = None
    if embeddings_dict is not None:
        print(f"DEBUG: dash_app.py - update_embeddings: embeddings_dict is populated. Type: {type(embeddings_dict)}")
        print("DEBUG: dash_app.py - update_embeddings: Converting np.ndarray to lists for dcc.Store...")
        serializable_embeddings_dict = numpy_to_list_recursive(embeddings_dict)
        print("DEBUG: dash_app.py - update_embeddings: Conversion finished.")
        # Basic check for numpy arrays (should be none now)
        try:
            # Example path, adjust if your structure is different or for a more thorough check
            sample_data_path = serializable_embeddings_dict.get("baseline", {}).get(seed, {}).get("layer1")
            if isinstance(sample_data_path, np.ndarray):
                print("DEBUG: dash_app.py - update_embeddings: WARNING! Found np.ndarray AFTER serialization attempt.")
            elif isinstance(sample_data_path, list):
                print("DEBUG: dash_app.py - update_embeddings: Verified sample data is now a list.")
        except Exception as e:
            print(f"DEBUG: dash_app.py - update_embeddings: Error during post-serialization check: {e}")
            pass
    else:
        print("DEBUG: dash_app.py - update_embeddings: embeddings_dict is None after load_and_embed_data.")

    print(f"DEBUG: dash_app.py - update_embeddings: About to return data to dcc.Store. Type of serializable_embeddings_dict: {type(serializable_embeddings_dict)}")
    return serializable_embeddings_dict, f"Loaded embeddings for {dataset.title()}, seed {seed}"

@app.callback(
    Output("trajectory-graph", "figure"),
    [Input("update-button", "n_clicks")],
    [State("embeddings-store", "data"),
     State("dataset-dropdown", "value"),
     State("samples-slider", "value"),
     State("highlight-slider", "value"),
     State("visualization-controls", "value"),
     State("highlight-controls", "value"),
     State("trajectory-point-metric-dropdown", "value"),
     State("seed-dropdown", "value")]  # Add seed to the state parameters
)
def update_visualization(n_clicks, stored_data, dataset, n_samples, n_highlights, 
                        vis_controls, highlight_controls, selected_point_metric, seed):
    print(f"DEBUG: dash_app.py - update_visualization callback triggered. n_clicks: {n_clicks}")
    print(f"DEBUG: dash_app.py - selected dataset: {dataset}, n_samples: {n_samples}, n_highlights: {n_highlights}")
    print(f"DEBUG: dash_app.py - vis_controls: {vis_controls}, highlight_controls: {highlight_controls}")
    print(f"DEBUG: dash_app.py - selected_point_metric for trajectory points: {selected_point_metric}")

    if not stored_data:
        print("DEBUG: dash_app.py - update_visualization: No stored_data (embeddings) found. Returning empty figure.")
        return go.Figure(layout={"title": "No embeddings data loaded. Please select a dataset and seed."})

    embeddings_dict = list_to_numpy_recursive(stored_data)
    
    if not embeddings_dict:
        print("DEBUG: dash_app.py - update_visualization: embeddings_dict is empty after list_to_numpy_recursive. Returning empty figure.")
        return go.Figure(layout={"title": "Embeddings data is empty after processing."})

    # Load metadata for class info
    print(f"DEBUG: dash_app.py - update_visualization: Loading metadata for dataset: {dataset}")
    metadata = load_dataset_metadata(dataset)
    
    # Load activations to get per-sample labels
    try:
        config = get_baseline_config(dataset)
        activations = load_activations(dataset, config, seed)
        
        # Extract per-sample class labels
        if 'labels' in activations and isinstance(activations['labels'], dict) and 'test' in activations['labels']:
            # If we have a test/train split, use the test set labels (last epoch)
            labels_array = activations['labels']['test']
            if labels_array.ndim > 1:  # Multiple epochs
                sample_class_labels = labels_array[-1]  # Use the last epoch
            else:
                sample_class_labels = labels_array
            print(f"DEBUG: Got per-sample class labels with shape {sample_class_labels.shape}")
        else:
            # Fallback to unique class labels
            print("DEBUG: Could not find per-sample labels, using unique classes from metadata")
            sample_class_labels = metadata.get("class_labels")
    except Exception as e:
        print(f"WARNING: Error getting per-sample labels: {e}")
        sample_class_labels = metadata.get("class_labels")
        
    # Load per-class, per-layer metrics for both baseline and regularized models
    layer_class_metrics_data_baseline = None
    layer_class_metrics_data_regularized = None
    
    try:
        if "baseline" in vis_controls:
            baseline_config = get_baseline_config(dataset)
            layer_class_metrics_data_baseline = load_layer_class_metrics(dataset, baseline_config, seed)
            print("DEBUG: Loaded baseline metrics successfully")
            
        if "regularized" in vis_controls:
            regularized_config = get_best_config(dataset)
            layer_class_metrics_data_regularized = load_layer_class_metrics(dataset, regularized_config, seed)
            print("DEBUG: Loaded regularized metrics successfully")
    except Exception as e:
        print(f"WARNING: Error loading layer class metrics: {e}")
        # Continue with None values for metrics

    # Compute highlight indices using the new fractured-off detection
    highlight_indices = None
    if "high_frag" in highlight_controls:
        try:
            # Use the regularized model's data for highlighting if available, else baseline
            config_for_highlighting = get_best_config(dataset) if "regularized" in vis_controls else get_baseline_config(dataset)
            highlight_indices = select_samples(
                dataset=dataset,
                config=config_for_highlighting,
                seed=seed,
                k_frag=n_highlights,
                k_norm=0  # We're only interested in fractured samples
            )
            print(f"DEBUG: Selected {len(highlight_indices) if highlight_indices else 0} samples for highlighting")
        except Exception as e:
            print(f"WARNING: Error computing highlight indices: {e}")
            highlight_indices = []

    # Create sample color values from metrics if available
    sample_color_values = None
    if selected_point_metric and "baseline" in vis_controls and layer_class_metrics_data_baseline:
        # Extract per-class metric values from the first layer
        first_layer = next(iter(layer_class_metrics_data_baseline))
        metrics_by_class = layer_class_metrics_data_baseline[first_layer]
        
        # Get per-sample fracture scores based on selected metric
        if hasattr(sample_class_labels, '__len__') and len(sample_class_labels) > 2:
            values = []
            for sample_idx in range(len(sample_class_labels)):
                class_str = str(sample_class_labels[sample_idx])
                if class_str in metrics_by_class and selected_point_metric in metrics_by_class[class_str]:
                    values.append(metrics_by_class[class_str][selected_point_metric])
                else:
                    values.append(np.nan)
            sample_color_values = np.array(values)

    # Filter embeddings_dict to only include configs that are in vis_controls
    filtered_embeddings_dict = {
        config_name: config_data 
        for config_name, config_data in embeddings_dict.items()
        if config_name in vis_controls
    }

    print(f"DEBUG: dash_app.py - update_visualization: About to call build_single_scene_figure")
    print(f"  dataset: {dataset}")
    print(f"  n_samples: {n_samples}")
    print(f"  highlight_indices count: {len(highlight_indices) if highlight_indices else 0}")
    print(f"  Using sample_color_values: {sample_color_values is not None}")
    print(f"  Using per-sample class_labels: {hasattr(sample_class_labels, '__len__') and len(sample_class_labels) > 2}")
    print(f"  Using layer_separation: {LAYER_SEPARATION_OFFSET}")
    
    # Use build_single_scene_figure instead of plot_dataset_trajectories
    fig = build_single_scene_figure(
        embeddings_dict=filtered_embeddings_dict,
        samples_to_plot=np.arange(n_samples),
        max_samples=n_samples,
        highlight_indices=highlight_indices,
        class_labels=sample_class_labels,
        sample_color_values=sample_color_values,
        color_value_name=selected_point_metric if selected_point_metric else "Score",
        show_arrows="arrows" in vis_controls,
        normalize="normalize" in vis_controls,
        layer_separation=LAYER_SEPARATION_OFFSET
    )
    
    return fig

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

if __name__ == "__main__":
    print("Starting Dash app...")
    print("Navigate to http://127.0.0.1:8050/ in your web browser.")
    app.run_server(debug=True) 