"""
GPT-2 Token Path Visualization Tab for the Neural Network Trajectories Dashboard.

This module provides the layout and callbacks for the GPT-2 token path visualization tab,
which enables interactive exploration of token paths through 3-layer windows.
"""

import dash
from dash import dcc, html, dash_table, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import GPT-2 token Sankey diagram visualization
from visualization.gpt2_token_sankey import (
    extract_token_paths,
    generate_token_sankey_diagram,
    get_token_path_stats,
    create_token_path_comparison,
    create_3layer_window_sankey
)

# Import GPT-2 attention Sankey diagram visualization
from visualization.gpt2_attention_sankey import (
    extract_attention_flow,
    generate_attention_sankey_diagram,
    create_attention_token_comparison,
    create_token_attention_window_visualization
)

# Import GPT-2 token movement metrics visualization
from visualization.gpt2_token_movement import (
    calculate_token_movement_metrics,
    create_token_trajectory_plot,
    create_token_velocity_heatmap,
    create_mobility_ranking_plot,
    create_token_movement_visualization
)

# Import GPT-2 attention-to-token path correlation visualization
from visualization.gpt2_attention_correlation import (
    calculate_correlation_metrics,
    create_correlation_heatmap,
    create_token_correlation_scatter,
    create_attention_path_correlation_visualization
)

# Constants
GPT2_RESULTS_DIR = os.path.join(parent_dir, "results", "gpt2")
DEFAULT_HEIGHT = 600
DEFAULT_WIDTH = 1000


def find_gpt2_analysis_results() -> List[Dict[str, Any]]:
    """
    Find and list available GPT-2 analysis results.
    
    Returns:
        List of dictionaries with available GPT-2 analysis metadata
    """
    available_results = []
    
    # Check if results directory exists
    if not os.path.exists(GPT2_RESULTS_DIR):
        return []
    
    # Find all metadata files
    for root, dirs, files in os.walk(GPT2_RESULTS_DIR):
        for file in files:
            if file.endswith("_metadata.json"):
                try:
                    # Load metadata
                    metadata_path = os.path.join(root, file)
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Extract relevant information
                    result_info = {
                        "model_type": metadata.get("model_type", "unknown"),
                        "metadata_file": metadata_path,
                        "windows": list(metadata.get("layer_files", {}).keys()),
                        "config": metadata.get("config", {}),
                        "display_name": f"{metadata.get('model_type', 'GPT-2')} - {Path(metadata_path).parent.name}"
                    }
                    
                    available_results.append(result_info)
                except Exception as e:
                    print(f"Error loading GPT-2 metadata from {file}: {e}")
    
    return available_results


def load_window_data(metadata_file: str, window_name: str) -> Dict[str, Any]:
    """
    Load window data from GPT-2 analysis results.
    
    Args:
        metadata_file: Path to metadata file
        window_name: Name of the window to load
        
    Returns:
        Dictionary with window data
    """
    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Check if window exists
    if window_name not in metadata.get("layer_files", {}):
        raise ValueError(f"Window {window_name} not found in metadata")
    
    # Get window metadata
    metadata_dir = os.path.dirname(metadata_file)
    window_metadata_file = os.path.join(metadata_dir, window_name, f"{window_name}_metadata.json")
    
    with open(window_metadata_file, 'r') as f:
        window_metadata = json.load(f)
    
    # Load activations
    window_activations = {}
    for layer_name, layer_file in metadata["layer_files"][window_name].items():
        try:
            window_activations[layer_name] = np.load(layer_file)
        except Exception as e:
            print(f"Error loading activation file {layer_file}: {e}")
    
    # Return window data
    return {
        "activations": window_activations,
        "metadata": window_metadata.get("metadata", {}),
        "window_layers": window_metadata.get("window_layers", [])
    }


def load_apa_results(metadata_file: str, window_name: str) -> Dict[str, Any]:
    """
    Load APA analysis results for a GPT-2 window.
    
    Args:
        metadata_file: Path to metadata file
        window_name: Name of the window to load
        
    Returns:
        Dictionary with APA analysis results
    """
    # Get paths
    metadata_dir = os.path.dirname(metadata_file)
    results_dir = os.path.join(metadata_dir, "results", window_name)
    
    # Check if results directory exists
    if not os.path.exists(results_dir):
        return {"clusters": {}}
    
    # Load window data to get layer names
    window_data = load_window_data(metadata_file, window_name)
    
    # Load cluster labels for each layer
    analysis_result = {"clusters": {}}
    for layer_name in window_data["activations"].keys():
        labels_file = os.path.join(results_dir, f"{layer_name}_labels.npy")
        if os.path.exists(labels_file):
            analysis_result["clusters"][layer_name] = {
                "labels": np.load(labels_file)
            }
    
    return analysis_result


def create_gpt2_token_tab():
    """
    Create the layout for the GPT-2 Token Path tab.
    
    Returns:
        Dash Tab component
    """
    # Find available GPT-2 analysis results
    gpt2_results = find_gpt2_analysis_results()
    
    # Create options for dropdown
    result_options = []
    window_options = []
    
    if gpt2_results:
        # Create options for results dropdown
        result_options = [
            {"label": result["display_name"], "value": result["metadata_file"]}
            for result in gpt2_results
        ]
        
        # Create initial window options
        if gpt2_results[0]["windows"]:
            window_options = [
                {"label": window_name, "value": window_name}
                for window_name in gpt2_results[0]["windows"]
            ]
    
    return dcc.Tab(label="GPT-2 Token Paths", children=[
        html.Div([
            html.H4("GPT-2 Token Path Analysis", style={"textAlign": "center"}),
            
            # Results selection
            html.Div([
                html.Div([
                    html.Label("GPT-2 Analysis:"),
                    dcc.Dropdown(
                        id="gpt2-result-dropdown",
                        options=result_options,
                        value=result_options[0]["value"] if result_options else None,
                        clearable=False
                    )
                ], style={"width": "60%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Label("Window:"),
                    dcc.Dropdown(
                        id="gpt2-window-dropdown",
                        options=window_options,
                        value=window_options[0]["value"] if window_options else None,
                        clearable=False
                    )
                ], style={"width": "40%", "display": "inline-block", "padding": "10px"})
            ], style={"display": "flex", "flexWrap": "wrap"}),
            
            # Token filtering options
            html.Div([
                html.Div([
                    html.Label("Token Filter:"),
                    dcc.Input(
                        id="gpt2-token-filter",
                        type="text",
                        placeholder="Enter tokens to filter (comma-separated)",
                        style={"width": "100%"}
                    )
                ], style={"width": "60%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Label("Min Path Count:"),
                    dcc.Input(
                        id="gpt2-min-path-count",
                        type="number",
                        min=1,
                        max=100,
                        value=2,
                        style={"width": "100%"}
                    )
                ], style={"width": "40%", "display": "inline-block", "padding": "10px"})
            ], style={"display": "flex", "flexWrap": "wrap"}),
            
            # Visualizations
            html.Div([
                # Visualization type selector
                html.Div([
                    html.Label("Visualization Type:"),
                    dcc.RadioItems(
                        id="gpt2-viz-type",
                        options=[
                            {"label": "Token Path Flow", "value": "token_path"},
                            {"label": "Attention Flow", "value": "attention_flow"},
                            {"label": "Token Movement", "value": "token_movement"},
                            {"label": "Path-Attention Correlation", "value": "path_attention_correlation"}
                        ],
                        value="token_path",
                        inline=True
                    )
                ], style={"width": "100%", "padding": "10px"}),
                
                # Movement metrics selector (initially hidden)
                html.Div([
                    html.Div([
                        html.Label("Movement Metric:"),
                        dcc.RadioItems(
                            id="gpt2-movement-metric",
                            options=[
                                {"label": "Token Trajectories", "value": "trajectory"},
                                {"label": "Activation Velocity", "value": "velocity"},
                                {"label": "Mobility Ranking", "value": "mobility"}
                            ],
                            value="trajectory",
                            inline=True
                        )
                    ], style={"width": "100%", "padding": "10px"}),
                ], id="gpt2-movement-settings", style={"width": "100%", "display": "none"}),
                
                # Correlation metrics selector (initially hidden)
                html.Div([
                    html.Div([
                        html.Label("Correlation View:"),
                        dcc.RadioItems(
                            id="gpt2-correlation-view",
                            options=[
                                {"label": "Layer Correlation Heatmap", "value": "heatmap"},
                                {"label": "Token Correlation Scatter", "value": "scatter"}
                            ],
                            value="heatmap",
                            inline=True
                        )
                    ], style={"width": "100%", "padding": "10px"}),
                ], id="gpt2-correlation-settings", style={"width": "100%", "display": "none"}),
                
                # Sankey diagram
                html.Div([
                    html.H5(id="gpt2-sankey-title", style={"textAlign": "center"}),
                    dcc.Loading(
                        id="gpt2-sankey-loading",
                        type="circle",
                        children=dcc.Graph(
                            id="gpt2-sankey-diagram",
                            figure=go.Figure().update_layout(title="Select a GPT-2 analysis result"),
                            style={"height": f"{DEFAULT_HEIGHT}px"}
                        )
                    )
                ], style={"width": "100%", "padding": "10px"}),
                
                # Attention settings (initially hidden)
                html.Div([
                    html.Div([
                        html.Label("Min Attention Threshold:"),
                        dcc.Slider(
                            id="gpt2-attention-threshold",
                            min=0.01,
                            max=0.5,
                            step=0.01,
                            value=0.05,
                            marks={i/100: str(i/100) for i in range(5, 55, 10)}
                        )
                    ], style={"width": "50%", "display": "inline-block", "padding": "10px"}),
                    
                    html.Div([
                        html.Label("Max Attention Edges:"),
                        dcc.Slider(
                            id="gpt2-max-edges",
                            min=50,
                            max=500,
                            step=50,
                            value=250,
                            marks={i: str(i) for i in range(50, 550, 100)}
                        )
                    ], style={"width": "50%", "display": "inline-block", "padding": "10px"})
                ], id="gpt2-attention-settings", style={"width": "100%", "display": "none"}),
                
                
                # Token statistics and path comparison
                html.Div([
                    html.Div([
                        html.H5("Token Statistics", style={"textAlign": "center"}),
                        dcc.Loading(
                            id="gpt2-stats-loading",
                            type="circle",
                            children=html.Div(id="gpt2-token-stats")
                        )
                    ], style={"width": "40%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
                    
                    html.Div([
                        html.H5(id="gpt2-comparison-title", style={"textAlign": "center"}),
                        dcc.Loading(
                            id="gpt2-comparison-loading",
                            type="circle",
                            children=dcc.Graph(
                                id="gpt2-path-comparison",
                                figure=go.Figure().update_layout(title="Select tokens to compare"),
                                style={"height": f"{DEFAULT_HEIGHT - 100}px"}
                            )
                        )
                    ], style={"width": "60%", "display": "inline-block", "padding": "10px"})
                ], style={"display": "flex", "flexWrap": "wrap"}),
                
                # Most fragmented tokens
                html.Div([
                    html.H5("Most Fragmented Tokens", style={"textAlign": "center"}),
                    dcc.Loading(
                        id="gpt2-fragmented-loading",
                        type="circle",
                        children=html.Div(id="gpt2-fragmented-tokens")
                    )
                ], style={"width": "100%", "padding": "10px"})
            ], style={"padding": "10px"})
        ], style={"margin": "10px", "padding": "10px"})
    ])


def register_gpt2_token_callbacks(app):
    """
    Register callbacks for the GPT-2 Token Path tab.
    
    Args:
        app: Dash application instance
    """
    # Update window dropdown when result is selected
    @app.callback(
        Output("gpt2-window-dropdown", "options"),
        Output("gpt2-window-dropdown", "value"),
        Input("gpt2-result-dropdown", "value")
    )
    def update_window_dropdown(metadata_file):
        if not metadata_file:
            return [], None
        
        try:
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get available windows
            windows = list(metadata.get("layer_files", {}).keys())
            
            # Create options
            options = [{"label": window, "value": window} for window in windows]
            
            # Set default value
            value = windows[0] if windows else None
            
            return options, value
        except Exception as e:
            print(f"Error loading metadata from {metadata_file}: {e}")
            return [], None
    
    # Update visualizations when inputs change
    # Update settings visibility based on visualization type
    @app.callback(
        Output("gpt2-attention-settings", "style"),
        Output("gpt2-movement-settings", "style"),
        Output("gpt2-correlation-settings", "style"),
        Output("gpt2-sankey-title", "children"),
        Output("gpt2-comparison-title", "children"),
        Input("gpt2-viz-type", "value")
    )
    def update_viz_type(viz_type):
        # Default - all settings hidden
        attention_style = {"width": "100%", "display": "none"}
        movement_style = {"width": "100%", "display": "none"}
        correlation_style = {"width": "100%", "display": "none"}
        sankey_title = "Token Path Flow"
        comparison_title = "Token Path Comparison"
        
        # Show settings based on visualization type
        if viz_type == "attention_flow":
            attention_style = {"width": "100%", "display": "block"}
            sankey_title = "Attention Flow Between Tokens"
            comparison_title = "Token vs Attention Comparison"
        elif viz_type == "token_movement":
            movement_style = {"width": "100%", "display": "block"}
            sankey_title = "Token Movement Visualization"
            comparison_title = "Token Movement Metrics"
        elif viz_type == "path_attention_correlation":
            correlation_style = {"width": "100%", "display": "block"}
            sankey_title = "Path-Attention Correlation"
            comparison_title = "Token Correlation Analysis"
        else:  # Default to token_path
            sankey_title = "Token Path Flow"
            comparison_title = "Token Path Comparison"
        
        return attention_style, movement_style, correlation_style, sankey_title, comparison_title
    
    @app.callback(
        Output("gpt2-sankey-diagram", "figure"),
        Output("gpt2-path-comparison", "figure"),
        Output("gpt2-token-stats", "children"),
        Output("gpt2-fragmented-tokens", "children"),
        Input("gpt2-result-dropdown", "value"),
        Input("gpt2-window-dropdown", "value"),
        Input("gpt2-token-filter", "value"),
        Input("gpt2-min-path-count", "value"),
        Input("gpt2-viz-type", "value"),
        Input("gpt2-attention-threshold", "value"),
        Input("gpt2-max-edges", "value"),
        Input("gpt2-movement-metric", "value"),
        Input("gpt2-correlation-view", "value")
    )
    def update_visualizations(metadata_file, window_name, token_filter, min_path_count, viz_type, attention_threshold, max_edges, movement_metric, correlation_view):
        if not metadata_file or not window_name:
            empty_fig = go.Figure().update_layout(title="Select a GPT-2 analysis result")
            return empty_fig, empty_fig, "No data selected", "No data selected"
        
        try:
            # Load window data
            window_data = load_window_data(metadata_file, window_name)
            
            # Load APA results
            apa_results = load_apa_results(metadata_file, window_name)
            
            # Parse token filter
            highlight_tokens = []
            if token_filter:
                highlight_tokens = [t.strip() for t in token_filter.split(",") if t.strip()]
            
            # Set min path count
            if not min_path_count or min_path_count < 1:
                min_path_count = 1
            
            # Create visualizations based on selected type
            if viz_type == "attention_flow":
                viz_results = create_token_attention_window_visualization(
                    window_data,
                    apa_results,
                    highlight_tokens=highlight_tokens,
                    min_attention=attention_threshold,
                    max_edges=max_edges,
                    save_html=False  # Don't save to files in the dashboard
                )
            elif viz_type == "token_movement":
                # Create token movement visualization
                movement_viz = create_token_movement_visualization(
                    window_data,
                    apa_results,
                    highlight_tokens=highlight_tokens,
                    top_n=20,  # Show top 20 tokens
                    save_html=False  # Don't save to files in the dashboard
                )
                
                # Select appropriate visualization based on metric
                if movement_metric == "trajectory":
                    main_fig = movement_viz["trajectory_plot"]
                    comparison_fig = movement_viz["mobility_ranking"]
                elif movement_metric == "velocity":
                    main_fig = movement_viz["velocity_heatmap"]
                    comparison_fig = movement_viz["trajectory_plot"]
                else:  # Default to mobility
                    main_fig = movement_viz["mobility_ranking"]
                    comparison_fig = movement_viz["velocity_heatmap"]
                
                # Create custom results structure to match expected format
                viz_results = {
                    "sankey": main_fig,
                    "comparison": comparison_fig,
                    "stats": {
                        "total_tokens": len(movement_viz["metrics"]["token_metrics"]),
                        "avg_mobility": movement_viz["metrics"]["global_metrics"].get("avg_path_length", 0.0)
                    }
                }
            elif viz_type == "path_attention_correlation":
                # Create correlation visualization
                correlation_viz = create_attention_path_correlation_visualization(
                    window_data,
                    apa_results,
                    highlight_tokens=highlight_tokens,
                    save_html=False  # Don't save to files in the dashboard
                )
                
                # Select appropriate visualization based on view
                if correlation_view == "heatmap":
                    main_fig = correlation_viz["correlation_heatmap"]
                    comparison_fig = correlation_viz["token_scatter"]
                else:  # Default to scatter
                    main_fig = correlation_viz["token_scatter"]
                    comparison_fig = correlation_viz["correlation_heatmap"]
                
                # Create custom results structure to match expected format
                viz_results = {
                    "sankey": main_fig,
                    "comparison": comparison_fig,
                    "stats": {
                        "total_tokens": len(correlation_viz["metrics"]["token_correlations"]) if "token_correlations" in correlation_viz["metrics"] else 0,
                        "global_correlation": correlation_viz["metrics"]["global_metrics"].get("avg_path_attention_correlation", 0.0) if "global_metrics" in correlation_viz["metrics"] else 0.0
                    }
                }
            else:  # Default to token_path
                viz_results = create_3layer_window_sankey(
                    window_data,
                    apa_results,
                    highlight_tokens=highlight_tokens,
                    min_path_count=min_path_count,
                    save_html=False  # Don't save to files in the dashboard
                )
            
            # Extract results
            sankey_fig = viz_results["sankey"]
            comparison_fig = viz_results["comparison"]
            stats = viz_results["stats"]
            
            # Create token stats display
            token_stats_div = html.Div([
                html.P(f"Total Tokens: {stats['total_tokens']}"),
                html.P(f"Unique Tokens: {stats['unique_tokens']}"),
                html.P(f"Total Paths: {stats['paths']['total']}"),
                html.P(f"Unique Paths: {stats['paths']['unique']}"),
                html.P(f"Highlighted Tokens: {len(highlight_tokens)}"),
                html.P(f"Fragmentation Range: {min_frag:.2f} - {max_frag:.2f}")
            ])
            
            # Create most fragmented tokens display
            if stats["most_fragmented_tokens"]:
                # Create table
                fragmented_table = dash_table.DataTable(
                    id="gpt2-fragmented-table",
                    columns=[
                        {"name": "Token", "id": "token"},
                        {"name": "Occurrences", "id": "occurrences"},
                        {"name": "Unique Paths", "id": "unique_paths"},
                        {"name": "Fragmentation", "id": "fragmentation_score"}
                    ],
                    data=[
                        {
                            "token": t["token"],
                            "occurrences": t["occurrences"],
                            "unique_paths": t["unique_paths"],
                            "fragmentation_score": f"{t['fragmentation_score']:.3f}"
                        }
                        for t in stats["most_fragmented_tokens"]
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center"},
                    style_header={"fontWeight": "bold"},
                    page_size=10
                )
                
                fragmented_tokens_div = html.Div([
                    html.P("Tokens with most diverse paths through clusters:"),
                    fragmented_table,
                    html.P("Click on a token to add it to the filter above for visualization.", style={"fontStyle": "italic"})
                ])
            else:
                fragmented_tokens_div = html.P("No fragmented tokens found")
            
            return sankey_fig, comparison_fig, token_stats_div, fragmented_tokens_div
        except Exception as e:
            import traceback
            traceback.print_exc()
            empty_fig = go.Figure().update_layout(title=f"Error: {str(e)}")
            return empty_fig, empty_fig, f"Error: {str(e)}", f"Error: {str(e)}"
    
    # Update token filter when clicking on a token in the fragmented tokens table
    @app.callback(
        Output("gpt2-token-filter", "value"),
        Input("gpt2-fragmented-table", "active_cell"),
        State("gpt2-fragmented-table", "data"),
        State("gpt2-token-filter", "value")
    )
    def update_token_filter(active_cell, table_data, current_filter):
        if active_cell is None or table_data is None:
            return current_filter
        
        # Get selected token
        row = active_cell["row"]
        if row < len(table_data):
            selected_token = table_data[row]["token"]
            
            # Add to current filter
            if current_filter:
                # Check if token is already in filter
                tokens = [t.strip() for t in current_filter.split(",")]
                if selected_token not in tokens:
                    return f"{current_filter}, {selected_token}"
                return current_filter
            else:
                return selected_token
        
        return current_filter