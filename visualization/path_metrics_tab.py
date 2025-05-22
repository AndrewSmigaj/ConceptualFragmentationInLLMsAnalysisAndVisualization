"""
Path Metrics Tab for the Dash application.

This module provides a tab for displaying path-based metrics in the
neural network trajectory explorer dashboard.
"""

import os
import json
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback
from dash.dependencies import Input, Output, State

from visualization.path_metrics import (
    load_path_data,
    calculate_all_path_metrics,
    plot_path_metrics
)
from visualization.traj_plot import get_friendly_layer_name

def create_path_metrics_tab():
    """Create the path metrics tab for the dashboard."""
    return dcc.Tab(
        label="Path-Based Metrics",
        value="path-metrics-tab",
        children=[
            html.Div([
                html.Div([
                    html.Label("Dataset:"),
                    dcc.Dropdown(
                        id="pm-dataset-dropdown",
                        options=[
                            {"label": "Titanic", "value": "titanic"},
                            {"label": "Heart", "value": "heart"}
                        ],
                        value="heart"
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Label("Seed:"),
                    dcc.Dropdown(
                        id="pm-seed-dropdown",
                        options=[
                            {"label": f"Seed {s}", "value": s} for s in range(3)
                        ],
                        value=0
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Button(
                        "Update Metrics", 
                        id="pm-update-button", 
                        n_clicks=0,
                        style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px"}
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "20px"}),
            ]),
            
            html.Div([
                html.Div([
                    html.Label("Metric Selection:"),
                    dcc.Checklist(
                        id="pm-metric-selection",
                        options=[
                            {"label": "Entropy Fragmentation", "value": "entropy_fragmentation"},
                            {"label": "Path Coherence", "value": "path_coherence"},
                            {"label": "Cluster Stability", "value": "cluster_stability"},
                            {"label": "Membership Overlap", "value": "path_membership_overlap"},
                            {"label": "Conceptual Purity", "value": "conceptual_purity"}
                        ],
                        value=["entropy_fragmentation", "path_coherence", "conceptual_purity"],
                        style={"fontSize": "14px", "lineHeight": "25px"}
                    ),
                ], style={"width": "100%", "padding": "10px"}),
            ]),
            
            # Loading indicator for path metrics
            dcc.Loading(
                id="loading-path-metrics",
                type="circle",
                children=[
                    # Combined metrics figure
                    html.Div([
                        html.H3("Path-Based Metrics Overview", style={"textAlign": "center"}),
                        dcc.Graph(id="path-metrics-combined-graph", style={"height": "500px"})
                    ], id="path-metrics-combined-section"),
                    
                    # Individual metric figures
                    html.Div(id="path-metrics-individual-graphs"),
                ]
            ),
            
            # Path metrics data store
            dcc.Store(id="path-metrics-store")
        ]
    )

def register_path_metrics_callbacks(app):
    """Register callbacks for the path metrics tab."""
    
    @app.callback(
        Output("path-metrics-store", "data"),
        [Input("pm-update-button", "n_clicks")],
        [State("pm-dataset-dropdown", "value"),
         State("pm-seed-dropdown", "value")]
    )
    def update_path_metrics(n_clicks, dataset, seed):
        """Calculate path metrics and store the results."""
        if n_clicks == 0 or not dataset or seed is None:
            return None
        
        try:
            # Calculate all metrics
            metrics = calculate_all_path_metrics(dataset, seed)
            
            if not metrics:
                return {"error": "No metrics data available"}
            
            # Convert NumPy values to lists for JSON serialization
            metrics_json = {}
            for metric_name, metric_values in metrics.items():
                if metric_name == "layers":
                    metrics_json[metric_name] = metric_values
                elif isinstance(metric_values, dict):
                    metrics_json[metric_name] = {
                        layer: float(value) for layer, value in metric_values.items()
                    }
            
            return metrics_json
        
        except Exception as e:
            import traceback
            error_msg = f"Error calculating path metrics: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {"error": str(e)}
    
    @app.callback(
        [Output("path-metrics-combined-graph", "figure"),
         Output("path-metrics-individual-graphs", "children")],
        [Input("path-metrics-store", "data"),
         Input("pm-metric-selection", "value")]
    )
    def update_path_metrics_graphs(metrics_data, selected_metrics):
        """Update the path metrics visualizations."""
        if not metrics_data or "error" in metrics_data:
            error_msg = metrics_data.get("error", "No path metrics data available") if metrics_data else "No path metrics data available"
            empty_fig = go.Figure().update_layout(
                title=error_msg,
                height=500
            )
            return empty_fig, []
        
        try:
            # Filter metrics based on selection
            filtered_metrics = {}
            for metric_name, metric_values in metrics_data.items():
                if metric_name == "layers":
                    filtered_metrics[metric_name] = metric_values
                elif metric_name in selected_metrics:
                    filtered_metrics[metric_name] = metric_values
            
            # Generate plots
            figures = plot_path_metrics(filtered_metrics, get_friendly_layer_name, height=400, width=800)
            
            if not figures:
                empty_fig = go.Figure().update_layout(
                    title="No metrics data to display",
                    height=500
                )
                return empty_fig, []
            
            # Get combined figure
            combined_fig = figures.pop("combined", None)
            if not combined_fig:
                combined_fig = go.Figure().update_layout(
                    title="No combined metrics data available",
                    height=500
                )
            
            # Create individual metric sections
            individual_graphs = []
            for metric_name, fig in figures.items():
                metric_title = metric_name.replace('_', ' ').title()
                individual_graphs.append(
                    html.Div([
                        html.H4(metric_title, style={"textAlign": "center"}),
                        dcc.Graph(
                            id=f"path-metrics-{metric_name}-graph",
                            figure=fig,
                            style={"height": "400px"}
                        )
                    ], style={"marginTop": "20px", "marginBottom": "20px"})
                )
            
            return combined_fig, individual_graphs
        
        except Exception as e:
            import traceback
            error_msg = f"Error updating path metrics graphs: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            empty_fig = go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=500
            )
            return empty_fig, []