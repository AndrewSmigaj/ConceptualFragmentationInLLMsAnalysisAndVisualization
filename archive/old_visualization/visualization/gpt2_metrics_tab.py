"""
GPT-2 Metrics Dashboard Tab.

This module provides a unified dashboard tab for GPT-2 metrics visualization,
integrating various analysis types including attention patterns, token paths,
and concept metrics into a single coherent interface.
"""

import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

# Import GPT-2 specific visualization components
from visualization.gpt2_token_sankey import create_token_sankey_diagram
from visualization.gpt2_attention_sankey import create_attention_sankey_diagram
from visualization.gpt2_head_agreement import create_head_agreement_network
from visualization.gpt2_attention_correlation import calculate_correlation_metrics
from visualization.gpt2_token_movement import create_token_trajectory_plot
from visualization.gpt2_concept_purity import create_concept_purity_heatmap
from visualization.gpt2_cluster_stability import create_cluster_stability_plot

# Import persistence functionality
from concept_fragmentation.persistence import GPT2AnalysisPersistence, save_gpt2_analysis, load_gpt2_analysis


def create_gpt2_metrics_tab():
    """Create the layout for the GPT-2 metrics dashboard tab."""
    return html.Div([
        html.H4("GPT-2 Model Analysis Dashboard", style={"textAlign": "center"}),
        
        # Layer selection
        html.Div([
            html.Label("Select Layers:"),
            dcc.Dropdown(
                id="gpt2-layers-dropdown",
                multi=True,
                placeholder="Select layers to analyze (default: all)",
            ),
        ], style={"width": "100%", "marginBottom": "20px"}),
        
        # Summary metrics card
        html.Div([
            html.H5("GPT-2 Model Metrics Summary", style={"marginBottom": "10px"}),
            html.Div(id="gpt2-metrics-summary", style={"padding": "15px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"}),
        ], style={"marginBottom": "20px"}),
        
        # Main metrics visualization tabs
        dcc.Tabs([
            # Attention Metrics Tab
            dcc.Tab(label="Attention Metrics", children=[
                html.Div([
                    html.H6("Attention Pattern Analysis", style={"marginTop": "15px"}),
                    
                    # Attention visualization selector
                    dcc.RadioItems(
                        id="attention-viz-type",
                        options=[
                            {"label": "Attention Flow Sankey", "value": "sankey"},
                            {"label": "Head Agreement Network", "value": "agreement"},
                            {"label": "Attention Entropy Heatmap", "value": "entropy"},
                        ],
                        value="sankey",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    ),
                    
                    # Attention visualization container
                    dcc.Loading(
                        id="attention-viz-loading",
                        type="circle",
                        children=[
                            html.Div(id="attention-viz-container", style={"height": "600px"})
                        ]
                    ),
                    
                    # Attention metrics table
                    html.H6("Attention Metrics by Layer", style={"marginTop": "15px"}),
                    html.Div(id="attention-metrics-table")
                ])
            ]),
            
            # Token Path Metrics Tab
            dcc.Tab(label="Token Path Metrics", children=[
                html.Div([
                    html.H6("Token Path Analysis", style={"marginTop": "15px"}),
                    
                    # Token path visualization selector
                    dcc.RadioItems(
                        id="token-path-viz-type",
                        options=[
                            {"label": "Token Path Sankey", "value": "sankey"},
                            {"label": "Token Movement Trajectories", "value": "trajectories"},
                            {"label": "Path Fragmentation Heatmap", "value": "fragmentation"},
                        ],
                        value="sankey",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    ),
                    
                    # Token selection for individual analysis
                    html.Div([
                        html.Label("Select Tokens for Individual Analysis:"),
                        dcc.Dropdown(
                            id="token-selection-dropdown",
                            multi=True,
                            placeholder="Select tokens to analyze",
                        )
                    ], style={"marginTop": "10px", "marginBottom": "10px"}),
                    
                    # Token path visualization container
                    dcc.Loading(
                        id="token-path-viz-loading",
                        type="circle",
                        children=[
                            html.Div(id="token-path-viz-container", style={"height": "600px"})
                        ]
                    ),
                    
                    # Token path metrics table
                    html.H6("Token Path Metrics", style={"marginTop": "15px"}),
                    html.Div(id="token-path-metrics-table")
                ])
            ]),
            
            # Concept Metrics Tab
            dcc.Tab(label="Concept Metrics", children=[
                html.Div([
                    html.H6("Concept Organization Analysis", style={"marginTop": "15px"}),
                    
                    # Concept visualization selector
                    dcc.RadioItems(
                        id="concept-viz-type",
                        options=[
                            {"label": "Concept Purity Heatmap", "value": "purity"},
                            {"label": "Cluster Stability Plot", "value": "stability"},
                            {"label": "Layer Similarity Matrix", "value": "similarity"},
                        ],
                        value="purity",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    ),
                    
                    # Concept visualization container
                    dcc.Loading(
                        id="concept-viz-loading",
                        type="circle",
                        children=[
                            html.Div(id="concept-viz-container", style={"height": "600px"})
                        ]
                    ),
                    
                    # Concept metrics table
                    html.H6("Concept Organization Metrics", style={"marginTop": "15px"}),
                    html.Div(id="concept-metrics-table")
                ])
            ]),
            
            # Path-Attention Correlation Tab
            dcc.Tab(label="Path-Attention Correlation", children=[
                html.Div([
                    html.H6("Path-Attention Correlation Analysis", style={"marginTop": "15px"}),
                    
                    # Correlation visualization selector
                    dcc.RadioItems(
                        id="correlation-viz-type",
                        options=[
                            {"label": "Layer Correlation Heatmap", "value": "heatmap"},
                            {"label": "Token Correlation Bar Chart", "value": "token_chart"},
                            {"label": "Combined View", "value": "combined"},
                        ],
                        value="heatmap",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    ),
                    
                    # Correlation visualization container
                    dcc.Loading(
                        id="correlation-viz-loading",
                        type="circle",
                        children=[
                            html.Div(id="correlation-viz-container", style={"height": "600px"})
                        ]
                    ),
                    
                    # Correlation metrics summary
                    html.H6("Correlation Metrics Summary", style={"marginTop": "15px"}),
                    html.Div(id="correlation-metrics-summary")
                ])
            ]),
            
            # Linked Token-Attention View Tab
            dcc.Tab(label="Linked Token-Attention View", children=[
                html.Div([
                    html.H6("Synchronized Token Path and Attention View", style={"marginTop": "15px"}),
                    
                    # Token selection
                    html.Div([
                        html.Label("Select Tokens to Highlight:"),
                        dcc.Dropdown(
                            id="linked-token-selection",
                            multi=True,
                            placeholder="Select tokens to highlight"
                        ),
                    ], style={"marginBottom": "15px"}),
                    
                    # Layer selection
                    html.Div([
                        html.Label("Select Layers to Display:"),
                        dcc.Dropdown(
                            id="linked-layer-selection",
                            multi=True,
                            placeholder="Select layers to display"
                        ),
                    ], style={"marginBottom": "15px"}),
                    
                    # Linked visualizations
                    html.Div([
                        # Token paths Sankey
                        html.Div([
                            html.H6("Token Paths", style={"textAlign": "center"}),
                            dcc.Loading(
                                id="linked-token-path-loading",
                                type="circle",
                                children=[
                                    html.Div(id="linked-token-path-container", style={"height": "400px"})
                                ]
                            )
                        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}),
                        
                        # Attention Sankey
                        html.Div([
                            html.H6("Attention Flow", style={"textAlign": "center"}),
                            dcc.Loading(
                                id="linked-attention-loading",
                                type="circle",
                                children=[
                                    html.Div(id="linked-attention-container", style={"height": "400px"})
                                ]
                            )
                        ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"})
                    ], style={"marginBottom": "15px"}),
                    
                    # Metrics for selected tokens
                    html.Div([
                        html.H6("Selected Token Metrics", style={"marginTop": "15px"}),
                        html.Div(id="linked-token-metrics")
                    ])
                ])
            ]),
            
            # LLM Analysis Tab
            dcc.Tab(label="LLM Analysis", children=[
                html.Div([
                    html.H6("LLM-Generated Analysis", style={"marginTop": "15px"}),
                    
                    # LLM analysis selector
                    dcc.RadioItems(
                        id="llm-analysis-viz-type",
                        options=[
                            {"label": "Attention Pattern Narratives", "value": "attention"},
                            {"label": "Token Movement Narratives", "value": "movement"},
                            {"label": "Concept Purity Analysis", "value": "concepts"},
                            {"label": "Path-Attention Correlation", "value": "correlation"},
                        ],
                        value="attention",
                        labelStyle={"display": "inline-block", "marginRight": "10px"}
                    ),
                    
                    # Generate button
                    html.Div([
                        html.Button(
                            "Generate Selected Analysis", 
                            id="generate-gpt2-llm-button",
                            style={"backgroundColor": "#4CAF50", "color": "white", "padding": "8px", "margin": "10px"}
                        ),
                    ], style={"textAlign": "center"}),
                    
                    # LLM analysis container
                    dcc.Loading(
                        id="llm-analysis-viz-loading",
                        type="circle",
                        children=[
                            html.Div(id="llm-analysis-viz-container", style={"minHeight": "300px"})
                        ]
                    )
                ])
            ])
        ]),
        
        # Store components for data
        dcc.Store(id="gpt2-metrics-data"),
        dcc.Store(id="gpt2-selected-layers"),
        
        # Persistence controls
        html.Div([
            html.H5("Analysis Persistence", style={"marginTop": "20px", "marginBottom": "10px"}),
            html.Div([
                # Save controls
                html.Div([
                    html.Label("Save Current Analysis:"),
                    dcc.Input(
                        id="save-analysis-name",
                        placeholder="Enter analysis name",
                        type="text",
                        style={"width": "200px", "marginRight": "10px"}
                    ),
                    html.Button(
                        "Save Analysis",
                        id="save-analysis-button",
                        style={"backgroundColor": "#4CAF50", "color": "white", "padding": "5px 10px", "marginRight": "10px"}
                    ),
                    html.Div(id="save-analysis-status", style={"display": "inline-block", "marginLeft": "10px"})
                ], style={"marginBottom": "10px"}),
                
                # Load controls
                html.Div([
                    html.Label("Load Saved Analysis:"),
                    dcc.Dropdown(
                        id="saved-analyses-dropdown",
                        placeholder="Select saved analysis",
                        style={"width": "300px", "display": "inline-block", "marginRight": "10px"}
                    ),
                    html.Button(
                        "Load Analysis",
                        id="load-analysis-button",
                        style={"backgroundColor": "#2196F3", "color": "white", "padding": "5px 10px", "marginRight": "10px"}
                    ),
                    html.Button(
                        "Delete Analysis",
                        id="delete-analysis-button",
                        style={"backgroundColor": "#f44336", "color": "white", "padding": "5px 10px", "marginRight": "10px"}
                    ),
                    html.Div(id="load-analysis-status", style={"display": "inline-block", "marginLeft": "10px"})
                ], style={"marginBottom": "10px"}),
                
                # Export controls
                html.Div([
                    html.Label("Export Analysis:"),
                    dcc.Dropdown(
                        id="export-format-dropdown",
                        options=[
                            {"label": "JSON", "value": "json"},
                            {"label": "CSV", "value": "csv"},
                            {"label": "Pickle", "value": "pickle"}
                        ],
                        value="json",
                        style={"width": "100px", "display": "inline-block", "marginRight": "10px"}
                    ),
                    html.Button(
                        "Export",
                        id="export-analysis-button",
                        style={"backgroundColor": "#FF9800", "color": "white", "padding": "5px 10px", "marginRight": "10px"}
                    ),
                    html.Div(id="export-analysis-status", style={"display": "inline-block", "marginLeft": "10px"})
                ])
            ], style={"padding": "15px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
        ], style={"marginBottom": "20px"}),
        
        # Hidden stores for persistence
        dcc.Store(id="persistence-manager"),
        dcc.Store(id="current-analysis-id"),
    ])


def register_gpt2_metrics_callbacks(app):
    """Register callbacks for the GPT-2 metrics dashboard tab."""
    
    # Initialize persistence manager
    persistence_manager = GPT2AnalysisPersistence()
    
    # Callback to initialize persistence manager
    @app.callback(
        Output("persistence-manager", "data"),
        [Input("cluster-paths-data-store", "data")]
    )
    def initialize_persistence_manager(cluster_paths_data):
        """Initialize the persistence manager when data is loaded."""
        if cluster_paths_data:
            return {"initialized": True}
        return {"initialized": False}
    
    # Callback to populate saved analyses dropdown
    @app.callback(
        Output("saved-analyses-dropdown", "options"),
        [Input("persistence-manager", "data")]
    )
    def update_saved_analyses_dropdown(persistence_data):
        """Update the dropdown with available saved analyses."""
        if not persistence_data or not persistence_data.get("initialized"):
            return []
        
        try:
            analyses = persistence_manager.list_analyses()
            options = [
                {
                    "label": f"{analysis.get('model_name', 'Unknown')} - {analysis.get('timestamp', '')[:19]}",
                    "value": analysis.get('analysis_id', '')
                }
                for analysis in analyses
            ]
            return options
        except Exception as e:
            print(f"Error loading saved analyses: {e}")
            return []
    
    # Callback to save analysis
    @app.callback(
        [Output("save-analysis-status", "children"),
         Output("current-analysis-id", "data")],
        [Input("save-analysis-button", "n_clicks")],
        [State("save-analysis-name", "value"),
         State("cluster-paths-data-store", "data")]
    )
    def save_current_analysis(n_clicks, analysis_name, cluster_paths_data):
        """Save the current analysis results."""
        if not n_clicks or not analysis_name or not cluster_paths_data:
            return "", None
        
        try:
            # Extract model info from cluster paths data
            model_name = cluster_paths_data.get("model_type", "GPT-2")
            input_text = cluster_paths_data.get("input_text", "")
            
            # Save analysis
            analysis_id = persistence_manager.save_analysis_results(
                analysis_data=cluster_paths_data,
                model_name=model_name,
                input_text=input_text,
                metadata={"user_name": analysis_name}
            )
            
            return html.Div([
                html.Span("✓ Saved successfully!", style={"color": "green"}),
                html.Br(),
                html.Small(f"ID: {analysis_id}", style={"color": "gray"})
            ]), analysis_id
            
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={"color": "red"}), None
    
    # Callback to load analysis
    @app.callback(
        [Output("cluster-paths-data-store", "data"),
         Output("load-analysis-status", "children")],
        [Input("load-analysis-button", "n_clicks")],
        [State("saved-analyses-dropdown", "value"),
         State("cluster-paths-data-store", "data")]
    )
    def load_saved_analysis(n_clicks, selected_analysis_id, current_data):
        """Load a saved analysis."""
        if not n_clicks or not selected_analysis_id:
            return current_data, ""
        
        try:
            # Load analysis
            loaded_data = persistence_manager.load_analysis_results(selected_analysis_id)
            
            if loaded_data:
                # Extract the analysis data portion
                analysis_data = loaded_data.get("analysis_data", {})
                
                return analysis_data, html.Div("✓ Loaded successfully!", style={"color": "green"})
            else:
                return current_data, html.Div("Analysis not found", style={"color": "red"})
                
        except Exception as e:
            return current_data, html.Div(f"Error: {str(e)}", style={"color": "red"})
    
    # Callback to export analysis
    @app.callback(
        Output("export-analysis-status", "children"),
        [Input("export-analysis-button", "n_clicks")],
        [State("current-analysis-id", "data"),
         State("export-format-dropdown", "value")]
    )
    def export_current_analysis(n_clicks, current_analysis_id, export_format):
        """Export the current analysis."""
        if not n_clicks:
            return ""
        
        if not current_analysis_id:
            return html.Div("No analysis to export (save first)", style={"color": "orange"})
        
        try:
            # Export analysis
            export_path = persistence_manager.export_analysis(
                analysis_id=current_analysis_id,
                export_format=export_format,
                include_visualizations=True
            )
            
            return html.Div([
                html.Span("✓ Exported successfully!", style={"color": "green"}),
                html.Br(),
                html.Small(f"Path: {export_path}", style={"color": "gray"})
            ])
            
        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={"color": "red"})
    
    # Define callback to handle drill-down clicks
    @app.callback(
        [
            Output("current-drill-down", "children"),
            Output("drill-down-panel", "style"),
            Output("drill-down-title", "children"),
            Output("drill-down-content", "children")
        ],
        [
            Input("token-count-link", "n_clicks"),
            Input("cluster-count-link", "n_clicks"),
            Input("archetypal-paths-link", "n_clicks"),
            Input("token-paths-link", "n_clicks"),
            Input("attention-data-link", "n_clicks"),
            Input("concept-metrics-link", "n_clicks"),
            Input("fragmentation-link", "n_clicks"),
            Input({"type": "layer-cluster-link", "index": dash.dependencies.ALL}, "n_clicks")
        ],
        [
            State("current-drill-down", "children"),
            State("cluster-paths-data-store", "data")
        ]
    )
    def update_drill_down_panel(*args):
        """Update the drill-down panel based on which metric was clicked."""
        # Extract inputs and states
        inputs = args[:-2]  # All inputs
        states = args[-2:]  # All states
        current_drill_down = states[0]
        cluster_paths_data = states[1]
        
        # Check which input triggered the callback
        ctx = dash.callback_context
        if not ctx.triggered:
            # No clicks yet
            return current_drill_down, {"display": "none"}, "", ""
        
        # Get the ID of the component that triggered the callback
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        
        # Default return values
        panel_style = {"display": "block", "marginTop": "20px", "padding": "15px", "backgroundColor": "#f0f0f0", "borderRadius": "5px"}
        drill_down_title = "Drill-Down Analysis"
        drill_down_content = ""
        
        # Handle layer cluster link clicks (pattern matching callback)
        if "index" in trigger_id:
            # Parse the JSON to get the layer name
            import json
            trigger_dict = json.loads(trigger_id)
            if trigger_dict.get("type") == "layer-cluster-link":
                layer_name = trigger_dict.get("index")
                drill_down_title = f"Clusters in Layer {layer_name}"
                
                # Get clusters for this layer
                if cluster_paths_data and "cluster_labels" in cluster_paths_data:
                    cluster_labels = cluster_paths_data["cluster_labels"]
                    if layer_name in cluster_labels:
                        # Get number of tokens in each cluster
                        clusters = cluster_labels[layer_name]
                        if isinstance(clusters, list) or isinstance(clusters, np.ndarray):
                            cluster_counts = {}
                            for cluster_id in clusters:
                                cluster_id = int(cluster_id)  # Convert to int (may be numpy type)
                                if cluster_id >= 0:  # Skip -1 (no cluster)
                                    if cluster_id not in cluster_counts:
                                        cluster_counts[cluster_id] = 0
                                    cluster_counts[cluster_id] += 1
                            
                            # Create table of cluster statistics
                            drill_down_content = html.Div([
                                html.Table([
                                    # Header
                                    html.Thead(html.Tr([
                                        html.Th("Cluster ID", style={"textAlign": "left", "padding": "8px"}),
                                        html.Th("Token Count", style={"textAlign": "left", "padding": "8px"}),
                                        html.Th("Actions", style={"textAlign": "left", "padding": "8px"})
                                    ])),
                                    # Body
                                    html.Tbody([
                                        html.Tr([
                                            html.Td(str(cluster_id), style={"padding": "8px"}),
                                            html.Td(str(count), style={"padding": "8px"}),
                                            html.Td(
                                                html.Button(
                                                    "View Tokens",
                                                    id={"type": "view-cluster-tokens", "layer": layer_name, "cluster": cluster_id},
                                                    style={"backgroundColor": "#4CAF50", "color": "white", "border": "none", "padding": "5px 10px", "cursor": "pointer"}
                                                ),
                                                style={"padding": "8px"}
                                            )
                                        ]) for cluster_id, count in sorted(cluster_counts.items())
                                    ])
                                ], style={"width": "100%", "borderCollapse": "collapse"}),
                                
                                # Container for token list (populated when View Tokens is clicked)
                                html.Div(id="cluster-tokens-container")
                            ])
                        else:
                            drill_down_content = html.Div("Cluster data format not supported.")
                    else:
                        drill_down_content = html.Div(f"No cluster data available for layer {layer_name}.")
                else:
                    drill_down_content = html.Div("No cluster data available.")
                
                return layer_name, panel_style, drill_down_title, drill_down_content
        
        # Handle fixed ID clicks
        if trigger_id == "token-count-link":
            drill_down_title = "Token Analysis"
            
            if cluster_paths_data and "token_metadata" in cluster_paths_data:
                token_metadata = cluster_paths_data["token_metadata"]
                tokens = token_metadata.get("tokens", [])
                positions = token_metadata.get("positions", [])
                
                if tokens and positions and len(tokens) == len(positions):
                    # Create token selection dropdown for drilling down
                    token_options = [
                        {"label": f"'{token}' (pos: {position})", "value": str(i)}
                        for i, (token, position) in enumerate(zip(tokens, positions))
                    ]
                    
                    drill_down_content = html.Div([
                        html.Div([
                            html.Label("Select Tokens for Detailed Analysis:"),
                            dcc.Dropdown(
                                id="drill-down-token-selection",
                                options=token_options,
                                multi=True,
                                placeholder="Select tokens to analyze",
                                style={"width": "100%"}
                            ),
                        ], style={"marginBottom": "15px"}),
                        
                        # Container for token metrics when selected
                        html.Div(id="drill-down-token-metrics")
                    ])
                else:
                    drill_down_content = html.Div("No token data available.")
            else:
                drill_down_content = html.Div("No token data available.")
            
            return "token-count", panel_style, drill_down_title, drill_down_content
            
        elif trigger_id == "cluster-count-link":
            drill_down_title = "Cluster Distribution Analysis"
            
            if cluster_paths_data and "cluster_labels" in cluster_paths_data:
                cluster_labels = cluster_paths_data["cluster_labels"]
                
                # Create cluster count by layer visualization
                from plotly.subplots import make_subplots
                import plotly.graph_objects as go
                
                # Get cluster counts by layer
                cluster_counts_by_layer = {}
                for layer, clusters in cluster_labels.items():
                    if isinstance(clusters, np.ndarray) or isinstance(clusters, list):
                        counts = {}
                        for cluster_id in clusters:
                            cluster_id = int(cluster_id)  # Convert to int (may be numpy type)
                            if cluster_id >= 0:  # Skip -1 (no cluster)
                                if cluster_id not in counts:
                                    counts[cluster_id] = 0
                                counts[cluster_id] += 1
                        cluster_counts_by_layer[layer] = counts
                
                # Create bar chart
                fig = go.Figure()
                
                for layer, counts in cluster_counts_by_layer.items():
                    fig.add_trace(go.Bar(
                        name=layer,
                        x=list(range(max(counts.keys()) + 1)),
                        y=[counts.get(i, 0) for i in range(max(counts.keys()) + 1)],
                        hovertemplate='Cluster %{x}<br>Count: %{y}'
                    ))
                
                fig.update_layout(
                    title="Token Distribution Across Clusters by Layer",
                    xaxis_title="Cluster ID",
                    yaxis_title="Token Count",
                    legend_title="Layer",
                    barmode='group'
                )
                
                drill_down_content = html.Div([
                    dcc.Graph(figure=fig, style={"height": "400px"}),
                    html.Div([
                        html.P("Click on a layer in the legend to isolate it, or double-click to see all layers.")
                    ], style={"marginTop": "10px"})
                ])
            else:
                drill_down_content = html.Div("No cluster data available.")
            
            return "cluster-count", panel_style, drill_down_title, drill_down_content
            
        elif trigger_id == "archetypal-paths-link":
            drill_down_title = "Archetypal Path Analysis"
            
            if cluster_paths_data and "path_archetypes" in cluster_paths_data:
                path_archetypes = cluster_paths_data["path_archetypes"]
                
                # Create path archetype table with sample tokens
                rows = []
                for i, path in enumerate(path_archetypes):
                    path_id = path.get("id", i)
                    path_layers = path.get("layers", [])
                    path_clusters = path.get("clusters", [])
                    path_count = path.get("count", 0)
                    example_tokens = path.get("example_tokens", [])
                    
                    # Format path as layer -> cluster sequence
                    path_str = " → ".join([f"{layer}:{cluster}" for layer, cluster in zip(path_layers, path_clusters)])
                    
                    # Format example tokens (limit to 5)
                    if len(example_tokens) > 5:
                        token_str = ", ".join(example_tokens[:5]) + f" (and {len(example_tokens) - 5} more)"
                    else:
                        token_str = ", ".join(example_tokens)
                    
                    rows.append({
                        "Path ID": path_id,
                        "Path": path_str,
                        "Token Count": path_count,
                        "Example Tokens": token_str
                    })
                
                drill_down_content = html.Div([
                    html.Table([
                        # Header
                        html.Thead(html.Tr([
                            html.Th(col, style={"textAlign": "left", "padding": "8px"})
                            for col in ["Path ID", "Path", "Token Count", "Example Tokens"]
                        ])),
                        # Body
                        html.Tbody([
                            html.Tr([
                                html.Td(row[col], style={"padding": "8px"})
                                for col in ["Path ID", "Path", "Token Count", "Example Tokens"]
                            ]) for row in rows
                        ])
                    ], style={"width": "100%", "borderCollapse": "collapse"}),
                    
                    # Container for path details (populated when a path is clicked)
                    html.Div(id="path-details-container")
                ])
            else:
                drill_down_content = html.Div("No path archetype data available.")
            
            return "archetypal-paths", panel_style, drill_down_title, drill_down_content
            
        elif trigger_id == "token-paths-link":
            drill_down_title = "Token Path Analysis"
            
            # Redirect to Token Path Metrics tab with Sankey visualization
            # Use JavaScript to programmatically select the tab
            app.clientside_callback(
                """
                function(n_clicks) {
                    // Find and click the Token Path Metrics tab
                    const tabs = document.querySelectorAll('.rc-tabs-tab');
                    for (let tab of tabs) {
                        if (tab.textContent.includes('Token Path Metrics')) {
                            tab.click();
                            break;
                        }
                    }
                    return n_clicks;
                }
                """,
                Output("token-paths-link", "n_clicks"),
                [Input("token-paths-link", "n_clicks")],
                prevent_initial_call=True
            )
            
            # Still need to provide some content for the drill-down panel
            drill_down_content = html.Div([
                html.P("Redirecting to Token Path Metrics tab...")
            ])
            
            return "token-paths", {"display": "none"}, drill_down_title, drill_down_content
            
        elif trigger_id == "attention-data-link":
            drill_down_title = "Attention Analysis"
            
            # Redirect to Attention Metrics tab
            app.clientside_callback(
                """
                function(n_clicks) {
                    // Find and click the Attention Metrics tab
                    const tabs = document.querySelectorAll('.rc-tabs-tab');
                    for (let tab of tabs) {
                        if (tab.textContent.includes('Attention Metrics')) {
                            tab.click();
                            break;
                        }
                    }
                    return n_clicks;
                }
                """,
                Output("attention-data-link", "n_clicks"),
                [Input("attention-data-link", "n_clicks")],
                prevent_initial_call=True
            )
            
            drill_down_content = html.Div([
                html.P("Redirecting to Attention Metrics tab...")
            ])
            
            return "attention-data", {"display": "none"}, drill_down_title, drill_down_content
            
        elif trigger_id == "concept-metrics-link":
            drill_down_title = "Concept Metrics Analysis"
            
            # Redirect to Concept Metrics tab
            app.clientside_callback(
                """
                function(n_clicks) {
                    // Find and click the Concept Metrics tab
                    const tabs = document.querySelectorAll('.rc-tabs-tab');
                    for (let tab of tabs) {
                        if (tab.textContent.includes('Concept Metrics')) {
                            tab.click();
                            break;
                        }
                    }
                    return n_clicks;
                }
                """,
                Output("concept-metrics-link", "n_clicks"),
                [Input("concept-metrics-link", "n_clicks")],
                prevent_initial_call=True
            )
            
            drill_down_content = html.Div([
                html.P("Redirecting to Concept Metrics tab...")
            ])
            
            return "concept-metrics", {"display": "none"}, drill_down_title, drill_down_content
            
        elif trigger_id == "fragmentation-link":
            drill_down_title = "Path Fragmentation Analysis"
            
            if cluster_paths_data and "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"]:
                fragmentation_data = cluster_paths_data["similarity"]["fragmentation_scores"]
                scores = fragmentation_data.get("scores", [])
                tokens = fragmentation_data.get("tokens", [])
                
                if scores and tokens and len(scores) == len(tokens):
                    # Create a sorted list of tokens by fragmentation score
                    token_scores = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)
                    
                    # Create token selection dropdown for drilling down
                    token_options = [
                        {"label": f"'{token}' (score: {score:.4f})", "value": token}
                        for token, score in token_scores
                    ]
                    
                    # Create bar chart of top fragmented tokens
                    top_tokens = token_scores[:20]  # Top 20 tokens
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(data=go.Bar(
                        x=[token for token, _ in top_tokens],
                        y=[score for _, score in top_tokens],
                        marker_color='crimson'
                    ))
                    
                    fig.update_layout(
                        title="Top 20 Tokens by Fragmentation Score",
                        xaxis_title="Token",
                        yaxis_title="Fragmentation Score",
                        xaxis_tickangle=-45
                    )
                    
                    drill_down_content = html.Div([
                        dcc.Graph(figure=fig, style={"height": "400px"}),
                        html.Div([
                            html.Label("Select Token for Path Analysis:"),
                            dcc.Dropdown(
                                id="fragmentation-token-dropdown",
                                options=token_options,
                                placeholder="Select a token to view its path",
                                style={"width": "100%"}
                            )
                        ], style={"marginTop": "15px", "marginBottom": "15px"}),
                        html.Div(id="fragmentation-token-detail")
                    ])
                else:
                    drill_down_content = html.Div("No fragmentation data available.")
            else:
                drill_down_content = html.Div("No fragmentation data available.")
            
            return "fragmentation", panel_style, drill_down_title, drill_down_content
        
        # Default fallback
        return current_drill_down, {"display": "none"}, "", ""
    
    # Callback for token selection in drill-down panel
    @app.callback(
        Output("drill-down-token-metrics", "children"),
        [Input("drill-down-token-selection", "value")],
        [State("cluster-paths-data-store", "data")]
    )
    def update_drill_down_token_metrics(selected_tokens, cluster_paths_data):
        """Update the token metrics in the drill-down panel."""
        if not selected_tokens or not cluster_paths_data or "token_paths" not in cluster_paths_data:
            return html.Div("Select tokens to view their metrics.")
        
        token_paths = cluster_paths_data["token_paths"]
        token_metadata = cluster_paths_data.get("token_metadata", {})
        tokens = token_metadata.get("tokens", [])
        
        # Convert selected tokens to integers
        token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
        
        # Create metrics table and visualizations for selected tokens
        metrics_data = []
        
        for token_id in token_indices:
            if token_id < len(tokens) and str(token_id) in token_paths:
                token_text = tokens[token_id]
                path_data = token_paths[str(token_id)]
                
                metrics_data.append({
                    "Token": f"'{token_text}'",
                    "Position": path_data.get("position", "N/A"),
                    "Path Length": f"{path_data.get('path_length', 'N/A'):.4f}" if path_data.get('path_length') is not None else "N/A",
                    "Cluster Changes": path_data.get("cluster_changes", "N/A"),
                    "Mobility Score": f"{path_data.get('mobility_score', 'N/A'):.4f}" if path_data.get('mobility_score') is not None else "N/A"
                })
        
        if not metrics_data:
            return html.Div("No metrics available for selected tokens.")
        
        # Create token paths visualization
        from visualization.gpt2_token_sankey import create_token_sankey_diagram
        
        try:
            # Create token paths Sankey diagram
            fig = create_token_sankey_diagram(
                token_paths=token_paths,
                selected_tokens=token_indices,
                highlight_selected=True
            )
            
            # Create results div with table and visualization
            results = html.Div([
                html.H6("Selected Token Metrics", style={"marginTop": "15px"}),
                html.Table([
                    # Header
                    html.Thead(html.Tr([
                        html.Th(col, style={"textAlign": "left", "padding": "8px"})
                        for col in ["Token", "Position", "Path Length", "Cluster Changes", "Mobility Score"]
                    ])),
                    # Body
                    html.Tbody([
                        html.Tr([
                            html.Td(row[col], style={"padding": "8px"})
                            for col in ["Token", "Position", "Path Length", "Cluster Changes", "Mobility Score"]
                        ]) for row in metrics_data
                    ])
                ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "15px"}),
                
                html.H6("Token Path Visualization", style={"marginTop": "15px"}),
                dcc.Graph(figure=fig, style={"height": "400px"})
            ])
            
            return results
        except Exception as e:
            return html.Div(f"Error creating token visualization: {str(e)}")
    
    # Callback for fragmentation token selection
    @app.callback(
        Output("fragmentation-token-detail", "children"),
        [Input("fragmentation-token-dropdown", "value")],
        [State("cluster-paths-data-store", "data")]
    )
    def update_fragmentation_token_detail(selected_token, cluster_paths_data):
        """Update the token detail view for fragmentation analysis."""
        if not selected_token or not cluster_paths_data:
            return html.Div("Select a token to view its path details.")
        
        # Get token paths and layer information
        token_paths = cluster_paths_data.get("token_paths", {})
        layers = cluster_paths_data.get("layers", [])
        
        # Find the token in the token paths
        token_path_data = None
        token_position = None
        
        for token_id, path_data in token_paths.items():
            if "token_text" in path_data and path_data["token_text"] == selected_token:
                token_path_data = path_data
                token_position = path_data.get("position")
                break
        
        if not token_path_data:
            return html.Div(f"No path data found for token '{selected_token}'")
        
        # Get the cluster path
        cluster_path = token_path_data.get("cluster_path", [])
        
        # Create a table showing the path through layers
        path_rows = []
        prev_cluster = None
        
        for i, (layer, cluster) in enumerate(zip(layers, cluster_path)):
            # Check if cluster changed from previous layer
            changed = prev_cluster is not None and cluster != prev_cluster
            prev_cluster = cluster
            
            path_rows.append({
                "Layer": layer,
                "Cluster": cluster,
                "Changed": "Yes" if changed else "No"
            })
        
        # Create visualization of token path
        from visualization.gpt2_token_sankey import create_token_sankey_diagram
        
        try:
            # Find the token index
            token_idx = None
            for idx, data in token_paths.items():
                if "token_text" in data and data["token_text"] == selected_token:
                    token_idx = int(idx)
                    break
            
            if token_idx is not None:
                # Create token path visualization
                fig = create_token_sankey_diagram(
                    token_paths=token_paths,
                    selected_tokens=[token_idx],
                    highlight_selected=True
                )
                
                # Return results
                return html.Div([
                    html.H6(f"Path for Token '{selected_token}' (Position: {token_position})", style={"marginTop": "15px"}),
                    html.Table([
                        # Header
                        html.Thead(html.Tr([
                            html.Th(col, style={"textAlign": "left", "padding": "8px"})
                            for col in ["Layer", "Cluster", "Changed"]
                        ])),
                        # Body
                        html.Tbody([
                            html.Tr([
                                html.Td(row[col], style={
                                    "padding": "8px",
                                    "backgroundColor": "#ffeeee" if col == "Changed" and row[col] == "Yes" else "transparent"
                                })
                                for col in ["Layer", "Cluster", "Changed"]
                            ]) for row in path_rows
                        ])
                    ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "15px"}),
                    
                    html.H6("Token Path Visualization", style={"marginTop": "15px"}),
                    dcc.Graph(figure=fig, style={"height": "400px"})
                ])
            else:
                return html.Div(f"Could not find token index for '{selected_token}'")
        except Exception as e:
            return html.Div(f"Error creating visualization: {str(e)}")
    
    # Callback for viewing cluster tokens
    @app.callback(
        Output("cluster-tokens-container", "children"),
        [Input({"type": "view-cluster-tokens", "layer": dash.dependencies.ALL, "cluster": dash.dependencies.ALL}, "n_clicks")],
        [State("cluster-paths-data-store", "data")]
    )
    def show_cluster_tokens(n_clicks_list, cluster_paths_data):
        """Show tokens in the selected cluster."""
        if not n_clicks_list or not any(n_clicks for n_clicks in n_clicks_list if n_clicks) or not cluster_paths_data:
            return html.Div()
        
        # Get which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            return html.Div()
        
        # Get the ID of the button that was clicked
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        import json
        button_data = json.loads(button_id)
        
        layer_name = button_data.get("layer")
        cluster_id = button_data.get("cluster")
        
        if not layer_name or cluster_id is None:
            return html.Div()
        
        # Get tokens in this cluster
        token_metadata = cluster_paths_data.get("token_metadata", {})
        cluster_labels = cluster_paths_data.get("cluster_labels", {})
        
        tokens = token_metadata.get("tokens", [])
        positions = token_metadata.get("positions", [])
        
        if not tokens or not positions or layer_name not in cluster_labels:
            return html.Div("No token data available.")
        
        # Find tokens in this cluster
        cluster_tokens = []
        clusters = cluster_labels[layer_name]
        
        for i, cluster in enumerate(clusters):
            if int(cluster) == int(cluster_id):
                if i < len(tokens):
                    token_text = tokens[i]
                    position = positions[i] if i < len(positions) else "N/A"
                    cluster_tokens.append({"token": token_text, "position": position})
        
        if not cluster_tokens:
            return html.Div(f"No tokens found in cluster {cluster_id} for layer {layer_name}.")
        
        # Create table of tokens
        return html.Div([
            html.H6(f"Tokens in Cluster {cluster_id} (Layer {layer_name})", style={"marginTop": "20px"}),
            html.Table([
                # Header
                html.Thead(html.Tr([
                    html.Th(col, style={"textAlign": "left", "padding": "8px"})
                    for col in ["Token", "Position"]
                ])),
                # Body
                html.Tbody([
                    html.Tr([
                        html.Td(f"'{row['token']}'", style={"padding": "8px"}),
                        html.Td(str(row["position"]), style={"padding": "8px"})
                    ]) for row in sorted(cluster_tokens, key=lambda x: x["position"])
                ])
            ], style={"width": "100%", "borderCollapse": "collapse"})
        ])
    
    @app.callback(
        [Output("gpt2-layers-dropdown", "options"),
         Output("gpt2-layers-dropdown", "value")],
        [Input("cluster-paths-data-store", "data")]
    )
    def update_gpt2_layer_options(cluster_paths_data):
        """Update the layer selection dropdown options."""
        if not cluster_paths_data or "layers" not in cluster_paths_data:
            return [], []
        
        layers = cluster_paths_data.get("layers", [])
        # Create options for dropdown
        options = [{"label": layer, "value": layer} for layer in layers]
        # Select all layers by default
        values = layers
        
        return options, values
    
    @app.callback(
        Output("gpt2-selected-layers", "data"),
        [Input("gpt2-layers-dropdown", "value"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_selected_layers(selected_layers, cluster_paths_data):
        """Update the selected layers store."""
        if not cluster_paths_data:
            return []
        
        # Default to all layers if none selected
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        return selected_layers
    
    @app.callback(
        Output("gpt2-metrics-summary", "children"),
        [Input("cluster-paths-data-store", "data")]
    )
    def update_gpt2_metrics_summary(cluster_paths_data):
        """Update the GPT-2 metrics summary card."""
        if not cluster_paths_data:
            return html.Div("No GPT-2 analysis data available.")
        
        # Extract model information
        model_type = cluster_paths_data.get("model_type", "Unknown")
        num_layers = len(cluster_paths_data.get("layers", []))
        num_tokens = len(cluster_paths_data.get("token_metadata", {}).get("tokens", []))
        
        # Extract cluster information
        total_clusters = 0
        clusters_by_layer = {}
        
        for layer_name, clusters in cluster_paths_data.get("cluster_labels", {}).items():
            if isinstance(clusters, np.ndarray):
                num_clusters = len(np.unique(clusters))
            elif isinstance(clusters, list):
                num_clusters = len(set(clusters))
            else:
                num_clusters = 0
                
            clusters_by_layer[layer_name] = num_clusters
            total_clusters += num_clusters
        
        # Extract path information
        total_paths = len(cluster_paths_data.get("path_archetypes", []))
        
        # Extract attention information
        has_attention = "attention_data" in cluster_paths_data
        
        # Create summary card with clickable metrics
        summary = html.Div([
            # Model information
            html.Div([
                html.Div("Model Information", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "5px"}),
                html.Table([
                    html.Tr([
                        html.Td("Model Type:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(model_type)
                    ]),
                    html.Tr([
                        html.Td("Layers:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(str(num_layers))
                    ]),
                    html.Tr([
                        html.Td("Tokens:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                str(num_tokens),
                                id="token-count-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Td("Total Clusters:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                str(total_clusters),
                                id="cluster-count-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Td("Archetypal Paths:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                str(total_paths),
                                id="archetypal-paths-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Td("Has Attention Data:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td("Yes" if has_attention else "No")
                    ])
                ], style={"marginBottom": "15px"})
            ], style={"marginBottom": "15px", "width": "33%", "display": "inline-block", "verticalAlign": "top"}),
            
            # Cluster information
            html.Div([
                html.Div("Clusters by Layer", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "5px"}),
                html.Table([
                    html.Tr([
                        html.Td(layer + ":", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                str(num_clusters),
                                id={"type": "layer-cluster-link", "index": layer},
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"}
                            )
                        )
                    ]) for layer, num_clusters in clusters_by_layer.items()
                ], style={"marginBottom": "15px"})
            ], style={"marginBottom": "15px", "width": "33%", "display": "inline-block", "verticalAlign": "top"}),
            
            # Analysis status
            html.Div([
                html.Div("Analysis Status", style={"fontWeight": "bold", "fontSize": "16px", "marginBottom": "5px"}),
                html.Table([
                    html.Tr([
                        html.Td("Token Paths:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                "Available" if "token_paths" in cluster_paths_data else "Not Available",
                                id="token-paths-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"} if "token_paths" in cluster_paths_data else {}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Td("Attention Analysis:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                "Available" if "attention_data" in cluster_paths_data else "Not Available",
                                id="attention-data-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"} if "attention_data" in cluster_paths_data else {}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Td("Concept Metrics:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                "Available" if "cluster_metrics" in cluster_paths_data else "Not Available",
                                id="concept-metrics-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"} if "cluster_metrics" in cluster_paths_data else {}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Td("Path Fragmentation:", style={"paddingRight": "10px", "fontWeight": "bold"}),
                        html.Td(
                            html.A(
                                "Available" if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"] else "Not Available",
                                id="fragmentation-link",
                                style={"textDecoration": "underline", "cursor": "pointer", "color": "#007bff"} if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"] else {}
                            )
                        )
                    ])
                ], style={"marginBottom": "15px"})
            ], style={"marginBottom": "15px", "width": "33%", "display": "inline-block", "verticalAlign": "top"}),
            
            # Hidden div to store currently selected drill-down
            html.Div(id="current-drill-down", style={"display": "none"}),
            
            # Drill-down detail panel (hidden by default)
            html.Div([
                html.H6("Drill-Down Analysis", id="drill-down-title"),
                html.Div(id="drill-down-content")
            ], id="drill-down-panel", style={"display": "none", "marginTop": "20px", "padding": "15px", "backgroundColor": "#f0f0f0", "borderRadius": "5px"})
        ])
        
        return summary
    
    # Callbacks for attention metrics tab
    @app.callback(
        Output("attention-viz-container", "children"),
        [Input("attention-viz-type", "value"),
         Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_attention_visualization(viz_type, selected_layers, cluster_paths_data):
        """Update the attention visualization based on the selected type."""
        if not cluster_paths_data or "attention_data" not in cluster_paths_data:
            return html.Div("No attention data available for visualization.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Create appropriate visualization based on selection
        if viz_type == "sankey":
            # Create attention flow Sankey diagram
            attention_data = cluster_paths_data["attention_data"]
            token_metadata = cluster_paths_data.get("token_metadata", {})
            
            fig = create_attention_sankey_diagram(
                attention_data=attention_data,
                token_metadata=token_metadata,
                selected_layers=selected_layers
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "agreement":
            # Create head agreement network
            attention_data = cluster_paths_data["attention_data"]
            
            fig = create_head_agreement_network(
                attention_data=attention_data,
                selected_layers=selected_layers
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "entropy":
            # Create attention entropy heatmap
            attention_data = cluster_paths_data["attention_data"]
            
            # Extract entropy values for each layer
            entropy_by_layer = {}
            for layer, layer_data in attention_data.items():
                if layer in selected_layers and "entropy" in layer_data:
                    entropy_by_layer[layer] = layer_data["entropy"]
            
            # Create heatmap
            if entropy_by_layer:
                layers = list(entropy_by_layer.keys())
                values = list(entropy_by_layer.values())
                
                fig = go.Figure(data=go.Heatmap(
                    z=[values],
                    x=layers,
                    y=["Entropy"],
                    colorscale="Viridis",
                    colorbar={"title": "Entropy"}
                ))
                
                fig.update_layout(
                    title="Attention Entropy by Layer",
                    xaxis_title="Layer",
                    yaxis_title="Metric"
                )
                
                return dcc.Graph(figure=fig, style={"height": "100%"})
            else:
                return html.Div("No entropy data available for selected layers.")
        
        # Default fallback
        return html.Div("Please select a visualization type.")
    
    @app.callback(
        Output("attention-metrics-table", "children"),
        [Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_attention_metrics_table(selected_layers, cluster_paths_data):
        """Update the attention metrics table."""
        if not cluster_paths_data or "attention_data" not in cluster_paths_data:
            return html.Div("No attention metrics available.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Extract metrics for each layer
        attention_data = cluster_paths_data["attention_data"]
        
        # Create metrics data
        metrics_data = []
        
        for layer in selected_layers:
            if layer in attention_data:
                layer_data = attention_data[layer]
                
                metrics_data.append({
                    "Layer": layer,
                    "Entropy": f"{layer_data.get('entropy', 'N/A'):.4f}" if layer_data.get('entropy') is not None else "N/A",
                    "Head Agreement": f"{layer_data.get('head_agreement', 'N/A'):.4f}" if layer_data.get('head_agreement') is not None else "N/A",
                    "Num Heads": layer_data.get('num_heads', 'N/A')
                })
        
        if not metrics_data:
            return html.Div("No metrics available for selected layers.")
        
        # Create table
        table = html.Table([
            # Header
            html.Thead(html.Tr([
                html.Th(col, style={"textAlign": "left", "padding": "8px"})
                for col in ["Layer", "Entropy", "Head Agreement", "Num Heads"]
            ])),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(row[col], style={"padding": "8px"})
                    for col in ["Layer", "Entropy", "Head Agreement", "Num Heads"]
                ]) for row in metrics_data
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
        
        return table
    
    # Callbacks for token path metrics tab
    @app.callback(
        [Output("token-selection-dropdown", "options"),
         Output("token-selection-dropdown", "value")],
        [Input("cluster-paths-data-store", "data")]
    )
    def update_token_selection_dropdown(cluster_paths_data):
        """Update the token selection dropdown."""
        if not cluster_paths_data or "token_metadata" not in cluster_paths_data:
            return [], []
        
        # Extract token information
        token_metadata = cluster_paths_data["token_metadata"]
        tokens = token_metadata.get("tokens", [])
        positions = token_metadata.get("positions", [])
        
        if not tokens or len(tokens) != len(positions):
            return [], []
        
        # Create options for dropdown
        options = [
            {"label": f"'{token}' (pos: {position})", "value": str(i)}
            for i, (token, position) in enumerate(zip(tokens, positions))
        ]
        
        # Default to no selection
        value = []
        
        return options, value
    
    @app.callback(
        Output("token-path-viz-container", "children"),
        [Input("token-path-viz-type", "value"),
         Input("token-selection-dropdown", "value"),
         Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_token_path_visualization(viz_type, selected_tokens, selected_layers, cluster_paths_data):
        """Update the token path visualization based on the selected type."""
        if not cluster_paths_data:
            return html.Div("No cluster paths data available for visualization.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Create appropriate visualization based on selection
        if viz_type == "sankey":
            # Create token path Sankey diagram
            if "token_paths" not in cluster_paths_data:
                return html.Div("No token path data available.")
            
            token_paths = cluster_paths_data["token_paths"]
            
            # Convert selected tokens to integers
            token_indices = []
            if selected_tokens:
                token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
            
            fig = create_token_sankey_diagram(
                token_paths=token_paths,
                selected_tokens=token_indices if token_indices else None,
                selected_layers=selected_layers
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "trajectories":
            # Create token movement trajectories
            if "token_paths" not in cluster_paths_data or "activations" not in cluster_paths_data:
                return html.Div("No token path or activation data available.")
            
            token_paths = cluster_paths_data["token_paths"]
            activations = cluster_paths_data["activations"]
            
            # Convert selected tokens to integers
            token_indices = []
            if selected_tokens:
                token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
            
            fig = create_token_trajectory_plot(
                token_paths=token_paths,
                activations=activations,
                selected_tokens=token_indices if token_indices else None,
                selected_layers=selected_layers
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "fragmentation":
            # Create fragmentation heatmap
            if "similarity" not in cluster_paths_data or "fragmentation_scores" not in cluster_paths_data["similarity"]:
                return html.Div("No fragmentation data available.")
            
            fragmentation_data = cluster_paths_data["similarity"]["fragmentation_scores"]
            
            # Extract scores and create heatmap
            scores = fragmentation_data.get("scores", [])
            tokens = fragmentation_data.get("tokens", [])
            
            if not scores or not tokens:
                return html.Div("No fragmentation scores available.")
            
            # Create heatmap data
            if selected_tokens:
                # Filter to selected tokens
                token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
                filtered_scores = [scores[i] for i in token_indices if i < len(scores)]
                filtered_tokens = [tokens[i] for i in token_indices if i < len(tokens)]
                
                if not filtered_scores:
                    return html.Div("No fragmentation data for selected tokens.")
                    
                z_data = [filtered_scores]
                x_data = [f"'{token}'" for token in filtered_tokens]
            else:
                # Use top 20 most fragmented tokens
                paired_data = [(token, score) for token, score in zip(tokens, scores)]
                sorted_data = sorted(paired_data, key=lambda x: x[1], reverse=True)[:20]
                
                z_data = [[score for _, score in sorted_data]]
                x_data = [f"'{token}'" for token, _ in sorted_data]
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=x_data,
                y=["Fragmentation"],
                colorscale="Reds",
                colorbar={"title": "Fragmentation Score"}
            ))
            
            fig.update_layout(
                title="Token Path Fragmentation Scores",
                xaxis_title="Token",
                yaxis_title="Metric",
                xaxis={"tickangle": 45}
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
        
        # Default fallback
        return html.Div("Please select a visualization type.")
    
    @app.callback(
        Output("token-path-metrics-table", "children"),
        [Input("token-selection-dropdown", "value"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_token_path_metrics_table(selected_tokens, cluster_paths_data):
        """Update the token path metrics table."""
        if not cluster_paths_data or "token_paths" not in cluster_paths_data:
            return html.Div("No token path metrics available.")
        
        token_paths = cluster_paths_data["token_paths"]
        token_metadata = cluster_paths_data.get("token_metadata", {})
        
        # Extract token text information
        tokens = token_metadata.get("tokens", [])
        
        # Create metrics data
        metrics_data = []
        
        if selected_tokens:
            # Filter to selected tokens
            token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
            
            for token_id in token_indices:
                if token_id < len(tokens) and str(token_id) in token_paths:
                    token_text = tokens[token_id]
                    path_data = token_paths[str(token_id)]
                    
                    metrics_data.append({
                        "Token": f"'{token_text}'",
                        "Position": path_data.get("position", "N/A"),
                        "Path Length": f"{path_data.get('path_length', 'N/A'):.4f}" if path_data.get('path_length') is not None else "N/A",
                        "Cluster Changes": path_data.get("cluster_changes", "N/A"),
                        "Mobility Score": f"{path_data.get('mobility_score', 'N/A'):.4f}" if path_data.get('mobility_score') is not None else "N/A"
                    })
        else:
            # Show metrics for all tokens (limited to first 20)
            for token_id, path_data in list(token_paths.items())[:20]:
                token_idx = int(token_id)
                if token_idx < len(tokens):
                    token_text = tokens[token_idx]
                    
                    metrics_data.append({
                        "Token": f"'{token_text}'",
                        "Position": path_data.get("position", "N/A"),
                        "Path Length": f"{path_data.get('path_length', 'N/A'):.4f}" if path_data.get('path_length') is not None else "N/A",
                        "Cluster Changes": path_data.get("cluster_changes", "N/A"),
                        "Mobility Score": f"{path_data.get('mobility_score', 'N/A'):.4f}" if path_data.get('mobility_score') is not None else "N/A"
                    })
        
        if not metrics_data:
            return html.Div("No metrics available for selected tokens.")
        
        # Create table
        table = html.Table([
            # Header
            html.Thead(html.Tr([
                html.Th(col, style={"textAlign": "left", "padding": "8px"})
                for col in ["Token", "Position", "Path Length", "Cluster Changes", "Mobility Score"]
            ])),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(row[col], style={"padding": "8px"})
                    for col in ["Token", "Position", "Path Length", "Cluster Changes", "Mobility Score"]
                ]) for row in metrics_data
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
        
        return table
    
    # Callbacks for concept metrics tab
    @app.callback(
        Output("concept-viz-container", "children"),
        [Input("concept-viz-type", "value"),
         Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_concept_visualization(viz_type, selected_layers, cluster_paths_data):
        """Update the concept visualization based on the selected type."""
        if not cluster_paths_data or "cluster_metrics" not in cluster_paths_data:
            return html.Div("No concept metrics available for visualization.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Create appropriate visualization based on selection
        if viz_type == "purity":
            # Create concept purity heatmap
            cluster_metrics = cluster_paths_data["cluster_metrics"]
            
            # Filter to selected layers
            filtered_metrics = {layer: metrics for layer, metrics in cluster_metrics.items() if layer in selected_layers}
            
            fig = create_concept_purity_heatmap(filtered_metrics)
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "stability":
            # Create cluster stability plot
            if "cluster_stability" not in cluster_paths_data:
                return html.Div("No cluster stability data available.")
            
            stability = cluster_paths_data["cluster_stability"]
            
            fig = create_cluster_stability_plot(stability, selected_layers)
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "similarity":
            # Create layer similarity matrix
            if "layer_similarity" not in cluster_paths_data:
                # Calculate layer similarity using centroids
                if "unique_centroids" not in cluster_paths_data or "id_mapping" not in cluster_paths_data:
                    return html.Div("No layer similarity data available.")
                
                # Calculate layer similarity
                centroids = cluster_paths_data["unique_centroids"]
                id_mapping = cluster_paths_data["id_mapping"]
                
                # Group centroids by layer
                layer_centroids = {}
                
                for centroid_id, mapping in id_mapping.items():
                    if centroid_id in centroids:
                        layer_name = mapping["layer_name"]
                        if layer_name not in layer_centroids:
                            layer_centroids[layer_name] = []
                        
                        layer_centroids[layer_name].append(centroids[centroid_id])
                
                # Calculate pairwise similarity between layers
                similarity_matrix = {}
                
                for layer1 in selected_layers:
                    if layer1 not in layer_centroids:
                        continue
                        
                    for layer2 in selected_layers:
                        if layer2 not in layer_centroids:
                            continue
                            
                        if layer1 == layer2:
                            similarity_matrix[(layer1, layer2)] = 1.0
                            continue
                        
                        # Calculate cosine similarity between all centroid pairs
                        similarities = []
                        
                        for centroid1 in layer_centroids[layer1]:
                            for centroid2 in layer_centroids[layer2]:
                                # Convert to numpy arrays if needed
                                c1 = np.array(centroid1)
                                c2 = np.array(centroid2)
                                
                                # Calculate cosine similarity
                                similarity = np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2))
                                similarities.append(similarity)
                        
                        # Use average similarity
                        if similarities:
                            similarity_matrix[(layer1, layer2)] = np.mean(similarities)
                        else:
                            similarity_matrix[(layer1, layer2)] = 0.0
            else:
                # Use provided layer similarity
                similarity_matrix = cluster_paths_data["layer_similarity"]
                
                # Filter to selected layers
                similarity_matrix = {
                    k: v for k, v in similarity_matrix.items() 
                    if k[0] in selected_layers and k[1] in selected_layers
                }
            
            # Create heatmap data
            if not similarity_matrix:
                return html.Div("No layer similarity data available for selected layers.")
                
            # Create z data for heatmap
            layers = sorted(set([k[0] for k in similarity_matrix.keys()]))
            z_data = []
            
            for layer1 in layers:
                row = []
                for layer2 in layers:
                    similarity = similarity_matrix.get((layer1, layer2), 0.0)
                    row.append(similarity)
                z_data.append(row)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                x=layers,
                y=layers,
                colorscale="Viridis",
                colorbar={"title": "Similarity"}
            ))
            
            fig.update_layout(
                title="Layer Similarity Matrix",
                xaxis_title="Layer",
                yaxis_title="Layer"
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
        
        # Default fallback
        return html.Div("Please select a visualization type.")
    
    @app.callback(
        Output("concept-metrics-table", "children"),
        [Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_concept_metrics_table(selected_layers, cluster_paths_data):
        """Update the concept metrics table."""
        if not cluster_paths_data or "cluster_metrics" not in cluster_paths_data:
            return html.Div("No concept metrics available.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Extract metrics for each layer
        cluster_metrics = cluster_paths_data["cluster_metrics"]
        
        # Create metrics data
        metrics_data = []
        
        for layer in selected_layers:
            if layer in cluster_metrics:
                layer_data = cluster_metrics[layer]
                
                metrics_data.append({
                    "Layer": layer,
                    "Purity": f"{layer_data.get('purity', 'N/A'):.4f}" if layer_data.get('purity') is not None else "N/A",
                    "Silhouette": f"{layer_data.get('silhouette', 'N/A'):.4f}" if layer_data.get('silhouette') is not None else "N/A",
                    "Num Clusters": layer_data.get('num_clusters', 'N/A')
                })
        
        if not metrics_data:
            return html.Div("No metrics available for selected layers.")
        
        # Create table
        table = html.Table([
            # Header
            html.Thead(html.Tr([
                html.Th(col, style={"textAlign": "left", "padding": "8px"})
                for col in ["Layer", "Purity", "Silhouette", "Num Clusters"]
            ])),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(row[col], style={"padding": "8px"})
                    for col in ["Layer", "Purity", "Silhouette", "Num Clusters"]
                ]) for row in metrics_data
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
        
        return table
    
    # Callbacks for path-attention correlation tab
    @app.callback(
        Output("correlation-viz-container", "children"),
        [Input("correlation-viz-type", "value"),
         Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_correlation_visualization(viz_type, selected_layers, cluster_paths_data):
        """Update the correlation visualization based on the selected type."""
        if not cluster_paths_data or "token_paths" not in cluster_paths_data or "attention_data" not in cluster_paths_data:
            return html.Div("Both token path and attention data required for correlation analysis.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Calculate correlation metrics
        from visualization.gpt2_attention_correlation import calculate_correlation_metrics
        from visualization.gpt2_token_sankey import extract_token_paths
        from visualization.gpt2_attention_sankey import extract_attention_flow
        
        # Extract paths and attention flow
        token_paths = extract_token_paths(
            activations=cluster_paths_data.get("activations", {}),
            token_metadata=cluster_paths_data.get("token_metadata", {}),
            cluster_labels=cluster_paths_data.get("cluster_labels", {})
        )
        
        attention_flow = extract_attention_flow(
            attention_data=cluster_paths_data["attention_data"],
            token_metadata=cluster_paths_data.get("token_metadata", {})
        )
        
        # Calculate correlation metrics
        correlation_metrics = calculate_correlation_metrics(
            token_paths=token_paths,
            attention_flow=attention_flow,
            cluster_labels=cluster_paths_data.get("cluster_labels", {}),
            layer_names=selected_layers
        )
        
        # Create appropriate visualization based on selection
        if viz_type == "heatmap":
            # Create layer correlation heatmap
            layer_transitions = correlation_metrics.get("layer_transitions", {})
            
            if not layer_transitions:
                return html.Div("No correlation data available for selected layers.")
            
            # Create heatmap data
            layers = selected_layers
            if len(layers) < 2:
                return html.Div("At least two layers required for transition analysis.")
            
            # Create labels for layer transitions
            transition_labels = [f"{layers[i]} → {layers[i+1]}" for i in range(len(layers)-1)]
            correlation_values = []
            
            for i in range(len(layers)-1):
                layer1 = layers[i]
                layer2 = layers[i+1]
                
                # Get correlation for this transition
                transition_key = (layer1, layer2)
                transition_data = layer_transitions.get(transition_key, {})
                
                correlation_values.append(transition_data.get("correlation", 0))
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=[correlation_values],
                x=transition_labels,
                y=["Correlation"],
                colorscale="RdBu",
                zmid=0,
                colorbar={"title": "Correlation"}
            ))
            
            fig.update_layout(
                title="Path-Attention Correlation by Layer Transition",
                xaxis_title="Layer Transition",
                yaxis_title="Metric",
                height=300
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "token_chart":
            # Create token correlation bar chart
            token_correlations = correlation_metrics.get("token_correlations", {})
            
            if not token_correlations:
                return html.Div("No token correlation data available.")
            
            # Sort tokens by correlation value
            sorted_tokens = sorted(
                [(token, stats.get("avg_correlation", 0)) 
                 for token, stats in token_correlations.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top 20 tokens
            top_tokens = sorted_tokens[:20]
            
            # Create bar chart
            fig = go.Figure(data=go.Bar(
                x=[token for token, _ in top_tokens],
                y=[corr for _, corr in top_tokens],
                marker_color='rgb(55, 83, 109)'
            ))
            
            fig.update_layout(
                title="Token Path-Attention Correlation",
                xaxis_title="Token",
                yaxis_title="Correlation",
                xaxis={"tickangle": 45}
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
            
        elif viz_type == "combined":
            # Create combined view with both visualizations
            # First, create layer correlation heatmap
            layer_transitions = correlation_metrics.get("layer_transitions", {})
            token_correlations = correlation_metrics.get("token_correlations", {})
            
            if not layer_transitions or not token_correlations:
                return html.Div("No correlation data available for selected layers.")
            
            # Create combined figure with subplots
            from plotly.subplots import make_subplots
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Layer Transition Correlation", "Top Token Correlation"))
            
            # Add layer transition heatmap
            layers = selected_layers
            if len(layers) >= 2:
                # Create labels for layer transitions
                transition_labels = [f"{layers[i]} → {layers[i+1]}" for i in range(len(layers)-1)]
                correlation_values = []
                
                for i in range(len(layers)-1):
                    layer1 = layers[i]
                    layer2 = layers[i+1]
                    
                    # Get correlation for this transition
                    transition_key = (layer1, layer2)
                    transition_data = layer_transitions.get(transition_key, {})
                    
                    correlation_values.append(transition_data.get("correlation", 0))
                
                fig.add_trace(
                    go.Heatmap(
                        z=[correlation_values],
                        x=transition_labels,
                        y=["Correlation"],
                        colorscale="RdBu",
                        zmid=0,
                        colorbar={"title": "Correlation"}
                    ),
                    row=1, col=1
                )
            
            # Add token correlation bar chart
            # Sort tokens by correlation value
            sorted_tokens = sorted(
                [(token, stats.get("avg_correlation", 0)) 
                 for token, stats in token_correlations.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]
            
            fig.add_trace(
                go.Bar(
                    x=[token for token, _ in top_tokens],
                    y=[corr for _, corr in top_tokens],
                    marker_color='rgb(55, 83, 109)'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                showlegend=False
            )
            
            return dcc.Graph(figure=fig, style={"height": "100%"})
        
        # Default fallback
        return html.Div("Please select a visualization type.")
    
    @app.callback(
        Output("correlation-metrics-summary", "children"),
        [Input("gpt2-selected-layers", "data"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_correlation_metrics_summary(selected_layers, cluster_paths_data):
        """Update the correlation metrics summary."""
        if not cluster_paths_data or "token_paths" not in cluster_paths_data or "attention_data" not in cluster_paths_data:
            return html.Div("Both token path and attention data required for correlation metrics.")
        
        # Filter to selected layers
        if not selected_layers:
            selected_layers = cluster_paths_data.get("layers", [])
        
        # Calculate correlation metrics
        from visualization.gpt2_attention_correlation import calculate_correlation_metrics
        from visualization.gpt2_token_sankey import extract_token_paths
        from visualization.gpt2_attention_sankey import extract_attention_flow
        
        # Extract paths and attention flow
        token_paths = extract_token_paths(
            activations=cluster_paths_data.get("activations", {}),
            token_metadata=cluster_paths_data.get("token_metadata", {}),
            cluster_labels=cluster_paths_data.get("cluster_labels", {})
        )
        
        attention_flow = extract_attention_flow(
            attention_data=cluster_paths_data["attention_data"],
            token_metadata=cluster_paths_data.get("token_metadata", {})
        )
        
        # Calculate correlation metrics
        correlation_metrics = calculate_correlation_metrics(
            token_paths=token_paths,
            attention_flow=attention_flow,
            cluster_labels=cluster_paths_data.get("cluster_labels", {}),
            layer_names=selected_layers
        )
        
        # Extract global metrics
        global_metrics = correlation_metrics.get("global_metrics", {})
        
        # Create summary table
        summary = html.Div([
            html.Table([
                html.Tr([
                    html.Td("Average Path-Attention Correlation:", style={"fontWeight": "bold", "paddingRight": "15px"}),
                    html.Td(f"{global_metrics.get('avg_path_attention_correlation', 0):.4f}")
                ]),
                html.Tr([
                    html.Td("Attention Follows Paths Score:", style={"fontWeight": "bold", "paddingRight": "15px"}),
                    html.Td(f"{global_metrics.get('attention_follows_paths', 0):.4f}")
                ]),
                html.Tr([
                    html.Td("Paths Follow Attention Score:", style={"fontWeight": "bold", "paddingRight": "15px"}),
                    html.Td(f"{global_metrics.get('paths_follow_attention', 0):.4f}")
                ])
            ], style={"marginBottom": "15px"})
        ])
        
        return summary
    
    # Callbacks for LLM analysis tab
    @app.callback(
        Output("llm-analysis-viz-container", "children"),
        [Input("llm-analysis-viz-type", "value"),
         Input("llm-analysis-store", "data")]
    )
    def update_llm_analysis_visualization(analysis_type, llm_analysis):
        """Update the LLM analysis visualization."""
        if not llm_analysis:
            return html.Div([
                html.P("No LLM analysis available."),
                html.P("Generate LLM analysis using the LLM Analysis tab first.")
            ])
        
        # Display appropriate analysis based on selection
        if analysis_type == "attention":
            # Display attention pattern narratives
            if "attention_patterns" not in llm_analysis:
                return html.Div([
                    html.P("No attention pattern analysis available."),
                    html.P("Generate LLM analysis with 'GPT-2 Attention Patterns' selected.")
                ])
            
            # Extract attention patterns
            attention_patterns = llm_analysis["attention_patterns"]
            provider = llm_analysis.get("provider", "unknown")
            
            # Create cards for each layer
            attention_cards = []
            
            for layer_name, pattern_analysis in attention_patterns.items():
                # Create card for this layer's attention pattern
                card = html.Div([
                    # Header with layer name
                    html.Div([
                        html.Span(f"Layer: {layer_name}", style={"fontWeight": "bold", "fontSize": "16px"})
                    ]),
                    
                    # Analysis text
                    html.Div([
                        html.Div("Analysis:", style={"fontWeight": "bold", "marginTop": "10px"}),
                        html.Div(pattern_analysis.get('narrative', 'No analysis available'), 
                                 style={"marginTop": "5px", "padding": "10px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
                    ])
                ], style={
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "marginBottom": "15px",
                    "backgroundColor": "white"
                })
                
                attention_cards.append(card)
            
            if not attention_cards:
                return html.Div("No attention pattern narratives available.")
            
            # Create header
            header = html.Div([
                html.H5(f"Attention Pattern Analysis Generated by {provider.title()}"),
                html.P(f"Analysis of attention patterns across {len(attention_patterns)} layers")
            ])
            
            return html.Div([header, html.Div(attention_cards)])
            
        elif analysis_type == "movement":
            # Display token movement narratives
            if "token_movement" not in llm_analysis:
                return html.Div([
                    html.P("No token movement analysis available."),
                    html.P("Generate LLM analysis with 'GPT-2 Token Movement' selected.")
                ])
            
            # Extract token movement
            token_movement = llm_analysis["token_movement"]
            provider = llm_analysis.get("provider", "unknown")
            
            # Create cards for each token
            token_cards = []
            
            for token_id, movement_analysis in token_movement.items():
                # Get token text
                token_text = movement_analysis.get('token_text', f"Token {token_id}")
                
                # Create card for this token
                card = html.Div([
                    # Header with token info
                    html.Div([
                        html.Span(f"Token: {token_text}", style={"fontWeight": "bold", "fontSize": "16px"})
                    ]),
                    
                    # Analysis text
                    html.Div([
                        html.Div("Analysis:", style={"fontWeight": "bold", "marginTop": "10px"}),
                        html.Div(movement_analysis.get('narrative', 'No analysis available'), 
                                 style={"marginTop": "5px", "padding": "10px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
                    ])
                ], style={
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "marginBottom": "15px",
                    "backgroundColor": "white"
                })
                
                token_cards.append(card)
            
            if not token_cards:
                return html.Div("No token movement narratives available.")
            
            # Create header
            header = html.Div([
                html.H5(f"Token Movement Analysis Generated by {provider.title()}"),
                html.P(f"Analysis of token movements for {len(token_movement)} tokens")
            ])
            
            return html.Div([header, html.Div(token_cards)])
            
        elif analysis_type == "concepts":
            # Display concept purity analysis
            if "concept_purity" not in llm_analysis:
                return html.Div([
                    html.P("No concept purity analysis available."),
                    html.P("Generate LLM analysis with 'GPT-2 Concept Purity' selected.")
                ])
            
            # Extract concept purity
            concept_purity = llm_analysis["concept_purity"]
            provider = llm_analysis.get("provider", "unknown")
            
            # Create card for concept analysis
            card = html.Div([
                # Analysis text
                html.Div([
                    html.Div("Analysis:", style={"fontWeight": "bold", "marginTop": "10px"}),
                    html.Div(concept_purity.get('narrative', 'No analysis available'), 
                             style={"marginTop": "5px", "padding": "10px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
                ])
            ], style={
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "padding": "15px",
                "marginBottom": "15px",
                "backgroundColor": "white"
            })
            
            # Create header
            header = html.Div([
                html.H5(f"Concept Purity Analysis Generated by {provider.title()}"),
                html.P("Analysis of concept organization across layers")
            ])
            
            return html.Div([header, card])
            
        elif analysis_type == "correlation":
            # Display path-attention correlation analysis
            if "token_path_comparison" not in llm_analysis:
                return html.Div([
                    html.P("No path-attention correlation analysis available."),
                    html.P("Generate LLM analysis with 'Token Path Comparison' selected.")
                ])
            
            # Extract correlation analysis
            correlation = llm_analysis["token_path_comparison"]
            provider = llm_analysis.get("provider", "unknown")
            
            # Create card for correlation analysis
            correlation_cards = []
            
            for correlation_type, analysis in correlation.items():
                card = html.Div([
                    # Header
                    html.Div([
                        html.Span(f"Analysis Type: {correlation_type}", style={"fontWeight": "bold", "fontSize": "16px"})
                    ]),
                    
                    # Analysis text
                    html.Div([
                        html.Div("Analysis:", style={"fontWeight": "bold", "marginTop": "10px"}),
                        html.Div(analysis, style={"marginTop": "5px", "padding": "10px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
                    ])
                ], style={
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                    "padding": "15px",
                    "marginBottom": "15px",
                    "backgroundColor": "white"
                })
                
                correlation_cards.append(card)
            
            if not correlation_cards:
                return html.Div("No correlation analysis available.")
            
            # Create header
            header = html.Div([
                html.H5(f"Path-Attention Correlation Analysis Generated by {provider.title()}"),
                html.P("Analysis of correlations between token paths and attention patterns")
            ])
            
            return html.Div([header, html.Div(correlation_cards)])
        
        # Default fallback
        return html.Div("Please select an analysis type.")
    
    # Callbacks for the Linked Token-Attention View tab
    @app.callback(
        [Output("linked-token-selection", "options"),
         Output("linked-token-selection", "value"),
         Output("linked-layer-selection", "options"),
         Output("linked-layer-selection", "value")],
        [Input("cluster-paths-data-store", "data")]
    )
    def update_linked_view_options(cluster_paths_data):
        """Update the token and layer selection options for the linked view."""
        if not cluster_paths_data:
            return [], [], [], []
        
        # Extract token information
        token_metadata = cluster_paths_data.get("token_metadata", {})
        tokens = token_metadata.get("tokens", [])
        positions = token_metadata.get("positions", [])
        
        # Create token options
        token_options = []
        if tokens and len(tokens) == len(positions):
            token_options = [
                {"label": f"'{token}' (pos: {position})", "value": str(i)}
                for i, (token, position) in enumerate(zip(tokens, positions))
            ]
        
        # Extract layer information
        layers = cluster_paths_data.get("layers", [])
        
        # Create layer options
        layer_options = [{"label": layer, "value": layer} for layer in layers]
        
        # Default to first 3 layers if available
        default_layers = layers[:min(3, len(layers))]
        
        return token_options, [], layer_options, default_layers
    
    @app.callback(
        [Output("linked-token-path-container", "children"),
         Output("linked-attention-container", "children")],
        [Input("linked-token-selection", "value"),
         Input("linked-layer-selection", "value"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_linked_visualizations(selected_tokens, selected_layers, cluster_paths_data):
        """Update the linked token path and attention flow visualizations."""
        if not cluster_paths_data or not selected_layers:
            return html.Div("Select layers to visualize."), html.Div("Select layers to visualize.")
        
        # Extract data
        token_paths = cluster_paths_data.get("token_paths", {})
        attention_data = cluster_paths_data.get("attention_data", {})
        token_metadata = cluster_paths_data.get("token_metadata", {})
        
        if not token_paths or not attention_data:
            return html.Div("Token path or attention data not available."), html.Div("Token path or attention data not available.")
        
        # Convert selected tokens to integers
        token_indices = []
        if selected_tokens:
            token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
        
        # Create token path visualization
        try:
            from visualization.gpt2_token_sankey import create_token_sankey_diagram
            
            token_path_fig = create_token_sankey_diagram(
                token_paths=token_paths,
                selected_tokens=token_indices if token_indices else None,
                selected_layers=selected_layers,
                highlight_selected=True  # Explicitly highlight selected tokens
            )
            
            token_path_graph = dcc.Graph(
                id="linked-token-path-graph",
                figure=token_path_fig,
                style={"height": "100%"},
                config={"displayModeBar": True}
            )
        except Exception as e:
            token_path_graph = html.Div(f"Error creating token path visualization: {str(e)}")
        
        # Create attention flow visualization
        try:
            from visualization.gpt2_attention_sankey import create_attention_sankey_diagram
            
            attention_fig = create_attention_sankey_diagram(
                attention_data=attention_data,
                token_metadata=token_metadata,
                selected_layers=selected_layers,
                highlight_tokens=token_indices if token_indices else None  # Highlight same tokens
            )
            
            attention_graph = dcc.Graph(
                id="linked-attention-graph",
                figure=attention_fig,
                style={"height": "100%"},
                config={"displayModeBar": True}
            )
        except Exception as e:
            attention_graph = html.Div(f"Error creating attention flow visualization: {str(e)}")
        
        return token_path_graph, attention_graph
    
    @app.callback(
        Output("linked-token-metrics", "children"),
        [Input("linked-token-selection", "value"),
         Input("cluster-paths-data-store", "data")]
    )
    def update_linked_token_metrics(selected_tokens, cluster_paths_data):
        """Update the metrics for selected tokens in the linked view."""
        if not cluster_paths_data or not selected_tokens:
            return html.Div("Select tokens to view their metrics.")
        
        # Extract data
        token_paths = cluster_paths_data.get("token_paths", {})
        attention_data = cluster_paths_data.get("attention_data", {})
        token_metadata = cluster_paths_data.get("token_metadata", {})
        
        if not token_paths:
            return html.Div("Token path data not available.")
        
        # Get token text information
        tokens = token_metadata.get("tokens", [])
        
        # Convert selected tokens to integers
        token_indices = [int(token_id) for token_id in selected_tokens if token_id.isdigit()]
        
        # Create metrics table for selected tokens
        metric_rows = []
        
        for token_id in token_indices:
            if token_id < len(tokens) and str(token_id) in token_paths:
                token_text = tokens[token_id]
                path_data = token_paths[str(token_id)]
                
                # Calculate additional correlation metrics if available
                attention_corr = "N/A"
                if "token_correlations" in cluster_paths_data.get("similarity", {}):
                    token_corrs = cluster_paths_data["similarity"]["token_correlations"]
                    if token_text in token_corrs:
                        attention_corr = f"{token_corrs[token_text]:.4f}"
                
                metric_rows.append({
                    "Token": f"'{token_text}'",
                    "Position": path_data.get("position", "N/A"),
                    "Path Length": f"{path_data.get('path_length', 'N/A'):.4f}" if path_data.get('path_length') is not None else "N/A",
                    "Cluster Changes": path_data.get("cluster_changes", "N/A"),
                    "Mobility Score": f"{path_data.get('mobility_score', 'N/A'):.4f}" if path_data.get('mobility_score') is not None else "N/A",
                    "Attention Correlation": attention_corr
                })
        
        if not metric_rows:
            return html.Div("No metrics available for selected tokens.")
        
        # Create metrics table
        table = html.Table([
            # Header
            html.Thead(html.Tr([
                html.Th(col, style={"textAlign": "left", "padding": "8px"})
                for col in ["Token", "Position", "Path Length", "Cluster Changes", "Mobility Score", "Attention Correlation"]
            ])),
            # Body
            html.Tbody([
                html.Tr([
                    html.Td(row[col], style={"padding": "8px"})
                    for col in ["Token", "Position", "Path Length", "Cluster Changes", "Mobility Score", "Attention Correlation"]
                ]) for row in metric_rows
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
        
        return table
    
    @app.callback(
        Output("generate-gpt2-llm-button", "n_clicks"),
        [Input("generate-gpt2-llm-button", "n_clicks"),
         Input("llm-analysis-viz-type", "value")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value")]
    )
    def generate_gpt2_llm_analysis(n_clicks, analysis_type, dataset, seed):
        """Generate LLM analysis for the selected GPT-2 analysis type."""
        if n_clicks is None or n_clicks == 0:
            return 0
        
        # Map analysis type to LLM analysis type
        analysis_map = {
            "attention": "attention_patterns",
            "movement": "token_movement",
            "concepts": "concept_purity",
            "correlation": "token_path_comparison"
        }
        
        llm_analysis_type = analysis_map.get(analysis_type)
        
        if llm_analysis_type:
            # Use JavaScript to trigger the LLM analysis generation
            app.clientside_callback(
                """
                function(n_clicks, analysis_type, dataset, seed) {
                    // Set the LLM analysis type
                    document.getElementById("llm-analysis-type").value = [analysis_type];
                    
                    // Click the generate button
                    document.getElementById("generate-llm-button").click();
                    
                    return n_clicks;
                }
                """,
                Output("generate-gpt2-llm-button", "n_clicks"),
                [Input("generate-gpt2-llm-button", "n_clicks")],
                [State("llm-analysis-viz-type", "value"),
                 State("dataset-dropdown", "value"),
                 State("seed-dropdown", "value")]
            )
        
        return 0