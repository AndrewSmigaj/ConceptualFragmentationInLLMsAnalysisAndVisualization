"""
LLM Integration Tab for the Neural Network Trajectories Dashboard.

This module provides the layout and callbacks for the LLM integration tab,
which allows users to generate cluster labels and path narratives using LLMs.
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import importlib
import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional

# Add a function to find and load existing LLM results files
def find_existing_llm_results(dataset: str, seed: int) -> Optional[Dict[str, Any]]:
    """Find and load existing LLM results for the given dataset and seed."""
    # Define paths to check
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "llm")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "llm")
    
    # Check results directory first
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            if filename.startswith(f"{dataset}_seed_{seed}") and filename.endswith(".json"):
                results_path = os.path.join(results_dir, filename)
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    print(f"Loaded existing LLM results from {results_path}")
                    return results
                except Exception as e:
                    print(f"Error loading LLM results from {results_path}: {e}")
    
    # Then check data directory
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.startswith(f"{dataset}_seed_{seed}") and filename.endswith(".json"):
                results_path = os.path.join(data_dir, filename)
                try:
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    print(f"Loaded existing LLM results from {results_path}")
                    return results
                except Exception as e:
                    print(f"Error loading LLM results from {results_path}: {e}")
    
    # If we reach here, no results were found
    print(f"No existing LLM results found for {dataset} seed {seed}")
    return None

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import LLM analysis module
from concept_fragmentation.llm.analysis import ClusterAnalysis
from concept_fragmentation.llm.factory import LLMClientFactory


def create_llm_tab():
    """Create the layout for the LLM Integration tab."""
    return dcc.Tab(label="LLM Analysis", children=[
        html.Div([
            html.H4("LLM-Enhanced Analysis", style={"textAlign": "center"}),
            
            # LLM Provider Selection
            html.Div([
                html.Label("LLM Provider:"),
                dcc.RadioItems(
                    id="llm-provider-radio",
                    options=[
                        {"label": "Grok (Meta)", "value": "grok"},
                        {"label": "Claude (Anthropic)", "value": "claude"},
                        {"label": "GPT (OpenAI)", "value": "openai"},
                        {"label": "Gemini (Google)", "value": "gemini"}
                    ],
                    value="grok",
                    inline=True
                ),
            ], style={"width": "100%", "padding": "10px"}),
            
            # Analysis Type Selection
            html.Div([
                html.Div([
                    html.Label("Analysis Type:"),
                    dcc.Checklist(
                        id="llm-analysis-type",
                        options=[
                            {"label": "Cluster Labeling", "value": "cluster_labels"},
                            {"label": "Path Narratives", "value": "path_narratives"}
                        ],
                        value=["cluster_labels"],
                        inline=True
                    ),
                ], style={"width": "50%", "display": "inline-block", "padding": "10px"}),
                
                html.Div([
                    html.Label("Path Selection:"),
                    dcc.RadioItems(
                        id="llm-path-selection",
                        options=[
                            {"label": "All Paths", "value": "all"},
                            {"label": "Top 10 Most Common", "value": "top_common"},
                            {"label": "Top 10 Most Fragmented", "value": "top_fragmented"},
                            {"label": "Top 10 Least Fragmented", "value": "bottom_fragmented"}
                        ],
                        value="top_common",
                        inline=True
                    ),
                ], style={"width": "50%", "display": "inline-block", "padding": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
            
            # Generate Button
            html.Div([
                html.Button(
                    "Generate LLM Analysis", 
                    id="generate-llm-button", 
                    n_clicks=0,
                    style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px", "margin": "10px"}
                ),
            ], style={"textAlign": "center"}),
            
            # Status Area
            html.Div(id="llm-status-area", style={"padding": "10px", "margin": "10px"}),
            
            # Analysis Results Tabs
            dcc.Tabs([
                # Cluster Labels Tab
                dcc.Tab(label="Cluster Labels", children=[
                    dcc.Loading(
                        id="loading-cluster-labels",
                        type="circle",
                        children=[
                            html.Div(id="cluster-labels-container", style={"padding": "15px"})
                        ]
                    )
                ]),
                
                # Path Narratives Tab
                dcc.Tab(label="Path Narratives", children=[
                    dcc.Loading(
                        id="loading-path-narratives",
                        type="circle",
                        children=[
                            html.Div(id="path-narratives-container", style={"padding": "15px"})
                        ]
                    )
                ]),
            ]),
            
            # Store for LLM analysis results
            dcc.Store(id="llm-analysis-store"),
        ])
    ])


def register_llm_callbacks(app):
    """Register callbacks for the LLM Integration tab."""
    
    @app.callback(
        [Output("llm-status-area", "children"),
         Output("llm-analysis-store", "data")],
        [Input("generate-llm-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("llm-provider-radio", "value"),
         State("llm-analysis-type", "value"),
         State("llm-path-selection", "value"),
         State("llm-analysis-store", "data")]
    )
    def generate_llm_analysis(n_clicks, dataset, seed, provider, analysis_types, path_selection, current_analysis):
        """Generate LLM analysis for the current dataset."""
        if n_clicks == 0:
            # Try to load existing results even if button hasn't been clicked
            existing_results = find_existing_llm_results(dataset, seed)
            if existing_results:
                return html.Div([
                    html.P("Loaded existing LLM analysis from saved results.", style={"color": "green"}),
                    html.P(f"Provider: {existing_results.get('provider', 'unknown')}"),
                    html.P(f"Model: {existing_results.get('model', 'unknown')}"),
                    html.P(f"Found {len(existing_results.get('cluster_labels', {}))} cluster labels and {len(existing_results.get('path_narratives', {}))} path narratives.")
                ]), existing_results
            return html.Div("Click 'Generate LLM Analysis' to start analysis."), None
        
        if not dataset or not seed:
            return html.Div("Please select a dataset and seed."), None
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Try to load cluster paths data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return html.Div(f"Error: {error_msg}", style={"color": "red"}), None
            
            # Extract centroids
            centroids = {}
            id_mapping = {}
            if "id_mapping" in cluster_paths_data:
                id_mapping = cluster_paths_data["id_mapping"]
                
                if "unique_centroids" in cluster_paths_data:
                    # Create centroids dict with proper cluster IDs
                    for unique_id, mapping in id_mapping.items():
                        if unique_id in cluster_paths_data["unique_centroids"]:
                            layer_name = mapping["layer_name"]
                            original_id = mapping["original_id"]
                            cluster_id = f"{layer_name}C{original_id}"
                            centroids[cluster_id] = np.array(cluster_paths_data["unique_centroids"][unique_id])
            
            # Initialize the analyzer
            analyzer = ClusterAnalysis(provider=provider)
            
            # Initialize results dict
            results = current_analysis or {}
            
            # If we haven't generated cluster labels or they're requested again
            if "cluster_labels" in analysis_types and (
                "cluster_labels" not in results or 
                results.get("provider") != provider
            ):
                if centroids:
                    # Generate cluster labels
                    status_div = html.Div("Generating cluster labels...", id="llm-status-text")
                    dash.callback_context.record_timing("llm-status-update", "Generating cluster labels...")
                    
                    # Grab current labels if available for augmentation
                    current_labels = results.get("cluster_labels", {})
                    
                    # Generate labels for clusters
                    try:
                        cluster_labels = analyzer.label_clusters_sync(centroids)
                        
                        # Store the results
                        results["cluster_labels"] = cluster_labels
                        results["provider"] = provider
                        results["model"] = analyzer.model
                        
                        # Update status
                        status_div = html.Div(f"Generated {len(cluster_labels)} cluster labels successfully.", style={"color": "green"})
                    except Exception as e:
                        status_div = html.Div(f"Error generating cluster labels: {str(e)}", style={"color": "red"})
                else:
                    status_div = html.Div("No centroids found in data.", style={"color": "orange"})
            else:
                status_div = html.Div("Using existing cluster labels.")
            
            # If path narratives are requested
            if "path_narratives" in analysis_types:
                # Extract paths based on selection
                paths = {}
                if "path_archetypes" in cluster_paths_data:
                    archetypes = cluster_paths_data["path_archetypes"]
                    
                    # Filter archetypes based on selection
                    if path_selection == "top_common":
                        # Sort by count (descending)
                        sorted_archetypes = sorted(enumerate(archetypes), key=lambda x: x[1].get("count", 0), reverse=True)
                        selected_archetypes = sorted_archetypes[:10]
                    elif path_selection == "top_fragmented":
                        # Check if fragmentation scores exist
                        if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"]:
                            frag_scores = cluster_paths_data["similarity"]["fragmentation_scores"].get("scores", [])
                            # Sort by fragmentation score (descending)
                            sorted_indices = sorted(range(len(frag_scores)), key=lambda i: frag_scores[i] if i < len(frag_scores) else 0, reverse=True)
                            selected_archetypes = [(i, archetypes[i]) for i in sorted_indices[:10] if i < len(archetypes)]
                        else:
                            selected_archetypes = list(enumerate(archetypes))[:10]
                    elif path_selection == "bottom_fragmented":
                        # Check if fragmentation scores exist
                        if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"]:
                            frag_scores = cluster_paths_data["similarity"]["fragmentation_scores"].get("scores", [])
                            # Sort by fragmentation score (ascending)
                            sorted_indices = sorted(range(len(frag_scores)), key=lambda i: frag_scores[i] if i < len(frag_scores) else 1)
                            selected_archetypes = [(i, archetypes[i]) for i in sorted_indices[:10] if i < len(archetypes)]
                        else:
                            selected_archetypes = list(enumerate(archetypes))[:10]
                    else:  # All paths
                        selected_archetypes = list(enumerate(archetypes))
                    
                    # Extract paths from selected archetypes
                    for i, archetype in selected_archetypes:
                        if "path" in archetype:
                            path_str = archetype["path"]
                            path_parts = path_str.split("→")
                            
                            # Get layer information
                            layers = cluster_paths_data.get("layers", [])
                            
                            # Create the path with full cluster IDs
                            path = []
                            for j, part in enumerate(path_parts):
                                if j < len(layers):
                                    layer = layers[j]
                                    cluster_id = f"{layer}C{part}"
                                    path.append(cluster_id)
                                else:
                                    # Fallback if layer information is missing
                                    path.append(f"L{j}C{part}")
                            
                            paths[i] = path
                
                # If we have paths and cluster labels, generate narratives
                if paths and "cluster_labels" in results:
                    # Get convergent points
                    convergent_points = {}
                    if "similarity" in cluster_paths_data and "convergent_paths" in cluster_paths_data["similarity"]:
                        conv_paths = cluster_paths_data["similarity"]["convergent_paths"]
                        layers = cluster_paths_data.get("layers", [])
                        
                        for path_id_str, convergences in conv_paths.items():
                            try:
                                path_id = int(path_id_str)
                                if path_id in paths:
                                    path_convergences = []
                                    
                                    for conv in convergences:
                                        early_layer = conv.get("early_layer", 0)
                                        late_layer = conv.get("late_layer", 0)
                                        early_cluster = conv.get("early_cluster", "?")
                                        late_cluster = conv.get("late_cluster", "?")
                                        similarity = conv.get("similarity", 0)
                                        
                                        # Create cluster IDs
                                        early_layer_name = layers[early_layer] if early_layer < len(layers) else f"L{early_layer}"
                                        late_layer_name = layers[late_layer] if late_layer < len(layers) else f"L{late_layer}"
                                        
                                        early_id = f"{early_layer_name}C{early_cluster}"
                                        late_id = f"{late_layer_name}C{late_cluster}"
                                        
                                        path_convergences.append((early_id, late_id, similarity))
                                    
                                    convergent_points[path_id] = path_convergences
                            except ValueError:
                                continue
                    
                    # Get fragmentation scores
                    fragmentation_scores = {}
                    if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"]:
                        frag_data = cluster_paths_data["similarity"]["fragmentation_scores"]
                        
                        if "scores" in frag_data:
                            scores = frag_data["scores"]
                            
                            # Create a dictionary mapping path IDs to scores
                            for i, score in enumerate(scores):
                                if i in paths:
                                    fragmentation_scores[i] = score
                    
                    # Get demographic info
                    demographic_info = {}
                    for i, archetype in enumerate(cluster_paths_data.get("path_archetypes", [])):
                        if i in paths and "demo_stats" in archetype:
                            demographic_info[i] = archetype["demo_stats"]
                            
                            # Add any other interesting statistics
                            if "survived_rate" in archetype:
                                if i not in demographic_info:
                                    demographic_info[i] = {}
                                demographic_info[i]["survival_rate"] = archetype["survived_rate"]
                    
                    # Update status
                    status_div = html.Div([
                        status_div,
                        html.Div(f"Generating narratives for {len(paths)} paths...")
                    ])
                    dash.callback_context.record_timing("llm-status-update", "Generating path narratives...")
                    
                    # Generate narratives
                    try:
                        path_narratives = analyzer.generate_path_narratives_sync(
                            paths,
                            results["cluster_labels"],
                            centroids,
                            convergent_points,
                            fragmentation_scores,
                            demographic_info
                        )
                        
                        # Store the results
                        results["path_narratives"] = path_narratives
                        results["provider"] = provider
                        results["model"] = analyzer.model
                        
                        # Update status
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Generated {len(path_narratives)} path narratives successfully.", style={"color": "green"})
                        ])
                    except Exception as e:
                        import traceback
                        error_msg = f"Error generating path narratives: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Error generating path narratives: {str(e)}", style={"color": "red"})
                        ])
                else:
                    status_div = html.Div([
                        status_div,
                        html.Div("No paths or cluster labels found for narrative generation.", style={"color": "orange"})
                    ])
            
            return status_div, results
        except Exception as e:
            import traceback
            error_msg = f"Error in LLM analysis: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return html.Div(f"Error: {str(e)}", style={"color": "red"}), None
    
    @app.callback(
        Output("cluster-labels-container", "children"),
        [Input("llm-analysis-store", "data")]
    )
    def update_cluster_labels(analysis_data):
        """Update the cluster labels table."""
        if not analysis_data or "cluster_labels" not in analysis_data:
            return html.Div("No cluster labels available. Generate LLM analysis first.")
        
        # Extract cluster labels
        cluster_labels = analysis_data["cluster_labels"]
        provider = analysis_data.get("provider", "unknown")
        model = analysis_data.get("model", "unknown")
        
        # Convert to DataFrame for table
        rows = []
        for cluster_id, label in cluster_labels.items():
            # Parse layer and cluster number from cluster ID
            layer = "Unknown"
            cluster = "Unknown"
            if cluster_id.startswith("layer") or cluster_id.startswith("L"):
                parts = cluster_id.split("C")
                if len(parts) == 2:
                    layer = parts[0]
                    cluster = parts[1]
            
            rows.append({
                "Cluster ID": cluster_id,
                "Layer": layer,
                "Cluster": cluster,
                "Label": label
            })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Check if sorting columns exist before sorting
        if 'Layer' in df.columns and 'Cluster' in df.columns:
            df = df.sort_values(by=["Layer", "Cluster"])
        elif 'Cluster ID' in df.columns:
            # Fall back to sorting just by Cluster ID if Layer column is missing
            df = df.sort_values(by=["Cluster ID"])
        
        # Create table
        table = dash_table.DataTable(
            id='cluster-labels-table',
            columns=[{"name": col, "id": col} for col in df.columns],
            data=df.to_dict('records'),
            sort_action="native",
            filter_action="native",
            page_size=20,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '8px',
                'minWidth': '100px',
                'width': '150px',
                'maxWidth': '300px',
                'whiteSpace': 'normal'
            },
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ]
        )
        
        # Create header with model info
        header = html.Div([
            html.H5(f"Cluster Labels Generated by {provider.title()} ({model})"),
            html.P(f"Total clusters labeled: {len(cluster_labels)}")
        ])
        
        return html.Div([header, table])
    
    @app.callback(
        Output("path-narratives-container", "children"),
        [Input("llm-analysis-store", "data")]
    )
    def update_path_narratives(analysis_data):
        """Update the path narratives."""
        if not analysis_data or "path_narratives" not in analysis_data:
            return html.Div("No path narratives available. Generate LLM analysis first.")
        
        # Extract path narratives
        path_narratives = analysis_data["path_narratives"]
        cluster_labels = analysis_data.get("cluster_labels", {})
        provider = analysis_data.get("provider", "unknown")
        model = analysis_data.get("model", "unknown")
        
        # Create the narratives view
        narrative_cards = []
        
        # Import the dashboard module dynamically to avoid circular imports
        dash_app = importlib.import_module("visualization.dash_app")
        
        # Load cluster paths data to get path information
        dataset = dash_app.current_dataset
        seed = dash_app.current_seed
        cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
        
        # Get archetypes for path info
        archetypes = cluster_paths_data.get("path_archetypes", [])
        
        # Get fragmentation scores if available
        fragmentation_scores = {}
        if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"]:
            frag_data = cluster_paths_data["similarity"]["fragmentation_scores"]
            if "scores" in frag_data:
                fragmentation_scores = {i: score for i, score in enumerate(frag_data["scores"])}
        
        # Sort path IDs for consistent display
        sorted_path_ids = sorted(path_narratives.keys())
        
        for path_id in sorted_path_ids:
            narrative = path_narratives[path_id]
            
            # Get path structure if available
            path_str = ""
            if path_id < len(archetypes) and "path" in archetypes[path_id]:
                path_str = archetypes[path_id]["path"]
            
            # Get count if available
            count = 0
            if path_id < len(archetypes) and "count" in archetypes[path_id]:
                count = archetypes[path_id]["count"]
            
            # Get fragmentation score if available
            frag_score = fragmentation_scores.get(path_id, None)
            
            # Create card for this narrative
            card = html.Div([
                # Header with path ID and stats
                html.Div([
                    html.Span(f"Path {path_id}", style={"fontWeight": "bold", "fontSize": "18px"}),
                    html.Span(f" ({count} samples)", style={"color": "#666", "marginLeft": "10px"})
                ]),
                
                # Path structure with labels
                html.Div([
                    html.Span("Path: ", style={"fontWeight": "bold"}),
                    html.Span(path_str)
                ], style={"marginTop": "5px"}),
                
                # Fragmentation score if available
                html.Div([
                    html.Span("Fragmentation: ", style={"fontWeight": "bold"}),
                    html.Span(f"{frag_score:.3f}" if frag_score is not None else "N/A",
                             style={
                                "color": "red" if frag_score and frag_score > 0.7 else 
                                         "green" if frag_score and frag_score < 0.3 else 
                                         "orange"
                             })
                ], style={"marginTop": "5px"}) if frag_score is not None else None,
                
                # Narrative text
                html.Div([
                    html.Div("Narrative:", style={"fontWeight": "bold", "marginTop": "10px"}),
                    html.Div(narrative, style={"marginTop": "5px", "padding": "10px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
                ])
            ], style={
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "padding": "15px",
                "marginBottom": "15px",
                "backgroundColor": "white",
                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
            })
            
            narrative_cards.append(card)
        
        # Create header with model info
        header = html.Div([
            html.H5(f"Path Narratives Generated by {provider.title()} ({model})"),
            html.P(f"Total paths analyzed: {len(path_narratives)}")
        ])
        
        return html.Div([header, html.Div(narrative_cards)])
        

def check_provider_availability():
    """Check which LLM providers are available."""
    available_providers = []
    
    # Check for API keys
    try:
        from concept_fragmentation.llm.api_keys import (
            OPENAI_KEY,
            OPENAI_API_BASE,
            XAI_API_KEY,
            GEMINI_API_KEY
        )
        
        if XAI_API_KEY:
            available_providers.append("grok")
        
        if OPENAI_KEY:
            available_providers.append("openai")
        
        # Claude typically uses ANTHROPIC_API_KEY - check environment
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
            available_providers.append("claude")
        
        if GEMINI_API_KEY:
            available_providers.append("gemini")
            
    except ImportError:
        # Check environment variables
        if os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"):
            available_providers.append("grok")
        
        if os.environ.get("OPENAI_API_KEY"):
            available_providers.append("openai")
        
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY"):
            available_providers.append("claude")
        
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            available_providers.append("gemini")
    
    return available_providers