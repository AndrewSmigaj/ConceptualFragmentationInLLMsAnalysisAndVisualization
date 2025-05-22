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
from typing import Dict, List, Tuple, Any, Optional, Union

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
                            {"label": "Path Narratives", "value": "path_narratives"},
                            {"label": "GPT-2 Attention Patterns", "value": "attention_patterns"},
                            {"label": "GPT-2 Token Movement", "value": "token_movement"},
                            {"label": "GPT-2 Concept Purity", "value": "concept_purity"},
                            {"label": "Token Path Comparison", "value": "token_path_comparison"}
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
            dcc.Tabs(id="llm-result-tabs", children=[
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
                
                # Attention Patterns Tab
                dcc.Tab(label="Attention Patterns", children=[
                    dcc.Loading(
                        id="loading-attention-patterns",
                        type="circle",
                        children=[
                            html.Div(id="attention-patterns-container", style={"padding": "15px"})
                        ]
                    )
                ]),
                
                # Token Movement Tab
                dcc.Tab(label="Token Movement", children=[
                    dcc.Loading(
                        id="loading-token-movement",
                        type="circle",
                        children=[
                            html.Div(id="token-movement-container", style={"padding": "15px"})
                        ]
                    )
                ]),
                
                # Concept Purity Tab
                dcc.Tab(label="Concept Purity", children=[
                    dcc.Loading(
                        id="loading-concept-purity",
                        type="circle",
                        children=[
                            html.Div(id="concept-purity-container", style={"padding": "15px"})
                        ]
                    )
                ]),
                
                # Token Path Comparison Tab
                dcc.Tab(label="Token Path Comparison", children=[
                    dcc.Loading(
                        id="loading-token-path-comparison",
                        type="circle",
                        children=[
                            html.Div(id="token-path-comparison-container", style={"padding": "15px"})
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
            
            # If GPT-2 attention patterns analysis is requested
            if "attention_patterns" in analysis_types:
                # Check if we're dealing with a GPT-2 model data
                is_gpt2_data = False
                model_type = cluster_paths_data.get("model_type", "").lower()
                if "gpt2" in model_type or "transformer" in model_type:
                    is_gpt2_data = True
                
                if is_gpt2_data and "attention_data" in cluster_paths_data:
                    # Extract attention data
                    attention_data = cluster_paths_data["attention_data"]
                    
                    # Update status
                    status_div = html.Div([
                        status_div,
                        html.Div("Generating attention pattern analysis...")
                    ])
                    dash.callback_context.record_timing("llm-status-update", "Generating attention pattern analysis...")
                    
                    try:
                        # Import GPT-2 prompts module
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from concept_fragmentation.llm.gpt2_prompts import generate_attention_pattern_prompt
                        
                        # Prepare attention pattern data for each layer
                        attention_patterns = {}
                        
                        # Process each layer's attention data
                        for layer_name, layer_data in attention_data.items():
                            # Extract metrics
                            entropy = layer_data.get("entropy", 0.0)
                            head_agreement = layer_data.get("head_agreement", 0.0)
                            num_heads = layer_data.get("num_heads", 12)  # Default for GPT-2
                            
                            # Extract notable patterns (top 5)
                            notable_patterns = []
                            if "notable_patterns" in layer_data:
                                for pattern in layer_data["notable_patterns"][:5]:
                                    pattern_desc = f"{pattern['description']} (score: {pattern['score']:.3f})"
                                    notable_patterns.append(pattern_desc)
                            
                            # Extract token-to-token examples (top 5)
                            token_examples = []
                            if "token_attention" in layer_data:
                                for example in layer_data["token_attention"][:5]:
                                    example_desc = f"'{example['source']}' → '{example['target']}' (strength: {example['strength']:.3f})"
                                    token_examples.append(example_desc)
                            
                            # Generate prompt for this layer
                            prompt = generate_attention_pattern_prompt(
                                layer_name=layer_name,
                                attention_entropy=entropy,
                                head_agreement=head_agreement,
                                num_heads=num_heads,
                                attention_patterns=notable_patterns or ["Pattern information not available"],
                                token_attention_examples=token_examples or ["Token attention examples not available"]
                            )
                            
                            # Get narrative from LLM
                            response = analyzer.client.generate(prompt)
                            
                            # Store results
                            attention_patterns[layer_name] = {
                                "entropy": entropy,
                                "head_agreement": head_agreement,
                                "num_heads": num_heads,
                                "narrative": response
                            }
                        
                        # Store the results
                        results["attention_patterns"] = attention_patterns
                        
                        # Update status
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Generated attention pattern analysis for {len(attention_patterns)} layers successfully.", 
                                    style={"color": "green"})
                        ])
                    except Exception as e:
                        import traceback
                        error_msg = f"Error generating attention pattern analysis: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Error generating attention pattern analysis: {str(e)}", style={"color": "red"})
                        ])
                else:
                    status_div = html.Div([
                        status_div,
                        html.Div("No GPT-2 attention data found for analysis.", style={"color": "orange"})
                    ])
            
            # If GPT-2 token movement analysis is requested
            if "token_movement" in analysis_types:
                # Check if we're dealing with a GPT-2 model data with token paths
                is_gpt2_data = False
                model_type = cluster_paths_data.get("model_type", "").lower()
                if "gpt2" in model_type or "transformer" in model_type:
                    is_gpt2_data = True
                
                if is_gpt2_data and "token_paths" in cluster_paths_data:
                    # Extract token path data
                    token_paths = cluster_paths_data["token_paths"]
                    
                    # Update status
                    status_div = html.Div([
                        status_div,
                        html.Div("Generating token movement analysis...")
                    ])
                    dash.callback_context.record_timing("llm-status-update", "Generating token movement analysis...")
                    
                    try:
                        # Import GPT-2 prompts module
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from concept_fragmentation.llm.gpt2_prompts import generate_token_movement_prompt
                        
                        # Prepare token movement data
                        token_movement = {}
                        
                        # Calculate average metrics for comparison
                        total_tokens = len(token_paths)
                        all_path_lengths = [path.get("path_length", 0) for path in token_paths.values()]
                        all_cluster_changes = [path.get("cluster_changes", 0) for path in token_paths.values()]
                        
                        avg_path_length = sum(all_path_lengths) / max(len(all_path_lengths), 1)
                        avg_cluster_changes = sum(all_cluster_changes) / max(len(all_cluster_changes), 1)
                        
                        # Rank tokens by mobility
                        if total_tokens > 0:
                            mobility_scores = {token_id: path.get("mobility_score", 0) 
                                            for token_id, path in token_paths.items()}
                            ranked_tokens = sorted(mobility_scores.keys(), 
                                                key=lambda k: mobility_scores[k], reverse=True)
                            token_ranks = {token_id: idx+1 for idx, token_id in enumerate(ranked_tokens)}
                        else:
                            token_ranks = {}
                        
                        # Select top N most mobile tokens for analysis
                        tokens_to_analyze = 10
                        selected_tokens = list(token_ranks.items())[:tokens_to_analyze] if token_ranks else []
                        
                        # Process each selected token
                        for token_id, rank in selected_tokens:
                            path_data = token_paths.get(token_id, {})
                            
                            # Extract path metrics
                            token_text = path_data.get("token_text", f"Token {token_id}")
                            token_position = path_data.get("position", 0)
                            cluster_path = path_data.get("cluster_path", [])
                            path_length = path_data.get("path_length", 0)
                            cluster_changes = path_data.get("cluster_changes", 0)
                            mobility_score = path_data.get("mobility_score", 0)
                            
                            # Convert cluster path to format needed for prompt
                            formatted_path = []
                            for i, cluster_id in enumerate(cluster_path):
                                layer_name = f"L{i}"
                                formatted_path.append((layer_name, cluster_id))
                            
                            # Generate prompt for this token
                            prompt = generate_token_movement_prompt(
                                token_text=token_text,
                                token_position=token_position,
                                cluster_path=formatted_path,
                                path_length=path_length,
                                cluster_changes=cluster_changes,
                                mobility_score=mobility_score,
                                avg_path_length=avg_path_length,
                                avg_cluster_changes=avg_cluster_changes,
                                mobility_ranking=rank,
                                total_tokens=total_tokens
                            )
                            
                            # Get narrative from LLM
                            response = analyzer.client.generate(prompt)
                            
                            # Create path string representation for display
                            cluster_path_str = " → ".join([f"L{i}C{cluster_id}" for i, cluster_id in enumerate(cluster_path)])
                            
                            # Store results
                            token_movement[token_id] = {
                                "token_text": token_text,
                                "position": token_position,
                                "path_length": path_length,
                                "cluster_changes": cluster_changes,
                                "mobility_score": mobility_score,
                                "mobility_ranking": rank,
                                "cluster_path_str": cluster_path_str,
                                "narrative": response
                            }
                        
                        # Store the results
                        results["token_movement"] = token_movement
                        
                        # Update status
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Generated token movement analysis for {len(token_movement)} tokens successfully.", 
                                    style={"color": "green"})
                        ])
                    except Exception as e:
                        import traceback
                        error_msg = f"Error generating token movement analysis: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Error generating token movement analysis: {str(e)}", style={"color": "red"})
                        ])
                else:
                    status_div = html.Div([
                        status_div,
                        html.Div("No GPT-2 token path data found for analysis.", style={"color": "orange"})
                    ])
            
            # If GPT-2 concept purity analysis is requested
            if "concept_purity" in analysis_types:
                # Check if we're dealing with a GPT-2 model data
                is_gpt2_data = False
                model_type = cluster_paths_data.get("model_type", "").lower()
                if "gpt2" in model_type or "transformer" in model_type:
                    is_gpt2_data = True
                
                if is_gpt2_data and "cluster_metrics" in cluster_paths_data:
                    # Extract cluster metrics data
                    cluster_metrics = cluster_paths_data["cluster_metrics"]
                    
                    # Update status
                    status_div = html.Div([
                        status_div,
                        html.Div("Generating concept purity analysis...")
                    ])
                    dash.callback_context.record_timing("llm-status-update", "Generating concept purity analysis...")
                    
                    try:
                        # Import GPT-2 prompts module
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from concept_fragmentation.llm.gpt2_prompts import generate_concept_purity_prompt
                        
                        # Extract layer metrics
                        layer_metrics = {}
                        for layer_name, metrics in cluster_metrics.items():
                            if "purity" in metrics and "silhouette" in metrics:
                                layer_metrics[layer_name] = {
                                    "purity": metrics["purity"],
                                    "silhouette": metrics["silhouette"]
                                }
                        
                        # Calculate overall metrics
                        avg_silhouette = sum([m["silhouette"] for m in layer_metrics.values()]) / len(layer_metrics) if layer_metrics else 0
                        cluster_stability = cluster_paths_data.get("cluster_stability", 0)
                        
                        # Find best and worst layers
                        sorted_layers = sorted(layer_metrics.items(), key=lambda x: x[1]["purity"], reverse=True)
                        best_layer = sorted_layers[0][0] if sorted_layers else "Unknown"
                        best_purity = sorted_layers[0][1]["purity"] if sorted_layers else 0
                        
                        worst_layer = sorted_layers[-1][0] if sorted_layers else "Unknown"
                        worst_purity = sorted_layers[-1][1]["purity"] if sorted_layers else 0
                        
                        # Generate prompt for concept purity analysis
                        prompt = generate_concept_purity_prompt(
                            layer_metrics=layer_metrics,
                            avg_silhouette=avg_silhouette,
                            cluster_stability=cluster_stability,
                            best_layer=best_layer,
                            best_purity=best_purity,
                            worst_layer=worst_layer,
                            worst_purity=worst_purity
                        )
                        
                        # Get narrative from LLM
                        response = analyzer.client.generate(prompt)
                        
                        # Store results
                        concept_purity_results = {
                            "layer_metrics": layer_metrics,
                            "avg_silhouette": avg_silhouette,
                            "cluster_stability": cluster_stability,
                            "best_layer": best_layer,
                            "best_purity": best_purity,
                            "worst_layer": worst_layer,
                            "worst_purity": worst_purity,
                            "narrative": response
                        }
                        
                        # Store the results
                        results["concept_purity"] = concept_purity_results
                        
                        # Update status
                        status_div = html.Div([
                            status_div,
                            html.Div("Generated concept purity analysis successfully.", style={"color": "green"})
                        ])
                    except Exception as e:
                        import traceback
                        error_msg = f"Error generating concept purity analysis: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Error generating concept purity analysis: {str(e)}", style={"color": "red"})
                        ])
                else:
                    status_div = html.Div([
                        status_div,
                        html.Div("No GPT-2 cluster metrics found for concept purity analysis.", style={"color": "orange"})
                    ])
            
            # If token path comparison analysis is requested
            if "token_path_comparison" in analysis_types:
                # Check if we have the necessary data
                if "attention_data" in cluster_paths_data and "token_paths" in cluster_paths_data:
                    # Update status
                    status_div = html.Div([
                        status_div,
                        html.Div("Generating token path comparison analysis...")
                    ])
                    dash.callback_context.record_timing("llm-status-update", "Generating token path comparison analysis...")
                    
                    try:
                        # Import correlation module
                        from visualization.gpt2_attention_correlation import calculate_correlation_metrics
                        
                        # Import GPT-2 prompts module
                        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                        from concept_fragmentation.llm.gpt2_prompts import generate_path_attention_correlation_prompt
                        
                        # Get token paths and attention data
                        token_paths = cluster_paths_data["token_paths"]
                        attention_data = cluster_paths_data["attention_data"]
                        
                        # Calculate correlation metrics
                        correlation_metrics = calculate_correlation_metrics(
                            token_paths=token_paths,
                            attention_flow=attention_data,
                            cluster_labels=cluster_paths_data.get("cluster_labels", {}),
                            layer_names=cluster_paths_data.get("layers", [])
                        )
                        
                        # Check if we have token correlations
                        token_correlations = correlation_metrics.get("token_correlations", {})
                        if token_correlations:
                            # Get tokens with strongest/weakest correlation
                            sorted_tokens = sorted(
                                [(token, stats.get("avg_correlation", 0)) 
                                 for token, stats in token_correlations.items()],
                                key=lambda x: x[1],
                                reverse=True
                            )
                            
                            # Get strongest and weakest tokens
                            strong_examples = sorted_tokens[:5]  # Top 5
                            weak_examples = sorted_tokens[-5:]   # Bottom 5
                            
                            # Generate prompt for correlation analysis
                            prompt = generate_path_attention_correlation_prompt(
                                correlation_metrics=correlation_metrics,
                                strong_examples=strong_examples,
                                weak_examples=weak_examples
                            )
                            
                            # Get narrative from LLM
                            response = analyzer.client.generate(prompt)
                            
                            # Store the results
                            results["token_path_comparison"] = {
                                "overall": response,
                                "metrics": correlation_metrics
                            }
                            
                            # Update status
                            status_div = html.Div([
                                status_div,
                                html.Div("Generated token path comparison analysis successfully.", 
                                         style={"color": "green"})
                            ])
                        else:
                            status_div = html.Div([
                                status_div,
                                html.Div("No token correlation data found for analysis.", style={"color": "orange"})
                            ])
                    except Exception as e:
                        import traceback
                        error_msg = f"Error generating token path comparison analysis: {str(e)}\n{traceback.format_exc()}"
                        print(error_msg)
                        status_div = html.Div([
                            status_div,
                            html.Div(f"Error generating token path comparison analysis: {str(e)}", style={"color": "red"})
                        ])
                else:
                    status_div = html.Div([
                        status_div,
                        html.Div("Token path or attention data not found for comparison analysis.", style={"color": "orange"})
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
    
    @app.callback(
        Output("attention-patterns-container", "children"),
        [Input("llm-analysis-store", "data")]
    )
    def update_attention_patterns(analysis_data):
        """Update the attention patterns analysis."""
        if not analysis_data or "attention_patterns" not in analysis_data:
            return html.Div("No attention pattern analysis available. Generate GPT-2 analysis first.")
        
        # Extract attention patterns analysis
        attention_patterns = analysis_data["attention_patterns"]
        provider = analysis_data.get("provider", "unknown")
        model = analysis_data.get("model", "unknown")
        
        # Create the attention patterns view
        pattern_cards = []
        
        # Import the dashboard module dynamically to avoid circular imports
        dash_app = importlib.import_module("visualization.dash_app")
        
        # Load model data to get attention information
        dataset = dash_app.current_dataset
        seed = dash_app.current_seed
        
        # Sort layer names for consistent display
        sorted_layer_names = sorted(attention_patterns.keys())
        
        for layer_name in sorted_layer_names:
            pattern_analysis = attention_patterns[layer_name]
            
            # Create card for this layer's attention pattern
            card = html.Div([
                # Header with layer name
                html.Div([
                    html.Span(f"Layer: {layer_name}", style={"fontWeight": "bold", "fontSize": "18px"})
                ]),
                
                # Attention metrics
                html.Div([
                    html.Div("Attention Metrics:", style={"fontWeight": "bold", "marginTop": "10px"}),
                    html.Div([
                        html.Div([
                            html.Span("Entropy: ", style={"fontWeight": "bold"}),
                            html.Span(f"{pattern_analysis.get('entropy', 'N/A'):.3f}" if 'entropy' in pattern_analysis else "N/A")
                        ], style={"marginRight": "20px", "display": "inline-block"}),
                        html.Div([
                            html.Span("Head Agreement: ", style={"fontWeight": "bold"}),
                            html.Span(f"{pattern_analysis.get('head_agreement', 'N/A'):.3f}" if 'head_agreement' in pattern_analysis else "N/A")
                        ], style={"marginRight": "20px", "display": "inline-block"}),
                        html.Div([
                            html.Span("Heads: ", style={"fontWeight": "bold"}),
                            html.Span(f"{pattern_analysis.get('num_heads', 'N/A')}" if 'num_heads' in pattern_analysis else "N/A")
                        ], style={"display": "inline-block"})
                    ], style={"marginTop": "5px"})
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
                "backgroundColor": "white",
                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
            })
            
            pattern_cards.append(card)
        
        # Create header with model info
        header = html.Div([
            html.H5(f"Attention Pattern Analysis Generated by {provider.title()} ({model})"),
            html.P(f"Total layers analyzed: {len(attention_patterns)}")
        ])
        
        return html.Div([header, html.Div(pattern_cards)])
    
    @app.callback(
        Output("token-movement-container", "children"),
        [Input("llm-analysis-store", "data")]
    )
    def update_token_movement(analysis_data):
        """Update the token movement analysis."""
        if not analysis_data or "token_movement" not in analysis_data:
            return html.Div("No token movement analysis available. Generate GPT-2 analysis first.")
        
        # Extract token movement analysis
        token_movement = analysis_data["token_movement"]
        provider = analysis_data.get("provider", "unknown")
        model = analysis_data.get("model", "unknown")
        
        # Create the token movement view
        token_cards = []
        
        # Import the dashboard module dynamically to avoid circular imports
        dash_app = importlib.import_module("visualization.dash_app")
        
        # Sort token IDs for consistent display
        sorted_token_ids = sorted(token_movement.keys())
        
        for token_id in sorted_token_ids:
            movement_analysis = token_movement[token_id]
            
            # Get token text and position
            token_text = movement_analysis.get('token_text', f"Token {token_id}")
            token_position = movement_analysis.get('position', 'Unknown')
            
            # Get movement metrics
            path_length = movement_analysis.get('path_length', 'N/A')
            cluster_changes = movement_analysis.get('cluster_changes', 'N/A')
            mobility_score = movement_analysis.get('mobility_score', 'N/A')
            mobility_rank = movement_analysis.get('mobility_ranking', 'N/A')
            
            # Create card for this token's movement pattern
            card = html.Div([
                # Header with token info
                html.Div([
                    html.Span(f"Token: \"{token_text}\"", style={"fontWeight": "bold", "fontSize": "18px"}),
                    html.Span(f" (position {token_position})", style={"color": "#666", "marginLeft": "10px"})
                ]),
                
                # Movement metrics
                html.Div([
                    html.Div("Movement Metrics:", style={"fontWeight": "bold", "marginTop": "10px"}),
                    html.Div([
                        html.Div([
                            html.Span("Path Length: ", style={"fontWeight": "bold"}),
                            html.Span(f"{path_length:.3f}" if isinstance(path_length, (int, float)) else path_length)
                        ], style={"marginRight": "20px", "display": "inline-block"}),
                        html.Div([
                            html.Span("Cluster Changes: ", style={"fontWeight": "bold"}),
                            html.Span(f"{cluster_changes}" if isinstance(cluster_changes, (int, float)) else cluster_changes)
                        ], style={"marginRight": "20px", "display": "inline-block"}),
                        html.Div([
                            html.Span("Mobility Score: ", style={"fontWeight": "bold"}),
                            html.Span(f"{mobility_score:.3f}" if isinstance(mobility_score, (int, float)) else mobility_score)
                        ], style={"display": "inline-block"})
                    ], style={"marginTop": "5px"}),
                    html.Div([
                        html.Span("Mobility Ranking: ", style={"fontWeight": "bold"}),
                        html.Span(f"{mobility_rank}" if isinstance(mobility_rank, (int, float)) else mobility_rank)
                    ], style={"marginTop": "5px"})
                ]),
                
                # Cluster path if available
                html.Div([
                    html.Div("Cluster Path:", style={"fontWeight": "bold", "marginTop": "10px"}),
                    html.Div(movement_analysis.get('cluster_path_str', 'Path not available'), 
                             style={"marginTop": "5px", "fontSize": "12px", "color": "#666"})
                ]) if 'cluster_path_str' in movement_analysis else None,
                
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
                "backgroundColor": "white",
                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
            })
            
            token_cards.append(card)
        
        # Create header with model info
        header = html.Div([
            html.H5(f"Token Movement Analysis Generated by {provider.title()} ({model})"),
            html.P(f"Total tokens analyzed: {len(token_movement)}")
        ])
        
        return html.Div([header, html.Div(token_cards)])
    
    @app.callback(
        Output("concept-purity-container", "children"),
        [Input("llm-analysis-store", "data")]
    )
    def update_concept_purity(analysis_data):
        """Update the concept purity analysis."""
        if not analysis_data or "concept_purity" not in analysis_data:
            return html.Div("No concept purity analysis available. Generate GPT-2 analysis first.")
        
        # Extract concept purity analysis
        concept_purity = analysis_data["concept_purity"]
        provider = analysis_data.get("provider", "unknown")
        model = analysis_data.get("model", "unknown")
        
        # Get overall metrics
        avg_silhouette = concept_purity.get('avg_silhouette', 'N/A')
        cluster_stability = concept_purity.get('cluster_stability', 'N/A')
        best_layer = concept_purity.get('best_layer', 'N/A')
        best_purity = concept_purity.get('best_purity', 'N/A')
        worst_layer = concept_purity.get('worst_layer', 'N/A')
        worst_purity = concept_purity.get('worst_purity', 'N/A')
        
        # Get layer metrics
        layer_metrics = concept_purity.get('layer_metrics', {})
        
        # Create the concept purity view
        metrics_card = html.Div([
            # Overall metrics
            html.Div([
                html.Div("Overall Metrics:", style={"fontWeight": "bold", "fontSize": "16px"}),
                html.Div([
                    html.Div([
                        html.Span("Average Silhouette: ", style={"fontWeight": "bold"}),
                        html.Span(f"{avg_silhouette:.3f}" if isinstance(avg_silhouette, (int, float)) else avg_silhouette)
                    ], style={"marginRight": "20px", "display": "inline-block"}),
                    html.Div([
                        html.Span("Cluster Stability: ", style={"fontWeight": "bold"}),
                        html.Span(f"{cluster_stability:.3f}" if isinstance(cluster_stability, (int, float)) else cluster_stability)
                    ], style={"display": "inline-block"})
                ], style={"marginTop": "5px"}),
                html.Div([
                    html.Div([
                        html.Span("Best Layer: ", style={"fontWeight": "bold"}),
                        html.Span(f"{best_layer} (purity: {best_purity:.3f})" if isinstance(best_purity, (int, float)) else f"{best_layer} (purity: {best_purity})")
                    ], style={"marginRight": "20px", "display": "inline-block"}),
                    html.Div([
                        html.Span("Worst Layer: ", style={"fontWeight": "bold"}),
                        html.Span(f"{worst_layer} (purity: {worst_purity:.3f})" if isinstance(worst_purity, (int, float)) else f"{worst_layer} (purity: {worst_purity})")
                    ], style={"display": "inline-block"})
                ], style={"marginTop": "5px"})
            ], style={"marginBottom": "20px"}),
            
            # Layer metrics
            html.Div([
                html.Div("Layer Metrics:", style={"fontWeight": "bold", "fontSize": "16px"}),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Layer", style={"textAlign": "left", "padding": "8px"}),
                            html.Th("Purity Score", style={"textAlign": "left", "padding": "8px"}),
                            html.Th("Silhouette Score", style={"textAlign": "left", "padding": "8px"})
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(layer, style={"padding": "8px"}),
                                html.Td(f"{metrics.get('purity', 'N/A'):.3f}" if isinstance(metrics.get('purity'), (int, float)) else metrics.get('purity', 'N/A'), 
                                        style={"padding": "8px"}),
                                html.Td(f"{metrics.get('silhouette', 'N/A'):.3f}" if isinstance(metrics.get('silhouette'), (int, float)) else metrics.get('silhouette', 'N/A'), 
                                        style={"padding": "8px"})
                            ]) for layer, metrics in sorted(layer_metrics.items())
                        ])
                    ], style={"width": "100%", "borderCollapse": "collapse"})
                ], style={"marginTop": "10px"})
            ], style={"marginBottom": "20px"})
        ], style={
            "border": "1px solid #ddd",
            "borderRadius": "5px",
            "padding": "15px",
            "marginBottom": "15px",
            "backgroundColor": "white",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
        })
        
        # Narrative section
        narrative_card = html.Div([
            html.Div("Analysis:", style={"fontWeight": "bold", "fontSize": "16px"}),
            html.Div(concept_purity.get('narrative', 'No analysis available'), 
                     style={"marginTop": "10px", "padding": "15px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
        ], style={
            "border": "1px solid #ddd",
            "borderRadius": "5px",
            "padding": "15px",
            "marginBottom": "15px",
            "backgroundColor": "white",
            "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
        })
        
        # Create header with model info
        header = html.Div([
            html.H5(f"Concept Purity Analysis Generated by {provider.title()} ({model})"),
            html.P("Analysis of how well-defined and distinct clusters are across layers")
        ])
        
        return html.Div([header, metrics_card, narrative_card])
    
    @app.callback(
        Output("token-path-comparison-container", "children"),
        [Input("llm-analysis-store", "data")]
    )
    def update_token_path_comparison(analysis_data):
        """Update the token path comparison analysis."""
        if not analysis_data or "token_path_comparison" not in analysis_data:
            return html.Div("No token path comparison analysis available. Generate LLM analysis with 'Token Path Comparison' selected.")
        
        # Extract token path comparison analysis
        token_path_comparison = analysis_data["token_path_comparison"]
        provider = analysis_data.get("provider", "unknown")
        model = analysis_data.get("model", "unknown")
        
        # Create the token path comparison view
        comparison_cards = []
        
        for correlation_type, analysis in token_path_comparison.items():
            # Create card for this comparison
            card = html.Div([
                # Header with correlation type
                html.Div([
                    html.Span(f"Correlation Type: {correlation_type}", style={"fontWeight": "bold", "fontSize": "18px"})
                ]),
                
                # Narrative text
                html.Div([
                    html.Div("Analysis:", style={"fontWeight": "bold", "marginTop": "10px"}),
                    html.Div(analysis, style={"marginTop": "5px", "padding": "10px", "backgroundColor": "#f8f8f8", "borderRadius": "5px"})
                ])
            ], style={
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "padding": "15px",
                "marginBottom": "15px",
                "backgroundColor": "white",
                "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
            })
            
            comparison_cards.append(card)
        
        if not comparison_cards:
            return html.Div("No token path comparison analysis available.")
        
        # Create header with model info
        header = html.Div([
            html.H5(f"Token Path Comparison Analysis Generated by {provider.title()} ({model})"),
            html.P(f"Analysis of correlations between token paths and attention patterns")
        ])
        
        return html.Div([header, html.Div(comparison_cards)])

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