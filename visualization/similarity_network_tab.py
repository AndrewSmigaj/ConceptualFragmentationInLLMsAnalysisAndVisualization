"""
Similarity Network Tab for the Neural Network Trajectories Dashboard.

This module provides the layout and callbacks for the similarity network tab,
which visualizes similarity relationships between clusters across different layers.
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import importlib

from visualization.similarity_network_viz import (
    build_similarity_network, 
    compute_force_directed_layout,
    create_similarity_network_plot,
    create_top_similar_clusters_table,
    create_convergent_paths_summary
)

# Create the layout for the Similarity Network tab
def create_similarity_network_tab():
    """Create the layout for the Similarity Network tab."""
    return dcc.Tab(label="Similarity Network", children=[
        html.Div([
            html.H4("Cluster Similarity Network Visualization", style={"textAlign": "center"}),
            
            # Controls row
            html.Div([
                # Left column: Similarity threshold
                html.Div([
                    html.Label("Similarity Threshold:"),
                    dcc.Slider(
                        id="similarity-threshold-slider",
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        value=0.5,
                        marks={i/10: str(i/10) for i in range(0, 11, 1)},
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                # Middle column: Max links control
                html.Div([
                    html.Label("Maximum Links:"),
                    dcc.Slider(
                        id="max-links-slider",
                        min=10,
                        max=300,
                        step=10,
                        value=100,
                        marks={i: str(i) for i in range(0, 301, 50)},
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                # Right column: Configuration selection
                html.Div([
                    html.Label("Configuration:"),
                    dcc.RadioItems(
                        id="similarity-config-radio",
                        options=[
                            {"label": "Baseline", "value": "baseline"},
                            {"label": "Regularized", "value": "regularized"}
                        ],
                        value="baseline",
                        inline=True
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
            
            # Display row
            html.Div([
                # Update button
                html.Button(
                    "Update Network", 
                    id="update-network-button", 
                    n_clicks=0,
                    style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px", "margin": "10px"}
                ),
            ], style={"textAlign": "center"}),
            
            # Network visualization
            dcc.Loading(
                id="loading-similarity-network",
                type="circle",
                children=[
                    dcc.Graph(
                        id="similarity-network-graph",
                        style={"height": "600px"},
                        config={
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "similarity_network",
                                "height": 600,
                                "width": 1000
                            }
                        }
                    )
                ]
            ),
            
            # Network Statistics
            html.Div([
                html.H5("Network Statistics", style={"textAlign": "center"}),
                html.Div(id="similarity-network-stats", style={"padding": "10px"})
            ]),
            
            # Tabs for more details
            dcc.Tabs([
                # Top Similar Clusters tab
                dcc.Tab(label="Top Similar Clusters", children=[
                    dcc.Loading(
                        id="loading-similar-clusters",
                        type="circle",
                        children=[
                            html.Div(id="top-similar-clusters-container", style={"padding": "15px"})
                        ]
                    )
                ]),
                
                # Convergent Paths tab
                dcc.Tab(label="Similarity-Convergent Paths", children=[
                    dcc.Loading(
                        id="loading-convergent-paths",
                        type="circle",
                        children=[
                            html.Div(id="convergent-paths-container", style={"padding": "15px"})
                        ]
                    )
                ]),
            ]),
        ])
    ])

# Register callbacks for the Similarity Network tab
def register_similarity_callbacks(app):
    """Register callbacks for the Similarity Network tab."""
    
    @app.callback(
        Output("similarity-network-graph", "figure"),
        [Input("update-network-button", "n_clicks")],
        [State("similarity-threshold-slider", "value"),
         State("max-links-slider", "value"),
         State("similarity-config-radio", "value"),
         State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("embeddings-store", "data"),
         State("layer-clusters-store", "data")]
    )
    def update_similarity_network(n_clicks, threshold, max_links, config, dataset, seed, 
                                 stored_embeddings, stored_clusters):
        """Update the similarity network visualization."""
        if n_clicks == 0 or not dataset or not stored_clusters:
            # Return empty figure on initial load
            return go.Figure().update_layout(
                title="Click 'Update Network' to generate similarity network",
                height=600
            )
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Convert stored clusters from JSON
            clusters = dash_app.list_to_numpy_recursive(stored_clusters)
            
            # Check if we have data for the selected configuration
            if config not in clusters:
                return go.Figure().update_layout(
                    title=f"No cluster data for {config} configuration",
                    height=600
                )
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return go.Figure().update_layout(
                    title="No similarity data available. Run cluster_paths.py with --compute_similarity option.",
                    height=600
                )
            
            # Extract similarity data
            similarity_data = cluster_paths_data["similarity"]
            
            # Get similarity matrix
            raw_similarity = similarity_data.get("raw_similarity", {})
            normalized_similarity = similarity_data.get("normalized_similarity", {})
            
            # Convert serialized matrix back to dictionary with tuple keys
            similarity_matrix = {}
            for key, value in normalized_similarity.items():
                if "," in key:
                    id1, id2 = map(int, key.split(","))
                    similarity_matrix[(id1, id2)] = float(value)
            
            # Get cluster ID mapping
            id_to_layer_cluster = {}
            if "id_mapping" in cluster_paths_data:
                for unique_id_str, mapping in cluster_paths_data["id_mapping"].items():
                    unique_id = int(unique_id_str)
                    id_to_layer_cluster[unique_id] = (
                        mapping["layer_name"],
                        mapping["original_id"],
                        mapping["layer_idx"]
                    )
            
            # Build network graph
            G = build_similarity_network(
                similarity_matrix,
                id_to_layer_cluster,
                threshold=threshold,
                max_links=max_links
            )
            
            # Get friendly layer name function from dash_app
            get_friendly_layer_name = dash_app.get_friendly_layer_name
            
            # Compute layout positions
            positions = compute_force_directed_layout(G, iterations=100)
            
            # Create network visualization
            fig = create_similarity_network_plot(
                G,
                positions,
                id_to_layer_cluster,
                height=600,
                width=1000,
                get_friendly_layer_name=get_friendly_layer_name
            )
            
            return fig
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating similarity network: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=600
            )
    
    @app.callback(
        Output("similarity-network-stats", "children"),
        [Input("update-network-button", "n_clicks")],
        [State("similarity-threshold-slider", "value"),
         State("max-links-slider", "value"),
         State("similarity-config-radio", "value"),
         State("dataset-dropdown", "value"),
         State("seed-dropdown", "value")]
    )
    def update_network_stats(n_clicks, threshold, max_links, config, dataset, seed):
        """Update the network statistics panel."""
        if n_clicks == 0 or not dataset:
            return "Click 'Update Network' to see statistics."
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return "No similarity data available."
            
            # Extract similarity data
            similarity_data = cluster_paths_data["similarity"]
            
            # Get similarity matrix
            normalized_similarity = similarity_data.get("normalized_similarity", {})
            
            # Convert serialized matrix back to dictionary with tuple keys
            similarity_matrix = {}
            for key, value in normalized_similarity.items():
                if "," in key:
                    id1, id2 = map(int, key.split(","))
                    similarity_matrix[(id1, id2)] = float(value)
            
            # Filter by threshold
            strong_connections = {k: v for k, v in similarity_matrix.items() if v >= threshold}
            
            # Get similarity statistics
            total_connections = len(similarity_matrix)
            strong_connection_count = len(strong_connections)
            if total_connections > 0:
                strong_percent = (strong_connection_count / total_connections) * 100
            else:
                strong_percent = 0
                
            avg_similarity = sum(similarity_matrix.values()) / max(1, len(similarity_matrix))
            
            # Count connections between different layers
            cross_layer_connections = sum(1 for (id1, id2), sim in strong_connections.items() 
                                        if "id_mapping" in cluster_paths_data and 
                                        str(id1) in cluster_paths_data["id_mapping"] and 
                                        str(id2) in cluster_paths_data["id_mapping"] and
                                        cluster_paths_data["id_mapping"][str(id1)]["layer_idx"] != 
                                        cluster_paths_data["id_mapping"][str(id2)]["layer_idx"])
            
            # Get fragmentation statistics if available
            frag_stats = {}
            if "fragmentation_scores" in similarity_data:
                frag_data = similarity_data["fragmentation_scores"]
                frag_stats = {
                    "mean": frag_data.get("mean", 0),
                    "median": frag_data.get("median", 0),
                    "high_count": len(frag_data.get("high_fragmentation_paths", [])),
                    "low_count": len(frag_data.get("low_fragmentation_paths", []))
                }
            
            # Create statistics panel
            stats_elements = [
                html.Div([
                    html.Div([
                        html.Strong("Total Connections: "),
                        html.Span(f"{total_connections}")
                    ], style={"marginBottom": "5px"}),
                    html.Div([
                        html.Strong("Strong Connections (≥ threshold): "),
                        html.Span(f"{strong_connection_count} ({strong_percent:.1f}%)")
                    ], style={"marginBottom": "5px"}),
                    html.Div([
                        html.Strong("Cross-Layer Connections: "),
                        html.Span(f"{cross_layer_connections}")
                    ], style={"marginBottom": "5px"}),
                    html.Div([
                        html.Strong("Average Similarity: "),
                        html.Span(f"{avg_similarity:.3f}")
                    ], style={"marginBottom": "5px"})
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"})
            ]
            
            # Add fragmentation stats if available
            if frag_stats:
                stats_elements.append(html.Div([
                    html.Div([
                        html.Strong("Mean Fragmentation: "),
                        html.Span(f"{frag_stats['mean']:.3f}")
                    ], style={"marginBottom": "5px"}),
                    html.Div([
                        html.Strong("Median Fragmentation: "),
                        html.Span(f"{frag_stats['median']:.3f}")
                    ], style={"marginBottom": "5px"}),
                    html.Div([
                        html.Strong("High Fragmentation Paths: "),
                        html.Span(f"{frag_stats['high_count']}")
                    ], style={"marginBottom": "5px"}),
                    html.Div([
                        html.Strong("Low Fragmentation Paths: "),
                        html.Span(f"{frag_stats['low_count']}")
                    ], style={"marginBottom": "5px"})
                ], style={"width": "50%", "display": "inline-block", "verticalAlign": "top"}))
            
            return html.Div(stats_elements, style={"display": "flex"})
            
        except Exception as e:
            return f"Error calculating network statistics: {str(e)}"
    
    @app.callback(
        Output("top-similar-clusters-container", "children"),
        [Input("update-network-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value")]
    )
    def update_top_similar_clusters(n_clicks, dataset, seed):
        """Update the top similar clusters table."""
        if n_clicks == 0 or not dataset:
            return "Click 'Update Network' to see top similar clusters."
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return "No similarity data available."
            
            # Extract similarity data
            similarity_data = cluster_paths_data["similarity"]
            
            # Get top similar clusters
            top_similar = similarity_data.get("top_similar_clusters", {})
            
            # Convert JSON structure to expected format
            top_similar_clusters = {}
            for cluster_id_str, similar_list in top_similar.items():
                cluster_id = int(cluster_id_str)
                similar_pairs = []
                for pair in similar_list:
                    if len(pair) == 2:
                        similar_id = int(pair[0])
                        similarity = float(pair[1])
                        similar_pairs.append((similar_id, similarity))
                top_similar_clusters[cluster_id] = similar_pairs
            
            # Get cluster ID mapping
            id_to_layer_cluster = {}
            if "id_mapping" in cluster_paths_data:
                for unique_id_str, mapping in cluster_paths_data["id_mapping"].items():
                    unique_id = int(unique_id_str)
                    id_to_layer_cluster[unique_id] = (
                        mapping["layer_name"],
                        mapping["original_id"],
                        mapping["layer_idx"]
                    )
            
            # Get friendly layer name function
            get_friendly_layer_name = dash_app.get_friendly_layer_name
            
            # Create table
            df = create_top_similar_clusters_table(
                top_similar_clusters,
                id_to_layer_cluster,
                get_friendly_layer_name=get_friendly_layer_name
            )
            
            if df.empty:
                return "No similar clusters found above threshold."
            
            # Create data table
            table = dash_table.DataTable(
                id='similar-clusters-table',
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict('records'),
                sort_action="native",
                filter_action="native",
                page_size=10,
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
                    },
                    {
                        'if': {
                            'filter_query': '{Similarity} >= 0.8',
                            'column_id': 'Similarity'
                        },
                        'backgroundColor': 'rgb(183, 226, 196)',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {
                            'filter_query': '{Similarity} < 0.5',
                            'column_id': 'Similarity'
                        },
                        'backgroundColor': 'rgb(255, 229, 229)'
                    }
                ]
            )
            
            return html.Div([
                html.H5(f"Top Similar Clusters (Dataset: {dataset}, Seed: {seed})"),
                table
            ])
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating similar clusters table: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"Error: {str(e)}"
    
    @app.callback(
        Output("convergent-paths-container", "children"),
        [Input("update-network-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value")]
    )
    def update_convergent_paths(n_clicks, dataset, seed):
        """Update the convergent paths table."""
        if n_clicks == 0 or not dataset:
            return "Click 'Update Network' to see similarity-convergent paths."
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return "No similarity data available."
            
            # Extract similarity data
            similarity_data = cluster_paths_data["similarity"]
            
            # Get convergent paths data
            convergent_paths = {}
            if "convergent_paths" in similarity_data:
                for path_idx_str, convergences in similarity_data["convergent_paths"].items():
                    try:
                        path_idx = int(path_idx_str)
                        convergent_paths[path_idx] = convergences
                    except ValueError:
                        # Skip if path_idx is not an integer
                        continue
            
            # Get human-readable convergent paths
            human_readable_paths = similarity_data.get("human_readable_convergent_paths", {})
            
            # Get fragmentation scores
            fragmentation_scores = {}
            if "fragmentation_scores" in similarity_data and "scores" in similarity_data["fragmentation_scores"]:
                scores = similarity_data["fragmentation_scores"]["scores"]
                for i, score in enumerate(scores):
                    fragmentation_scores[str(i)] = score
            
            # Create summary DataFrame
            df = create_convergent_paths_summary(
                convergent_paths,
                human_readable_paths,
                fragmentation_scores
            )
            
            if df.empty:
                return "No similarity-convergent paths found."
            
            # Create data table
            table = dash_table.DataTable(
                id='convergent-paths-table',
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict('records'),
                sort_action="native",
                filter_action="native",
                page_size=10,
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
                    },
                    {
                        'if': {
                            'filter_query': '{Strongest Similarity} >= 0.8',
                            'column_id': 'Strongest Similarity'
                        },
                        'backgroundColor': 'rgb(183, 226, 196)',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {
                            'filter_query': '{Number of Convergences} >= 3',
                            'column_id': 'Number of Convergences'
                        },
                        'backgroundColor': 'rgb(200, 220, 240)',
                        'fontWeight': 'bold'
                    }
                ]
            )
            
            # Add explanation about similarity-convergent paths
            explanation = html.Div([
                html.P(
                    "Similarity-convergent paths indicate where clusters in later layers have high similarity to clusters in earlier layers. "
                    "This suggests conceptual preservation or re-emergence across different parts of the network."
                ),
                html.P(
                    "Paths with more convergences and higher similarity are more likely to represent important conceptual patterns."
                )
            ], style={"marginBottom": "15px"})
            
            return html.Div([
                html.H5(f"Similarity-Convergent Paths (Dataset: {dataset}, Seed: {seed})"),
                explanation,
                table
            ])
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating convergent paths table: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"Error: {str(e)}"