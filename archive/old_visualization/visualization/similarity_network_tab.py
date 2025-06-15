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

# Import cross-layer visualization functions
from visualization.cross_layer_viz import (
    plot_centroid_similarity_heatmap,
    plot_membership_overlap_sankey,
    plot_trajectory_fragmentation_bars,
    plot_path_density_network
)

# Import metrics for cross-layer analysis
from concept_fragmentation.metrics.cross_layer_metrics import (
    compute_centroid_similarity, 
    compute_membership_overlap,
    compute_trajectory_fragmentation, 
    compute_path_density
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
                
                # Cross-Layer Visualizations tab
                dcc.Tab(label="Cross-Layer Visualizations", children=[
                    dcc.Loading(
                        id="loading-cross-layer-viz",
                        type="circle",
                        children=[
                            html.Div([
                                # Centroid Similarity Section
                                html.Div([
                                    html.H4("Centroid Similarity Heatmaps", style={"textAlign": "center"}),
                                    dcc.Graph(id="sim-centroid-similarity-graph")
                                ], id="sim-centroid-similarity-section", style={"marginTop": "20px"}),
                                
                                # Membership Overlap Section
                                html.Div([
                                    html.H4("Membership Overlap Flow", style={"textAlign": "center"}),
                                    dcc.Graph(id="sim-membership-overlap-graph")
                                ], id="sim-membership-overlap-section", style={"marginTop": "20px"}),
                                
                                # Trajectory Fragmentation Section
                                html.Div([
                                    html.H4("Trajectory Fragmentation by Layer", style={"textAlign": "center"}),
                                    dcc.Graph(id="sim-trajectory-fragmentation-graph")
                                ], id="sim-trajectory-fragmentation-section", style={"marginTop": "20px"}),
                                
                                # Path Density Section
                                html.Div([
                                    html.H4("Inter-Cluster Path Density", style={"textAlign": "center"}),
                                    dcc.Graph(id="sim-path-density-graph")
                                ], id="sim-path-density-section", style={"marginTop": "20px"}),
                            ])
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
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                
                if error_type == "missing_data":
                    # Create a figure with detailed instructions
                    fig = go.Figure()
                    
                    # Add error message as annotation
                    fig.add_annotation(
                        text="Similarity Data Not Found",
                        x=0.5, y=0.9,
                        xref="paper", yref="paper",
                        font=dict(size=24, color="red"),
                        showarrow=False
                    )
                    
                    # Add explanation
                    fig.add_annotation(
                        text=cluster_paths_data.get("message", "Missing similarity data"),
                        x=0.5, y=0.8,
                        xref="paper", yref="paper",
                        font=dict(size=16),
                        showarrow=False
                    )
                    
                    # Add command to run
                    cmd = cluster_paths_data.get("command", "")
                    if cmd:
                        fig.add_annotation(
                            text="To generate the required data, run:",
                            x=0.5, y=0.6,
                            xref="paper", yref="paper",
                            font=dict(size=16),
                            showarrow=False
                        )
                        
                        fig.add_annotation(
                            text=cmd,
                            x=0.5, y=0.5,
                            xref="paper", yref="paper",
                            font=dict(family="Courier New", size=14),
                            showarrow=False
                        )
                    
                    # Update layout with more visual details
                    fig.update_layout(
                        title="Missing Similarity Data",
                        height=600,
                        plot_bgcolor="rgba(240, 240, 240, 0.8)"  # Light gray background
                    )
                    
                    return fig
                
                # Handle other error types
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return go.Figure().update_layout(
                    title=f"Error: {error_msg}",
                    height=600
                )
            
            # Check if similarity data exists in the returned data
            if "similarity" not in cluster_paths_data:
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
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                
                if error_type == "missing_data":
                    # Create a more informative error message
                    message = cluster_paths_data.get("message", "Missing similarity data")
                    command = cluster_paths_data.get("command", "")
                    
                    return html.Div([
                        html.Div(
                            "Missing Similarity Data", 
                            style={"color": "red", "fontWeight": "bold", "fontSize": 18, "marginBottom": "10px"}
                        ),
                        html.Div(message, style={"marginBottom": "10px"}),
                        html.Div("Run this command to generate the data:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                        html.Pre(command, style={"backgroundColor": "#f0f0f0", "padding": "10px", "borderRadius": "3px"})
                    ])
                
                # Handle other error types
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return f"Error: {error_msg}"
            
            # Check for similarity data
            if "similarity" not in cluster_paths_data:
                return "No similarity data available. Run cluster_paths.py with --compute_similarity option."
            
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
                
            # Calculate similarity distribution
            similarity_values = list(similarity_matrix.values())
            
            # Define similarity categories
            low_threshold = 0.3  # Below this is considered low
            medium_threshold = 0.7  # Below this is medium, above is high
            
            # Count connections in each category
            low_connections = sum(1 for v in similarity_values if v < low_threshold)
            medium_connections = sum(1 for v in similarity_values if low_threshold <= v < medium_threshold)
            high_connections = sum(1 for v in similarity_values if v >= medium_threshold)
            
            # Calculate percentages
            if total_connections > 0:
                low_percent = (low_connections / total_connections) * 100
                medium_percent = (medium_connections / total_connections) * 100
                high_percent = (high_connections / total_connections) * 100
            else:
                low_percent = medium_percent = high_percent = 0
                
            # Calculate statistics for similarity distribution
            similarity_distribution = {
                "low": {"count": low_connections, "percent": low_percent},
                "medium": {"count": medium_connections, "percent": medium_percent},
                "high": {"count": high_connections, "percent": high_percent},
                "threshold": threshold
            }
            
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
            
            # Calculate network density
            # Get total number of clusters to determine max possible connections
            total_clusters = 0
            if "id_mapping" in cluster_paths_data:
                total_clusters = len(cluster_paths_data["id_mapping"])
            
            # Maximum possible connections between clusters
            max_possible_connections = total_clusters * (total_clusters - 1) // 2 if total_clusters > 1 else 0
            
            # Calculate network density (ratio of actual to possible connections)
            network_density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
            
            # Calculate cluster distribution per layer
            cluster_counts = {}
            layer_connections = {}
            
            # Track cross-layer connectivity - which layers connect to which
            # Format: {(layer1_idx, layer2_idx): count}
            cross_layer_matrix = {}
            # Format: {layer_idx: {connected_layer_idx: count}}
            layer_connectivity = {}
            
            if "id_mapping" in cluster_paths_data:
                # Count clusters per layer
                for cluster_id, mapping in cluster_paths_data["id_mapping"].items():
                    layer_name = mapping["layer_name"]
                    layer_idx = mapping["layer_idx"]
                    
                    if layer_name not in cluster_counts:
                        cluster_counts[layer_name] = 0
                        layer_connections[layer_name] = 0
                        
                    # Initialize layer connectivity tracking
                    if layer_idx not in layer_connectivity:
                        layer_connectivity[layer_idx] = {}
                        
                    cluster_counts[layer_name] += 1
                
                # Count connections per layer and cross-layer connections
                for (id1, id2), similarity in strong_connections.items():
                    if str(id1) in cluster_paths_data["id_mapping"] and str(id2) in cluster_paths_data["id_mapping"]:
                        # Get layer info for both clusters
                        layer1 = cluster_paths_data["id_mapping"][str(id1)]["layer_name"]
                        layer2 = cluster_paths_data["id_mapping"][str(id2)]["layer_name"]
                        layer1_idx = cluster_paths_data["id_mapping"][str(id1)]["layer_idx"]
                        layer2_idx = cluster_paths_data["id_mapping"][str(id2)]["layer_idx"]
                        
                        # Count connections for each layer
                        layer_connections[layer1] = layer_connections.get(layer1, 0) + 1
                        layer_connections[layer2] = layer_connections.get(layer2, 0) + 1
                        
                        # Add to cross-layer connectivity matrix if different layers
                        if layer1_idx != layer2_idx:
                            # Sort indices to ensure we don't double count
                            min_idx = min(layer1_idx, layer2_idx)
                            max_idx = max(layer1_idx, layer2_idx)
                            key = (min_idx, max_idx)
                            
                            # Increment connection count
                            cross_layer_matrix[key] = cross_layer_matrix.get(key, 0) + 1
                            
                            # Track directional connectivity for each layer
                            if layer1_idx not in layer_connectivity:
                                layer_connectivity[layer1_idx] = {}
                            if layer2_idx not in layer_connectivity:
                                layer_connectivity[layer2_idx] = {}
                                
                            layer_connectivity[layer1_idx][layer2_idx] = layer_connectivity[layer1_idx].get(layer2_idx, 0) + 1
                            layer_connectivity[layer2_idx][layer1_idx] = layer_connectivity[layer2_idx].get(layer1_idx, 0) + 1
            
            # Sort layers by their natural order (layer1, layer2, etc.)
            sorted_layers = sorted(cluster_counts.keys(), key=lambda x: 
                                  int(x.replace("layer", "")) if x.startswith("layer") and x[5:].isdigit() 
                                  else (0 if x == "input" else (999 if x == "output" else 500)))
            
            # Create ordered dictionaries for display
            ordered_cluster_counts = {layer: cluster_counts[layer] for layer in sorted_layers}
            ordered_layer_connections = {layer: layer_connections.get(layer, 0) for layer in sorted_layers}
            
            # Calculate metrics for each layer
            layer_metrics = {}
            for layer in sorted_layers:
                # Connectivity ratio (connections per cluster)
                cluster_count = cluster_counts[layer]
                connection_count = layer_connections.get(layer, 0)
                
                # Calculate connectivity ratio (avg connections per cluster)
                connectivity_ratio = connection_count / max(1, cluster_count)
                
                layer_metrics[layer] = {
                    "cluster_count": cluster_count,
                    "connection_count": connection_count,
                    "connectivity_ratio": connectivity_ratio
                }
            
            # Calculate layer-to-layer connectivity metrics
            layer_pairs = []
            
            # Import the function for friendly layer names
            from visualization.traj_plot import get_friendly_layer_name
            
            # Get mapping from layer_idx to layer_name
            idx_to_layer_name = {}
            for _, mapping in cluster_paths_data.get("id_mapping", {}).items():
                idx_to_layer_name[mapping["layer_idx"]] = mapping["layer_name"]
                
            # Process cross-layer connections
            for (idx1, idx2), count in cross_layer_matrix.items():
                if idx1 in idx_to_layer_name and idx2 in idx_to_layer_name:
                    layer1 = idx_to_layer_name[idx1]
                    layer2 = idx_to_layer_name[idx2]
                    
                    # Get friendly names
                    try:
                        layer1_name = get_friendly_layer_name(layer1)
                        layer2_name = get_friendly_layer_name(layer2)
                    except Exception:
                        # Fall back to original names if there's an error
                        layer1_name = layer1
                        layer2_name = layer2
                    
                    # Calculate the layer distance
                    layer_distance = abs(idx2 - idx1)
                    
                    # Add to layer pairs
                    layer_pairs.append({
                        "layer1": layer1_name,
                        "layer2": layer2_name,
                        "layer1_idx": idx1,
                        "layer2_idx": idx2,
                        "count": count,
                        "distance": layer_distance
                    })
            
            # Sort by count (descending)
            layer_pairs = sorted(layer_pairs, key=lambda x: x["count"], reverse=True)
            
            # Calculate adjacency statistics - count connections to immediately adjacent layers
            adjacent_connections = sum(1 for pair in layer_pairs if pair["distance"] == 1)
            adjacent_percent = (adjacent_connections / max(1, len(layer_pairs))) * 100 if layer_pairs else 0
            
            # Count long-distance connections (distance > 2)
            long_distance_connections = sum(1 for pair in layer_pairs if pair["distance"] > 2)
            long_distance_percent = (long_distance_connections / max(1, len(layer_pairs))) * 100 if layer_pairs else 0
            
            # Calculate layer connectivity stats
            connectivity_stats = {
                "total_layer_pairs": len(layer_pairs),
                "adjacent_connections": adjacent_connections,
                "adjacent_percent": adjacent_percent,
                "long_distance_connections": long_distance_connections,
                "long_distance_percent": long_distance_percent,
                "layer_pairs": layer_pairs[:5],  # Only include top 5 for display
                "layer_connectivity": layer_connectivity
            }
            
            # Prepare statistics data
            stats_data = {
                "total_connections": total_connections,
                "strong_connections": strong_connection_count,
                "strong_percent": strong_percent,
                "cross_layer_connections": cross_layer_connections,
                "avg_similarity": avg_similarity,
                "network_density": network_density,
                "frag_stats": frag_stats if frag_stats else None,
                "total_clusters": total_clusters,
                "cluster_counts": ordered_cluster_counts,
                "layer_connections": ordered_layer_connections,
                "layer_metrics": layer_metrics,
                "similarity_distribution": similarity_distribution,
                "connectivity_stats": connectivity_stats
            }
            
            # Create formatted statistics panel
            return create_statistics_panel(stats_data)
        except Exception as e:
            return f"Error calculating network statistics: {str(e)}"

def create_statistics_panel(stats_data):
    """
    Create a styled statistics panel with sections.
    
    Args:
        stats_data: Dictionary containing the statistics to display
        
    Returns:
        Dash HTML component with the formatted statistics panel
    """
    # Style definitions for consistency
    section_style = {
        "margin": "0 0 15px 0",
        "padding": "10px",
        "borderRadius": "5px",
        "backgroundColor": "#f9f9f9",
        "boxShadow": "0px 0px 5px rgba(0,0,0,0.1)"
    }
    
    section_title_style = {
        "fontWeight": "bold",
        "borderBottom": "1px solid #ddd",
        "paddingBottom": "5px",
        "marginBottom": "10px"
    }
    
    stat_row_style = {
        "display": "flex",
        "justifyContent": "space-between",
        "marginBottom": "5px",
        "fontSize": "14px"
    }
    
    stat_value_style = {
        "fontWeight": "bold"
    }
    
    # Create sections
    sections = []
    
    # 1. Summary Section
    summary_section = html.Div([
        html.Div("Network Summary", style=section_title_style),
        html.Div([
            html.Div([
                html.Span("Total Clusters:", style={"width": "65%"}),
                html.Span(f"{stats_data.get('total_clusters', 0)}", style=stat_value_style)
            ], style=stat_row_style),
            html.Div([
                html.Span("Total Connections:", style={"width": "65%"}),
                html.Span(f"{stats_data.get('total_connections', 0)}", style=stat_value_style)
            ], style=stat_row_style),
            html.Div([
                html.Span("Strong Connections:", style={"width": "65%"}),
                html.Span(
                    f"{stats_data.get('strong_connections', 0)} " +
                    f"({stats_data.get('strong_percent', 0):.1f}%)", 
                    style=stat_value_style
                )
            ], style=stat_row_style),
            html.Div([
                html.Span("Cross-Layer Connections:", style={"width": "65%"}),
                html.Span(f"{stats_data.get('cross_layer_connections', 0)}", style=stat_value_style)
            ], style=stat_row_style),
            html.Div([
                html.Span("Average Similarity:", style={"width": "65%"}),
                html.Span(f"{stats_data.get('avg_similarity', 0):.3f}", style=stat_value_style)
            ], style=stat_row_style),
            html.Div([
                html.Span("Network Density:", style={"width": "65%"}),
                html.Span(f"{stats_data.get('network_density', 0):.3f}", style=stat_value_style)
            ], style=stat_row_style)
        ])
    ], style=section_style)
    sections.append(summary_section)
    
    # 2. Cluster Distribution Section
    if stats_data.get('cluster_counts'):
        cluster_counts = stats_data['cluster_counts']
        layer_metrics = stats_data.get('layer_metrics', {})
        
        # Create rows for each layer
        cluster_rows = []
        
        for layer, count in cluster_counts.items():
            # Get layer metrics
            metrics = layer_metrics.get(layer, {})
            connection_count = metrics.get('connection_count', 0)
            connectivity_ratio = metrics.get('connectivity_ratio', 0)
            
            # Create a row for this layer
            layer_row = html.Div([
                html.Div([
                    html.Span(f"{layer}:", style={"width": "30%", "fontWeight": "bold"}),
                    html.Span(f"{count} clusters", style={"width": "30%"}),
                    html.Span(f"{connection_count} connections", style={"width": "40%"})
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "3px"}),
                
                # Add a mini bar to visualize connectivity ratio
                html.Div([
                    html.Div(
                        style={
                            "width": f"{min(100, max(5, connectivity_ratio * 15))}%",  # Scale for visibility
                            "height": "4px",
                            "backgroundColor": "#4CAF50",
                            "borderRadius": "2px"
                        }
                    )
                ], style={"width": "100%", "backgroundColor": "#f0f0f0", "height": "4px", "marginBottom": "8px"})
            ])
            
            cluster_rows.append(layer_row)
        
        # Create the cluster distribution section
        cluster_section = html.Div([
            html.Div("Cluster Distribution", style=section_title_style),
            html.Div(cluster_rows),
            
            # Add a legend for the connectivity bars
            html.Div([
                html.Div("Bar length indicates average connections per cluster", 
                        style={"fontSize": "11px", "color": "#666", "marginTop": "5px", "fontStyle": "italic"})
            ])
        ], style=section_style)
        
        sections.append(cluster_section)
    
    # 3. Similarity Distribution Section
    if stats_data.get('similarity_distribution'):
        sim_dist = stats_data['similarity_distribution']
        
        # Get distribution values
        low = sim_dist.get('low', {})
        medium = sim_dist.get('medium', {})
        high = sim_dist.get('high', {})
        threshold = sim_dist.get('threshold', 0.5)
        
        # Calculate bar widths based on percentages (min 5% for visibility)
        low_width = max(5, low.get('percent', 0))
        medium_width = max(5, medium.get('percent', 0))
        high_width = max(5, high.get('percent', 0))
        
        # Create the similarity distribution section
        similarity_section = html.Div([
            html.Div("Similarity Distribution", style=section_title_style),
            
            # Distribution bars
            html.Div([
                # Low similarity bar (red)
                html.Div([
                    html.Div(style={
                        "width": f"{low_width}%",
                        "height": "15px",
                        "backgroundColor": "#f44336",  # Red
                        "borderRadius": "3px 0 0 3px" if low_width > 5 else "3px",
                        "display": "inline-block"
                    }),
                    # Medium similarity bar (yellow/orange)
                    html.Div(style={
                        "width": f"{medium_width}%",
                        "height": "15px",
                        "backgroundColor": "#ff9800",  # Orange
                        "display": "inline-block"
                    }),
                    # High similarity bar (green)
                    html.Div(style={
                        "width": f"{high_width}%",
                        "height": "15px",
                        "backgroundColor": "#4caf50",  # Green
                        "borderRadius": "0 3px 3px 0" if high_width > 5 else "3px",
                        "display": "inline-block"
                    })
                ], style={"width": "100%", "whiteSpace": "nowrap", "marginBottom": "5px"}),
                
                # Legend with counts
                html.Div([
                    html.Div([
                        html.Div(style={
                            "width": "12px", 
                            "height": "12px", 
                            "backgroundColor": "#f44336", 
                            "display": "inline-block",
                            "marginRight": "5px",
                            "borderRadius": "2px"
                        }),
                        html.Span(f"Low (<0.3): {low.get('count', 0)} ({low.get('percent', 0):.1f}%)")
                    ], style={"display": "inline-block", "marginRight": "15px"}),
                    
                    html.Div([
                        html.Div(style={
                            "width": "12px", 
                            "height": "12px", 
                            "backgroundColor": "#ff9800", 
                            "display": "inline-block",
                            "marginRight": "5px",
                            "borderRadius": "2px"
                        }),
                        html.Span(f"Medium (0.3-0.7): {medium.get('count', 0)} ({medium.get('percent', 0):.1f}%)")
                    ], style={"display": "inline-block", "marginRight": "15px"}),
                    
                    html.Div([
                        html.Div(style={
                            "width": "12px", 
                            "height": "12px", 
                            "backgroundColor": "#4caf50", 
                            "display": "inline-block",
                            "marginRight": "5px",
                            "borderRadius": "2px"
                        }),
                        html.Span(f"High (>0.7): {high.get('count', 0)} ({high.get('percent', 0):.1f}%)")
                    ], style={"display": "inline-block"})
                ], style={"fontSize": "13px", "marginTop": "5px"}),
                
                # Current threshold indicator
                html.Div([
                    html.Div("Current threshold: ", style={"display": "inline-block", "marginRight": "5px"}),
                    html.Div(f"{threshold:.2f}", style={
                        "display": "inline-block", 
                        "fontWeight": "bold",
                        "backgroundColor": "#e0e0e0",
                        "padding": "2px 6px",
                        "borderRadius": "3px"
                    })
                ], style={"fontSize": "13px", "marginTop": "10px"})
            ])
        ], style=section_style)
        
        sections.append(similarity_section)
    else:
        # Fallback if no similarity distribution data
        similarity_section = html.Div([
            html.Div("Similarity Distribution", style=section_title_style),
            html.Div("No similarity distribution data available.")
        ], style=section_style)
        sections.append(similarity_section)
    
    # 4. Cross-Layer Connectivity Section
    if stats_data.get('connectivity_stats'):
        conn_stats = stats_data['connectivity_stats']
        
        # Create rows for layer pairs
        layer_pair_rows = []
        for pair in conn_stats.get('layer_pairs', []):
            layer_pair_rows.append(
                html.Div([
                    html.Span(f"{pair['layer1']} → {pair['layer2']}:", style={"width": "65%"}),
                    html.Span(f"{pair['count']} connections", style=stat_value_style)
                ], style=stat_row_style)
            )
        
        # Create connection heatmap indicator
        layer_connectivity = conn_stats.get('layer_connectivity', {})
        layer_indices = sorted(layer_connectivity.keys())
        max_connections = 1  # Avoid division by zero
        
        # Find maximum connections for color scaling
        for idx1 in layer_indices:
            for idx2 in layer_connectivity.get(idx1, {}).keys():
                max_connections = max(max_connections, layer_connectivity[idx1].get(idx2, 0))
        
        # Create layer connectivity heatmap
        heat_cells = []
        if layer_indices:
            # Create header row with layer indices
            header_row = html.Div([html.Span("→", style={"width": "15%"})] + [
                html.Span(f"L{idx}", style={"width": f"{85/max(1, len(layer_indices))}%", "textAlign": "center", "fontSize": "10px"})
                for idx in layer_indices
            ], style={"display": "flex", "marginBottom": "2px", "fontWeight": "bold"})
            
            heat_cells.append(header_row)
            
            # Create rows for each layer
            for idx1 in layer_indices:
                # Create cells for this row
                cells = [html.Span(f"L{idx1}", style={"width": "15%", "fontWeight": "bold", "fontSize": "10px"})]
                
                # Add a cell for each target layer
                for idx2 in layer_indices:
                    count = layer_connectivity.get(idx1, {}).get(idx2, 0)
                    if idx1 == idx2:
                        # Diagonal cell (same layer)
                        intensity = "#f0f0f0"  # Light gray for diagonal
                    else:
                        # Color intensity based on connection count
                        intensity_val = min(1.0, count / max_connections)
                        # Green gradient from light to dark
                        r = int(240 - intensity_val * 150)
                        g = int(240 - intensity_val * 40)
                        b = int(240 - intensity_val * 140)
                        intensity = f"rgb({r},{g},{b})"
                    
                    # Create the cell with background color
                    cell_content = ""
                    if count > 0:
                        cell_content = str(count)
                        
                    cell = html.Span(
                        cell_content,
                        style={
                            "width": f"{85/max(1, len(layer_indices))}%", 
                            "backgroundColor": intensity,
                            "textAlign": "center",
                            "fontSize": "10px",
                            "padding": "2px 0",
                            "margin": "1px",
                            "borderRadius": "2px"
                        }
                    )
                    cells.append(cell)
                
                # Add this row to the heatmap
                row = html.Div(cells, style={"display": "flex", "marginBottom": "1px"})
                heat_cells.append(row)
        
        # Create the cross-layer connectivity section
        connectivity_section = html.Div([
            html.Div("Cross-Layer Connectivity", style=section_title_style),
            
            # Metrics subsection
            html.Div([
                html.Div([
                    html.Span("Total Cross-Layer Pairs:", style={"width": "65%"}),
                    html.Span(f"{conn_stats.get('total_layer_pairs', 0)}", style=stat_value_style)
                ], style=stat_row_style),
                html.Div([
                    html.Span("Adjacent Layer Connections:", style={"width": "65%"}),
                    html.Span(
                        f"{conn_stats.get('adjacent_connections', 0)} " +
                        f"({conn_stats.get('adjacent_percent', 0):.1f}%)", 
                        style=stat_value_style
                    )
                ], style=stat_row_style),
                html.Div([
                    html.Span("Long-Distance Connections (>2 layers):", style={"width": "65%"}),
                    html.Span(
                        f"{conn_stats.get('long_distance_connections', 0)} " +
                        f"({conn_stats.get('long_distance_percent', 0):.1f}%)", 
                        style=stat_value_style
                    )
                ], style=stat_row_style),
            ], style={"marginBottom": "10px"}),
            
            # Top layer pairs subsection
            html.Div([
                html.Div("Top Layer Pairs", style={"fontWeight": "bold", "marginBottom": "5px", "fontSize": "13px"}),
                html.Div(layer_pair_rows)
            ], style={"marginBottom": "15px"}) if layer_pair_rows else None,
            
            # Layer connectivity heatmap
            html.Div([
                html.Div("Layer Connectivity Matrix", style={"fontWeight": "bold", "marginBottom": "5px", "fontSize": "13px"}),
                html.Div(heat_cells, style={"marginTop": "5px"}),
                html.Div("Color intensity indicates connection strength between layers", 
                        style={"fontSize": "11px", "fontStyle": "italic", "color": "#666", "marginTop": "5px"})
            ]) if heat_cells else None
        ], style=section_style)
        
        sections.append(connectivity_section)
    
    # 5. Fragmentation Statistics Section
    if stats_data.get('frag_stats'):
        frag_stats = stats_data['frag_stats']
        fragmentation_section = html.Div([
            html.Div("Path Fragmentation", style=section_title_style),
            html.Div([
                html.Div([
                    html.Span("Mean Fragmentation:", style={"width": "65%"}),
                    html.Span(f"{frag_stats.get('mean', 0):.3f}", style=stat_value_style)
                ], style=stat_row_style),
                html.Div([
                    html.Span("Median Fragmentation:", style={"width": "65%"}),
                    html.Span(f"{frag_stats.get('median', 0):.3f}", style=stat_value_style)
                ], style=stat_row_style),
                html.Div([
                    html.Span("High Fragmentation Paths:", style={"width": "65%"}),
                    html.Span(f"{frag_stats.get('high_count', 0)}", style=stat_value_style)
                ], style=stat_row_style),
                html.Div([
                    html.Span("Low Fragmentation Paths:", style={"width": "65%"}),
                    html.Span(f"{frag_stats.get('low_count', 0)}", style=stat_value_style)
                ], style=stat_row_style)
            ])
        ], style=section_style)
        sections.append(fragmentation_section)
    
    # Combine all sections into a panel
    panel = html.Div(sections, style={"width": "100%", "maxWidth": "600px"})
    
    return panel
    
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
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                
                if error_type == "missing_data":
                    # Create a more informative error message
                    message = cluster_paths_data.get("message", "Missing similarity data")
                    command = cluster_paths_data.get("command", "")
                    
                    return html.Div([
                        html.Div(
                            "Missing Similarity Data", 
                            style={"color": "red", "fontWeight": "bold", "fontSize": 18, "marginBottom": "10px"}
                        ),
                        html.Div(message, style={"marginBottom": "10px"}),
                        html.Div("Run this command to generate the data:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                        html.Pre(command, style={"backgroundColor": "#f0f0f0", "padding": "10px", "borderRadius": "3px"})
                    ])
                
                # Handle other error types
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return f"Error: {error_msg}"
            
            # Check for similarity data
            if "similarity" not in cluster_paths_data:
                return "No similarity data available. Run cluster_paths.py with --compute_similarity option."
            
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
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                
                if error_type == "missing_data":
                    # Create a more informative error message
                    message = cluster_paths_data.get("message", "Missing similarity data")
                    command = cluster_paths_data.get("command", "")
                    
                    return html.Div([
                        html.Div(
                            "Missing Similarity Data", 
                            style={"color": "red", "fontWeight": "bold", "fontSize": 18, "marginBottom": "10px"}
                        ),
                        html.Div(message, style={"marginBottom": "10px"}),
                        html.Div("Run this command to generate the data:", style={"fontWeight": "bold", "marginBottom": "5px"}),
                        html.Pre(command, style={"backgroundColor": "#f0f0f0", "padding": "10px", "borderRadius": "3px"})
                    ])
                
                # Handle other error types
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return f"Error: {error_msg}"
            
            # Check for similarity data
            if "similarity" not in cluster_paths_data:
                return "No similarity data available. Run cluster_paths.py with --compute_similarity option."
            
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
            
    # Cross-Layer Visualizations Callbacks
    
    @app.callback(
        Output("sim-centroid-similarity-graph", "figure"),
        [Input("update-network-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("similarity-config-radio", "value"),
         State("layer-clusters-store", "data")]
    )
    def update_centroid_similarity_graph(n_clicks, dataset, seed, config, stored_clusters):
        """Update centroid similarity heatmap visualization."""
        if n_clicks == 0 or not dataset or not stored_clusters:
            return go.Figure().update_layout(
                title="Click 'Update Network' to generate visualizations",
                height=500
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
                    height=500
                )
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return go.Figure().update_layout(
                    title=f"Error: {error_msg}",
                    height=500
                )
            
            # Extract information from cluster_paths_data
            if "unique_centroid_distances" in cluster_paths_data and "id_mapping" in cluster_paths_data:
                # Build centroid similarity matrices from the existing data
                centroid_similarity = {}
                
                # Extract layer information
                layer_to_clusters = {}
                for unique_id, mapping in cluster_paths_data["id_mapping"].items():
                    layer_name = mapping["layer_name"]
                    if layer_name not in layer_to_clusters:
                        layer_to_clusters[layer_name] = []
                    layer_to_clusters[layer_name].append(int(unique_id))
                
                # Sort layers by index
                sorted_layers = sorted(layer_to_clusters.keys(), key=lambda x: 
                                 int(x.replace("layer", "")) if x.startswith("layer") and x[5:].isdigit() 
                                 else (0 if x == "input" else 999))
                
                # Build similarity matrices for consecutive layer pairs
                for i in range(len(sorted_layers) - 1):
                    layer1 = sorted_layers[i]
                    layer2 = sorted_layers[i + 1]
                    
                    # Get clusters in each layer
                    clusters1 = layer_to_clusters[layer1]
                    clusters2 = layer_to_clusters[layer2]
                    
                    # Create similarity matrix
                    sim_matrix = np.zeros((len(clusters1), len(clusters2)))
                    
                    # Fill in similarity values
                    for i1, cluster1 in enumerate(clusters1):
                        for i2, cluster2 in enumerate(clusters2):
                            key = f"{cluster1},{cluster2}"
                            if key in cluster_paths_data.get("unique_centroid_distances", {}):
                                # Convert distance to similarity (1 - normalized_distance)
                                distance = cluster_paths_data["unique_centroid_distances"][key]
                                similarity = 1.0 - distance
                                sim_matrix[i1, i2] = similarity
                    
                    # Store the similarity matrix
                    centroid_similarity[(layer1, layer2)] = sim_matrix
                
                # Create heatmap visualization
                from visualization.traj_plot import get_friendly_layer_name
                fig = plot_centroid_similarity_heatmap(
                    centroid_similarity,
                    colorscale="Viridis",
                    height=500,
                    width=1000,
                    get_friendly_layer_name=get_friendly_layer_name
                )
                
                return fig
            else:
                return go.Figure().update_layout(
                    title="No centroid similarity data available in the cluster paths data",
                    height=500
                )
                
        except Exception as e:
            import traceback
            error_msg = f"Error creating centroid similarity graph: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=500
            )

    @app.callback(
        Output("sim-membership-overlap-graph", "figure"),
        [Input("update-network-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("similarity-config-radio", "value"),
         State("layer-clusters-store", "data")]
    )
    def update_membership_overlap_graph(n_clicks, dataset, seed, config, stored_clusters):
        """Update membership overlap Sankey diagram."""
        if n_clicks == 0 or not dataset or not stored_clusters:
            return go.Figure().update_layout(
                title="Click 'Update Network' to generate visualizations",
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
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return go.Figure().update_layout(
                    title=f"Error: {error_msg}",
                    height=600
                )
            
            # Extract path information for calculating membership overlap
            if "paths" in cluster_paths_data and "id_mapping" in cluster_paths_data:
                # Prepare data structures for membership overlap
                label_clusters = {}
                membership_overlap = {}
                
                # Extract layer mapping
                layer_to_idx = {}
                id_to_layer = {}
                for unique_id, mapping in cluster_paths_data["id_mapping"].items():
                    layer_name = mapping["layer_name"]
                    original_id = mapping["original_id"]
                    layer_to_idx[layer_name] = mapping["layer_idx"]
                    id_to_layer[int(unique_id)] = (layer_name, original_id)
                
                # Sort layers by index
                sorted_layers = sorted(layer_to_idx.keys(), key=lambda x: layer_to_idx[x])
                
                # Initialize label clusters for each layer
                for layer in sorted_layers:
                    label_clusters[layer] = {
                        "labels": []
                    }
                
                # Process paths to create clusters for each layer
                paths = cluster_paths_data["paths"]
                for path in paths:
                    for layer_idx, cluster_id in enumerate(path):
                        if cluster_id >= 0 and layer_idx < len(sorted_layers):
                            layer = sorted_layers[layer_idx]
                            label_clusters[layer]["labels"].append(cluster_id)
                
                # Convert lists to arrays
                for layer in label_clusters:
                    if "labels" in label_clusters[layer] and label_clusters[layer]["labels"]:
                        label_clusters[layer]["labels"] = np.array(label_clusters[layer]["labels"])
                
                # Compute membership overlap between consecutive layers
                min_overlap = 0.1  # Minimum overlap to display
                
                for i in range(len(sorted_layers) - 1):
                    layer1 = sorted_layers[i]
                    layer2 = sorted_layers[i + 1]
                    
                    if "labels" in label_clusters[layer1] and "labels" in label_clusters[layer2]:
                        overlap = compute_membership_overlap(
                            {layer1: label_clusters[layer1], layer2: label_clusters[layer2]},
                            [(layer1, layer2)]
                        )
                        
                        if (layer1, layer2) in overlap:
                            membership_overlap[(layer1, layer2)] = overlap[(layer1, layer2)]
                
                # Create Sankey diagram
                from visualization.traj_plot import get_friendly_layer_name
                fig = plot_membership_overlap_sankey(
                    membership_overlap,
                    label_clusters,
                    min_overlap=min_overlap,
                    height=600,
                    width=1000,
                    get_friendly_layer_name=get_friendly_layer_name
                )
                
                return fig
            else:
                return go.Figure().update_layout(
                    title="No path data available for membership overlap calculation",
                    height=600
                )
                
        except Exception as e:
            import traceback
            error_msg = f"Error creating membership overlap graph: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=600
            )

    @app.callback(
        Output("sim-trajectory-fragmentation-graph", "figure"),
        [Input("update-network-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("similarity-config-radio", "value"),
         State("layer-clusters-store", "data")]
    )
    def update_trajectory_fragmentation_graph(n_clicks, dataset, seed, config, stored_clusters):
        """Update trajectory fragmentation bar chart."""
        if n_clicks == 0 or not dataset or not stored_clusters:
            return go.Figure().update_layout(
                title="Click 'Update Network' to generate visualizations",
                height=400
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
                    height=400
                )
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return go.Figure().update_layout(
                    title=f"Error: {error_msg}",
                    height=400
                )
            
            # Check if fragmentation scores are available
            if "similarity" in cluster_paths_data and "fragmentation_scores" in cluster_paths_data["similarity"]:
                # Extract fragmentation scores by layer
                frag_data = cluster_paths_data["similarity"]["fragmentation_scores"]
                
                # Check if layer scores are available
                if "layer_scores" in frag_data:
                    fragmentation_scores = frag_data["layer_scores"]
                    
                    # Use the provided fragmentation scores
                    from visualization.traj_plot import get_friendly_layer_name
                    fig = plot_trajectory_fragmentation_bars(
                        fragmentation_scores,
                        height=400,
                        width=800,
                        get_friendly_layer_name=get_friendly_layer_name
                    )
                    
                    return fig
                else:
                    return go.Figure().update_layout(
                        title="No layer-specific fragmentation scores available",
                        height=400
                    )
            else:
                return go.Figure().update_layout(
                    title="No fragmentation data available in the cluster paths data",
                    height=400
                )
                
        except Exception as e:
            import traceback
            error_msg = f"Error creating trajectory fragmentation graph: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=400
            )

    @app.callback(
        Output("sim-path-density-graph", "figure"),
        [Input("update-network-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("similarity-config-radio", "value"),
         State("layer-clusters-store", "data")]
    )
    def update_path_density_graph(n_clicks, dataset, seed, config, stored_clusters):
        """Update path density network graph."""
        if n_clicks == 0 or not dataset or not stored_clusters:
            return go.Figure().update_layout(
                title="Click 'Update Network' to generate visualizations",
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
            
            # Check for error status in the response
            if isinstance(cluster_paths_data, dict) and "status" in cluster_paths_data and cluster_paths_data["status"] == "error":
                error_type = cluster_paths_data.get("error_type", "unknown")
                error_msg = cluster_paths_data.get("message", "Unknown error occurred")
                return go.Figure().update_layout(
                    title=f"Error: {error_msg}",
                    height=600
                )
            
            # Create path graph from cluster_paths_data
            if "paths" in cluster_paths_data and "id_mapping" in cluster_paths_data:
                # Create a NetworkX graph
                G = nx.DiGraph()
                
                # Add nodes (clusters)
                for unique_id, mapping in cluster_paths_data["id_mapping"].items():
                    G.add_node(
                        int(unique_id),
                        layer=mapping["layer_name"],
                        cluster_id=mapping["original_id"],
                        layer_idx=mapping["layer_idx"]
                    )
                
                # Process paths to add edges (connections between clusters)
                paths = cluster_paths_data["paths"]
                edges = {}
                
                for path in paths:
                    for i in range(len(path) - 1):
                        source = path[i]
                        target = path[i + 1]
                        
                        if source >= 0 and target >= 0:  # Skip invalid cluster IDs
                            edge = (source, target)
                            if edge in edges:
                                edges[edge] += 1
                            else:
                                edges[edge] = 1
                
                # Add weighted edges to the graph
                for (source, target), weight in edges.items():
                    if source in G and target in G:  # Ensure nodes exist
                        # Normalize weight (divide by total paths)
                        norm_weight = weight / len(paths)
                        
                        # Add edge with weight attribute
                        G.add_edge(source, target, weight=norm_weight)
                
                # Create network visualization
                from visualization.traj_plot import get_friendly_layer_name
                fig = plot_path_density_network(
                    G,
                    layout="multipartite",
                    height=600,
                    width=1000,
                    get_friendly_layer_name=get_friendly_layer_name
                )
                
                return fig
            else:
                return go.Figure().update_layout(
                    title="No path data available for path density visualization",
                    height=600
                )
                
        except Exception as e:
            import traceback
            error_msg = f"Error creating path density graph: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=600
            )