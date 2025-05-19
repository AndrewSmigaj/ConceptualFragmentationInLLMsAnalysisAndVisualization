"""
Path Fragmentation View for the Neural Network Trajectories Dashboard.

This module provides the layout and callbacks for the path fragmentation tab,
which visualizes the fragmentation scores for different paths through the network.
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import importlib
from typing import Dict, List, Tuple, Any, Optional

def create_path_fragmentation_tab():
    """Create the layout for the Path Fragmentation tab."""
    return dcc.Tab(label="Path Fragmentation", children=[
        html.Div([
            html.H4("Path Fragmentation Analysis", style={"textAlign": "center"}),
            
            # Controls row
            html.Div([
                # Left column: Configuration selection
                html.Div([
                    html.Label("Configuration:"),
                    dcc.RadioItems(
                        id="path-config-radio",
                        options=[
                            {"label": "Baseline", "value": "baseline"},
                            {"label": "Regularized", "value": "regularized"}
                        ],
                        value="baseline",
                        inline=True
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                # Middle column: Fragmentation filter
                html.Div([
                    html.Label("Fragmentation Filter:"),
                    dcc.RadioItems(
                        id="fragmentation-filter-radio",
                        options=[
                            {"label": "All Paths", "value": "all"},
                            {"label": "High Fragmentation", "value": "high"},
                            {"label": "Low Fragmentation", "value": "low"}
                        ],
                        value="all",
                        inline=True
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
                
                # Right column: Sort order
                html.Div([
                    html.Label("Sort By:"),
                    dcc.Dropdown(
                        id="path-sort-dropdown",
                        options=[
                            {"label": "Fragmentation Score (High to Low)", "value": "frag_desc"},
                            {"label": "Fragmentation Score (Low to High)", "value": "frag_asc"},
                            {"label": "Path Length (High to Low)", "value": "length_desc"},
                            {"label": "Path Length (Low to High)", "value": "length_asc"},
                            {"label": "Frequency (High to Low)", "value": "freq_desc"}
                        ],
                        value="frag_desc"
                    ),
                ], style={"width": "30%", "display": "inline-block", "padding": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between"}),
            
            # Display row
            html.Div([
                # Update button
                html.Button(
                    "Update Paths", 
                    id="update-paths-button", 
                    n_clicks=0,
                    style={"backgroundColor": "#4CAF50", "color": "white", "padding": "10px", "margin": "10px"}
                ),
            ], style={"textAlign": "center"}),
            
            # Path fragmentation histogram
            dcc.Loading(
                id="loading-fragmentation-histogram",
                type="circle",
                children=[
                    dcc.Graph(
                        id="path-fragmentation-histogram",
                        style={"height": "300px"},
                        config={
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "path_fragmentation_histogram",
                                "height": 300,
                                "width": 800
                            }
                        }
                    )
                ]
            ),
            
            # Path table
            html.Div([
                html.H5("Paths by Fragmentation Score", style={"textAlign": "center"}),
                dcc.Loading(
                    id="loading-path-table",
                    type="circle",
                    children=[
                        html.Div(id="path-table-container", style={"padding": "15px"})
                    ]
                )
            ]),
            
            # Path details
            html.Div([
                html.H5("Path Details", style={"textAlign": "center"}),
                dcc.Loading(
                    id="loading-path-details",
                    type="circle",
                    children=[
                        html.Div(id="path-details-container", style={"padding": "15px"})
                    ]
                )
            ]),
            
            # Path visualization
            dcc.Loading(
                id="loading-path-visualization",
                type="circle",
                children=[
                    dcc.Graph(
                        id="path-visualization-graph",
                        style={"height": "400px", "display": "none"},
                        config={
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "path_visualization",
                                "height": 400,
                                "width": 800
                            }
                        }
                    )
                ]
            ),
        ])
    ])

def create_path_fragmentation_histogram(
    paths_data: Dict[str, Any],
    fragmentation_filter: str = "all"
) -> go.Figure:
    """
    Create a histogram of path fragmentation scores.
    
    Args:
        paths_data: Dictionary containing path data with fragmentation scores
        fragmentation_filter: Filter for fragmentation scores ('all', 'high', or 'low')
        
    Returns:
        Plotly Figure object with the fragmentation histogram
    """
    # Extract fragmentation scores
    if not paths_data or "similarity" not in paths_data or "fragmentation_scores" not in paths_data["similarity"]:
        return go.Figure().update_layout(
            title="No fragmentation data available",
            height=300
        )
    
    # Get scores from fragmentation data
    frag_data = paths_data["similarity"]["fragmentation_scores"]
    
    if "scores" not in frag_data or not frag_data["scores"]:
        return go.Figure().update_layout(
            title="No fragmentation scores available",
            height=300
        )
    
    scores = frag_data["scores"]
    
    # Filter scores based on selection
    if fragmentation_filter == "high":
        threshold = frag_data.get("high_threshold", 0.7)
        filtered_scores = [s for s in scores if s >= threshold]
        title = f"High Fragmentation Paths (≥ {threshold:.2f})"
    elif fragmentation_filter == "low":
        threshold = frag_data.get("low_threshold", 0.3)
        filtered_scores = [s for s in scores if s <= threshold]
        title = f"Low Fragmentation Paths (≤ {threshold:.2f})"
    else:
        filtered_scores = scores
        title = "All Path Fragmentation Scores"
    
    # Create histogram
    fig = px.histogram(
        x=filtered_scores,
        nbins=20,
        labels={"x": "Fragmentation Score"},
        title=title
    )
    
    # Add mean and median lines
    if filtered_scores:
        mean_score = np.mean(filtered_scores)
        median_score = np.median(filtered_scores)
        
        fig.add_shape(
            type="line",
            x0=mean_score,
            x1=mean_score,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
            name="Mean"
        )
        
        fig.add_shape(
            type="line",
            x0=median_score,
            x1=median_score,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="blue", width=2, dash="dot"),
            name="Median"
        )
        
        # Add annotations for mean and median
        fig.add_annotation(
            x=mean_score,
            y=1,
            text=f"Mean: {mean_score:.2f}",
            showarrow=True,
            arrowhead=1,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="red")
        )
        
        fig.add_annotation(
            x=median_score,
            y=0.85,
            text=f"Median: {median_score:.2f}",
            showarrow=True,
            arrowhead=1,
            xanchor="right",
            yanchor="bottom",
            font=dict(color="blue")
        )
    
    # Update layout
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            title="Fragmentation Score",
            range=[0, 1],
            tickformat=".1f"
        ),
        yaxis=dict(
            title="Number of Paths"
        )
    )
    
    return fig

def create_path_table(
    paths_data: Dict[str, Any],
    fragmentation_filter: str = "all",
    sort_by: str = "frag_desc",
    get_friendly_layer_name=None
) -> dash_table.DataTable:
    """
    Create a data table of paths with fragmentation scores.
    
    Args:
        paths_data: Dictionary containing path data with fragmentation scores
        fragmentation_filter: Filter for fragmentation scores ('all', 'high', or 'low')
        sort_by: Sort order for the paths
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Dash DataTable with path data
    """
    # Check for required data
    if not paths_data or "similarity" not in paths_data or "fragmentation_scores" not in paths_data["similarity"]:
        return html.Div("No fragmentation data available.")
    
    # Get fragmentation data and path archetypes
    frag_data = paths_data["similarity"]["fragmentation_scores"]
    archetypes = paths_data.get("path_archetypes", [])
    
    if not archetypes:
        return html.Div("No path archetypes available.")
    
    # Get scores
    scores = frag_data.get("scores", [])
    high_threshold = frag_data.get("high_threshold", 0.7)
    low_threshold = frag_data.get("low_threshold", 0.3)
    
    # Prepare data for table
    rows = []
    for i, archetype in enumerate(archetypes):
        # Skip if we don't have a fragmentation score for this path
        if i >= len(scores):
            continue
            
        frag_score = scores[i]
        path_str = archetype.get("path", "")
        count = archetype.get("count", 0)
        
        # Apply fragmentation filter
        if fragmentation_filter == "high" and frag_score < high_threshold:
            continue
        elif fragmentation_filter == "low" and frag_score > low_threshold:
            continue
        
        # Get layer structure of path
        path_clusters = path_str.split("→")
        path_length = len(path_clusters)
        
        # Get optional demographics info
        demographics = ""
        if "demo_stats" in archetype:
            demo_stats = archetype["demo_stats"]
            highlights = []
            
            # Check for sex stats
            if "sex" in demo_stats:
                sex_stats = demo_stats["sex"]
                for sex, percentage in sex_stats.items():
                    if percentage > 0.7:
                        highlights.append(f"{percentage:.0%} {sex.title()}")
            
            # Check for pclass (Titanic specific)
            if "pclass" in demo_stats:
                pclass_stats = demo_stats["pclass"]
                for pclass, percentage in pclass_stats.items():
                    if percentage > 0.6:
                        highlights.append(f"{percentage:.0%} Class {pclass}")
            
            if highlights:
                demographics = ", ".join(highlights)
        
        # Format path with friendly layer names if available
        formatted_path = path_str
        if get_friendly_layer_name:
            # Get layers from path data if available
            layers = paths_data.get("layers", [])
            if layers and len(layers) == len(path_clusters):
                friendly_clusters = []
                for layer_idx, cluster_id in enumerate(path_clusters):
                    layer_name = layers[layer_idx]
                    friendly_layer = get_friendly_layer_name(layer_name)
                    friendly_clusters.append(f"{friendly_layer} C{cluster_id}")
                formatted_path = " → ".join(friendly_clusters)
        
        # Add row to table data
        row = {
            "Path ID": i,
            "Path": formatted_path,
            "Raw Path": path_str,
            "Fragmentation": frag_score,
            "Count": count,
            "Path Length": path_length,
            "Demographics": demographics
        }
        
        # Add target-specific info if available (e.g., Titanic survival)
        if "survived_rate" in archetype:
            row["Survival Rate"] = archetype["survived_rate"]
        
        rows.append(row)
    
    # Sort rows based on specified order
    if sort_by == "frag_desc":
        rows.sort(key=lambda x: x["Fragmentation"], reverse=True)
    elif sort_by == "frag_asc":
        rows.sort(key=lambda x: x["Fragmentation"])
    elif sort_by == "length_desc":
        rows.sort(key=lambda x: x["Path Length"], reverse=True)
    elif sort_by == "length_asc":
        rows.sort(key=lambda x: x["Path Length"])
    elif sort_by == "freq_desc":
        rows.sort(key=lambda x: x["Count"], reverse=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Create conditional style for fragmentation column
    style_conditions = [
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'
        },
        {
            'if': {
                'filter_query': '{Fragmentation} >= 0.7',
                'column_id': 'Fragmentation'
            },
            'backgroundColor': 'rgb(255, 200, 200)',
            'fontWeight': 'bold'
        },
        {
            'if': {
                'filter_query': '{Fragmentation} <= 0.3',
                'column_id': 'Fragmentation'
            },
            'backgroundColor': 'rgb(200, 255, 200)',
            'fontWeight': 'bold'
        }
    ]
    
    # Create table
    columns = [col for col in df.columns if col != "Raw Path"]
    table = dash_table.DataTable(
        id='path-fragmentation-table',
        columns=[{"name": col, "id": col} for col in columns],
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
        style_data_conditional=style_conditions,
        row_selectable="single"
    )
    
    # Add explanation about fragmentation scores
    explanation = html.Div([
        html.P(
            "Path fragmentation measures how much a concept changes as it passes through the network. "
            "Higher fragmentation (closer to 1.0) means more concept drift and less conceptual preservation. "
            "Lower fragmentation (closer to 0.0) indicates better preservation of concepts across layers."
        ),
        html.P(
            "Sorting by fragmentation helps identify which paths exhibit the most or least conceptual consistency."
        )
    ], style={"marginBottom": "15px"})
    
    return html.Div([explanation, table])

def create_path_detail_view(
    selected_row: Dict[str, Any],
    paths_data: Dict[str, Any],
    get_friendly_layer_name=None
) -> html.Div:
    """
    Create a detailed view of a selected path with fragmentation information.
    
    Args:
        selected_row: The selected path row from the table
        paths_data: Dictionary containing path data with fragmentation scores
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Dash layout with detailed path information
    """
    if not selected_row or not paths_data:
        return html.Div("Select a path to view details.")
    
    # Get path ID and raw path from selected row
    path_id = selected_row.get("Path ID")
    raw_path = selected_row.get("Raw Path", "")
    
    # Get similarity data if available
    similarity_data = paths_data.get("similarity", {})
    
    # Get convergent paths if available
    convergent_paths = similarity_data.get("convergent_paths", {})
    path_convergences = convergent_paths.get(str(path_id), [])
    
    # Get layer information
    layers = paths_data.get("layers", [])
    
    # Parse the raw path
    cluster_ids = raw_path.split("→")
    
    # Create path step details
    steps = []
    for i, cluster_id in enumerate(cluster_ids):
        # Get layer name if available
        layer_name = layers[i] if i < len(layers) else f"Layer {i}"
        
        # Format with friendly name if available
        if get_friendly_layer_name:
            friendly_layer = get_friendly_layer_name(layer_name)
        else:
            friendly_layer = layer_name
        
        # Create step element
        step = html.Div([
            html.Div(f"{friendly_layer}", className="path-layer-name"),
            html.Div(f"Cluster {cluster_id}", className="path-cluster-id")
        ], className="path-step")
        
        steps.append(step)
        
        # Add arrow between steps except for the last one
        if i < len(cluster_ids) - 1:
            steps.append(html.Div("→", className="path-arrow"))
    
    # Create path steps display
    path_steps = html.Div(steps, className="path-steps-container", style={
        "display": "flex",
        "alignItems": "center",
        "flexWrap": "wrap",
        "margin": "10px 0",
        "padding": "10px",
        "backgroundColor": "#f8f8f8",
        "borderRadius": "5px"
    })
    
    # Create convergence details if available
    convergence_elements = []
    if path_convergences:
        # Sort convergences by similarity (descending)
        sorted_convergences = sorted(path_convergences, key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Create table rows for convergences
        convergence_rows = []
        for conv in sorted_convergences:
            early_layer = conv.get("early_layer", 0)
            late_layer = conv.get("late_layer", 0)
            early_cluster = conv.get("early_cluster", "?")
            late_cluster = conv.get("late_cluster", "?")
            similarity = conv.get("similarity", 0)
            
            # Format layer names if available
            early_layer_name = layers[early_layer] if early_layer < len(layers) else f"Layer {early_layer}"
            late_layer_name = layers[late_layer] if late_layer < len(layers) else f"Layer {late_layer}"
            
            if get_friendly_layer_name:
                early_layer_display = get_friendly_layer_name(early_layer_name)
                late_layer_display = get_friendly_layer_name(late_layer_name)
            else:
                early_layer_display = early_layer_name
                late_layer_display = late_layer_name
            
            # Create row
            convergence_rows.append({
                "Early Layer": early_layer_display,
                "Early Cluster": early_cluster,
                "Late Layer": late_layer_display,
                "Late Cluster": late_cluster,
                "Similarity": similarity,
                "Layer Distance": late_layer - early_layer
            })
        
        # Create DataFrame
        convergence_df = pd.DataFrame(convergence_rows)
        
        # Create table
        convergence_table = dash_table.DataTable(
            id='convergence-details-table',
            columns=[{"name": col, "id": col} for col in convergence_df.columns],
            data=convergence_df.to_dict('records'),
            sort_action="native",
            page_size=5,
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'left',
                'padding': '8px'
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
                        'filter_query': '{Similarity} >= 0.7',
                        'column_id': 'Similarity'
                    },
                    'backgroundColor': 'rgb(200, 255, 200)',
                    'fontWeight': 'bold'
                }
            ]
        )
        
        # Add convergence explanation
        convergence_elements = [
            html.H6("Similarity Convergences", style={"marginTop": "15px"}),
            html.P(
                "Convergence occurs when a cluster in a later layer has high similarity to a cluster in an earlier layer, "
                "indicating concept re-emergence or preservation despite intermediate transformations."
            ),
            convergence_table
        ]
    
    # Get metadata from archetypes if available
    metadata_elements = []
    if "path_archetypes" in paths_data and path_id < len(paths_data["path_archetypes"]):
        archetype = paths_data["path_archetypes"][path_id]
        
        # Get count and demographics
        count = archetype.get("count", 0)
        
        # Extract meaningful demographic information
        if "demo_stats" in archetype:
            demo_stats = archetype["demo_stats"]
            
            # Prepare demographic sections
            demo_sections = []
            
            # Process sex distribution if available
            if "sex" in demo_stats:
                sex_stats = demo_stats["sex"]
                sex_items = []
                for sex, percentage in sex_stats.items():
                    sex_items.append(html.Li(f"{sex.title()}: {percentage:.1%}"))
                
                if sex_items:
                    demo_sections.append(html.Div([
                        html.H6("Sex Distribution:", style={"marginBottom": "5px"}),
                        html.Ul(sex_items, style={"marginTop": "0"})
                    ]))
            
            # Process class distribution (Titanic specific)
            if "pclass" in demo_stats:
                pclass_stats = demo_stats["pclass"]
                pclass_items = []
                for pclass, percentage in pclass_stats.items():
                    pclass_items.append(html.Li(f"Class {pclass}: {percentage:.1%}"))
                
                if pclass_items:
                    demo_sections.append(html.Div([
                        html.H6("Passenger Class:", style={"marginBottom": "5px"}),
                        html.Ul(pclass_items, style={"marginTop": "0"})
                    ]))
            
            # Process age distribution if available
            if "age" in demo_stats:
                age_stats = demo_stats["age"]
                if isinstance(age_stats, dict):
                    age_items = []
                    if "mean" in age_stats:
                        age_items.append(html.Li(f"Mean: {age_stats['mean']:.1f}"))
                    if "median" in age_stats:
                        age_items.append(html.Li(f"Median: {age_stats['median']:.1f}"))
                    if "min" in age_stats and "max" in age_stats:
                        age_items.append(html.Li(f"Range: {age_stats['min']:.1f} - {age_stats['max']:.1f}"))
                    
                    if age_items:
                        demo_sections.append(html.Div([
                            html.H6("Age Statistics:", style={"marginBottom": "5px"}),
                            html.Ul(age_items, style={"marginTop": "0"})
                        ]))
            
            # Create demographics panel if we have sections
            if demo_sections:
                metadata_elements.append(html.Div([
                    html.H6("Demographics", style={"marginTop": "15px"}),
                    html.Div(demo_sections, style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "20px"
                    })
                ]))
        
        # Add target-specific info (e.g., Titanic survival)
        if "survived_rate" in archetype:
            survival_rate = archetype["survived_rate"]
            survival_color = "green" if survival_rate > 0.5 else "red"
            
            metadata_elements.append(html.Div([
                html.H6("Outcome Statistics", style={"marginTop": "15px"}),
                html.Div([
                    html.Strong("Survival Rate: "),
                    html.Span(f"{survival_rate:.1%}", style={"color": survival_color})
                ])
            ]))
    
    # Create path detail card
    details_card = html.Div([
        html.H6(f"Path {path_id} Structure", style={"marginBottom": "10px"}),
        path_steps,
        html.Div([
            html.Strong("Fragmentation Score: "),
            html.Span(f"{selected_row.get('Fragmentation', 0):.3f}", style={
                "fontWeight": "bold",
                "color": "red" if selected_row.get('Fragmentation', 0) > 0.7 else 
                         "green" if selected_row.get('Fragmentation', 0) < 0.3 else "black"
            })
        ], style={"marginTop": "10px"}),
        html.Div(convergence_elements) if convergence_elements else None,
        html.Div(metadata_elements) if metadata_elements else None
    ], style={
        "backgroundColor": "white",
        "padding": "15px",
        "borderRadius": "5px",
        "boxShadow": "0 2px 5px rgba(0,0,0,0.1)"
    })
    
    return details_card

def create_path_visualization(
    selected_row: Dict[str, Any],
    paths_data: Dict[str, Any],
    get_friendly_layer_name=None
) -> go.Figure:
    """
    Create a visualization of the selected path with fragmentation information.
    
    Args:
        selected_row: The selected path row from the table
        paths_data: Dictionary containing path data with fragmentation scores
        get_friendly_layer_name: Optional function to convert layer names to display names
        
    Returns:
        Plotly Figure with the path visualization
    """
    if not selected_row or not paths_data:
        # Return empty figure that won't be displayed
        return go.Figure()
    
    # Get path ID and raw path from selected row
    path_id = selected_row.get("Path ID")
    raw_path = selected_row.get("Raw Path", "")
    frag_score = selected_row.get("Fragmentation", 0)
    
    # Get similarity data if available
    similarity_data = paths_data.get("similarity", {})
    
    # Get convergent paths if available
    convergent_paths = similarity_data.get("convergent_paths", {})
    path_convergences = convergent_paths.get(str(path_id), [])
    
    # Get layer information
    layers = paths_data.get("layers", [])
    
    # Parse the raw path
    cluster_ids = raw_path.split("→")
    
    # Create figure with two subplots (main path and fragmentation details)
    fig = make_subplots(
        rows=2, 
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1,
        subplot_titles=("Path Structure with Convergences", "Layer-to-Layer Fragmentation")
    )
    
    # X-coordinates for the main path (evenly spaced)
    x_positions = np.linspace(0, 1, len(cluster_ids))
    
    # Y-coordinate for the main path (center)
    y_main = 0.5
    
    # Node sizes based on importance (you can adjust logic)
    node_sizes = [30] * len(cluster_ids)
    
    # Add nodes for the main path
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[y_main] * len(cluster_ids),
            mode="markers",
            marker=dict(
                size=node_sizes,
                color="rgba(50, 100, 200, 0.8)",
                line=dict(width=2, color="rgba(30, 60, 120, 1)")
            ),
            text=[f"{get_friendly_layer_name(layers[i]) if get_friendly_layer_name and i < len(layers) else f'Layer {i}'}<br>Cluster {cid}" 
                  for i, cid in enumerate(cluster_ids)],
            hoverinfo="text",
            name="Main Path"
        ),
        row=1, col=1
    )
    
    # Add lines connecting the nodes
    for i in range(len(cluster_ids) - 1):
        fig.add_trace(
            go.Scatter(
                x=[x_positions[i], x_positions[i+1]],
                y=[y_main, y_main],
                mode="lines",
                line=dict(width=2, color="rgba(50, 100, 200, 0.5)"),
                hoverinfo="none",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add convergence connections if available
    if path_convergences:
        for i, conv in enumerate(path_convergences):
            early_layer = conv.get("early_layer", 0)
            late_layer = conv.get("late_layer", 0)
            similarity = conv.get("similarity", 0)
            
            # Skip if layer indices are out of range
            if early_layer >= len(x_positions) or late_layer >= len(x_positions):
                continue
            
            # X-coordinates for the convergence connection
            x_early = x_positions[early_layer]
            x_late = x_positions[late_layer]
            
            # Calculate curve parameters (arc above the main path)
            x_mid = (x_early + x_late) / 2
            y_curve = y_main + 0.2 + (similarity * 0.2)  # Higher similarity = higher arc
            
            # Create a simple curve using intermediate points
            curve_x = [x_early, x_mid, x_late]
            curve_y = [y_main, y_curve, y_main]
            
            # Add the convergence curve
            fig.add_trace(
                go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode="lines",
                    line=dict(
                        width=1 + (similarity * 3),  # Width based on similarity
                        color=f"rgba(255, 100, 100, {similarity})",  # Color alpha based on similarity
                        dash="dot"
                    ),
                    text=f"Similarity: {similarity:.3f}<br>From Layer {early_layer} to Layer {late_layer}",
                    hoverinfo="text",
                    name=f"Convergence {i+1}" if i == 0 else "",
                    showlegend=i == 0  # Only show the first one in legend
                ),
                row=1, col=1
            )
    
    # Add layer-to-layer fragmentation in the second subplot
    if len(cluster_ids) > 1:
        # Calculate fragmentation between adjacent layers
        # For simplicity, we'll use an even distribution of the total fragmentation
        # In a real implementation, you would have the actual layer-to-layer values
        x_frag = []
        y_frag = []
        text_frag = []
        
        for i in range(len(cluster_ids) - 1):
            # Position between layers
            x_frag.append((x_positions[i] + x_positions[i+1]) / 2)
            
            # Use total fragmentation divided by number of transitions as placeholder
            # In real implementation, use actual values
            layer_frag = frag_score / (len(cluster_ids) - 1)
            y_frag.append(layer_frag)
            
            # Get layer names for hover text
            layer1 = layers[i] if i < len(layers) else f"Layer {i}"
            layer2 = layers[i+1] if i+1 < len(layers) else f"Layer {i+1}"
            
            if get_friendly_layer_name:
                layer1_name = get_friendly_layer_name(layer1)
                layer2_name = get_friendly_layer_name(layer2)
            else:
                layer1_name = layer1
                layer2_name = layer2
            
            text_frag.append(f"{layer1_name} → {layer2_name}<br>Fragmentation: {layer_frag:.3f}")
        
        # Add fragmentation bars
        fig.add_trace(
            go.Bar(
                x=x_frag,
                y=y_frag,
                marker=dict(
                    color=["rgba(255,100,100,0.7)" if f > 0.5 else "rgba(100,200,100,0.7)" for f in y_frag]
                ),
                text=text_frag,
                hoverinfo="text",
                name="Layer Fragmentation"
            ),
            row=2, col=1
        )
        
        # Add horizontal line for average fragmentation
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=frag_score,
            y1=frag_score,
            line=dict(
                color="red",
                width=2,
                dash="dash"
            ),
            row=2, col=1
        )
        
        # Add annotation for average fragmentation
        fig.add_annotation(
            x=0.1,
            y=frag_score + 0.05,
            text=f"Overall: {frag_score:.3f}",
            showarrow=False,
            font=dict(color="red"),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"Path {path_id} Visualization",
        showlegend=True,
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes for first subplot (main path)
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1, col=1
    )
    
    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1, col=1
    )
    
    # Update axes for second subplot (fragmentation)
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        title="Layer Transitions",
        row=2, col=1
    )
    
    fig.update_yaxes(
        title="Fragmentation",
        range=[0, 1],
        tickformat=".1f",
        row=2, col=1
    )
    
    return fig

def register_path_fragmentation_callbacks(app):
    """Register callbacks for the Path Fragmentation tab."""
    
    @app.callback(
        Output("path-fragmentation-histogram", "figure"),
        [Input("update-paths-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("fragmentation-filter-radio", "value")]
    )
    def update_fragmentation_histogram(n_clicks, dataset, seed, fragmentation_filter):
        """Update the path fragmentation histogram."""
        if n_clicks == 0 or not dataset:
            # Return empty figure on initial load
            return go.Figure().update_layout(
                title="Click 'Update Paths' to generate fragmentation histogram",
                height=300
            )
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return go.Figure().update_layout(
                    title="No similarity data available. Run cluster_paths.py with --compute_similarity option.",
                    height=300
                )
            
            # Create histogram
            fig = create_path_fragmentation_histogram(
                cluster_paths_data,
                fragmentation_filter
            )
            
            return fig
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating fragmentation histogram: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return go.Figure().update_layout(
                title=f"Error: {str(e)}",
                height=300
            )
    
    @app.callback(
        Output("path-table-container", "children"),
        [Input("update-paths-button", "n_clicks")],
        [State("dataset-dropdown", "value"),
         State("seed-dropdown", "value"),
         State("fragmentation-filter-radio", "value"),
         State("path-sort-dropdown", "value")]
    )
    def update_path_table(n_clicks, dataset, seed, fragmentation_filter, sort_by):
        """Update the path table with fragmentation scores."""
        if n_clicks == 0 or not dataset:
            return "Click 'Update Paths' to load path data."
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return "No similarity data available. Run cluster_paths.py with --compute_similarity option."
            
            # Get friendly layer name function
            get_friendly_layer_name = dash_app.get_friendly_layer_name
            
            # Create path table
            table = create_path_table(
                cluster_paths_data,
                fragmentation_filter,
                sort_by,
                get_friendly_layer_name
            )
            
            return table
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating path table: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"Error: {str(e)}"
    
    @app.callback(
        [Output("path-details-container", "children"),
         Output("path-visualization-graph", "figure"),
         Output("path-visualization-graph", "style")],
        [Input("path-fragmentation-table", "selected_rows")],
        [State("path-fragmentation-table", "data"),
         State("dataset-dropdown", "value"),
         State("seed-dropdown", "value")]
    )
    def update_path_details(selected_rows, rows_data, dataset, seed):
        """Update the path details when a row is selected."""
        if not selected_rows or not rows_data:
            return "Select a path to view details.", go.Figure(), {"display": "none"}
        
        try:
            # Import the dashboard module dynamically to avoid circular imports
            dash_app = importlib.import_module("visualization.dash_app")
            
            # Get the selected row
            selected_row = rows_data[selected_rows[0]]
            
            # Try to load cluster paths with similarity data
            cluster_paths_data = dash_app.load_cluster_paths_data(dataset, seed)
            
            if cluster_paths_data is None or "similarity" not in cluster_paths_data:
                return "No similarity data available.", go.Figure(), {"display": "none"}
            
            # Get friendly layer name function
            get_friendly_layer_name = dash_app.get_friendly_layer_name
            
            # Create path details view
            details_view = create_path_detail_view(
                selected_row,
                cluster_paths_data,
                get_friendly_layer_name
            )
            
            # Create path visualization
            fig = create_path_visualization(
                selected_row,
                cluster_paths_data,
                get_friendly_layer_name
            )
            
            return details_view, fig, {"height": "400px", "display": "block"}
            
        except Exception as e:
            import traceback
            error_msg = f"Error creating path details: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return f"Error: {str(e)}", go.Figure(), {"display": "none"}