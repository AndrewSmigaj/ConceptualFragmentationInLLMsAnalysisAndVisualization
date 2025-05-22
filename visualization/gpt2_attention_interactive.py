"""
Interactive filtering components for GPT-2 attention visualizations.

This module extends existing GPT-2 attention visualizations with interactive filtering
capabilities for attention heads, layers, and patterns, integrated with Dash components.
"""

import dash
from dash import dcc, html, callback, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import itertools

# Import existing attention visualization components
from visualization.gpt2_attention_sankey import (
    extract_attention_flow,
    generate_attention_sankey_diagram,
    create_attention_token_comparison
)
from visualization.gpt2_attention_correlation import (
    calculate_correlation_metrics,
    create_correlation_heatmap,
    create_token_correlation_scatter
)

# Constants for filtering
MIN_ATTENTION_THRESHOLD = 0.001
MAX_ATTENTION_THRESHOLD = 1.0
DEFAULT_ATTENTION_THRESHOLD = 0.05
MAX_DISPLAYED_EDGES = 500
DEFAULT_MAX_EDGES = 100


def create_attention_filter_controls() -> html.Div:
    """
    Create interactive filter controls for attention visualizations.
    
    Returns:
        Dash HTML component with filter controls
    """
    controls = html.Div([
        # Filter Section Header
        html.H4("Attention Filters", className="mb-3"),
        
        dbc.Row([
            # Attention Head Filtering
            dbc.Col([
                html.Label("Attention Heads:", className="form-label"),
                dcc.Dropdown(
                    id="attention-head-filter",
                    placeholder="Select attention heads to display",
                    multi=True,
                    value=None  # Will be populated dynamically
                ),
                html.Small("Select specific attention heads to analyze", className="form-text text-muted")
            ], width=6),
            
            # Layer Range Filtering
            dbc.Col([
                html.Label("Layer Range:", className="form-label"),
                dcc.RangeSlider(
                    id="layer-range-filter",
                    min=0,
                    max=12,  # Will be updated dynamically
                    step=1,
                    value=[0, 12],
                    marks={i: f"L{i}" for i in range(0, 13, 2)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Select range of layers to analyze", className="form-text text-muted")
            ], width=6)
        ], className="mb-3"),
        
        dbc.Row([
            # Attention Threshold Filtering
            dbc.Col([
                html.Label("Attention Threshold:", className="form-label"),
                dcc.Slider(
                    id="attention-threshold-filter",
                    min=MIN_ATTENTION_THRESHOLD,
                    max=MAX_ATTENTION_THRESHOLD,
                    step=0.001,
                    value=DEFAULT_ATTENTION_THRESHOLD,
                    marks={
                        0.001: "0.001",
                        0.01: "0.01", 
                        0.05: "0.05",
                        0.1: "0.1",
                        0.5: "0.5",
                        1.0: "1.0"
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Minimum attention weight to display", className="form-text text-muted")
            ], width=6),
            
            # Maximum Edges Filtering
            dbc.Col([
                html.Label("Max Edges:", className="form-label"),
                dcc.Slider(
                    id="max-edges-filter",
                    min=10,
                    max=MAX_DISPLAYED_EDGES,
                    step=10,
                    value=DEFAULT_MAX_EDGES,
                    marks={
                        10: "10",
                        50: "50",
                        100: "100",
                        250: "250",
                        500: "500"
                    },
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Maximum number of attention edges to display", className="form-text text-muted")
            ], width=6)
        ], className="mb-3"),
        
        dbc.Row([
            # Pattern Type Filtering
            dbc.Col([
                html.Label("Attention Patterns:", className="form-label"),
                dcc.Checklist(
                    id="attention-pattern-filter",
                    options=[
                        {"label": "Local (adjacent tokens)", "value": "local"},
                        {"label": "Global (distant tokens)", "value": "global"},
                        {"label": "Self-attention", "value": "self"},
                        {"label": "Forward flow", "value": "forward"},
                        {"label": "Backward flow", "value": "backward"}
                    ],
                    value=["local", "global", "self", "forward", "backward"],
                    inline=False,
                    className="form-check"
                ),
                html.Small("Select attention pattern types to include", className="form-text text-muted")
            ], width=6),
            
            # Token Highlighting
            dbc.Col([
                html.Label("Highlight Tokens:", className="form-label"),
                dcc.Input(
                    id="highlight-tokens-input",
                    type="text",
                    placeholder="Enter tokens separated by commas",
                    className="form-control mb-2"
                ),
                dbc.Button(
                    "Apply Token Highlighting",
                    id="apply-token-highlighting",
                    color="primary",
                    size="sm",
                    className="mb-2"
                ),
                html.Small("Highlight specific tokens in visualizations", className="form-text text-muted")
            ], width=6)
        ], className="mb-3"),
        
        # Advanced Filters (Collapsible)
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        html.Label("Head Specialization Filter:", className="form-label"),
                        dcc.Dropdown(
                            id="head-specialization-filter",
                            options=[
                                {"label": "Concept Formation Heads", "value": "concept_formation"},
                                {"label": "Syntactic Heads", "value": "syntactic"},
                                {"label": "Integration Heads", "value": "integration"},
                                {"label": "Position Heads", "value": "position"},
                                {"label": "Mixed Heads", "value": "mixed"}
                            ],
                            placeholder="Filter by head specialization",
                            multi=True
                        )
                    ], width=6),
                    
                    dbc.Col([
                        html.Label("Entropy Range:", className="form-label"),
                        dcc.RangeSlider(
                            id="entropy-range-filter",
                            min=0,
                            max=1,
                            step=0.05,
                            value=[0, 1],
                            marks={0: "0", 0.5: "0.5", 1: "1"},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Label("Correlation Threshold:", className="form-label"),
                        dcc.Slider(
                            id="correlation-threshold-filter",
                            min=-1,
                            max=1,
                            step=0.1,
                            value=0.3,
                            marks={-1: "-1", 0: "0", 0.5: "0.5", 1: "1"},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], width=6),
                    
                    dbc.Col([
                        dbc.Checklist(
                            id="advanced-pattern-filter",
                            options=[
                                {"label": "Show only high-entropy attention", "value": "high_entropy"},
                                {"label": "Show only low-entropy attention", "value": "low_entropy"},
                                {"label": "Show cross-cluster attention only", "value": "cross_cluster"},
                                {"label": "Show intra-cluster attention only", "value": "intra_cluster"}
                            ],
                            value=[],
                            className="form-check"
                        )
                    ], width=6)
                ])
            ], title="Advanced Filters")
        ], start_closed=True)
    ], className="mb-4")
    
    return controls


def filter_attention_data(
    attention_data: Dict[str, np.ndarray],
    attention_heads: Optional[List[int]] = None,
    layer_range: Optional[Tuple[int, int]] = None,
    attention_threshold: float = DEFAULT_ATTENTION_THRESHOLD,
    pattern_types: Optional[List[str]] = None,
    entropy_range: Optional[Tuple[float, float]] = None,
    correlation_threshold: Optional[float] = None,
    advanced_filters: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Apply filters to attention data.
    
    Args:
        attention_data: Dictionary mapping layer names to attention arrays
        attention_heads: List of head indices to include (None for all)
        layer_range: Tuple of (min_layer, max_layer) indices
        attention_threshold: Minimum attention value to include
        pattern_types: List of pattern types to include
        entropy_range: Tuple of (min_entropy, max_entropy) values
        correlation_threshold: Minimum correlation to include
        advanced_filters: List of advanced filter options
        
    Returns:
        Dictionary with filtered attention data and metadata
    """
    filtered_data = {}
    filter_stats = {
        "original_layers": len(attention_data),
        "filtered_layers": 0,
        "total_attention_values": 0,
        "filtered_attention_values": 0,
        "filters_applied": []
    }
    
    # Get layer names and sort them
    layer_names = sorted(attention_data.keys())
    
    # Apply layer range filter
    if layer_range is not None:
        min_layer, max_layer = layer_range
        
        # Extract layer numbers for filtering
        layer_indices = []
        for layer_name in layer_names:
            # Extract number from layer name (e.g., "layer_0" -> 0)
            try:
                if "layer" in layer_name:
                    layer_num = int(layer_name.split("_")[-1])
                    if min_layer <= layer_num <= max_layer:
                        layer_indices.append(layer_name)
            except (ValueError, IndexError):
                # If we can't parse layer number, include it
                layer_indices.append(layer_name)
        
        layer_names = layer_indices
        filter_stats["filters_applied"].append(f"layer_range_{min_layer}_{max_layer}")
    
    # Process each layer
    for layer_name in layer_names:
        if layer_name not in attention_data:
            continue
        
        attention = attention_data[layer_name]
        original_shape = attention.shape
        
        # Handle different attention tensor shapes
        if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
            batch_size, n_heads, seq_len, _ = attention.shape
        elif len(attention.shape) == 3:  # [batch, seq_len, seq_len] or [heads, seq_len, seq_len]
            if attention.shape[0] <= 16:  # Assume this is heads dimension
                n_heads, seq_len = attention.shape[0], attention.shape[1]
                attention = attention.reshape(1, n_heads, seq_len, seq_len)
                batch_size = 1
            else:  # Assume this is batch dimension
                batch_size, seq_len = attention.shape[0], attention.shape[1]
                attention = attention.reshape(batch_size, 1, seq_len, seq_len)
                n_heads = 1
        else:
            # Skip unsupported shapes
            continue
        
        filter_stats["total_attention_values"] += attention.size
        
        # Apply attention head filter
        if attention_heads is not None:
            # Filter to only selected heads
            selected_heads = [h for h in attention_heads if h < n_heads]
            if selected_heads:
                attention = attention[:, selected_heads]
                filter_stats["filters_applied"].append(f"heads_{len(selected_heads)}")
        
        # Apply attention threshold filter
        if attention_threshold > MIN_ATTENTION_THRESHOLD:
            attention = np.where(attention >= attention_threshold, attention, 0)
            filter_stats["filters_applied"].append(f"threshold_{attention_threshold}")
        
        # Apply pattern type filters
        if pattern_types is not None and len(pattern_types) < 5:  # Not all patterns selected
            filtered_attention = np.zeros_like(attention)
            
            for batch_idx in range(attention.shape[0]):
                for head_idx in range(attention.shape[1]):
                    head_attention = attention[batch_idx, head_idx]
                    
                    # Create pattern masks
                    pattern_mask = np.zeros_like(head_attention, dtype=bool)
                    
                    if "local" in pattern_types:
                        # Local attention (within window of ±2)
                        for i in range(seq_len):
                            for j in range(max(0, i-2), min(seq_len, i+3)):
                                pattern_mask[i, j] = True
                    
                    if "global" in pattern_types:
                        # Global attention (beyond window of ±2)
                        for i in range(seq_len):
                            for j in range(seq_len):
                                if abs(i - j) > 2:
                                    pattern_mask[i, j] = True
                    
                    if "self" in pattern_types:
                        # Self-attention (diagonal)
                        np.fill_diagonal(pattern_mask, True)
                    
                    if "forward" in pattern_types:
                        # Forward flow (upper triangle)
                        upper_triangle = np.triu(np.ones_like(head_attention, dtype=bool), k=1)
                        pattern_mask = pattern_mask | upper_triangle
                    
                    if "backward" in pattern_types:
                        # Backward flow (lower triangle)
                        lower_triangle = np.tril(np.ones_like(head_attention, dtype=bool), k=-1)
                        pattern_mask = pattern_mask | lower_triangle
                    
                    # Apply pattern mask
                    filtered_attention[batch_idx, head_idx] = head_attention * pattern_mask
            
            attention = filtered_attention
            filter_stats["filters_applied"].append(f"patterns_{len(pattern_types)}")
        
        # Apply entropy filter
        if entropy_range is not None:
            min_entropy, max_entropy = entropy_range
            
            for batch_idx in range(attention.shape[0]):
                for head_idx in range(attention.shape[1]):
                    head_attention = attention[batch_idx, head_idx]
                    
                    # Calculate entropy for each position
                    entropies = []
                    for pos in range(seq_len):
                        attn_dist = head_attention[pos]
                        attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
                        entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-10))
                        normalized_entropy = entropy / np.log2(seq_len) if seq_len > 1 else 0
                        entropies.append(normalized_entropy)
                    
                    # Filter positions by entropy
                    entropy_mask = np.array([(min_entropy <= e <= max_entropy) for e in entropies])
                    
                    # Apply entropy filter
                    attention[batch_idx, head_idx] = attention[batch_idx, head_idx] * entropy_mask[:, np.newaxis]
            
            filter_stats["filters_applied"].append(f"entropy_{min_entropy}_{max_entropy}")
        
        # Apply advanced filters
        if advanced_filters:
            for filter_type in advanced_filters:
                if filter_type == "high_entropy":
                    # Keep only high-entropy attention (> 0.7)
                    entropy_threshold = 0.7
                    # Implementation similar to entropy_range filter
                    
                elif filter_type == "low_entropy":
                    # Keep only low-entropy attention (< 0.3)
                    entropy_threshold = 0.3
                    # Implementation similar to entropy_range filter
                    
                elif filter_type == "cross_cluster":
                    # Keep only attention between different clusters
                    # This would require cluster labels - placeholder for now
                    pass
                    
                elif filter_type == "intra_cluster":
                    # Keep only attention within same clusters
                    # This would require cluster labels - placeholder for now
                    pass
        
        # Store filtered attention
        filtered_data[layer_name] = attention
        filter_stats["filtered_attention_values"] += attention.size
        filter_stats["filtered_layers"] += 1
    
    # Calculate filter efficiency
    if filter_stats["total_attention_values"] > 0:
        filter_stats["retention_rate"] = filter_stats["filtered_attention_values"] / filter_stats["total_attention_values"]
    else:
        filter_stats["retention_rate"] = 0.0
    
    return {
        "filtered_attention_data": filtered_data,
        "filter_stats": filter_stats,
        "layer_names": list(filtered_data.keys())
    }


def create_filtered_attention_sankey(
    attention_data: Dict[str, np.ndarray],
    token_metadata: Dict[str, Any],
    filter_params: Dict[str, Any],
    highlight_tokens: Optional[List[str]] = None,
    max_edges: int = DEFAULT_MAX_EDGES
) -> go.Figure:
    """
    Create filtered attention Sankey diagram.
    
    Args:
        attention_data: Raw attention data
        token_metadata: Token metadata
        filter_params: Filter parameters
        highlight_tokens: Tokens to highlight
        max_edges: Maximum edges to display
        
    Returns:
        Plotly Figure with filtered Sankey diagram
    """
    # Apply filters
    filtered_result = filter_attention_data(attention_data, **filter_params)
    filtered_attention = filtered_result["filtered_attention_data"]
    
    if not filtered_attention:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No attention data remains after filtering",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Extract attention flow
    attention_flow = extract_attention_flow(
        attention_data=filtered_attention,
        token_metadata=token_metadata,
        min_attention=filter_params.get("attention_threshold", DEFAULT_ATTENTION_THRESHOLD)
    )
    
    # Generate Sankey diagram
    fig = generate_attention_sankey_diagram(
        attention_flow=attention_flow,
        layer_names=filtered_result["layer_names"],
        highlight_tokens=highlight_tokens,
        max_edges=max_edges,
        title=f"Filtered Attention Flow ({len(filtered_result['filter_stats']['filters_applied'])} filters applied)"
    )
    
    # Add filter information to title
    filter_info = filtered_result["filter_stats"]
    retention_rate = filter_info["retention_rate"] * 100
    
    fig.update_layout(
        title=dict(
            text=(
                f"Filtered Attention Flow<br>"
                f"<sub>Filters: {', '.join(filter_info['filters_applied'])} | "
                f"Retention: {retention_rate:.1f}% | "
                f"Layers: {filter_info['filtered_layers']}/{filter_info['original_layers']}</sub>"
            )
        )
    )
    
    return fig


def create_attention_head_analysis(
    attention_data: Dict[str, np.ndarray],
    layer_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze attention head specialization for filtering.
    
    Args:
        attention_data: Dictionary mapping layer names to attention arrays
        layer_names: Optional list of layer names to analyze
        
    Returns:
        Dictionary with head analysis results
    """
    if layer_names is None:
        layer_names = sorted(attention_data.keys())
    
    head_analysis = {
        "specialization_by_layer": {},
        "head_options": [],
        "layer_head_mapping": {}
    }
    
    for layer_name in layer_names:
        if layer_name not in attention_data:
            continue
        
        attention = attention_data[layer_name]
        
        # Handle different tensor shapes
        if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
            batch_size, n_heads, seq_len, _ = attention.shape
        elif len(attention.shape) == 3 and attention.shape[0] <= 16:  # [heads, seq_len, seq_len]
            n_heads, seq_len = attention.shape[0], attention.shape[1]
            attention = attention.reshape(1, n_heads, seq_len, seq_len)
            batch_size = 1
        else:
            continue
        
        layer_specialization = {}
        
        for head_idx in range(n_heads):
            # Extract head attention (average across batch)
            head_attention = attention[:, head_idx].mean(axis=0)
            
            # Calculate head characteristics
            # 1. Diagonal vs off-diagonal attention
            diagonal_strength = np.mean(np.diag(head_attention))
            off_diagonal_strength = np.mean(head_attention - np.diag(np.diag(head_attention)))
            
            # 2. Local vs global attention
            local_attention = 0
            global_attention = 0
            local_count = 0
            global_count = 0
            
            for i in range(seq_len):
                for j in range(seq_len):
                    if abs(i - j) <= 2:  # Local window
                        local_attention += head_attention[i, j]
                        local_count += 1
                    else:  # Global
                        global_attention += head_attention[i, j]
                        global_count += 1
            
            local_attention = local_attention / local_count if local_count > 0 else 0
            global_attention = global_attention / global_count if global_count > 0 else 0
            
            # 3. Attention entropy
            entropy_values = []
            for pos in range(seq_len):
                attn_dist = head_attention[pos]
                attn_dist = attn_dist / (attn_dist.sum() + 1e-10)
                entropy = -np.sum(attn_dist * np.log2(attn_dist + 1e-10))
                normalized_entropy = entropy / np.log2(seq_len) if seq_len > 1 else 0
                entropy_values.append(normalized_entropy)
            
            mean_entropy = np.mean(entropy_values)
            
            # Classify head specialization
            if diagonal_strength > 0.5:
                specialization = "position"
            elif local_attention > global_attention * 2:
                specialization = "syntactic"
            elif mean_entropy < 0.3:
                specialization = "concept_formation"
            elif mean_entropy > 0.7:
                specialization = "integration"
            else:
                specialization = "mixed"
            
            # Store head analysis
            head_id = f"{layer_name}_head_{head_idx}"
            layer_specialization[head_idx] = {
                "head_id": head_id,
                "specialization": specialization,
                "diagonal_strength": float(diagonal_strength),
                "off_diagonal_strength": float(off_diagonal_strength),
                "local_attention": float(local_attention),
                "global_attention": float(global_attention),
                "mean_entropy": float(mean_entropy)
            }
            
            # Add to global head options
            head_analysis["head_options"].append({
                "label": f"{layer_name} Head {head_idx} ({specialization})",
                "value": head_id
            })
        
        head_analysis["specialization_by_layer"][layer_name] = layer_specialization
        head_analysis["layer_head_mapping"][layer_name] = list(range(n_heads))
    
    return head_analysis


def create_filter_summary_stats(filter_stats: Dict[str, Any]) -> html.Div:
    """
    Create summary statistics display for applied filters.
    
    Args:
        filter_stats: Filter statistics from filter_attention_data
        
    Returns:
        Dash HTML component with filter summary
    """
    return html.Div([
        html.H5("Filter Summary", className="mb-2"),
        dbc.Row([
            dbc.Col([
                html.P([
                    html.Strong("Filters Applied: "),
                    html.Span(f"{len(filter_stats.get('filters_applied', []))}")
                ], className="mb-1"),
                html.P([
                    html.Strong("Retention Rate: "),
                    html.Span(f"{filter_stats.get('retention_rate', 0) * 100:.1f}%")
                ], className="mb-1")
            ], width=6),
            dbc.Col([
                html.P([
                    html.Strong("Layers: "),
                    html.Span(f"{filter_stats.get('filtered_layers', 0)}/{filter_stats.get('original_layers', 0)}")
                ], className="mb-1"),
                html.P([
                    html.Strong("Active Filters: "),
                    html.Span(", ".join(filter_stats.get('filters_applied', [])))
                ], className="mb-1", style={"font-size": "0.9em"})
            ], width=6)
        ])
    ], className="p-3 bg-light rounded mb-3")


# Dash callbacks for interactive filtering
@callback(
    [Output("attention-head-filter", "options"),
     Output("layer-range-filter", "max"),
     Output("layer-range-filter", "marks"),
     Output("layer-range-filter", "value")],
    [Input("gpt2-analysis-selector", "value")]
)
def update_filter_options(selected_analysis):
    """Update filter options based on selected analysis."""
    if not selected_analysis:
        return [], 12, {}, [0, 12]
    
    try:
        # Load analysis data to get available heads and layers
        # This would be implemented based on the actual data structure
        # For now, return default options
        
        # Mock head options (would be populated from actual data)
        head_options = [
            {"label": f"Layer {i} Head {j}", "value": f"layer_{i}_head_{j}"}
            for i in range(12) for j in range(12)
        ]
        
        # Mock layer range (would be populated from actual data)
        max_layers = 11
        layer_marks = {i: f"L{i}" for i in range(0, max_layers + 1, 2)}
        layer_value = [0, max_layers]
        
        return head_options, max_layers, layer_marks, layer_value
        
    except Exception as e:
        print(f"Error updating filter options: {e}")
        return [], 12, {}, [0, 12]


@callback(
    [Output("filtered-attention-visualization", "figure"),
     Output("filter-summary-stats", "children")],
    [Input("attention-head-filter", "value"),
     Input("layer-range-filter", "value"),
     Input("attention-threshold-filter", "value"),
     Input("max-edges-filter", "value"),
     Input("attention-pattern-filter", "value"),
     Input("apply-token-highlighting", "n_clicks")],
    [State("highlight-tokens-input", "value"),
     State("gpt2-analysis-selector", "value")]
)
def update_filtered_visualization(head_filter, layer_range, threshold, max_edges,
                                pattern_filter, highlight_clicks, highlight_tokens,
                                selected_analysis):
    """Update visualization based on filter settings."""
    if not selected_analysis:
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text="Please select a GPT-2 analysis to visualize",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        return empty_fig, html.Div()
    
    try:
        # This would load actual data based on selected_analysis
        # For now, return a placeholder
        
        # Mock filter parameters
        filter_params = {
            "attention_heads": head_filter,
            "layer_range": tuple(layer_range) if layer_range else None,
            "attention_threshold": threshold,
            "pattern_types": pattern_filter
        }
        
        # Mock filter stats
        filter_stats = {
            "filters_applied": [f for f in ["heads", "layers", "threshold", "patterns"] if f],
            "retention_rate": 0.75,
            "filtered_layers": 8,
            "original_layers": 12
        }
        
        # Create placeholder figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Filtered Attention Visualization\n{len(filter_stats['filters_applied'])} filters applied",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        
        # Create filter summary
        summary_stats = create_filter_summary_stats(filter_stats)
        
        return fig, summary_stats
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error creating filtered visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return error_fig, html.Div()


# Main layout component for integration with existing dashboard
def create_interactive_attention_tab() -> html.Div:
    """
    Create the main interactive attention visualization tab.
    
    Returns:
        Dash HTML component with complete interactive attention interface
    """
    return html.Div([
        # Header
        html.H2("Interactive GPT-2 Attention Analysis", className="mb-4"),
        
        # Filter Controls
        create_attention_filter_controls(),
        
        # Filter Summary Stats
        html.Div(id="filter-summary-stats"),
        
        # Main Visualization
        dcc.Graph(
            id="filtered-attention-visualization",
            style={"height": "600px"},
            config={
                "displayModeBar": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "filtered_attention_visualization",
                    "height": 600,
                    "width": 1000,
                    "scale": 2
                }
            }
        ),
        
        # Additional Analysis Tabs
        dbc.Tabs([
            dbc.Tab(
                label="Head Specialization",
                tab_id="head-specialization-tab",
                children=[
                    html.Div(id="head-specialization-content", className="p-3")
                ]
            ),
            dbc.Tab(
                label="Pattern Analysis",
                tab_id="pattern-analysis-tab", 
                children=[
                    html.Div(id="pattern-analysis-content", className="p-3")
                ]
            ),
            dbc.Tab(
                label="Correlation Analysis",
                tab_id="correlation-analysis-tab",
                children=[
                    html.Div(id="correlation-analysis-content", className="p-3")
                ]
            )
        ], id="analysis-tabs", active_tab="head-specialization-tab", className="mt-4")
    ], className="container-fluid")


if __name__ == "__main__":
    # Example usage - would be integrated with main Dash app
    print("Interactive GPT-2 Attention Filtering Components Created")
    print("Integration points:")
    print("- create_attention_filter_controls(): Filter control components")
    print("- filter_attention_data(): Core filtering functionality")
    print("- create_filtered_attention_sankey(): Filtered visualization generation")
    print("- create_interactive_attention_tab(): Complete tab layout")