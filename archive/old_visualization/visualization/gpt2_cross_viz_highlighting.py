"""
Cross-visualization highlighting system for GPT-2 APA visualizations.

This module provides synchronized highlighting capabilities across different visualization types,
enabling coordinated exploration of token paths, attention patterns, and cluster relationships.
"""

import dash
from dash import dcc, html, callback, clientside_callback, ClientsideFunction
from dash.dependencies import Input, Output, State, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from collections import defaultdict
from dataclasses import dataclass
import uuid

# Import existing visualization components
from visualization.gpt2_token_sankey import generate_token_sankey_diagram
from visualization.gpt2_attention_sankey import generate_attention_sankey_diagram
from visualization.gpt2_attention_correlation import create_correlation_heatmap


@dataclass
class SelectionState:
    """
    Represents the current selection state across visualizations.
    """
    selected_tokens: Set[str] = None
    selected_attention_edges: Set[Tuple[str, str]] = None
    selected_clusters: Set[int] = None
    selected_layers: Set[str] = None
    selected_positions: Set[int] = None
    highlight_color: str = "#ff6b6b"
    
    def __post_init__(self):
        if self.selected_tokens is None:
            self.selected_tokens = set()
        if self.selected_attention_edges is None:
            self.selected_attention_edges = set()
        if self.selected_clusters is None:
            self.selected_clusters = set()
        if self.selected_layers is None:
            self.selected_layers = set()
        if self.selected_positions is None:
            self.selected_positions = set()


class CrossVisualizationHighlighter:
    """
    Manages synchronized highlighting across multiple GPT-2 visualizations.
    """
    
    def __init__(self):
        self.selection_state = SelectionState()
        self.visualization_registry = {}
        self.callback_registry = {}
    
    def register_visualization(self, viz_id: str, viz_type: str, data_mapping: Dict[str, Any]):
        """
        Register a visualization for cross-highlighting.
        
        Args:
            viz_id: Unique identifier for the visualization
            viz_type: Type of visualization ('token_sankey', 'attention_sankey', 'correlation_heatmap', etc.)
            data_mapping: Mapping between visualization elements and data identifiers
        """
        self.visualization_registry[viz_id] = {
            "type": viz_type,
            "data_mapping": data_mapping,
            "selection_handlers": self._get_selection_handlers(viz_type)
        }
    
    def _get_selection_handlers(self, viz_type: str) -> Dict[str, callable]:
        """Get appropriate selection handlers for visualization type."""
        handlers = {
            "token_sankey": {
                "on_node_click": self._handle_token_node_selection,
                "on_link_click": self._handle_token_link_selection,
                "update_highlights": self._update_token_sankey_highlights
            },
            "attention_sankey": {
                "on_node_click": self._handle_attention_node_selection,
                "on_link_click": self._handle_attention_link_selection,
                "update_highlights": self._update_attention_sankey_highlights
            },
            "correlation_heatmap": {
                "on_cell_click": self._handle_correlation_cell_selection,
                "update_highlights": self._update_correlation_highlights
            },
            "trajectory_plot": {
                "on_point_click": self._handle_trajectory_point_selection,
                "update_highlights": self._update_trajectory_highlights
            }
        }
        return handlers.get(viz_type, {})
    
    def _handle_token_node_selection(self, node_data: Dict[str, Any]):
        """Handle token node selection in Sankey diagram."""
        token_id = node_data.get("token_id")
        layer = node_data.get("layer")
        position = node_data.get("position")
        
        if token_id:
            if token_id in self.selection_state.selected_tokens:
                self.selection_state.selected_tokens.remove(token_id)
            else:
                self.selection_state.selected_tokens.add(token_id)
        
        if position is not None:
            if position in self.selection_state.selected_positions:
                self.selection_state.selected_positions.remove(position)
            else:
                self.selection_state.selected_positions.add(position)
        
        return self._propagate_selection_update("token_node", node_data)
    
    def _handle_attention_link_selection(self, link_data: Dict[str, Any]):
        """Handle attention link selection in attention Sankey."""
        source_token = link_data.get("source_token")
        target_token = link_data.get("target_token")
        
        if source_token and target_token:
            edge = (source_token, target_token)
            if edge in self.selection_state.selected_attention_edges:
                self.selection_state.selected_attention_edges.remove(edge)
            else:
                self.selection_state.selected_attention_edges.add(edge)
        
        return self._propagate_selection_update("attention_link", link_data)
    
    def _propagate_selection_update(self, selection_type: str, selection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Propagate selection update to all registered visualizations."""
        update_commands = {}
        
        for viz_id, viz_info in self.visualization_registry.items():
            viz_type = viz_info["type"]
            update_handler = viz_info["selection_handlers"].get("update_highlights")
            
            if update_handler:
                update_commands[viz_id] = update_handler(selection_type, selection_data)
        
        return update_commands
    
    def _update_token_sankey_highlights(self, selection_type: str, selection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update token Sankey diagram highlights based on selection."""
        highlight_updates = {
            "node_colors": {},
            "link_colors": {},
            "node_sizes": {}
        }
        
        # Highlight selected tokens
        for token_id in self.selection_state.selected_tokens:
            highlight_updates["node_colors"][token_id] = self.selection_state.highlight_color
            highlight_updates["node_sizes"][token_id] = 1.5  # Scale factor
        
        # Highlight related attention edges
        for source, target in self.selection_state.selected_attention_edges:
            # Find corresponding token path links
            link_key = f"{source}_{target}"
            highlight_updates["link_colors"][link_key] = self.selection_state.highlight_color
        
        return highlight_updates
    
    def _update_attention_sankey_highlights(self, selection_type: str, selection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update attention Sankey diagram highlights based on selection."""
        highlight_updates = {
            "node_colors": {},
            "link_colors": {},
            "link_widths": {}
        }
        
        # Highlight attention edges for selected tokens
        for token_id in self.selection_state.selected_tokens:
            # Find attention edges involving this token
            for source, target in self.selection_state.selected_attention_edges:
                if token_id in [source, target]:
                    edge_key = f"{source}_{target}"
                    highlight_updates["link_colors"][edge_key] = self.selection_state.highlight_color
                    highlight_updates["link_widths"][edge_key] = 2.0  # Scale factor
        
        # Highlight selected attention edges directly
        for source, target in self.selection_state.selected_attention_edges:
            edge_key = f"{source}_{target}"
            highlight_updates["link_colors"][edge_key] = self.selection_state.highlight_color
            highlight_updates["link_widths"][edge_key] = 2.0
        
        return highlight_updates


def create_synchronized_visualization_layout(
    token_sankey_data: Dict[str, Any],
    attention_sankey_data: Dict[str, Any],
    correlation_data: Dict[str, Any],
    layout_style: str = "grid"
) -> html.Div:
    """
    Create a synchronized visualization layout with cross-highlighting.
    
    Args:
        token_sankey_data: Data for token path Sankey diagram
        attention_sankey_data: Data for attention Sankey diagram
        correlation_data: Data for correlation heatmap
        layout_style: Layout style ('grid', 'tabs', 'accordion')
        
    Returns:
        Dash HTML component with synchronized visualizations
    """
    # Generate unique IDs for this layout instance
    layout_id = str(uuid.uuid4())[:8]
    
    # Create individual visualization components
    token_viz_id = f"token-sankey-{layout_id}"
    attention_viz_id = f"attention-sankey-{layout_id}"
    correlation_viz_id = f"correlation-heatmap-{layout_id}"
    
    # Selection state store
    selection_store = dcc.Store(
        id=f"selection-state-{layout_id}",
        data={
            "selected_tokens": [],
            "selected_attention_edges": [],
            "selected_clusters": [],
            "selected_layers": [],
            "selected_positions": []
        }
    )
    
    # Create visualizations based on layout style
    if layout_style == "grid":
        return create_grid_layout(
            token_viz_id, attention_viz_id, correlation_viz_id, 
            selection_store, layout_id
        )
    elif layout_style == "tabs":
        return create_tabbed_layout(
            token_viz_id, attention_viz_id, correlation_viz_id,
            selection_store, layout_id
        )
    else:  # accordion
        return create_accordion_layout(
            token_viz_id, attention_viz_id, correlation_viz_id,
            selection_store, layout_id
        )


def create_grid_layout(token_viz_id: str, attention_viz_id: str, correlation_viz_id: str,
                      selection_store: dcc.Store, layout_id: str) -> html.Div:
    """Create grid-based synchronized visualization layout."""
    
    return html.Div([
        selection_store,
        
        # Selection controls
        html.Div([
            html.H5("Selection Controls", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "Clear All Selections",
                        id=f"clear-selections-{layout_id}",
                        color="secondary",
                        size="sm"
                    )
                ], width=3),
                dbc.Col([
                    dbc.Button(
                        "Highlight Connected",
                        id=f"highlight-connected-{layout_id}",
                        color="primary",
                        size="sm"
                    )
                ], width=3),
                dbc.Col([
                    html.Div(id=f"selection-summary-{layout_id}", className="small text-muted")
                ], width=6)
            ])
        ], className="mb-3 p-3 bg-light rounded"),
        
        # Main visualization grid
        dbc.Row([
            # Token Sankey (left column)
            dbc.Col([
                html.H6("Token Paths", className="mb-2"),
                dcc.Graph(
                    id=token_viz_id,
                    style={"height": "500px"},
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawclosedpath"]
                    }
                )
            ], width=6),
            
            # Attention Sankey (right column)
            dbc.Col([
                html.H6("Attention Flow", className="mb-2"),
                dcc.Graph(
                    id=attention_viz_id,
                    style={"height": "500px"},
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"]
                    }
                )
            ], width=6)
        ], className="mb-4"),
        
        # Correlation heatmap (bottom row)
        dbc.Row([
            dbc.Col([
                html.H6("Path-Attention Correlation", className="mb-2"),
                dcc.Graph(
                    id=correlation_viz_id,
                    style={"height": "300px"}
                )
            ], width=12)
        ])
    ])


def create_tabbed_layout(token_viz_id: str, attention_viz_id: str, correlation_viz_id: str,
                        selection_store: dcc.Store, layout_id: str) -> html.Div:
    """Create tabbed synchronized visualization layout."""
    
    return html.Div([
        selection_store,
        
        # Selection status bar
        dbc.Alert(
            id=f"selection-status-{layout_id}",
            children="No selections active",
            color="info",
            className="mb-3"
        ),
        
        # Visualization tabs
        dbc.Tabs([
            dbc.Tab(
                label="Token Paths",
                tab_id="token-paths-tab",
                children=[
                    html.Div([
                        dcc.Graph(
                            id=token_viz_id,
                            style={"height": "600px"}
                        )
                    ], className="p-3")
                ]
            ),
            dbc.Tab(
                label="Attention Flow",
                tab_id="attention-flow-tab",
                children=[
                    html.Div([
                        dcc.Graph(
                            id=attention_viz_id,
                            style={"height": "600px"}
                        )
                    ], className="p-3")
                ]
            ),
            dbc.Tab(
                label="Correlation Analysis",
                tab_id="correlation-tab",
                children=[
                    html.Div([
                        dcc.Graph(
                            id=correlation_viz_id,
                            style={"height": "500px"}
                        )
                    ], className="p-3")
                ]
            )
        ], active_tab="token-paths-tab")
    ])


def create_selection_callbacks(layout_id: str):
    """
    Create Dash callbacks for synchronized selection handling.
    
    Args:
        layout_id: Unique identifier for this layout instance
    """
    
    # Token Sankey selection callback
    @callback(
        [Output(f"selection-state-{layout_id}", "data"),
         Output(f"attention-sankey-{layout_id}", "figure"),
         Output(f"correlation-heatmap-{layout_id}", "figure")],
        [Input(f"token-sankey-{layout_id}", "clickData")],
        [State(f"selection-state-{layout_id}", "data"),
         State(f"attention-sankey-{layout_id}", "figure"),
         State(f"correlation-heatmap-{layout_id}", "figure")]
    )
    def handle_token_sankey_selection(click_data, selection_state, attention_fig, correlation_fig):
        """Handle token Sankey node/link selection and update other visualizations."""
        
        if not click_data:
            return selection_state, attention_fig, correlation_fig
        
        # Parse click data
        point_data = click_data["points"][0]
        
        # Update selection state
        updated_state = dict(selection_state)
        
        if "customdata" in point_data:
            # Handle node selection
            node_info = point_data["customdata"]
            token_id = node_info.get("token_id")
            
            if token_id:
                if token_id in updated_state["selected_tokens"]:
                    updated_state["selected_tokens"].remove(token_id)
                else:
                    updated_state["selected_tokens"].append(token_id)
        
        # Update attention visualization with highlights
        updated_attention_fig = update_attention_highlights(
            attention_fig, updated_state["selected_tokens"]
        )
        
        # Update correlation visualization with highlights
        updated_correlation_fig = update_correlation_highlights(
            correlation_fig, updated_state["selected_tokens"]
        )
        
        return updated_state, updated_attention_fig, updated_correlation_fig
    
    # Attention Sankey selection callback
    @callback(
        [Output(f"selection-state-{layout_id}", "data", allow_duplicate=True),
         Output(f"token-sankey-{layout_id}", "figure"),
         Output(f"correlation-heatmap-{layout_id}", "figure", allow_duplicate=True)],
        [Input(f"attention-sankey-{layout_id}", "clickData")],
        [State(f"selection-state-{layout_id}", "data"),
         State(f"token-sankey-{layout_id}", "figure"),
         State(f"correlation-heatmap-{layout_id}", "figure")],
        prevent_initial_call=True
    )
    def handle_attention_sankey_selection(click_data, selection_state, token_fig, correlation_fig):
        """Handle attention Sankey selection and update other visualizations."""
        
        if not click_data:
            return selection_state, token_fig, correlation_fig
        
        # Parse attention click data
        point_data = click_data["points"][0]
        
        # Update selection state based on attention selection
        updated_state = dict(selection_state)
        
        if "source" in point_data and "target" in point_data:
            # Handle link selection
            source_idx = point_data["source"]
            target_idx = point_data["target"]
            edge = (source_idx, target_idx)
            
            if edge not in updated_state["selected_attention_edges"]:
                updated_state["selected_attention_edges"].append(edge)
            else:
                updated_state["selected_attention_edges"].remove(edge)
        
        # Update token visualization with highlights
        updated_token_fig = update_token_highlights(
            token_fig, updated_state["selected_attention_edges"]
        )
        
        # Update correlation visualization
        updated_correlation_fig = update_correlation_highlights(
            correlation_fig, updated_state["selected_tokens"]
        )
        
        return updated_state, updated_token_fig, updated_correlation_fig
    
    # Clear selections callback
    @callback(
        [Output(f"selection-state-{layout_id}", "data", allow_duplicate=True),
         Output(f"token-sankey-{layout_id}", "figure", allow_duplicate=True),
         Output(f"attention-sankey-{layout_id}", "figure", allow_duplicate=True),
         Output(f"correlation-heatmap-{layout_id}", "figure", allow_duplicate=True)],
        [Input(f"clear-selections-{layout_id}", "n_clicks")],
        [State(f"token-sankey-{layout_id}", "figure"),
         State(f"attention-sankey-{layout_id}", "figure"),
         State(f"correlation-heatmap-{layout_id}", "figure")],
        prevent_initial_call=True
    )
    def clear_all_selections(n_clicks, token_fig, attention_fig, correlation_fig):
        """Clear all selections across visualizations."""
        
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Reset selection state
        empty_state = {
            "selected_tokens": [],
            "selected_attention_edges": [],
            "selected_clusters": [],
            "selected_layers": [],
            "selected_positions": []
        }
        
        # Clear highlights from all visualizations
        cleared_token_fig = clear_highlights(token_fig)
        cleared_attention_fig = clear_highlights(attention_fig)
        cleared_correlation_fig = clear_highlights(correlation_fig)
        
        return empty_state, cleared_token_fig, cleared_attention_fig, cleared_correlation_fig
    
    # Selection summary callback
    @callback(
        Output(f"selection-summary-{layout_id}", "children"),
        [Input(f"selection-state-{layout_id}", "data")]
    )
    def update_selection_summary(selection_state):
        """Update selection summary display."""
        
        if not selection_state:
            return "No selections"
        
        summary_parts = []
        
        if selection_state.get("selected_tokens"):
            summary_parts.append(f"{len(selection_state['selected_tokens'])} tokens")
        
        if selection_state.get("selected_attention_edges"):
            summary_parts.append(f"{len(selection_state['selected_attention_edges'])} attention edges")
        
        if selection_state.get("selected_clusters"):
            summary_parts.append(f"{len(selection_state['selected_clusters'])} clusters")
        
        if not summary_parts:
            return "No selections"
        
        return f"Selected: {', '.join(summary_parts)}"


def update_token_highlights(fig: go.Figure, selected_attention_edges: List[Tuple[int, int]]) -> go.Figure:
    """Update token Sankey highlights based on attention edge selection."""
    
    updated_fig = go.Figure(fig)
    
    # Find tokens involved in selected attention edges
    highlighted_tokens = set()
    for source_idx, target_idx in selected_attention_edges:
        # Map attention indices to token IDs (implementation depends on data structure)
        highlighted_tokens.add(source_idx)
        highlighted_tokens.add(target_idx)
    
    # Update node colors for highlighted tokens
    if hasattr(updated_fig.data[0], 'node') and highlighted_tokens:
        node_colors = list(updated_fig.data[0].node.color) if updated_fig.data[0].node.color else ["lightblue"] * len(updated_fig.data[0].node.label)
        
        for i, token_idx in enumerate(highlighted_tokens):
            if token_idx < len(node_colors):
                node_colors[token_idx] = "#ff6b6b"  # Highlight color
        
        updated_fig.data[0].node.color = node_colors
    
    return updated_fig


def update_attention_highlights(fig: go.Figure, selected_tokens: List[str]) -> go.Figure:
    """Update attention Sankey highlights based on token selection."""
    
    updated_fig = go.Figure(fig)
    
    if not selected_tokens:
        return updated_fig
    
    # Update link colors for attention edges involving selected tokens
    if hasattr(updated_fig.data[0], 'link') and selected_tokens:
        link_colors = list(updated_fig.data[0].link.color) if updated_fig.data[0].link.color else ["lightgray"] * len(updated_fig.data[0].link.source)
        
        # Find links involving selected tokens (implementation depends on data mapping)
        for i, (source, target) in enumerate(zip(updated_fig.data[0].link.source, updated_fig.data[0].link.target)):
            # Check if this link involves any selected tokens
            if any(token in str(source) or token in str(target) for token in selected_tokens):
                link_colors[i] = "#ff6b6b"  # Highlight color
        
        updated_fig.data[0].link.color = link_colors
    
    return updated_fig


def update_correlation_highlights(fig: go.Figure, selected_tokens: List[str]) -> go.Figure:
    """Update correlation heatmap highlights based on token selection."""
    
    updated_fig = go.Figure(fig)
    
    if not selected_tokens:
        return updated_fig
    
    # Add annotation or overlay for selected tokens in correlation matrix
    for token in selected_tokens:
        # Add highlighting annotation (implementation depends on correlation data structure)
        updated_fig.add_annotation(
            text=f"Selected: {token}",
            xref="paper", yref="paper",
            x=0.02, y=0.98 - len(updated_fig.layout.annotations) * 0.05,
            showarrow=False,
            font=dict(size=10, color="#ff6b6b"),
            bgcolor="rgba(255, 255, 255, 0.8)"
        )
    
    return updated_fig


def clear_highlights(fig: go.Figure) -> go.Figure:
    """Clear all highlights from a visualization."""
    
    updated_fig = go.Figure(fig)
    
    # Reset colors to defaults (implementation depends on visualization type)
    if hasattr(updated_fig.data[0], 'node'):
        # Sankey diagram - reset node colors
        if updated_fig.data[0].node.color:
            default_color = "lightblue"
            updated_fig.data[0].node.color = [default_color] * len(updated_fig.data[0].node.label)
    
    if hasattr(updated_fig.data[0], 'link'):
        # Sankey diagram - reset link colors
        if updated_fig.data[0].link.color:
            default_color = "lightgray"
            updated_fig.data[0].link.color = [default_color] * len(updated_fig.data[0].link.source)
    
    # Clear annotations
    updated_fig.layout.annotations = []
    
    return updated_fig


# Client-side callback for performance optimization
clientside_callback(
    """
    function(clickData, currentSelections) {
        if (!clickData) {
            return window.dash_clientside.no_update;
        }
        
        // Handle rapid selection updates on the client side
        // This improves responsiveness for frequent interactions
        
        const point = clickData.points[0];
        const tokenId = point.customdata && point.customdata.token_id;
        
        if (tokenId) {
            const selections = currentSelections || {selected_tokens: []};
            const tokenIndex = selections.selected_tokens.indexOf(tokenId);
            
            if (tokenIndex > -1) {
                selections.selected_tokens.splice(tokenIndex, 1);
            } else {
                selections.selected_tokens.push(tokenId);
            }
            
            return selections;
        }
        
        return window.dash_clientside.no_update;
    }
    """,
    Output({"type": "selection-state", "index": MATCH}, "data"),
    [Input({"type": "token-sankey", "index": MATCH}, "clickData")],
    [State({"type": "selection-state", "index": MATCH}, "data")]
)


def create_cross_highlighting_demo() -> html.Div:
    """Create a demonstration of cross-visualization highlighting."""
    
    return html.Div([
        html.H3("Cross-Visualization Highlighting Demo", className="mb-4"),
        
        dbc.Alert([
            html.H5("How to Use:", className="alert-heading"),
            html.P("1. Click on tokens in the Token Paths visualization to highlight related attention patterns"),
            html.P("2. Click on attention edges in the Attention Flow to highlight involved tokens"),
            html.P("3. Use 'Clear All Selections' to reset all highlights"),
            html.P("4. The correlation heatmap updates to show relationships for selected elements")
        ], color="info", className="mb-4"),
        
        # Mock synchronized visualization layout
        create_synchronized_visualization_layout(
            token_sankey_data={},  # Would be populated with real data
            attention_sankey_data={},
            correlation_data={},
            layout_style="grid"
        )
    ])


if __name__ == "__main__":
    # Example usage
    print("Cross-Visualization Highlighting System for GPT-2 APA")
    print("=" * 60)
    print("Components created:")
    print("- CrossVisualizationHighlighter: Main coordination class")
    print("- create_synchronized_visualization_layout(): Layout generator")
    print("- Selection callbacks: Dash callback functions")
    print("- Highlight update functions: Visual update handlers")
    print("\nIntegration: Add to existing dashboard tabs for synchronized exploration")