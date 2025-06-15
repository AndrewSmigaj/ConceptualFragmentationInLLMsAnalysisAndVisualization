"""
VisualizationPanel component for displaying network visualizations.

This component shows:
- Visualization switcher (Sankey/Trajectory/3D)
- Active visualization
- Visualization controls
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, List
import plotly.graph_objects as go

class VisualizationPanel:
    """Panel for displaying network visualizations."""
    
    def __init__(self):
        """Initialize the VisualizationPanel."""
        self.id_prefix = "viz-panel"
        self.current_viz_type = "sankey"
        
    def create_component(self) -> html.Div:
        """Create and return the visualization panel."""
        return html.Div([
            # Store for current visualization type
            dcc.Store(id=f"{self.id_prefix}-current-type", data="sankey"),
            
            # Visualization container
            html.Div(
                id=f"{self.id_prefix}-container",
                className="viz-container p-3",
                style={"height": "calc(100% - 70px)", "position": "relative"},
                children=[
                    # Loading indicator
                    dcc.Loading(
                        id=f"{self.id_prefix}-loading",
                        type="circle",
                        children=html.Div(
                            id=f"{self.id_prefix}-content",
                            style={"height": "100%"},
                            children=[self._create_placeholder()]
                        )
                    )
                ]
            ),
            
            # Controls section
            html.Div([
                dbc.Row([
                    # Visualization type selector
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button(
                                "Sankey", 
                                id=f"{self.id_prefix}-sankey-btn",
                                color="primary", 
                                outline=True, 
                                size="sm", 
                                active=True
                            ),
                            dbc.Button(
                                "Trajectory", 
                                id=f"{self.id_prefix}-trajectory-btn",
                                color="primary", 
                                outline=True, 
                                size="sm"
                            ),
                            dbc.Button(
                                "3D Network", 
                                id=f"{self.id_prefix}-3d-btn",
                                color="primary", 
                                outline=True, 
                                size="sm", 
                                disabled=True
                            )
                        ], id=f"{self.id_prefix}-type-selector")
                    ], width="auto"),
                    
                    # Visualization controls
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Color by:", className="small me-2"),
                                dbc.Select(
                                    id=f"{self.id_prefix}-color-scheme",
                                    options=[
                                        {"label": "Cluster", "value": "cluster"},
                                        {"label": "Path", "value": "path"},
                                        {"label": "Stability", "value": "stability"},
                                        {"label": "Layer", "value": "layer"}
                                    ],
                                    value="cluster",
                                    size="sm",
                                    style={"width": "120px"}
                                )
                            ], width="auto", className="d-flex align-items-center"),
                            
                            dbc.Col([
                                html.Label("Layout:", className="small me-2"),
                                dbc.Select(
                                    id=f"{self.id_prefix}-layout",
                                    options=[
                                        {"label": "Standard", "value": "standard"},
                                        {"label": "Compact", "value": "compact"},
                                        {"label": "Expanded", "value": "expanded"}
                                    ],
                                    value="standard",
                                    size="sm",
                                    style={"width": "120px"}
                                )
                            ], width="auto", className="d-flex align-items-center"),
                            
                            dbc.Col([
                                html.Label("Highlight:", className="small me-2"),
                                dbc.Select(
                                    id=f"{self.id_prefix}-highlight",
                                    options=[
                                        {"label": "None", "value": "none"}
                                    ],
                                    value="none",
                                    size="sm",
                                    style={"width": "150px"}
                                )
                            ], width="auto", className="d-flex align-items-center")
                        ], className="g-2")
                    ], width=True),
                    
                    # Toggle options
                    dbc.Col([
                        dbc.Checklist(
                            id=f"{self.id_prefix}-toggles",
                            options=[
                                {"label": " Show all paths", "value": "show_all"},
                                {"label": " Normalize", "value": "normalize"},
                                {"label": " Compare windows", "value": "compare"}
                            ],
                            value=["normalize"],
                            inline=True,
                            switch=True,
                            className="small"
                        )
                    ], width="auto")
                ], className="align-items-center")
            ], className="p-2 border-top bg-light", style={"height": "70px"})
        ], className="h-100 d-flex flex-column")
    
    def _create_placeholder(self) -> html.Div:
        """Create placeholder content."""
        return html.Div([
            html.I(className="fas fa-chart-line fa-3x text-muted mb-3"),
            html.P("Run clustering to see visualizations", className="text-muted")
        ], className="text-center d-flex flex-column align-items-center justify-content-center h-100")
    
    def create_sankey_container(self) -> html.Div:
        """Create container for Sankey visualization."""
        return html.Div([
            dcc.Graph(
                id=f"{self.id_prefix}-sankey-graph",
                figure=self._create_empty_sankey(),
                config={'displayModeBar': False},
                style={"height": "100%"}
            )
        ], style={"height": "100%"})
    
    def create_trajectory_container(self) -> html.Div:
        """Create container for trajectory visualization."""
        return html.Div([
            dcc.Graph(
                id=f"{self.id_prefix}-trajectory-graph",
                figure=self._create_empty_trajectory(),
                config={'displayModeBar': False},
                style={"height": "100%"}
            )
        ], style={"height": "100%"})
    
    def _create_empty_sankey(self) -> go.Figure:
        """Create an empty Sankey diagram."""
        fig = go.Figure()
        fig.add_annotation(
            text="No Sankey data available.<br>Waiting for clustering results.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def _create_empty_trajectory(self) -> go.Figure:
        """Create an empty trajectory plot."""
        fig = go.Figure()
        fig.add_annotation(
            text="No trajectory data available.<br>Waiting for clustering results.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    def update_highlight_options(self, paths_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Update highlight dropdown options based on available paths."""
        options = [{"label": "None", "value": "none"}]
        
        if paths_data:
            # Add top paths as highlight options
            for i, path in enumerate(paths_data[:10]):  # Limit to top 10
                path_id = path.get('id', i)
                percentage = path.get('percentage', 0)
                options.append({
                    "label": f"Path {path_id} ({percentage:.1f}%)",
                    "value": f"path-{path_id}"
                })
        
        return options