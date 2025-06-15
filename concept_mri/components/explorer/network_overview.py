"""
NetworkOverview component for displaying network-wide metrics.

This component shows:
- Multi-metric visualization across all layers
- Window selection (Early/Middle/Late)
- Metric selection checkboxes
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

from .metrics_chart import MetricsChart
from .window_selector import WindowSelector

class NetworkOverview:
    """Network overview component showing metrics across all layers."""
    
    def __init__(self):
        """Initialize the NetworkOverview."""
        self.id_prefix = "network-overview"
        self.metrics_chart = MetricsChart()
        self.window_selector = WindowSelector()
        
    def create_component(self) -> html.Div:
        """Create and return the network overview component."""
        return html.Div([
            # Metrics visualization area
            html.Div([
                dcc.Graph(
                    id=f"{self.id_prefix}-metrics-chart",
                    figure=self.metrics_chart.create_empty_figure(),
                    config={'displayModeBar': False},
                    className="network-metrics-chart"
                )
            ], style={'height': '65%', 'position': 'relative'}),
            
            # Controls row
            html.Div([
                # Window selector buttons
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button(
                            "Early Window", 
                            id=f"{self.id_prefix}-early-btn",
                            color="primary", 
                            outline=True, 
                            size="sm",
                            className="window-selector-btn",
                            active=True
                        ),
                        dbc.Button(
                            "Middle Window", 
                            id=f"{self.id_prefix}-middle-btn",
                            color="primary", 
                            outline=True, 
                            size="sm",
                            className="window-selector-btn"
                        ),
                        dbc.Button(
                            "Late Window", 
                            id=f"{self.id_prefix}-late-btn",
                            color="primary", 
                            outline=True, 
                            size="sm",
                            className="window-selector-btn"
                        ),
                        dbc.Button(
                            "Custom", 
                            id=f"{self.id_prefix}-custom-btn",
                            color="primary", 
                            outline=True, 
                            size="sm",
                            className="window-selector-btn"
                        )
                    ], id=f"{self.id_prefix}-window-selector")
                ], className="mb-2 text-center"),
                
                # Metric checkboxes
                html.Div([
                    html.Label("Metrics: ", className="me-2 small fw-bold"),
                    dbc.Checklist(
                        id=f"{self.id_prefix}-metric-selector",
                        options=[
                            {"label": " Fragmentation", "value": "fragmentation"},
                            {"label": " Cohesion", "value": "cohesion"},
                            {"label": " Entropy", "value": "entropy"},
                            {"label": " Path Diversity", "value": "path_diversity"},
                            {"label": " Stability", "value": "stability"}
                        ],
                        value=["fragmentation", "cohesion", "entropy"],
                        inline=True,
                        switch=True,
                        className="small"
                    )
                ], className="d-flex align-items-center justify-content-center"),
                
                # Custom window modal
                self.window_selector.create_custom_window_modal()
            ], className="p-2", style={'height': '35%'})
        ], className="h-100 d-flex flex-column")
    
    def update_metrics_chart(self, 
                           metrics_data: Optional[Dict[str, List[float]]],
                           layer_names: Optional[List[str]],
                           selected_metrics: List[str],
                           window_selection: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Update the metrics chart with new data."""
        if not metrics_data or not layer_names:
            return self.metrics_chart.create_empty_figure()
        
        return self.metrics_chart.create_figure(
            metrics_data=metrics_data,
            layer_names=layer_names,
            selected_metrics=selected_metrics,
            window_selection=window_selection
        )