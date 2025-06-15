"""
Main NetworkExplorer container component.

This component coordinates the overall layout and communication between:
- NetworkOverview (top)
- ArchetypalPathsPanel (left)
- VisualizationPanel (center)
- DetailsPanel (right)
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Optional

# Import child components
from .network_overview import NetworkOverview
from .archetypal_paths_panel import ArchetypalPathsPanel
from .visualization_panel import VisualizationPanel
from .details_panel import DetailsPanel

# Import stores configuration
from .stores_config import create_stores

class NetworkExplorer:
    """Main container for the unified network exploration interface."""
    
    def __init__(self):
        """Initialize the NetworkExplorer."""
        self.id_prefix = "network-explorer"
        
        # Initialize child components
        self.network_overview = NetworkOverview()
        self.paths_panel = ArchetypalPathsPanel()
        self.viz_panel = VisualizationPanel()
        self.details_panel = DetailsPanel()
        
    def create_component(self) -> html.Div:
        """Create and return the complete network explorer layout."""
        return html.Div([
            # Header bar
            self._create_header(),
            
            # Main content area
            html.Div([
                # Network Overview (15% height)
                html.Div(
                    id=f"{self.id_prefix}-overview",
                    className="network-overview-section",
                    style={
                        "height": "15vh",
                        "borderBottom": "1px solid #dee2e6",
                        "overflow": "hidden"
                    },
                    children=[
                        self.network_overview.create_component()
                    ]
                ),
                
                # Explorer Workspace (85% height)
                html.Div([
                    dbc.Row([
                        # Archetypal Paths Panel (25% width)
                        dbc.Col(
                            id=f"{self.id_prefix}-paths-panel",
                            width=3,
                            className="paths-panel border-end h-100",
                            style={"overflow": "hidden"},
                            children=[
                                self.paths_panel.create_component()
                            ]
                        ),
                        
                        # Visualization Panel (50% width)
                        dbc.Col(
                            id=f"{self.id_prefix}-viz-panel",
                            width=6,
                            className="viz-panel h-100 px-3",
                            style={"overflow": "hidden"},
                            children=[
                                self.viz_panel.create_component()
                            ]
                        ),
                        
                        # Details Panel (25% width)
                        dbc.Col(
                            id=f"{self.id_prefix}-details-panel",
                            width=3,
                            className="details-panel border-start h-100",
                            style={"overflow": "hidden"},
                            children=[
                                self.details_panel.create_component()
                            ]
                        )
                    ], className="g-0 h-100")
                ], style={
                    "height": "calc(85vh - 48px)",  # Subtract header height
                    "overflow": "hidden"
                }, className="explorer-workspace")
            ], className="network-explorer-content flex-grow-1 overflow-hidden"),
            
            # State management stores - using centralized store configuration
            *create_stores()
        ], className="network-explorer-container vh-100 d-flex flex-column", 
           style={"overflow": "hidden"})
    
    def _create_header(self) -> html.Div:
        """Create the header bar with model/dataset info and controls."""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Span("Model: ", className="fw-bold me-2"),
                    html.Span("Not loaded", id=f"{self.id_prefix}-model-name"),
                    html.Span(" | ", className="mx-2"),
                    html.Span("Dataset: ", className="fw-bold me-2"),
                    html.Span("Not loaded", id=f"{self.id_prefix}-dataset-name")
                ], width="auto"),
                
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="fas fa-cog me-2"),
                            "Settings"
                        ], size="sm", outline=True),
                        dbc.Button([
                            html.I(className="fas fa-file-export me-2"),
                            "Export"
                        ], size="sm", outline=True)
                    ])
                ], width="auto", className="ms-auto")
            ], className="align-items-center")
        ], className="network-explorer-header p-2 border-bottom bg-light")