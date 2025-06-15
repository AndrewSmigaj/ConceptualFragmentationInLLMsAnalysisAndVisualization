"""
DetailsPanel component for showing entity details and analysis.

This component shows:
- EntityCard (dynamic based on selection)
- AnalysisPanel with LLM insights
"""

from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, List
import json

from .entity_card import EntityCard, ClusterCard, PathCard, SampleCard


class DetailsPanel:
    """Panel for displaying entity details and analysis."""
    
    def __init__(self):
        """Initialize the DetailsPanel."""
        self.id_prefix = "details-panel"
        self.current_selection = None
        self.entity_card = EntityCard()
        
    def create_component(self) -> html.Div:
        """Create and return the details panel."""
        return html.Div([
            # Header with close button
            html.Div([
                html.H5("Details", className="mb-0"),
                dbc.Button(
                    html.I(className="fas fa-times"),
                    id=f"{self.id_prefix}-close-btn",
                    color="link",
                    size="sm",
                    className="float-end",
                    style={"display": "none"}  # Hidden by default
                )
            ], className="details-panel-header p-3 border-bottom"),
            
            # Selection indicator
            html.Div(
                id=f"{self.id_prefix}-selection-indicator",
                className="text-muted small px-3 py-1"
            ),
            
            # Content area (scrollable)
            html.Div([
                # Entity Card section
                html.Div(
                    id=f"{self.id_prefix}-entity-container",
                    className="entity-card-section mb-3",
                    children=[self._create_placeholder()]
                ),
                
                # Analysis Panel section
                html.Div(
                    id=f"{self.id_prefix}-analysis",
                    className="analysis-section",
                    children=[self._create_analysis_panel()]
                )
            ], className="details-panel-content flex-grow-1 overflow-auto p-3"),
            
            # Actions footer
            html.Div([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-brain me-1"), "Analyze"],
                        id=f"{self.id_prefix}-analyze-btn",
                        color="primary",
                        size="sm",
                        disabled=True
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-chart-line me-1"), "Compare"],
                        id=f"{self.id_prefix}-compare-btn",
                        color="secondary",
                        size="sm",
                        disabled=True
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-download me-1"), "Export"],
                        id=f"{self.id_prefix}-export-btn",
                        color="secondary",
                        size="sm",
                        disabled=True
                    )
                ], size="sm", className="w-100")
            ], className="details-panel-footer p-3 border-top")
        ], className="details-panel h-100 d-flex flex-column")
    
    def _create_placeholder(self) -> html.Div:
        """Create placeholder content when nothing is selected."""
        return html.Div([
            html.Div([
                html.I(className="fas fa-mouse-pointer fa-3x text-muted mb-3"),
                html.P("Select an entity to view details", className="text-muted"),
                html.Small([
                    "You can select:",
                    html.Ul([
                        html.Li("Clusters from the network overview"),
                        html.Li("Paths from the archetypal paths panel"),
                        html.Li("Nodes from the Sankey diagram"),
                        html.Li("Points from the trajectory visualization")
                    ], className="text-start mt-2")
                ], className="text-muted")
            ], className="text-center py-5")
        ], className="h-100 d-flex align-items-center justify-content-center")
    
    def _create_analysis_panel(self) -> html.Div:
        """Create the analysis panel with category selector."""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-robot me-2"),
                "LLM Analysis",
                # Category selector in header
                html.Div([
                    dbc.Checklist(
                        id=f"{self.id_prefix}-analysis-categories",
                        options=[
                            {"label": "Interpretation", "value": "interpretation"},
                            {"label": "Bias Detection", "value": "bias"},
                            {"label": "Efficiency", "value": "efficiency"},
                            {"label": "Robustness", "value": "robustness"}
                        ],
                        value=["interpretation", "bias"],
                        inline=True,
                        switch=True,
                        className="small"
                    )
                ], className="mt-2")
            ], className="fw-bold"),
            dbc.CardBody([
                # Analysis content area
                html.Div(
                    id=f"{self.id_prefix}-analysis-content",
                    className="analysis-content overflow-auto",
                    style={"maxHeight": "40vh"},
                    children=[
                        # Placeholder content
                        dbc.Alert([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            html.Strong("Bias Alert: "),
                            "Potential demographic bias detected in path routing."
                        ], color="warning", className="mb-3"),
                        
                        html.Div([
                            html.H6("Interpretation:", className="mb-2"),
                            html.P([
                                "This path represents a progression through concepts related to...",
                                " Analysis will appear here when clustering is complete."
                            ], className="small")
                        ])
                    ]
                )
            ])
        ], className="h-100")
    
    def create_entity_card(self, entity_type: str, entity_data: Dict[str, Any]) -> html.Div:
        """Create appropriate entity card based on type.
        
        Args:
            entity_type: Type of entity ('cluster', 'path', 'sample')
            entity_data: Entity data
            
        Returns:
            Entity card component
        """
        if entity_type == 'cluster':
            card = ClusterCard()
            return card.create_card(entity_data)
        elif entity_type == 'path':
            card = PathCard()
            return card.create_card(entity_data)
        elif entity_type == 'sample':
            card = SampleCard()
            return card.create_card(entity_data)
        else:
            return self._create_placeholder()
    
    def get_selection_info(self, selection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract selection information from various sources.
        
        Args:
            selection_data: Selection data from stores
            
        Returns:
            Processed selection info
        """
        # Determine selection type and extract relevant data
        if not selection_data:
            return {"type": None, "data": None}
        
        # Check for cluster selection
        if "cluster_id" in selection_data:
            return {
                "type": "cluster",
                "data": {
                    "id": selection_data["cluster_id"],
                    "layer": selection_data.get("layer", "unknown"),
                    "label": selection_data.get("label", f"Cluster {selection_data['cluster_id']}"),
                    "size": selection_data.get("size", 0),
                    "samples": selection_data.get("samples", []),
                    "features": selection_data.get("features", {}),
                    "metrics": selection_data.get("metrics", {}),
                    "llm_analysis": selection_data.get("llm_analysis", None)
                }
            }
        
        # Check for path selection
        elif "path_id" in selection_data:
            return {
                "type": "path",
                "data": {
                    "id": selection_data["path_id"],
                    "sequence": selection_data.get("sequence", []),
                    "frequency": selection_data.get("frequency", 0),
                    "samples": selection_data.get("samples", []),
                    "transitions": selection_data.get("transitions", []),
                    "stability": selection_data.get("stability", 0),
                    "pattern": selection_data.get("pattern", "unknown"),
                    "llm_analysis": selection_data.get("llm_analysis", None)
                }
            }
        
        # Check for sample selection
        elif "sample_id" in selection_data:
            return {
                "type": "sample",
                "data": {
                    "id": selection_data["sample_id"],
                    "text": selection_data.get("text", ""),
                    "trajectory": selection_data.get("trajectory", []),
                    "clusters": selection_data.get("clusters", []),
                    "activations": selection_data.get("activations", {}),
                    "metadata": selection_data.get("metadata", {})
                }
            }
        
        return {"type": None, "data": None}