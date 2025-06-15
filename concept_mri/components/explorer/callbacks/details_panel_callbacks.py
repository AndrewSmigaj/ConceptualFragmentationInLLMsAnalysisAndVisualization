"""
Callbacks for DetailsPanel component.

Handles:
- Entity selection from various sources
- Dynamic card updates
- LLM analysis triggers
- Export functionality
"""

from dash import callback, Input, Output, State, ctx, ALL, html
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, List
import json
import pandas as pd

from ..details_panel import DetailsPanel
from ..entity_card import ClusterCard, PathCard, SampleCard

# Create instance for helper methods
details_panel = DetailsPanel()


def register_details_panel_callbacks(app):
    """Register callbacks for the DetailsPanel component."""
    
    @app.callback(
        [Output("details-panel-entity-container", "children"),
         Output("details-panel-selection-indicator", "children"),
         Output("details-panel-analyze-btn", "disabled"),
         Output("details-panel-compare-btn", "disabled"),
         Output("details-panel-export-btn", "disabled")],
        [Input("network-explorer-selection-store", "data"),
         Input("network-explorer-paths-store", "data"),
         Input("network-explorer-clustering-results-store", "data")]
    )
    def update_entity_card(selection_data, paths_data, clustering_data):
        """Update entity card based on current selection."""
        if not selection_data:
            return (
                details_panel._create_placeholder(),
                "No selection",
                True, True, True
            )
        
        # Get selection info
        selection_info = details_panel.get_selection_info(selection_data)
        
        if selection_info["type"] is None:
            return (
                details_panel._create_placeholder(),
                "No selection",
                True, True, True
            )
        
        # Create appropriate card
        entity_card = details_panel.create_entity_card(
            selection_info["type"],
            selection_info["data"]
        )
        
        # Create selection indicator text
        indicator_text = f"Selected: {selection_info['type'].capitalize()} - {selection_info['data'].get('id', 'Unknown')}"
        
        # Enable appropriate buttons
        analyze_disabled = selection_info["data"].get("llm_analysis") is not None
        compare_disabled = selection_info["type"] == "sample"  # Can't compare samples
        export_disabled = False  # Always allow export
        
        return (
            entity_card,
            indicator_text,
            analyze_disabled,
            compare_disabled,
            export_disabled
        )
    
    @app.callback(
        Output("details-panel-analysis-content", "children"),
        [Input("details-panel-analysis-categories", "value"),
         Input("network-explorer-selection-store", "data")]
    )
    def update_analysis_content(selected_categories, selection_data):
        """Update analysis panel content based on selected categories."""
        if not selection_data or not selected_categories:
            return html.P("Select analysis categories to view insights", 
                         className="text-muted text-center py-3")
        
        # Get selection info
        selection_info = details_panel.get_selection_info(selection_data)
        
        if not selection_info["data"] or not selection_info["data"].get("llm_analysis"):
            return html.P("No analysis available. Click 'Analyze' to generate insights.",
                         className="text-muted text-center py-3")
        
        llm_analysis = selection_info["data"]["llm_analysis"]
        content = []
        
        # Add content for each selected category
        if "bias" in selected_categories and "bias" in llm_analysis:
            content.append(
                dbc.Alert([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    html.Strong("Bias Alert: "),
                    llm_analysis["bias"]["summary"]
                ], color="warning", className="mb-3")
            )
        
        if "interpretation" in selected_categories and "interpretation" in llm_analysis:
            content.append(html.Div([
                html.H6("Interpretation:", className="mb-2"),
                html.P(llm_analysis["interpretation"]["description"], className="small"),
                # Add key insights if available
                html.Ul([
                    html.Li(insight, className="small")
                    for insight in llm_analysis["interpretation"].get("insights", [])
                ])
            ], className="mb-3"))
        
        if "efficiency" in selected_categories and "efficiency" in llm_analysis:
            content.append(html.Div([
                html.H6("Efficiency Analysis:", className="mb-2"),
                html.P(llm_analysis["efficiency"]["summary"], className="small"),
                # Add metrics if available
                html.Div([
                    html.Span(f"{metric}: ", className="fw-bold"),
                    html.Span(f"{value:.3f} ")
                    for metric, value in llm_analysis["efficiency"].get("metrics", {}).items()
                ], className="small")
            ], className="mb-3"))
        
        if "robustness" in selected_categories and "robustness" in llm_analysis:
            content.append(html.Div([
                html.H6("Robustness Analysis:", className="mb-2"),
                html.P(llm_analysis["robustness"]["assessment"], className="small"),
                # Add vulnerability warnings if any
                *[dbc.Alert(vuln, color="danger", className="small")
                  for vuln in llm_analysis["robustness"].get("vulnerabilities", [])]
            ], className="mb-3"))
        
        return content if content else html.P("No analysis available for selected categories",
                                             className="text-muted text-center py-3")
    
    @app.callback(
        [Output("network-explorer-selection-store", "data", allow_duplicate=True),
         Output("llm-analysis-trigger-store", "data")],
        Input("details-panel-analyze-btn", "n_clicks"),
        State("network-explorer-selection-store", "data"),
        prevent_initial_call=True
    )
    def trigger_llm_analysis(n_clicks, selection_data):
        """Trigger LLM analysis for selected entity."""
        if not n_clicks or not selection_data:
            return selection_data, None
        
        # Create analysis trigger
        selection_info = details_panel.get_selection_info(selection_data)
        
        analysis_trigger = {
            "entity_type": selection_info["type"],
            "entity_id": selection_info["data"]["id"],
            "entity_data": selection_info["data"],
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return selection_data, analysis_trigger
    
    @app.callback(
        Output("download-entity-data", "data"),
        Input("details-panel-export-btn", "n_clicks"),
        State("network-explorer-selection-store", "data"),
        prevent_initial_call=True
    )
    def export_entity_data(n_clicks, selection_data):
        """Export selected entity data."""
        if not n_clicks or not selection_data:
            return None
        
        selection_info = details_panel.get_selection_info(selection_data)
        
        # Prepare export data
        export_data = {
            "entity_type": selection_info["type"],
            "entity_data": selection_info["data"],
            "export_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Convert to JSON string
        json_str = json.dumps(export_data, indent=2)
        
        # Create filename
        filename = f"{selection_info['type']}_{selection_info['data']['id']}_export.json"
        
        return {
            "content": json_str,
            "filename": filename,
            "type": "application/json"
        }
    
    @app.callback(
        Output("network-explorer-comparison-store", "data"),
        [Input("details-panel-compare-btn", "n_clicks"),
         Input({"type": "selectable-entity", "id": ALL}, "n_clicks")],
        [State("network-explorer-selection-store", "data"),
         State("network-explorer-comparison-store", "data")],
        prevent_initial_call=True
    )
    def handle_comparison_mode(compare_clicks, entity_clicks, selection_data, comparison_data):
        """Handle comparison mode and multi-selection."""
        ctx_triggered = ctx.triggered_id
        
        if not ctx_triggered:
            return comparison_data
        
        # Check if compare button was clicked
        if ctx_triggered == "details-panel-compare-btn":
            # Toggle comparison mode
            if comparison_data and comparison_data.get("active"):
                # Deactivate comparison mode
                return {"active": False, "entities": []}
            else:
                # Activate comparison mode with current selection
                if selection_data:
                    selection_info = details_panel.get_selection_info(selection_data)
                    return {
                        "active": True,
                        "entities": [{
                            "type": selection_info["type"],
                            "id": selection_info["data"]["id"],
                            "data": selection_info["data"]
                        }]
                    }
                else:
                    return {"active": True, "entities": []}
        
        # Handle entity clicks in comparison mode
        if comparison_data and comparison_data.get("active"):
            # Add/remove entity from comparison
            # This would need the actual entity data from the click
            pass
        
        return comparison_data