"""
WindowSelector component for selecting network windows.

This component provides predefined window selections (Early/Middle/Late)
and custom window definition capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
from dash import html, dcc
import dash_bootstrap_components as dbc

class WindowSelector:
    """Component for selecting analysis windows in the network."""
    
    def __init__(self):
        """Initialize the WindowSelector."""
        self.id_prefix = "window-selector"
        
    def get_predefined_windows(self, n_layers: int) -> Dict[str, Tuple[int, int]]:
        """
        Get predefined window definitions based on number of layers.
        
        Args:
            n_layers: Total number of layers in the network
            
        Returns:
            Dictionary mapping window names to (start, end) layer indices
        """
        if n_layers <= 3:
            return {
                'early': (0, n_layers - 1),
                'middle': (0, n_layers - 1),
                'late': (0, n_layers - 1)
            }
        
        # Calculate window boundaries
        third = n_layers // 3
        
        return {
            'early': (0, third),
            'middle': (third, 2 * third),
            'late': (2 * third, n_layers - 1)
        }
    
    def create_custom_window_modal(self) -> dbc.Modal:
        """Create modal for custom window definition."""
        return dbc.Modal([
            dbc.ModalHeader("Define Custom Window"),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Start Layer"),
                            dbc.Select(
                                id=f"{self.id_prefix}-custom-start",
                                options=[],  # Will be populated dynamically
                                value=""
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("End Layer"),
                            dbc.Select(
                                id=f"{self.id_prefix}-custom-end",
                                options=[],  # Will be populated dynamically
                                value=""
                            )
                        ], width=6)
                    ]),
                    dbc.Alert(
                        "Select start and end layers for your custom window.",
                        id=f"{self.id_prefix}-custom-feedback",
                        color="info",
                        className="mt-3"
                    )
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id=f"{self.id_prefix}-custom-cancel", 
                          color="secondary", outline=True),
                dbc.Button("Apply", id=f"{self.id_prefix}-custom-apply",
                          color="primary")
            ])
        ], id=f"{self.id_prefix}-custom-modal", is_open=False)
    
    def format_window_label(self, window_type: str, start_layer: int, 
                           end_layer: int, layer_names: List[str]) -> str:
        """
        Format a human-readable window label.
        
        Args:
            window_type: Type of window ('early', 'middle', 'late', 'custom')
            start_layer: Start layer index
            end_layer: End layer index
            layer_names: List of layer names
            
        Returns:
            Formatted window label
        """
        if window_type == 'custom':
            return f"Custom ({layer_names[start_layer]}-{layer_names[end_layer]})"
        
        window_name = window_type.capitalize()
        return f"{window_name} Window ({layer_names[start_layer]}-{layer_names[end_layer]})"