"""
PathCard component for displaying individual path information.

This component creates interactive cards showing:
- Path sequence
- Frequency and percentage
- Stability indicator
- Quick statistics
"""

from dash import html
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional

class PathCard:
    """Component for creating path information cards."""
    
    def __init__(self):
        """Initialize the PathCard."""
        self.stability_colors = {
            'stable': 'success',
            'divergent': 'warning',
            'convergent': 'info',
            'fragmented': 'danger'
        }
        
    def create_card(self, 
                   path_id: str,
                   path_sequence: List[str],
                   frequency: int,
                   percentage: float,
                   stability: str,
                   metadata: Optional[Dict[str, Any]] = None) -> html.Div:
        """
        Create a path card component.
        
        Args:
            path_id: Unique identifier for the path
            path_sequence: List of cluster IDs in the path
            frequency: Number of samples following this path
            percentage: Percentage of total samples
            stability: Stability indicator ('stable', 'divergent', etc.)
            metadata: Optional additional metadata
            
        Returns:
            Path card component
        """
        # Format path sequence for display
        path_str = " → ".join(path_sequence)
        
        # Create card
        card = dbc.Card([
            dbc.CardBody([
                # Header row with path number and badges
                html.Div([
                    html.Strong(f"Path {path_id} ", className="me-2"),
                    dbc.Badge(f"{percentage:.1f}%", color="primary", className="me-2"),
                    dbc.Badge(f"{frequency} samples", color="secondary", className="me-2"),
                    dbc.Badge(
                        stability.capitalize(),
                        color=self.stability_colors.get(stability, 'secondary'),
                        pill=True
                    )
                ], className="d-flex align-items-center mb-2"),
                
                # Path sequence
                html.Div([
                    html.Small(path_str, className="font-monospace text-muted")
                ], className="mb-2"),
                
                # Additional metadata if provided
                self._create_metadata_section(metadata) if metadata else None
            ], className="p-3")
        ], className="mb-2 path-card shadow-sm")
        
        return html.Div(card, id=f"path-card-{path_id}", className="path-card-wrapper")
    
    def _create_metadata_section(self, metadata: Dict[str, Any]) -> html.Div:
        """Create metadata section for the card."""
        items = []
        
        # Add demographic info if available
        if 'demographics' in metadata:
            demo = metadata['demographics']
            demo_items = []
            
            for key, value in demo.items():
                if isinstance(value, dict):
                    # Format distribution
                    top_item = max(value.items(), key=lambda x: x[1])
                    demo_items.append(f"{key}: {top_item[0]} ({top_item[1]:.0%})")
                else:
                    demo_items.append(f"{key}: {value}")
            
            if demo_items:
                items.append(html.Small(" | ".join(demo_items[:3]), className="text-muted"))
        
        # Add metrics if available
        if 'metrics' in metadata:
            metrics = metadata['metrics']
            if 'fragmentation' in metrics:
                items.append(
                    html.Small(f"Fragmentation: {metrics['fragmentation']:.3f}", 
                              className="text-muted")
                )
        
        return html.Div(items) if items else None
    
    def create_mini_card(self, 
                        path_id: str,
                        path_sequence: List[str],
                        percentage: float) -> html.Div:
        """Create a compact version of the path card."""
        path_str = " → ".join(path_sequence[:3])
        if len(path_sequence) > 3:
            path_str += " → ..."
            
        return html.Div([
            html.Span(f"Path {path_id}", className="fw-bold me-2"),
            html.Span(f"{percentage:.1f}%", className="text-muted me-2"),
            html.Small(path_str, className="font-monospace text-muted")
        ], className="p-2 border-bottom")