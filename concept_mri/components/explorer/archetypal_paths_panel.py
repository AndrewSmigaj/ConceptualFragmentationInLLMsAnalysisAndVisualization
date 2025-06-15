"""
ArchetypalPathsPanel component for displaying and filtering paths.

This component shows:
- List of archetypal paths with frequency and stability indicators
- Path filters (frequency threshold, pattern matching)
- Path statistics (coverage, distribution)
"""

from dash import html, dcc, callback, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from typing import Dict, List, Any, Optional
import json

from .path_card import PathCard

class ArchetypalPathsPanel:
    """Panel for displaying and exploring archetypal paths."""
    
    def __init__(self):
        """Initialize the ArchetypalPathsPanel."""
        self.id_prefix = "archetypal-paths"
        self.path_card = PathCard()
        
    def create_component(self) -> html.Div:
        """Create and return the archetypal paths panel."""
        return html.Div([
            # Panel header
            html.Div([
                html.H5("Archetypal Paths", className="mb-0"),
                html.Small("No data loaded", 
                          id=f"{self.id_prefix}-window-label",
                          className="text-muted")
            ], className="p-3 border-bottom"),
            
            # Path list container
            html.Div(
                id=f"{self.id_prefix}-list",
                className="path-list-container overflow-auto",
                style={"height": "calc(100% - 250px)"},
                children=[
                    # Will be populated by callbacks
                    self._create_loading_placeholder()
                ]
            ),
            
            # Filters section
            html.Div([
                html.H6("Filters", className="mb-2"),
                
                # Search input
                dbc.Input(
                    id=f"{self.id_prefix}-search",
                    placeholder="Search paths...",
                    size="sm",
                    className="mb-2",
                    value=""
                ),
                
                # Frequency filter
                html.Div([
                    html.Label("Frequency", className="small"),
                    dbc.Select(
                        id=f"{self.id_prefix}-frequency-filter",
                        options=[
                            {"label": "All", "value": "0"},
                            {"label": "> 10%", "value": "10"},
                            {"label": "> 5%", "value": "5"},
                            {"label": "> 1%", "value": "1"}
                        ],
                        value="1",
                        size="sm"
                    )
                ], className="mb-2"),
                
                # Pattern filter
                html.Div([
                    html.Label("Pattern", className="small"),
                    dbc.Select(
                        id=f"{self.id_prefix}-pattern-filter",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Stable", "value": "stable"},
                            {"label": "Divergent", "value": "divergent"},
                            {"label": "Convergent", "value": "convergent"},
                            {"label": "Fragmented", "value": "fragmented"}
                        ],
                        value="all",
                        size="sm"
                    )
                ])
            ], className="p-3 border-top", style={"height": "200px"}),
            
            # Statistics footer
            html.Div([
                html.Small([
                    "Coverage: ",
                    html.Span("0%", id=f"{self.id_prefix}-coverage"),
                    " | Paths shown: ",
                    html.Span("0/0", id=f"{self.id_prefix}-count")
                ], className="text-muted")
            ], className="p-2 border-top text-center", style={"height": "50px"})
        ], className="h-100 d-flex flex-column")
    
    def _create_loading_placeholder(self) -> html.Div:
        """Create a loading placeholder."""
        return html.Div([
            html.I(className="fas fa-spinner fa-spin fa-2x text-muted mb-3"),
            html.P("Run clustering to see archetypal paths", className="text-muted")
        ], className="text-center mt-5")
    
    def create_path_list(self, paths_data: List[Dict[str, Any]], 
                        window_label: str = "All Layers") -> List[html.Div]:
        """Create a list of path cards from paths data."""
        if not paths_data:
            return [self._create_loading_placeholder()]
        
        path_cards = []
        for path_info in paths_data:
            path_id = path_info.get('id', 0)
            
            # Determine stability pattern
            stability = self._determine_stability(path_info)
            
            card = self.path_card.create_card(
                path_id=str(path_id),
                path_sequence=path_info.get('sequence', []),
                frequency=path_info.get('frequency', 0),
                percentage=path_info.get('percentage', 0.0),
                stability=stability,
                metadata=path_info.get('metadata', {})
            )
            
            # Wrap in a div with click handler ID
            path_cards.append(
                html.Div(
                    card,
                    id={'type': 'path-card-wrapper', 'index': path_id},
                    className="path-card-wrapper"
                )
            )
        
        return path_cards
    
    def _determine_stability(self, path_info: Dict[str, Any]) -> str:
        """Determine the stability pattern of a path."""
        # Use fragmentation score if available
        fragmentation = path_info.get('fragmentation', 0.5)
        
        if fragmentation < 0.3:
            return 'stable'
        elif fragmentation < 0.5:
            return 'convergent'
        elif fragmentation < 0.7:
            return 'divergent'
        else:
            return 'fragmented'
    
    def filter_paths(self, all_paths: List[Dict[str, Any]], 
                    search_term: str,
                    frequency_threshold: float,
                    pattern_filter: str) -> List[Dict[str, Any]]:
        """Filter paths based on criteria."""
        filtered = all_paths
        
        # Apply frequency filter
        if frequency_threshold > 0:
            filtered = [p for p in filtered 
                       if p.get('percentage', 0) >= frequency_threshold]
        
        # Apply pattern filter
        if pattern_filter != 'all':
            filtered = [p for p in filtered 
                       if self._determine_stability(p) == pattern_filter]
        
        # Apply search filter
        if search_term:
            search_lower = search_term.lower()
            filtered = [p for p in filtered
                       if any(search_lower in str(cluster).lower() 
                             for cluster in p.get('sequence', []))]
        
        return filtered
    
    def calculate_coverage(self, shown_paths: List[Dict[str, Any]], 
                          all_paths: List[Dict[str, Any]]) -> float:
        """Calculate the coverage percentage of shown paths."""
        if not all_paths:
            return 0.0
        
        shown_samples = sum(p.get('frequency', 0) for p in shown_paths)
        total_samples = sum(p.get('frequency', 0) for p in all_paths)
        
        if total_samples == 0:
            return 0.0
        
        return (shown_samples / total_samples) * 100