"""
EntityCard component for displaying detailed information about selected entities.

This component dynamically displays:
- PathCard for selected paths
- ClusterCard for selected clusters
- SampleCard for selected samples
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional, Union
import plotly.graph_objects as go


class EntityCard:
    """Base class for entity cards."""
    
    def create_card(self, data: Dict[str, Any]) -> html.Div:
        """Create a generic entity card.
        
        Args:
            data: Entity data
            
        Returns:
            Card component
        """
        return dbc.Card([
            dbc.CardHeader("Entity Details"),
            dbc.CardBody("Select an entity to view details")
        ])


class ClusterCard(EntityCard):
    """Card for displaying cluster information."""
    
    def create_card(self, data: Dict[str, Any]) -> html.Div:
        """Create cluster information card.
        
        Args:
            data: Cluster data including:
                - id: Cluster identifier
                - layer: Layer name
                - label: Semantic label
                - size: Number of samples
                - features: Top features
                - metrics: Cluster metrics
                
        Returns:
            Cluster card component
        """
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-layer-group me-2"),
                f"Cluster {data.get('id', 'Unknown')} - {data.get('layer', 'Unknown Layer')}"
            ], className="fw-bold"),
            dbc.CardBody([
                # Semantic label
                html.H5(data.get('label', 'Unlabeled Cluster'), className="mb-3"),
                
                # Basic stats
                html.Div([
                    html.Div([
                        html.Span("Size: ", className="fw-bold"),
                        html.Span(f"{data.get('size', 0)} samples")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Cohesion: ", className="fw-bold"),
                        html.Span(f"{data.get('metrics', {}).get('cohesion', 0):.3f}")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Separation: ", className="fw-bold"),
                        html.Span(f"{data.get('metrics', {}).get('separation', 0):.3f}")
                    ], className="mb-2")
                ]),
                
                # Top features
                html.Hr(),
                html.H6("Top Features:", className="mb-2"),
                self._create_feature_list(data.get('features', {})),
                
                # Mini visualization
                html.Hr(),
                html.H6("Cluster Distribution:", className="mb-2"),
                self._create_cluster_viz(data)
            ])
        ])
    
    def _create_feature_list(self, features: Dict[str, float]) -> html.Div:
        """Create list of top features."""
        if not features:
            return html.P("No feature data available", className="text-muted small")
        
        # Sort features by importance
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return html.Ul([
            html.Li([
                html.Span(f"{name}: ", className="fw-bold"),
                html.Span(f"{value:.3f}")
            ], className="small")
            for name, value in sorted_features
        ], className="mb-0")
    
    def _create_cluster_viz(self, data: Dict[str, Any]) -> html.Div:
        """Create small cluster visualization."""
        # Create a simple bar chart of feature importances
        features = data.get('features', {})
        if not features:
            return html.P("No visualization available", className="text-muted small")
        
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f[1] for f in sorted_features],
                y=[f[0] for f in sorted_features],
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis_title=None,
            yaxis_title=None,
            font=dict(size=10)
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})


class PathCard(EntityCard):
    """Card for displaying path information."""
    
    def create_card(self, data: Dict[str, Any]) -> html.Div:
        """Create path information card.
        
        Args:
            data: Path data including:
                - id: Path identifier
                - sequence: Cluster sequence
                - frequency: Path frequency
                - samples: Sample indices
                - transitions: Transition details
                - stability: Path stability metric
                - pattern: Path pattern type
                
        Returns:
            Path card component
        """
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-route me-2"),
                f"Path {data.get('id', 'Unknown')}"
            ], className="fw-bold"),
            dbc.CardBody([
                # Pattern type
                html.H5(data.get('pattern', 'Unknown Pattern'), className="mb-3"),
                
                # Basic stats
                html.Div([
                    html.Div([
                        html.Span("Frequency: ", className="fw-bold"),
                        html.Span(f"{data.get('frequency', 0)} occurrences")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Samples: ", className="fw-bold"),
                        html.Span(f"{len(data.get('samples', []))}")
                    ], className="mb-1"),
                    html.Div([
                        html.Span("Stability: ", className="fw-bold"),
                        html.Span(f"{data.get('stability', 0):.3f}")
                    ], className="mb-2")
                ]),
                
                # Path sequence
                html.Hr(),
                html.H6("Cluster Sequence:", className="mb-2"),
                self._create_sequence_display(data.get('sequence', [])),
                
                # Transitions
                html.Hr(),
                html.H6("Key Transitions:", className="mb-2"),
                self._create_transition_list(data.get('transitions', []))
            ])
        ])
    
    def _create_sequence_display(self, sequence: List[str]) -> html.Div:
        """Create visual display of cluster sequence."""
        if not sequence:
            return html.P("No sequence data available", className="text-muted small")
        
        elements = []
        for i, cluster in enumerate(sequence):
            if i > 0:
                elements.append(html.I(className="fas fa-arrow-right mx-1 text-muted"))
            elements.append(html.Span(cluster, className="badge bg-primary"))
        
        return html.Div(elements, className="d-flex align-items-center flex-wrap")
    
    def _create_transition_list(self, transitions: List[Dict[str, Any]]) -> html.Div:
        """Create list of key transitions."""
        if not transitions:
            return html.P("No transition data available", className="text-muted small")
        
        return html.Ul([
            html.Li([
                html.Span(f"{t.get('from', '?')} → {t.get('to', '?')}: ", 
                         className="fw-bold"),
                html.Span(f"{t.get('weight', 0):.3f}")
            ], className="small")
            for t in transitions[:3]  # Show top 3 transitions
        ], className="mb-0")


class SampleCard(EntityCard):
    """Card for displaying sample information."""
    
    def create_card(self, data: Dict[str, Any]) -> html.Div:
        """Create sample information card.
        
        Args:
            data: Sample data including:
                - id: Sample identifier
                - text: Sample text (if available)
                - trajectory: Cluster trajectory
                - clusters: Cluster assignments
                - activations: Activation values
                - metadata: Additional metadata
                
        Returns:
            Sample card component
        """
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-file-alt me-2"),
                f"Sample {data.get('id', 'Unknown')}"
            ], className="fw-bold"),
            dbc.CardBody([
                # Sample text (if available)
                self._create_text_display(data.get('text', '')),
                
                # Trajectory
                html.H6("Cluster Trajectory:", className="mb-2 mt-3"),
                self._create_trajectory_display(data.get('trajectory', [])),
                
                # Metadata
                html.Hr(),
                html.H6("Metadata:", className="mb-2"),
                self._create_metadata_display(data.get('metadata', {})),
                
                # Mini activation heatmap
                html.Hr(),
                html.H6("Activation Pattern:", className="mb-2"),
                self._create_activation_viz(data.get('activations', {}))
            ])
        ])
    
    def _create_text_display(self, text: str) -> html.Div:
        """Create text display if available."""
        if not text:
            return html.Div()
        
        # Truncate long text
        display_text = text[:200] + "..." if len(text) > 200 else text
        
        return html.Div([
            html.H6("Content:", className="mb-2"),
            html.P(display_text, className="small font-monospace bg-light p-2"),
        ])
    
    def _create_trajectory_display(self, trajectory: List[int]) -> html.Div:
        """Create visual display of sample trajectory."""
        if not trajectory:
            return html.P("No trajectory data available", className="text-muted small")
        
        elements = []
        for i, cluster in enumerate(trajectory):
            if i > 0:
                elements.append(html.I(className="fas fa-arrow-right mx-1 text-muted small"))
            elements.append(html.Span(f"C{cluster}", className="badge bg-info"))
        
        return html.Div(elements, className="d-flex align-items-center flex-wrap")
    
    def _create_metadata_display(self, metadata: Dict[str, Any]) -> html.Div:
        """Create metadata display."""
        if not metadata:
            return html.P("No metadata available", className="text-muted small")
        
        return html.Ul([
            html.Li([
                html.Span(f"{key}: ", className="fw-bold"),
                html.Span(str(value))
            ], className="small")
            for key, value in list(metadata.items())[:5]  # Show first 5 items
        ], className="mb-0")
    
    def _create_activation_viz(self, activations: Dict[str, Any]) -> html.Div:
        """Create small activation visualization."""
        if not activations:
            return html.P("No activation data available", className="text-muted small")
        
        # Create a simple line chart of activation magnitudes
        layers = sorted(activations.keys())
        values = [activations.get(layer, {}).get('magnitude', 0) for layer in layers]
        
        if not values:
            return html.P("No activation magnitudes available", className="text-muted small")
        
        fig = go.Figure(data=[
            go.Scatter(
                x=list(range(len(layers))),
                y=values,
                mode='lines+markers',
                line=dict(color='coral', width=2),
                marker=dict(size=8)
            )
        ])
        
        fig.update_layout(
            height=120,
            margin=dict(l=20, r=20, t=10, b=20),
            showlegend=False,
            xaxis=dict(
                title=None, 
                tickmode='array', 
                tickvals=list(range(len(layers))), 
                ticktext=[f"L{i}" for i in range(len(layers))],
                tickfont=dict(size=9)
            ),
            yaxis=dict(title=None, tickfont=dict(size=9))
        )
        
        return dcc.Graph(figure=fig, config={'displayModeBar': False})