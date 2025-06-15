"""
MetricsChart component for visualizing network metrics across layers.

This component creates multi-line plots showing various metrics
(fragmentation, cohesion, entropy, etc.) across network layers.
"""

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MetricsChart:
    """Component for creating network metrics visualizations."""
    
    def __init__(self):
        """Initialize the MetricsChart."""
        self.default_colors = {
            'fragmentation': '#e74c3c',
            'cohesion': '#3498db',
            'entropy': '#2ecc71',
            'path_diversity': '#f39c12',
            'stability': '#9b59b6'
        }
        
    def create_figure(self, 
                     metrics_data: Dict[str, List[float]],
                     layer_names: List[str],
                     selected_metrics: List[str],
                     window_selection: Optional[Dict[str, Any]] = None) -> go.Figure:
        """
        Create a multi-line plot of network metrics.
        
        Args:
            metrics_data: Dictionary mapping metric names to lists of values
            layer_names: List of layer names
            selected_metrics: List of metrics to display
            window_selection: Optional window selection for highlighting
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add traces for each selected metric
        for metric in selected_metrics:
            if metric in metrics_data:
                fig.add_trace(go.Scatter(
                    x=layer_names,
                    y=metrics_data[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(
                        color=self.default_colors.get(metric, '#95a5a6'),
                        width=2
                    ),
                    marker=dict(size=8),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  'Layer: %{x}<br>' +
                                  'Value: %{y:.3f}<br>' +
                                  '<extra></extra>'
                ))
        
        # Add window highlighting if selected
        if window_selection:
            start_idx = window_selection.get('start_layer', 0)
            end_idx = window_selection.get('end_layer', len(layer_names) - 1)
            
            # Add shaded region for selected window
            fig.add_vrect(
                x0=layer_names[start_idx],
                x1=layer_names[end_idx],
                fillcolor='rgba(52, 152, 219, 0.1)',
                layer='below',
                line_width=0
            )
        
        # Update layout
        fig.update_layout(
            height=200,
            margin=dict(l=50, r=20, t=30, b=40),
            xaxis=dict(
                title='Layer',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title='Value',
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)',
                range=[0, 1.05]
            ),
            hovermode='x unified',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_empty_figure(self) -> go.Figure:
        """Create an empty figure with message."""
        fig = go.Figure()
        
        fig.add_annotation(
            text="No metrics data available.<br>Run clustering to see metrics.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        
        fig.update_layout(
            height=200,
            margin=dict(l=50, r=20, t=30, b=40),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white'
        )
        
        return fig