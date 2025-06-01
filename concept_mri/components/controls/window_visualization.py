"""
Visualization utilities for Layer Window Manager.
Creates interactive plots for metrics and window visualization.
"""
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from concept_mri.config.settings import THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR

logger = logging.getLogger(__name__)


def create_metrics_plot(
    metrics: Dict[str, List[float]],
    windows: Dict[str, Dict],
    suggested_boundaries: Optional[List[int]] = None,
    num_layers: int = None,
    height: int = 300
) -> go.Figure:
    """
    Create an interactive plot showing layer metrics with window boundaries.
    
    Args:
        metrics: Dictionary of metric names to values
        windows: Current window configuration
        suggested_boundaries: Optional list of suggested boundary indices
        num_layers: Total number of layers
        height: Plot height in pixels
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Determine x-axis values
    if num_layers and metrics:
        # For metrics, x values are between layers (transitions)
        first_metric = next(iter(metrics.values()))
        if first_metric and len(first_metric) == num_layers - 1:
            x_values = list(range(num_layers - 1))
            x_axis_title = "Layer Transition"
        else:
            x_values = list(range(len(first_metric) if first_metric else 0))
            x_axis_title = "Layer Index"
    else:
        x_values = []
        x_axis_title = "Layer"
    
    # Plot metrics
    colors = [THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR, '#2ca02c']
    for i, (name, values) in enumerate(metrics.items()):
        if values is None:
            continue
            
        fig.add_trace(go.Scatter(
            x=x_values,
            y=values,
            mode='lines',
            name=name.capitalize(),
            line=dict(
                color=colors[i % len(colors)],
                width=2
            ),
            hovertemplate='%{y:.3f}<extra></extra>'
        ))
    
    # Add window boundaries as vertical lines
    if windows:
        for window_name, window_config in windows.items():
            # Add start boundary
            if window_config['start'] > 0:
                fig.add_vline(
                    x=window_config['start'] - 0.5,
                    line_dash="solid",
                    line_color=window_config.get('color', 'gray'),
                    annotation_text=f"{window_name} start",
                    annotation_position="top"
                )
    
    # Add suggested boundaries (if in experimental mode)
    if suggested_boundaries:
        for boundary in suggested_boundaries:
            fig.add_vline(
                x=boundary,
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                annotation_text="Suggested",
                annotation_position="bottom"
            )
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            title=x_axis_title,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            range=[-0.5, (num_layers or len(x_values)) - 0.5]
        ),
        yaxis=dict(
            title="Metric Value",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        clickmode='event+select'
    )
    
    return fig


def create_window_preview(
    windows: Dict[str, Dict],
    num_layers: int,
    height: int = 150
) -> go.Figure:
    """
    Create a visual preview of window configuration.
    
    Args:
        windows: Window configuration dictionary
        num_layers: Total number of layers
        height: Plot height in pixels
        
    Returns:
        Plotly figure showing window layout
    """
    fig = go.Figure()
    
    if not windows:
        # Show empty state
        fig.add_annotation(
            x=num_layers / 2,
            y=0.5,
            text="No windows configured",
            showarrow=False,
            font=dict(size=14, color='gray')
        )
    else:
        # Create a horizontal bar for each window
        y_pos = 0
        for name, config in windows.items():
            # Create filled rectangle for window
            fig.add_trace(go.Scatter(
                x=[config['start'], config['start'], config['end'] + 0.9, config['end'] + 0.9, config['start']],
                y=[y_pos, y_pos + 0.8, y_pos + 0.8, y_pos, y_pos],
                fill='toself',
                fillcolor=config.get('color', THEME_COLOR),
                line=dict(color=config.get('color', THEME_COLOR)),
                mode='lines',
                name=name,
                hovertemplate=f"{name}<br>Layers {config['start']}-{config['end']}<extra></extra>",
                showlegend=False
            ))
            
            # Add window label
            fig.add_annotation(
                x=(config['start'] + config['end']) / 2,
                y=y_pos + 0.4,
                text=f"<b>{name}</b><br>L{config['start']}-L{config['end']}",
                showarrow=False,
                font=dict(color='white', size=11),
                bgcolor=config.get('color', THEME_COLOR),
                borderpad=4
            )
            
            y_pos += 1
    
    # Update layout
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            title="Layer Index",
            range=[-0.5, num_layers - 0.5],
            tickmode='linear',
            tick0=0,
            dtick=max(1, num_layers // 10)
        ),
        yaxis=dict(
            visible=False,
            range=[-0.5, max(len(windows), 1)]
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest'
    )
    
    return fig


def find_suggested_boundaries(
    metrics: Dict[str, List[float]],
    num_windows: int = 3,
    min_distance: int = 2
) -> List[int]:
    """
    Find suggested window boundaries based on metrics.
    
    Args:
        metrics: Dictionary of metrics
        num_windows: Desired number of windows
        min_distance: Minimum distance between boundaries
        
    Returns:
        List of suggested boundary indices
    """
    # Use stability metric if available
    if 'stability' in metrics and metrics['stability']:
        values = metrics['stability']
    else:
        # Use first available metric
        values = next((v for v in metrics.values() if v), [])
    
    if not values or len(values) < num_windows:
        return []
    
    # Find peaks (high instability = good boundaries)
    from concept_mri.components.controls.window_detection_utils import WindowDetectionMetrics
    peaks = WindowDetectionMetrics.detect_peaks(
        values,
        min_prominence=0.2,
        min_distance=min_distance
    )
    
    # If we have too many peaks, select the most prominent
    if len(peaks) > num_windows - 1:
        # Sort by peak value and take the highest
        peak_values = [(i, values[i]) for i in peaks]
        peak_values.sort(key=lambda x: x[1], reverse=True)
        peaks = [p[0] for p in peak_values[:num_windows-1]]
        peaks.sort()
    
    return peaks


def create_window_config_from_boundaries(
    boundaries: List[int],
    num_layers: int,
    colors: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Create window configuration from boundary indices.
    
    Args:
        boundaries: List of boundary indices
        num_layers: Total number of layers
        colors: Optional list of colors for windows
        
    Returns:
        Window configuration dictionary
    """
    if not colors:
        colors = [THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR, '#2ca02c', '#9467bd']
    
    # Add start and end if not present
    boundaries = sorted(set([0] + boundaries + [num_layers]))
    
    # Create windows
    windows = {}
    window_names = generate_window_names(len(boundaries) - 1)
    
    for i in range(len(boundaries) - 1):
        windows[window_names[i]] = {
            'start': boundaries[i],
            'end': boundaries[i+1] - 1,
            'color': colors[i % len(colors)]
        }
    
    return windows


def generate_window_names(num_windows: int) -> List[str]:
    """Generate appropriate names for windows based on count."""
    if num_windows == 2:
        return ['Early', 'Late']
    elif num_windows == 3:
        return ['Early', 'Middle', 'Late']
    elif num_windows == 4:
        return ['First', 'Second', 'Third', 'Fourth']
    else:
        return [f'Window {i+1}' for i in range(num_windows)]