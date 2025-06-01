"""
Enhanced Layer Window Manager with automatic boundary detection and metric visualization.

This component provides advanced layer windowing capabilities including:
- Automatic boundary detection using multiple metrics
- Interactive metric visualization for manual window placement
- Scalable UI for deep networks (50-100+ layers)
- Progressive disclosure of complexity
"""

from dash import html, dcc, Input, Output, State, callback, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from dataclasses import dataclass, asdict
from enum import Enum
import colorsys

from concept_mri.config.settings import THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR


class BoundaryDetectionMethod(Enum):
    """Available methods for automatic boundary detection."""
    STABILITY_GRADIENT = "stability_gradient"
    CLUSTERING_TRANSITION = "clustering_transition"
    ATTENTION_SHIFT = "attention_shift"
    REPRESENTATION_JUMP = "representation_jump"
    ENSEMBLE = "ensemble"


class MetricType(Enum):
    """Types of metrics to visualize."""
    STABILITY = "stability"
    ENTROPY = "entropy"
    CLUSTER_COHERENCE = "cluster_coherence"
    ATTENTION_FOCUS = "attention_focus"
    ACTIVATION_MAGNITUDE = "activation_magnitude"
    GRADIENT_FLOW = "gradient_flow"


@dataclass
class LayerMetrics:
    """Container for layer-wise metrics."""
    layer_index: int
    layer_name: str
    stability: float
    entropy: float
    cluster_coherence: float
    attention_focus: Optional[float] = None
    activation_magnitude: float = 0.0
    gradient_flow: Optional[float] = None


@dataclass
class SuggestedBoundary:
    """Container for a suggested boundary with reasoning."""
    layer_index: int
    confidence: float
    method: BoundaryDetectionMethod
    reasoning: str
    supporting_metrics: Dict[str, float]


class EnhancedLayerWindowManager:
    """Enhanced layer window manager with automatic boundary detection."""
    
    def __init__(self):
        """Initialize the enhanced layer window manager."""
        self.id_prefix = "enhanced-layer-window"
        self.metric_colors = {
            MetricType.STABILITY: "#00A6A6",  # Teal
            MetricType.ENTROPY: "#0066CC",     # Blue
            MetricType.CLUSTER_COHERENCE: "#2ca02c",  # Green
            MetricType.ATTENTION_FOCUS: "#FF6B35",    # Orange
            MetricType.ACTIVATION_MAGNITUDE: "#9467bd",  # Purple
            MetricType.GRADIENT_FLOW: "#8c564b"  # Brown
        }
    
    def create_component(self) -> dbc.Card:
        """Create and return the enhanced component layout."""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-layer-group me-2"),
                "Enhanced Layer Window Configuration",
                dbc.Badge(
                    "Experimental",
                    color="warning",
                    className="ms-2",
                    pill=True
                ),
                html.Div(
                    id=f"{self.id_prefix}-layer-count",
                    className="ms-auto"
                )
            ], className="fw-bold d-flex align-items-center"),
            
            dbc.CardBody([
                # Stores
                dcc.Store(id=f"{self.id_prefix}-metrics-store", data={}),
                dcc.Store(id=f"{self.id_prefix}-boundaries-store", data={}),
                dcc.Store(id=f"{self.id_prefix}-windows-store", data={}),
                
                # Main content tabs
                dbc.Tabs([
                    dbc.Tab(label="Quick Setup", tab_id="quick"),
                    dbc.Tab(label="Metric Explorer", tab_id="metrics"),
                    dbc.Tab(label="Window Builder", tab_id="builder"),
                    dbc.Tab(label="Analysis", tab_id="analysis")
                ], id=f"{self.id_prefix}-tabs", active_tab="quick"),
                
                html.Div(id=f"{self.id_prefix}-tab-content", className="mt-3")
            ])
        ])
    
    def create_quick_setup_tab(self) -> html.Div:
        """Create the quick setup tab for fast configuration."""
        return html.Div([
            # Automatic detection section
            dbc.Card([
                dbc.CardHeader("Automatic Boundary Detection", className="fw-bold"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Detection Method"),
                            dbc.Select(
                                id=f"{self.id_prefix}-detection-method",
                                options=[
                                    {"label": "Ensemble (Recommended)", "value": "ensemble"},
                                    {"label": "Stability Gradient", "value": "stability_gradient"},
                                    {"label": "Clustering Transition", "value": "clustering_transition"},
                                    {"label": "Attention Pattern Shift", "value": "attention_shift"},
                                    {"label": "Representation Jump", "value": "representation_jump"}
                                ],
                                value="ensemble"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Sensitivity"),
                            dbc.Slider(
                                id=f"{self.id_prefix}-sensitivity",
                                min=0.1,
                                max=1.0,
                                step=0.1,
                                value=0.5,
                                marks={
                                    0.1: "Low",
                                    0.5: "Medium",
                                    1.0: "High"
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ], className="mb-3"),
                    
                    dbc.Button(
                        [
                            html.I(className="fas fa-magic me-2"),
                            "Detect Boundaries"
                        ],
                        id=f"{self.id_prefix}-detect-btn",
                        color="primary",
                        className="w-100"
                    )
                ])
            ], className="mb-3"),
            
            # Suggested boundaries
            html.Div(id=f"{self.id_prefix}-suggested-boundaries"),
            
            # Quick presets
            dbc.Card([
                dbc.CardHeader("Quick Presets", className="fw-bold"),
                dbc.CardBody([
                    dbc.RadioItems(
                        id=f"{self.id_prefix}-quick-preset",
                        options=[
                            {"label": "Use Suggested Boundaries", "value": "suggested"},
                            {"label": "Equal Segments", "value": "equal"},
                            {"label": "Functional Groups", "value": "functional"},
                            {"label": "Custom", "value": "custom"}
                        ],
                        value="suggested",
                        className="mb-3"
                    ),
                    
                    # Number of windows for equal segments
                    html.Div(
                        id=f"{self.id_prefix}-equal-config",
                        style={"display": "none"},
                        children=[
                            dbc.Label("Number of Windows"),
                            dbc.Input(
                                id=f"{self.id_prefix}-num-windows",
                                type="number",
                                min=2,
                                max=10,
                                value=3
                            )
                        ]
                    )
                ])
            ], className="mb-3"),
            
            # Window preview
            html.Div(id=f"{self.id_prefix}-quick-preview")
        ])
    
    def create_metric_explorer_tab(self) -> html.Div:
        """Create the metric explorer tab for interactive analysis."""
        return html.Div([
            # Metric selection
            dbc.Card([
                dbc.CardHeader("Select Metrics to Display", className="fw-bold"),
                dbc.CardBody([
                    dbc.Checklist(
                        id=f"{self.id_prefix}-metric-selection",
                        options=[
                            {"label": "Representation Stability", "value": "stability"},
                            {"label": "Activation Entropy", "value": "entropy"},
                            {"label": "Cluster Coherence", "value": "cluster_coherence"},
                            {"label": "Attention Focus (Transformers)", "value": "attention_focus"},
                            {"label": "Activation Magnitude", "value": "activation_magnitude"},
                            {"label": "Gradient Flow", "value": "gradient_flow"}
                        ],
                        value=["stability", "entropy", "cluster_coherence"],
                        inline=True
                    )
                ])
            ], className="mb-3"),
            
            # Interactive metric plot
            dcc.Graph(
                id=f"{self.id_prefix}-metric-plot",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawrect', 'eraseshape']
                },
                style={'height': '400px'}
            ),
            
            # Instructions
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Click and drag on the plot to create window boundaries. ",
                "Double-click to remove the last boundary."
            ], color="info", className="mt-2"),
            
            # Metric insights
            html.Div(id=f"{self.id_prefix}-metric-insights"),
            
            # Selected windows from plot
            html.Div(id=f"{self.id_prefix}-plot-windows")
        ])
    
    def create_window_builder_tab(self) -> html.Div:
        """Create the window builder tab for manual configuration."""
        return html.Div([
            # Current windows
            dbc.Card([
                dbc.CardHeader([
                    "Current Windows",
                    dbc.Button(
                        html.I(className="fas fa-plus"),
                        id=f"{self.id_prefix}-add-window-btn",
                        color="success",
                        size="sm",
                        className="ms-auto"
                    )
                ], className="d-flex align-items-center"),
                dbc.CardBody([
                    html.Div(id=f"{self.id_prefix}-window-list")
                ])
            ], className="mb-3"),
            
            # Window visualization
            dbc.Card([
                dbc.CardHeader("Window Layout Visualization", className="fw-bold"),
                dbc.CardBody([
                    dcc.Graph(
                        id=f"{self.id_prefix}-window-viz",
                        config={'displayModeBar': False},
                        style={'height': '200px'}
                    )
                ])
            ], className="mb-3"),
            
            # Validation warnings
            html.Div(id=f"{self.id_prefix}-validation-warnings")
        ])
    
    def create_analysis_tab(self) -> html.Div:
        """Create the analysis tab showing window quality metrics."""
        return html.Div([
            # Window quality analysis
            dbc.Card([
                dbc.CardHeader("Window Quality Analysis", className="fw-bold"),
                dbc.CardBody([
                    html.Div(id=f"{self.id_prefix}-quality-analysis")
                ])
            ], className="mb-3"),
            
            # Comparative metrics
            dbc.Card([
                dbc.CardHeader("Comparative Metrics", className="fw-bold"),
                dbc.CardBody([
                    dcc.Graph(
                        id=f"{self.id_prefix}-comparative-plot",
                        config={'displayModeBar': False},
                        style={'height': '300px'}
                    )
                ])
            ], className="mb-3"),
            
            # Recommendations
            dbc.Card([
                dbc.CardHeader("Recommendations", className="fw-bold"),
                dbc.CardBody([
                    html.Div(id=f"{self.id_prefix}-recommendations")
                ])
            ])
        ])
    
    def create_suggested_boundary_card(
        self, 
        boundary: SuggestedBoundary,
        layer_names: List[str]
    ) -> dbc.Card:
        """Create a card for a suggested boundary."""
        confidence_color = "success" if boundary.confidence > 0.8 else "warning" if boundary.confidence > 0.6 else "danger"
        
        return dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6(
                            f"Boundary after {layer_names[boundary.layer_index]}",
                            className="mb-1"
                        ),
                        html.Small(
                            boundary.reasoning,
                            className="text-muted"
                        )
                    ], width=8),
                    dbc.Col([
                        dbc.Progress(
                            value=boundary.confidence * 100,
                            label=f"{boundary.confidence:.0%}",
                            color=confidence_color,
                            className="mb-1"
                        ),
                        html.Small(
                            boundary.method.value.replace("_", " ").title(),
                            className="text-muted"
                        )
                    ], width=4)
                ]),
                
                # Supporting metrics
                html.Details([
                    html.Summary("Supporting Metrics", className="text-muted small mt-2"),
                    html.Ul([
                        html.Li(f"{k}: {v:.3f}")
                        for k, v in boundary.supporting_metrics.items()
                    ], className="small mb-0")
                ])
            ])
        ], className="mb-2")
    
    def create_metric_plot(
        self,
        layer_metrics: List[LayerMetrics],
        selected_metrics: List[str],
        suggested_boundaries: List[SuggestedBoundary]
    ) -> go.Figure:
        """Create an interactive metric plot."""
        if not layer_metrics:
            return go.Figure()
        
        # Create subplots for each selected metric
        fig = make_subplots(
            rows=len(selected_metrics),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[m.replace("_", " ").title() for m in selected_metrics]
        )
        
        # Layer indices
        layer_indices = [m.layer_index for m in layer_metrics]
        
        # Plot each metric
        for i, metric_name in enumerate(selected_metrics, 1):
            metric_type = MetricType(metric_name)
            color = self.metric_colors.get(metric_type, "#000000")
            
            # Get metric values
            values = [getattr(m, metric_name, 0) for m in layer_metrics]
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=values,
                    mode='lines+markers',
                    name=metric_name.replace("_", " ").title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    showlegend=False
                ),
                row=i,
                col=1
            )
            
            # Add suggested boundaries
            for boundary in suggested_boundaries:
                if boundary.layer_index < len(layer_indices):
                    fig.add_vline(
                        x=boundary.layer_index + 0.5,
                        line_dash="dash",
                        line_color="gray",
                        opacity=boundary.confidence * 0.5,
                        row=i,
                        col=1
                    )
        
        # Update layout
        fig.update_xaxes(title_text="Layer Index", row=len(selected_metrics), col=1)
        fig.update_layout(
            height=150 * len(selected_metrics),
            margin=dict(l=60, r=20, t=40, b=40),
            hovermode='x unified',
            dragmode='drawrect'
        )
        
        # Configure drawing tools
        fig.update_layout(
            newshape=dict(
                line_color="rgba(0, 166, 166, 0.5)",
                fillcolor="rgba(0, 166, 166, 0.1)"
            )
        )
        
        return fig
    
    def create_window_visualization(
        self,
        windows: Dict[str, Dict],
        total_layers: int,
        layer_names: List[str]
    ) -> go.Figure:
        """Create an enhanced window visualization."""
        if not windows or total_layers == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        # Create a horizontal bar chart representation
        window_height = 0.8
        y_pos = 0
        
        for window_name, window_info in windows.items():
            start = window_info['start']
            end = window_info['end']
            color = window_info.get('color', THEME_COLOR)
            
            # Add window bar
            fig.add_trace(go.Bar(
                x=[end - start + 1],
                y=[y_pos],
                name=window_name,
                orientation='h',
                base=start,
                marker_color=color,
                text=f"{window_name}<br>{layer_names[start]} - {layer_names[end]}",
                textposition='inside',
                hovertemplate=f"<b>{window_name}</b><br>" +
                             f"Layers: {start}-{end}<br>" +
                             f"Size: {end-start+1} layers<br>" +
                             "<extra></extra>",
                showlegend=False,
                width=window_height
            ))
            
            y_pos += 1
        
        # Add layer markers for reference
        if total_layers <= 50:  # Show individual layers for smaller networks
            for i in range(total_layers):
                fig.add_vline(
                    x=i + 0.5,
                    line_dash="dot",
                    line_color="lightgray",
                    opacity=0.5
                )
        
        # Update layout
        fig.update_layout(
            xaxis=dict(
                title="Layer Index",
                range=[-0.5, total_layers],
                tickmode='linear' if total_layers <= 20 else 'auto'
            ),
            yaxis=dict(
                visible=False,
                range=[-0.5, len(windows) - 0.5]
            ),
            height=max(150, 60 * len(windows)),
            margin=dict(l=20, r=20, t=20, b=40),
            plot_bgcolor='rgba(248, 249, 250, 0.5)',
            bargap=0.2,
            barmode='overlay'
        )
        
        return fig
    
    def calculate_window_quality_metrics(
        self,
        windows: Dict[str, Dict],
        layer_metrics: List[LayerMetrics]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate quality metrics for each window."""
        quality_metrics = {}
        
        for window_name, window_info in windows.items():
            start = window_info['start']
            end = window_info['end']
            
            # Get metrics for this window
            window_metrics = layer_metrics[start:end+1]
            
            if window_metrics:
                # Calculate average metrics
                avg_stability = np.mean([m.stability for m in window_metrics])
                avg_entropy = np.mean([m.entropy for m in window_metrics])
                avg_coherence = np.mean([m.cluster_coherence for m in window_metrics])
                
                # Calculate variance (lower is better for coherent windows)
                var_stability = np.var([m.stability for m in window_metrics])
                var_entropy = np.var([m.entropy for m in window_metrics])
                
                quality_metrics[window_name] = {
                    'avg_stability': avg_stability,
                    'avg_entropy': avg_entropy,
                    'avg_coherence': avg_coherence,
                    'stability_variance': var_stability,
                    'entropy_variance': var_entropy,
                    'size': end - start + 1
                }
        
        return quality_metrics
    
    def generate_recommendations(
        self,
        quality_metrics: Dict[str, Dict[str, float]],
        total_layers: int
    ) -> List[str]:
        """Generate recommendations based on window quality metrics."""
        recommendations = []
        
        # Check for very small windows
        small_windows = [
            name for name, metrics in quality_metrics.items()
            if metrics['size'] < max(2, total_layers // 10)
        ]
        if small_windows:
            recommendations.append(
                f"Consider merging small windows: {', '.join(small_windows)}. "
                f"They may not capture meaningful semantic boundaries."
            )
        
        # Check for high variance windows
        high_variance_windows = [
            name for name, metrics in quality_metrics.items()
            if metrics['stability_variance'] > 0.1
        ]
        if high_variance_windows:
            recommendations.append(
                f"Windows {', '.join(high_variance_windows)} show high internal variance. "
                f"Consider splitting them at transition points."
            )
        
        # Check for low coherence
        low_coherence_windows = [
            name for name, metrics in quality_metrics.items()
            if metrics['avg_coherence'] < 0.5
        ]
        if low_coherence_windows:
            recommendations.append(
                f"Windows {', '.join(low_coherence_windows)} have low cluster coherence. "
                f"This may indicate conceptual transitions within the window."
            )
        
        if not recommendations:
            recommendations.append(
                "Window configuration looks good! The boundaries appear to align well "
                "with the network's semantic structure."
            )
        
        return recommendations


# Callback functions would be implemented in the main app.py file
def register_callbacks(app, manager: EnhancedLayerWindowManager):
    """Register callbacks for the enhanced layer window manager."""
    
    @app.callback(
        Output(f"{manager.id_prefix}-tab-content", "children"),
        Input(f"{manager.id_prefix}-tabs", "active_tab")
    )
    def render_tab_content(active_tab):
        """Render content based on active tab."""
        if active_tab == "quick":
            return manager.create_quick_setup_tab()
        elif active_tab == "metrics":
            return manager.create_metric_explorer_tab()
        elif active_tab == "builder":
            return manager.create_window_builder_tab()
        elif active_tab == "analysis":
            return manager.create_analysis_tab()
        return html.Div()
    
    # Additional callbacks would handle:
    # - Automatic boundary detection
    # - Metric plot interactions
    # - Window creation/editing
    # - Quality analysis updates