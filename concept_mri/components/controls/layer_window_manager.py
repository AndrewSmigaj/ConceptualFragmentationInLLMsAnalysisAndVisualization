"""
Layer Window Manager component for Concept MRI.
Allows users to split neural networks into analyzable windows for APA.
"""
from dash import html, dcc, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import logging

from concept_mri.config.settings import THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR

logger = logging.getLogger(__name__)

# Standard window presets
WINDOW_PRESETS = {
    'gpt2': {
        'name': 'GPT-2 Style',
        'description': 'Standard GPT-2 windows (L0-3, L4-7, L8-11)',
        'windows': {
            'Early': {'start': 0, 'end': 3},
            'Middle': {'start': 4, 'end': 7},
            'Late': {'start': 8, 'end': 11}
        }
    },
    'thirds': {
        'name': 'Thirds',
        'description': 'Divide network into three equal parts',
        'windows': {}  # Computed dynamically
    },
    'quarters': {
        'name': 'Quarters', 
        'description': 'Divide network into four equal parts',
        'windows': {}  # Computed dynamically
    },
    'halves': {
        'name': 'Halves',
        'description': 'Split network into two halves',
        'windows': {}  # Computed dynamically
    }
}

# Theme colors for windows
WINDOW_COLORS = [THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR, '#2ca02c', '#9467bd', '#8c564b']


class LayerWindowManager:
    """Manages layer windowing for neural network analysis."""
    
    def __init__(self):
        """Initialize the layer window manager."""
        self.id_prefix = "layer-window"
    
    def create_component(self):
        """Create and return the component layout."""
        return create_layer_window_manager()


def create_layer_window_manager():
    """Create the layer window manager interface."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-layer-group me-2"),
            "Layer Window Configuration",
            html.Span(id="layer-count-badge", className="ms-auto")
        ], className="fw-bold d-flex align-items-center"),
        
        dbc.CardBody([
            # Stores for state management
            dcc.Store(id='window-config-store', data={}),
            dcc.Store(id='window-metrics-store', data={}),
            
            # Alert when no model is loaded
            dbc.Alert(
                "Please upload a model to configure layer windows.",
                id="window-alert-no-model",
                color="info",
                is_open=True,
                dismissable=False
            ),
            
            # Main content (hidden until model loads)
            html.Div(id="window-manager-content", style={'display': 'none'}, children=[
                # Mode selector
                dbc.RadioItems(
                    id="window-mode",
                    options=[
                        {"label": "Manual Configuration", "value": "manual"},
                        {"label": html.Span([
                            "Auto-detect Boundaries ",
                            html.Span("(Experimental) 🧪", className="text-warning")
                        ]), "value": "auto"}
                    ],
                    value="manual",
                    className="mb-3"
                ),
                
                # Manual mode content
                html.Div(id="manual-mode-content", children=[
                    # Preset selector
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Window Presets:"),
                            dbc.Select(
                                id="window-presets",
                                options=[
                                    {"label": "Custom", "value": "custom"},
                                    {"label": "GPT-2 Style (L0-3, L4-7, L8-11)", "value": "gpt2"},
                                    {"label": "Thirds", "value": "thirds"},
                                    {"label": "Quarters", "value": "quarters"},
                                    {"label": "Halves", "value": "halves"}
                                ],
                                value="custom"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Br(),
                            dbc.Button(
                                "Apply Preset",
                                id="apply-preset-btn",
                                color="primary",
                                className="mt-1"
                            )
                        ], width=6)
                    ], className="mb-3")
                ]),
                
                # Auto mode content (hidden by default)
                html.Div(id="auto-mode-content", style={'display': 'none'}, children=[
                    dbc.Alert(
                        [
                            html.I(className="fas fa-flask me-2"),
                            "Experimental Feature: Auto-detection uses metrics to suggest window boundaries."
                        ],
                        color="warning",
                        className="mb-3"
                    ),
                    dbc.Button(
                        "Detect Boundaries",
                        id="detect-boundaries-btn",
                        color="primary",
                        className="mb-3"
                    )
                ]),
                
                # Metrics visualization
                dbc.Card([
                    dbc.CardHeader("Layer Metrics", className="py-2"),
                    dbc.CardBody([
                        html.Div(id="metrics-plot", children=[
                            html.P("Metrics will appear here after model analysis.", 
                                   className="text-muted text-center")
                        ]),
                        html.Small(
                            "📍 Click on the plot to add window boundaries",
                            className="text-muted"
                        )
                    ])
                ], color="light", className="mb-3"),
                
                # Current windows display
                html.Div(id="current-windows", children=[
                    html.H6("Current Windows"),
                    html.Div(id="windows-list", children=[
                        html.P("No windows configured yet.", className="text-muted")
                    ])
                ]),
                
                # Custom window builder
                html.Details([
                    html.Summary("Add Custom Window", className="fw-bold mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id="custom-window-name",
                                placeholder="Window name",
                                size="sm"
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Input(
                                id="custom-window-start",
                                type="number",
                                placeholder="Start",
                                min=0,
                                size="sm"
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Input(
                                id="custom-window-end",
                                type="number",
                                placeholder="End",
                                min=0,
                                size="sm"
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Input(
                                id="custom-window-color",
                                type="color",
                                value=THEME_COLOR,
                                style={"width": "40px", "height": "31px"}
                            )
                        ], width=2),
                        dbc.Col([
                            dbc.Button(
                                "Add",
                                id="add-custom-window-btn",
                                color="success",
                                size="sm"
                            )
                        ], width=3)
                    ], className="mt-2")
                ], className="mt-3")
            ])
        ])
    ], className="mb-3")


def compute_preset_windows(preset: str, num_layers: int) -> Dict[str, Dict]:
    """
    Compute window configuration for a given preset and layer count.
    
    Args:
        preset: Preset name from WINDOW_PRESETS
        num_layers: Total number of layers in the model
        
    Returns:
        Dictionary of window configurations
    """
    if preset not in WINDOW_PRESETS:
        return {}
    
    preset_config = WINDOW_PRESETS[preset]
    
    # For GPT-2, use fixed windows but adjust if model is smaller
    if preset == 'gpt2':
        windows = {}
        for name, config in preset_config['windows'].items():
            start = config['start']
            end = min(config['end'], num_layers - 1)
            if start < num_layers:
                windows[name] = {
                    'start': start,
                    'end': end,
                    'color': WINDOW_COLORS[len(windows) % len(WINDOW_COLORS)]
                }
        return windows
    
    # For dynamic presets
    elif preset == 'thirds':
        if num_layers < 3:
            return {'All': {'start': 0, 'end': num_layers - 1, 'color': WINDOW_COLORS[0]}}
        
        size = num_layers / 3
        return {
            'Early': {
                'start': 0,
                'end': int(size) - 1,
                'color': WINDOW_COLORS[0]
            },
            'Middle': {
                'start': int(size),
                'end': int(2 * size) - 1,
                'color': WINDOW_COLORS[1]
            },
            'Late': {
                'start': int(2 * size),
                'end': num_layers - 1,
                'color': WINDOW_COLORS[2]
            }
        }
    
    elif preset == 'quarters':
        if num_layers < 4:
            return compute_preset_windows('thirds', num_layers)
        
        size = num_layers / 4
        windows = {}
        names = ['First', 'Second', 'Third', 'Fourth']
        for i, name in enumerate(names):
            start = int(i * size)
            end = int((i + 1) * size) - 1 if i < 3 else num_layers - 1
            windows[name] = {
                'start': start,
                'end': end,
                'color': WINDOW_COLORS[i]
            }
        return windows
    
    elif preset == 'halves':
        if num_layers < 2:
            return {'All': {'start': 0, 'end': num_layers - 1, 'color': WINDOW_COLORS[0]}}
        
        mid = num_layers // 2
        return {
            'First Half': {
                'start': 0,
                'end': mid - 1,
                'color': WINDOW_COLORS[0]
            },
            'Second Half': {
                'start': mid,
                'end': num_layers - 1,
                'color': WINDOW_COLORS[1]
            }
        }
    
    return {}


def create_window_summary_cards(windows: Dict[str, Dict]) -> List[dbc.Card]:
    """Create summary cards for configured windows."""
    if not windows:
        return [html.P("No windows configured.", className="text-muted")]
    
    cards = []
    for name, config in windows.items():
        card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6(name, className="mb-1"),
                        html.Small(f"Layers {config['start']}-{config['end']}")
                    ], width=8),
                    dbc.Col([
                        html.Div(
                            style={
                                'width': '30px',
                                'height': '30px',
                                'backgroundColor': config.get('color', THEME_COLOR),
                                'borderRadius': '5px',
                                'float': 'right'
                            }
                        )
                    ], width=4)
                ]),
                dbc.ButtonGroup([
                    dbc.Button("Edit", size="sm", color="link", 
                              id={'type': 'edit-window', 'name': name}),
                    dbc.Button("Delete", size="sm", color="link",
                              id={'type': 'delete-window', 'name': name})
                ], className="mt-2")
            ])
        ], className="mb-2")
        cards.append(card)
    
    return cards