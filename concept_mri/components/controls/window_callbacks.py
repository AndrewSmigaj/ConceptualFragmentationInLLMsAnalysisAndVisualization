"""
Callbacks for Layer Window Manager component.
"""
from dash import Input, Output, State, callback, callback_context, no_update, ALL, MATCH, html, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import json
from typing import Dict, Any, List, Optional
import logging

from .layer_window_manager import (
    compute_preset_windows, 
    create_window_summary_cards,
    WINDOW_COLORS
)
from .window_visualization import (
    create_metrics_plot,
    create_window_preview,
    find_suggested_boundaries,
    create_window_config_from_boundaries
)
from .window_detection_utils import WindowDetectionMetrics

logger = logging.getLogger(__name__)


def register_window_callbacks(app):
    """Register all callbacks for the layer window manager."""
    
    @app.callback(
        [Output('window-alert-no-model', 'is_open'),
         Output('window-manager-content', 'style'),
         Output('layer-count-badge', 'children')],
        Input('model-store', 'data')
    )
    def update_window_manager_visibility(model_data):
        """Show/hide window manager based on model availability."""
        if not model_data or not model_data.get('num_layers'):
            return True, {'display': 'none'}, ""
        
        num_layers = model_data['num_layers']
        
        # Disable for single-layer networks
        if num_layers <= 1:
            badge = dbc.Badge("Single layer - windowing disabled", color="secondary")
            return True, {'display': 'none'}, badge
        
        badge = dbc.Badge(f"{num_layers} layers", color="primary")
        return False, {'display': 'block'}, badge
    
    @app.callback(
        [Output('manual-mode-content', 'style'),
         Output('auto-mode-content', 'style')],
        Input('window-mode', 'value')
    )
    def toggle_mode_content(mode):
        """Show/hide content based on selected mode."""
        if mode == 'manual':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
    @app.callback(
        Output('window-metrics-store', 'data'),
        [Input('model-store', 'data')],
        State('window-metrics-store', 'data')
    )
    def compute_metrics_on_model_load(model_data, current_metrics):
        """Compute window detection metrics when model loads."""
        if not model_data or not model_data.get('activations'):
            return current_metrics or {}
        
        # Check if we already computed metrics for this model
        model_id = model_data.get('model_id', '')
        if current_metrics and current_metrics.get('model_id') == model_id:
            return current_metrics
        
        try:
            # Get activations and optional clusters
            activations = model_data.get('activations', {})
            clusters = model_data.get('clusters', None)
            
            # Compute metrics using existing code
            metrics_calc = WindowDetectionMetrics()
            raw_metrics = metrics_calc.compute_boundary_metrics(
                activations_dict=activations,
                clusters_dict=clusters,
                compute_all=False  # Just stability for now
            )
            
            # Normalize for display
            display_metrics = metrics_calc.normalize_metrics_for_display(raw_metrics)
            
            # Aggregate if deep network
            num_layers = model_data.get('num_layers', 0)
            if num_layers > 50:
                display_metrics = metrics_calc.aggregate_metrics_for_deep_networks(
                    display_metrics,
                    target_points=50
                )
            
            return {
                'model_id': model_id,
                'raw': raw_metrics,
                'display': display_metrics,
                'num_layers': num_layers
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return {}
    
    @app.callback(
        Output('metrics-plot', 'children'),
        [Input('window-metrics-store', 'data'),
         Input('window-config-store', 'data')]
    )
    def update_metrics_plot(metrics_data, window_config):
        """Update the metrics visualization plot."""
        if not metrics_data or not metrics_data.get('display'):
            return html.P("No metrics available. Ensure model analysis is complete.", 
                         className="text-muted text-center")
        
        display_metrics = metrics_data['display']
        num_layers = metrics_data.get('num_layers', 0)
        windows = window_config.get('windows', {}) if window_config else {}
        
        # Create the plot
        fig = create_metrics_plot(
            metrics=display_metrics,
            windows=windows,
            num_layers=num_layers
        )
        
        return dcc.Graph(
            figure=fig,
            id='metrics-plot-graph',
            config={'displayModeBar': False},
            style={'width': '100%'}
        )
    
    @app.callback(
        [Output('window-config-store', 'data'),
         Output('windows-list', 'children')],
        [Input('apply-preset-btn', 'n_clicks'),
         Input('add-custom-window-btn', 'n_clicks'),
         Input('detect-boundaries-btn', 'n_clicks'),
         Input({'type': 'delete-window', 'name': ALL}, 'n_clicks')],
        [State('window-presets', 'value'),
         State('model-store', 'data'),
         State('custom-window-name', 'value'),
         State('custom-window-start', 'value'),
         State('custom-window-end', 'value'),
         State('custom-window-color', 'value'),
         State('window-config-store', 'data'),
         State('window-metrics-store', 'data')]
    )
    def handle_window_updates(preset_clicks, add_clicks, detect_clicks, delete_clicks,
                            preset_value, model_data, custom_name, custom_start, 
                            custom_end, custom_color, current_config, metrics_data):
        """Handle all window configuration updates."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Initialize config
        config = current_config or {'windows': {}}
        num_layers = model_data.get('num_layers', 0) if model_data else 0
        
        # Handle preset application
        if trigger_id == 'apply-preset-btn' and preset_value and preset_value != 'custom':
            windows = compute_preset_windows(preset_value, num_layers)
            config['windows'] = windows
            config['mode'] = 'preset'
            config['preset'] = preset_value
        
        # Handle custom window addition
        elif trigger_id == 'add-custom-window-btn':
            if custom_name and custom_start is not None and custom_end is not None:
                start = int(custom_start)
                end = int(custom_end)
                
                # Validate
                if 0 <= start <= end < num_layers:
                    windows = config.get('windows', {})
                    
                    # Use next available color if not specified
                    if not custom_color:
                        used_colors = [w.get('color') for w in windows.values()]
                        available_colors = [c for c in WINDOW_COLORS if c not in used_colors]
                        custom_color = available_colors[0] if available_colors else WINDOW_COLORS[0]
                    
                    windows[custom_name] = {
                        'start': start,
                        'end': end,
                        'color': custom_color
                    }
                    config['windows'] = windows
                    config['mode'] = 'custom'
        
        # Handle auto-detection
        elif trigger_id == 'detect-boundaries-btn':
            if metrics_data and metrics_data.get('display'):
                # Find boundaries
                boundaries = find_suggested_boundaries(
                    metrics_data['display'],
                    num_windows=3
                )
                
                # Create windows from boundaries
                if boundaries:
                    windows = create_window_config_from_boundaries(
                        boundaries,
                        num_layers
                    )
                    config['windows'] = windows
                    config['mode'] = 'auto'
                    config['boundaries'] = boundaries
        
        # Handle window deletion
        elif 'delete-window' in trigger_id:
            # Parse the window name from the trigger
            trigger_dict = json.loads(trigger_id)
            window_name = trigger_dict.get('name')
            
            if window_name and window_name in config.get('windows', {}):
                del config['windows'][window_name]
        
        # Create window summary cards
        window_cards = create_window_summary_cards(config.get('windows', {}))
        
        return config, window_cards
    
    @app.callback(
        Output('metrics-plot-graph', 'figure', allow_duplicate=True),
        Input('metrics-plot-graph', 'clickData'),
        [State('metrics-plot-graph', 'figure'),
         State('window-config-store', 'data'),
         State('model-store', 'data')],
        prevent_initial_call=True
    )
    def handle_plot_click(click_data, current_figure, window_config, model_data):
        """Handle clicks on the metrics plot to add boundaries."""
        if not click_data or not current_figure:
            raise PreventUpdate
        
        # Get clicked x position
        x_pos = click_data['points'][0]['x']
        
        # Round to nearest integer (layer boundary)
        boundary = round(x_pos)
        
        # Add a visual indicator for the new boundary
        fig = go.Figure(current_figure)
        
        # Add temporary boundary line
        fig.add_vline(
            x=boundary,
            line_dash="dot",
            line_color="orange",
            opacity=0.7,
            annotation_text="Click to confirm",
            annotation_position="top"
        )
        
        return fig
    
    @app.callback(
        [Output('custom-window-start', 'max'),
         Output('custom-window-end', 'max')],
        Input('model-store', 'data')
    )
    def update_custom_window_bounds(model_data):
        """Update the bounds for custom window inputs based on model."""
        if not model_data:
            return 0, 0
        
        num_layers = model_data.get('num_layers', 0)
        return num_layers - 1, num_layers - 1