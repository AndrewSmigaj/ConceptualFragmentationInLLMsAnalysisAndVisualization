"""
Callbacks for NetworkOverview component.

Handles:
- Window selection updates
- Metric selection updates
- Chart updates based on clustering results
"""

from dash import callback, Input, Output, State, ALL, ctx
from typing import Dict, List, Any, Optional, Tuple
import plotly.graph_objects as go
import numpy as np

from ..network_overview import NetworkOverview
from ..window_selector import WindowSelector
from ..stores_config import get_store_id

# Create instances for helper methods
network_overview = NetworkOverview()
window_selector = WindowSelector()

def register_network_overview_callbacks(app):
    """Register callbacks for the NetworkOverview component."""
    
    @app.callback(
        [Output("network-overview-early-btn", "active"),
         Output("network-overview-middle-btn", "active"),
         Output("network-overview-late-btn", "active"),
         Output("network-overview-custom-btn", "active"),
         Output(get_store_id("window_config"), "data", allow_duplicate=True)],
        [Input("network-overview-early-btn", "n_clicks"),
         Input("network-overview-middle-btn", "n_clicks"),
         Input("network-overview-late-btn", "n_clicks"),
         Input("network-overview-custom-btn", "n_clicks")],
        [State(get_store_id("clustering_results"), "data"),
         State(get_store_id("window_config"), "data")],
        prevent_initial_call=True
    )
    def handle_window_selection(early_clicks, middle_clicks, late_clicks, custom_clicks,
                               clustering_results, current_window_config):
        """Handle window selection button clicks."""
        triggered_id = ctx.triggered_id
        
        if not triggered_id:
            # Return current state
            current_window = current_window_config.get('current_window', 'early')
            return (
                current_window == 'early',
                current_window == 'middle',
                current_window == 'late',
                current_window == 'custom',
                current_window_config
            )
        
        button_id = triggered_id
        
        # Default to 12 layers if no clustering results
        n_layers = 12
        if clustering_results and 'clusters_per_layer' in clustering_results:
            n_layers = len(clustering_results['clusters_per_layer'])
        
        # Get predefined windows
        windows = window_selector.get_predefined_windows(n_layers)
        
        # Update window config based on selection
        updated_config = current_window_config.copy()
        
        if 'early' in button_id:
            updated_config['current_window'] = 'early'
            button_states = (True, False, False, False)
        elif 'middle' in button_id:
            updated_config['current_window'] = 'middle'
            button_states = (False, True, False, False)
        elif 'late' in button_id:
            updated_config['current_window'] = 'late'
            button_states = (False, False, True, False)
        elif 'custom' in button_id:
            updated_config['current_window'] = 'custom'
            button_states = (False, False, False, True)
        else:
            # Default to early
            updated_config['current_window'] = 'early'
            button_states = (True, False, False, False)
        
        # Update the windows configuration if needed
        if 'windows' not in updated_config:
            updated_config['windows'] = windows
        
        return *button_states, updated_config
    
    @app.callback(
        Output("network-overview-metrics-chart", "figure"),
        [Input(get_store_id("clustering_results"), "data"),
         Input("network-overview-metric-selector", "value"),
         Input(get_store_id("window_config"), "data")]
    )
    def update_metrics_chart(clustering_results, selected_metrics, window_config):
        """Update the metrics chart based on selected metrics and window."""
        if not clustering_results or not clustering_results.get('completed'):
            return network_overview.metrics_chart.create_empty_figure()
        
        # Compute metrics from clustering results
        metrics_data = _compute_metrics_from_clustering(clustering_results)
        
        # Extract layer names and metric values
        layer_names = metrics_data.get('layer_names', [])
        metric_values = metrics_data.get('metrics', {})
        
        # Filter to only selected metrics
        filtered_metrics = {
            metric: values for metric, values in metric_values.items()
            if metric in selected_metrics
        }
        
        if not filtered_metrics or not layer_names:
            return network_overview.metrics_chart.create_empty_figure()
        
        # Apply window filtering if needed
        window_type = window_config.get('current_window', 'full')
        if window_type != 'full' and window_type in window_config.get('windows', {}):
            window = window_config['windows'][window_type]
            start_idx = window['start']
            end_idx = window['end'] + 1
            
            # Filter layer names and metrics to window
            layer_names = layer_names[start_idx:end_idx]
            filtered_metrics = {
                metric: values[start_idx:end_idx]
                for metric, values in filtered_metrics.items()
            }
        
        # Create the figure
        return network_overview.metrics_chart.create_figure(
            metrics_data=filtered_metrics,
            layer_names=layer_names,
            selected_metrics=selected_metrics,
            window_selection=window_config
        )
    
    @app.callback(
        Output(get_store_id("selection"), "data", allow_duplicate=True),
        Input("network-overview-metrics-chart", "clickData"),
        State(get_store_id("clustering_results"), "data"),
        prevent_initial_call=True
    )
    def handle_metrics_chart_click(click_data, clustering_results):
        """Handle clicks on the metrics chart to select layers."""
        if not click_data or not clustering_results:
            return {"entity_type": None, "entity_id": None, "entity_data": None, "source_component": None}
        
        # Extract layer information from click
        points = click_data.get('points', [])
        if not points:
            return {"entity_type": None, "entity_id": None, "entity_data": None, "source_component": None}
        
        point = points[0]
        layer_idx = point.get('x', 0)
        layer_names = sorted(clustering_results.get('clusters_per_layer', {}).keys())
        
        if layer_idx < len(layer_names):
            layer_name = layer_names[layer_idx]
            layer_data = clustering_results['clusters_per_layer'].get(layer_name, {})
            
            return {
                "entity_type": "layer",
                "entity_id": layer_name,
                "entity_data": {
                    "id": layer_name,
                    "index": layer_idx,
                    "n_clusters": layer_data.get('n_clusters', 0),
                    "metrics": {
                        "fragmentation": point.get('y', 0),
                        "metric_type": point.get('curveNumber', 0)
                    }
                },
                "source_component": "network_overview"
            }
        
        return {"entity_type": None, "entity_id": None, "entity_data": None, "source_component": None}


def _compute_metrics_from_clustering(clustering_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metrics from clustering results."""
    # Extract layer information
    clusters_per_layer = clustering_results.get('clusters_per_layer', {})
    if not clusters_per_layer:
        return {}
        
        # Extract layer information
        clusters_per_layer = clustering_results.get('clusters_per_layer', {})
        if not clusters_per_layer:
            return {}
        
        layer_names = sorted(clusters_per_layer.keys())
        n_layers = len(layer_names)
        
        # Initialize metrics
        metrics = {
            'fragmentation': [],
            'cohesion': [],
            'entropy': [],
            'path_diversity': [],
            'stability': []
        }
        
        # Compute simple placeholder metrics (in real implementation, these would be actual calculations)
        for i, layer in enumerate(layer_names):
            layer_data = clusters_per_layer[layer]
            n_clusters = layer_data.get('n_clusters', 1)
            
            # Simple heuristic metrics
            metrics['fragmentation'].append(min(n_clusters / 10, 1.0))
            metrics['cohesion'].append(1.0 - (n_clusters / 20))
            metrics['entropy'].append(0.5 + 0.3 * (i / n_layers))
            metrics['path_diversity'].append(0.3 + 0.4 * (n_clusters / 15))
            metrics['stability'].append(0.8 - 0.3 * abs(i - n_layers/2) / (n_layers/2))
        
        # Add real metrics if available
        if 'metrics' in clustering_results:
            real_metrics = clustering_results['metrics']
            # Override with real values if they exist
            for metric_name, metric_values in real_metrics.items():
                if metric_name in metrics and isinstance(metric_values, list):
                    metrics[metric_name] = metric_values[:n_layers]
        
        return {
            'layer_names': layer_names,
            'metrics': metrics
        }