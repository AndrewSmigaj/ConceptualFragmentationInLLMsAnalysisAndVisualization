"""
Callbacks for ArchetypalPathsPanel component.

Handles:
- Path list updates based on clustering results
- Path filtering (search, frequency, pattern)
- Path selection and highlighting
- Window label updates
"""

from dash import callback, Input, Output, State, ALL, ctx
from typing import Dict, List, Any, Optional, Tuple
import json

from ..archetypal_paths_panel import ArchetypalPathsPanel
from ..window_selector import WindowSelector
from ..stores_config import get_store_id

# Create instances for helper methods
paths_panel = ArchetypalPathsPanel()
window_selector = WindowSelector()

def register_archetypal_paths_callbacks(app):
    """Register callbacks for the ArchetypalPathsPanel component."""
    
    @app.callback(
        Output(get_store_id("paths_analysis"), "data"),
        Input(get_store_id("clustering_results"), "data")
    )
    def extract_paths_from_clustering(clustering_results):
        """Extract and format paths data from clustering results."""
        if not clustering_results or not clustering_results.get('completed'):
            return {'paths': [], 'patterns': {}, 'statistics': {}, 'llm_analysis': None}
        
        # Extract paths information
        paths_dict = clustering_results.get('paths', {})
        cluster_labels = clustering_results.get('cluster_labels', {})
        total_samples = clustering_results.get('total_samples', 0)
        
        if not paths_dict:
            return {'paths': [], 'patterns': {}, 'statistics': {}, 'llm_analysis': None}
        
        # Initialize pattern tracking
        patterns = {'stable': [], 'transitional': [], 'fragmented': []}
        
        # Format paths for display
        formatted_paths = []
        for path_id, path_data in paths_dict.items():
            # PathData should already be in the correct format from main callbacks
            if isinstance(path_data, dict) and 'sequence' in path_data:
                # Add labeled sequence for display
                labeled_sequence = []
                for cluster_id in path_data['sequence']:
                    label = cluster_labels.get(cluster_id, cluster_id)
                    labeled_sequence.append(f"{cluster_id}: {label}")
                
                path_data['labeled_sequence'] = labeled_sequence
                path_data['percentage'] = (path_data.get('frequency', 0) / total_samples * 100) if total_samples > 0 else 0
                
                formatted_paths.append(path_data)
                
                # Track pattern
                pattern = path_data.get('pattern', 'fragmented')
                patterns[pattern].append(path_id)
        
        # Sort by frequency
        formatted_paths.sort(key=lambda x: x.get('frequency', 0), reverse=True)
        
        # Calculate statistics
        statistics = {
            'total_paths': len(formatted_paths),
            'total_samples': total_samples,
            'pattern_distribution': {
                pattern: len(path_ids) for pattern, path_ids in patterns.items()
            },
            'avg_frequency': sum(p.get('frequency', 0) for p in formatted_paths) / len(formatted_paths) if formatted_paths else 0
        }
        
        return {
            'paths': formatted_paths,
            'patterns': patterns,
            'statistics': statistics,
            'llm_analysis': None
        }
    
    @app.callback(
        [Output("archetypal-paths-list", "children"),
         Output("archetypal-paths-coverage", "children"),
         Output("archetypal-paths-count", "children"),
         Output("archetypal-paths-window-label", "children")],
        [Input(get_store_id("paths_analysis"), "data"),
         Input(get_store_id("window_config"), "data"),
         Input("archetypal-paths-search", "value"),
         Input("archetypal-paths-frequency-filter", "value"),
         Input("archetypal-paths-pattern-filter", "value")],
        [State(get_store_id("clustering_results"), "data")]
    )
    def update_path_list(paths_data, window_config, search_term, 
                        frequency_filter, pattern_filter, clustering_results):
        """Update the path list based on filters and window selection."""
        if not paths_data or not paths_data.get('paths'):
            return (
                [paths_panel._create_loading_placeholder()],
                "0%",
                "0/0",
                "No data loaded"
            )
        
        all_paths = paths_data['paths']
        
        # Apply filters
        frequency_threshold = float(frequency_filter) if frequency_filter else 0
        filtered_paths = paths_panel.filter_paths(
            all_paths,
            search_term or "",
            frequency_threshold,
            pattern_filter
        )
        
        # Filter by window if needed
        current_window = window_config.get('current_window', 'full')
        if current_window != 'full' and current_window in window_config.get('windows', {}):
            # Filter paths based on window layers
            window = window_config['windows'][current_window]
            start_layer = window['start']
            end_layer = window['end']
            
            # Filter paths that have transitions within this window
            window_filtered_paths = []
            for path in filtered_paths:
                # Check if path has activity in the window
                # This is a simplified check - in reality would check actual transitions
                if path.get('transitions'):
                    # Check if any transition is within the window range
                    has_window_activity = any(
                        start_layer <= t.get('layer', 0) <= end_layer
                        for t in path.get('transitions', [])
                    )
                    if has_window_activity:
                        window_filtered_paths.append(path)
                else:
                    # If no transition data, include all paths for now
                    window_filtered_paths.append(path)
            
            filtered_paths = window_filtered_paths
        
        # Create path cards
        path_cards = paths_panel.create_path_list(filtered_paths)
        
        # Calculate statistics
        coverage = paths_panel.calculate_coverage(filtered_paths, all_paths)
        count_str = f"{len(filtered_paths)}/{len(all_paths)}"
        
        # Format window label
        window_label = "All Layers"
        if window_config and clustering_results:
            current_window = window_config.get('current_window', 'full')
            if current_window != 'full' and current_window in window_config.get('windows', {}):
                window = window_config['windows'][current_window]
                layer_names = list(clustering_results.get('clusters_per_layer', {}).keys())
                if layer_names:
                    window_label = window_selector.format_window_label(
                        current_window,
                        window['start'],
                        window['end'],
                        layer_names
                    )
            else:
                window_label = "All Layers"
        
        return (
            path_cards,
            f"{coverage:.1f}%",
            count_str,
            f"Window: {window_label}"
        )
    
    @app.callback(
        Output(get_store_id("selection"), "data", allow_duplicate=True),
        [Input({'type': 'path-card-wrapper', 'index': ALL}, 'n_clicks')],
        [State(get_store_id("selection"), "data"),
         State(get_store_id("paths_analysis"), "data")],
        prevent_initial_call=True
    )
    def handle_path_selection(n_clicks_list, current_selection, paths_data):
        """Handle path card clicks for selection."""
        if not ctx.triggered or not any(n_clicks_list):
            return current_selection or {"entity_type": None, "entity_id": None, "entity_data": None, "source_component": None}
        
        # Get the clicked path ID
        triggered_id = ctx.triggered_id
        if not triggered_id or 'index' not in triggered_id:
            return current_selection
        
        path_id = triggered_id['index']
        
        # Find the path data
        path_info = None
        if paths_data and 'paths' in paths_data:
            for path in paths_data['paths']:
                if path['id'] == path_id:
                    path_info = path
                    break
        
        if not path_info:
            return current_selection
        
        # Update selection using the new store format
        return {
            'entity_type': 'path',
            'entity_id': str(path_id),
            'entity_data': path_info,
            'source_component': 'archetypal_paths_panel'
        }
    
    @app.callback(
        [Output({'type': 'path-card-wrapper', 'index': ALL}, 'className')],
        [Input(get_store_id("highlight"), "data")],
        [State({'type': 'path-card-wrapper', 'index': ALL}, 'id')],
        prevent_initial_call=False
    )
    def update_path_highlighting(highlight_data, path_card_ids):
        """Update path card highlighting based on highlight store."""
        if not path_card_ids:
            return []
        
        # Default class names
        class_names = []
        
        for card_id in path_card_ids:
            path_id = str(card_id.get('index', ''))
            base_class = "path-card-wrapper"
            
            # Check if this path should be highlighted
            if highlight_data and highlight_data.get('highlight_ids'):
                if path_id in highlight_data['highlight_ids'] or f"path-{path_id}" in highlight_data['highlight_ids']:
                    class_names.append(f"{base_class} highlighted-path")
                else:
                    # Dim non-highlighted paths when something is highlighted
                    class_names.append(f"{base_class} dimmed-path")
            else:
                # No highlighting active
                class_names.append(base_class)
        
        return [class_names]