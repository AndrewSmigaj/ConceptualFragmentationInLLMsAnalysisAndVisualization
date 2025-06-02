"""
Sankey diagram wrapper component for Concept MRI.
Wraps existing SankeyGenerator from concept_fragmentation.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from concept_fragmentation.visualization.sankey import SankeyGenerator, SankeyConfig, PathInfo
from concept_mri.config.settings import DEFAULT_TOP_N_PATHS, DEFAULT_SANKEY_HEIGHT, THEME_COLOR

class SankeyWrapper:
    """
    Wrapper for SankeyGenerator providing Dash integration.
    """
    
    def __init__(self, component_id: str = "sankey-diagram"):
        self.component_id = component_id
        # Create SankeyGenerator with colored paths enabled
        self.sankey_config = SankeyConfig(
            colored_paths=True,
            show_purity=False,  # We don't have purity data in basic clustering
            top_n_paths=DEFAULT_TOP_N_PATHS,
            height=DEFAULT_SANKEY_HEIGHT,
            width=1600,
            generate_summary=False
        )
        self.sankey_generator = SankeyGenerator(config=self.sankey_config)
        self.current_data = None
        
    def create_component(self) -> dbc.Card:
        """Create the Sankey diagram component."""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-stream me-2"),
                "Concept Flow Visualization",
                dbc.Button(
                    html.I(className="fas fa-cog"),
                    id=f"{self.component_id}-settings-btn",
                    color="link",
                    size="sm",
                    className="float-end"
                )
            ], className="fw-bold"),
            dbc.CardBody([
                # Controls row
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Top N Paths", html_for=f"{self.component_id}-top-n"),
                        dbc.Input(
                            id=f"{self.component_id}-top-n",
                            type="number",
                            value=DEFAULT_TOP_N_PATHS,
                            min=5,
                            max=100,
                            step=5
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Color By", html_for=f"{self.component_id}-color-by"),
                        dbc.Select(
                            id=f"{self.component_id}-color-by",
                            options=[
                                {"label": "Cluster", "value": "cluster"},
                                {"label": "Frequency", "value": "frequency"},
                                {"label": "Stability", "value": "stability"}
                            ],
                            value="cluster"
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Window", html_for=f"{self.component_id}-window"),
                        dbc.Select(
                            id=f"{self.component_id}-window",
                            options=[
                                {"label": "Full Network", "value": "full"}
                            ],
                            value="full",
                            placeholder="Configure windows first"
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-sync me-2"), "Update"],
                            id=f"{self.component_id}-update-btn",
                            color="primary",
                            className="mt-4"
                        )
                    ], width=3)
                ], className="mb-3"),
                
                # Hierarchy level selector
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Clustering Level", html_for=f"{self.component_id}-hierarchy"),
                        dbc.RadioItems(
                            id=f"{self.component_id}-hierarchy",
                            options=[
                                {"label": "Macro", "value": "macro"},
                                {"label": "Meso", "value": "meso"}, 
                                {"label": "Micro", "value": "micro"}
                            ],
                            value="meso",
                            inline=True
                        )
                    ], width=12)
                ], className="mb-3"),
                
                # Sankey diagram container
                dcc.Loading(
                    id=f"{self.component_id}-loading",
                    type="default",
                    color=THEME_COLOR,
                    children=[
                        html.Div(
                            id=f"{self.component_id}-container",
                            style={"height": f"{DEFAULT_SANKEY_HEIGHT}px"}
                        )
                    ]
                ),
                
                # Path information panel
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Selected Path Information"),
                            html.Div(id=f"{self.component_id}-path-info")
                        ])
                    ], className="mt-3"),
                    id=f"{self.component_id}-path-collapse",
                    is_open=False
                ),
                
                # Export button
                dbc.Button(
                    [html.I(className="fas fa-download me-2"), "Export Diagram"],
                    id=f"{self.component_id}-export-btn",
                    color="secondary",
                    size="sm",
                    className="mt-3"
                )
            ])
        ])
    
    def generate_sankey(
        self,
        clustering_data: Dict[str, Any],
        path_data: Dict[str, Any],
        cluster_labels: Dict[str, str],
        window_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate window-aware and hierarchy-aware Sankey diagram.
        
        Args:
            clustering_data: Clustering results
            path_data: Path analysis results
            cluster_labels: LLM-generated cluster labels
            window_config: Window configuration from Layer Window Manager
            config: Optional configuration overrides
            
        Returns:
            Plotly figure data
        """
        if not clustering_data:
            return self._create_empty_figure()
            
        # Use paths from clustering data if path_data not provided separately
        if not path_data:
            path_data = clustering_data.get('paths', {})
        
        # Update configuration with user settings
        self.sankey_config.top_n_paths = config.get('top_n', DEFAULT_TOP_N_PATHS) if config else DEFAULT_TOP_N_PATHS
        self.sankey_config.colored_paths = config.get('color_by', 'cluster') == 'cluster' if config else True
        
        # Extract window and hierarchy settings
        window_name = config.get('window', 'full') if config else 'full'
        hierarchy_level = config.get('hierarchy', 'meso') if config else 'meso'
        
        # Filter paths based on window
        paths = path_data.get('paths', [])
        if window_config and window_name != 'full':
            window = window_config.get('windows', {}).get(window_name, {})
            if window:
                # Filter paths to only include transitions within the window
                start_layer = window['start']
                end_layer = window['end']
                paths = self._filter_paths_by_window(paths, start_layer, end_layer)
        
        # Get appropriate clustering data for hierarchy level
        if hierarchy_level in clustering_data.get('hierarchy_results', {}):
            hierarchy_clustering = clustering_data['hierarchy_results'][hierarchy_level]
        else:
            hierarchy_clustering = clustering_data
        
        # Convert clustering data to format expected by unified SankeyGenerator
        sankey_data = self._convert_to_sankey_format(
            clustering_data, path_data, cluster_labels, hierarchy_level, window_name
        )
        
        # Use the unified SankeyGenerator
        try:
            fig = self.sankey_generator.create_sankey(sankey_data, window=window_name)
            return fig.to_dict()
        except Exception as e:
            print(f"Error creating Sankey with unified generator: {e}")
            # Fallback to simple version
            fig = self._create_sankey_from_paths(path_data, cluster_labels, clustering_data, sankey_config)
            return fig.to_dict()
    
    def _convert_to_sankey_format(
        self,
        clustering_data: Dict[str, Any],
        path_data: Dict[str, List[str]],
        cluster_labels: Dict[str, str],
        hierarchy_level: str,
        window_name: str
    ) -> Dict[str, Any]:
        """Convert Concept MRI clustering data to unified Sankey format."""
        # Get layer names from clustering data
        layer_names = sorted(clustering_data.get('clusters_per_layer', {}).keys())
        layers = list(range(len(layer_names)))
        
        # Convert paths to the expected format
        archetypal_paths = []
        for path_id, path in path_data.items():
            # Convert path from ["L0_C1", "L1_C2", ...] to [1, 2, ...]
            numeric_path = []
            for node in path:
                # Extract cluster number from node ID (e.g., "L0_C1" -> 1)
                parts = node.split('_')
                if len(parts) == 2 and parts[1].startswith('C'):
                    cluster_num = int(parts[1][1:])
                    numeric_path.append(cluster_num)
            
            # Create PathInfo structure
            path_info = {
                'path': numeric_path,
                'frequency': 1,  # Default frequency, could be enhanced
                'representative_words': [],  # Not available in basic clustering
                'semantic_labels': [cluster_labels.get(node, node) for node in path]
            }
            archetypal_paths.append(path_info)
        
        # Create windowed analysis structure
        windowed_analysis = {
            window_name: {
                'layers': layers,
                'total_paths': clustering_data.get('metrics', {}).get('total_samples', len(path_data)),
                'unique_paths': len(path_data),
                'archetypal_paths': archetypal_paths
            }
        }
        
        # Create labels structure
        labels_dict = {}
        for layer_idx, layer_name in enumerate(layer_names):
            layer_key = f"layer_{layer_idx}"
            labels_dict[layer_key] = {}
            
            # Get cluster info for this layer
            if layer_name in clustering_data.get('clusters_per_layer', {}):
                n_clusters = clustering_data['clusters_per_layer'][layer_name].get('n_clusters', 0)
                for cluster_id in range(n_clusters):
                    cluster_key = f"L{layer_idx}_C{cluster_id}"
                    original_key = f"{layer_name.replace('layer_', 'L')}_C{cluster_id}"
                    labels_dict[layer_key][cluster_key] = {
                        'label': cluster_labels.get(original_key, f"Cluster {cluster_id}")
                    }
        
        # Create the final data structure
        sankey_data = {
            'windowed_analysis': windowed_analysis,
            'labels': {
                'all': labels_dict  # The unified generator expects a config key
            },
            'purity_data': None  # Not available in basic clustering
        }
        
        return sankey_data
    
    def _filter_paths_by_window(
        self,
        paths: List[Dict[str, Any]],
        start_layer: int,
        end_layer: int
    ) -> List[Dict[str, Any]]:
        """Filter paths to only include transitions within a window."""
        filtered_paths = []
        for path in paths:
            # Check if all transitions in the path are within the window
            transitions = path.get('transitions', [])
            if all(start_layer <= t['layer'] <= end_layer for t in transitions):
                filtered_paths.append(path)
        return filtered_paths
    
    def _create_sankey_from_paths(
        self,
        path_data: Dict[str, List[str]],
        cluster_labels: Dict[str, str],
        clustering_data: Dict[str, Any],
        config: SankeyConfig
    ) -> go.Figure:
        """Create Sankey figure from clustering paths."""
        if not path_data:
            return self._create_empty_figure()
            
        # Build node list and link data
        nodes = []
        node_map = {}
        links = {'source': [], 'target': [], 'value': []}
        
        # Get unique nodes from paths
        for path_id, path in path_data.items():
            for node in path:
                if node not in node_map:
                    node_map[node] = len(nodes)
                    # Use cluster label if available, otherwise use node ID
                    label = cluster_labels.get(node, node)
                    nodes.append(label)
        
        # Count transitions
        transition_counts = {}
        for path_id, path in path_data.items():
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                key = (source, target)
                transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Build links
        for (source, target), count in transition_counts.items():
            links['source'].append(node_map[source])
            links['target'].append(node_map[target])
            links['value'].append(count)
        
        # Create colors for nodes (by layer)
        node_colors = []
        for node in nodes:
            if 'L0' in node or 'layer_0' in node:
                node_colors.append('#1f77b4')
            elif 'L1' in node or 'layer_1' in node:
                node_colors.append('#ff7f0e')
            elif 'L2' in node or 'layer_2' in node:
                node_colors.append('#2ca02c')
            else:
                node_colors.append('#d62728')
        
        # Create Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value']
            )
        )])
        
        # Update layout
        fig.update_layout(
            title=f"Concept Flow Visualization (Top {len(path_data)} paths)",
            font_size=10,
            height=config.height
        )
        
        return fig
    
    def _create_empty_figure(self) -> Dict[str, Any]:
        """Create empty figure when no data available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available. Please run clustering analysis first.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            height=DEFAULT_SANKEY_HEIGHT,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig.to_dict()
    
    def register_callbacks(self, app):
        """Register Dash callbacks for interactivity."""
        
        @app.callback(
            [Output(f"{self.component_id}-window", "options"),
             Output(f"{self.component_id}-window", "value")],
            Input("window-config-store", "data"),
            prevent_initial_call=False
        )
        def update_window_options(window_config):
            """Update window dropdown options based on Layer Window Manager configuration."""
            options = [{"label": "Full Network", "value": "full"}]
            
            if window_config and window_config.get('windows'):
                for window_name, window_data in window_config['windows'].items():
                    label = f"{window_name} (L{window_data['start']}-L{window_data['end']})"
                    options.append({"label": label, "value": window_name})
            
            return options, "full"
        
        @app.callback(
            [Output(f"{self.component_id}-container", "children"),
             Output(f"{self.component_id}-path-collapse", "is_open")],
            [Input(f"{self.component_id}-update-btn", "n_clicks"),
             Input("clustering-store", "data")],
            [State(f"{self.component_id}-top-n", "value"),
             State(f"{self.component_id}-color-by", "value"),
             State(f"{self.component_id}-window", "value"),
             State(f"{self.component_id}-hierarchy", "value"),
             State("window-config-store", "data")],
            prevent_initial_call=False
        )
        def update_sankey(n_clicks, clustering_data, top_n, color_by, window,
                         hierarchy, window_config):
            """Update Sankey diagram based on settings."""
            print(f"DEBUG Sankey: n_clicks={n_clicks}, has_clustering_data={bool(clustering_data)}")
            if clustering_data:
                print(f"DEBUG Sankey: clustering_data keys={list(clustering_data.keys())}")
                print(f"DEBUG Sankey: completed={clustering_data.get('completed')}")
                print(f"DEBUG Sankey: has paths={bool(clustering_data.get('paths'))}")
                
            if not clustering_data:
                return dcc.Graph(
                    figure=self._create_empty_figure(),
                    id=f"{self.component_id}-graph",
                    config={'displayModeBar': False}
                ), False
            
            # Generate configuration
            config = {
                'top_n': top_n or DEFAULT_TOP_N_PATHS,
                'color_by': color_by,
                'window': window,
                'hierarchy': hierarchy
            }
            
            # Extract path data and labels from clustering data
            path_data = clustering_data.get('paths', {})
            cluster_labels = clustering_data.get('cluster_labels', {})
            
            # Generate Sankey
            fig_data = self.generate_sankey(
                clustering_data,
                path_data,
                cluster_labels,
                window_config,
                config
            )
            
            graph = dcc.Graph(
                figure=fig_data,
                id=f"{self.component_id}-graph",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
                }
            )
            
            return graph, False
        
        @app.callback(
            Output(f"{self.component_id}-path-info", "children"),
            Input(f"{self.component_id}-graph", "clickData"),
            State("path-analysis-store", "data"),
            prevent_initial_call=True
        )
        def display_path_info(click_data, path_data):
            """Display information about clicked path."""
            if not click_data or not path_data:
                return "Click on a path to see details."
            
            # Extract path information from click data
            # In real implementation, this would show path statistics
            return html.Div([
                html.P("Path: L1_C0 → L2_C1 → L3_C0"),
                html.P("Frequency: 15 samples"),
                html.P("Stability: 0.85"),
                html.P("Dominant features: Feature 1, Feature 3, Feature 7")
            ])
        
        @app.callback(
            Output("download-sankey", "data"),
            Input(f"{self.component_id}-export-btn", "n_clicks"),
            State(f"{self.component_id}-graph", "figure"),
            prevent_initial_call=True
        )
        def export_sankey(n_clicks, figure):
            """Export Sankey diagram."""
            if not n_clicks or not figure:
                return None
            
            # Convert to HTML for export
            import plotly.io as pio
            html_str = pio.to_html(go.Figure(figure), include_plotlyjs='cdn')
            
            return {
                "content": html_str,
                "filename": "sankey_diagram.html"
            }