"""
Stepped trajectory visualization component for Concept MRI.
Shows cluster assignments across layers as step functions.
"""
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import numpy as np

from concept_mri.config.settings import THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR


class SteppedTrajectoryVisualization:
    """Creates stepped trajectory plots showing cluster transitions across layers."""
    
    def __init__(self, component_id: str = "stepped-trajectory"):
        self.component_id = component_id
        self.current_data = None
        
    def create_component(self) -> dbc.Card:
        """Create the stepped trajectory visualization component."""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-chart-line me-2"),
                "Stepped Trajectory Analysis",
                dbc.Button(
                    html.I(className="fas fa-expand"),
                    id=f"{self.component_id}-fullscreen-btn",
                    color="link",
                    size="sm",
                    className="float-end"
                )
            ], className="fw-bold"),
            dbc.CardBody([
                # Controls row
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Display Mode", html_for=f"{self.component_id}-mode"),
                        dbc.RadioItems(
                            id=f"{self.component_id}-mode",
                            options=[
                                {"label": "Individual Paths", "value": "individual"},
                                {"label": "Aggregated", "value": "aggregated"},
                                {"label": "Heatmap", "value": "heatmap"}
                            ],
                            value="individual",
                            inline=True
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Sample Size", html_for=f"{self.component_id}-samples"),
                        dbc.Input(
                            id=f"{self.component_id}-samples",
                            type="number",
                            value=50,
                            min=10,
                            max=500,
                            step=10
                        )
                    ], width=3),
                    dbc.Col([
                        dbc.Label("Window", html_for=f"{self.component_id}-window"),
                        dbc.Select(
                            id=f"{self.component_id}-window",
                            options=[
                                {"label": "Full Network", "value": "full"}
                            ],
                            value="full"
                        )
                    ], width=3)
                ], className="mb-3"),
                
                # Zoom controls
                dbc.Row([
                    dbc.Col([
                        dbc.ButtonGroup([
                            dbc.Button(
                                html.I(className="fas fa-search-plus"),
                                id=f"{self.component_id}-zoom-in",
                                color="secondary",
                                size="sm"
                            ),
                            dbc.Button(
                                html.I(className="fas fa-search-minus"),
                                id=f"{self.component_id}-zoom-out",
                                color="secondary",
                                size="sm"
                            ),
                            dbc.Button(
                                html.I(className="fas fa-home"),
                                id=f"{self.component_id}-reset-view",
                                color="secondary",
                                size="sm"
                            )
                        ], className="me-3"),
                        html.Small("Use scroll to zoom, drag to pan", className="text-muted")
                    ], width=12)
                ], className="mb-3"),
                
                # Hierarchy selector
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
                
                # Visualization container
                dcc.Loading(
                    id=f"{self.component_id}-loading",
                    type="default",
                    color=THEME_COLOR,
                    children=[
                        html.Div(
                            id=f"{self.component_id}-container",
                            style={"height": "600px"}
                        )
                    ]
                ),
                
                # Legend and statistics
                dbc.Row([
                    dbc.Col([
                        html.Div(id=f"{self.component_id}-legend")
                    ], width=8),
                    dbc.Col([
                        html.Div(id=f"{self.component_id}-stats")
                    ], width=4)
                ], className="mt-3")
            ])
        ])
    
    def generate_stepped_plot(
        self,
        clustering_data: Dict[str, Any],
        window_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate stepped trajectory visualization.
        
        Args:
            clustering_data: Clustering results with trajectories
            window_config: Window configuration from Layer Window Manager
            config: Visualization configuration
            
        Returns:
            Plotly figure data
        """
        if not clustering_data:
            return self._create_empty_figure()
        
        mode = config.get('mode', 'individual') if config else 'individual'
        n_samples = config.get('n_samples', 50) if config else 50
        window = config.get('window', 'full') if config else 'full'
        hierarchy = config.get('hierarchy', 'meso') if config else 'meso'
        
        # Get layer names and filter by window
        layers = list(clustering_data.get('clusters_per_layer', {}).keys())
        if window_config and window != 'full':
            window_data = window_config.get('windows', {}).get(window, {})
            if window_data:
                start_idx = window_data['start']
                end_idx = window_data['end'] + 1
                layers = layers[start_idx:end_idx]
        
        if mode == 'individual':
            fig = self._create_individual_paths_plot(clustering_data, layers, n_samples, hierarchy)
        elif mode == 'aggregated':
            fig = self._create_aggregated_plot(clustering_data, layers, hierarchy)
        else:  # heatmap
            fig = self._create_heatmap_plot(clustering_data, layers, hierarchy)
        
        return fig.to_dict()
    
    def _create_individual_paths_plot(
        self,
        clustering_data: Dict[str, Any],
        layers: List[str],
        n_samples: int,
        hierarchy: str
    ) -> go.Figure:
        """Create plot showing individual cluster assignment paths."""
        fig = go.Figure()
        
        # Mock data - in real implementation, extract from clustering_data
        n_clusters = 5 if hierarchy == 'macro' else 10 if hierarchy == 'meso' else 20
        
        # Generate sample trajectories
        for i in range(min(n_samples, 50)):
            # Create stepped line for each sample
            x = []
            y = []
            
            for j, layer in enumerate(layers):
                # Add two points per layer for step effect
                cluster = np.random.randint(0, n_clusters)
                x.extend([j, j+1])
                y.extend([cluster, cluster])
            
            fig.add_trace(go.Scatter(
                x=x[:-1],  # Remove last duplicate
                y=y[:-1],
                mode='lines',
                line=dict(
                    color=f'rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.3)',
                    width=1
                ),
                showlegend=False,
                hovertemplate='Layer %{x}<br>Cluster %{y}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Individual Cluster Trajectories ({hierarchy.capitalize()} level)",
            xaxis=dict(
                title="Layer",
                tickmode='array',
                tickvals=list(range(len(layers))),
                ticktext=[l.replace('layer_', 'L') for l in layers],
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)'
            ),
            yaxis=dict(
                title="Cluster ID",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                dtick=1
            ),
            hovermode='closest',
            height=600
        )
        
        return fig
    
    def _create_aggregated_plot(
        self,
        clustering_data: Dict[str, Any],
        layers: List[str],
        hierarchy: str
    ) -> go.Figure:
        """Create aggregated view showing cluster population flows."""
        fig = go.Figure()
        
        # Mock data - in real implementation, compute from clustering_data
        n_clusters = 5 if hierarchy == 'macro' else 10 if hierarchy == 'meso' else 20
        
        # Create area plot for each cluster
        for cluster_id in range(n_clusters):
            # Generate mock population data
            populations = []
            for layer in layers:
                # Mock: random population with some continuity
                if not populations:
                    pop = np.random.randint(5, 20)
                else:
                    pop = max(0, populations[-1] + np.random.randint(-3, 4))
                populations.append(pop)
            
            fig.add_trace(go.Scatter(
                x=list(range(len(layers))),
                y=populations,
                mode='lines',
                stackgroup='one',
                name=f'Cluster {cluster_id}',
                hovertemplate='%{y} samples<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Cluster Population Flow ({hierarchy.capitalize()} level)",
            xaxis=dict(
                title="Layer",
                tickmode='array',
                tickvals=list(range(len(layers))),
                ticktext=[l.replace('layer_', 'L') for l in layers]
            ),
            yaxis=dict(title="Number of Samples"),
            hovermode='x unified',
            height=600
        )
        
        return fig
    
    def _create_heatmap_plot(
        self,
        clustering_data: Dict[str, Any],
        layers: List[str],
        hierarchy: str
    ) -> go.Figure:
        """Create heatmap showing cluster transition probabilities."""
        # Mock transition matrix - in real implementation, compute from data
        n_clusters = 5 if hierarchy == 'macro' else 10 if hierarchy == 'meso' else 20
        
        # Create subplot for each layer transition
        n_transitions = len(layers) - 1
        fig = make_subplots(
            rows=1, cols=n_transitions,
            subplot_titles=[f"L{i}→L{i+1}" for i in range(n_transitions)],
            horizontal_spacing=0.02
        )
        
        for i in range(n_transitions):
            # Generate mock transition matrix
            transition_matrix = np.random.rand(n_clusters, n_clusters)
            transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
            
            fig.add_trace(
                go.Heatmap(
                    z=transition_matrix,
                    colorscale='Blues',
                    showscale=(i == n_transitions - 1),
                    hovertemplate='From C%{y} to C%{x}<br>Prob: %{z:.2f}<extra></extra>'
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title=f"Cluster Transition Probabilities ({hierarchy.capitalize()} level)",
            height=400 + n_clusters * 20
        )
        
        # Update axes
        for i in range(n_transitions):
            fig.update_xaxis(title="To", row=1, col=i+1)
            if i == 0:
                fig.update_yaxis(title="From", row=1, col=i+1)
        
        return fig
    
    def _create_empty_figure(self) -> Dict[str, Any]:
        """Create empty figure when no data available."""
        fig = go.Figure()
        fig.add_annotation(
            text="No clustering data available. Please run analysis first.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            height=600,
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
            """Update window dropdown based on Layer Window Manager."""
            options = [{"label": "Full Network", "value": "full"}]
            
            if window_config and window_config.get('windows'):
                for window_name, window_data in window_config['windows'].items():
                    label = f"{window_name} (L{window_data['start']}-L{window_data['end']})"
                    options.append({"label": label, "value": window_name})
            
            return options, "full"
        
        @app.callback(
            Output(f"{self.component_id}-container", "children"),
            [Input("clustering-store", "data"),
             Input(f"{self.component_id}-mode", "value"),
             Input(f"{self.component_id}-samples", "value"),
             Input(f"{self.component_id}-window", "value"),
             Input(f"{self.component_id}-hierarchy", "value")],
            State("window-config-store", "data"),
            prevent_initial_call=False
        )
        def update_visualization(clustering_data, mode, n_samples, window, 
                               hierarchy, window_config):
            """Update stepped trajectory visualization."""
            if not clustering_data:
                return dcc.Graph(
                    figure=self._create_empty_figure(),
                    id=f"{self.component_id}-graph",
                    config={'displayModeBar': False}
                )
            
            config = {
                'mode': mode,
                'n_samples': n_samples,
                'window': window,
                'hierarchy': hierarchy
            }
            
            fig_data = self.generate_stepped_plot(
                clustering_data,
                window_config,
                config
            )
            
            return dcc.Graph(
                figure=fig_data,
                id=f"{self.component_id}-graph",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'stepped_trajectory_{hierarchy}_{window}'
                    }
                }
            )
        
        @app.callback(
            Output(f"{self.component_id}-stats", "children"),
            [Input(f"{self.component_id}-graph", "hoverData"),
             Input(f"{self.component_id}-hierarchy", "value")],
            State("clustering-store", "data"),
            prevent_initial_call=True
        )
        def update_stats(hover_data, hierarchy, clustering_data):
            """Update statistics based on hover."""
            if not hover_data or not clustering_data:
                return html.P("Hover over the plot to see statistics", className="text-muted")
            
            # Extract hover information
            point = hover_data['points'][0]
            
            # Mock statistics - in real implementation, compute from data
            return html.Div([
                html.H6("Cluster Statistics"),
                html.P(f"Hierarchy: {hierarchy.capitalize()}"),
                html.P(f"Layer: {point.get('x', 'N/A')}"),
                html.P(f"Cluster: {point.get('y', 'N/A')}"),
                html.P(f"Population: {np.random.randint(10, 100)}"),
                html.P(f"Stability: {np.random.uniform(0.7, 0.95):.2f}")
            ])