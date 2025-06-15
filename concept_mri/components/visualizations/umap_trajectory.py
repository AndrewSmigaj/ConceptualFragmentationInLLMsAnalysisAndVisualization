"""
UMAP-based trajectory visualization component for Concept MRI.
Shows activation vectors reduced to 3D with layer offsets.
"""
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import numpy as np
from umap import UMAP

from concept_mri.config.settings import THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR
from concept_mri.core.activation_manager import activation_manager


class UMAPTrajectoryVisualization:
    """Creates UMAP-reduced 3D trajectory plots with layer offsets."""
    
    def __init__(self, component_id: str = "umap-trajectory"):
        self.component_id = component_id
        self.current_data = None
        self.LAYER_SEPARATION_OFFSET = 10.0  # Y-axis offset between layers for better visual separation
        
    def create_component(self) -> dbc.Card:
        """Create the UMAP trajectory visualization component."""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-route me-2"),
                "Activation Trajectory Analysis",
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
                        dbc.Label("Color By", html_for=f"{self.component_id}-color-by"),
                        dbc.Select(
                            id=f"{self.component_id}-color-by",
                            options=[
                                {"label": "Cluster", "value": "cluster"},
                                {"label": "Sample Index", "value": "index"},
                                {"label": "Activation Magnitude", "value": "magnitude"}
                            ],
                            value="cluster"
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Sample Size", html_for=f"{self.component_id}-samples"),
                        dbc.Input(
                            id=f"{self.component_id}-samples",
                            type="number",
                            value=50,
                            min=10,
                            max=200,
                            step=10
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("UMAP Neighbors", html_for=f"{self.component_id}-neighbors"),
                        dbc.Input(
                            id=f"{self.component_id}-neighbors",
                            type="number",
                            value=15,
                            min=5,
                            max=50,
                            step=5
                        )
                    ], width=4)
                ], className="mb-3"),
                
                # View controls
                dbc.Row([
                    dbc.Col([
                        dbc.Checklist(
                            id=f"{self.component_id}-options",
                            options=[
                                {"label": "Show arrows", "value": "arrows"},
                                {"label": "Show paths", "value": "paths"},
                                {"label": "Show cluster centers", "value": "centers"},
                                {"label": "Show layer planes", "value": "planes"},
                                {"label": "Normalize embeddings", "value": "normalize"}
                            ],
                            value=["arrows", "paths", "normalize"],
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
                            style={"height": "700px"}
                        )
                    ]
                ),
                
                # Statistics panel
                html.Div(id=f"{self.component_id}-stats", className="mt-3")
            ])
        ])
    
    def generate_umap_trajectory(
        self,
        model_data: Dict[str, Any],
        clustering_data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate UMAP trajectory visualization.
        
        Args:
            model_data: Model data including activations
            clustering_data: Clustering results
            config: Visualization configuration
            
        Returns:
            Plotly figure data
        """
        if not model_data or not clustering_data:
            return self._create_empty_figure()
        
        # Get activations
        activations = self._get_activations_from_model_data(model_data)
        if not activations:
            return self._create_empty_figure()
        
        # Extract configuration
        n_samples = config.get('n_samples', 50) if config else 50
        n_neighbors = config.get('n_neighbors', 15) if config else 15
        color_by = config.get('color_by', 'cluster') if config else 'cluster'
        show_options = config.get('show_options', ['arrows', 'paths']) if config else ['arrows', 'paths']
        
        # Prepare data for UMAP
        layer_names = sorted(activations.keys())
        
        # Apply layer filtering if specified
        layer_range = config.get('layer_range') if config else None
        if layer_range:
            start_idx, end_idx = layer_range
            layer_names = layer_names[start_idx:end_idx]
        
        if not layer_names:
            return self._create_empty_figure()
        
        n_samples = min(n_samples, activations[layer_names[0]].shape[0])
        
        # Sample indices
        sample_indices = np.random.choice(
            activations[layer_names[0]].shape[0], 
            size=n_samples, 
            replace=False
        )
        
        # Run UMAP separately for each layer first, then combine
        embeddings_list = []
        
        for layer_idx, layer_name in enumerate(layer_names):
            layer_acts = activations[layer_name][sample_indices]
            
            # Run UMAP on this layer
            if layer_acts.shape[1] >= n_neighbors:  # Ensure enough features
                reducer = UMAP(n_components=3, n_neighbors=min(n_neighbors, layer_acts.shape[1]-1), random_state=42)
                layer_embeddings = reducer.fit_transform(layer_acts)
            else:
                # If too few features, use PCA or direct projection
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                layer_embeddings = pca.fit_transform(layer_acts)
            
            # Normalize embeddings if requested
            if 'normalize' in show_options:
                # Center the embeddings
                layer_embeddings = layer_embeddings - np.mean(layer_embeddings, axis=0)
                
                # Scale to unit variance (this ensures consistent spread across layers)
                std = np.std(layer_embeddings, axis=0)
                std[std == 0] = 1  # Avoid division by zero
                layer_embeddings = layer_embeddings / std
                
                # Further scale to a fixed radius for visual consistency
                # This ensures all layers have the same visual size
                target_radius = 3.0  # Adjust this for desired layer size
                layer_embeddings = layer_embeddings * target_radius
            
            embeddings_list.append(layer_embeddings)
        
        # Stack all embeddings
        embeddings_3d = np.vstack(embeddings_list)
        
        # Create layer indices
        layer_indices = []
        for layer_idx in range(len(layer_names)):
            layer_indices.extend([layer_idx] * n_samples)
        
        # Add layer offsets
        for i, layer_idx in enumerate(layer_indices):
            embeddings_3d[i, 1] += layer_idx * self.LAYER_SEPARATION_OFFSET
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter points
        colors = self._get_colors(clustering_data, sample_indices, layer_indices, color_by, n_samples, len(layer_names))
        
        fig.add_trace(go.Scatter3d(
            x=embeddings_3d[:, 0],
            y=embeddings_3d[:, 1],
            z=embeddings_3d[:, 2],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                colorscale='Viridis' if color_by != 'cluster' else None,
                showscale=color_by != 'cluster',
                opacity=0.8
            ),
            text=[f"Layer {layer_names[layer_indices[i]]}<br>Sample {sample_indices[i % n_samples]}" 
                  for i in range(len(embeddings_3d))],
            hoverinfo='text',
            name='Activations'
        ))
        
        # Add paths between layers
        if 'paths' in show_options:
            for sample_idx in range(n_samples):
                path_x = []
                path_y = []
                path_z = []
                
                for layer_idx in range(len(layer_names)):
                    point_idx = layer_idx * n_samples + sample_idx
                    path_x.append(embeddings_3d[point_idx, 0])
                    path_y.append(embeddings_3d[point_idx, 1])
                    path_z.append(embeddings_3d[point_idx, 2])
                
                fig.add_trace(go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=path_z,
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.3)', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add arrows
        if 'arrows' in show_options:
            self._add_arrows(fig, embeddings_3d, n_samples, len(layer_names))
        
        # Add layer planes
        if 'planes' in show_options:
            self._add_layer_planes(fig, embeddings_3d, layer_names, n_samples)
        
        # Update layout
        fig.update_layout(
            title="Activation Trajectories Through Network Layers (UMAP 3D)",
            scene=dict(
                xaxis_title="UMAP-1",
                yaxis_title="Layer Progression",
                zaxis_title="UMAP-3",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=700,
            showlegend=False
        )
        
        return fig.to_dict()
    
    def _get_activations_from_model_data(self, model_data):
        """Get activations from model data, handling both session storage and direct storage."""
        if not model_data:
            return None
            
        # Try to get from session storage first
        session_id = model_data.get('activation_session_id')
        if session_id:
            activations = activation_manager.get_activations(session_id)
            if activations is not None:
                return activations
        
        # Fall back to direct storage
        return model_data.get('activations', {})
    
    def _get_colors(self, clustering_data, sample_indices, layer_indices, color_by, n_samples, n_layers):
        """Get colors for points based on coloring mode."""
        if color_by == 'cluster':
            colors = []
            clusters_per_layer = clustering_data.get('clusters_per_layer', {})
            layer_names = sorted(clusters_per_layer.keys())
            
            for i in range(len(layer_indices)):
                layer_idx = layer_indices[i]
                sample_idx = sample_indices[i % n_samples]
                
                if layer_idx < len(layer_names):
                    layer_name = layer_names[layer_idx]
                    if layer_name in clusters_per_layer:
                        labels = clusters_per_layer[layer_name].get('labels', [])
                        if sample_idx < len(labels):
                            colors.append(labels[sample_idx])
                        else:
                            colors.append(0)
                    else:
                        colors.append(0)
                else:
                    colors.append(0)
                    
        elif color_by == 'index':
            colors = [sample_indices[i % n_samples] for i in range(n_samples * n_layers)]
        else:  # magnitude
            colors = np.random.rand(n_samples * n_layers)  # Mock magnitude
            
        return colors
    
    def _add_arrows(self, fig, embeddings, n_samples, n_layers):
        """Add directional arrows between consecutive layers."""
        for layer_idx in range(n_layers - 1):
            for sample_idx in range(n_samples):
                start_idx = layer_idx * n_samples + sample_idx
                end_idx = (layer_idx + 1) * n_samples + sample_idx
                
                # Calculate arrow direction
                direction = embeddings[end_idx] - embeddings[start_idx]
                
                # Add small cone for arrow
                mid_point = (embeddings[start_idx] + embeddings[end_idx]) / 2
                
                fig.add_trace(go.Cone(
                    x=[mid_point[0]],
                    y=[mid_point[1]],
                    z=[mid_point[2]],
                    u=[direction[0]],
                    v=[direction[1]],
                    w=[direction[2]],
                    sizemode="absolute",
                    sizeref=0.5,
                    showscale=False,
                    colorscale=[[0, 'gray'], [1, 'gray']],
                    opacity=0.6,
                    hoverinfo='skip'
                ))
    
    def _add_layer_planes(self, fig, embeddings, layer_names, n_samples):
        """Add transparent planes at each layer level."""
        for layer_idx, layer_name in enumerate(layer_names):
            layer_y = layer_idx * self.LAYER_SEPARATION_OFFSET
            
            # Get bounds for this layer
            start_idx = layer_idx * n_samples
            end_idx = (layer_idx + 1) * n_samples
            layer_points = embeddings[start_idx:end_idx]
            
            x_min, x_max = layer_points[:, 0].min(), layer_points[:, 0].max()
            z_min, z_max = layer_points[:, 2].min(), layer_points[:, 2].max()
            
            # Add plane
            xx, zz = np.meshgrid(
                np.linspace(x_min - 1, x_max + 1, 10),
                np.linspace(z_min - 1, z_max + 1, 10)
            )
            yy = np.full_like(xx, layer_y)
            
            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=zz,
                opacity=0.2,
                colorscale=[[0, 'lightblue'], [1, 'lightblue']],
                showscale=False,
                hoverinfo='skip',
                name=layer_name
            ))
            
            # Add layer label
            fig.add_trace(go.Scatter3d(
                x=[x_max + 2],
                y=[layer_y],
                z=[z_max + 2],
                mode='text',
                text=[layer_name.replace('layer_', 'Layer ')],
                textposition='middle right',
                showlegend=False,
                hoverinfo='skip'
            ))
    
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
            height=700,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
        return fig.to_dict()
    
    def register_callbacks(self, app):
        """Register Dash callbacks for interactivity."""
        
        @app.callback(
            Output(f"{self.component_id}-container", "children"),
            [Input("model-store", "data"),
             Input("clustering-store", "data"),
             Input(f"{self.component_id}-color-by", "value"),
             Input(f"{self.component_id}-samples", "value"),
             Input(f"{self.component_id}-neighbors", "value"),
             Input(f"{self.component_id}-options", "value")],
            prevent_initial_call=False
        )
        def update_visualization(model_data, clustering_data, color_by, n_samples, n_neighbors, options):
            """Update UMAP trajectory visualization."""
            if not model_data or not clustering_data:
                return dcc.Graph(
                    figure=self._create_empty_figure(),
                    id=f"{self.component_id}-graph",
                    config={'displayModeBar': True}
                )
            
            config = {
                'color_by': color_by,
                'n_samples': n_samples,
                'n_neighbors': n_neighbors,
                'show_options': options or []
            }
            
            fig_data = self.generate_umap_trajectory(
                model_data,
                clustering_data,
                config
            )
            
            return dcc.Graph(
                figure=fig_data,
                id=f"{self.component_id}-graph",
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': f'umap_trajectory_{color_by}'
                    }
                }
            )
        
        @app.callback(
            Output(f"{self.component_id}-stats", "children"),
            Input(f"{self.component_id}-graph", "hoverData"),
            State("clustering-store", "data"),
            prevent_initial_call=True
        )
        def update_stats(hover_data, clustering_data):
            """Update statistics based on hover."""
            if not hover_data or not clustering_data:
                return html.P("Hover over points to see details", className="text-muted")
            
            # Extract hover information
            point = hover_data['points'][0]
            text = point.get('text', '')
            
            return html.Div([
                html.H6("Point Details"),
                html.P(text),
                html.Hr(),
                html.P(f"Total paths: {clustering_data.get('metrics', {}).get('unique_paths', 'N/A')}"),
                html.P(f"Total samples: {clustering_data.get('metrics', {}).get('total_samples', 'N/A')}")
            ])