"""
Clustering configuration panel for Concept MRI.
"""
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Any

from config.settings import (
    DEFAULT_CLUSTERING_ALGORITHM, DEFAULT_METRIC,
    CLUSTERING_ALGORITHMS, CLUSTERING_METRICS,
    MIN_CLUSTERS, MAX_CLUSTERS, THEME_COLOR
)

def create_clustering_panel():
    """Create the clustering configuration panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-project-diagram me-2"),
            "Clustering Configuration"
        ], className="fw-bold"),
        dbc.CardBody([
            # Algorithm selection
            dbc.Row([
                dbc.Col([
                    dbc.Label("Clustering Algorithm", html_for="clustering-algorithm"),
                    dbc.RadioItems(
                        id='clustering-algorithm',
                        options=[
                            {'label': 'K-Means', 'value': 'kmeans'},
                            {'label': 'DBSCAN', 'value': 'dbscan'}
                        ],
                        value=DEFAULT_CLUSTERING_ALGORITHM,
                        inline=True
                    )
                ], width=12)
            ], className='mb-3'),
            
            # K-Means specific options
            html.Div(id='kmeans-options', children=[
                # K selection method
                dbc.Row([
                    dbc.Col([
                        dbc.Label("K Selection Method", html_for="k-selection-method"),
                        dbc.RadioItems(
                            id='k-selection-method',
                            options=[
                                {'label': 'Automatic (Gap Statistic)', 'value': 'gap'},
                                {'label': 'Automatic (Silhouette)', 'value': 'silhouette'},
                                {'label': 'Automatic (Elbow)', 'value': 'elbow'},
                                {'label': 'Manual', 'value': 'manual'}
                            ],
                            value=DEFAULT_METRIC,
                            inline=False
                        )
                    ], width=12)
                ], className='mb-3'),
                
                # Manual k selection
                html.Div(id='manual-k-section', children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Number of Clusters (k)", html_for="manual-k-slider"),
                            dcc.Slider(
                                id='manual-k-slider',
                                min=MIN_CLUSTERS,
                                max=MAX_CLUSTERS,
                                value=5,
                                step=1,
                                marks={i: str(i) for i in range(MIN_CLUSTERS, MAX_CLUSTERS+1, 2)},
                                tooltip={"placement": "bottom", "always_visible": False}
                            )
                        ], width=12)
                    ])
                ], style={'display': 'none'})
            ]),
            
            # DBSCAN specific options
            html.Div(id='dbscan-options', style={'display': 'none'}, children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Epsilon (ε)", html_for="dbscan-eps"),
                        dbc.Input(
                            id='dbscan-eps',
                            type='number',
                            value=0.5,
                            step=0.1,
                            min=0.1
                        ),
                        dbc.FormText("Maximum distance between points in a cluster")
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Min Samples", html_for="dbscan-min-samples"),
                        dbc.Input(
                            id='dbscan-min-samples',
                            type='number',
                            value=5,
                            step=1,
                            min=1
                        ),
                        dbc.FormText("Minimum points to form a cluster")
                    ], width=6)
                ], className='mb-3')
            ]),
            
            # Advanced options (collapsible)
            html.Details([
                html.Summary("Advanced Options", className='fw-bold mb-2'),
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Distance Metric", html_for="distance-metric"),
                            dbc.Select(
                                id='distance-metric',
                                options=[
                                    {'label': 'Euclidean', 'value': 'euclidean'},
                                    {'label': 'Cosine', 'value': 'cosine'},
                                    {'label': 'Manhattan', 'value': 'manhattan'}
                                ],
                                value='euclidean'
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Random Seed", html_for="random-seed"),
                            dbc.Input(
                                id='random-seed',
                                type='number',
                                value=42,
                                placeholder='Leave empty for random'
                            )
                        ], width=6)
                    ], className='mb-3'),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Initialization Method", html_for="init-method"),
                            dbc.Select(
                                id='init-method',
                                options=[
                                    {'label': 'K-Means++', 'value': 'k-means++'},
                                    {'label': 'Random', 'value': 'random'}
                                ],
                                value='k-means++'
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Max Iterations", html_for="max-iter"),
                            dbc.Input(
                                id='max-iter',
                                type='number',
                                value=300,
                                min=10,
                                max=1000
                            )
                        ], width=6)
                    ])
                ])
            ], open=False, className='mb-3'),
            
            # Metric visualization area
            html.Div(id='metric-visualization', className='mb-3'),
            
            # Run clustering button
            dbc.Button(
                [html.I(className="fas fa-play me-2"), "Run Clustering"],
                id='run-clustering-btn',
                color='primary',
                size='lg',
                className='w-100',
                disabled=True
            ),
            
            # Progress/status feedback
            dbc.Progress(
                id='clustering-progress',
                value=0,
                striped=True,
                animated=True,
                className='mt-3',
                style={'display': 'none'}
            ),
            
            # Status message
            dbc.Alert(
                "Configure clustering parameters and click 'Run Clustering' to begin.",
                id='clustering-status',
                color='info',
                is_open=True,
                className='mt-3'
            )
        ])
    ])

def create_metric_visualization(metric_data: Dict[str, Any]):
    """Create visualization for clustering metrics."""
    if not metric_data:
        return None
    
    metric_type = metric_data.get('type', 'gap')
    
    if metric_type == 'gap':
        # Gap statistic plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metric_data['k_values'],
            y=metric_data['gap_values'],
            mode='lines+markers',
            name='Gap Statistic',
            line=dict(color=THEME_COLOR, width=2),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=metric_data['k_values'],
            y=metric_data['gap_values'],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='star'
            ),
            name='Optimal k',
            showlegend=True,
            visible=True,
            selectedpoints=[metric_data['optimal_k'] - MIN_CLUSTERS]
        ))
        fig.update_layout(
            title="Gap Statistic Analysis",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Gap Statistic",
            hovermode='x unified',
            height=300
        )
        
    elif metric_type == 'silhouette':
        # Silhouette score plot
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metric_data['k_values'],
            y=metric_data['silhouette_scores'],
            marker_color=[THEME_COLOR if k != metric_data['optimal_k'] else 'red' 
                         for k in metric_data['k_values']],
            text=[f"{score:.3f}" for score in metric_data['silhouette_scores']],
            textposition='outside'
        ))
        fig.update_layout(
            title="Silhouette Score Analysis",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            height=300,
            showlegend=False
        )
        
    elif metric_type == 'elbow':
        # Elbow method plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=metric_data['k_values'],
            y=metric_data['inertia_values'],
            mode='lines+markers',
            name='Inertia',
            line=dict(color=THEME_COLOR, width=2),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="Elbow Method Analysis",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Within-Cluster Sum of Squares",
            hovermode='x unified',
            height=300
        )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def register_clustering_panel_callbacks(app):
    """Register callbacks for clustering panel."""
    
    @app.callback(
        [Output('kmeans-options', 'style'),
         Output('dbscan-options', 'style')],
        Input('clustering-algorithm', 'value')
    )
    def toggle_algorithm_options(algorithm):
        """Show/hide algorithm-specific options."""
        if algorithm == 'kmeans':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
    @app.callback(
        Output('manual-k-section', 'style'),
        Input('k-selection-method', 'value')
    )
    def toggle_manual_k(method):
        """Show/hide manual k selection."""
        if method == 'manual':
            return {'display': 'block'}
        return {'display': 'none'}
    
    @app.callback(
        Output('run-clustering-btn', 'disabled'),
        [Input('model-store', 'data'),
         Input('dataset-store', 'data')]
    )
    def enable_clustering_button(model_data, dataset_data):
        """Enable clustering button when both model and dataset are loaded."""
        return not (model_data and dataset_data)
    
    @app.callback(
        [Output('clustering-store', 'data'),
         Output('clustering-progress', 'style'),
         Output('clustering-progress', 'value'),
         Output('clustering-status', 'children'),
         Output('clustering-status', 'color'),
         Output('metric-visualization', 'children')],
        Input('run-clustering-btn', 'n_clicks'),
        [State('clustering-algorithm', 'value'),
         State('k-selection-method', 'value'),
         State('manual-k-slider', 'value'),
         State('distance-metric', 'value'),
         State('random-seed', 'value'),
         State('model-store', 'data'),
         State('dataset-store', 'data')],
        prevent_initial_call=True
    )
    def run_clustering(n_clicks, algorithm, k_method, manual_k, distance, seed, 
                      model_data, dataset_data):
        """Run clustering with selected parameters."""
        if not n_clicks or not model_data or not dataset_data:
            return None, {'display': 'none'}, 0, "Ready to run clustering.", "info", None
        
        # Show progress
        progress_style = {'display': 'block'}
        
        # Mock clustering process (would integrate with real clustering)
        # Step 1: Extract activations (30%)
        # Step 2: Run clustering (60%)
        # Step 3: Calculate metrics (90%)
        # Step 4: Generate results (100%)
        
        # Mock results
        clustering_results = {
            'algorithm': algorithm,
            'k_method': k_method,
            'num_clusters': manual_k if k_method == 'manual' else 7,
            'distance_metric': distance,
            'random_seed': seed,
            'clusters_per_layer': {
                f'layer_{i}': {
                    'n_clusters': 5 + i % 3,
                    'silhouette_score': 0.7 - i * 0.05,
                    'labels': [0, 1, 2, 1, 0] * 20  # Mock labels
                }
                for i in range(5)
            }
        }
        
        # Mock metric data
        metric_data = {
            'type': k_method if k_method != 'manual' else 'gap',
            'k_values': list(range(2, 11)),
            'gap_values': [0.2, 0.3, 0.5, 0.8, 0.9, 0.85, 0.8, 0.75, 0.7],
            'optimal_k': 7,
            'silhouette_scores': [0.3, 0.4, 0.5, 0.7, 0.8, 0.75, 0.7, 0.65, 0.6],
            'inertia_values': [100, 80, 60, 45, 35, 30, 28, 27, 26]
        }
        
        metric_viz = create_metric_visualization(metric_data)
        
        return (
            clustering_results,
            progress_style,
            100,
            f"Clustering complete! Found optimal k={clustering_results['num_clusters']} clusters per layer.",
            "success",
            metric_viz
        )