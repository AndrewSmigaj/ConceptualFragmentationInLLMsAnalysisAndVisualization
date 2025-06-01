"""
Clustering configuration panel for Concept MRI.
"""
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Any
import numpy as np

from concept_mri.config.settings import (
    DEFAULT_CLUSTERING_ALGORITHM, DEFAULT_METRIC,
    CLUSTERING_ALGORITHMS, CLUSTERING_METRICS,
    MIN_CLUSTERS, MAX_CLUSTERS, THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR
)

# Import ETS functionality
from concept_fragmentation.metrics.explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_ets_statistics,
    explain_ets_similarity
)


class ClusteringPanel:
    """Clustering panel component."""
    
    def __init__(self):
        """Initialize the clustering panel."""
        self.id_prefix = "clustering"
    
    def create_component(self):
        """Create and return the component layout."""
        return create_clustering_panel()

def get_k_for_hierarchy(hierarchy_level, n_samples, algorithm='kmeans'):
    """Calculate appropriate k based on hierarchy level and sample size."""
    # Base calculation
    sqrt_n = np.sqrt(n_samples)
    
    # Hierarchy multipliers
    hierarchy_multipliers = {
        1: 0.5,   # Macro: fewer clusters
        2: 1.0,   # Meso: balanced
        3: 2.0    # Micro: more clusters
    }
    
    multiplier = hierarchy_multipliers.get(hierarchy_level, 1.0)
    
    # Calculate k with bounds
    if algorithm == 'kmeans':
        k = int(np.ceil(sqrt_n * multiplier))
        k = max(2, min(k, n_samples // 2))  # At least 2, at most n/2
    elif algorithm == 'ets':
        # For ETS, adjust threshold percentile instead
        # Lower hierarchy = higher percentile (looser clusters)
        percentiles = {1: 30, 2: 10, 3: 5}
        return percentiles.get(hierarchy_level, 10)
    
    return k

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
                            {'label': 'DBSCAN', 'value': 'dbscan'},
                            {'label': 'ETS (Explainable Threshold)', 'value': 'ets'}
                        ],
                        value=DEFAULT_CLUSTERING_ALGORITHM,
                        inline=True
                    )
                ], width=12)
            ], className='mb-3'),
            
            # Hierarchy level control
            dbc.Row([
                dbc.Col([
                    dbc.Label("Clustering Hierarchy", html_for="hierarchy-level"),
                    html.Div([
                        dcc.Slider(
                            id='hierarchy-level',
                            min=1,
                            max=3,
                            value=2,
                            step=1,
                            marks={
                                1: {'label': 'Macro', 'style': {'color': THEME_COLOR}},
                                2: {'label': 'Meso', 'style': {'color': SECONDARY_COLOR}},
                                3: {'label': 'Micro', 'style': {'color': ACCENT_COLOR}}
                            },
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),
                        dbc.FormText("Macro: Few large clusters | Meso: Balanced | Micro: Many small clusters")
                    ])
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
            
            # ETS specific options
            html.Div(id='ets-options', style={'display': 'none'}, children=[
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    "ETS creates transparent clusters where membership is based on dimension-wise thresholds."
                ], color="info", className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Threshold Mode", html_for="ets-threshold-mode"),
                        dbc.RadioItems(
                            id='ets-threshold-mode',
                            options=[
                                {'label': 'Automatic', 'value': 'auto'},
                                {'label': 'Manual', 'value': 'manual'}
                            ],
                            value='auto',
                            inline=True
                        )
                    ], width=12)
                ], className='mb-3'),
                
                # Automatic threshold options
                html.Div(id='ets-auto-options', children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Threshold Percentile", html_for="ets-percentile"),
                            html.Div([
                                dcc.Slider(
                                    id='ets-percentile',
                                    min=1,
                                    max=50,
                                    value=10,
                                    step=1,
                                    marks={
                                        1: '1%',
                                        10: '10%',
                                        25: '25%',
                                        50: '50%'
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                dbc.FormText("Lower percentile = tighter clusters, Higher = looser clusters")
                            ])
                        ], width=12)
                    ], className='mb-3'),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Minimum Threshold", html_for="ets-min-threshold"),
                            dbc.Input(
                                id='ets-min-threshold',
                                type='number',
                                value=0.00001,
                                step=0.00001,
                                min=0,
                                placeholder="1e-5"
                            ),
                            dbc.FormText("Minimum threshold to avoid numerical issues")
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Batch Size", html_for="ets-batch-size"),
                            dbc.Input(
                                id='ets-batch-size',
                                type='number',
                                value=1000,
                                step=100,
                                min=100,
                                max=10000
                            ),
                            dbc.FormText("For processing large datasets")
                        ], width=6)
                    ])
                ]),
                
                # Manual threshold options (hidden by default)
                html.Div(id='ets-manual-options', style={'display': 'none'}, children=[
                    dbc.Alert(
                        "Manual threshold configuration will be available after initial analysis.",
                        color="warning"
                    )
                ]),
                
                # ETS-specific visualizations
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Checkbox(
                                id='ets-show-explanations',
                                label='Show cluster explanations',
                                value=True
                            ),
                            dbc.Checkbox(
                                id='ets-compute-similarity',
                                label='Compute similarity matrix (slower)',
                                value=False
                            )
                        ])
                    ], width=12)
                ], className='mt-3')
            ]),
            
            # Store for hierarchy results
            dcc.Store(id='hierarchy-results-store', data={}),
            
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

def create_ets_visualization(ets_results: Dict[str, Any]):
    """Create visualization for ETS clustering results."""
    if not ets_results:
        return None
    
    # Create a multi-panel visualization
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cluster Sizes', 
            'Active Dimensions per Cluster',
            'Dimension Importance',
            'Threshold Distribution'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    stats = ets_results.get('statistics', {})
    
    # Cluster sizes
    if 'cluster_sizes' in ets_results:
        sizes = ets_results['cluster_sizes']
        fig.add_trace(
            go.Bar(
                x=list(range(len(sizes))),
                y=sizes,
                name='Cluster Size',
                marker_color=THEME_COLOR
            ),
            row=1, col=1
        )
    
    # Active dimensions per cluster
    if 'active_dimensions' in stats:
        active_dims = stats['active_dimensions']
        if 'per_cluster' in active_dims:
            fig.add_trace(
                go.Bar(
                    x=list(range(len(active_dims['per_cluster']))),
                    y=active_dims['per_cluster'],
                    name='Active Dimensions',
                    marker_color=SECONDARY_COLOR
                ),
                row=1, col=2
            )
    
    # Dimension importance
    if 'dimension_importance' in stats:
        importance = stats['dimension_importance']
        dims = list(importance.keys())[:20]  # Show top 20 dimensions
        values = [importance[d] for d in dims]
        fig.add_trace(
            go.Bar(
                x=dims,
                y=values,
                name='Importance',
                marker_color=ACCENT_COLOR
            ),
            row=2, col=1
        )
    
    # Threshold distribution
    if 'thresholds' in ets_results:
        fig.add_trace(
            go.Histogram(
                x=ets_results['thresholds'],
                nbinsx=30,
                name='Thresholds',
                marker_color='#2ca02c'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="ETS Clustering Analysis"
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})

def register_clustering_panel_callbacks(app):
    """Register callbacks for clustering panel."""
    
    @app.callback(
        [Output('kmeans-options', 'style'),
         Output('dbscan-options', 'style'),
         Output('ets-options', 'style')],
        Input('clustering-algorithm', 'value')
    )
    def toggle_algorithm_options(algorithm):
        """Show/hide algorithm-specific options."""
        if algorithm == 'kmeans':
            return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}
        elif algorithm == 'dbscan':
            return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
        else:  # ets
            return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    
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
        [Output('ets-auto-options', 'style'),
         Output('ets-manual-options', 'style')],
        Input('ets-threshold-mode', 'value')
    )
    def toggle_ets_threshold_mode(mode):
        """Show/hide ETS threshold options."""
        if mode == 'auto':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
    @app.callback(
        [Output('manual-k-slider', 'min'),
         Output('manual-k-slider', 'max'),
         Output('manual-k-slider', 'value'),
         Output('manual-k-slider', 'marks')],
        [Input('hierarchy-level', 'value')]
    )
    def update_k_range_for_hierarchy(hierarchy_level):
        """Update k slider range based on hierarchy level."""
        ranges = {
            1: (2, 10, 5),    # Macro: 2-10 clusters
            2: (3, 20, 10),   # Meso: 3-20 clusters
            3: (5, 50, 20)    # Micro: 5-50 clusters
        }
        
        min_k, max_k, default_k = ranges.get(hierarchy_level, (3, 20, 10))
        
        # Create marks
        if max_k <= 20:
            marks = {i: str(i) for i in range(min_k, max_k+1, 2)}
        else:
            marks = {i: str(i) for i in range(min_k, max_k+1, 10)}
            marks[min_k] = str(min_k)
            marks[max_k] = str(max_k)
        
        return min_k, max_k, default_k, marks
    
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
         State('dataset-store', 'data'),
         State('hierarchy-level', 'value'),
         # ETS-specific states
         State('ets-threshold-mode', 'value'),
         State('ets-percentile', 'value'),
         State('ets-min-threshold', 'value'),
         State('ets-batch-size', 'value'),
         State('ets-show-explanations', 'value'),
         State('ets-compute-similarity', 'value')],
        prevent_initial_call=True
    )
    def run_clustering(n_clicks, algorithm, k_method, manual_k, distance, seed, 
                      model_data, dataset_data, hierarchy_level, ets_mode, ets_percentile, 
                      ets_min_threshold, ets_batch_size, ets_show_explanations, ets_compute_similarity):
        """Run clustering with selected parameters."""
        if not n_clicks or not model_data or not dataset_data:
            return None, {'display': 'none'}, 0, "Ready to run clustering.", "info", None
        
        # Show progress
        progress_style = {'display': 'block'}
        
        # Process hierarchical clustering
        hierarchy_results = {}
        hierarchy_names = {1: 'macro', 2: 'meso', 3: 'micro'}
        
        # Handle ETS clustering
        if algorithm == 'ets':
            try:
                # Extract activations from model data
                activations = model_data.get('activations', {})
                
                # Adjust threshold based on hierarchy level
                if ets_mode == 'auto':
                    ets_percentile = get_k_for_hierarchy(hierarchy_level, 100, 'ets')
                
                # Process ETS for each layer
                ets_results = {'algorithm': 'ets', 'hierarchy': hierarchy_names[hierarchy_level], 'clusters_per_layer': {}}
                
                for layer_name, layer_activations in activations.items():
                    # Reshape if needed (assuming shape is [samples, features])
                    if isinstance(layer_activations, np.ndarray) and len(layer_activations.shape) == 2:
                        # Run ETS clustering
                        threshold_percentile = ets_percentile / 100.0  # Convert to 0-1 range
                        
                        cluster_labels, thresholds = compute_ets_clustering(
                            activations=layer_activations,
                            threshold_percentile=threshold_percentile,
                            min_threshold=ets_min_threshold,
                            batch_size=int(ets_batch_size),
                            verbose=False,
                            return_similarity=ets_compute_similarity
                        )
                        
                        # Compute statistics if requested
                        stats = None
                        if ets_show_explanations:
                            stats = compute_ets_statistics(
                                layer_activations, 
                                cluster_labels, 
                                thresholds
                            )
                        
                        # Store results
                        ets_results['clusters_per_layer'][layer_name] = {
                            'labels': cluster_labels.tolist(),
                            'thresholds': thresholds.tolist(),
                            'n_clusters': len(np.unique(cluster_labels)),
                            'statistics': stats
                        }
                
                # Create ETS-specific visualization
                ets_viz_data = {
                    'cluster_sizes': np.bincount(cluster_labels).tolist(),
                    'thresholds': thresholds.tolist(),
                    'statistics': stats
                }
                metric_viz = create_ets_visualization(ets_viz_data)
                
                clustering_results = ets_results
                status_msg = f"ETS clustering complete! Found {stats['n_clusters']} explainable clusters."
                
            except Exception as e:
                return None, {'display': 'none'}, 0, f"ETS clustering failed: {str(e)}", "danger", None
        
        else:
            # Mock clustering process for other algorithms
            # Adjust k based on hierarchy if using automatic
            n_samples = 100  # Mock sample size
            if k_method != 'manual':
                manual_k = get_k_for_hierarchy(hierarchy_level, n_samples, algorithm)
            
            # Mock results
            clustering_results = {
                'algorithm': algorithm,
                'hierarchy': hierarchy_names[hierarchy_level],
                'k_method': k_method,
                'num_clusters': manual_k,
                'distance_metric': distance,
                'random_seed': seed,
                'clusters_per_layer': {
                    f'layer_{i}': {
                        'n_clusters': manual_k,
                        'silhouette_score': 0.7 - i * 0.05,
                        'labels': [j % manual_k for j in range(100)]  # Mock labels
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
            status_msg = f"Clustering complete! Found optimal k={clustering_results['num_clusters']} clusters per layer."
        
        return (
            clustering_results,
            progress_style,
            100,
            status_msg,
            "success",
            metric_viz
        )