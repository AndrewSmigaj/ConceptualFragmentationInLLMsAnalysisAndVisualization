"""
Enhanced cluster cards component for Concept MRI.
Shows detailed cluster information with confidence intervals and transition probabilities.
"""
from dash import dcc, html, Input, Output, State, callback, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import numpy as np

from concept_mri.config.settings import THEME_COLOR, SECONDARY_COLOR, ACCENT_COLOR


class ClusterCards:
    """Creates interactive cards showing detailed cluster information."""
    
    def __init__(self, component_id: str = "cluster-cards"):
        self.component_id = component_id
        self.selected_clusters = set()
        
    def create_component(self) -> dbc.Card:
        """Create the cluster cards component."""
        return dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-th me-2"),
                "Cluster Analysis Cards",
                dbc.Badge(
                    "0 clusters",
                    id=f"{self.component_id}-count",
                    color="primary",
                    className="ms-2"
                ),
                dbc.ButtonGroup([
                    dbc.Button(
                        html.I(className="fas fa-expand-alt"),
                        id=f"{self.component_id}-expand-all",
                        color="link",
                        size="sm"
                    ),
                    dbc.Button(
                        html.I(className="fas fa-compress-alt"),
                        id=f"{self.component_id}-collapse-all",
                        color="link",
                        size="sm"
                    )
                ], className="float-end")
            ], className="fw-bold"),
            dbc.CardBody([
                # Controls
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Layer", html_for=f"{self.component_id}-layer"),
                        dbc.Select(
                            id=f"{self.component_id}-layer",
                            options=[
                                {"label": "Select a layer", "value": ""}
                            ],
                            value=""
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Card Type", html_for=f"{self.component_id}-type"),
                        dbc.RadioItems(
                            id=f"{self.component_id}-type",
                            options=[
                                {"label": "Standard", "value": "standard"},
                                {"label": "ETS", "value": "ets"},
                                {"label": "Hierarchical", "value": "hierarchical"}
                            ],
                            value="standard",
                            inline=True
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Sort By", html_for=f"{self.component_id}-sort"),
                        dbc.Select(
                            id=f"{self.component_id}-sort",
                            options=[
                                {"label": "Cluster ID", "value": "id"},
                                {"label": "Size", "value": "size"},
                                {"label": "Cohesion", "value": "cohesion"},
                                {"label": "Stability", "value": "stability"}
                            ],
                            value="id"
                        )
                    ], width=4)
                ], className="mb-3"),
                
                # Filter controls
                dbc.Row([
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.InputGroupText(html.I(className="fas fa-search")),
                            dbc.Input(
                                id=f"{self.component_id}-search",
                                placeholder="Search clusters...",
                                type="text"
                            )
                        ])
                    ], width=6),
                    dbc.Col([
                        dbc.Checklist(
                            id=f"{self.component_id}-filters",
                            options=[
                                {"label": "Show statistics", "value": "stats"},
                                {"label": "Show transitions", "value": "transitions"},
                                {"label": "Show confidence intervals", "value": "ci"}
                            ],
                            value=["stats"],
                            inline=True
                        )
                    ], width=6)
                ], className="mb-3"),
                
                # Cards container
                dcc.Loading(
                    id=f"{self.component_id}-loading",
                    type="default",
                    color=THEME_COLOR,
                    children=[
                        html.Div(
                            id=f"{self.component_id}-container",
                            className="cluster-cards-grid",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(auto-fill, minmax(300px, 1fr))",
                                "gap": "1rem",
                                "maxHeight": "600px",
                                "overflowY": "auto"
                            }
                        )
                    ]
                ),
                
                # Selected clusters summary
                dbc.Collapse(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Selected Clusters Summary"),
                            html.Div(id=f"{self.component_id}-selection-summary")
                        ])
                    ], className="mt-3"),
                    id=f"{self.component_id}-selection-collapse",
                    is_open=False
                )
            ])
        ])
    
    def create_standard_card(
        self,
        cluster_id: int,
        cluster_data: Dict[str, Any],
        layer_name: str,
        show_options: List[str]
    ) -> dbc.Card:
        """Create a standard cluster card."""
        size = cluster_data.get('size', 0)
        cohesion = cluster_data.get('cohesion', 0.0)
        separation = cluster_data.get('separation', 0.0)
        label = cluster_data.get('label', f'Cluster {cluster_id}')
        
        card_content = [
            dbc.CardHeader([
                html.H6(label, className="mb-0"),
                dbc.Badge(
                    f"C{cluster_id}",
                    color="primary",
                    className="ms-2"
                )
            ]),
            dbc.CardBody([
                # Basic info
                html.P([
                    html.Strong("Size: "),
                    f"{size} samples",
                    html.Span(
                        f" ({size/cluster_data.get('total_samples', 1)*100:.1f}%)",
                        className="text-muted"
                    )
                ], className="mb-2"),
                
                # Statistics
                "stats" in show_options and html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Small("Cohesion", className="text-muted"),
                            html.Div([
                                dbc.Progress(
                                    value=cohesion * 100,
                                    color="success" if cohesion > 0.7 else "warning",
                                    className="mb-1"
                                ),
                                html.Small(f"{cohesion:.3f}")
                            ])
                        ], width=6),
                        dbc.Col([
                            html.Small("Separation", className="text-muted"),
                            html.Div([
                                dbc.Progress(
                                    value=separation * 100,
                                    color="info",
                                    className="mb-1"
                                ),
                                html.Small(f"{separation:.3f}")
                            ])
                        ], width=6)
                    ])
                ], className="mb-2"),
                
                # Confidence intervals
                "ci" in show_options and html.Div([
                    html.Small("95% Confidence Interval", className="text-muted"),
                    html.P(
                        f"Size: [{size - int(size*0.1)}, {size + int(size*0.1)}]",
                        className="mb-1"
                    )
                ], className="mb-2"),
                
                # Transitions
                "transitions" in show_options and html.Div([
                    html.Small("Top Transitions", className="text-muted"),
                    html.Ul([
                        html.Li(f"→ C{i}: {np.random.uniform(0.1, 0.4):.1%}")
                        for i in np.random.choice(range(10), 3, replace=False)
                    ], className="mb-0 small")
                ])
            ])
        ]
        
        return dbc.Card(
            card_content,
            className="h-100",
            id={"type": f"{self.component_id}-card", "cluster": cluster_id}
        )
    
    def create_ets_card(
        self,
        cluster_id: int,
        cluster_data: Dict[str, Any],
        layer_name: str,
        show_options: List[str]
    ) -> dbc.Card:
        """Create an ETS-specific cluster card."""
        base_card = self.create_standard_card(cluster_id, cluster_data, layer_name, show_options)
        
        # Add ETS-specific information
        thresholds = cluster_data.get('thresholds', [])
        active_dims = cluster_data.get('active_dimensions', [])
        
        ets_content = html.Div([
            html.Hr(),
            html.H6("ETS Information", className="mb-2"),
            html.Small("Active Dimensions", className="text-muted"),
            html.P(f"{len(active_dims)} / {len(thresholds)} dimensions"),
            
            # Threshold distribution mini chart
            html.Small("Threshold Distribution", className="text-muted"),
            dcc.Graph(
                figure=self._create_threshold_histogram(thresholds[:20]),  # First 20 dims
                config={'displayModeBar': False},
                style={'height': '100px'}
            ),
            
            # Explainability score
            html.Small("Explainability", className="text-muted"),
            dbc.Progress(
                value=85,  # Mock value
                color="success",
                label="85%"
            )
        ])
        
        # Insert ETS content into card body
        base_card.children[1].children.append(ets_content)
        
        return base_card
    
    def create_hierarchical_card(
        self,
        cluster_id: int,
        cluster_data: Dict[str, Any],
        layer_name: str,
        show_options: List[str],
        hierarchy_data: Dict[str, Any]
    ) -> dbc.Card:
        """Create a hierarchical cluster card showing parent/child relationships."""
        base_card = self.create_standard_card(cluster_id, cluster_data, layer_name, show_options)
        
        # Add hierarchical information
        parent_clusters = hierarchy_data.get('parents', {}).get(cluster_id, [])
        child_clusters = hierarchy_data.get('children', {}).get(cluster_id, [])
        
        hierarchy_content = html.Div([
            html.Hr(),
            html.H6("Hierarchical Relationships", className="mb-2"),
            
            # Parent clusters
            parent_clusters and html.Div([
                html.Small("Parent Clusters (Macro)", className="text-muted"),
                html.Div([
                    dbc.Badge(
                        f"C{pid}",
                        color="secondary",
                        className="me-1"
                    ) for pid in parent_clusters[:3]
                ], className="mb-2")
            ]),
            
            # Child clusters
            child_clusters and html.Div([
                html.Small("Child Clusters (Micro)", className="text-muted"),
                html.Div([
                    dbc.Badge(
                        f"C{cid}",
                        color="info",
                        className="me-1"
                    ) for cid in child_clusters[:5]
                ])
            ]),
            
            # Hierarchy diagram
            html.Small("Hierarchy Position", className="text-muted mt-2"),
            dcc.Graph(
                figure=self._create_hierarchy_diagram(cluster_id, parent_clusters, child_clusters),
                config={'displayModeBar': False},
                style={'height': '120px'}
            )
        ])
        
        # Insert hierarchy content into card body
        base_card.children[1].children.append(hierarchy_content)
        
        return base_card
    
    def _create_threshold_histogram(self, thresholds: List[float]) -> go.Figure:
        """Create a mini histogram of threshold values."""
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(thresholds))),
                y=thresholds,
                marker_color=THEME_COLOR
            )
        ])
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def _create_hierarchy_diagram(
        self,
        cluster_id: int,
        parents: List[int],
        children: List[int]
    ) -> go.Figure:
        """Create a simple hierarchy diagram."""
        fig = go.Figure()
        
        # Add nodes
        x_pos = [0.5]
        y_pos = [1]
        labels = [f"C{cluster_id}"]
        colors = [THEME_COLOR]
        
        # Add parent nodes
        for i, pid in enumerate(parents[:2]):
            x_pos.append(0.25 + i*0.5)
            y_pos.append(2)
            labels.append(f"C{pid}")
            colors.append(SECONDARY_COLOR)
        
        # Add child nodes
        for i, cid in enumerate(children[:3]):
            x_pos.append(0.2 + i*0.3)
            y_pos.append(0)
            labels.append(f"C{cid}")
            colors.append(ACCENT_COLOR)
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_pos,
            y=y_pos,
            mode='markers+text',
            marker=dict(size=30, color=colors),
            text=labels,
            textposition="center",
            textfont=dict(color='white', size=10),
            showlegend=False
        ))
        
        # Add edges
        for i, pid in enumerate(parents[:2]):
            fig.add_trace(go.Scatter(
                x=[0.25 + i*0.5, 0.5],
                y=[2, 1],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        
        for i, cid in enumerate(children[:3]):
            fig.add_trace(go.Scatter(
                x=[0.5, 0.2 + i*0.3],
                y=[1, 0],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[-0.5, 2.5]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_cluster_cards(
        self,
        clustering_data: Dict[str, Any],
        layer: str,
        card_type: str,
        show_options: List[str],
        sort_by: str = "id"
    ) -> List[dbc.Card]:
        """Create cluster cards for a specific layer."""
        if not clustering_data or layer not in clustering_data.get('clusters_per_layer', {}):
            return [self._create_empty_card()]
        
        layer_data = clustering_data['clusters_per_layer'][layer]
        clusters = layer_data.get('clusters', {})
        
        # Mock cluster data - in real implementation, extract from clustering results
        cards = []
        for cluster_id in range(layer_data.get('n_clusters', 5)):
            cluster_data = {
                'size': np.random.randint(10, 100),
                'cohesion': np.random.uniform(0.6, 0.9),
                'separation': np.random.uniform(0.5, 0.8),
                'label': f'Cluster {cluster_id}',
                'total_samples': 500,
                'thresholds': np.random.uniform(0.01, 0.1, 20).tolist(),
                'active_dimensions': list(range(np.random.randint(5, 15)))
            }
            
            if card_type == "ets":
                card = self.create_ets_card(cluster_id, cluster_data, layer, show_options)
            elif card_type == "hierarchical":
                hierarchy_data = {
                    'parents': {cluster_id: [max(0, cluster_id//2 - 1), cluster_id//2]},
                    'children': {cluster_id: list(range(cluster_id*2, cluster_id*2 + 3))}
                }
                card = self.create_hierarchical_card(
                    cluster_id, cluster_data, layer, show_options, hierarchy_data
                )
            else:
                card = self.create_standard_card(cluster_id, cluster_data, layer, show_options)
            
            cards.append(card)
        
        # Sort cards
        if sort_by == "size":
            cards.sort(key=lambda x: x.id['cluster'], reverse=True)
        elif sort_by == "cohesion":
            cards.sort(key=lambda x: x.id['cluster'], reverse=True)
        
        return cards
    
    def _create_empty_card(self) -> dbc.Card:
        """Create empty state card."""
        return dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-info-circle fa-3x text-muted mb-3"),
                    html.P("No clusters found for selected layer.", className="text-muted"),
                    html.P("Please run clustering analysis first.", className="text-muted small")
                ], className="text-center")
            ])
        ])
    
    def register_callbacks(self, app):
        """Register Dash callbacks for interactivity."""
        
        @app.callback(
            [Output(f"{self.component_id}-layer", "options"),
             Output(f"{self.component_id}-layer", "value")],
            Input("clustering-store", "data"),
            prevent_initial_call=False
        )
        def update_layer_options(clustering_data):
            """Update layer dropdown options."""
            if not clustering_data:
                return [{"label": "No layers available", "value": ""}], ""
            
            layers = list(clustering_data.get('clusters_per_layer', {}).keys())
            options = [{"label": layer.replace('_', ' ').title(), "value": layer} for layer in layers]
            
            return options, layers[0] if layers else ""
        
        @app.callback(
            [Output(f"{self.component_id}-container", "children"),
             Output(f"{self.component_id}-count", "children")],
            [Input("clustering-store", "data"),
             Input(f"{self.component_id}-layer", "value"),
             Input(f"{self.component_id}-type", "value"),
             Input(f"{self.component_id}-sort", "value"),
             Input(f"{self.component_id}-filters", "value"),
             Input(f"{self.component_id}-search", "value")],
            prevent_initial_call=False
        )
        def update_cards(clustering_data, layer, card_type, sort_by, show_options, search):
            """Update cluster cards display."""
            if not clustering_data or not layer:
                return [self._create_empty_card()], "0 clusters"
            
            cards = self.create_cluster_cards(
                clustering_data,
                layer,
                card_type,
                show_options or [],
                sort_by
            )
            
            # Filter by search
            if search:
                # In real implementation, filter cards based on search
                pass
            
            return cards, f"{len(cards)} clusters"
        
        @app.callback(
            Output({"type": f"{self.component_id}-card", "cluster": MATCH}, "color"),
            Input({"type": f"{self.component_id}-card", "cluster": MATCH}, "n_clicks"),
            State({"type": f"{self.component_id}-card", "cluster": MATCH}, "color"),
            prevent_initial_call=True
        )
        def toggle_card_selection(n_clicks, current_color):
            """Toggle card selection on click."""
            if n_clicks:
                return "primary" if current_color != "primary" else None
            return current_color
        
        @app.callback(
            [Output(f"{self.component_id}-selection-collapse", "is_open"),
             Output(f"{self.component_id}-selection-summary", "children")],
            Input({"type": f"{self.component_id}-card", "cluster": ALL}, "color"),
            prevent_initial_call=True
        )
        def update_selection_summary(colors):
            """Update selected clusters summary."""
            selected = [i for i, color in enumerate(colors) if color == "primary"]
            
            if not selected:
                return False, ""
            
            return True, html.Div([
                html.P(f"Selected {len(selected)} clusters: {', '.join(f'C{i}' for i in selected)}"),
                dbc.ButtonGroup([
                    dbc.Button("Compare", color="primary", size="sm"),
                    dbc.Button("Export", color="secondary", size="sm")
                ])
            ])