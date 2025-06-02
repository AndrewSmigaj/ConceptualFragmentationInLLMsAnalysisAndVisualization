"""
Feedforward Networks Analysis Tab for Concept MRI.
Orchestrates components for analyzing generic feedforward networks.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import json
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import callbacks
from .ff_networks_callbacks import register_ff_networks_callbacks

# Import LLM analysis capabilities
try:
    from concept_fragmentation.llm.analysis import ClusterAnalysis
    from local_config import OPENAI_KEY
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ClusterAnalysis = None
    OPENAI_KEY = None

# Import components with error handling
try:
    from concept_mri.components.controls.model_upload import ModelUploadPanel
    from concept_mri.components.controls.dataset_upload import DatasetUploadPanel
    from concept_mri.components.controls.clustering_panel import ClusteringPanel, register_clustering_panel_callbacks
    from concept_mri.components.controls.api_keys_panel import APIKeysPanel
    from concept_mri.components.controls.layer_window_manager import LayerWindowManager
    from concept_mri.components.controls.window_callbacks import register_window_callbacks
    from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
    from concept_mri.components.visualizations.stepped_trajectory import SteppedTrajectoryVisualization
    from concept_mri.components.visualizations.cluster_cards import ClusterCards
except ImportError as e:
    print(f"Warning: Could not import component: {e}")
    # Define mock classes if imports fail
    ModelUploadPanel = DatasetUploadPanel = ClusteringPanel = APIKeysPanel = SankeyWrapper = None
    LayerWindowManager = SteppedTrajectoryVisualization = ClusterCards = None
    register_clustering_panel_callbacks = register_window_callbacks = None


def create_ff_networks_tab(model_data: Optional[Dict] = None, 
                          clustering_data: Optional[Dict] = None) -> dbc.Container:
    """
    Create the feedforward networks analysis tab.
    
    Args:
        model_data: Stored model data from model upload
        clustering_data: Stored clustering results
        
    Returns:
        Tab content container
    """
    # Check if components are available
    if not all([ModelUploadPanel, DatasetUploadPanel, ClusteringPanel, APIKeysPanel, SankeyWrapper]):
        return _create_error_layout("Some components are not available. Please check imports.")
    
    # Initialize components
    model_upload = ModelUploadPanel()
    dataset_upload = DatasetUploadPanel()
    clustering_panel = ClusteringPanel()
    api_keys_panel = APIKeysPanel()
    layer_window_manager = LayerWindowManager() if LayerWindowManager else None
    sankey_wrapper = SankeyWrapper("ff-sankey")
    stepped_trajectory = SteppedTrajectoryVisualization("ff-stepped") if SteppedTrajectoryVisualization else None
    cluster_cards = ClusterCards("ff-cluster-cards") if ClusterCards else None
    
    # Calculate current progress
    progress = _calculate_progress(model_data, clustering_data)
    
    # Create tab layout
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-network-wired me-2"),
                    "Feedforward Network Analysis"
                ], className="mb-0"),
                html.P("Analyze concept organization in feedforward neural networks", 
                       className="text-muted")
            ])
        ], className="mb-4"),
        
        # Progress tracker
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span(f"Progress: {progress}%", className="mb-2 d-block"),
                    dbc.Progress(
                        id="ff-progress",
                        value=progress,
                        striped=True,
                        animated=progress < 100,
                        color=_get_progress_color(progress),
                        className="mb-3"
                    )
                ])
            ])
        ]),
        
        # Alert area for messages
        html.Div(id="ff-alert-area"),
        
        # Main workflow area
        dbc.Card([
            dbc.CardBody([
                # Workflow steps
                _create_workflow_steps(model_data, clustering_data, 
                                     model_upload, dataset_upload, 
                                     clustering_panel, api_keys_panel,
                                     layer_window_manager, sankey_wrapper,
                                     stepped_trajectory, cluster_cards)
            ])
        ]),
        
        # Hidden stores for state management
        dcc.Store(id="ff-analysis-results", data={}),
        dcc.Store(id="ff-current-step", data=_get_current_step(model_data, clustering_data)),
        
        # Loading overlay
        dcc.Loading(
            id="ff-loading",
            type="default",
            children=html.Div(id="ff-loading-output", style={"display": "none"})
        )
        
    ], fluid=True, className="p-4")


def _create_workflow_steps(model_data, clustering_data, 
                          model_upload, dataset_upload, 
                          clustering_panel, api_keys_panel,
                          layer_window_manager, sankey_wrapper,
                          stepped_trajectory, cluster_cards):
    """Create the workflow steps based on current state."""
    current_step = _get_current_step(model_data, clustering_data)
    
    steps = []
    
    # Step 1: Model Upload
    steps.append(_create_step_card(
        "1. Upload Model",
        model_upload.create_component() if model_upload else _create_placeholder("Model Upload"),
        is_active=(current_step == 1),
        is_complete=(current_step > 1)
    ))
    
    # Step 2: Dataset Upload (only show if model is uploaded)
    if current_step >= 2:
        steps.append(_create_step_card(
            "2. Upload Dataset",
            dataset_upload.create_component() if dataset_upload else _create_placeholder("Dataset Upload"),
            is_active=(current_step == 2),
            is_complete=(current_step > 2)
        ))
    
    # Step 3: Configuration (only show if dataset is uploaded)
    if current_step >= 3:
        config_content = html.Div([
            dbc.Row([
                dbc.Col([
                    layer_window_manager.create_component() if layer_window_manager else _create_placeholder("Layer Window Manager")
                ], width=12)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    api_keys_panel.create_component() if api_keys_panel else _create_placeholder("API Keys")
                ], width=6),
                dbc.Col([
                    clustering_panel.create_component() if clustering_panel else _create_placeholder("Clustering Config")
                ], width=6)
            ])
        ])
        
        steps.append(_create_step_card(
            "3. Configure Analysis",
            config_content,
            is_active=(current_step == 3),
            is_complete=(current_step > 3)
        ))
        
        # Add Run Analysis button if on step 3
        if current_step == 3:
            steps.append(dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="fas fa-play me-2"), "Run Analysis"],
                        id="ff-run-analysis-btn",
                        color="primary",
                        size="lg",
                        className="w-100 mt-3"
                    )
                ], width={"size": 6, "offset": 3})
            ]))
    
    # Step 4: Results (only show if analysis is complete)
    if current_step >= 4:
        results_content = dbc.Tabs([
            dbc.Tab([
                sankey_wrapper.create_component() if sankey_wrapper else _create_placeholder("Sankey Diagram")
            ], label="Concept Flow"),
            dbc.Tab([
                stepped_trajectory.create_component() if stepped_trajectory else _create_placeholder("Stepped Trajectory")
            ], label="Trajectories"),
            dbc.Tab([
                cluster_cards.create_component() if cluster_cards else _create_placeholder("Cluster Cards")
            ], label="Cluster Details"),
            dbc.Tab([
                _create_metrics_dashboard(clustering_data)
            ], label="Metrics"),
            dbc.Tab([
                _create_llm_analysis_panel(clustering_data)
            ], label="LLM Analysis"),
            dbc.Tab([
                _create_export_panel()
            ], label="Export")
        ])
        
        steps.append(_create_step_card(
            "4. Analysis Results",
            results_content,
            is_active=(current_step == 4),
            is_complete=False
        ))
    
    return html.Div(steps)


def _create_step_card(title, content, is_active=False, is_complete=False):
    """Create a card for a workflow step."""
    # Determine styling based on state
    if is_complete:
        header_color = "success"
        icon = "fas fa-check-circle"
    elif is_active:
        header_color = "primary"
        icon = "fas fa-circle"
    else:
        header_color = "secondary"
        icon = "fas fa-circle"
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className=f"{icon} me-2"),
            title
        ], className=f"bg-{header_color} text-white"),
        dbc.CardBody(content) if is_active or is_complete else None
    ], className="mb-3")


def _create_placeholder(component_name):
    """Create a placeholder for missing components."""
    return dbc.Alert(
        f"{component_name} component not available",
        color="warning"
    )


def _create_error_layout(error_message):
    """Create an error layout when components can't be loaded."""
    return dbc.Container([
        dbc.Alert([
            html.H4("Component Loading Error", className="alert-heading"),
            html.P(error_message)
        ], color="danger")
    ])


def _create_metrics_dashboard(clustering_data):
    """Create a simple metrics dashboard."""
    if not clustering_data or not clustering_data.get('completed'):
        return html.Div([
            html.P("No analysis results available yet.", className="text-muted text-center")
        ])
    
    # Extract metrics (using mock data for now)
    metrics = clustering_data.get('metrics', {})
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Clusters"),
                    html.H2(metrics.get('total_clusters', 25))
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Unique Paths"),
                    html.H2(metrics.get('unique_paths', 150))
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Fragmentation"),
                    html.H2(f"{metrics.get('fragmentation', 0.23):.2f}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Stability"),
                    html.H2(f"{metrics.get('stability', 0.87):.2f}")
                ])
            ])
        ], width=3)
    ])


def _create_export_panel():
    """Create export options panel."""
    return dbc.Card([
        dbc.CardBody([
            html.H5("Export Options"),
            dbc.ListGroup([
                dbc.ListGroupItem([
                    html.Div([
                        html.H6("Analysis Report (JSON)"),
                        html.Small("Complete analysis results and metrics")
                    ]),
                    dbc.Button(
                        [html.I(className="fas fa-download me-1"), "Download"],
                        id="ff-export-json",
                        size="sm",
                        color="primary",
                        className="float-end"
                    )
                ]),
                dbc.ListGroupItem([
                    html.Div([
                        html.H6("Cluster Labels (CSV)"),
                        html.Small("Cluster IDs and generated labels")
                    ]),
                    dbc.Button(
                        [html.I(className="fas fa-download me-1"), "Download"],
                        id="ff-export-csv",
                        size="sm",
                        color="primary",
                        className="float-end"
                    )
                ]),
                dbc.ListGroupItem([
                    html.Div([
                        html.H6("Visualizations (HTML)"),
                        html.Small("Interactive charts and diagrams")
                    ]),
                    dbc.Button(
                        [html.I(className="fas fa-download me-1"), "Download"],
                        id="ff-export-html",
                        size="sm",
                        color="primary",
                        className="float-end"
                    )
                ])
            ])
        ])
    ])


def _calculate_progress(model_data, clustering_data):
    """Calculate overall progress percentage."""
    progress = 0
    
    # Check model upload (25%)
    if model_data and model_data.get('model_loaded'):
        progress += 25
        
        # Check dataset upload (25%)
        if model_data.get('dataset'):
            progress += 25
            
            # Check if analysis has been configured (25%)
            if model_data.get('analysis_configured'):
                progress += 25
    
    # Check if clustering is complete (25%)
    if clustering_data and clustering_data.get('completed'):
        progress += 25
    
    return progress


def _get_progress_color(progress):
    """Get progress bar color based on percentage."""
    if progress < 25:
        return "danger"
    elif progress < 50:
        return "warning"
    elif progress < 75:
        return "info"
    elif progress < 100:
        return "primary"
    else:
        return "success"


def _get_current_step(model_data, clustering_data):
    """Determine current workflow step."""
    if not model_data or not model_data.get('model_loaded'):
        return 1  # Model upload
    elif not model_data.get('dataset'):
        return 2  # Dataset upload
    elif not model_data.get('analysis_configured') and not clustering_data:
        return 3  # Configuration
    else:
        return 4  # Results


def _create_llm_analysis_panel(clustering_data):
    """Create LLM analysis panel with category selection and results display."""
    if not LLM_AVAILABLE:
        return dbc.Alert(
            "LLM analysis not available. Please ensure concept_fragmentation and API keys are properly configured.",
            color="warning"
        )
    
    if not clustering_data or not clustering_data.get('completed'):
        return html.Div([
            html.P("No clustering results available. Run clustering analysis first.", 
                   className="text-muted text-center")
        ])
    
    return dbc.Card([
        dbc.CardBody([
            html.H5("LLM-Powered Analysis", className="mb-3"),
            
            # Analysis category selection
            dbc.Card([
                dbc.CardBody([
                    html.H6("Select Analysis Categories"),
                    dbc.Checklist(
                        id="llm-analysis-categories",
                        options=[
                            {"label": "Interpretation - Conceptual paths and transformations", "value": "interpretation"},
                            {"label": "Bias Detection - Demographic routing patterns", "value": "bias"},
                            {"label": "Efficiency - Redundancy and compression opportunities", "value": "efficiency"},
                            {"label": "Robustness - Stability and vulnerability analysis", "value": "robustness"}
                        ],
                        value=["interpretation", "bias"],  # Default selections
                        inline=False
                    )
                ])
            ], className="mb-3"),
            
            # Run analysis button
            dbc.Button(
                [html.I(className="fas fa-brain me-2"), "Run LLM Analysis"],
                id="run-llm-analysis-btn",
                color="primary",
                className="mb-3",
                disabled=False
            ),
            
            # Loading spinner
            dcc.Loading(
                id="llm-analysis-loading",
                children=[
                    # Results display area
                    html.Div(id="llm-analysis-results", className="mt-3")
                ]
            ),
            
            # Store for analysis results
            dcc.Store(id="llm-analysis-store")
        ])
    ])


# Add callback for LLM analysis
def register_llm_analysis_callback(app):
    """Register the LLM analysis callback with the app."""
    @app.callback(
        [Output("llm-analysis-results", "children"),
         Output("llm-analysis-store", "data")],
        [Input("run-llm-analysis-btn", "n_clicks")],
        [State("llm-analysis-categories", "value"),
         State("clustering-store", "data")],
        prevent_initial_call=True
    )
    def run_llm_analysis(n_clicks, selected_categories, clustering_data):
        """Run LLM analysis on clustering results."""
        if not n_clicks or not clustering_data or not LLM_AVAILABLE:
            raise PreventUpdate
        
        # Initialize analyzer
        try:
        analyzer = ClusterAnalysis(
            provider="openai",
            api_key=OPENAI_KEY,
            model="gpt-4",
            use_cache=True
        )
    except Exception as e:
        return dbc.Alert(f"Error initializing LLM: {str(e)}", color="danger"), None
    
    # Extract data from clustering results
    # TODO: Ensure clustering_data has the correct format
    paths = clustering_data.get('paths', {})
    cluster_labels = clustering_data.get('cluster_labels', {})
    path_demographic_info = clustering_data.get('path_demographic_info', {})
    fragmentation_scores = clustering_data.get('fragmentation_scores', {})
    
    if not paths:
        return dbc.Alert("No paths found in clustering results.", color="warning"), None
    
    try:
        # Run comprehensive analysis
        analysis_text = analyzer.generate_path_narratives_sync(
            paths=paths,
            cluster_labels=cluster_labels,
            path_demographic_info=path_demographic_info,
            fragmentation_scores=fragmentation_scores,
            analysis_categories=selected_categories
        )
        
        # Parse and display results
        result_components = []
        
        # Split analysis by sections
        sections = analysis_text.split('\n\n')
        current_section = None
        current_content = []
        
        for line in analysis_text.split('\n'):
            if line.strip().endswith(':') and line.strip().upper() in ['INTERPRETATION:', 'BIAS ANALYSIS:', 'EFFICIENCY:', 'ROBUSTNESS:']:
                # Save previous section
                if current_section and current_content:
                    result_components.append(
                        dbc.Card([
                            dbc.CardHeader(html.H6(current_section)),
                            dbc.CardBody([
                                html.P(content) for content in current_content
                            ])
                        ], className="mb-3")
                    )
                # Start new section
                current_section = line.strip()[:-1]
                current_content = []
            elif line.strip() and current_section:
                current_content.append(line.strip())
        
        # Add last section
        if current_section and current_content:
            result_components.append(
                dbc.Card([
                    dbc.CardHeader(html.H6(current_section)),
                    dbc.CardBody([
                        html.P(content) for content in current_content
                    ])
                ], className="mb-3")
            )
        
        # If no sections were parsed, just display the full text
        if not result_components:
            result_components = [
                dbc.Card([
                    dbc.CardBody([
                        html.Pre(analysis_text, style={"whiteSpace": "pre-wrap"})
                    ])
                ])
            ]
        
        # Add export button
        result_components.append(
            dbc.Button(
                [html.I(className="fas fa-download me-2"), "Export Analysis"],
                id="export-llm-analysis-btn",
                color="secondary",
                size="sm",
                className="mt-2"
            )
        )
        
        return result_components, {"analysis": analysis_text, "categories": selected_categories}
        
    except Exception as e:
        return dbc.Alert(f"Error during analysis: {str(e)}", color="danger"), None