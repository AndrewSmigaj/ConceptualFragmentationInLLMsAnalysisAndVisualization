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

# Import workflow manager
from concept_mri.components.workflow_manager import WorkflowManager

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
    components = {
        'model_upload': ModelUploadPanel(),
        'dataset_upload': DatasetUploadPanel(),
        'clustering_panel': ClusteringPanel(),
        'api_keys_panel': APIKeysPanel(),
        'layer_window_manager': LayerWindowManager() if LayerWindowManager else None,
        'sankey_wrapper': SankeyWrapper("ff-sankey"),
        'stepped_trajectory': SteppedTrajectoryVisualization("ff-stepped") if SteppedTrajectoryVisualization else None,
        'cluster_cards': ClusterCards("ff-cluster-cards") if ClusterCards else None
    }
    
    # Initialize workflow manager
    workflow_manager = WorkflowManager(components, llm_available=LLM_AVAILABLE)
    
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
        
        # Alert area for messages
        html.Div(id="ff-alert-area"),
        
        # Main workflow area - managed by WorkflowManager
        workflow_manager.create_layout(),
        
        # Hidden stores for state management
        dcc.Store(id="ff-analysis-results", data={}),
        
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
                ], width=12, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    clustering_panel.create_component() if clustering_panel else _create_placeholder("Clustering Config")
                ], width=12)
            ])
        ])
        
        steps.append(_create_step_card(
            "3. Configure Analysis",
            config_content,
            is_active=(current_step == 3),
            is_complete=(current_step > 3)
        ))
    
    # Step 4: Results (only show if clustering is complete)
    if clustering_data and clustering_data.get('completed'):
        results_content = _create_results_section(clustering_data, sankey_wrapper, 
                                                 stepped_trajectory, cluster_cards)
        steps.append(_create_step_card(
            "4. Analysis Results",
            results_content,
            is_active=True,
            is_complete=True
        ))
    
    return html.Div(steps)


def _create_step_card(title, content, is_active=False, is_complete=False):
    """Create a styled step card."""
    # Determine styling based on state
    if is_complete:
        border_color = "success"
        header_bg = "success"
        text_color = "white"
        icon = "fas fa-check-circle"
    elif is_active:
        border_color = "primary"
        header_bg = "primary"
        text_color = "white"
        icon = "fas fa-circle-notch fa-spin"
    else:
        border_color = "secondary"
        header_bg = "light"
        text_color = "dark"
        icon = "fas fa-circle"
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className=f"{icon} me-2"),
            title
        ], className=f"bg-{header_bg} text-{text_color}"),
        dbc.CardBody(content)
    ], className=f"mb-3 border-{border_color}", style={"borderWidth": "2px"})


def _create_results_section(clustering_data, sankey_wrapper, stepped_trajectory, cluster_cards):
    """Create the results visualization section."""
    return dbc.Tabs([
        dbc.Tab(label="Concept Flow", tab_id="sankey-tab"),
        dbc.Tab(label="Trajectories", tab_id="trajectory-tab"),
        dbc.Tab(label="Cluster Details", tab_id="clusters-tab"),
        dbc.Tab(label="Metrics", tab_id="metrics-tab"),
        dbc.Tab(label="LLM Analysis", tab_id="llm-tab")
    ], id="results-tabs", active_tab="sankey-tab", children=[
        dbc.Card([
            dbc.CardBody([
                html.Div(id="results-tab-content", children=[
                    # Sankey diagram (default view)
                    sankey_wrapper.create_component() if sankey_wrapper else _create_placeholder("Sankey Diagram")
                ])
            ])
        ], className="mt-3")
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


def _create_placeholder(name):
    """Create a placeholder component."""
    return dbc.Alert(
        f"{name} component not available",
        color="warning",
        className="text-center"
    )


def _create_error_layout(message):
    """Create an error layout."""
    return dbc.Container([
        dbc.Alert([
            html.H4("Error Loading Tab", className="alert-heading"),
            html.P(message)
        ], color="danger")
    ], className="p-4")


def _create_llm_analysis_panel(clustering_data):
    """Create LLM analysis panel with category selection and results display."""
    if not LLM_AVAILABLE:
        return dbc.Alert(
            "LLM analysis not available. Please ensure concept_fragmentation and API keys are properly configured.",
            color="warning"
        )
    
    # Check if clustering data has paths
    if not clustering_data or not clustering_data.get('paths'):
        return dbc.Alert(
            "No clustering results available. Please run clustering first.",
            color="info"
        )
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-brain me-2"),
            "LLM-Powered Analysis"
        ]),
        dbc.CardBody([
            # Analysis category selection
            html.Div([
                html.H6("Select Analysis Categories:"),
                dbc.Checklist(
                    id="llm-analysis-categories",
                    options=[
                        {"label": "Interpretation - Conceptual understanding of paths", "value": "interpretation"},
                        {"label": "Bias Detection - Demographic routing patterns", "value": "bias"},
                        {"label": "Efficiency - Computational considerations", "value": "efficiency"},
                        {"label": "Robustness - Stability analysis", "value": "robustness"}
                    ],
                    value=["interpretation", "bias"],  # Default selections
                    inline=False,
                    className="mb-3"
                )
            ]),
            
            # Run analysis button
            dbc.Button(
                [html.I(className="fas fa-play me-2"), "Run LLM Analysis"],
                id="run-llm-analysis-btn",
                color="primary",
                size="lg",
                className="mb-3"
            ),
            
            # Results display area
            html.Div(id="llm-analysis-results"),
            
            # Store for analysis results
            dcc.Store(id="llm-analysis-store")
        ])
    ])


# Export the callback registration function
def get_ff_networks_callbacks():
    """Get callback registration function with closure over module variables."""
    def register_callbacks(app):
        # Register workflow callbacks
        # We need to create a workflow manager instance for callbacks
        components = {
            'model_upload': ModelUploadPanel() if ModelUploadPanel else None,
            'dataset_upload': DatasetUploadPanel() if DatasetUploadPanel else None,
            'clustering_panel': ClusteringPanel() if ClusteringPanel else None,
            'api_keys_panel': APIKeysPanel() if APIKeysPanel else None,
            'layer_window_manager': LayerWindowManager() if LayerWindowManager else None,
            'sankey_wrapper': SankeyWrapper("ff-sankey") if SankeyWrapper else None,
            'stepped_trajectory': SteppedTrajectoryVisualization("ff-stepped") if SteppedTrajectoryVisualization else None,
            'cluster_cards': ClusterCards("ff-cluster-cards") if ClusterCards else None
        }
        workflow_manager = WorkflowManager(components, llm_available=LLM_AVAILABLE)
        workflow_manager.register_callbacks(app)
        
        # Register LLM analysis callbacks
        register_ff_networks_callbacks(app, LLM_AVAILABLE, ClusterAnalysis, OPENAI_KEY)
    return register_callbacks