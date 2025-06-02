"""
Workflow Manager for Concept MRI.

Manages the multi-step analysis workflow with proper state management
and component persistence.
"""
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from typing import Dict, Any, Optional, Tuple, List
from enum import IntEnum


class WorkflowStep(IntEnum):
    """Enumeration of workflow steps."""
    MODEL_UPLOAD = 1
    DATASET_UPLOAD = 2
    CONFIGURATION = 3
    RESULTS = 4


class WorkflowManager:
    """
    Manages the multi-step analysis workflow.
    
    This class ensures components maintain their state throughout the workflow
    by keeping them mounted and controlling visibility via CSS.
    """
    
    def __init__(self, components: Dict[str, Any], llm_available: bool = False):
        """
        Initialize the workflow manager.
        
        Args:
            components: Dictionary containing initialized components:
                - model_upload: ModelUploadPanel instance
                - dataset_upload: DatasetUploadPanel instance
                - clustering_panel: ClusteringPanel instance
                - layer_window_manager: LayerWindowManager instance (optional)
                - sankey_wrapper: SankeyWrapper instance
                - stepped_trajectory: SteppedTrajectory instance (optional)
                - cluster_cards: ClusterCards instance (optional)
            llm_available: Whether LLM analysis is available
        """
        self.components = components
        self.llm_available = llm_available
        
    def create_layout(self) -> html.Div:
        """
        Create the workflow layout with all steps.
        
        All components are rendered but visibility is controlled via CSS.
        """
        return html.Div([
            # Workflow state store
            dcc.Store(id='workflow-state', data={'current_step': WorkflowStep.MODEL_UPLOAD}),
            
            # Progress indicator
            html.Div(id='workflow-progress', children=self._create_progress_bar(0)),
            
            # Step containers - all rendered, visibility controlled by callbacks
            html.Div([
                # Step 1: Model Upload (always visible)
                self._create_step_container(
                    step_id='step-1-model',
                    title="Step 1: Upload Model",
                    content=self.components['model_upload'].create_component(),
                    icon="fas fa-brain"
                ),
                
                # Step 2: Dataset Upload
                self._create_step_container(
                    step_id='step-2-dataset',
                    title="Step 2: Upload Dataset",
                    content=self.components['dataset_upload'].create_component(),
                    icon="fas fa-database",
                    initial_style={'display': 'none'}
                ),
                
                # Step 3: Configuration
                self._create_step_container(
                    step_id='step-3-config',
                    title="Step 3: Configure Analysis",
                    content=self._create_config_content(),
                    icon="fas fa-cogs",
                    initial_style={'display': 'none'}
                ),
                
                # Step 4: Results
                self._create_step_container(
                    step_id='step-4-results',
                    title="Step 4: Analysis Results",
                    content=self._create_results_content(),
                    icon="fas fa-chart-line",
                    initial_style={'display': 'none'}
                )
            ])
        ])
    
    def _create_step_container(self, step_id: str, title: str, content: Any, 
                             icon: str, initial_style: Optional[Dict] = None) -> html.Div:
        """Create a container for a workflow step."""
        style = initial_style or {}
        
        return html.Div([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className=f"{icon} me-2"),
                    html.Span(title),
                    html.I(className="fas fa-check-circle text-success ms-auto", 
                          id=f"{step_id}-check", style={'display': 'none'})
                ], className="d-flex align-items-center"),
                dbc.CardBody(content)
            ], className="mb-3")
        ], id=step_id, style=style)
    
    def _create_config_content(self) -> html.Div:
        """Create the configuration step content."""
        content = [
            dbc.Row([
                dbc.Col([
                    self.components['clustering_panel'].create_component()
                ], width=12)
            ])
        ]
        
        # Add layer window manager if available
        if self.components.get('layer_window_manager'):
            content.insert(0, dbc.Row([
                dbc.Col([
                    self.components['layer_window_manager'].create_component()
                ], width=12, className="mb-3")
            ]))
            
        return html.Div(content)
    
    def _create_results_content(self) -> html.Div:
        """Create the results step content."""
        # Create tabs for different result views
        return html.Div([
            dbc.Tabs(id="results-tabs", active_tab="sankey-tab", children=[
                dbc.Tab(label="Concept Flow", tab_id="sankey-tab"),
                dbc.Tab(label="Trajectories", tab_id="trajectory-tab"),
                dbc.Tab(label="Cluster Details", tab_id="clusters-tab"),
                dbc.Tab(label="LLM Analysis", tab_id="llm-tab")
            ]),
            # Tab content container
            html.Div(id="results-tab-content", className="mt-3")
        ])
    
    def _create_llm_analysis_placeholder(self) -> html.Div:
        """Create placeholder for LLM analysis panel."""
        if not self.llm_available:
            return dbc.Alert(
                "LLM analysis not available. Please ensure concept_fragmentation and API keys are properly configured.",
                color="warning"
            )
        return dbc.Alert(
            "Run clustering first to enable LLM analysis.",
            color="info"
        )
    
    def _create_progress_bar(self, progress: int) -> html.Div:
        """Create a progress bar component."""
        return html.Div([
            html.Span(f"Progress: {progress}%", className="mb-2 d-block"),
            dbc.Progress(
                value=progress,
                color="success" if progress == 100 else "primary",
                className="mb-3",
                style={"height": "20px"}
            )
        ])
    
    def register_callbacks(self, app):
        """Register all workflow-related callbacks."""
        
        @app.callback(
            [Output('workflow-state', 'data'),
             Output('workflow-progress', 'children'),
             # Step visibility outputs
             Output('step-2-dataset', 'style'),
             Output('step-3-config', 'style'),
             Output('step-4-results', 'style'),
             # Step completion indicators
             Output('step-1-model-check', 'style'),
             Output('step-2-dataset-check', 'style'),
             Output('step-3-config-check', 'style')],
            [Input('model-store', 'data'),
             Input('dataset-store', 'data'),
             Input('clustering-store', 'data')],
            State('workflow-state', 'data')
        )
        def update_workflow_state(model_data, dataset_data, clustering_data, current_state):
            """
            Update workflow state based on available data.
            This callback manages the entire workflow progression.
            """
            # Determine current step and progress
            current_step = WorkflowStep.MODEL_UPLOAD
            progress = 0
            
            # Check model upload
            has_model = model_data and model_data.get('model_loaded')
            if has_model:
                current_step = WorkflowStep.DATASET_UPLOAD
                progress = 25
            
            # Check dataset upload
            has_dataset = dataset_data and dataset_data.get('filename')
            if has_model and has_dataset:
                current_step = WorkflowStep.CONFIGURATION
                progress = 50
            
            # Check clustering completion
            has_clustering = clustering_data and clustering_data.get('completed')
            if has_model and has_dataset and has_clustering:
                current_step = WorkflowStep.RESULTS
                progress = 100
            
            # Update workflow state
            new_state = {'current_step': current_step}
            
            # Create progress bar
            progress_bar = self._create_progress_bar(progress)
            
            # Step visibility styles
            show = {'display': 'block'}
            hide = {'display': 'none'}
            
            # Determine what to show/hide
            show_dataset = show if current_step >= WorkflowStep.DATASET_UPLOAD else hide
            show_config = show if current_step >= WorkflowStep.CONFIGURATION else hide
            show_results = show if current_step >= WorkflowStep.RESULTS else hide
            
            # Completion indicators
            model_check = show if has_model else hide
            dataset_check = show if has_dataset else hide
            config_check = show if has_clustering else hide
            
            return (
                new_state,
                progress_bar,
                show_dataset,
                show_config,
                show_results,
                model_check,
                dataset_check,
                config_check
            )
        
        # Results tab content callback
        @app.callback(
            Output('results-tab-content', 'children'),
            Input('results-tabs', 'active_tab'),
            State('clustering-store', 'data'),
            prevent_initial_call=False
        )
        def render_results_tab_content(active_tab, clustering_data):
            """
            Render content for the selected results tab.
            """
            if not clustering_data or not clustering_data.get('completed'):
                return dbc.Alert(
                    "No results available. Please run clustering analysis first.",
                    color="info"
                )
            
            # Render appropriate component based on selected tab
            if active_tab == 'sankey-tab':
                if self.components.get('sankey_wrapper'):
                    return self.components['sankey_wrapper'].create_component()
                else:
                    return dbc.Alert("Sankey visualization not available", color="warning")
                    
            elif active_tab == 'trajectory-tab':
                if self.components.get('umap_trajectory'):
                    return self.components['umap_trajectory'].create_component()
                elif self.components.get('stepped_trajectory'):
                    return self.components['stepped_trajectory'].create_component()
                else:
                    return dbc.Alert("Trajectory visualization not available", color="warning")
                    
            elif active_tab == 'clusters-tab':
                if self.components.get('cluster_cards'):
                    return self.components['cluster_cards'].create_component()
                else:
                    return dbc.Alert("Cluster details not available", color="warning")
                    
            elif active_tab == 'llm-tab':
                return self._create_llm_analysis_panel()
            
            return html.Div("Select a tab to view results.")
    
    def _create_llm_analysis_panel(self) -> html.Div:
        """Create LLM analysis panel with category selection and results display."""
        if not self.llm_available:
            return dbc.Alert(
                "LLM analysis not available. Please ensure concept_fragmentation and API keys are properly configured.",
                color="warning"
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