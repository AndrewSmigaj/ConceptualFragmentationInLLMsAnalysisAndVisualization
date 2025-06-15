"""
Main layout component for Concept MRI application.
"""
from dash import dcc, html
import dash_bootstrap_components as dbc
from concept_mri.config.settings import APP_NAME, APP_SUBTITLE, THEME_COLOR

def create_header():
    """Create the application header with clinical styling."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.NavbarBrand([
                        html.I(className="fas fa-brain me-2", style={'color': THEME_COLOR}),
                        html.Span(APP_NAME, className="fw-bold"),
                        html.Span(f" | {APP_SUBTITLE}", className="text-muted small ms-2")
                    ], className="fs-4")
                ], width=6),
                dbc.Col([
                    dbc.Nav([
                        dbc.NavItem(
                            dbc.Button(
                                [html.I(className="fas fa-cog me-2"), "Settings"],
                                id="settings-button",
                                color="light",
                                className="me-2",
                                size="sm"
                            )
                        ),
                        dbc.NavItem(
                            dbc.Button(
                                [html.I(className="fas fa-question-circle me-2"), "Help"],
                                id="help-button",
                                color="light",
                                size="sm"
                            )
                        )
                    ], className="ms-auto", navbar=True)
                ], width=6, className="text-end")
            ], className="w-100")
        ], fluid=True),
        color="white",
        dark=False,
        className="shadow-sm mb-4"
    )

def create_tabs():
    """Create the main tab navigation."""
    return dcc.Tabs(
        id='main-tabs',
        value='ff-tab',
        children=[
            dcc.Tab(
                label='FF Networks',
                value='ff-tab',
                className='custom-tab',
                selected_className='custom-tab-selected'
            ),
            dcc.Tab(
                label='GPT (Coming Soon)',
                value='gpt-tab',
                className='custom-tab',
                selected_className='custom-tab-selected',
                disabled=True
            )
        ],
        className="mb-3"
    )

def create_storage_components():
    """Create storage components for state management."""
    import uuid
    from datetime import datetime
    
    # Generate a unique session ID
    session_id = f"session_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return html.Div([
        # Session ID store - this identifies the current session
        dcc.Store(id='session-id-store', storage_type='session', data=session_id),
        # Session storage for model data
        dcc.Store(id='model-store', storage_type='session'),
        # Session storage for dataset info
        dcc.Store(id='dataset-store', storage_type='session'),
        # Session storage for clustering results
        dcc.Store(id='clustering-store', storage_type='session'),
        # Session storage for window configuration
        dcc.Store(id='window-config-store', storage_type='session'),
        # Session storage for path analysis results
        dcc.Store(id='path-analysis-store', storage_type='session'),
        # Session storage for cluster labels
        dcc.Store(id='cluster-labels-store', storage_type='session'),
        # Session storage for hierarchy results
        dcc.Store(id='hierarchy-results-store', storage_type='session'),
        # Local storage for LLM API keys
        dcc.Store(id='api-keys-store', storage_type='local'),
        # Memory storage for temporary UI state
        dcc.Store(id='ui-state-store', storage_type='memory'),
        # Download component for exports
        dcc.Download(id='download-sankey'),
        # Interval for periodic tasks (if needed)
        dcc.Interval(id='interval-component', interval=60*1000, disabled=True),
        # Hidden alert for activation status (used by callbacks)
        dbc.Alert(id='activation-status', is_open=False, dismissable=True, style={'display': 'none'})
    ])

def create_modals():
    """Create modal dialogs for settings and help."""
    settings_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Settings")),
        dbc.ModalBody([
            html.P("Settings panel will be implemented here."),
            html.P("This will include API key configuration and preferences.")
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-settings", className="ms-auto")
        ])
    ], id="settings-modal", is_open=False)
    
    help_modal = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Help & Documentation")),
        dbc.ModalBody([
            html.H5("Getting Started"),
            html.P("1. Upload your neural network model (.pt, .pth, .onnx)"),
            html.P("2. Upload your dataset (.csv, .npz, .pkl)"),
            html.P("3. Configure clustering parameters"),
            html.P("4. Explore the visualizations!"),
            html.Hr(),
            html.H5("About Concept MRI"),
            html.P(
                "Concept MRI provides insights into how neural networks organize "
                "concepts through their layers using Concept Trajectory Analysis (CTA)."
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-help", className="ms-auto")
        ])
    ], id="help-modal", is_open=False)
    
    return html.Div([settings_modal, help_modal])

def create_footer():
    """Create a simple footer."""
    return html.Footer([
        html.Hr(className="mt-5"),
        html.Div([
            html.P([
                "Concept MRI v1.0.0 | ",
                html.A("Documentation", href="#", className="text-muted"),
                " | ",
                html.A("GitHub", href="#", className="text-muted")
            ], className="text-center text-muted small")
        ], className="mb-3")
    ])

def create_layout():
    """Create the complete application layout."""
    return html.Div([
        # Storage components
        create_storage_components(),
        
        # Header
        create_header(),
        
        # Main container
        dbc.Container([
            # Tabs
            create_tabs(),
            
            # Tab content container
            html.Div(id='tab-content', className="mt-4"),
            
            # Loading overlay
            dcc.Loading(
                id="loading-overlay",
                type="circle",
                color=THEME_COLOR,
                children=html.Div(id="loading-output")
            ),
            
            # Footer
            create_footer()
        ], fluid=True),
        
        # Modals
        create_modals()
    ], className="min-vh-100 d-flex flex-column")