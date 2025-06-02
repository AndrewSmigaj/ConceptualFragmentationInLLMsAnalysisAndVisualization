"""
Concept MRI - Neural Network Analysis Tool
Main application entry point.
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash_uploader as du
import logging
import os
from datetime import datetime
from pathlib import Path

from concept_mri.config.settings import (
    APP_NAME, APP_VERSION, DEBUG, PORT, HOST,
    THEME_COLOR, SECRET_KEY
)

# Set up logging - create a simple debug log
log_dir = Path("logs/concept_mri")
log_dir.mkdir(parents=True, exist_ok=True)

# Create debug log filename
debug_log = log_dir / f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create a specific logger for our debug messages
debug_logger = logging.getLogger('concept_mri.debug')
debug_logger.setLevel(logging.DEBUG)

# Create handler for debug file only
debug_handler = logging.FileHandler(debug_log)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
debug_logger.addHandler(debug_handler)

# Also print to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
debug_logger.addHandler(console_handler)

# Don't propagate to root logger to avoid noise
debug_logger.propagate = False

logger = debug_logger
logger.info(f"=== Concept MRI Debug Log Started ===")
logger.info(f"Debug log file: {debug_log}")

# Make debug logger available to other modules
def get_debug_logger():
    """Get the debug logger for Concept MRI."""
    return logging.getLogger('concept_mri.debug')

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
    ],
    suppress_callback_exceptions=True,
    title=APP_NAME,
    update_title="Loading..."
)

# Set secret key for sessions
app.server.secret_key = SECRET_KEY

# Configure dash-uploader
from concept_mri.config.settings import UPLOAD_FOLDER_ROOT
du.configure_upload(app, UPLOAD_FOLDER_ROOT, use_upload_id=True)

# Import layout after app initialization
from concept_mri.components.layouts.main_layout import create_layout

# Import tab creators at top level to avoid dynamic import issues
from concept_mri.tabs.ff_networks import create_ff_networks_tab
from concept_mri.tabs.gpt_placeholder import create_gpt_placeholder

# Import activation manager for session cleanup
from concept_mri.core.activation_manager import activation_manager

# Set the app layout
app.layout = create_layout()

# Tab content callback - only re-render when switching tabs
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(active_tab):
    """Render content based on active tab."""
    logger.debug(f"Rendering tab: {active_tab}")
    if active_tab == 'ff-tab':
        # Pass None for data since WorkflowManager will read from stores
        return create_ff_networks_tab(None, None)
    elif active_tab == 'gpt-tab':
        return create_gpt_placeholder()
    else:
        return html.Div("Select a tab to begin.")

# Modal callbacks
@app.callback(
    Output('settings-modal', 'is_open'),
    [Input('settings-button', 'n_clicks'),
     Input('close-settings', 'n_clicks')],
    State('settings-modal', 'is_open')
)
def toggle_settings_modal(n1, n2, is_open):
    """Toggle settings modal."""
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output('help-modal', 'is_open'),
    [Input('help-button', 'n_clicks'),
     Input('close-help', 'n_clicks')],
    State('help-modal', 'is_open')
)
def toggle_help_modal(n1, n2, is_open):
    """Toggle help modal."""
    if n1 or n2:
        return not is_open
    return is_open

# Callback registry
def register_all_callbacks(app):
    """Register all component callbacks."""
    logger.info("Starting callback registration...")
    # Import callback registration functions
    from concept_mri.components.controls.clustering_panel import register_clustering_panel_callbacks
    from concept_mri.components.controls.window_callbacks import register_window_callbacks
    from concept_mri.components.controls.model_upload import register_model_upload_callbacks
    from concept_mri.components.controls.dataset_upload import register_dataset_upload_callbacks
    from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
    from concept_mri.components.visualizations.stepped_trajectory import SteppedTrajectoryVisualization
    from concept_mri.components.visualizations.cluster_cards import ClusterCards
    from concept_mri.tabs.ff_networks import get_ff_networks_callbacks
    from concept_mri.components.callbacks.activation_extraction_callback import register_activation_extraction_callback
    
    # Register control callbacks
    register_clustering_panel_callbacks(app)
    register_window_callbacks(app)
    register_model_upload_callbacks(app)
    register_dataset_upload_callbacks(app)
    
    # Register activation extraction callback
    register_activation_extraction_callback(app)
    
    # Register tab callbacks
    ff_callbacks = get_ff_networks_callbacks()
    ff_callbacks(app)
    
    # Register visualization callbacks
    sankey = SankeyWrapper("ff-sankey")
    sankey.register_callbacks(app)
    
    stepped = SteppedTrajectoryVisualization("ff-stepped")
    stepped.register_callbacks(app)
    
    cluster_cards = ClusterCards("ff-cluster-cards")
    cluster_cards.register_callbacks(app)

# Register all callbacks
register_all_callbacks(app)

# Run the app
if __name__ == '__main__':
    print(f"""
    ╔════════════════════════════════════════╗
    ║         🏥 Concept MRI v{APP_VERSION}        ║
    ║     Neural Network Analysis Tool       ║
    ╚════════════════════════════════════════╝
    
    Starting server at http://{HOST}:{PORT}
    Debug mode: {DEBUG}
    """)
    
    app.run_server(
        debug=DEBUG,
        host=HOST,
        port=PORT
    )