"""
Concept MRI - Neural Network Analysis Tool
Main application entry point.
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from concept_mri.config.settings import (
    APP_NAME, APP_VERSION, DEBUG, PORT, HOST,
    THEME_COLOR, SECRET_KEY
)

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

# Import layout after app initialization
from concept_mri.components.layouts.main_layout import create_layout

# Set the app layout
app.layout = create_layout()

# Tab content callback
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
    State('model-store', 'data'),
    State('clustering-store', 'data')
)
def render_tab_content(active_tab, model_data, clustering_data):
    """Render content based on active tab."""
    if active_tab == 'ff-tab':
        # Import here to avoid circular imports
        from concept_mri.tabs.ff_networks import create_ff_networks_tab
        return create_ff_networks_tab(model_data, clustering_data)
    elif active_tab == 'gpt-tab':
        from concept_mri.tabs.gpt_placeholder import create_gpt_placeholder
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
    # Import callback registration functions
    from concept_mri.components.controls.clustering_panel import register_clustering_panel_callbacks
    from concept_mri.components.controls.window_callbacks import register_window_callbacks
    from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
    from concept_mri.components.visualizations.stepped_trajectory import SteppedTrajectoryVisualization
    from concept_mri.components.visualizations.cluster_cards import ClusterCards
    
    # Register control callbacks
    register_clustering_panel_callbacks(app)
    register_window_callbacks(app)
    
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