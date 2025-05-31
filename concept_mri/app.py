"""
Concept MRI - Neural Network Analysis Tool
Main application entry point.
"""
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from config.settings import (
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
from components.layouts.main_layout import create_layout

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
        from tabs.ff_networks import create_ff_networks_tab
        return create_ff_networks_tab(model_data, clustering_data)
    elif active_tab == 'gpt-tab':
        from tabs.gpt_placeholder import create_gpt_placeholder
        return create_gpt_placeholder()
    else:
        return html.Div("Select a tab to begin.")

# Callback registry placeholder
def register_callbacks(app):
    """Register all component callbacks."""
    # Import and register callbacks from components
    # This will be populated as we create components
    pass

# Register all callbacks
register_callbacks(app)

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