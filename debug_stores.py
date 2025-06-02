"""
Debug script to check store behavior
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import json

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Store Debug Test"),
    
    dbc.Row([
        dbc.Col([
            dbc.Button("Update Model", id="btn-model", color="primary", className="m-2"),
            dbc.Button("Update Dataset", id="btn-dataset", color="success", className="m-2"),
            dbc.Button("Update Combined", id="btn-combined", color="info", className="m-2"),
        ])
    ]),
    
    html.Hr(),
    
    dbc.Row([
        dbc.Col([
            html.H4("Model Store:"),
            html.Pre(id="model-display", style={"backgroundColor": "#f0f0f0", "padding": "10px"})
        ], width=4),
        dbc.Col([
            html.H4("Dataset Store:"),
            html.Pre(id="dataset-display", style={"backgroundColor": "#f0f0f0", "padding": "10px"})
        ], width=4),
        dbc.Col([
            html.H4("Combined Store:"),
            html.Pre(id="combined-display", style={"backgroundColor": "#f0f0f0", "padding": "10px"})
        ], width=4),
    ]),
    
    # Stores
    dcc.Store(id='model-store', storage_type='session'),
    dcc.Store(id='dataset-store', storage_type='session'),
    dcc.Store(id='combined-store', storage_type='session'),
])

# Model store callback
@app.callback(
    Output('model-store', 'data'),
    Input('btn-model', 'n_clicks'),
    State('model-store', 'data'),
    prevent_initial_call=True
)
def update_model(n_clicks, current_data):
    print(f"Model callback triggered. Current data: {current_data}")
    new_data = current_data or {}
    new_data['model_loaded'] = True
    new_data['click_count'] = n_clicks
    return new_data

# Dataset store callback
@app.callback(
    Output('dataset-store', 'data'),
    Input('btn-dataset', 'n_clicks'),
    State('dataset-store', 'data'),
    prevent_initial_call=True
)
def update_dataset(n_clicks, current_data):
    print(f"Dataset callback triggered. Current data: {current_data}")
    new_data = current_data or {}
    new_data['dataset_loaded'] = True
    new_data['click_count'] = n_clicks
    return new_data

# Combined store callback
@app.callback(
    Output('combined-store', 'data'),
    [Input('model-store', 'data'),
     Input('dataset-store', 'data')]
)
def update_combined(model_data, dataset_data):
    print(f"Combined callback triggered. Model: {model_data}, Dataset: {dataset_data}")
    combined = {}
    if model_data:
        combined['model'] = model_data
    if dataset_data:
        combined['dataset'] = dataset_data
    return combined

# Display callbacks
@app.callback(
    Output('model-display', 'children'),
    Input('model-store', 'data')
)
def display_model(data):
    return json.dumps(data, indent=2) if data else "None"

@app.callback(
    Output('dataset-display', 'children'),
    Input('dataset-store', 'data')
)
def display_dataset(data):
    return json.dumps(data, indent=2) if data else "None"

@app.callback(
    Output('combined-display', 'children'),
    Input('combined-store', 'data')
)
def display_combined(data):
    return json.dumps(data, indent=2) if data else "None"

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)