"""
Model upload component for Concept MRI.
"""
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import dash_uploader as du
from pathlib import Path
import json

from concept_mri.config.settings import (
    ALLOWED_MODEL_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
    UPLOAD_FOLDER_ROOT, THEME_COLOR
)


class ModelUploadPanel:
    """Model upload panel component."""
    
    def __init__(self):
        """Initialize the model upload panel."""
        self.id_prefix = "model"
    
    def create_component(self):
        """Create and return the component layout."""
        return create_model_upload()

def create_model_upload():
    """Create the model upload interface."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-brain me-2"),
            "Model Upload"
        ], className="fw-bold"),
        dbc.CardBody([
            # Upload area
            du.Upload(
                id='model-upload',
                max_file_size=MAX_UPLOAD_SIZE_MB,
                filetypes=ALLOWED_MODEL_EXTENSIONS,
                upload_id='model-uploader',
                text='Drag and drop or click to upload your model',
                text_completed='Model uploaded: ',
                default_style={
                    'minHeight': 150,
                    'borderWidth': 2,
                    'borderStyle': 'dashed',
                    'borderColor': THEME_COLOR,
                    'borderRadius': 10,
                    'textAlign': 'center',
                    'padding': '2rem',
                    'backgroundColor': '#f8f9fa'
                },
                max_files=1
            ),
            
            # Model info display
            html.Div(id='model-info', className='mt-3'),
            
            # Validation feedback
            dbc.Alert(
                "Please upload a model file to begin analysis.",
                id='model-alert',
                color='info',
                is_open=True,
                className='mt-3'
            )
        ])
    ])

def create_model_info_display(model_data):
    """Create a display of model information."""
    if not model_data:
        return None
    
    return dbc.Card([
        dbc.CardHeader("Model Architecture"),
        dbc.CardBody([
            html.H6(f"File: {model_data.get('filename', 'Unknown')}"),
            html.P(f"Type: {model_data.get('type', 'Unknown')}", className='mb-1'),
            html.P(f"Size: {model_data.get('size_mb', 0):.2f} MB", className='mb-1'),
            
            html.Hr(),
            
            html.H6("Architecture Summary:"),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Small("Layers:", className='text-muted'),
                        html.P(f"{model_data.get('num_layers', 'Unknown')}", className='fw-bold mb-0')
                    ], width=4),
                    dbc.Col([
                        html.Small("Parameters:", className='text-muted'),
                        html.P(f"{model_data.get('num_params', 'Unknown'):,}", className='fw-bold mb-0')
                    ], width=4),
                    dbc.Col([
                        html.Small("Input Dim:", className='text-muted'),
                        html.P(f"{model_data.get('input_dim', 'Unknown')}", className='fw-bold mb-0')
                    ], width=4)
                ])
            ]),
            
            # Layer details
            html.Div([
                html.H6("Layer Details:", className='mt-3'),
                html.Div(id='layer-details')
            ]) if model_data.get('layers') else None
        ])
    ], className='mt-3', color='light')

def register_model_upload_callbacks(app):
    """Register callbacks for model upload component."""
    
    @app.callback(
        [Output('model-store', 'data'),
         Output('model-info', 'children'),
         Output('model-alert', 'children'),
         Output('model-alert', 'color'),
         Output('model-alert', 'is_open')],
        Input('model-upload', 'isCompleted'),
        State('model-upload', 'fileNames'),
        State('model-upload', 'upload_id'),
        prevent_initial_call=True
    )
    def handle_model_upload(is_completed, filenames, upload_id):
        """Handle model file upload."""
        if not is_completed or not filenames:
            return None, None, "Upload failed. Please try again.", "danger", True
        
        # Get uploaded file path
        filename = filenames[0]
        file_path = Path(UPLOAD_FOLDER_ROOT) / upload_id / filename
        
        # Here we would normally call model_interface.py to load the model
        # For now, create mock data
        model_data = {
            'filename': filename,
            'path': str(file_path),
            'type': 'PyTorch' if filename.endswith('.pt') or filename.endswith('.pth') else 'Unknown',
            'size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
            'num_layers': 5,  # Mock data
            'num_params': 125000,  # Mock data
            'input_dim': 13,  # Mock data
            'layers': [  # Mock data
                {'name': 'fc1', 'type': 'Linear', 'shape': '(13, 64)'},
                {'name': 'fc2', 'type': 'Linear', 'shape': '(64, 32)'},
                {'name': 'fc3', 'type': 'Linear', 'shape': '(32, 16)'},
                {'name': 'fc4', 'type': 'Linear', 'shape': '(16, 8)'},
                {'name': 'output', 'type': 'Linear', 'shape': '(8, 2)'}
            ]
        }
        
        info_display = create_model_info_display(model_data)
        
        return (
            model_data,
            info_display,
            f"Model '{filename}' loaded successfully!",
            "success",
            True
        )
    
    @app.callback(
        Output('layer-details', 'children'),
        Input('model-store', 'data'),
        prevent_initial_call=True
    )
    def display_layer_details(model_data):
        """Display detailed layer information."""
        if not model_data or not model_data.get('layers'):
            return None
        
        rows = []
        for i, layer in enumerate(model_data['layers']):
            rows.append(
                html.Tr([
                    html.Td(f"Layer {i+1}"),
                    html.Td(layer['name']),
                    html.Td(layer['type']),
                    html.Td(layer['shape'])
                ])
            )
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Index"),
                    html.Th("Name"),
                    html.Th("Type"),
                    html.Th("Shape")
                ])
            ]),
            html.Tbody(rows)
        ], size='sm', hover=True, className='mb-0')