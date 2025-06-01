"""
Dataset upload component for Concept MRI.
"""


class DatasetUploadPanel:
    """Dataset upload panel component."""
    
    def __init__(self):
        """Initialize the dataset upload panel."""
        self.id_prefix = "dataset"
    
    def create_component(self):
        """Create and return the component layout."""
        return create_dataset_upload()
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import dash_uploader as du
from pathlib import Path
import pandas as pd
import numpy as np
import json

from concept_mri.config.settings import (
    ALLOWED_DATA_EXTENSIONS, MAX_UPLOAD_SIZE_MB,
    UPLOAD_FOLDER_ROOT, THEME_COLOR
)

def create_dataset_upload():
    """Create the dataset upload interface."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-database me-2"),
            "Dataset Upload"
        ], className="fw-bold"),
        dbc.CardBody([
            # Upload area
            du.Upload(
                id='dataset-upload',
                max_file_size=MAX_UPLOAD_SIZE_MB,
                filetypes=ALLOWED_DATA_EXTENSIONS,
                upload_id='dataset-uploader',
                text='Drag and drop or click to upload your dataset',
                text_completed='Dataset uploaded: ',
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
            
            # Dataset info display
            html.Div(id='dataset-info', className='mt-3'),
            
            # Feature mapping section
            html.Div(id='feature-mapping-section', className='mt-3'),
            
            # Validation feedback
            dbc.Alert(
                "Please upload a dataset file to begin analysis.",
                id='dataset-alert',
                color='info',
                is_open=True,
                className='mt-3'
            )
        ])
    ])

def create_dataset_info_display(dataset_data):
    """Create a display of dataset information."""
    if not dataset_data:
        return None
    
    return dbc.Card([
        dbc.CardHeader("Dataset Information"),
        dbc.CardBody([
            html.H6(f"File: {dataset_data.get('filename', 'Unknown')}"),
            
            dbc.Row([
                dbc.Col([
                    html.Small("Samples:", className='text-muted'),
                    html.P(f"{dataset_data.get('num_samples', 0):,}", className='fw-bold mb-0')
                ], width=4),
                dbc.Col([
                    html.Small("Features:", className='text-muted'),
                    html.P(f"{dataset_data.get('num_features', 0)}", className='fw-bold mb-0')
                ], width=4),
                dbc.Col([
                    html.Small("Size:", className='text-muted'),
                    html.P(f"{dataset_data.get('size_mb', 0):.2f} MB", className='fw-bold mb-0')
                ], width=4)
            ]),
            
            html.Hr(),
            
            # Data preview
            html.H6("Data Preview:"),
            html.Div(id='data-preview-container', className='mb-3'),
            
            # Feature statistics
            html.Details([
                html.Summary("Feature Statistics", className='fw-bold mb-2'),
                html.Div(id='feature-stats-container')
            ], open=False)
        ])
    ], className='mt-3', color='light')

def create_feature_mapping_interface(features):
    """Create interface for mapping feature names."""
    if not features:
        return None
    
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-tags me-2"),
            "Feature Name Mapping"
        ]),
        dbc.CardBody([
            html.P("Provide human-readable names for your features:", className='text-muted'),
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dbc.Label(f"Feature {i} ({feat}):", html_for=f"feature-name-{i}")
                    ], width=6),
                    dbc.Col([
                        dbc.Input(
                            id=f"feature-name-{i}",
                            type="text",
                            placeholder=f"Enter name for {feat}",
                            value=feat,
                            size="sm"
                        )
                    ], width=6)
                ], className='mb-2') for i, feat in enumerate(features[:10])  # Show first 10
            ]),
            
            # Show remaining count if more than 10
            html.Small(f"... and {len(features) - 10} more features", className='text-muted') 
            if len(features) > 10 else None,
            
            dbc.Button(
                "Save Feature Names",
                id='save-feature-names',
                color='primary',
                size='sm',
                className='mt-3'
            )
        ])
    ], className='mt-3')

def register_dataset_upload_callbacks(app):
    """Register callbacks for dataset upload component."""
    
    @app.callback(
        [Output('dataset-store', 'data'),
         Output('dataset-info', 'children'),
         Output('feature-mapping-section', 'children'),
         Output('dataset-alert', 'children'),
         Output('dataset-alert', 'color'),
         Output('dataset-alert', 'is_open')],
        Input('dataset-upload', 'isCompleted'),
        State('dataset-upload', 'fileNames'),
        State('dataset-upload', 'upload_id'),
        prevent_initial_call=True
    )
    def handle_dataset_upload(is_completed, filenames, upload_id):
        """Handle dataset file upload."""
        if not is_completed or not filenames:
            return None, None, None, "Upload failed. Please try again.", "danger", True
        
        # Get uploaded file path
        filename = filenames[0]
        file_path = Path(UPLOAD_FOLDER_ROOT) / upload_id / filename
        
        # Mock data loading (would normally load actual data)
        dataset_data = {
            'filename': filename,
            'path': str(file_path),
            'num_samples': 303,  # Mock data
            'num_features': 13,  # Mock data
            'size_mb': file_path.stat().st_size / (1024 * 1024) if file_path.exists() else 0,
            'features': [f'feature_{i}' for i in range(13)],  # Mock features
            'preview': {  # Mock preview data
                'columns': [f'feature_{i}' for i in range(5)] + ['target'],
                'data': [
                    [63, 1, 3, 145, 233, 0],
                    [37, 1, 2, 130, 250, 0],
                    [41, 0, 1, 130, 204, 0],
                    [56, 1, 1, 120, 236, 1],
                    [57, 0, 0, 120, 354, 1]
                ]
            }
        }
        
        info_display = create_dataset_info_display(dataset_data)
        feature_mapping = create_feature_mapping_interface(dataset_data['features'])
        
        return (
            dataset_data,
            info_display,
            feature_mapping,
            f"Dataset '{filename}' loaded successfully!",
            "success",
            True
        )
    
    @app.callback(
        Output('data-preview-container', 'children'),
        Input('dataset-store', 'data'),
        prevent_initial_call=True
    )
    def display_data_preview(dataset_data):
        """Display preview of the dataset."""
        if not dataset_data or 'preview' not in dataset_data:
            return "No preview available"
        
        preview = dataset_data['preview']
        
        return dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in preview['columns']],
            data=[dict(zip(preview['columns'], row)) for row in preview['data']],
            style_cell={'textAlign': 'center'},
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': THEME_COLOR,
                'color': 'white',
                'fontWeight': 'bold'
            },
            page_size=5
        )
    
    @app.callback(
        Output('feature-stats-container', 'children'),
        Input('dataset-store', 'data'),
        prevent_initial_call=True
    )
    def display_feature_stats(dataset_data):
        """Display feature statistics."""
        if not dataset_data:
            return "No statistics available"
        
        # Mock statistics
        stats = [
            dbc.Row([
                dbc.Col(html.Small(f"Feature {i}:", className='text-muted'), width=3),
                dbc.Col(html.Small(f"Mean: {np.random.randn():.2f}"), width=3),
                dbc.Col(html.Small(f"Std: {np.random.rand():.2f}"), width=3),
                dbc.Col(html.Small(f"Range: [{np.random.randint(0, 50)}, {np.random.randint(51, 100)}]"), width=3)
            ], className='mb-1') for i in range(min(5, dataset_data['num_features']))
        ]
        
        return html.Div(stats)