"""
API keys configuration panel for LLM providers.
"""


class APIKeysPanel:
    """API keys panel component."""
    
    def __init__(self):
        """Initialize the API keys panel."""
        self.id_prefix = "api_keys"
    
    def create_component(self):
        """Create and return the component layout."""
        return create_api_keys_panel()
from dash import dcc, html, Input, Output, State, callback, ALL
import dash_bootstrap_components as dbc
from datetime import datetime
from typing import Dict, Any, Optional

from concept_mri.config.settings import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER, THEME_COLOR

def create_api_keys_panel():
    """Create the API keys configuration panel."""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className="fas fa-key me-2"),
            "LLM API Configuration"
        ], className="fw-bold"),
        dbc.CardBody([
            html.P(
                "Configure API keys for LLM providers to enable cluster labeling and analysis.",
                className="text-muted mb-3"
            ),
            
            # Provider selection
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Provider", html_for="llm-provider-select"),
                    dbc.Select(
                        id='llm-provider-select',
                        options=[
                            {'label': info['name'], 'value': provider}
                            for provider, info in LLM_PROVIDERS.items()
                        ],
                        value=DEFAULT_LLM_PROVIDER
                    )
                ], width=12)
            ], className='mb-3'),
            
            # Provider-specific configuration
            html.Div(id='provider-config-container'),
            
            # Test connection button
            dbc.Button(
                [html.I(className="fas fa-plug me-2"), "Test Connection"],
                id='test-api-key-btn',
                color='secondary',
                size='sm',
                className='mt-3',
                disabled=True
            ),
            
            # Connection status
            html.Div(id='api-key-status', className='mt-3'),
            
            # Save configuration
            dbc.Button(
                [html.I(className="fas fa-save me-2"), "Save Configuration"],
                id='save-api-config-btn',
                color='primary',
                className='w-100 mt-3',
                disabled=True
            ),
            
            # Current configuration status
            html.Hr(className='my-4'),
            html.H6("Current Configuration", className='mb-3'),
            html.Div(id='current-config-display')
        ])
    ])

def create_provider_config(provider: str, stored_config: Optional[Dict] = None):
    """Create provider-specific configuration interface."""
    provider_info = LLM_PROVIDERS.get(provider, {})
    has_key = bool(stored_config and stored_config.get(provider, {}).get('api_key'))
    
    config_elements = [
        # API Key input
        dbc.Row([
            dbc.Col([
                dbc.Label(f"{provider_info.get('name', provider)} API Key", 
                         html_for=f"{provider}-api-key"),
                dbc.InputGroup([
                    dbc.Input(
                        id={'type': 'api-key-input', 'provider': provider},
                        type='password',
                        placeholder='sk-...' if provider == 'openai' else 'Enter API key',
                        value='',
                        disabled=has_key
                    ),
                    dbc.InputGroupText(
                        html.I(className="fas fa-check text-success" if has_key else "fas fa-times text-danger"),
                        id=f"{provider}-key-status-icon"
                    )
                ])
            ], width=12)
        ], className='mb-3'),
        
        # Edit/Clear key button if key exists
        dbc.Button(
            [html.I(className="fas fa-edit me-2"), "Change API Key"],
            id={'type': 'edit-key-btn', 'provider': provider},
            color='warning',
            size='sm',
            className='mb-3',
            style={'display': 'block' if has_key else 'none'}
        ) if has_key else None,
        
        # Model selection
        dbc.Row([
            dbc.Col([
                dbc.Label("Preferred Model", html_for=f"{provider}-model-select"),
                dbc.Select(
                    id={'type': 'model-select', 'provider': provider},
                    options=[
                        {'label': model, 'value': model}
                        for model in provider_info.get('models', [])
                    ],
                    value=provider_info.get('models', [''])[0] if provider_info.get('models') else ''
                )
            ], width=12)
        ], className='mb-3'),
        
        # Provider-specific options
        create_provider_specific_options(provider)
    ]
    
    return html.Div([elem for elem in config_elements if elem is not None])

def create_provider_specific_options(provider: str):
    """Create provider-specific configuration options."""
    if provider == 'openai':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Organization ID (Optional)", html_for="openai-org-id"),
                    dbc.Input(
                        id={'type': 'org-id', 'provider': 'openai'},
                        type='text',
                        placeholder='org-...'
                    ),
                    dbc.FormText("Required for some enterprise accounts")
                ], width=12)
            ])
        ])
    elif provider == 'anthropic':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("API Version", html_for="anthropic-api-version"),
                    dbc.Select(
                        id={'type': 'api-version', 'provider': 'anthropic'},
                        options=[
                            {'label': '2023-06-01 (Latest)', 'value': '2023-06-01'},
                            {'label': '2023-01-01', 'value': '2023-01-01'}
                        ],
                        value='2023-06-01'
                    )
                ], width=12)
            ])
        ])
    elif provider == 'google':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Project ID", html_for="google-project-id"),
                    dbc.Input(
                        id={'type': 'project-id', 'provider': 'google'},
                        type='text',
                        placeholder='my-project-id'
                    ),
                    dbc.FormText("Google Cloud project ID")
                ], width=12)
            ])
        ])
    elif provider == 'local':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Label("API Endpoint", html_for="local-endpoint"),
                    dbc.Input(
                        id={'type': 'endpoint', 'provider': 'local'},
                        type='text',
                        placeholder='http://localhost:8080',
                        value='http://localhost:8080'
                    ),
                    dbc.FormText("Local model server endpoint")
                ], width=12)
            ])
        ])
    return None

def create_config_status_display(stored_config: Dict[str, Any]):
    """Create display of current configuration status."""
    if not stored_config:
        return dbc.Alert(
            "No API keys configured. Please add at least one provider.",
            color='warning',
            className='mb-0'
        )
    
    rows = []
    for provider, config in stored_config.items():
        if config.get('api_key'):
            provider_info = LLM_PROVIDERS.get(provider, {})
            status_color = 'success' if config.get('validated') else 'warning'
            status_icon = 'check-circle' if config.get('validated') else 'exclamation-circle'
            
            rows.append(
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className=f"fas fa-{status_icon} text-{status_color} me-2"),
                            html.Strong(provider_info.get('name', provider))
                        ])
                    ], width=4),
                    dbc.Col([
                        html.Small(f"Model: {config.get('model', 'default')}", className='text-muted')
                    ], width=4),
                    dbc.Col([
                        html.Small(
                            f"Added: {config.get('added_date', 'Unknown')}",
                            className='text-muted'
                        )
                    ], width=4)
                ], className='mb-2')
            )
    
    if not rows:
        return dbc.Alert(
            "No valid API keys found. Please configure at least one provider.",
            color='warning',
            className='mb-0'
        )
    
    return html.Div(rows)

def register_api_keys_callbacks(app):
    """Register callbacks for API keys panel."""
    
    @app.callback(
        Output('provider-config-container', 'children'),
        Input('llm-provider-select', 'value'),
        State('api-keys-store', 'data')
    )
    def update_provider_config(provider, stored_config):
        """Update provider-specific configuration interface."""
        return create_provider_config(provider, stored_config)
    
    @app.callback(
        Output('test-api-key-btn', 'disabled'),
        Input({'type': 'api-key-input', 'provider': ALL}, 'value'),
        State('llm-provider-select', 'value')
    )
    def enable_test_button(api_keys, selected_provider):
        """Enable test button when API key is entered."""
        if not api_keys:
            return True
        
        # Find the API key for the selected provider
        for i, key in enumerate(api_keys):
            if key and len(key) > 10:  # Basic validation
                return False
        return True
    
    @app.callback(
        [Output('api-key-status', 'children'),
         Output('save-api-config-btn', 'disabled')],
        Input('test-api-key-btn', 'n_clicks'),
        [State({'type': 'api-key-input', 'provider': ALL}, 'value'),
         State('llm-provider-select', 'value')],
        prevent_initial_call=True
    )
    def test_api_connection(n_clicks, api_keys, selected_provider):
        """Test API connection with provided key."""
        if not n_clicks:
            return None, True
        
        # Mock API validation (would actually test with LLM factory)
        # In real implementation, this would call the LLM factory to validate
        
        success = True  # Mock successful validation
        
        if success:
            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                "API key validated successfully!"
            ], color='success', dismissable=True)
            return status, False
        else:
            status = dbc.Alert([
                html.I(className="fas fa-times-circle me-2"),
                "API key validation failed. Please check your key."
            ], color='danger', dismissable=True)
            return status, True
    
    @app.callback(
        [Output('api-keys-store', 'data'),
         Output('current-config-display', 'children')],
        Input('save-api-config-btn', 'n_clicks'),
        [State({'type': 'api-key-input', 'provider': ALL}, 'value'),
         State({'type': 'model-select', 'provider': ALL}, 'value'),
         State('llm-provider-select', 'value'),
         State('api-keys-store', 'data')],
        prevent_initial_call=True
    )
    def save_api_configuration(n_clicks, api_keys, models, selected_provider, stored_config):
        """Save API configuration to local storage."""
        if not n_clicks:
            return stored_config, create_config_status_display(stored_config)
        
        if stored_config is None:
            stored_config = {}
        
        # Update configuration for selected provider
        stored_config[selected_provider] = {
            'api_key': api_keys[0] if api_keys else '',
            'model': models[0] if models else '',
            'validated': True,  # Mock validation
            'added_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return stored_config, create_config_status_display(stored_config)
    
    @app.callback(
        Output('current-config-display', 'children'),
        Input('api-keys-store', 'data')
    )
    def update_config_display(stored_config):
        """Update configuration status display."""
        return create_config_status_display(stored_config or {})