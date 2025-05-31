"""
Placeholder for GPT analysis tab.
"""
from dash import html
import dash_bootstrap_components as dbc


def create_gpt_placeholder():
    """Create placeholder content for GPT tab."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-brain fa-3x text-muted mb-3"),
                    html.H3("GPT Analysis Coming Soon", className="text-muted"),
                    html.P(
                        "This tab will provide specialized analysis tools for GPT and other transformer models.",
                        className="text-muted"
                    )
                ], className="text-center mt-5")
            ])
        ])
    ], className="p-4")