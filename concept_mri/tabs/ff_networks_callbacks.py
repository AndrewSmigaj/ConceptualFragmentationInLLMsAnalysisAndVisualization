"""
Callbacks for the Feedforward Networks tab.
"""
from dash import Output, Input, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc


def register_ff_networks_callbacks(app, LLM_AVAILABLE, ClusterAnalysis, OPENAI_KEY):
    """Register callbacks for the FF Networks tab."""
    
    @app.callback(
        [Output("llm-analysis-results", "children"),
         Output("llm-analysis-store", "data")],
        [Input("run-llm-analysis-btn", "n_clicks")],
        [State("llm-analysis-categories", "value"),
         State("clustering-store", "data")],
        prevent_initial_call=True
    )
    def run_llm_analysis(n_clicks, selected_categories, clustering_data):
        """Run LLM analysis on clustering results."""
        if not n_clicks or not clustering_data or not LLM_AVAILABLE:
            raise PreventUpdate
        
        # Initialize analyzer
        try:
            analyzer = ClusterAnalysis(
                provider="openai",
                api_key=OPENAI_KEY,
                model="gpt-4",
                use_cache=True
            )
        except Exception as e:
            return dbc.Alert(f"Error initializing LLM: {str(e)}", color="danger"), None
        
        # Extract data from clustering results
        paths = clustering_data.get('paths', {})
        cluster_labels = clustering_data.get('cluster_labels', {})
        path_demographic_info = clustering_data.get('path_demographic_info', {})
        fragmentation_scores = clustering_data.get('fragmentation_scores', {})
        
        if not paths:
            return dbc.Alert("No paths found in clustering results.", color="warning"), None
        
        try:
            # Run comprehensive analysis
            analysis_text = analyzer.generate_path_narratives_sync(
                paths=paths,
                cluster_labels=cluster_labels,
                path_demographic_info=path_demographic_info,
                fragmentation_scores=fragmentation_scores,
                analysis_categories=selected_categories
            )
            
            # Parse and display results
            result_components = []
            
            # Split analysis by sections
            current_section = None
            current_content = []
            
            for line in analysis_text.split('\n'):
                if line.strip().endswith(':') and line.strip().upper() in ['INTERPRETATION:', 'BIAS ANALYSIS:', 'EFFICIENCY:', 'ROBUSTNESS:']:
                    # Save previous section
                    if current_section and current_content:
                        result_components.append(
                            dbc.Card([
                                dbc.CardHeader(html.H6(current_section)),
                                dbc.CardBody([
                                    html.P(content) for content in current_content
                                ])
                            ], className="mb-3")
                        )
                    # Start new section
                    current_section = line.strip()[:-1]
                    current_content = []
                elif line.strip() and current_section:
                    current_content.append(line.strip())
            
            # Add last section
            if current_section and current_content:
                result_components.append(
                    dbc.Card([
                        dbc.CardHeader(html.H6(current_section)),
                        dbc.CardBody([
                            html.P(content) for content in current_content
                        ])
                    ], className="mb-3")
                )
            
            # If no sections were parsed, just display the full text
            if not result_components:
                result_components = [
                    dbc.Card([
                        dbc.CardBody([
                            html.Pre(analysis_text, style={"whiteSpace": "pre-wrap"})
                        ])
                    ])
                ]
            
            # Add export button
            result_components.append(
                dbc.Button(
                    [html.I(className="fas fa-download me-2"), "Export Analysis"],
                    id="export-llm-analysis-btn",
                    color="secondary",
                    size="sm",
                    className="mt-2"
                )
            )
            
            return result_components, {"analysis": analysis_text, "categories": selected_categories}
            
        except Exception as e:
            return dbc.Alert(f"Error during analysis: {str(e)}", color="danger"), None