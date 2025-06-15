"""
Callbacks for VisualizationPanel component.

Handles:
- Visualization type switching
- Visualization updates based on clustering data
- Control updates (color scheme, layout, highlight)
- Integration with existing Sankey and Trajectory components
"""

from dash import callback, Input, Output, State, ctx
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go

from ..visualization_panel import VisualizationPanel
from ..stores_config import get_store_id

# Create instance for helper methods
viz_panel = VisualizationPanel()

def register_visualization_panel_callbacks(app):
    """Register callbacks for the VisualizationPanel component."""
    
    @app.callback(
        [Output("viz-panel-sankey-btn", "active"),
         Output("viz-panel-trajectory-btn", "active"),
         Output("viz-panel-3d-btn", "active"),
         Output("viz-panel-current-type", "data")],
        [Input("viz-panel-sankey-btn", "n_clicks"),
         Input("viz-panel-trajectory-btn", "n_clicks"),
         Input("viz-panel-3d-btn", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_viz_type_selection(sankey_clicks, trajectory_clicks, threeds_clicks):
        """Handle visualization type selection."""
        triggered_id = ctx.triggered_id
        
        if not triggered_id:
            return True, False, False, "sankey"
        
        if "sankey" in triggered_id:
            return True, False, False, "sankey"
        elif "trajectory" in triggered_id:
            return False, True, False, "trajectory"
        elif "3d" in triggered_id:
            return False, False, True, "3d"
        
        return True, False, False, "sankey"
    
    @app.callback(
        Output("viz-panel-content", "children"),
        [Input("viz-panel-current-type", "data"),
         Input("network-explorer-clustering-results-store", "data"),
         Input("network-explorer-window-selection-store", "data")],
        [State("viz-panel-color-scheme", "value"),
         State("viz-panel-layout", "value"),
         State("viz-panel-highlight", "value"),
         State("viz-panel-toggles", "value")]
    )
    def update_visualization_content(viz_type, clustering_results, window_selection,
                                   color_scheme, layout, highlight, toggles):
        """Update visualization content based on type and data."""
        if not clustering_results or not clustering_results.get('completed'):
            return viz_panel._create_placeholder()
        
        # Check toggles
        show_all = "show_all" in toggles if toggles else False
        normalize = "normalize" in toggles if toggles else True
        compare = "compare" in toggles if toggles else False
        
        if viz_type == "sankey":
            # For now, return the container - will be integrated with actual Sankey
            return viz_panel.create_sankey_container()
        elif viz_type == "trajectory":
            # For now, return the container - will be integrated with actual Trajectory
            return viz_panel.create_trajectory_container()
        else:
            return viz_panel._create_placeholder()
    
    @app.callback(
        Output("viz-panel-highlight", "options"),
        Input("network-explorer-paths-store", "data")
    )
    def update_highlight_options(paths_data):
        """Update highlight dropdown options based on available paths."""
        if not paths_data or 'paths' not in paths_data:
            return [{"label": "None", "value": "none"}]
        
        return viz_panel.update_highlight_options(paths_data['paths'])
    
    # Placeholder callbacks for actual visualizations
    # These will be replaced when integrating with existing components
    
    @app.callback(
        Output("viz-panel-sankey-graph", "figure", allow_duplicate=True),
        [Input(get_store_id("clustering_results"), "data"),
         Input(get_store_id("window_config"), "data"),
         Input("viz-panel-color-scheme", "value"),
         Input(get_store_id("highlight"), "data")],
        [State(get_store_id("paths_analysis"), "data")],
        prevent_initial_call=True
    )
    def update_sankey_visualization(clustering_results, window_config, 
                                  color_scheme, highlight_data, paths_data):
        """Update Sankey diagram based on clustering results and highlighting."""
        if not clustering_results or not clustering_results.get('completed'):
            return viz_panel._create_empty_sankey()
        
        # Import and use the actual SankeyWrapper
        from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
        
        # Create SankeyWrapper instance
        sankey_wrapper = SankeyWrapper()
        
        # Extract data from clustering results
        paths = clustering_results.get('paths', {})
        cluster_labels = clustering_results.get('cluster_labels', {})
        
        if not paths:
            return viz_panel._create_empty_sankey()
        
        # Configure based on visualization settings
        config = {
            'top_n': 25,  # Show top 25 paths
            'color_by': color_scheme,
            'window': window_config.get('current_window', 'full') if window_config else 'full',
            'hierarchy': 'meso'  # Default to meso level
        }
        
        # Use window config directly (already in correct format)
        window_name = window_config.get('current_window', 'full') if window_config else 'full'
        
        # Generate Sankey using the wrapper
        try:
            # Pass the correct window name
            if window_name != 'full':
                config['window'] = window_name
            
            fig_dict = sankey_wrapper.generate_sankey(
                clustering_data=clustering_results,
                path_data=paths,
                cluster_labels=cluster_labels,
                window_config=window_config,
                config=config
            )
            
            # Convert dict back to figure
            fig = go.Figure(fig_dict)
            
            # Apply cross-component highlighting
            if highlight_data and highlight_data.get('highlight_type'):
                highlighted_ids = highlight_data.get('highlight_ids', [])
                highlight_color = highlight_data.get('color', 'orange')
                
                # Modify Sankey to highlight specific paths or clusters
                if 'data' in fig_dict and len(fig_dict['data']) > 0:
                    sankey_trace = fig_dict['data'][0]
                    
                    # Highlight nodes (clusters)
                    if 'node' in sankey_trace:
                        node_colors = []
                        for i, label in enumerate(sankey_trace['node'].get('label', [])):
                            # Check if this node should be highlighted
                            is_highlighted = any(
                                h_id in label for h_id in highlighted_ids
                            )
                            if is_highlighted:
                                node_colors.append(highlight_color)
                            else:
                                # Use default color
                                node_colors.append(sankey_trace['node'].get('color', ['#1f77b4'])[i] if i < len(sankey_trace['node'].get('color', [])) else '#1f77b4')
                        
                        sankey_trace['node']['color'] = node_colors
                    
                    # Highlight links (paths)
                    if 'link' in sankey_trace and highlight_data.get('highlight_type') == 'path':
                        link_colors = sankey_trace['link'].get('color', [])
                        # For path highlighting, we'd need to identify which links belong to the highlighted path
                        # This requires more complex logic based on the path structure
            
            # Adjust layout for the panel
            fig.update_layout(
                height=None,  # Let it use container height
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        except Exception as e:
            print(f"Error generating Sankey: {e}")
            import traceback
            traceback.print_exc()
            return viz_panel._create_empty_sankey()
    
    @app.callback(
        Output("viz-panel-trajectory-graph", "figure", allow_duplicate=True),
        [Input(get_store_id("clustering_results"), "data"),
         Input(get_store_id("window_config"), "data"),
         Input("viz-panel-color-scheme", "value"),
         Input("viz-panel-toggles", "value"),
         Input(get_store_id("highlight"), "data")],
        [State("model-store", "data")],
        prevent_initial_call=True
    )
    def update_trajectory_visualization(clustering_results, window_config,
                                      color_scheme, toggles, highlight_data, model_data):
        """Update trajectory visualization based on clustering results and highlighting."""
        if not clustering_results or not clustering_results.get('completed'):
            return viz_panel._create_empty_trajectory()
        
        # Import and use the actual UMAPTrajectoryVisualization
        from concept_mri.components.visualizations.umap_trajectory import UMAPTrajectoryVisualization
        
        # Create trajectory visualizer instance
        trajectory_viz = UMAPTrajectoryVisualization()
        
        # Check if we have model data with activations
        if not model_data:
            return viz_panel._create_empty_trajectory()
        
        # Prepare options based on toggles
        show_options = []
        if toggles:
            if "normalize" in toggles:
                show_options.append("normalize")
            # Map panel toggles to trajectory options
            show_options.extend(["arrows", "paths"])  # Default to showing arrows and paths
        else:
            show_options = ["arrows", "paths", "normalize"]
        
        # Determine sample size based on window
        n_samples = 50  # Default
        if window_selection and window_selection.get('type') != 'full':
            # Use fewer samples for window views
            n_samples = 30
        
        # Add window filtering to config
        trajectory_config = {
            'n_samples': n_samples,
            'n_neighbors': 15,
            'color_by': color_scheme if color_scheme in ['cluster', 'index', 'magnitude'] else 'cluster',
            'show_options': show_options
        }
        
        # Add window range if not full view
        if window_config and window_config.get('current_window') != 'full':
            current_window = window_config['current_window']
            if current_window in window_config.get('windows', {}):
                window = window_config['windows'][current_window]
                trajectory_config['layer_range'] = (window['start'], window['end'] + 1)
        
        # Generate trajectory using the existing component
        try:
            fig_dict = trajectory_viz.generate_umap_trajectory(
                model_data=model_data,
                clustering_data=clustering_results,
                config=trajectory_config
            )
            
            # Convert dict to figure
            fig = go.Figure(fig_dict)
            
            # Apply window filtering
            if window_config and window_config.get('current_window') != 'full':
                current_window = window_config['current_window']
                if current_window in window_config.get('windows', {}):
                    window = window_config['windows'][current_window]
                    start_layer = window['start']
                    end_layer = window['end']
                    
                    fig.update_layout(
                        title=f"Activation Trajectories - {current_window.title()} Window (Layers {start_layer}-{end_layer})"
                    )
            
            # Apply highlighting
            if highlight_data and highlight_data.get('highlight_type') == 'sample':
                highlighted_ids = highlight_data.get('highlight_ids', [])
                highlight_color = highlight_data.get('color', '#2ca02c')
                
                # Update trace colors for highlighted samples
                if 'data' in fig.to_dict():
                    for trace in fig.data:
                        if hasattr(trace, 'marker') and hasattr(trace.marker, 'color'):
                            # Check if this trace represents highlighted samples
                            # This would require matching sample IDs
                            pass
            
            # Adjust layout for the panel
            fig.update_layout(
                height=None,  # Let it use container height
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            print(f"Error generating trajectory: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to a 2D representation if 3D fails
            try:
                # Simple 2D trajectory as fallback
                from sklearn.decomposition import PCA
                
                # Get activations
                activations = trajectory_viz._get_activations_from_model_data(model_data)
                if not activations:
                    return viz_panel._create_empty_trajectory()
                
                # Get first layer for simple 2D view
                first_layer_key = sorted(activations.keys())[0]
                first_layer_acts = activations[first_layer_key][:50]  # Limit samples
                
                # Reduce to 2D
                if first_layer_acts.shape[1] > 2:
                    pca = PCA(n_components=2)
                    reduced = pca.fit_transform(first_layer_acts)
                else:
                    reduced = first_layer_acts[:, :2]
                
                # Create simple 2D scatter
                fig = go.Figure(data=go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=list(range(len(reduced))),
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=[f"Sample {i}" for i in range(len(reduced))],
                    hoverinfo='text'
                ))
                
                fig.update_layout(
                    title="Activation Space (2D Projection)",
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                return fig
                
            except Exception as e2:
                print(f"Error in fallback trajectory: {e2}")
                return viz_panel._create_empty_trajectory()