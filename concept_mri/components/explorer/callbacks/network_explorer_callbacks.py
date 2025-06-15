"""
Main callbacks for NetworkExplorer component.

These callbacks coordinate data flow between all child components
and manage the overall state of the Network Explorer.
"""

from dash import callback, Input, Output, State, ctx, ALL
from typing import Dict, Any, Optional, List
import json
import pandas as pd

from ..stores_config import get_store_id, DEFAULT_STORES


def register_network_explorer_callbacks(app):
    """Register main callbacks for the NetworkExplorer component."""
    
    @app.callback(
        [Output(get_store_id("clustering_results"), "data"),
         Output(get_store_id("paths_analysis"), "data"),
         Output("network-explorer-model-name", "children"),
         Output("network-explorer-dataset-name", "children")],
        [Input("clustering-store", "data"),  # From main app
         Input("model-store", "data")],      # From main app
        prevent_initial_call=False
    )
    def initialize_network_explorer(clustering_data, model_data):
        """Initialize Network Explorer with clustering results and model data."""
        # Initialize with defaults
        clustering_results = DEFAULT_STORES["clustering_results"].copy()
        paths_analysis = DEFAULT_STORES["paths_analysis"].copy()
        model_name = "Not loaded"
        dataset_name = "Not loaded"
        
        # Update model info if available
        if model_data:
            model_name = model_data.get("model_name", "Unknown Model")
            dataset_name = model_data.get("dataset_name", "Unknown Dataset")
        
        # Process clustering data if available
        if clustering_data and clustering_data.get("completed"):
            # Update clustering results
            clustering_results.update({
                "completed": True,
                "model_name": model_name,
                "timestamp": pd.Timestamp.now().isoformat(),
                "clusters_per_layer": clustering_data.get("clusters_per_layer", {}),
                "total_samples": clustering_data.get("metrics", {}).get("total_samples", 0),
                "unique_paths": clustering_data.get("metrics", {}).get("unique_paths", 0),
                "cluster_labels": clustering_data.get("cluster_labels", {}),
                "metrics": clustering_data.get("metrics", {}),
                "hierarchy_results": clustering_data.get("hierarchy_results", None)
            })
            
            # Extract and analyze paths
            if "paths" in clustering_data:
                paths_data = clustering_data["paths"]
                
                # Convert paths to our PathData format
                formatted_paths = []
                patterns = {"stable": [], "transitional": [], "fragmented": []}
                
                for path_id, path_info in paths_data.items():
                    # Determine pattern type based on path characteristics
                    pattern = _determine_path_pattern(path_info)
                    
                    formatted_path = {
                        "path_id": path_id,
                        "sequence": path_info.get("sequence", []),
                        "frequency": path_info.get("frequency", 1),
                        "samples": path_info.get("samples", []),
                        "stability": path_info.get("stability", 0.0),
                        "pattern": pattern,
                        "transitions": path_info.get("transitions", [])
                    }
                    
                    formatted_paths.append(formatted_path)
                    patterns[pattern].append(path_id)
                
                # Update paths in clustering results
                clustering_results["paths"] = {p["path_id"]: p for p in formatted_paths}
                
                # Update paths analysis
                paths_analysis.update({
                    "paths": formatted_paths,
                    "patterns": patterns,
                    "statistics": {
                        "total_paths": len(formatted_paths),
                        "pattern_distribution": {
                            pattern: len(paths) for pattern, paths in patterns.items()
                        }
                    }
                })
        
        return clustering_results, paths_analysis, model_name, dataset_name
    
    @app.callback(
        Output(get_store_id("window_config"), "data"),
        [Input("window-config-store", "data")],  # From Layer Window Manager
        State(get_store_id("window_config"), "data"),
        prevent_initial_call=True
    )
    def update_window_configuration(window_manager_config, current_config):
        """Update window configuration from Layer Window Manager."""
        if not window_manager_config:
            return current_config
        
        # Merge with current config
        updated_config = current_config.copy()
        
        if "windows" in window_manager_config:
            updated_config["windows"] = window_manager_config["windows"]
        
        if "current_window" in window_manager_config:
            updated_config["current_window"] = window_manager_config["current_window"]
        
        return updated_config
    
    @app.callback(
        Output(get_store_id("selection"), "data", allow_duplicate=True),
        [Input({"type": "path-card", "index": ALL}, "n_clicks"),
         Input("viz-panel-sankey-graph", "clickData"),
         Input("viz-panel-trajectory-graph", "clickData"),
         Input("network-metrics-chart", "clickData")],
        State(get_store_id("paths_analysis"), "data"),
        prevent_initial_call=True
    )
    def handle_entity_selection(path_clicks, sankey_click, trajectory_click, 
                              metrics_click, paths_data):
        """Handle entity selection from various sources."""
        ctx_triggered = ctx.triggered_id
        
        if not ctx_triggered:
            return DEFAULT_STORES["selection"]
        
        selection_data = DEFAULT_STORES["selection"].copy()
        
        # Handle path card clicks
        if isinstance(ctx_triggered, dict) and ctx_triggered.get("type") == "path-card":
            clicked_index = ctx_triggered["index"]
            if paths_data and clicked_index < len(paths_data["paths"]):
                path = paths_data["paths"][clicked_index]
                selection_data.update({
                    "entity_type": "path",
                    "entity_id": path["path_id"],
                    "entity_data": path,
                    "source_component": "paths_panel"
                })
        
        # Handle Sankey clicks
        elif ctx_triggered == "viz-panel-sankey-graph" and sankey_click:
            # Extract node or link information from click data
            point_data = sankey_click.get("points", [{}])[0]
            if "label" in point_data:  # Node click
                selection_data.update({
                    "entity_type": "cluster",
                    "entity_id": point_data["label"],
                    "entity_data": {
                        "id": point_data["label"],
                        "layer": _extract_layer_from_label(point_data["label"]),
                        "label": point_data.get("customdata", {}).get("semantic_label", ""),
                        "size": point_data.get("value", 0)
                    },
                    "source_component": "sankey"
                })
        
        # Handle trajectory clicks
        elif ctx_triggered == "viz-panel-trajectory-graph" and trajectory_click:
            point_data = trajectory_click.get("points", [{}])[0]
            if "pointIndex" in point_data:
                selection_data.update({
                    "entity_type": "sample",
                    "entity_id": f"sample_{point_data['pointIndex']}",
                    "entity_data": {
                        "id": f"sample_{point_data['pointIndex']}",
                        "text": point_data.get("text", ""),
                        "trajectory": point_data.get("customdata", {}).get("trajectory", []),
                        "metadata": point_data.get("customdata", {})
                    },
                    "source_component": "trajectory"
                })
        
        # Handle metrics chart clicks
        elif ctx_triggered == "network-metrics-chart" and metrics_click:
            point_data = metrics_click.get("points", [{}])[0]
            if "x" in point_data:  # Layer click
                layer_idx = point_data["x"]
                selection_data.update({
                    "entity_type": "layer",
                    "entity_id": f"layer_{layer_idx}",
                    "entity_data": {
                        "id": f"layer_{layer_idx}",
                        "index": layer_idx,
                        "metrics": point_data.get("customdata", {})
                    },
                    "source_component": "metrics_chart"
                })
        
        return selection_data
    
    @app.callback(
        Output(get_store_id("highlight"), "data"),
        [Input(get_store_id("selection"), "data"),
         Input("details-panel-compare-btn", "n_clicks")],
        [State(get_store_id("highlight"), "data"),
         State(get_store_id("comparison"), "data")],
        prevent_initial_call=True
    )
    def update_highlighting(selection_data, compare_clicks, current_highlight, comparison_data):
        """Update cross-component highlighting based on selection."""
        ctx_triggered_id = ctx.triggered_id
        
        # Clear highlighting if no selection
        if not selection_data or not selection_data.get("entity_type"):
            return DEFAULT_STORES["highlight"]
        
        highlight_data = current_highlight.copy() if current_highlight else DEFAULT_STORES["highlight"].copy()
        
        # Handle comparison mode
        if ctx_triggered_id == "details-panel-compare-btn" and comparison_data and comparison_data.get("active"):
            # Highlight all compared entities
            highlight_ids = [e["id"] for e in comparison_data.get("entities", [])]
            highlight_data.update({
                "highlight_type": "comparison",
                "highlight_ids": highlight_ids,
                "source_component": "comparison",
                "color": "orange"
            })
        else:
            # Regular selection highlighting
            entity_type = selection_data["entity_type"]
            entity_id = selection_data["entity_id"]
            
            if entity_type == "path":
                # Highlight the path and its clusters
                path_data = selection_data.get("entity_data", {})
                highlight_ids = [entity_id] + path_data.get("sequence", [])
                highlight_color = "#ff7f0e"  # Orange for paths
            elif entity_type == "cluster":
                # Highlight the cluster and related paths
                highlight_ids = [entity_id]
                highlight_color = "#1f77b4"  # Blue for clusters
            elif entity_type == "sample":
                # Highlight the sample trajectory
                highlight_ids = [entity_id]
                highlight_color = "#2ca02c"  # Green for samples
            else:
                highlight_ids = []
                highlight_color = None
            
            highlight_data.update({
                "highlight_type": entity_type,
                "highlight_ids": highlight_ids,
                "source_component": selection_data.get("source_component"),
                "color": highlight_color
            })
        
        return highlight_data
    
    @app.callback(
        Output("llm-analysis-trigger", "data"),
        Input("llm-analysis-trigger-store", "data"),
        prevent_initial_call=True
    )
    def propagate_llm_trigger(trigger_data):
        """Propagate LLM analysis trigger to main app."""
        return trigger_data


def _determine_path_pattern(path_info: Dict[str, Any]) -> str:
    """Determine the pattern type of a path based on its characteristics.
    
    Args:
        path_info: Path information dictionary
        
    Returns:
        Pattern type: "stable", "transitional", or "fragmented"
    """
    sequence = path_info.get("sequence", [])
    stability = path_info.get("stability", 0)
    
    if not sequence:
        return "fragmented"
    
    # Check for stable pattern (same cluster across layers)
    unique_clusters = len(set(sequence))
    if unique_clusters == 1:
        return "stable"
    
    # Check stability metric
    if stability > 0.7:
        return "stable"
    elif stability > 0.3:
        return "transitional"
    else:
        return "fragmented"


def _extract_layer_from_label(label: str) -> str:
    """Extract layer information from a cluster label.
    
    Args:
        label: Cluster label (e.g., "L0_C1" or "layer_0_cluster_1")
        
    Returns:
        Layer identifier
    """
    if label.startswith("L") and "_" in label:
        # Format: L0_C1
        return label.split("_")[0]
    elif "layer_" in label:
        # Format: layer_0_cluster_1
        parts = label.split("_")
        if len(parts) >= 2:
            return f"layer_{parts[1]}"
    
    return "unknown"