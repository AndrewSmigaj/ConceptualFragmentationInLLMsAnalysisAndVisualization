"""
Visualization validation utilities for testing.

This module provides validators for Plotly figures, Dash components, and other
visualization elements used in GPT-2 analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import plotly.graph_objects as go
from dash import html, dcc
import json


class PlotlyFigureValidator:
    """Validate Plotly figure structures and content."""
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
    
    def validate_sankey_diagram(
        self,
        figure: go.Figure,
        expected_nodes: Optional[int] = None,
        expected_links: Optional[int] = None,
        check_node_labels: bool = True,
        check_link_values: bool = True
    ) -> bool:
        """
        Validate Sankey diagram structure.
        
        Args:
            figure: Plotly figure to validate
            expected_nodes: Expected number of nodes
            expected_links: Expected number of links
            check_node_labels: Whether to validate node labels
            check_link_values: Whether to validate link values
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if figure has data
            if not figure.data:
                self.validation_errors.append("Figure has no data")
                return False
            
            # Find Sankey trace
            sankey_trace = None
            for trace in figure.data:
                if trace.type == "sankey":
                    sankey_trace = trace
                    break
            
            if sankey_trace is None:
                self.validation_errors.append("No Sankey trace found in figure")
                return False
            
            # Validate nodes
            if not hasattr(sankey_trace, 'node') or sankey_trace.node is None:
                self.validation_errors.append("Sankey trace has no node data")
                return False
            
            nodes = sankey_trace.node
            
            # Check node count
            if expected_nodes is not None:
                if not hasattr(nodes, 'label') or len(nodes.label) != expected_nodes:
                    self.validation_errors.append(
                        f"Expected {expected_nodes} nodes, got {len(nodes.label) if hasattr(nodes, 'label') else 0}"
                    )
                    return False
            
            # Check node labels if requested
            if check_node_labels and hasattr(nodes, 'label'):
                for i, label in enumerate(nodes.label):
                    if not isinstance(label, str) or not label.strip():
                        self.validation_errors.append(f"Node {i} has invalid label: {label}")
                        return False
            
            # Validate links
            if not hasattr(sankey_trace, 'link') or sankey_trace.link is None:
                self.validation_errors.append("Sankey trace has no link data")
                return False
            
            links = sankey_trace.link
            
            # Check link count
            if expected_links is not None:
                if not hasattr(links, 'source') or len(links.source) != expected_links:
                    self.validation_errors.append(
                        f"Expected {expected_links} links, got {len(links.source) if hasattr(links, 'source') else 0}"
                    )
                    return False
            
            # Check link structure
            required_link_attrs = ['source', 'target', 'value']
            for attr in required_link_attrs:
                if not hasattr(links, attr):
                    self.validation_errors.append(f"Links missing required attribute: {attr}")
                    return False
                
                link_data = getattr(links, attr)
                if not isinstance(link_data, (list, tuple, np.ndarray)):
                    self.validation_errors.append(f"Link {attr} is not a list/array")
                    return False
            
            # Check link values if requested
            if check_link_values and hasattr(links, 'value'):
                for i, value in enumerate(links.value):
                    if not isinstance(value, (int, float)) or value < 0:
                        self.validation_errors.append(f"Link {i} has invalid value: {value}")
                        return False
            
            # Check link indices are valid
            num_nodes = len(nodes.label) if hasattr(nodes, 'label') else 0
            for i, (source, target) in enumerate(zip(links.source, links.target)):
                if not (0 <= source < num_nodes):
                    self.validation_errors.append(f"Link {i} has invalid source index: {source}")
                    return False
                if not (0 <= target < num_nodes):
                    self.validation_errors.append(f"Link {i} has invalid target index: {target}")
                    return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_heatmap(
        self,
        figure: go.Figure,
        expected_dimensions: Optional[Tuple[int, int]] = None,
        check_colorscale: bool = True,
        check_hover_data: bool = True
    ) -> bool:
        """
        Validate heatmap structure.
        
        Args:
            figure: Plotly figure to validate
            expected_dimensions: Expected (rows, cols) dimensions
            check_colorscale: Whether to validate colorscale
            check_hover_data: Whether to validate hover data
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if figure has data
            if not figure.data:
                self.validation_errors.append("Figure has no data")
                return False
            
            # Find heatmap trace
            heatmap_trace = None
            for trace in figure.data:
                if trace.type == "heatmap":
                    heatmap_trace = trace
                    break
            
            if heatmap_trace is None:
                self.validation_errors.append("No heatmap trace found in figure")
                return False
            
            # Check Z data
            if not hasattr(heatmap_trace, 'z') or heatmap_trace.z is None:
                self.validation_errors.append("Heatmap has no Z data")
                return False
            
            z_data = heatmap_trace.z
            
            # Check dimensions
            if expected_dimensions is not None:
                if len(z_data) != expected_dimensions[0]:
                    self.validation_errors.append(f"Expected {expected_dimensions[0]} rows, got {len(z_data)}")
                    return False
                
                if len(z_data[0]) != expected_dimensions[1]:
                    self.validation_errors.append(f"Expected {expected_dimensions[1]} columns, got {len(z_data[0])}")
                    return False
            
            # Check data types
            for i, row in enumerate(z_data):
                for j, value in enumerate(row):
                    if not isinstance(value, (int, float, np.number)):
                        self.validation_errors.append(f"Invalid data type at ({i}, {j}): {type(value)}")
                        return False
            
            # Check colorscale if requested
            if check_colorscale and hasattr(heatmap_trace, 'colorscale'):
                colorscale = heatmap_trace.colorscale
                if colorscale is not None and not isinstance(colorscale, (str, list)):
                    self.validation_errors.append(f"Invalid colorscale type: {type(colorscale)}")
                    return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_network_graph(
        self,
        figure: go.Figure,
        expected_nodes: Optional[int] = None,
        expected_edges: Optional[int] = None,
        check_node_positions: bool = True
    ) -> bool:
        """
        Validate network graph structure.
        
        Args:
            figure: Plotly figure to validate
            expected_nodes: Expected number of nodes
            expected_edges: Expected number of edges
            check_node_positions: Whether to validate node positions
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if figure has data
            if not figure.data:
                self.validation_errors.append("Figure has no data")
                return False
            
            # Find scatter traces (nodes and edges)
            node_traces = []
            edge_traces = []
            
            for trace in figure.data:
                if trace.type == "scatter":
                    if hasattr(trace, 'mode') and 'markers' in trace.mode:
                        node_traces.append(trace)
                    elif hasattr(trace, 'mode') and 'lines' in trace.mode:
                        edge_traces.append(trace)
            
            if not node_traces:
                self.validation_errors.append("No node traces found")
                return False
            
            # Validate nodes
            total_nodes = sum(len(trace.x) for trace in node_traces if hasattr(trace, 'x'))
            if expected_nodes is not None and total_nodes != expected_nodes:
                self.validation_errors.append(f"Expected {expected_nodes} nodes, got {total_nodes}")
                return False
            
            # Check node positions if requested
            if check_node_positions:
                for trace in node_traces:
                    if not hasattr(trace, 'x') or not hasattr(trace, 'y'):
                        self.validation_errors.append("Node trace missing x or y coordinates")
                        return False
                    
                    if len(trace.x) != len(trace.y):
                        self.validation_errors.append("Node trace x and y coordinates length mismatch")
                        return False
            
            # Validate edges if present
            if expected_edges is not None:
                total_edges = len(edge_traces)
                if total_edges != expected_edges:
                    self.validation_errors.append(f"Expected {expected_edges} edge traces, got {total_edges}")
                    return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_bar_chart(
        self,
        figure: go.Figure,
        expected_bars: Optional[int] = None,
        check_values: bool = True
    ) -> bool:
        """
        Validate bar chart structure.
        
        Args:
            figure: Plotly figure to validate
            expected_bars: Expected number of bars
            check_values: Whether to validate bar values
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if figure has data
            if not figure.data:
                self.validation_errors.append("Figure has no data")
                return False
            
            # Find bar trace
            bar_trace = None
            for trace in figure.data:
                if trace.type == "bar":
                    bar_trace = trace
                    break
            
            if bar_trace is None:
                self.validation_errors.append("No bar trace found in figure")
                return False
            
            # Check bar count
            if expected_bars is not None:
                if not hasattr(bar_trace, 'x') or len(bar_trace.x) != expected_bars:
                    self.validation_errors.append(
                        f"Expected {expected_bars} bars, got {len(bar_trace.x) if hasattr(bar_trace, 'x') else 0}"
                    )
                    return False
            
            # Check required attributes
            required_attrs = ['x', 'y']
            for attr in required_attrs:
                if not hasattr(bar_trace, attr):
                    self.validation_errors.append(f"Bar trace missing required attribute: {attr}")
                    return False
            
            # Check values if requested
            if check_values and hasattr(bar_trace, 'y'):
                for i, value in enumerate(bar_trace.y):
                    if not isinstance(value, (int, float, np.number)):
                        self.validation_errors.append(f"Bar {i} has invalid value: {value}")
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors from last validation."""
        return self.validation_errors.copy()


class DashComponentValidator:
    """Validate Dash component structures."""
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
    
    def validate_tab_structure(
        self,
        tab_component: Any,
        expected_tabs: Optional[int] = None,
        check_labels: bool = True
    ) -> bool:
        """
        Validate tab component structure.
        
        Args:
            tab_component: Dash tab component
            expected_tabs: Expected number of tabs
            check_labels: Whether to validate tab labels
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if component is a Tabs component
            if not hasattr(tab_component, 'children'):
                self.validation_errors.append("Component is not a valid Tabs component")
                return False
            
            # Get children (should be Tab components)
            children = tab_component.children
            if not isinstance(children, list):
                self.validation_errors.append("Tab component children is not a list")
                return False
            
            # Check tab count
            if expected_tabs is not None and len(children) != expected_tabs:
                self.validation_errors.append(f"Expected {expected_tabs} tabs, got {len(children)}")
                return False
            
            # Check tab labels if requested
            if check_labels:
                for i, child in enumerate(children):
                    if not hasattr(child, 'label'):
                        self.validation_errors.append(f"Tab {i} has no label")
                        return False
                    
                    if not isinstance(child.label, str) or not child.label.strip():
                        self.validation_errors.append(f"Tab {i} has invalid label: {child.label}")
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_dropdown_options(
        self,
        dropdown: dcc.Dropdown,
        expected_options: Optional[List[Dict[str, str]]] = None,
        check_values: bool = True
    ) -> bool:
        """
        Validate dropdown options.
        
        Args:
            dropdown: Dash dropdown component
            expected_options: Expected dropdown options
            check_values: Whether to validate option values
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if component has options
            if not hasattr(dropdown, 'options'):
                self.validation_errors.append("Dropdown has no options attribute")
                return False
            
            options = dropdown.options
            if options is None:
                options = []
            
            # Check expected options if provided
            if expected_options is not None:
                if len(options) != len(expected_options):
                    self.validation_errors.append(
                        f"Expected {len(expected_options)} options, got {len(options)}"
                    )
                    return False
                
                for i, (actual, expected) in enumerate(zip(options, expected_options)):
                    if actual != expected:
                        self.validation_errors.append(f"Option {i} mismatch: {actual} != {expected}")
                        return False
            
            # Check option structure if requested
            if check_values:
                for i, option in enumerate(options):
                    if not isinstance(option, dict):
                        self.validation_errors.append(f"Option {i} is not a dictionary")
                        return False
                    
                    required_keys = ['label', 'value']
                    for key in required_keys:
                        if key not in option:
                            self.validation_errors.append(f"Option {i} missing key: {key}")
                            return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_table_data(
        self,
        table: Union[html.Table, Any],
        expected_columns: Optional[List[str]] = None,
        expected_rows: Optional[int] = None,
        check_data_types: bool = True
    ) -> bool:
        """
        Validate table data structure.
        
        Args:
            table: Dash table component
            expected_columns: Expected column names
            expected_rows: Expected number of rows
            check_data_types: Whether to validate data types
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Handle different table types
            if hasattr(table, 'children'):
                # HTML table
                children = table.children
                if not isinstance(children, list):
                    self.validation_errors.append("Table children is not a list")
                    return False
                
                # Find thead and tbody
                thead = None
                tbody = None
                
                for child in children:
                    if hasattr(child, 'type'):
                        if child.type == 'Thead':
                            thead = child
                        elif child.type == 'Tbody':
                            tbody = child
                
                # Validate header if present
                if thead and expected_columns:
                    if not hasattr(thead, 'children') or not thead.children:
                        self.validation_errors.append("Table header has no children")
                        return False
                    
                    # Get header row
                    header_row = thead.children[0] if isinstance(thead.children, list) else thead.children
                    if not hasattr(header_row, 'children'):
                        self.validation_errors.append("Header row has no children")
                        return False
                    
                    # Check column count
                    header_cells = header_row.children
                    if len(header_cells) != len(expected_columns):
                        self.validation_errors.append(
                            f"Expected {len(expected_columns)} columns, got {len(header_cells)}"
                        )
                        return False
                
                # Validate body if present
                if tbody and expected_rows:
                    if not hasattr(tbody, 'children'):
                        self.validation_errors.append("Table body has no children")
                        return False
                    
                    rows = tbody.children
                    if len(rows) != expected_rows:
                        self.validation_errors.append(f"Expected {expected_rows} rows, got {len(rows)}")
                        return False
            
            elif hasattr(table, 'data'):
                # DataTable
                data = table.data
                if expected_rows is not None and len(data) != expected_rows:
                    self.validation_errors.append(f"Expected {expected_rows} rows, got {len(data)}")
                    return False
                
                if expected_columns is not None and data:
                    actual_columns = list(data[0].keys())
                    if set(actual_columns) != set(expected_columns):
                        self.validation_errors.append(f"Column mismatch: {actual_columns} != {expected_columns}")
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_div_structure(
        self,
        div: html.Div,
        expected_children: Optional[int] = None,
        check_styles: bool = False
    ) -> bool:
        """
        Validate div structure.
        
        Args:
            div: HTML div component
            expected_children: Expected number of children
            check_styles: Whether to validate styles
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check if component has children
            if not hasattr(div, 'children'):
                self.validation_errors.append("Div has no children attribute")
                return False
            
            children = div.children
            if children is None:
                children = []
            elif not isinstance(children, list):
                children = [children]
            
            # Check child count
            if expected_children is not None and len(children) != expected_children:
                self.validation_errors.append(f"Expected {expected_children} children, got {len(children)}")
                return False
            
            # Check styles if requested
            if check_styles and hasattr(div, 'style'):
                style = div.style
                if style is not None and not isinstance(style, dict):
                    self.validation_errors.append(f"Invalid style type: {type(style)}")
                    return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors from last validation."""
        return self.validation_errors.copy()


class DataStructureValidator:
    """Validate data structures used in visualizations."""
    
    def __init__(self):
        """Initialize the validator."""
        self.validation_errors = []
    
    def validate_token_paths(
        self,
        token_paths: Dict[str, Dict[str, Any]],
        expected_tokens: Optional[int] = None,
        check_path_structure: bool = True
    ) -> bool:
        """
        Validate token paths data structure.
        
        Args:
            token_paths: Token paths dictionary
            expected_tokens: Expected number of tokens
            check_path_structure: Whether to validate path structure
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check basic structure
            if not isinstance(token_paths, dict):
                self.validation_errors.append("Token paths is not a dictionary")
                return False
            
            # Check if token_paths is empty when structure checking is enabled
            if check_path_structure and len(token_paths) == 0:
                self.validation_errors.append("Token paths is empty")
                return False
            
            # Check token count
            if expected_tokens is not None and len(token_paths) != expected_tokens:
                self.validation_errors.append(f"Expected {expected_tokens} tokens, got {len(token_paths)}")
                return False
            
            # Check path structure if requested
            if check_path_structure:
                required_keys = ['token_text', 'position', 'cluster_path']
                
                for token_id, path_data in token_paths.items():
                    if not isinstance(path_data, dict):
                        self.validation_errors.append(f"Token {token_id} path data is not a dictionary")
                        return False
                    
                    for key in required_keys:
                        if key not in path_data:
                            self.validation_errors.append(f"Token {token_id} missing key: {key}")
                            return False
                    
                    # Check cluster path
                    cluster_path = path_data['cluster_path']
                    if not isinstance(cluster_path, list):
                        self.validation_errors.append(f"Token {token_id} cluster_path is not a list")
                        return False
                    
                    for cluster_id in cluster_path:
                        if not isinstance(cluster_id, (int, np.integer)):
                            self.validation_errors.append(f"Token {token_id} has invalid cluster_id: {cluster_id}")
                            return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def validate_attention_data(
        self,
        attention_data: Dict[str, Dict[str, Any]],
        expected_layers: Optional[List[str]] = None,
        check_weights: bool = True
    ) -> bool:
        """
        Validate attention data structure.
        
        Args:
            attention_data: Attention data dictionary
            expected_layers: Expected layer names
            check_weights: Whether to validate attention weights
            
        Returns:
            True if validation passes, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Check basic structure
            if not isinstance(attention_data, dict):
                self.validation_errors.append("Attention data is not a dictionary")
                return False
            
            # Check layers
            if expected_layers is not None:
                if set(attention_data.keys()) != set(expected_layers):
                    self.validation_errors.append(f"Layer mismatch: {list(attention_data.keys())} != {expected_layers}")
                    return False
            
            # Check layer data structure
            for layer, layer_data in attention_data.items():
                if not isinstance(layer_data, dict):
                    self.validation_errors.append(f"Layer {layer} data is not a dictionary")
                    return False
                
                # Check required keys
                required_keys = ['entropy', 'head_agreement', 'num_heads']
                for key in required_keys:
                    if key not in layer_data:
                        self.validation_errors.append(f"Layer {layer} missing key: {key}")
                        return False
                
                # Check data types for numeric values
                if not isinstance(layer_data['entropy'], (int, float, np.number)):
                    self.validation_errors.append(f"Layer {layer} entropy is not a number: {type(layer_data['entropy'])}")
                    return False
                    
                if not isinstance(layer_data['head_agreement'], (int, float, np.number)):
                    self.validation_errors.append(f"Layer {layer} head_agreement is not a number: {type(layer_data['head_agreement'])}")
                    return False
                    
                if not isinstance(layer_data['num_heads'], (int, np.integer)):
                    self.validation_errors.append(f"Layer {layer} num_heads is not an integer: {type(layer_data['num_heads'])}")
                    return False
                
                # Check attention weights if requested
                if check_weights and 'weights' in layer_data:
                    weights = layer_data['weights']
                    if not isinstance(weights, np.ndarray):
                        self.validation_errors.append(f"Layer {layer} weights is not a numpy array")
                        return False
                    
                    # Check dimensions (batch, heads, seq_len, seq_len)
                    if len(weights.shape) != 4:
                        self.validation_errors.append(f"Layer {layer} weights has wrong number of dimensions: {weights.shape}")
                        return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Exception during validation: {str(e)}")
            return False
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors from last validation."""
        return self.validation_errors.copy()


# Convenience functions for common validations
def validate_figure_basic_structure(figure: go.Figure) -> bool:
    """Basic validation for any Plotly figure."""
    validator = PlotlyFigureValidator()
    
    # Check if figure exists and has data
    if figure is None:
        return False
    
    if not hasattr(figure, 'data') or not figure.data:
        return False
    
    # Check layout
    if not hasattr(figure, 'layout'):
        return False
    
    return True


def validate_dash_component_exists(component: Any) -> bool:
    """Basic validation that a Dash component exists and is properly structured."""
    if component is None:
        return False
    
    # Check if it's a Dash component
    if not hasattr(component, 'to_plotly_json'):
        return False
    
    return True


def print_validation_errors(validator: Union[PlotlyFigureValidator, DashComponentValidator, DataStructureValidator]):
    """Print validation errors for debugging."""
    errors = validator.get_validation_errors()
    if errors:
        print("Validation Errors:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    else:
        print("No validation errors found.")