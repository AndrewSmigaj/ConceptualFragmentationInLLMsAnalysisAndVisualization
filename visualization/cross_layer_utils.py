"""
Utility functions for cross-layer metrics in the visualization dashboard.

This module provides helper functions for serializing/deserializing NetworkX graphs
and other complex objects used in cross-layer metrics.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

def networkx_to_dict(graph: nx.Graph) -> Dict[str, Any]:
    """
    Convert a NetworkX graph to a JSON-serializable dictionary.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        Dictionary representation of the graph
    """
    # Create a dictionary representation of the graph
    graph_dict = {
        "nodes": [],
        "edges": [],
        "directed": isinstance(graph, nx.DiGraph)
    }
    
    # Add nodes with attributes
    for node in graph.nodes():
        node_attrs = {key: value for key, value in graph.nodes[node].items()}
        graph_dict["nodes"].append({
            "id": node,
            "attributes": node_attrs
        })
    
    # Add edges with attributes
    for u, v in graph.edges():
        edge_attrs = {key: value for key, value in graph.edges[u, v].items()}
        graph_dict["edges"].append({
            "source": u,
            "target": v,
            "attributes": edge_attrs
        })
    
    return graph_dict

def dict_to_networkx(graph_dict: Dict[str, Any]) -> nx.Graph:
    """
    Convert a dictionary representation of a graph back to a NetworkX graph.
    
    Args:
        graph_dict: Dictionary representation of a graph
        
    Returns:
        NetworkX graph object
    """
    # Create appropriate graph type
    if graph_dict.get("directed", False):
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add nodes with attributes
    for node_data in graph_dict.get("nodes", []):
        node_id = node_data.get("id")
        node_attrs = node_data.get("attributes", {})
        G.add_node(node_id, **node_attrs)
    
    # Add edges with attributes
    for edge_data in graph_dict.get("edges", []):
        source = edge_data.get("source")
        target = edge_data.get("target")
        edge_attrs = edge_data.get("attributes", {})
        G.add_edge(source, target, **edge_attrs)
    
    return G

def serialize_cross_layer_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize cross-layer metrics for JSON storage.
    
    Args:
        metrics: Dictionary of cross-layer metrics
        
    Returns:
        Serialized metrics dictionary
    """
    serialized = {}
    
    # Handle each metric type
    for key, value in metrics.items():
        # Handle NetworkX graph
        if key == "path_graph" and isinstance(value, nx.Graph):
            serialized[key] = networkx_to_dict(value)
        # Handle error messages
        elif key.endswith("_error"):
            serialized[key] = str(value)
        # Handle numpy arrays in nested dictionaries
        elif isinstance(value, dict):
            serialized[key] = serialize_nested_dict(value)
        # Handle other types
        else:
            serialized[key] = value
    
    return serialized

def serialize_nested_dict(nested_dict: Dict) -> Dict:
    """
    Recursively serialize a nested dictionary containing numpy arrays.
    
    Args:
        nested_dict: Dictionary potentially containing numpy arrays
        
    Returns:
        Serialized dictionary
    """
    result = {}
    
    for key, value in nested_dict.items():
        # Convert tuple keys to strings for JSON compatibility
        if isinstance(key, tuple):
            # Join the tuple elements with a separator that won't be in the strings
            str_key = str(key)
        else:
            str_key = key
            
        # Handle numpy arrays
        if isinstance(value, np.ndarray):
            result[str_key] = value.tolist()
        # Recurse into nested dictionaries
        elif isinstance(value, dict):
            result[str_key] = serialize_nested_dict(value)
        # Convert tuples to lists for JSON serialization
        elif isinstance(value, tuple):
            result[str_key] = list(value)
        # Handle other types
        else:
            result[str_key] = value
    
    return result

def deserialize_cross_layer_metrics(serialized: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deserialize cross-layer metrics from JSON storage.
    
    Args:
        serialized: Serialized metrics dictionary
        
    Returns:
        Deserialized metrics dictionary
    """
    deserialized = {}
    
    # Handle each metric type
    for key, value in serialized.items():
        # Handle NetworkX graph
        if key == "path_graph" and isinstance(value, dict):
            deserialized[key] = dict_to_networkx(value)
        # Handle nested dictionaries
        elif isinstance(value, dict):
            deserialized[key] = deserialize_nested_dict(value)
        # Handle other types
        else:
            deserialized[key] = value
    
    return deserialized

def deserialize_nested_dict(nested_dict: Dict) -> Dict:
    """
    Recursively deserialize a nested dictionary containing numpy arrays.
    
    Args:
        nested_dict: Serialized dictionary
        
    Returns:
        Deserialized dictionary
    """
    result = {}
    
    for key, value in nested_dict.items():
        # Check if the key is a string representation of a tuple
        # Match patterns like "('layer1', 'layer2')" or "(1, 2, 3)"
        is_tuple_key = False
        if isinstance(key, str) and key.startswith("(") and key.endswith(")") and "," in key:
            try:
                # Try to safely evaluate the string as a tuple
                # This handles cases like "('layer1', 'layer2')"
                orig_key = eval(key)
                if isinstance(orig_key, tuple):
                    is_tuple_key = True
                    target_key = orig_key
            except (SyntaxError, NameError, ValueError):
                # If eval fails, it wasn't a proper tuple representation
                is_tuple_key = False
                target_key = key
        else:
            target_key = key
            
        # Convert lists back to numpy arrays
        if isinstance(value, list):
            # Handle list of lists (2D array)
            if value and isinstance(value[0], list):
                processed_value = np.array(value)
            # Handle regular lists
            else:
                try:
                    processed_value = np.array(value)
                except:
                    processed_value = value
        # Recurse into nested dictionaries
        elif isinstance(value, dict):
            processed_value = deserialize_nested_dict(value)
        # Handle other types
        else:
            processed_value = value
            
        # Add to result dictionary with the appropriate key
        result[target_key] = processed_value
    
    return result