"""
Generate Sankey diagram visualizations for cluster membership overlap.

This script creates Sankey diagrams showing how datapoints flow through
different clusters across layers, with colors representing original class labels.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any, Union
import argparse

# Define constants
COLORS = px.colors.qualitative.Plotly  # Default Plotly color sequence


def load_data(data_dir: str, dataset: str = "titanic", seed: int = 0) -> Tuple[Dict, Optional[List[int]]]:
    """
    Load path data and class labels.
    
    Args:
        data_dir: Directory containing data files
        dataset: Dataset name
        seed: Random seed used for training
        
    Returns:
        Tuple of (paths, labels)
    """
    # Load path data
    path_file = os.path.join(data_dir, f"{dataset}_seed_{seed}_paths.json")
    with open(path_file, 'r') as f:
        path_data = json.load(f)
    
    # Try to load labels if available
    labels_file = os.path.join(data_dir, f"{dataset}_seed_{seed}_labels.json")
    labels = None
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    
    return path_data, labels


def prepare_sankey_data(path_data: Dict, labels: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Prepare data for Sankey diagram.
    
    Args:
        path_data: Dictionary with path information
        labels: Optional list of class labels for datapoints
        
    Returns:
        Dictionary with Sankey diagram data
    """
    paths = path_data.get("unique_paths", [])
    layers = path_data.get("layers", [])
    
    # Create node lists - one for each layer's clusters
    nodes = []
    node_labels = []
    node_colors = []
    
    # Map to track node indices
    node_map = {}
    
    # Create layer-specific colors
    layer_colors = {
        layer: f"rgba({50 + i * 50}, {100 + i * 30}, {150 + i * 20}, 0.8)" 
        for i, layer in enumerate(layers)
    }
    
    # Process each layer
    for layer_idx, layer in enumerate(layers):
        # Find unique clusters in this layer
        clusters = set(path[layer_idx] for path in paths)
        
        # Add nodes for each cluster
        for cluster in sorted(clusters):
            # Create node ID
            node_id = f"{layer}_C{cluster}"
            node_idx = len(nodes)
            node_map[node_id] = node_idx
            
            # Add node
            nodes.append(node_id)
            node_labels.append(f"{layer} Cluster {cluster}")
            node_colors.append(layer_colors[layer])
    
    # Create links
    sources = []
    targets = []
    values = []
    link_colors = []
    
    # Use labels for coloring if available
    if labels:
        unique_labels = sorted(set(labels))
        label_colors = {
            label: COLORS[i % len(COLORS)]
            for i, label in enumerate(unique_labels)
        }
    
    # Process connections between layers
    for layer_idx in range(len(layers) - 1):
        layer1 = layers[layer_idx]
        layer2 = layers[layer_idx + 1]
        
        # Track transitions with label information
        transitions = {}
        
        # Count transitions
        for path_idx, path in enumerate(paths):
            # Get clusters for consecutive layers
            cluster1 = path[layer_idx]
            cluster2 = path[layer_idx + 1]
            
            # Create source and target node IDs
            source_id = f"{layer1}_C{cluster1}"
            target_id = f"{layer2}_C{cluster2}"
            
            # Get label for coloring (if available)
            if labels and path_idx < len(labels):
                label = labels[path_idx]
                # Initialize if first occurrence of this transition
                if (source_id, target_id) not in transitions:
                    transitions[(source_id, target_id)] = {}
                # Count by label
                transitions[(source_id, target_id)][label] = transitions[(source_id, target_id)].get(label, 0) + 1
            else:
                # If no labels, just count transitions
                transitions[(source_id, target_id)] = transitions.get((source_id, target_id), 0) + 1
        
        # Add links for each transition
        for (source_id, target_id), value in transitions.items():
            source_idx = node_map[source_id]
            target_idx = node_map[target_id]
            
            # If label-specific counts are available
            if isinstance(value, dict):
                # Create a link for each label
                for label, count in value.items():
                    sources.append(source_idx)
                    targets.append(target_idx)
                    values.append(count)
                    link_colors.append(label_colors[label])
            else:
                # Single link for all transitions
                sources.append(source_idx)
                targets.append(target_idx)
                values.append(value)
                link_colors.append("rgba(150, 150, 150, 0.5)")  # Default gray
    
    # Return prepared Sankey data
    return {
        "nodes": nodes,
        "node_labels": node_labels,
        "node_colors": node_colors,
        "sources": sources,
        "targets": targets,
        "values": values,
        "link_colors": link_colors
    }


def generate_sankey_diagram(sankey_data: Dict[str, Any], 
                            output_dir: str,
                            title: str = "Membership Overlap Across Layers"):
    """
    Generate Sankey diagram visualization.
    
    Args:
        sankey_data: Dictionary with Sankey data
        output_dir: Directory to save output files
        title: Title for the diagram
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig = go.Figure(data=[go.Sankey(
        # Define nodes
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "black", width = 0.5),
            label = sankey_data["node_labels"],
            color = sankey_data["node_colors"]
        ),
        # Define links
        link = dict(
            source = sankey_data["sources"],
            target = sankey_data["targets"],
            value = sankey_data["values"],
            color = sankey_data["link_colors"]
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text=title,
        font_size=12,
        height=800,
        width=1200
    )
    
    # Save figure
    output_file = os.path.join(output_dir, "membership_overlap_sankey.html")
    fig.write_html(output_file)
    print(f"Sankey diagram saved to {output_file}")
    
    # Also save as static image
    static_file = os.path.join(output_dir, "membership_overlap_sankey.png")
    fig.write_image(static_file, width=1200, height=800, scale=2)
    print(f"Static image saved to {static_file}")
    
    return fig


def calculate_mixing_metrics(sankey_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate mixing metrics for the Sankey diagram.
    
    Args:
        sankey_data: Dictionary with Sankey data
        
    Returns:
        Dictionary with mixing metrics
    """
    # Create source-target flow matrix
    sources = np.array(sankey_data["sources"])
    targets = np.array(sankey_data["targets"])
    values = np.array(sankey_data["values"])
    
    # Get unique node indices
    all_nodes = set(sources) | set(targets)
    node_count = len(all_nodes)
    
    # Create flow matrix
    flow_matrix = np.zeros((node_count, node_count))
    for i in range(len(sources)):
        flow_matrix[sources[i], targets[i]] = values[i]
    
    # Calculate metrics
    metrics = {}
    
    # 1. Entropy of outgoing connections per node
    outgoing_entropy = []
    for i in range(node_count):
        outgoing = flow_matrix[i, :]
        # Skip nodes with no outgoing connections
        if np.sum(outgoing) == 0:
            continue
            
        # Normalize to get probabilities
        probs = outgoing / np.sum(outgoing)
        # Remove zeros
        probs = probs[probs > 0]
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        outgoing_entropy.append(entropy)
    
    # Average entropy across nodes
    metrics["avg_outgoing_entropy"] = np.mean(outgoing_entropy) if outgoing_entropy else 0
    
    # 2. Purity - how much flows stay within class
    # Requires class information, which we don't have directly
    # Instead, use as proxy: how concentrated flows are
    
    # Get total flow per source and target
    source_flow = np.sum(flow_matrix, axis=1)
    target_flow = np.sum(flow_matrix, axis=0)
    
    # Calculate concentration of flows
    source_concentration = np.sum(source_flow**2) / (np.sum(source_flow)**2)
    target_concentration = np.sum(target_flow**2) / (np.sum(target_flow)**2)
    
    metrics["source_concentration"] = source_concentration
    metrics["target_concentration"] = target_concentration
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generate Sankey diagram for cluster membership overlap.")
    parser.add_argument("--data_dir", type=str, default="data/cluster_paths",
                       help="Directory containing cluster path data")
    parser.add_argument("--output_dir", type=str, default="results/figures",
                       help="Directory to save output figures")
    parser.add_argument("--dataset", type=str, default="titanic",
                       help="Dataset name")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    path_data, labels = load_data(args.data_dir, args.dataset, args.seed)
    
    # If labels not available, generate synthetic ones for visualization
    if not labels:
        print("No labels found, generating synthetic labels for visualization.")
        # Just split paths into two classes for demonstration
        n_paths = len(path_data.get("unique_paths", []))
        labels = [0] * (n_paths // 2) + [1] * (n_paths - n_paths // 2)
    
    # Prepare Sankey data
    sankey_data = prepare_sankey_data(path_data, labels)
    
    # Generate Sankey diagram
    generate_sankey_diagram(sankey_data, args.output_dir, 
                          title=f"{args.dataset.title()} Dataset: Cluster Membership Flow")
    
    # Calculate and display mixing metrics
    metrics = calculate_mixing_metrics(sankey_data)
    print("\nMixing metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nSankey diagram generation complete!")


if __name__ == "__main__":
    main()