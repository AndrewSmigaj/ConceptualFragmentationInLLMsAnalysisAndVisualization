"""
Command-line tool for batch cross-layer analysis of transformer models.

This script performs comprehensive cross-layer analysis on transformer model
activations and attention patterns, generating detailed reports and
visualizations.

Usage:
    python -m concept_fragmentation.analysis.analyze_transformer_cross_layer \
        --activation-path /path/to/activations.npy \
        --attention-path /path/to/attention.npy \
        --output-dir /path/to/output \
        --config /path/to/config.json

Configuration options (in JSON):
    - similarity_metric: Metric for layer similarity (cosine, cka, correlation)
    - dimensionality_reduction: Whether to apply dimensionality reduction
    - n_components: Number of components for dimensionality reduction
    - pca_components: Number of PCA components for visualization
    - focus_tokens: List of tokens to focus trajectory analysis on
    - cache_dir: Directory for caching results
"""

import argparse
import json
import logging
import numpy as np
import os
import sys
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import necessary modules
from concept_fragmentation.analysis.transformer_cross_layer import (
    TransformerCrossLayerAnalyzer,
    CrossLayerTransformerAnalysisResult
)
from concept_fragmentation.metrics.transformer_metrics import (
    TransformerMetricsCalculator
)
from concept_fragmentation.analysis.transformer_dimensionality import (
    TransformerDimensionalityReducer
)

# Default configuration
DEFAULT_CONFIG = {
    "similarity_metric": "cosine",
    "dimensionality_reduction": True,
    "n_components": 50,
    "pca_components": 3,
    "random_state": 42,
    "use_cache": True,
    "verbose": True
}

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from JSON file, using defaults for missing values.
    
    Args:
        config_path: Path to configuration JSON file
        
    Returns:
        Merged configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            
            # Update config with user values
            config.update(user_config)
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    return config

def load_data(
    activation_path: str, 
    attention_path: str,
    token_info_path: Optional[str] = None,
    class_labels_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load transformer data from specified paths.
    
    Args:
        activation_path: Path to activation data
        attention_path: Path to attention data
        token_info_path: Optional path to token information
        class_labels_path: Optional path to class labels
        
    Returns:
        Dictionary with loaded data
    """
    data = {}
    
    # Load activations
    try:
        logger.info(f"Loading activations from {activation_path}")
        if activation_path.endswith('.npy'):
            activations = np.load(activation_path, allow_pickle=True).item()
        elif activation_path.endswith('.pt'):
            activations = torch.load(activation_path)
            # Convert torch dictionary to numpy if needed
            if isinstance(activations, dict) and all(isinstance(v, torch.Tensor) for v in activations.values()):
                activations = {k: v.detach().cpu().numpy() for k, v in activations.items()}
        else:
            raise ValueError(f"Unsupported activation file format: {activation_path}")
        
        data['layer_activations'] = activations
    except Exception as e:
        logger.error(f"Error loading activations: {e}")
        raise
    
    # Load attention data
    try:
        logger.info(f"Loading attention data from {attention_path}")
        if attention_path.endswith('.npy'):
            attention_data = np.load(attention_path, allow_pickle=True).item()
        elif attention_path.endswith('.pt'):
            attention_data = torch.load(attention_path)
            # Convert torch dictionary to numpy if needed
            if isinstance(attention_data, dict) and all(isinstance(v, torch.Tensor) for v in attention_data.values()):
                attention_data = {k: v.detach().cpu().numpy() for k, v in attention_data.items()}
        else:
            raise ValueError(f"Unsupported attention file format: {attention_path}")
        
        data['attention_data'] = attention_data
    except Exception as e:
        logger.error(f"Error loading attention data: {e}")
        raise
    
    # Load token info if provided
    if token_info_path:
        try:
            logger.info(f"Loading token info from {token_info_path}")
            if token_info_path.endswith('.npy'):
                token_info = np.load(token_info_path, allow_pickle=True).item()
            elif token_info_path.endswith('.json'):
                with open(token_info_path, 'r') as f:
                    token_info = json.load(f)
            elif token_info_path.endswith('.pt'):
                token_info = torch.load(token_info_path)
            else:
                raise ValueError(f"Unsupported token info file format: {token_info_path}")
            
            data['token_info'] = token_info
            
            # Extract token IDs and strings
            if 'token_ids' in token_info:
                data['token_ids'] = token_info['token_ids']
            if 'token_strings' in token_info:
                data['token_strings'] = token_info['token_strings']
        except Exception as e:
            logger.warning(f"Error loading token info: {e}")
    
    # Load class labels if provided
    if class_labels_path:
        try:
            logger.info(f"Loading class labels from {class_labels_path}")
            if class_labels_path.endswith('.npy'):
                class_labels = np.load(class_labels_path)
            elif class_labels_path.endswith('.txt'):
                class_labels = np.loadtxt(class_labels_path)
            elif class_labels_path.endswith('.pt'):
                class_labels = torch.load(class_labels_path)
                if isinstance(class_labels, torch.Tensor):
                    class_labels = class_labels.detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported class labels file format: {class_labels_path}")
            
            data['class_labels'] = class_labels
        except Exception as e:
            logger.warning(f"Error loading class labels: {e}")
    
    return data

def generate_plots(
    result: CrossLayerTransformerAnalysisResult,
    output_dir: str
) -> List[str]:
    """
    Generate plots from analysis results.
    
    Args:
        result: CrossLayerTransformerAnalysisResult to visualize
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated plot files
    """
    plot_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Layer Similarity Heatmap
    try:
        if "layer_similarity" in result.plot_data:
            plt.figure(figsize=(10, 8))
            
            # Extract data for heatmap
            similarity_data = result.plot_data["layer_similarity"]
            layers = sorted(set([d["layer1"] for d in similarity_data] + [d["layer2"] for d in similarity_data]))
            
            # Create similarity matrix
            sim_matrix = np.zeros((len(layers), len(layers)))
            
            for item in similarity_data:
                i = layers.index(item["layer1"])
                j = layers.index(item["layer2"])
                sim_matrix[i, j] = item["similarity"]
                sim_matrix[j, i] = item["similarity"]  # Mirror for full matrix
            
            # Plot heatmap
            sns.heatmap(
                sim_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="viridis",
                xticklabels=layers,
                yticklabels=layers
            )
            
            plt.title("Layer Similarity Matrix")
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"layer_similarity_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
            logger.info(f"Generated layer similarity heatmap: {plot_path}")
    except Exception as e:
        logger.warning(f"Error generating layer similarity heatmap: {e}")
    
    # 2. Attention Flow Direction Network
    try:
        if "attention_flow" in result.plot_data:
            plt.figure(figsize=(12, 8))
            
            # Extract data for flow diagram
            flow_data = result.plot_data["attention_flow"]
            
            # Create network graph
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes for each layer
            layers = sorted(set([d["source"] for d in flow_data] + [d["target"] for d in flow_data]))
            for i, layer in enumerate(layers):
                G.add_node(layer, pos=(i, 0))
            
            # Add edges with attributes
            for item in flow_data:
                G.add_edge(
                    item["source"],
                    item["target"],
                    weight=abs(item["entropy_diff"]) * 3,
                    color="red" if item["direction"] < 0 else "green",
                    label=f"{item['entropy_diff']:.2f}"
                )
            
            # Get positions
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
            nx.draw_networkx_labels(G, pos, font_size=12)
            
            # Draw edges with different colors based on direction
            for edge in G.edges(data=True):
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(edge[0], edge[1])],
                    width=edge[2]['weight'],
                    edge_color=edge[2]['color'],
                    arrows=True,
                    arrowsize=20
                )
            
            # Draw edge labels
            edge_labels = {(edge[0], edge[1]): edge[2]['label'] for edge in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title("Attention Flow Direction")
            plt.axis('off')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"attention_flow_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
            logger.info(f"Generated attention flow network: {plot_path}")
    except Exception as e:
        logger.warning(f"Error generating attention flow network: {e}")
    
    # 3. Token Trajectory Heatmap
    try:
        if "token_trajectories" in result.plot_data:
            plt.figure(figsize=(14, 8))
            
            # Extract data for token trajectories
            trajectory_data = result.plot_data["token_trajectories"]
            
            # Get unique tokens and layers
            tokens = sorted(set([d["token"] for d in trajectory_data]))
            layers = sorted(set([d["source"] for d in trajectory_data] + [d["target"] for d in trajectory_data]))
            
            # Create transition matrix
            n_tokens = len(tokens)
            n_transitions = len(layers) - 1
            stability_matrix = np.zeros((n_tokens, n_transitions))
            
            # Fill in stability values
            for item in trajectory_data:
                token_idx = tokens.index(item["token"])
                source_idx = layers.index(item["source"])
                target_idx = layers.index(item["target"])
                
                if target_idx == source_idx + 1:  # Only use consecutive layers
                    stability_matrix[token_idx, source_idx] = item["stability"]
            
            # Plot heatmap
            sns.heatmap(
                stability_matrix,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                xticklabels=[f"{layers[i]}->{layers[i+1]}" for i in range(len(layers)-1)],
                yticklabels=tokens
            )
            
            plt.title("Token Stability Across Layers")
            plt.xlabel("Layer Transitions")
            plt.ylabel("Tokens")
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"token_stability_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
            logger.info(f"Generated token stability heatmap: {plot_path}")
    except Exception as e:
        logger.warning(f"Error generating token stability heatmap: {e}")
    
    # 4. Class Separation Bar Chart
    try:
        if "class_separation" in result.plot_data:
            plt.figure(figsize=(10, 6))
            
            # Extract data for class separation
            separation_data = result.plot_data["class_separation"]
            
            # Create bar chart
            layers = [d["layer"] for d in separation_data]
            separation = [d["separation"] for d in separation_data]
            
            plt.bar(layers, separation, color="skyblue")
            plt.axhline(y=np.mean(separation), color='r', linestyle='--', label=f"Mean: {np.mean(separation):.2f}")
            
            plt.title("Class Separation Across Layers")
            plt.xlabel("Layer")
            plt.ylabel("Class Separation")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"class_separation_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
            logger.info(f"Generated class separation bar chart: {plot_path}")
    except Exception as e:
        logger.warning(f"Error generating class separation bar chart: {e}")
    
    # 5. Representation Divergence Network
    try:
        if "representation_divergence" in result.plot_data:
            plt.figure(figsize=(12, 8))
            
            # Extract data for divergence visualization
            divergence_data = result.plot_data["representation_divergence"]
            
            # Create network graph
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes for each layer
            layers = sorted(set([d["source"] for d in divergence_data] + [d["target"] for d in divergence_data]))
            for i, layer in enumerate(layers):
                G.add_node(layer, pos=(i, 0))
            
            # Add edges with attributes
            for item in divergence_data:
                G.add_edge(
                    item["source"],
                    item["target"],
                    weight=item["divergence"] * 5,
                    label=f"{item['divergence']:.2f}"
                )
            
            # Get positions
            pos = nx.get_node_attributes(G, 'pos')
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightgreen")
            nx.draw_networkx_labels(G, pos, font_size=12)
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                width=[G[u][v]['weight'] for u, v in G.edges()],
                arrows=True,
                arrowsize=20
            )
            
            # Draw edge labels
            edge_labels = {(edge[0], edge[1]): edge[2]['label'] for edge in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
            plt.title("Representation Divergence Between Layers")
            plt.axis('off')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"representation_divergence_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            
            plot_paths.append(plot_path)
            logger.info(f"Generated representation divergence network: {plot_path}")
    except Exception as e:
        logger.warning(f"Error generating representation divergence network: {e}")
    
    return plot_paths

def generate_markdown_report(
    result: CrossLayerTransformerAnalysisResult,
    plot_paths: List[str],
    output_dir: str
) -> str:
    """
    Generate a Markdown report summarizing the analysis results.
    
    Args:
        result: CrossLayerTransformerAnalysisResult to report on
        plot_paths: List of paths to generated plots
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"cross_layer_analysis_report_{timestamp}.md")
    
    # Get result summary
    summary = result.get_summary()
    
    # Convert plot paths to relative paths for the report
    rel_plot_paths = []
    for path in plot_paths:
        rel_plot_paths.append(os.path.basename(path))
    
    # Generate report content
    report_content = [
        "# Transformer Cross-Layer Analysis Report",
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    
    # Add summary metrics to the table
    for key, value in summary.items():
        if key == "focused_tokens":
            # Format list of tokens
            token_str = ", ".join(value) if isinstance(value, list) else str(value)
            report_content.append(f"| {key} | {token_str} |")
        elif isinstance(value, float):
            report_content.append(f"| {key} | {value:.4f} |")
        else:
            report_content.append(f"| {key} | {value} |")
    
    # Add model details
    report_content.extend([
        "",
        "## Model Details",
        "",
        f"Analyzed {len(result.layers_analyzed)} layers and {summary.get('total_attention_heads', 0)} attention heads.",
        "",
        "### Layers Analyzed",
        "",
        "```",
        "\n".join(result.layers_analyzed),
        "```",
        "",
    ])
    
    # Add layer relationship details
    if result.layer_relationships and "similarity_matrix" in result.layer_relationships:
        report_content.extend([
            "## Layer Relationships",
            "",
            "### Layer Similarity Matrix",
            "",
            "| Layer Pair | Similarity |",
            "|------------|------------|",
        ])
        
        # Add similarities to the table
        for (layer1, layer2), sim in result.layer_relationships["similarity_matrix"].items():
            report_content.append(f"| {layer1} - {layer2} | {sim:.4f} |")
    
    # Add attention flow details
    if result.attention_flow and "layer_entropy" in result.attention_flow:
        report_content.extend([
            "",
            "## Attention Flow Analysis",
            "",
            "### Layer Entropy",
            "",
            "| Layer | Entropy |",
            "|-------|---------|",
        ])
        
        # Add entropy values to the table
        for layer, entropy in result.attention_flow["layer_entropy"].items():
            report_content.append(f"| {layer} | {entropy:.4f} |")
        
        # Add entropy differences
        if "layer_entropy_differences" in result.attention_flow:
            report_content.extend([
                "",
                "### Entropy Changes Between Layers",
                "",
                "| Layer Transition | Entropy Change | Direction |",
                "|------------------|----------------|-----------|",
            ])
            
            # Add entropy differences to the table
            for (layer1, layer2), diff in result.attention_flow["layer_entropy_differences"].items():
                direction = "More Focused" if diff < 0 else "More Dispersed"
                report_content.append(f"| {layer1} -> {layer2} | {diff:.4f} | {direction} |")
    
    # Add token trajectory details
    if result.token_trajectory and "focused_tokens" in result.token_trajectory:
        report_content.extend([
            "",
            "## Token Trajectory Analysis",
            "",
            "### Focused Tokens",
            "",
            "```",
            "\n".join(result.token_trajectory["focused_tokens"]),
            "```",
            "",
        ])
        
        # Add token stability
        if "avg_token_stability" in result.token_trajectory:
            report_content.extend([
                "### Average Token Stability Between Layers",
                "",
                "| Layer Transition | Stability |",
                "|------------------|-----------|",
            ])
            
            # Add stability values to the table
            for (layer1, layer2), stability in result.token_trajectory["avg_token_stability"].items():
                report_content.append(f"| {layer1} -> {layer2} | {stability:.4f} |")
    
    # Add representation evolution details
    if result.representation_evolution:
        report_content.extend([
            "",
            "## Representation Evolution Analysis",
            "",
        ])
        
        # Add class separation if available
        if "class_separation" in result.representation_evolution:
            report_content.extend([
                "### Class Separation by Layer",
                "",
                "| Layer | Separation |",
                "|-------|------------|",
            ])
            
            # Add class separation values to the table
            for layer, separation in result.representation_evolution["class_separation"].items():
                report_content.append(f"| {layer} | {separation:.4f} |")
        
        # Add representation divergence
        if "representation_divergence" in result.representation_evolution:
            report_content.extend([
                "",
                "### Representation Divergence Between Layers",
                "",
                "| Layer Transition | Divergence |",
                "|------------------|------------|",
            ])
            
            # Add divergence values to the table
            for (layer1, layer2), divergence in result.representation_evolution["representation_divergence"].items():
                report_content.append(f"| {layer1} -> {layer2} | {divergence:.4f} |")
    
    # Add visualizations
    if rel_plot_paths:
        report_content.extend([
            "",
            "## Visualizations",
            "",
        ])
        
        # Add images to the report
        for i, path in enumerate(rel_plot_paths):
            plot_type = path.split('_')[0].replace('.png', '')
            report_content.extend([
                f"### {plot_type.title()} Visualization",
                "",
                f"![{plot_type} Visualization]({path})",
                "",
            ])
    
    # Write report to file
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    logger.info(f"Generated Markdown report: {report_path}")
    return report_path

def save_results(
    result: CrossLayerTransformerAnalysisResult, 
    output_dir: str
) -> str:
    """
    Save analysis results to disk.
    
    Args:
        result: CrossLayerTransformerAnalysisResult to save
        output_dir: Directory to save results
        
    Returns:
        Path to saved results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"cross_layer_analysis_results_{timestamp}.json")
    
    # Prepare results for serialization
    serializable_results = {
        "summary": result.get_summary(),
        "layers_analyzed": result.layers_analyzed,
        "plot_data": result.plot_data
    }
    
    # Add layer relationships
    if result.layer_relationships:
        relationships = {}
        
        # Convert similarity matrix (handle tuple keys)
        if "similarity_matrix" in result.layer_relationships:
            similarity_matrix = {}
            for (layer1, layer2), value in result.layer_relationships["similarity_matrix"].items():
                similarity_matrix[f"{layer1}_{layer2}"] = value
            relationships["similarity_matrix"] = similarity_matrix
        
        # Add other serializable data
        for key, value in result.layer_relationships.items():
            if key != "similarity_matrix" and key != "graph":
                relationships[key] = value
            
        serializable_results["layer_relationships"] = relationships
    
    # Add attention flow
    if result.attention_flow:
        flow = {}
        
        # Handle dictionary with tuple keys
        for key, value in result.attention_flow.items():
            if isinstance(value, dict) and any(isinstance(k, tuple) for k in value.keys()):
                # Convert tuple keys to strings
                converted = {}
                for k, v in value.items():
                    if isinstance(k, tuple):
                        converted[f"{k[0]}_{k[1]}"] = v
                    else:
                        converted[k] = v
                flow[key] = converted
            elif key != "flow_graph":
                flow[key] = value
        
        serializable_results["attention_flow"] = flow
    
    # Add token trajectory
    if result.token_trajectory:
        trajectory = {}
        
        # Handle nested dictionaries with tuple keys
        for key, value in result.token_trajectory.items():
            if isinstance(value, dict):
                if any(isinstance(k, tuple) for k in value.keys()):
                    # Convert tuple keys to strings
                    converted = {}
                    for k, v in value.items():
                        converted[str(k)] = v
                    trajectory[key] = converted
                elif any(isinstance(v, dict) and any(isinstance(k2, tuple) for k2 in v.keys()) for v in value.values()):
                    # Handle nested dictionaries
                    converted = {}
                    for token_key, token_dict in value.items():
                        token_converted = {}
                        for k, v in token_dict.items():
                            if isinstance(k, tuple):
                                token_converted[f"{k[0]}_{k[1]}"] = v
                            else:
                                token_converted[k] = v
                        converted[token_key] = token_converted
                    trajectory[key] = converted
                else:
                    trajectory[key] = value
            else:
                trajectory[key] = value
        
        serializable_results["token_trajectory"] = trajectory
    
    # Add representation evolution
    if result.representation_evolution:
        evolution = {}
        
        # Handle dictionaries with tuple keys
        for key, value in result.representation_evolution.items():
            if isinstance(value, dict) and any(isinstance(k, tuple) for k in value.keys()):
                # Convert tuple keys to strings
                converted = {}
                for k, v in value.items():
                    if isinstance(k, tuple):
                        converted[f"{k[0]}_{k[1]}"] = v
                    else:
                        converted[k] = v
                evolution[key] = converted
            else:
                evolution[key] = value
        
        serializable_results["representation_evolution"] = evolution
    
    # Add embedding comparisons
    if result.embedding_comparisons:
        comparisons = {}
        
        # Handle dictionaries with numpy arrays and tuple keys
        for key, value in result.embedding_comparisons.items():
            if key == "embeddings":
                # Convert numpy arrays to lists
                embeddings = {}
                for layer, emb in value.items():
                    if isinstance(emb, np.ndarray):
                        embeddings[layer] = emb.tolist()
                    else:
                        embeddings[layer] = emb
                comparisons[key] = embeddings
            elif key == "embedding_similarities" and isinstance(value, dict):
                # Convert tuple keys to strings
                similarities = {}
                for k, v in value.items():
                    if isinstance(k, tuple):
                        similarities[f"{k[0]}_{k[1]}"] = v
                    else:
                        similarities[k] = v
                comparisons[key] = similarities
            else:
                comparisons[key] = value
        
        serializable_results["embedding_comparisons"] = comparisons
    
    # Write results to file
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved analysis results to {results_path}")
    return results_path

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze cross-layer relationships in transformer models."
    )
    
    parser.add_argument(
        "--activation-path",
        type=str,
        required=True,
        help="Path to activation data (.npy or .pt file)"
    )
    
    parser.add_argument(
        "--attention-path",
        type=str,
        required=True,
        help="Path to attention data (.npy or .pt file)"
    )
    
    parser.add_argument(
        "--token-info-path",
        type=str,
        default=None,
        help="Optional path to token information (.npy, .pt, or .json file)"
    )
    
    parser.add_argument(
        "--class-labels-path",
        type=str,
        default=None,
        help="Optional path to class labels (.npy, .pt, or .txt file)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./cross_layer_analysis_results",
        help="Directory to save analysis results and visualizations"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--focus-tokens",
        type=str,
        nargs='+',
        default=None,
        help="Optional tokens to focus on in trajectory analysis"
    )
    
    parser.add_argument(
        "--layer-order",
        type=str,
        nargs='+',
        default=None,
        help="Optional explicit layer ordering for analysis"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching results"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable generation of visualization plots"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command-line arguments
    if args.focus_tokens:
        config["focus_tokens"] = args.focus_tokens
    
    if args.layer_order:
        config["layer_order"] = args.layer_order
    
    if args.cache_dir:
        config["cache_dir"] = args.cache_dir
    
    if args.verbose:
        config["verbose"] = True
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load data
    data = load_data(
        activation_path=args.activation_path,
        attention_path=args.attention_path,
        token_info_path=args.token_info_path,
        class_labels_path=args.class_labels_path
    )
    
    # Create analyzer
    analyzer = TransformerCrossLayerAnalyzer(
        cache_dir=config.get("cache_dir"),
        use_cache=config.get("use_cache", True),
        random_state=config.get("random_state", 42)
    )
    
    # Run analysis
    logger.info("Running cross-layer analysis...")
    
    result = analyzer.analyze_transformer_cross_layer(
        layer_activations=data["layer_activations"],
        attention_data=data["attention_data"],
        token_ids=data.get("token_ids"),
        token_strings=data.get("token_strings"),
        class_labels=data.get("class_labels"),
        focus_tokens=config.get("focus_tokens"),
        layer_order=config.get("layer_order"),
        config=config
    )
    
    # Save results
    results_path = save_results(result, args.output_dir)
    
    # Generate visualizations if enabled
    plot_paths = []
    if not args.no_plots:
        logger.info("Generating visualizations...")
        plot_paths = generate_plots(result, args.output_dir)
    
    # Generate markdown report
    logger.info("Generating analysis report...")
    report_path = generate_markdown_report(result, plot_paths, args.output_dir)
    
    # Print summary
    print("\nTransformer Cross-Layer Analysis Summary:")
    print("-----------------------------------------")
    
    summary = result.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print("\nAnalysis Results:")
    print(f"- Results saved to: {results_path}")
    print(f"- Report saved to: {report_path}")
    
    if plot_paths:
        print(f"- {len(plot_paths)} visualizations saved to: {args.output_dir}")

if __name__ == "__main__":
    main()