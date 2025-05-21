"""
Path extraction and analysis for GPT-2 token-level representations.

This module provides specialized functions for extracting and analyzing paths
through clusters for token-level representations in GPT-2 models, extending
the Archetypal Path Analysis framework for transformer sequence data.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from collections import Counter, defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

# Import existing path analysis code
from concept_fragmentation.analysis.cross_layer_metrics import (
    compute_trajectory_fragmentation,
    compute_inter_cluster_path_density,
    extract_paths,
    validate_layer_order
)

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class TokenPathAnalysisResult:
    """
    Results from token path analysis.
    
    Attributes:
        token_paths: Dictionary mapping token indices to path arrays
        token_fragmentation: Dictionary mapping token indices to fragmentation scores
        token_path_entropy: Dictionary mapping token indices to path entropy values
        token_path_density: Dictionary mapping token indices to path density metrics
        top_token_paths: Dictionary mapping token indices to most common paths
        attention_weighted: Optional attention-weighted path metrics
        layer_names: List of layer names used in the analysis
        token_strings: Optional mapping of token indices to token strings
        metadata: Additional metadata about the analysis
    """
    token_paths: Dict[int, np.ndarray]
    token_fragmentation: Dict[int, float]
    token_path_entropy: Dict[int, float]
    token_path_density: Dict[int, Dict[str, Any]]
    top_token_paths: Dict[int, List[Dict[str, Any]]]
    attention_weighted: Optional[Dict[str, Any]] = None
    layer_names: List[str] = None
    token_strings: Optional[Dict[int, str]] = None
    metadata: Dict[str, Any] = None


def extract_token_paths(
    layer_clusters: Dict[str, Dict[int, np.ndarray]],
    layer_order: Optional[List[str]] = None
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Extract paths for token-level clusters.
    
    This function is specialized for token-level paths where the clustering
    has been done at the token level with the GPT2TokenClusterer.
    
    Args:
        layer_clusters: Dictionary mapping layer names to token cluster dictionaries
                       Each inner dictionary maps token indices to cluster arrays
        layer_order: Optional custom layer ordering
        
    Returns:
        Dictionary mapping token indices to path arrays, and list of layer names
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(list(layer_clusters.keys()))
    
    # Find common token indices across all layers
    token_indices = set()
    for layer_name in layer_order:
        if layer_name in layer_clusters:
            token_indices.update(layer_clusters[layer_name].keys())
    
    token_indices = sorted(token_indices)
    
    # Get batch size from the first available token in the first layer
    first_layer = layer_order[0]
    first_token = next(iter(layer_clusters[first_layer].keys()))
    batch_size = len(layer_clusters[first_layer][first_token])
    
    # Initialize token paths
    token_paths = {}
    
    for token_idx in token_indices:
        # Create array for this token's path
        token_path = np.zeros((batch_size, len(layer_order)), dtype=int)
        
        # Fill in the path for each layer
        for layer_idx, layer_name in enumerate(layer_order):
            if layer_name in layer_clusters and token_idx in layer_clusters[layer_name]:
                token_path[:, layer_idx] = layer_clusters[layer_name][token_idx]
        
        # Store this token's path
        token_paths[token_idx] = token_path
    
    return token_paths, layer_order


def extract_token_paths_from_results(
    layer_clusters: Dict[str, Any],
    layer_order: Optional[List[str]] = None
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """
    Extract token paths from clustering results.
    
    This function handles extracting paths from both GPT2TokenClusterer results
    and raw cluster assignments.
    
    Args:
        layer_clusters: Dictionary mapping layer names to clustering results
                       (either TokenClusteringResult objects or dictionaries)
        layer_order: Optional custom layer ordering
        
    Returns:
        Dictionary mapping token indices to path arrays, and list of layer names
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(list(layer_clusters.keys()))
    
    # Check the type of clustering results
    first_layer = layer_clusters[layer_order[0]]
    
    # Case 1: Results from GPT2TokenClusterer
    if hasattr(first_layer, "token_clusters") and hasattr(first_layer, "combined_clusters"):
        # Convert to the expected format
        token_cluster_dict = {}
        
        for layer_name, result in layer_clusters.items():
            token_cluster_dict[layer_name] = result.token_clusters
        
        # Call the standard extraction function
        return extract_token_paths(token_cluster_dict, layer_order)
    
    # Case 2: Already in the right format
    elif isinstance(first_layer, dict) and all(isinstance(k, int) for k in first_layer.keys()):
        return extract_token_paths(layer_clusters, layer_order)
    
    # Case 3: Raw combined cluster arrays
    elif isinstance(first_layer, np.ndarray) or (
        isinstance(first_layer, dict) and "combined_clusters" in first_layer):
        
        # Convert to token clusters format
        token_cluster_dict = {}
        
        for layer_name, result in layer_clusters.items():
            # Extract combined clusters
            if isinstance(result, np.ndarray):
                combined_clusters = result
            else:
                combined_clusters = result["combined_clusters"]
            
            # Extract token-level clusters
            batch_size, seq_len = combined_clusters.shape
            token_clusters = {}
            
            for tok_idx in range(seq_len):
                token_clusters[tok_idx] = combined_clusters[:, tok_idx]
            
            token_cluster_dict[layer_name] = token_clusters
        
        # Call the standard extraction function
        return extract_token_paths(token_cluster_dict, layer_order)
    
    # Unsupported format
    else:
        error = "Unsupported layer_clusters format"
        logger.error(error)
        return {}, layer_order


def compute_token_path_metrics(
    token_paths: Dict[int, np.ndarray],
    layer_names: List[str],
    token_strings: Optional[Dict[int, str]] = None,
    top_k_paths: int = 5
) -> TokenPathAnalysisResult:
    """
    Compute comprehensive path metrics for token-level paths.
    
    Args:
        token_paths: Dictionary mapping token indices to path arrays
        layer_names: List of layer names used in the paths
        token_strings: Optional mapping of token indices to token strings
        top_k_paths: Number of top paths to analyze for each token
        
    Returns:
        TokenPathAnalysisResult with comprehensive path metrics
    """
    # Initialize result containers
    token_fragmentation = {}
    token_path_entropy = {}
    token_path_density = {}
    top_token_paths = {}
    
    # Process each token
    for token_idx, paths in token_paths.items():
        token_name = token_strings[token_idx] if token_strings and token_idx in token_strings else f"token_{token_idx}"
        logger.info(f"Computing path metrics for {token_name}")
        
        # Calculate fragmentation metrics
        fragmentation = compute_trajectory_fragmentation(
            paths=paths,
            layer_names=layer_names
        )
        
        # Calculate path density
        path_density = compute_inter_cluster_path_density(
            paths=paths,
            layer_names=layer_names,
            min_density=0.05,
            max_steps=min(3, len(layer_names) - 1)
        )
        
        # Extract key metrics
        token_fragmentation[token_idx] = fragmentation["normalized_entropy"]
        token_path_entropy[token_idx] = fragmentation["overall_entropy"]
        token_path_density[token_idx] = {
            "summary": {
                "layer_pairs": len(path_density),
                "avg_density": np.mean([
                    np.mean(info["matrix"]) for info in path_density.values()
                ]) if path_density else 0.0
            },
            "details": path_density
        }
        
        # Find top paths for this token
        path_tuples = [tuple(path) for path in paths]
        path_counts = Counter(path_tuples)
        
        # Sort by frequency (descending)
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create human-readable path strings
        top_paths = []
        for path_tuple, count in sorted_paths[:top_k_paths]:
            path_str = "→".join(f"L{i}C{c}" for i, c in enumerate(path_tuple))
            top_paths.append({
                "path": list(path_tuple),
                "path_str": path_str,
                "count": count,
                "percentage": 100 * count / len(paths)
            })
        
        top_token_paths[token_idx] = top_paths
    
    # Return comprehensive results
    return TokenPathAnalysisResult(
        token_paths=token_paths,
        token_fragmentation=token_fragmentation,
        token_path_entropy=token_path_entropy,
        token_path_density=token_path_density,
        top_token_paths=top_token_paths,
        layer_names=layer_names,
        token_strings=token_strings,
        metadata={
            "n_tokens": len(token_paths),
            "n_layers": len(layer_names),
            "avg_fragmentation": np.mean(list(token_fragmentation.values())),
            "max_fragmentation": max(token_fragmentation.values()),
            "min_fragmentation": min(token_fragmentation.values())
        }
    )


def compute_attention_weighted_paths(
    token_paths: Dict[int, np.ndarray],
    attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
    layer_names: List[str],
    token_strings: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """
    Compute attention-weighted path metrics.
    
    Args:
        token_paths: Dictionary mapping token indices to path arrays
        attention_data: Dictionary mapping layer names to attention matrices
        layer_names: List of layer names used in the paths
        token_strings: Optional mapping of token indices to token strings
        
    Returns:
        Dictionary with attention-weighted path metrics
    """
    # Skip if no attention data
    if not attention_data:
        return {}
    
    # Initialize results
    attention_metrics = {}
    token_attention = {}
    weighted_fragmentation = {}
    
    # Compute token attention weights
    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name not in attention_data:
            continue
        
        # Get attention for this layer
        attention = attention_data[layer_name]
        
        # Convert to numpy if needed
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        # Get attention weights for tokens
        if len(attention.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
            # Average across heads
            avg_attention = attention.mean(axis=1)  # [batch_size, seq_len, seq_len]
            
            # Sum across source tokens to get token importance
            token_weights = avg_attention.sum(axis=1)  # [batch_size, seq_len]
            
            # Store for each token
            for token_idx in token_paths.keys():
                if token_idx < token_weights.shape[1]:
                    token_attention[(layer_name, token_idx)] = token_weights[:, token_idx]
        
        elif len(attention.shape) == 3:  # [n_heads, seq_len, seq_len]
            # Average across heads
            avg_attention = attention.mean(axis=0)  # [seq_len, seq_len]
            
            # Sum across source tokens to get token importance
            token_weights = avg_attention.sum(axis=0)  # [seq_len]
            
            # Store for each token
            for token_idx in token_paths.keys():
                if token_idx < len(token_weights):
                    token_attention[(layer_name, token_idx)] = np.array([token_weights[token_idx]])
    
    # Compute weighted fragmentation using token attention
    for token_idx, paths in token_paths.items():
        batch_size = paths.shape[0]
        
        # Get attention weights for this token
        token_weights = np.ones(batch_size)  # Default uniform weights
        
        # Average attention across available layers
        weight_count = 0
        for layer_name in layer_names:
            if (layer_name, token_idx) in token_attention:
                layer_weights = token_attention[(layer_name, token_idx)]
                # Only use weights if they match the batch size
                if len(layer_weights) == batch_size:
                    token_weights += layer_weights
                    weight_count += 1
        
        # Normalize weights
        if weight_count > 0:
            token_weights /= (weight_count + 1)  # +1 for the default weights
        
        # Normalize to sum to 1
        token_weights /= token_weights.sum()
        
        # Compute weighted path entropy
        path_tuples = [tuple(path) for path in paths]
        
        # Count paths with attention weights
        weighted_counts = defaultdict(float)
        for i, path in enumerate(path_tuples):
            weighted_counts[path] += token_weights[i]
        
        # Compute weighted entropy
        total_weight = sum(weighted_counts.values())
        probs = np.array([w / total_weight for w in weighted_counts.values()])
        weighted_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(min(len(paths), len(weighted_counts)))
        weighted_fragmentation[token_idx] = weighted_entropy / max_entropy if max_entropy > 0 else 0
    
    # Store results
    attention_metrics["token_attention"] = token_attention
    attention_metrics["weighted_fragmentation"] = weighted_fragmentation
    attention_metrics["avg_weighted_fragmentation"] = np.mean(list(weighted_fragmentation.values()))
    
    return attention_metrics


def create_token_path_visualizations(
    token_paths: Dict[int, np.ndarray],
    token_fragmentation: Dict[int, float],
    layer_names: List[str],
    token_strings: Optional[Dict[int, str]] = None,
    output_dir: str = "token_path_visualizations",
    top_k_tokens: int = 10,
    show_plots: bool = False
) -> Dict[str, str]:
    """
    Create visualizations for token paths.
    
    Args:
        token_paths: Dictionary mapping token indices to path arrays
        token_fragmentation: Dictionary mapping token indices to fragmentation scores
        layer_names: List of layer names used in the paths
        token_strings: Optional mapping of token indices to token strings
        output_dir: Directory to save visualizations
        top_k_tokens: Number of top tokens to visualize
        show_plots: Whether to display plots (in addition to saving)
        
    Returns:
        Dictionary mapping token indices to visualization file paths
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort tokens by fragmentation (descending)
    sorted_tokens = sorted(token_fragmentation.items(), key=lambda x: x[1], reverse=True)
    
    # Limit to top-k and bottom-k tokens
    top_tokens = [t for t, _ in sorted_tokens[:top_k_tokens]]
    bottom_tokens = [t for t, _ in sorted_tokens[-top_k_tokens:]]
    
    # Select tokens to visualize
    tokens_to_visualize = set(top_tokens + bottom_tokens)
    
    # Initialize result container
    visualization_paths = {}
    
    # Create visualizations for each selected token
    for token_idx in tokens_to_visualize:
        token_name = token_strings[token_idx] if token_strings and token_idx in token_strings else f"token_{token_idx}"
        logger.info(f"Creating visualization for {token_name}")
        
        # Get paths for this token
        paths = token_paths[token_idx]
        fragmentation = token_fragmentation[token_idx]
        
        # Convert paths to strings for counting
        path_tuples = [tuple(path) for path in paths]
        path_counts = Counter(path_tuples)
        
        # Sort by frequency (descending)
        sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create Sankey diagram for top paths
        plt.figure(figsize=(12, 8))
        
        # Create path flow matrix
        n_layers = len(layer_names)
        
        # Initialize counters for each layer transition
        transition_counts = {}
        
        for layer_idx in range(n_layers - 1):
            src_layer = layer_names[layer_idx]
            tgt_layer = layer_names[layer_idx + 1]
            transition_counts[(src_layer, tgt_layer)] = defaultdict(int)
        
        # Count transitions between clusters
        for path, count in sorted_paths:
            for layer_idx in range(n_layers - 1):
                src_layer = layer_names[layer_idx]
                tgt_layer = layer_names[layer_idx + 1]
                src_cluster = path[layer_idx]
                tgt_cluster = path[layer_idx + 1]
                transition_counts[(src_layer, tgt_layer)][(src_cluster, tgt_cluster)] += count
        
        # Prepare data for visualization
        plt.subplot(2, 1, 1)
        
        # Create heatmap showing path distribution
        path_matrix = np.zeros((min(10, len(sorted_paths)), n_layers))
        
        for i, (path, _) in enumerate(sorted_paths[:10]):
            for j, cluster in enumerate(path):
                path_matrix[i, j] = cluster
        
        # Create path labels
        path_labels = [f"Path {i+1}: {count} ({100*count/len(paths):.1f}%)" 
                     for i, (_, count) in enumerate(sorted_paths[:10])]
        
        # Create heatmap
        sns.heatmap(
            path_matrix,
            cmap="viridis",
            annot=True,
            fmt="d",
            xticklabels=layer_names,
            yticklabels=path_labels,
            cbar_kws={"label": "Cluster ID"}
        )
        
        plt.title(f"Token: {token_name} (Fragmentation: {fragmentation:.4f})")
        plt.ylabel("Path")
        plt.xlabel("Layer")
        
        # Create fragmentation comparison subplot
        plt.subplot(2, 1, 2)
        
        # Compute transition entropies
        transition_entropies = []
        
        for layer_idx in range(n_layers - 1):
            src_layer = layer_names[layer_idx]
            tgt_layer = layer_names[layer_idx + 1]
            
            # Get clusters at this layer
            src_clusters = paths[:, layer_idx]
            
            # Count transitions for each source cluster
            transitions = {}
            for src, tgt in zip(paths[:, layer_idx], paths[:, layer_idx + 1]):
                if src not in transitions:
                    transitions[src] = Counter()
                transitions[src][tgt] += 1
            
            # Compute entropy for each source cluster
            cluster_entropies = []
            
            for src, counts in transitions.items():
                total = sum(counts.values())
                probs = np.array([count / total for count in counts.values()])
                
                if len(probs) > 0:
                    ent = -np.sum(probs * np.log2(probs + 1e-10))
                    cluster_entropies.append(ent)
            
            # Average entropy for this transition
            if cluster_entropies:
                avg_entropy = np.mean(cluster_entropies)
                transition_entropies.append(avg_entropy)
            else:
                transition_entropies.append(0.0)
        
        # Plot transition entropies
        plt.bar(
            range(len(transition_entropies)),
            transition_entropies,
            tick_label=[f"{layer_names[i]}->{layer_names[i+1]}" for i in range(n_layers - 1)]
        )
        
        plt.title(f"Transition Entropy for {token_name}")
        plt.ylabel("Entropy")
        plt.xlabel("Layer Transition")
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        file_path = os.path.join(output_dir, f"token_{token_idx}_paths.png")
        plt.savefig(file_path)
        
        # Show if requested
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Store visualization path
        visualization_paths[token_idx] = file_path
    
    # Create comparative visualization of token fragmentation
    plt.figure(figsize=(12, 6))
    
    # Sort tokens by index for visualization
    token_indices = sorted(token_fragmentation.keys())
    
    # Get values and labels
    values = [token_fragmentation[idx] for idx in token_indices]
    labels = [token_strings[idx] if token_strings and idx in token_strings else f"token_{idx}" 
             for idx in token_indices]
    
    # Create bar chart
    plt.bar(range(len(values)), values, tick_label=labels)
    plt.title("Token Fragmentation Comparison")
    plt.ylabel("Fragmentation Score")
    plt.xlabel("Token")
    plt.xticks(rotation=90)
    
    # Draw mean line
    mean_frag = np.mean(values)
    plt.axhline(mean_frag, color='r', linestyle='--', 
                label=f"Mean: {mean_frag:.4f}")
    
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    file_path = os.path.join(output_dir, "token_fragmentation_comparison.png")
    plt.savefig(file_path)
    
    # Show if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Add to visualization paths
    visualization_paths["comparison"] = file_path
    
    return visualization_paths


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data
    batch_size = 4
    seq_len = 16
    n_layers = 3
    
    # Example layer clusters (simplified form for demonstration)
    layer_clusters = {}
    
    for i in range(n_layers):
        layer_name = f"layer{i}"
        
        # Create token clusters for this layer
        token_clusters = {}
        
        for tok_idx in range(seq_len):
            # Random cluster assignments for this token
            token_clusters[tok_idx] = np.random.randint(0, 8, size=batch_size)
        
        layer_clusters[layer_name] = token_clusters
    
    # Extract token paths
    token_paths, layer_names = extract_token_paths(layer_clusters)
    
    print(f"Extracted paths for {len(token_paths)} tokens across {len(layer_names)} layers")
    
    # Example token strings
    token_strings = {i: f"token_{chr(97 + i)}" for i in range(seq_len)}
    
    # Compute path metrics
    path_analysis = compute_token_path_metrics(
        token_paths=token_paths,
        layer_names=layer_names,
        token_strings=token_strings
    )
    
    print("\nToken Path Analysis Results:")
    print(f"Average fragmentation: {path_analysis.metadata['avg_fragmentation']:.4f}")
    print(f"Min fragmentation: {path_analysis.metadata['min_fragmentation']:.4f}")
    print(f"Max fragmentation: {path_analysis.metadata['max_fragmentation']:.4f}")
    
    # Show top 3 tokens with highest fragmentation
    top_tokens = sorted(path_analysis.token_fragmentation.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("\nTop 3 tokens by fragmentation:")
    for token_idx, fragmentation in top_tokens:
        token_name = token_strings[token_idx]
        top_path = path_analysis.top_token_paths[token_idx][0]
        print(f"{token_name}: {fragmentation:.4f} - Top path: {top_path['path_str']} ({top_path['percentage']:.2f}%)")
    
    # Create visualizations
    viz_paths = create_token_path_visualizations(
        token_paths=token_paths,
        token_fragmentation=path_analysis.token_fragmentation,
        layer_names=layer_names,
        token_strings=token_strings,
        show_plots=False
    )
    
    print(f"\nCreated {len(viz_paths)} visualizations")