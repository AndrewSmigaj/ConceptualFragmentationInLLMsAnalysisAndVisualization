"""
Integration of attention patterns with fragmentation metrics for GPT-2.

This module provides methods for integrating transformer attention patterns
with Archetypal Path Analysis, enabling attention-aware fragmentation metrics
and insights into how attention influences concept flow in GPT-2 models.
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

# Import existing metrics
from concept_fragmentation.analysis.cross_layer_metrics import (
    compute_trajectory_fragmentation,
    compute_inter_cluster_path_density,
    extract_paths,
    validate_layer_order
)

# Import path extraction
from concept_fragmentation.analysis.gpt2_path_extraction import (
    extract_token_paths,
    extract_token_paths_from_results,
    TokenPathAnalysisResult
)

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class AttentionFragmentationResult:
    """
    Results from attention-integrated fragmentation analysis.
    
    Attributes:
        standard_fragmentation: Standard fragmentation metrics without attention
        attention_weighted_fragmentation: Fragmentation metrics weighted by attention
        attention_flow_metrics: Metrics about attention flow across layers
        attention_path_correlation: Correlation between attention and path patterns
        layers_analyzed: List of layer names that were analyzed
        attention_influence: Metrics quantifying attention's influence on fragmentation
        window_analysis: Analysis results for specific layer windows
        visualization_data: Data prepared for visualizations
    """
    standard_fragmentation: Dict[str, Any]
    attention_weighted_fragmentation: Dict[str, Any]
    attention_flow_metrics: Dict[str, Any]
    attention_path_correlation: Dict[str, float]
    layers_analyzed: List[str]
    attention_influence: Dict[str, float]
    window_analysis: Optional[Dict[str, Dict[str, Any]]] = None
    visualization_data: Optional[Dict[str, Any]] = None


def compute_attention_matrices(
    attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
    layer_order: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Process raw attention data into standardized attention matrices.
    
    Args:
        attention_data: Dictionary mapping layer names to attention tensors
        layer_order: Optional layer ordering
        
    Returns:
        Dictionary mapping layer names to processed attention matrices
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(list(attention_data.keys()))
    
    # Initialize result container
    processed_attention = {}
    
    # Process each layer
    for layer_name in layer_order:
        if layer_name not in attention_data:
            continue
        
        # Get attention
        attention = attention_data[layer_name]
        
        # Convert to numpy if needed
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        # Process based on shape
        if len(attention.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
            # Average across heads
            processed_attention[layer_name] = attention.mean(axis=1)  # [batch_size, seq_len, seq_len]
        
        elif len(attention.shape) == 3:  # [n_heads, seq_len, seq_len] or [batch_size, seq_len, seq_len]
            # Check if first dimension is n_heads or batch_size
            if layer_name.endswith("_attention") or attention.shape[0] < 16:  # Heuristic for n_heads
                # This is [n_heads, seq_len, seq_len]
                processed_attention[layer_name] = attention.mean(axis=0)  # [seq_len, seq_len]
            else:
                # This is already [batch_size, seq_len, seq_len]
                processed_attention[layer_name] = attention
        
        elif len(attention.shape) == 2:  # [seq_len, seq_len]
            # Already in the right format
            processed_attention[layer_name] = np.expand_dims(attention, axis=0)  # [1, seq_len, seq_len]
    
    return processed_attention


def compute_attention_entropy(
    attention_matrices: Dict[str, np.ndarray],
    layer_order: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute entropy of attention distributions.
    
    Args:
        attention_matrices: Dictionary mapping layer names to attention matrices
        layer_order: Optional layer ordering
        
    Returns:
        Dictionary mapping layer names to attention entropy metrics
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(list(attention_matrices.keys()))
    
    # Initialize result container
    attention_entropy = {}
    
    # Process each layer
    for layer_name in layer_order:
        if layer_name not in attention_matrices:
            continue
        
        # Get attention matrix
        attention = attention_matrices[layer_name]
        
        # Initialize metrics for this layer
        layer_metrics = {}
        
        # Calculate entropy for each position
        batch_size, seq_len, _ = attention.shape
        position_entropies = np.zeros((batch_size, seq_len))
        
        for b in range(batch_size):
            for pos in range(seq_len):
                # Get attention distribution for this position
                attn_dist = attention[b, pos]
                
                # Ensure it's a valid probability distribution
                attn_dist = np.clip(attn_dist, 1e-10, 1.0)
                attn_dist = attn_dist / attn_dist.sum()
                
                # Calculate entropy
                entropy = -np.sum(attn_dist * np.log2(attn_dist))
                
                # Normalize by maximum possible entropy
                max_entropy = np.log2(seq_len)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                position_entropies[b, pos] = normalized_entropy
        
        # Store metrics
        layer_metrics["position_entropies"] = position_entropies
        layer_metrics["mean_entropy"] = float(np.mean(position_entropies))
        layer_metrics["max_entropy"] = float(np.max(position_entropies))
        layer_metrics["min_entropy"] = float(np.min(position_entropies))
        
        # Store layer results
        attention_entropy[layer_name] = layer_metrics
    
    return attention_entropy


def compute_attention_weighted_clusters(
    layer_clusters: Dict[str, np.ndarray],
    attention_matrices: Dict[str, np.ndarray],
    layer_order: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Weight cluster assignments by attention.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        attention_matrices: Dictionary mapping layer names to attention matrices
        layer_order: Optional layer ordering
        
    Returns:
        Dictionary mapping layer names to attention-weighted cluster metrics
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(
            set(layer_clusters.keys()) & set(attention_matrices.keys())
        )
    
    # Initialize result container
    weighted_clusters = {}
    
    # Process each layer
    for layer_idx, layer_name in enumerate(layer_order):
        # Skip if we don't have both clusters and attention
        if layer_name not in layer_clusters or layer_name not in attention_matrices:
            continue
        
        # Get clusters and attention
        clusters = layer_clusters[layer_name]
        attention = attention_matrices[layer_name]
        
        # Ensure clusters are 2D
        if len(clusters.shape) != 2:
            logger.warning(f"Expected 2D clusters for layer {layer_name}, got {clusters.shape}")
            continue
        
        # Get dimensions
        batch_size, seq_len = clusters.shape
        _, attn_seq_len, _ = attention.shape
        
        # Skip if dimensions don't match
        if seq_len != attn_seq_len:
            logger.warning(f"Sequence length mismatch for layer {layer_name}: {seq_len} vs {attn_seq_len}")
            continue
        
        # Initialize metrics for this layer
        layer_metrics = {}
        
        # Compute attention-weighted cluster importance
        cluster_importance = np.zeros((batch_size, seq_len))
        
        for b in range(batch_size):
            # Sum attention received by each position
            importance = attention[b].sum(axis=0)
            cluster_importance[b] = importance
        
        # Compute weighted cluster assignments
        unique_clusters = np.unique(clusters)
        cluster_attention = {}
        
        for cluster_id in unique_clusters:
            # Create mask for this cluster
            mask = (clusters == cluster_id)
            
            # Calculate mean attention for this cluster
            cluster_attn = np.sum(cluster_importance * mask) / np.sum(mask) if np.sum(mask) > 0 else 0
            
            cluster_attention[int(cluster_id)] = float(cluster_attn)
        
        # Normalize cluster attention
        total_attention = sum(cluster_attention.values())
        normalized_attention = {
            k: v / total_attention if total_attention > 0 else 0
            for k, v in cluster_attention.items()
        }
        
        # Store metrics
        layer_metrics["cluster_importance"] = cluster_importance
        layer_metrics["cluster_attention"] = cluster_attention
        layer_metrics["normalized_attention"] = normalized_attention
        
        # Store layer results
        weighted_clusters[layer_name] = layer_metrics
    
    return weighted_clusters


def compute_attention_weighted_fragmentation(
    paths: np.ndarray,
    attention_matrices: Dict[str, np.ndarray],
    layer_names: List[str]
) -> Dict[str, Any]:
    """
    Compute path fragmentation metrics weighted by attention.
    
    Args:
        paths: Array of paths through clusters [n_samples, n_layers]
        attention_matrices: Dictionary mapping layer names to attention matrices
        layer_names: List of layer names corresponding to path columns
        
    Returns:
        Dictionary with attention-weighted fragmentation metrics
    """
    n_samples, n_layers = paths.shape
    
    # Skip if we don't have enough layers
    if n_layers < 2 or len(layer_names) < 2:
        return {
            "error": "Not enough layers for fragmentation analysis",
            "weighted_entropy": 0.0,
            "weighted_fragmentation": 0.0
        }
    
    # Convert paths to tuples for counting
    path_tuples = [tuple(path) for path in paths]
    
    # Calculate standard path counts and probabilities
    path_counts = Counter(path_tuples)
    path_probs = np.array([count / n_samples for count in path_counts.values()])
    
    # Calculate standard entropy
    standard_entropy = -np.sum(path_probs * np.log2(path_probs + 1e-10))
    
    # Normalize by maximum possible entropy
    max_entropy = np.log2(min(n_samples, len(path_counts)))
    standard_fragmentation = standard_entropy / max_entropy if max_entropy > 0 else 0
    
    # Calculate attention weights for each sample
    sample_weights = np.ones(n_samples)
    weight_count = 0
    
    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name not in attention_matrices:
            continue
        
        attention = attention_matrices[layer_name]
        batch_size, seq_len, _ = attention.shape
        
        # Skip if dimensions don't match
        if n_samples % seq_len != 0:
            continue
        
        # Reshape paths to match attention dimensions
        reshaped_paths = paths.reshape(batch_size, seq_len, n_layers)
        
        # Calculate attention weight for each position
        for b in range(batch_size):
            for pos in range(seq_len):
                # Get attention received by this position
                attn_received = attention[b, :, pos].sum()
                
                # Update sample weight
                sample_idx = b * seq_len + pos
                if sample_idx < n_samples:
                    sample_weights[sample_idx] += attn_received
                    weight_count += 1
    
    # Normalize weights
    if weight_count > 0:
        sample_weights = sample_weights / (1 + weight_count / n_samples)
    
    # Normalize to sum to 1
    sample_weights = sample_weights / sample_weights.sum()
    
    # Calculate weighted path counts
    weighted_counts = defaultdict(float)
    for i, path in enumerate(path_tuples):
        weighted_counts[path] += sample_weights[i]
    
    # Calculate weighted entropy
    weighted_probs = np.array([weighted_counts[path] / sum(weighted_counts.values()) 
                              for path in weighted_counts.keys()])
    weighted_entropy = -np.sum(weighted_probs * np.log2(weighted_probs + 1e-10))
    
    # Normalize by maximum possible entropy
    weighted_fragmentation = weighted_entropy / max_entropy if max_entropy > 0 else 0
    
    # Calculate attention influence score (how much attention changes fragmentation)
    attention_influence = abs(weighted_fragmentation - standard_fragmentation) / standard_fragmentation if standard_fragmentation > 0 else 0
    
    # Compare most common paths with and without attention weighting
    standard_top_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    weighted_top_paths = sorted(weighted_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    standard_top_indices = [paths.tolist().index(list(path)) for path, _ in standard_top_paths]
    weighted_top_indices = []
    for path, _ in weighted_top_paths:
        try:
            weighted_top_indices.append(paths.tolist().index(list(path)))
        except ValueError:
            # This shouldn't happen, but just in case
            weighted_top_indices.append(-1)
    
    # Create human-readable path strings
    standard_top_paths_readable = []
    for path, count in standard_top_paths:
        path_str = "→".join(f"L{i}C{c}" for i, c in enumerate(path))
        standard_top_paths_readable.append({
            "path": list(path),
            "path_str": path_str,
            "count": count,
            "percentage": 100 * count / n_samples
        })
    
    weighted_top_paths_readable = []
    for path, weight in weighted_top_paths:
        path_str = "→".join(f"L{i}C{c}" for i, c in enumerate(path))
        weighted_top_paths_readable.append({
            "path": list(path),
            "path_str": path_str,
            "weight": weight,
            "percentage": 100 * weight / sum(weighted_counts.values())
        })
    
    # Return comprehensive metrics
    return {
        "standard_entropy": float(standard_entropy),
        "standard_fragmentation": float(standard_fragmentation),
        "weighted_entropy": float(weighted_entropy),
        "weighted_fragmentation": float(weighted_fragmentation),
        "attention_influence": float(attention_influence),
        "standard_top_paths": standard_top_paths_readable,
        "weighted_top_paths": weighted_top_paths_readable,
        "standard_top_indices": standard_top_indices,
        "weighted_top_indices": weighted_top_indices,
        "path_count": len(path_counts),
        "weighted_path_count": len(weighted_counts)
    }


def compute_attention_flow_metrics(
    attention_matrices: Dict[str, np.ndarray],
    layer_order: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute metrics about attention flow across layers.
    
    Args:
        attention_matrices: Dictionary mapping layer names to attention matrices
        layer_order: Optional layer ordering
        
    Returns:
        Dictionary with attention flow metrics
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(list(attention_matrices.keys()))
    
    # Skip if we don't have enough layers
    if len(layer_order) < 2:
        return {
            "error": "Not enough layers for attention flow analysis",
            "attention_flow": {},
            "attention_consistency": 0.0
        }
    
    # Initialize result containers
    attention_flow = {}
    layer_attention_entropy = {}
    
    # Calculate attention entropy for each layer
    for layer_name in layer_order:
        if layer_name not in attention_matrices:
            continue
        
        # Get attention
        attention = attention_matrices[layer_name]
        
        # Calculate entropy
        batch_size, seq_len, _ = attention.shape
        position_entropies = np.zeros((batch_size, seq_len))
        
        for b in range(batch_size):
            for pos in range(seq_len):
                # Get attention distribution for this position
                attn_dist = attention[b, pos]
                
                # Ensure it's a valid probability distribution
                attn_dist = np.clip(attn_dist, 1e-10, 1.0)
                attn_dist = attn_dist / attn_dist.sum()
                
                # Calculate entropy
                entropy = -np.sum(attn_dist * np.log2(attn_dist))
                
                # Normalize by maximum possible entropy
                max_entropy = np.log2(seq_len)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                position_entropies[b, pos] = normalized_entropy
        
        # Store average entropy
        layer_attention_entropy[layer_name] = float(np.mean(position_entropies))
    
    # Calculate attention flow direction between consecutive layers
    for i in range(len(layer_order) - 1):
        layer1 = layer_order[i]
        layer2 = layer_order[i + 1]
        
        # Skip if we don't have both layers
        if layer1 not in layer_attention_entropy or layer2 not in layer_attention_entropy:
            continue
        
        # Calculate entropy change
        entropy1 = layer_attention_entropy[layer1]
        entropy2 = layer_attention_entropy[layer2]
        entropy_diff = entropy2 - entropy1
        
        # Calculate flow direction
        # Negative diff means attention becomes more focused
        # Positive diff means attention becomes more dispersed
        flow_direction = -1 if entropy_diff < 0 else 1
        
        # Store flow metrics
        attention_flow[(layer1, layer2)] = {
            "entropy_diff": float(entropy_diff),
            "flow_direction": flow_direction,
            "source_entropy": float(entropy1),
            "target_entropy": float(entropy2)
        }
    
    # Calculate attention pattern consistency between layers
    attention_consistency = {}
    
    for i, layer1 in enumerate(layer_order):
        for j, layer2 in enumerate(layer_order):
            if i >= j or layer1 not in attention_matrices or layer2 not in attention_matrices:
                continue
            
            # Get attention matrices
            attn1 = attention_matrices[layer1]
            attn2 = attention_matrices[layer2]
            
            # Calculate correlation between attention patterns
            batch_size = min(attn1.shape[0], attn2.shape[0])
            correlations = []
            
            for b in range(batch_size):
                # Flatten attention matrices
                flat1 = attn1[b].flatten()
                flat2 = attn2[b].flatten()
                
                # Calculate correlation
                corr = np.corrcoef(flat1, flat2)[0, 1]
                correlations.append(corr)
            
            # Store average correlation
            attention_consistency[(layer1, layer2)] = float(np.mean(correlations))
    
    # Calculate overall consistency
    consistency_values = list(attention_consistency.values())
    overall_consistency = float(np.mean(consistency_values)) if consistency_values else 0.0
    
    # Return comprehensive metrics
    return {
        "layer_attention_entropy": layer_attention_entropy,
        "attention_flow": attention_flow,
        "attention_consistency": attention_consistency,
        "overall_consistency": overall_consistency
    }


def integrate_attention_with_fragmentation(
    layer_clusters: Dict[str, np.ndarray],
    attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
    layer_order: Optional[List[str]] = None,
    window_size: int = 3
) -> AttentionFragmentationResult:
    """
    Integrate attention patterns with fragmentation metrics.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        attention_data: Dictionary mapping layer names to attention tensors
        layer_order: Optional layer ordering
        window_size: Size of analysis windows
        
    Returns:
        AttentionFragmentationResult with combined metrics
    """
    # Ensure layer order is consistent
    if layer_order is None:
        layer_order = validate_layer_order(
            set(layer_clusters.keys()) & set(attention_data.keys())
        )
    
    # Process attention data
    attention_matrices = compute_attention_matrices(
        attention_data=attention_data,
        layer_order=layer_order
    )
    
    # Extract paths
    paths, used_layer_names = extract_paths(
        layer_clusters={l: layer_clusters[l] for l in layer_order if l in layer_clusters}
    )
    
    # Calculate standard fragmentation
    standard_fragmentation = compute_trajectory_fragmentation(
        paths=paths,
        layer_names=used_layer_names
    )
    
    # Calculate attention-weighted fragmentation
    attention_weighted_fragmentation = compute_attention_weighted_fragmentation(
        paths=paths,
        attention_matrices={l: attention_matrices[l] for l in layer_order if l in attention_matrices},
        layer_names=used_layer_names
    )
    
    # Calculate attention flow metrics
    attention_flow_metrics = compute_attention_flow_metrics(
        attention_matrices=attention_matrices,
        layer_order=layer_order
    )
    
    # Calculate attention-weighted clusters
    weighted_clusters = compute_attention_weighted_clusters(
        layer_clusters=layer_clusters,
        attention_matrices=attention_matrices,
        layer_order=layer_order
    )
    
    # Calculate attention-path correlation
    attention_path_correlation = {}
    
    # Compute correlation between attention and path patterns
    if len(paths) > 0 and len(paths[0]) > 1:
        # Calculate path transition frequencies
        transition_freqs = {}
        
        for i in range(len(used_layer_names) - 1):
            layer1 = used_layer_names[i]
            layer2 = used_layer_names[i + 1]
            
            # Get transitions at this layer
            transition_counts = defaultdict(int)
            
            for path in paths:
                transition = (path[i], path[i + 1])
                transition_counts[transition] += 1
            
            # Normalize to frequencies
            total = sum(transition_counts.values())
            transition_freqs[(layer1, layer2)] = {
                transition: count / total 
                for transition, count in transition_counts.items()
            }
        
        # Calculate attention between clusters
        for i in range(len(used_layer_names) - 1):
            layer1 = used_layer_names[i]
            layer2 = used_layer_names[i + 1]
            
            # Skip if we don't have attention for both layers
            if layer1 not in attention_matrices or layer2 not in attention_matrices:
                continue
            
            # Get cluster assignments
            clusters1 = layer_clusters[layer1]
            clusters2 = layer_clusters[layer2]
            
            # Get attention from layer1 to layer2
            attention1 = attention_matrices[layer1]
            
            # Calculate attention between clusters
            unique_clusters1 = np.unique(clusters1)
            unique_clusters2 = np.unique(clusters2)
            
            cluster_attention = defaultdict(float)
            
            for c1 in unique_clusters1:
                for c2 in unique_clusters2:
                    transition = (c1, c2)
                    
                    # Skip if we don't have this transition
                    if transition not in transition_freqs.get((layer1, layer2), {}):
                        continue
                    
                    # Get positions for each cluster
                    pos1 = np.where(clusters1 == c1)
                    pos2 = np.where(clusters2 == c2)
                    
                    # Calculate average attention from positions in c1 to positions in c2
                    attn_sum = 0.0
                    count = 0
                    
                    for b1, p1 in zip(*pos1):
                        for b2, p2 in zip(*pos2):
                            if b1 == b2:  # Same batch element
                                attn_sum += attention1[b1, p1, p2]
                                count += 1
                    
                    if count > 0:
                        cluster_attention[transition] = attn_sum / count
            
            # Normalize cluster attention
            total_attention = sum(cluster_attention.values())
            if total_attention > 0:
                cluster_attention = {
                    k: v / total_attention for k, v in cluster_attention.items()
                }
            
            # Calculate correlation between transition frequencies and attention
            common_transitions = set(transition_freqs.get((layer1, layer2), {}).keys()) & set(cluster_attention.keys())
            
            if common_transitions:
                freq_values = [transition_freqs[(layer1, layer2)][t] for t in common_transitions]
                attn_values = [cluster_attention[t] for t in common_transitions]
                
                # Calculate correlation
                correlation = np.corrcoef(freq_values, attn_values)[0, 1]
                
                # Store result
                attention_path_correlation[(layer1, layer2)] = float(correlation)
    
    # Calculate global correlation
    correlation_values = list(attention_path_correlation.values())
    global_correlation = float(np.mean(correlation_values)) if correlation_values else 0.0
    attention_path_correlation["global"] = global_correlation
    
    # Calculate attention influence metrics
    attention_influence = {
        "fragmentation_change": abs(
            attention_weighted_fragmentation["weighted_fragmentation"] - 
            attention_weighted_fragmentation["standard_fragmentation"]
        ),
        "fragmentation_ratio": (
            attention_weighted_fragmentation["weighted_fragmentation"] / 
            attention_weighted_fragmentation["standard_fragmentation"]
        ) if attention_weighted_fragmentation["standard_fragmentation"] > 0 else 1.0,
        "attention_path_correlation": global_correlation,
        "path_ranking_change": len(
            set(attention_weighted_fragmentation["standard_top_indices"]) ^ 
            set(attention_weighted_fragmentation["weighted_top_indices"])
        ) / len(attention_weighted_fragmentation["standard_top_indices"])
    }
    
    # Implement window analysis if requested
    window_analysis = None
    
    if window_size > 0 and len(used_layer_names) >= window_size:
        window_analysis = {}
        
        for i in range(len(used_layer_names) - window_size + 1):
            # Define window
            window_layers = used_layer_names[i:i+window_size]
            window_name = f"window_{i+1}"
            
            # Extract clusters and attention for this window
            window_clusters = {l: layer_clusters[l] for l in window_layers if l in layer_clusters}
            window_attention = {l: attention_matrices[l] for l in window_layers if l in attention_matrices}
            
            # Skip if we don't have both clusters and attention
            if len(window_clusters) < window_size or len(window_attention) < window_size:
                continue
            
            # Extract paths for this window
            window_paths, _ = extract_paths(window_clusters)
            
            # Calculate attention-weighted fragmentation
            window_weighted_fragmentation = compute_attention_weighted_fragmentation(
                paths=window_paths,
                attention_matrices=window_attention,
                layer_names=window_layers
            )
            
            # Store results
            window_analysis[window_name] = window_weighted_fragmentation
    
    # Prepare visualization data
    visualization_data = {
        "attention_entropy": {
            layer: metrics.get("mean_entropy", 0.0)
            for layer, metrics in compute_attention_entropy(attention_matrices, layer_order).items()
        },
        "weighted_vs_standard": {
            "standard_fragmentation": attention_weighted_fragmentation["standard_fragmentation"],
            "weighted_fragmentation": attention_weighted_fragmentation["weighted_fragmentation"]
        },
        "top_paths": {
            "standard": attention_weighted_fragmentation["standard_top_paths"],
            "weighted": attention_weighted_fragmentation["weighted_top_paths"]
        },
        "attention_flow": {
            f"{layer1}_{layer2}": metrics["flow_direction"]
            for (layer1, layer2), metrics in attention_flow_metrics["attention_flow"].items()
        }
    }
    
    # Return comprehensive results
    return AttentionFragmentationResult(
        standard_fragmentation=standard_fragmentation,
        attention_weighted_fragmentation=attention_weighted_fragmentation,
        attention_flow_metrics=attention_flow_metrics,
        attention_path_correlation=attention_path_correlation,
        layers_analyzed=used_layer_names,
        attention_influence=attention_influence,
        window_analysis=window_analysis,
        visualization_data=visualization_data
    )


def create_attention_fragmentation_visualizations(
    result: AttentionFragmentationResult,
    output_dir: str = "attention_fragmentation_visualizations",
    show_plots: bool = False
) -> Dict[str, str]:
    """
    Create visualizations for attention-integrated fragmentation analysis.
    
    Args:
        result: AttentionFragmentationResult to visualize
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots (in addition to saving)
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result container
    visualization_paths = {}
    
    # 1. Create attention entropy vs. layer visualization
    plt.figure(figsize=(10, 6))
    
    entropy_data = result.visualization_data["attention_entropy"]
    
    if entropy_data:
        layers = list(entropy_data.keys())
        entropy_values = [entropy_data[layer] for layer in layers]
        
        plt.bar(range(len(layers)), entropy_values, tick_label=layers)
        plt.title("Attention Entropy by Layer")
        plt.ylabel("Normalized Entropy")
        plt.xlabel("Layer")
        plt.xticks(rotation=45)
        
        # Save figure
        file_path = os.path.join(output_dir, "attention_entropy.png")
        plt.savefig(file_path)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        visualization_paths["attention_entropy"] = file_path
    
    # 2. Create attention flow visualization
    plt.figure(figsize=(12, 8))
    
    flow_data = result.visualization_data["attention_flow"]
    
    if flow_data:
        # Create flow network
        import networkx as nx
        G = nx.DiGraph()
        
        # Add nodes for each layer
        layers = result.layers_analyzed
        for i, layer in enumerate(layers):
            G.add_node(layer, pos=(i, 0))
        
        # Add edges with attributes
        for edge, direction in flow_data.items():
            layer1, layer2 = edge.split("_")
            
            if layer1 in layers and layer2 in layers:
                G.add_edge(
                    layer1,
                    layer2,
                    direction=direction,
                    color="red" if direction < 0 else "green"
                )
        
        # Get positions
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue")
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Draw edges with different colors
        for u, v, data in G.edges(data=True):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=2.0,
                edge_color=data["color"],
                arrows=True,
                arrowsize=20
            )
        
        plt.title("Attention Flow Direction")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        file_path = os.path.join(output_dir, "attention_flow.png")
        plt.savefig(file_path)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        visualization_paths["attention_flow"] = file_path
    
    # 3. Create weighted vs. standard fragmentation
    plt.figure(figsize=(8, 6))
    
    weighted_data = result.visualization_data["weighted_vs_standard"]
    
    if weighted_data:
        labels = ["Standard", "Attention-weighted"]
        values = [weighted_data["standard_fragmentation"], weighted_data["weighted_fragmentation"]]
        
        plt.bar(range(len(labels)), values, tick_label=labels)
        plt.title("Standard vs. Attention-weighted Fragmentation")
        plt.ylabel("Fragmentation Score")
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.4f}", ha="center")
        
        # Save figure
        file_path = os.path.join(output_dir, "fragmentation_comparison.png")
        plt.savefig(file_path)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        visualization_paths["fragmentation_comparison"] = file_path
    
    # 4. Create top paths comparison
    plt.figure(figsize=(12, 8))
    
    top_paths = result.visualization_data["top_paths"]
    
    if top_paths and top_paths["standard"] and top_paths["weighted"]:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Standard top paths
        standard_paths = top_paths["standard"]
        standard_labels = [p["path_str"] for p in standard_paths]
        standard_values = [p["percentage"] for p in standard_paths]
        
        ax1.barh(range(len(standard_labels)), standard_values)
        ax1.set_yticks(range(len(standard_labels)))
        ax1.set_yticklabels(standard_labels)
        ax1.set_title("Top Paths (Standard)")
        ax1.set_xlabel("Percentage (%)")
        
        # Weighted top paths
        weighted_paths = top_paths["weighted"]
        weighted_labels = [p["path_str"] for p in weighted_paths]
        weighted_values = [p["percentage"] for p in weighted_paths]
        
        ax2.barh(range(len(weighted_labels)), weighted_values)
        ax2.set_yticks(range(len(weighted_labels)))
        ax2.set_yticklabels(weighted_labels)
        ax2.set_title("Top Paths (Attention-weighted)")
        ax2.set_xlabel("Percentage (%)")
        
        plt.tight_layout()
        
        # Save figure
        file_path = os.path.join(output_dir, "top_paths_comparison.png")
        plt.savefig(file_path)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        visualization_paths["top_paths_comparison"] = file_path
    
    # 5. Create window analysis comparison if available
    if result.window_analysis:
        plt.figure(figsize=(10, 6))
        
        # Extract window names and fragmentation values
        windows = list(result.window_analysis.keys())
        standard_values = [result.window_analysis[w]["standard_fragmentation"] for w in windows]
        weighted_values = [result.window_analysis[w]["weighted_fragmentation"] for w in windows]
        
        # Set up bar positions
        x = np.arange(len(windows))
        width = 0.35
        
        # Create grouped bar chart
        plt.bar(x - width/2, standard_values, width, label="Standard")
        plt.bar(x + width/2, weighted_values, width, label="Attention-weighted")
        
        plt.xlabel("Window")
        plt.ylabel("Fragmentation Score")
        plt.title("Fragmentation by Window")
        plt.xticks(x, windows)
        plt.legend()
        
        # Save figure
        file_path = os.path.join(output_dir, "window_comparison.png")
        plt.savefig(file_path)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        visualization_paths["window_comparison"] = file_path
    
    return visualization_paths


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create example data
    batch_size = 4
    seq_len = 16
    n_layers = 3
    
    # Example layer clusters
    layer_clusters = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        layer_clusters[layer_name] = np.random.randint(0, 8, size=(batch_size, seq_len))
    
    # Example attention matrices
    attention_data = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        
        # Generate random attention and normalize
        random_attention = np.random.rand(batch_size, seq_len, seq_len)
        
        # Convert to proper attention distribution
        for b in range(batch_size):
            for j in range(seq_len):
                random_attention[b, j] = random_attention[b, j] / random_attention[b, j].sum()
        
        attention_data[layer_name] = random_attention
    
    # Run analysis
    result = integrate_attention_with_fragmentation(
        layer_clusters=layer_clusters,
        attention_data=attention_data
    )
    
    # Print results
    print("\nAttention-Integrated Fragmentation Analysis:")
    print(f"Standard fragmentation: {result.attention_weighted_fragmentation['standard_fragmentation']:.4f}")
    print(f"Attention-weighted fragmentation: {result.attention_weighted_fragmentation['weighted_fragmentation']:.4f}")
    print(f"Attention influence: {result.attention_influence['fragmentation_change']:.4f}")
    print(f"Attention-path correlation: {result.attention_path_correlation['global']:.4f}")
    
    # Create visualizations
    viz_paths = create_attention_fragmentation_visualizations(
        result=result,
        show_plots=False
    )
    
    print(f"\nCreated {len(viz_paths)} visualizations")