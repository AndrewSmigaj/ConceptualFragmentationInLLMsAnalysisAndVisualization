"""
Token path metrics for transformer models.

This module provides specialized metrics for analyzing how token representations
evolve through transformer layers, including path coherence, path divergence,
and semantic stability. These metrics quantify how consistently tokens maintain
their meaning and relationships as they progress through the network.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
import networkx as nx
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics import pairwise_distances

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class TokenPathMetricsResult:
    """
    Results from token path metrics calculations.
    
    Attributes:
        path_coherence: Dictionary mapping token paths to coherence scores
        path_divergence: Dictionary mapping layer transitions to divergence scores
        semantic_stability: Dictionary mapping tokens to stability scores across layers
        neighborhood_preservation: Dictionary mapping tokens to neighborhood preservation scores
        token_influence: Dictionary mapping tokens to influence scores
        token_specialization: Dictionary mapping tokens to specialization scores
        tokens_analyzed: List of token identifiers that were analyzed
        layers_analyzed: List of layer names that were analyzed
        aggregated_metrics: Dictionary with aggregated metrics across all tokens
    """
    path_coherence: Dict[str, float]
    path_divergence: Dict[Tuple[str, str], Dict[str, float]]
    semantic_stability: Dict[str, float]
    neighborhood_preservation: Dict[str, Dict[str, float]]
    token_influence: Dict[str, float]
    token_specialization: Dict[str, float]
    tokens_analyzed: List[str]
    layers_analyzed: List[str]
    aggregated_metrics: Dict[str, float]


def calculate_token_path_coherence(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    metric: str = "cosine"
) -> Dict[str, float]:
    """
    Calculate path coherence for tokens across layers.
    
    Path coherence measures how consistently a token's representation
    changes between consecutive layers. Higher coherence indicates
    more predictable and smooth evolution of token representations.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping token identifiers to path coherence scores
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Skip if we have fewer than 2 layers
    if len(layer_names) < 2:
        logger.warning("Need at least 2 layers to calculate path coherence")
        return {}
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return {}
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Calculate path coherence for each token
    coherence_scores = {}
    
    for token_idx in valid_tokens:
        token_key = f"token_{token_idx}"
        
        # Calculate directional changes between consecutive layers
        directions = []
        
        for i in range(len(layer_names) - 1):
            layer1 = layer_names[i]
            layer2 = layer_names[i + 1]
            
            # Get token representations from each layer (using first batch item)
            rep1 = processed_representations[layer1][0, token_idx]
            rep2 = processed_representations[layer2][0, token_idx]
            
            # Calculate direction vector of change
            direction = rep2 - rep1
            
            # Normalize direction vector
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                
            directions.append(direction)
        
        # Calculate consistency of directions (coherence)
        # Higher coherence = more consistent directional changes
        direction_similarities = []
        
        for i in range(len(directions) - 1):
            dir1 = directions[i]
            dir2 = directions[i + 1]
            
            if metric == "cosine":
                # Cosine similarity between consecutive direction vectors
                similarity = 1 - cosine(dir1, dir2)
            elif metric == "euclidean":
                # Convert Euclidean distance to similarity
                dist = euclidean(dir1, dir2)
                # Normalize to [0, 1] range where 1 = identical directions
                similarity = 1 / (1 + dist)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            direction_similarities.append(similarity)
        
        # Average similarity = coherence score
        if direction_similarities:
            coherence = np.mean(direction_similarities)
        else:
            coherence = 0.0
            
        coherence_scores[token_key] = float(coherence)
    
    return coherence_scores


def calculate_token_path_divergence(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    metric: str = "cosine",
    token_strings: Optional[List[str]] = None
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Calculate where token paths diverge most significantly.
    
    Path divergence identifies the layer transitions where token
    representations undergo the most dramatic changes, which can
    reveal critical transformation points in the network.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        metric: Distance metric to use ('cosine', 'euclidean')
        token_strings: Optional list of string tokens for better identification
        
    Returns:
        Dictionary mapping layer pairs to divergence information
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Skip if we have fewer than 2 layers
    if len(layer_names) < 2:
        logger.warning("Need at least 2 layers to calculate path divergence")
        return {}
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return {}
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Create token names dictionary
    token_names = {}
    for idx in valid_tokens:
        if token_strings and idx < len(token_strings):
            token_names[idx] = token_strings[idx]
        else:
            token_names[idx] = f"token_{idx}"
    
    # Calculate divergence for each layer transition
    divergence_info = {}
    
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        layer_key = (layer1, layer2)
        
        token_divergences = {}
        
        for token_idx in valid_tokens:
            # Get token representations from each layer (using first batch item)
            rep1 = processed_representations[layer1][0, token_idx]
            rep2 = processed_representations[layer2][0, token_idx]
            
            # Calculate divergence based on metric
            if metric == "cosine":
                # Cosine distance as divergence
                similarity = 1 - cosine(rep1, rep2)
                divergence = 1 - similarity
            elif metric == "euclidean":
                # Normalized Euclidean distance as divergence
                divergence = euclidean(rep1, rep2) / np.sqrt(len(rep1))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            token_divergences[token_names[token_idx]] = float(divergence)
        
        # Sort tokens by divergence (descending)
        sorted_tokens = sorted(
            token_divergences.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Calculate overall layer divergence statistics
        values = list(token_divergences.values())
        stats = {
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "std": float(np.std(values)),
            "top_tokens": dict(sorted_tokens[:5]),  # Top 5 most divergent tokens
            "token_divergences": token_divergences
        }
        
        divergence_info[layer_key] = stats
    
    return divergence_info


def calculate_semantic_stability(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    metric: str = "cosine",
    layer_windows: int = 2
) -> Dict[str, float]:
    """
    Calculate semantic stability of tokens across different layer spans.
    
    Semantic stability measures how well token meanings are preserved across
    multiple layer transitions. Stable tokens retain similar semantic properties
    throughout the network, while unstable tokens undergo significant transformations.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        metric: Distance metric to use ('cosine', 'euclidean')
        layer_windows: Number of layer transitions to consider for stability
        
    Returns:
        Dictionary mapping token identifiers to semantic stability scores
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Skip if we have fewer than 2 layers
    if len(layer_names) < layer_windows + 1:
        logger.warning(f"Need at least {layer_windows + 1} layers to calculate semantic stability")
        return {}
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return {}
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Calculate semantic stability for each token
    stability_scores = {}
    
    for token_idx in valid_tokens:
        token_key = f"token_{token_idx}"
        
        # Calculate similarity between layers with different window sizes
        window_similarities = []
        
        for i in range(len(layer_names) - layer_windows):
            layer1 = layer_names[i]
            layer2 = layer_names[i + layer_windows]
            
            # Get token representations from each layer (using first batch item)
            rep1 = processed_representations[layer1][0, token_idx]
            rep2 = processed_representations[layer2][0, token_idx]
            
            # Calculate similarity based on metric
            if metric == "cosine":
                similarity = 1 - cosine(rep1, rep2)
            elif metric == "euclidean":
                # Convert distance to similarity
                dist = euclidean(rep1, rep2)
                similarity = 1 / (1 + dist)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
                
            window_similarities.append(similarity)
        
        # Calculate various stability metrics
        if window_similarities:
            # Average similarity across all windows
            avg_stability = float(np.mean(window_similarities))
            
            # Variance of stability (lower = more consistent)
            var_stability = float(np.var(window_similarities))
            
            # Minimum stability (worst case)
            min_stability = float(np.min(window_similarities))
            
            # Overall stability score as weighted combination
            # Higher weight to average, but penalize variance and boost minimum
            stability = 0.6 * avg_stability - 0.2 * var_stability + 0.2 * min_stability
            
            stability_scores[token_key] = max(0.0, min(1.0, stability))  # Clip to [0, 1]
        else:
            stability_scores[token_key] = 0.0
    
    return stability_scores


def calculate_neighborhood_preservation(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_neighbors: int = 5,
    metric: str = "cosine"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate neighborhood preservation of tokens across layers.
    
    Neighborhood preservation measures how consistently tokens maintain
    their nearest neighbors as they progress through layers. This reveals
    how well semantic relationships are preserved during processing.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        n_neighbors: Number of nearest neighbors to consider
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping token identifiers to neighborhood preservation scores
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Skip if we have fewer than 2 layers
    if len(layer_names) < 2:
        logger.warning("Need at least 2 layers to calculate neighborhood preservation")
        return {}
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return {}
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Calculate neighborhood preservation for each token
    preservation_scores = {}
    
    for token_idx in valid_tokens:
        token_key = f"token_{token_idx}"
        layer_scores = {}
        
        # Get neighborhoods for each layer
        layer_neighborhoods = {}
        
        for layer_name in layer_names:
            # Extract all token representations for this layer (first batch item)
            layer_reps = processed_representations[layer_name][0]
            
            # Calculate distances from this token to all others
            if metric == "cosine":
                # Use pairwise distances for efficiency
                distances = pairwise_distances(
                    layer_reps[token_idx].reshape(1, -1),
                    layer_reps,
                    metric="cosine"
                )[0]
            elif metric == "euclidean":
                distances = pairwise_distances(
                    layer_reps[token_idx].reshape(1, -1),
                    layer_reps,
                    metric="euclidean"
                )[0]
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            # Get indices of n nearest neighbors (excluding self)
            sorted_indices = np.argsort(distances)
            # Remove self (should be first with distance 0) and get n neighbors
            neighbor_indices = sorted_indices[1:n_neighbors+1]
            
            # Store neighborhood
            layer_neighborhoods[layer_name] = set(neighbor_indices)
        
        # Calculate preservation between consecutive layers
        for i in range(len(layer_names) - 1):
            layer1 = layer_names[i]
            layer2 = layer_names[i + 1]
            
            # Get neighborhoods
            neighborhood1 = layer_neighborhoods[layer1]
            neighborhood2 = layer_neighborhoods[layer2]
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(neighborhood1.intersection(neighborhood2))
            union = len(neighborhood1.union(neighborhood2))
            
            if union > 0:
                jaccard = intersection / union
            else:
                jaccard = 0.0
                
            layer_scores[(layer1, layer2)] = float(jaccard)
        
        # Calculate average preservation across all layer transitions
        if layer_scores:
            avg_preservation = float(np.mean(list(layer_scores.values())))
            preservation_scores[token_key] = {
                "average": avg_preservation,
                "layer_transitions": layer_scores
            }
        else:
            preservation_scores[token_key] = {
                "average": 0.0,
                "layer_transitions": {}
            }
    
    return preservation_scores


def calculate_token_influence(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_neighbors: int = 10,
    metric: str = "cosine"
) -> Dict[str, float]:
    """
    Calculate influence of tokens on other tokens across layers.
    
    Token influence measures how much a token affects the representations
    of other tokens. Influential tokens have many other tokens as neighbors
    and play a central role in shaping the activation space.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        n_neighbors: Number of nearest neighbors to consider for influence
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping token identifiers to influence scores
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Skip if no layers
    if not layer_names:
        logger.warning("No layers provided to calculate token influence")
        return {}
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return {}
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Calculate token influence based on how often each token appears
    # in the neighborhood of other tokens
    influence_counts = defaultdict(int)
    
    # For each layer, count how many times each token is a neighbor of other tokens
    for layer_name in layer_names:
        # Extract all token representations for this layer (first batch item)
        layer_reps = processed_representations[layer_name][0]
        
        # Calculate pairwise distances between all tokens
        if metric == "cosine":
            distances = pairwise_distances(layer_reps, layer_reps, metric="cosine")
        elif metric == "euclidean":
            distances = pairwise_distances(layer_reps, layer_reps, metric="euclidean")
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # For each token, find its neighbors and increment their influence count
        for token_idx in valid_tokens:
            # Get distances from this token to all others
            token_distances = distances[token_idx]
            
            # Get indices of n nearest neighbors (excluding self)
            sorted_indices = np.argsort(token_distances)
            # Remove self (should be first with distance 0) and get n neighbors
            neighbor_indices = sorted_indices[1:n_neighbors+1]
            
            # Increment influence count for each neighbor
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in valid_tokens:
                    influence_counts[neighbor_idx] += 1
    
    # Normalize influence scores to [0, 1] range
    max_possible_count = len(valid_tokens) * len(layer_names)
    influence_scores = {}
    
    for token_idx in valid_tokens:
        token_key = f"token_{token_idx}"
        influence_scores[token_key] = float(influence_counts[token_idx] / max_possible_count)
    
    return influence_scores


def calculate_token_specialization(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Calculate specialization of tokens across layers.
    
    Token specialization measures how distinctive a token's representation
    becomes in later layers. Specialized tokens develop unique features
    that differentiate them from the average token representation.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        
    Returns:
        Dictionary mapping token identifiers to specialization scores
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Skip if no layers or only one layer
    if len(layer_names) < 2:
        logger.warning("Need at least 2 layers to calculate token specialization")
        return {}
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return {}
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Calculate token specialization
    specialization_scores = {}
    
    for token_idx in valid_tokens:
        token_key = f"token_{token_idx}"
        
        # Calculate initial and final specialization
        # Initial layer (usually embeddings)
        initial_layer = layer_names[0]
        # Final layer
        final_layer = layer_names[-1]
        
        # Get representations from first and last layer
        initial_reps = processed_representations[initial_layer][0]  # [seq_len, hidden_dim]
        final_reps = processed_representations[final_layer][0]  # [seq_len, hidden_dim]
        
        # Calculate average representations
        initial_avg = np.mean(initial_reps, axis=0)  # [hidden_dim]
        final_avg = np.mean(final_reps, axis=0)  # [hidden_dim]
        
        # Calculate token's representations
        initial_token_rep = initial_reps[token_idx]  # [hidden_dim]
        final_token_rep = final_reps[token_idx]  # [hidden_dim]
        
        # Calculate distance from average in first layer
        initial_dist = cosine(initial_token_rep, initial_avg)
        
        # Calculate distance from average in last layer
        final_dist = cosine(final_token_rep, final_avg)
        
        # Specialization = how much the token has moved away from the average
        # compared to its initial distance
        specialization = final_dist - initial_dist
        
        # Normalize to [0, 1] range by clipping
        normalized_specialization = max(0.0, min(1.0, specialization))
        
        specialization_scores[token_key] = float(normalized_specialization)
    
    return specialization_scores


def analyze_token_paths(
    layer_representations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_indices: Optional[List[int]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    token_strings: Optional[List[str]] = None,
    metric: str = "cosine",
    n_neighbors: int = 5,
    layer_windows: int = 2
) -> TokenPathMetricsResult:
    """
    Comprehensively analyze token paths through transformer layers.
    
    This function combines multiple path analysis metrics to provide
    a holistic view of how tokens evolve through the network.
    
    Args:
        layer_representations: Dictionary mapping layer names to token representations
                             [batch_size, seq_len, hidden_dim]
        token_indices: Optional list of token indices to analyze (default: all tokens)
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        token_strings: Optional list of string tokens for better identification
        metric: Distance metric to use ('cosine', 'euclidean')
        n_neighbors: Number of nearest neighbors to consider for neighborhood metrics
        layer_windows: Number of layer transitions to consider for stability
        
    Returns:
        TokenPathMetricsResult with comprehensive token path metrics
    """
    # Ensure all inputs are numpy arrays
    processed_representations = {}
    for layer_name, representations in layer_representations.items():
        if isinstance(representations, torch.Tensor):
            processed_representations[layer_name] = representations.detach().cpu().numpy()
        else:
            processed_representations[layer_name] = representations
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_representations.keys())
    
    # Get dimensions from first layer
    first_layer_rep = processed_representations[layer_names[0]]
    if len(first_layer_rep.shape) != 3:
        logger.warning(f"Expected 3D tensor [batch_size, seq_len, hidden_dim], got {first_layer_rep.shape}")
        return None
    
    batch_size, seq_len, _ = first_layer_rep.shape
    
    # If no token indices specified, use all tokens
    if token_indices is None:
        token_indices = list(range(seq_len))
    
    # Apply token mask if provided
    valid_tokens = token_indices
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Filter token indices based on mask
        # For simplicity, we'll use the first batch item's mask
        if len(token_mask.shape) > 1:
            mask = token_mask[0]
        else:
            mask = token_mask
        
        valid_tokens = [idx for idx in token_indices if idx < len(mask) and mask[idx] > 0]
    
    # Create token strings for analysis
    tokens_analyzed = []
    for idx in valid_tokens:
        if token_strings and idx < len(token_strings):
            tokens_analyzed.append(token_strings[idx])
        else:
            tokens_analyzed.append(f"token_{idx}")
    
    # Calculate path coherence
    logger.info("Calculating token path coherence")
    path_coherence = calculate_token_path_coherence(
        layer_representations=processed_representations,
        token_indices=valid_tokens,
        token_mask=token_mask,
        metric=metric
    )
    
    # Calculate path divergence
    logger.info("Calculating token path divergence")
    path_divergence = calculate_token_path_divergence(
        layer_representations=processed_representations,
        token_indices=valid_tokens,
        token_mask=token_mask,
        metric=metric,
        token_strings=token_strings
    )
    
    # Calculate semantic stability
    logger.info("Calculating token semantic stability")
    semantic_stability = calculate_semantic_stability(
        layer_representations=processed_representations,
        token_indices=valid_tokens,
        token_mask=token_mask,
        metric=metric,
        layer_windows=layer_windows
    )
    
    # Calculate neighborhood preservation
    logger.info("Calculating token neighborhood preservation")
    neighborhood_preservation = calculate_neighborhood_preservation(
        layer_representations=processed_representations,
        token_indices=valid_tokens,
        token_mask=token_mask,
        n_neighbors=n_neighbors,
        metric=metric
    )
    
    # Calculate token influence
    logger.info("Calculating token influence")
    token_influence = calculate_token_influence(
        layer_representations=processed_representations,
        token_indices=valid_tokens,
        token_mask=token_mask,
        n_neighbors=n_neighbors,
        metric=metric
    )
    
    # Calculate token specialization
    logger.info("Calculating token specialization")
    token_specialization = calculate_token_specialization(
        layer_representations=processed_representations,
        token_indices=valid_tokens,
        token_mask=token_mask
    )
    
    # Calculate aggregated metrics
    avg_coherence = np.mean(list(path_coherence.values())) if path_coherence else 0.0
    avg_stability = np.mean(list(semantic_stability.values())) if semantic_stability else 0.0
    avg_influence = np.mean(list(token_influence.values())) if token_influence else 0.0
    avg_specialization = np.mean(list(token_specialization.values())) if token_specialization else 0.0
    
    # Calculate average neighborhood preservation
    avg_preservation = 0.0
    if neighborhood_preservation:
        preservation_values = [np.mean(list(data["layer_transitions"].values())) 
                              for data in neighborhood_preservation.values() 
                              if data["layer_transitions"]]
        avg_preservation = np.mean(preservation_values) if preservation_values else 0.0
    
    # Average divergence across all layer transitions
    avg_divergence = 0.0
    if path_divergence:
        divergence_values = [data["mean"] for data in path_divergence.values()]
        avg_divergence = np.mean(divergence_values) if divergence_values else 0.0
    
    # Create aggregated metrics dictionary
    aggregated_metrics = {
        "average_coherence": float(avg_coherence),
        "average_divergence": float(avg_divergence),
        "average_stability": float(avg_stability),
        "average_preservation": float(avg_preservation),
        "average_influence": float(avg_influence),
        "average_specialization": float(avg_specialization)
    }
    
    # Create result
    result = TokenPathMetricsResult(
        path_coherence=path_coherence,
        path_divergence=path_divergence,
        semantic_stability=semantic_stability,
        neighborhood_preservation=neighborhood_preservation,
        token_influence=token_influence,
        token_specialization=token_specialization,
        tokens_analyzed=tokens_analyzed,
        layers_analyzed=layer_names,
        aggregated_metrics=aggregated_metrics
    )
    
    return result