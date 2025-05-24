"""
Transformer-specific metrics for analyzing attention patterns and activations.

This module provides metrics specifically designed for transformer models,
including attention entropy, sparsity measures, and head attribution metrics.
These metrics help evaluate the architectural properties of transformer models
and provide insights into their internal mechanisms.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
from dataclasses import dataclass
import warnings
from scipy.stats import entropy
from scipy.spatial.distance import cosine, euclidean

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class AttentionMetricsResult:
    """
    Results from attention metrics calculations.
    
    Attributes:
        entropy: Entropy of attention distributions
        sparsity: Sparsity of attention matrices
        head_importance: Importance scores for each attention head
        layer_entropy: Entropy per layer (dict mapping layer to entropy)
        layer_sparsity: Sparsity per layer
        head_entropy: Entropy per head (dict mapping (layer, head) to entropy)
        head_sparsity: Sparsity per head
        input_tokens: Input token information if available
        aggregation_method: Method used for aggregation
        layers_analyzed: List of layer names that were analyzed
        heads_analyzed: List of (layer, head) tuples that were analyzed
        cross_head_agreement: Agreement scores between attention heads (if computed)
    """
    entropy: float
    sparsity: float
    head_importance: Dict[Tuple[str, int], float]
    layer_entropy: Dict[str, float]
    layer_sparsity: Dict[str, float]
    head_entropy: Dict[Tuple[str, int], float]
    head_sparsity: Dict[Tuple[str, int], float]
    input_tokens: Optional[List[str]] = None
    aggregation_method: str = "mean"
    layers_analyzed: List[str] = None
    heads_analyzed: List[Tuple[str, int]] = None
    cross_head_agreement: Optional[Dict[str, Dict[Tuple[int, int], float]]] = None


def calculate_attention_entropy(
    attention_probs: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate entropy of attention distributions.
    
    Higher entropy indicates more uniform attention distribution,
    while lower entropy indicates more focused attention.
    
    Args:
        attention_probs: Attention probabilities [batch_size, n_heads, seq_len, seq_len]
                        or [n_heads, seq_len, seq_len]
                        
    Returns:
        Entropy values with shape [batch_size, n_heads] or [n_heads]
    """
    # Determine if input is torch tensor
    is_torch = isinstance(attention_probs, torch.Tensor)
    
    # Convert to numpy for consistent processing if it's a torch tensor
    if is_torch:
        attention_probs = attention_probs.detach().cpu().numpy()
    
    # Handle different input shapes
    if len(attention_probs.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
        batch_size, n_heads, seq_len, _ = attention_probs.shape
        
        # Calculate entropy along last dimension (attended tokens)
        # entropy() expects each row to be a probability distribution
        ent = np.zeros((batch_size, n_heads))
        
        for b in range(batch_size):
            for h in range(n_heads):
                for s in range(seq_len):
                    # Calculate entropy for each query's attention distribution
                    probs = attention_probs[b, h, s]
                    
                    # Handle zero probabilities (replace with small epsilon to avoid NaN)
                    probs = np.clip(probs, 1e-10, 1.0)
                    probs = probs / probs.sum()  # Re-normalize
                    
                    ent[b, h] += entropy(probs) / seq_len
        
    elif len(attention_probs.shape) == 3:  # [n_heads, seq_len, seq_len]
        n_heads, seq_len, _ = attention_probs.shape
        
        # Calculate entropy along last dimension
        ent = np.zeros(n_heads)
        
        for h in range(n_heads):
            for s in range(seq_len):
                # Calculate entropy for each query's attention distribution
                probs = attention_probs[h, s]
                
                # Handle zero probabilities
                probs = np.clip(probs, 1e-10, 1.0)
                probs = probs / probs.sum()  # Re-normalize
                
                ent[h] += entropy(probs) / seq_len
    else:
        raise ValueError(f"Unsupported attention_probs shape: {attention_probs.shape}")
    
    # Convert back to torch if input was torch
    if is_torch:
        ent = torch.from_numpy(ent)
    
    return ent


def calculate_attention_sparsity(
    attention_probs: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.01
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate sparsity of attention matrices.
    
    Sparsity is defined as the fraction of elements smaller than the threshold.
    Higher sparsity means attention is more focused on a few tokens.
    
    Args:
        attention_probs: Attention probabilities [batch_size, n_heads, seq_len, seq_len]
                        or [n_heads, seq_len, seq_len]
        threshold: Threshold below which attention weights are considered "zero"
                        
    Returns:
        Sparsity values with shape [batch_size, n_heads] or [n_heads]
    """
    # Determine if input is torch tensor
    is_torch = isinstance(attention_probs, torch.Tensor)
    
    # Convert to numpy for consistent processing if it's a torch tensor
    if is_torch:
        attention_probs = attention_probs.detach().cpu().numpy()
    
    # Handle different input shapes
    if len(attention_probs.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
        batch_size, n_heads, seq_len, _ = attention_probs.shape
        
        sparsity_values = np.zeros((batch_size, n_heads))
        
        for b in range(batch_size):
            for h in range(n_heads):
                # Count elements below threshold
                below_threshold = (attention_probs[b, h] < threshold).sum()
                total_elements = seq_len * seq_len
                
                # Calculate sparsity as fraction of "zeros"
                sparsity_values[b, h] = float(below_threshold) / total_elements
        
    elif len(attention_probs.shape) == 3:  # [n_heads, seq_len, seq_len]
        n_heads, seq_len, _ = attention_probs.shape
        
        sparsity_values = np.zeros(n_heads)
        
        for h in range(n_heads):
            # Count elements below threshold
            below_threshold = (attention_probs[h] < threshold).sum()
            total_elements = seq_len * seq_len
            
            # Calculate sparsity as fraction of "zeros"
            sparsity_values[h] = float(below_threshold) / total_elements
    else:
        raise ValueError(f"Unsupported attention_probs shape: {attention_probs.shape}")
    
    # Convert back to torch if input was torch
    if is_torch:
        sparsity_values = torch.from_numpy(sparsity_values)
    
    return sparsity_values


def calculate_head_importance(
    attention_probs: Union[torch.Tensor, np.ndarray],
    outputs: Union[torch.Tensor, np.ndarray],
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Union[torch.Tensor, np.ndarray]:
    """
    Calculate importance of each attention head based on impact on outputs.
    
    For each attention head, we compute the importance as the gradient of the output
    with respect to the attention weights. This shows how much each head influences
    the final output.
    
    Args:
        attention_probs: Attention probabilities [batch_size, n_heads, seq_len, seq_len]
        outputs: Model outputs [batch_size, seq_len, hidden_size]
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        
    Returns:
        Importance scores for each head [batch_size, n_heads]
    """
    # Ensure we have torch tensors with gradients
    if not isinstance(attention_probs, torch.Tensor):
        attention_probs = torch.tensor(attention_probs, dtype=torch.float32)
    
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs, dtype=torch.float32)
    
    if token_mask is not None and not isinstance(token_mask, torch.Tensor):
        token_mask = torch.tensor(token_mask, dtype=torch.float32)
    
    # Get shapes
    batch_size, n_heads, seq_len, _ = attention_probs.shape
    
    # We need gradients for attention probs
    requires_grad = attention_probs.requires_grad
    if not requires_grad:
        attention_probs = attention_probs.detach().clone().requires_grad_(True)
    
    # Initialize importance scores
    importance = torch.zeros(batch_size, n_heads)
    
    # For each head, compute gradient of outputs w.r.t. attention
    for h in range(n_heads):
        # For computational efficiency, we'll use the Frobenius norm of the attention matrix
        # as a proxy for its influence on the output
        attention_norm = torch.norm(attention_probs[:, h], dim=(1, 2))
        
        # Use output norm as a measure of influence
        output_norm = torch.norm(outputs, dim=(1, 2))
        
        # Compute correlation between attention norm and output norm
        correlation = torch.zeros(batch_size)
        for b in range(batch_size):
            if token_mask is not None:
                # Only consider tokens that are not padding
                mask = token_mask[b].bool()
                if mask.sum() > 0:
                    correlation[b] = torch.corrcoef(torch.stack([
                        attention_norm[b], output_norm[b]
                    ]))[0, 1]
            else:
                correlation[b] = torch.corrcoef(torch.stack([
                    attention_norm[b], output_norm[b]
                ]))[0, 1]
        
        # Set importance scores
        importance[:, h] = correlation
    
    # Normalize importance scores to [0, 1]
    min_importance = importance.min(dim=1, keepdim=True)[0]
    max_importance = importance.max(dim=1, keepdim=True)[0]
    importance = (importance - min_importance) / (max_importance - min_importance + 1e-10)
    
    return importance


def calculate_cross_head_agreement(
    attention_probs: Union[torch.Tensor, np.ndarray],
    agreement_metric: str = "cosine",
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> Dict[Tuple[int, int], float]:
    """
    Calculate agreement between different attention heads.
    
    This metric quantifies how similarly different attention heads attend
    to the input sequence. High agreement indicates redundancy between heads,
    while low agreement suggests heads are specializing in different patterns.
    
    Args:
        attention_probs: Attention probabilities [batch_size, n_heads, seq_len, seq_len]
                        or [n_heads, seq_len, seq_len]
        agreement_metric: Metric to use for comparing heads ('cosine', 'correlation', 'kl')
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        
    Returns:
        Dictionary mapping head pairs (h1, h2) to agreement scores
    """
    # Determine if input is torch tensor
    is_torch = isinstance(attention_probs, torch.Tensor)
    
    # Convert to numpy for consistent processing if it's a torch tensor
    if is_torch:
        attention_probs = attention_probs.detach().cpu().numpy()
    
    # Apply token mask if provided
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Create masking tensors based on the input shape
        if len(attention_probs.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
            # Create mask with shape [batch_size, 1, seq_len, seq_len]
            mask_2d = np.expand_dims(token_mask, -1) * np.expand_dims(token_mask, -2)
            mask_3d = np.expand_dims(mask_2d, 1)
            # Apply mask
            attention_probs = attention_probs * mask_3d
        else:  # [n_heads, seq_len, seq_len]
            # We don't have batch dimension, so just use the first batch element
            if len(token_mask.shape) > 1:
                token_mask = token_mask[0]
            mask_2d = np.expand_dims(token_mask, -1) * np.expand_dims(token_mask, -2)
            # Apply mask
            attention_probs = attention_probs * mask_2d
    
    # Calculate head agreement
    head_agreement = {}
    
    if len(attention_probs.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
        batch_size, n_heads, seq_len, _ = attention_probs.shape
        
        for h1 in range(n_heads):
            for h2 in range(h1+1, n_heads):  # Only compute upper triangle
                agreements = []
                
                for b in range(batch_size):
                    # Get attention matrices for each head
                    attn1 = attention_probs[b, h1].flatten()
                    attn2 = attention_probs[b, h2].flatten()
                    
                    # Calculate agreement based on chosen metric
                    if agreement_metric == "cosine":
                        # Cosine similarity
                        norm1 = np.linalg.norm(attn1)
                        norm2 = np.linalg.norm(attn2)
                        
                        if norm1 > 0 and norm2 > 0:
                            agreement = np.dot(attn1, attn2) / (norm1 * norm2)
                        else:
                            agreement = 0.0
                    
                    elif agreement_metric == "correlation":
                        # Pearson correlation
                        if np.std(attn1) > 0 and np.std(attn2) > 0:
                            agreement = np.corrcoef(attn1, attn2)[0, 1]
                        else:
                            agreement = 0.0
                    
                    elif agreement_metric == "kl":
                        # KL divergence (symmetric version)
                        # Ensure valid probability distributions
                        attn1 = np.clip(attn1, 1e-10, 1.0)
                        attn1 = attn1 / np.sum(attn1)
                        attn2 = np.clip(attn2, 1e-10, 1.0)
                        attn2 = attn2 / np.sum(attn2)
                        
                        # Compute symmetric KL
                        kl1 = entropy(attn1, attn2)
                        kl2 = entropy(attn2, attn1)
                        
                        # Convert divergence to similarity (agreement)
                        # Higher divergence = lower agreement
                        agreement = 1.0 / (1.0 + 0.5 * (kl1 + kl2))
                    
                    else:
                        raise ValueError(f"Unsupported agreement metric: {agreement_metric}")
                    
                    agreements.append(agreement)
                
                # Average agreement across batch
                head_agreement[(h1, h2)] = float(np.mean(agreements))
        
    elif len(attention_probs.shape) == 3:  # [n_heads, seq_len, seq_len]
        n_heads, seq_len, _ = attention_probs.shape
        
        for h1 in range(n_heads):
            for h2 in range(h1+1, n_heads):  # Only compute upper triangle
                # Get attention matrices for each head
                attn1 = attention_probs[h1].flatten()
                attn2 = attention_probs[h2].flatten()
                
                # Calculate agreement based on chosen metric
                if agreement_metric == "cosine":
                    # Cosine similarity
                    norm1 = np.linalg.norm(attn1)
                    norm2 = np.linalg.norm(attn2)
                    
                    if norm1 > 0 and norm2 > 0:
                        agreement = np.dot(attn1, attn2) / (norm1 * norm2)
                    else:
                        agreement = 0.0
                
                elif agreement_metric == "correlation":
                    # Pearson correlation
                    if np.std(attn1) > 0 and np.std(attn2) > 0:
                        agreement = np.corrcoef(attn1, attn2)[0, 1]
                    else:
                        agreement = 0.0
                
                elif agreement_metric == "kl":
                    # KL divergence (symmetric version)
                    # Ensure valid probability distributions
                    attn1 = np.clip(attn1, 1e-10, 1.0)
                    attn1 = attn1 / np.sum(attn1)
                    attn2 = np.clip(attn2, 1e-10, 1.0)
                    attn2 = attn2 / np.sum(attn2)
                    
                    # Compute symmetric KL
                    kl1 = entropy(attn1, attn2)
                    kl2 = entropy(attn2, attn1)
                    
                    # Convert divergence to similarity (agreement)
                    # Higher divergence = lower agreement
                    agreement = 1.0 / (1.0 + 0.5 * (kl1 + kl2))
                
                else:
                    raise ValueError(f"Unsupported agreement metric: {agreement_metric}")
                
                head_agreement[(h1, h2)] = float(agreement)
    else:
        raise ValueError(f"Unsupported attention_probs shape: {attention_probs.shape}")
    
    return head_agreement


def aggregate_metrics(
    metrics: Union[torch.Tensor, np.ndarray],
    method: str = "mean",
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> float:
    """
    Aggregate metrics across batch and/or heads.
    
    Args:
        metrics: Metrics tensor [batch_size, n_heads] or [n_heads]
        method: Aggregation method ('mean', 'max', 'min', 'median')
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        
    Returns:
        Aggregated metric value
    """
    # Convert to numpy for consistent processing
    if isinstance(metrics, torch.Tensor):
        metrics = metrics.detach().cpu().numpy()
    
    if token_mask is not None:
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.detach().cpu().numpy()
        
        # Use token mask to compute weighted average if metrics has batch dimension
        if len(metrics.shape) == 2 and token_mask.shape[0] == metrics.shape[0]:
            # Compute weights based on number of tokens in each sequence
            weights = token_mask.sum(axis=1)
            weights = weights / weights.sum()
            
            # Apply weights along batch dimension
            if method == "mean":
                # Mean across heads, then weighted mean across batch
                head_means = metrics.mean(axis=1)
                return float((head_means * weights).sum())
    
    # Apply aggregation method
    if method == "mean":
        return float(metrics.mean())
    elif method == "max":
        return float(metrics.max())
    elif method == "min":
        return float(metrics.min())
    elif method == "median":
        return float(np.median(metrics))
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")


def analyze_attention_patterns(
    attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
    outputs: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]] = None,
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    token_info: Optional[Dict[str, Any]] = None,
    aggregation_method: str = "mean",
    include_head_importance: bool = True,
    include_cross_head_agreement: bool = False,
    agreement_metric: str = "cosine",
    sparsity_threshold: float = 0.01
) -> AttentionMetricsResult:
    """
    Analyze attention patterns and compute various metrics.
    
    Args:
        attention_data: Dictionary mapping layer names to attention probabilities
                      Each tensor should have shape [batch_size, n_heads, seq_len, seq_len]
        outputs: Optional dictionary mapping layer names to outputs
                [batch_size, seq_len, hidden_size]
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        token_info: Optional information about input tokens
        aggregation_method: Method for aggregating metrics ('mean', 'max', 'min', 'median')
        include_head_importance: Whether to calculate head importance metrics
        include_cross_head_agreement: Whether to calculate cross-head agreement metrics
        agreement_metric: Metric to use for head agreement ('cosine', 'correlation', 'kl')
        sparsity_threshold: Threshold for calculating attention sparsity
        
    Returns:
        AttentionMetricsResult with computed metrics
    """
    # Initialize result containers
    layer_entropy = {}
    layer_sparsity = {}
    head_entropy = {}
    head_sparsity = {}
    head_importance = {}
    cross_head_agreement = {} if include_cross_head_agreement else None
    
    # Layers and heads analyzed
    layers_analyzed = []
    heads_analyzed = []
    
    # Process each layer
    for layer_name, attention_probs in attention_data.items():
        # Add to layers analyzed
        layers_analyzed.append(layer_name)
        
        # Ensure we have the right shape
        if len(attention_probs.shape) not in (3, 4):
            logger.warning(f"Skipping layer {layer_name} with unexpected shape: {attention_probs.shape}")
            continue
        
        # Get the number of heads
        if len(attention_probs.shape) == 4:
            n_heads = attention_probs.shape[1]
        else:
            n_heads = attention_probs.shape[0]
        
        # Calculate entropy and sparsity
        ent = calculate_attention_entropy(attention_probs)
        sparse = calculate_attention_sparsity(attention_probs, threshold=sparsity_threshold)
        
        # Aggregate metrics for the layer
        layer_entropy[layer_name] = aggregate_metrics(ent, method=aggregation_method, token_mask=token_mask)
        layer_sparsity[layer_name] = aggregate_metrics(sparse, method=aggregation_method, token_mask=token_mask)
        
        # Process individual heads
        for h in range(n_heads):
            head_key = (layer_name, h)
            heads_analyzed.append(head_key)
            
            # Add head-specific metrics
            if len(ent.shape) == 2:
                head_entropy[head_key] = aggregate_metrics(
                    ent[:, h], method=aggregation_method, token_mask=token_mask
                )
                head_sparsity[head_key] = aggregate_metrics(
                    sparse[:, h], method=aggregation_method, token_mask=token_mask
                )
            else:
                head_entropy[head_key] = float(ent[h])
                head_sparsity[head_key] = float(sparse[h])
        
        # Calculate head importance if requested
        if include_head_importance and outputs is not None and layer_name in outputs:
            try:
                imp = calculate_head_importance(
                    attention_probs,
                    outputs[layer_name],
                    token_mask
                )
                
                for h in range(n_heads):
                    head_key = (layer_name, h)
                    
                    if len(imp.shape) == 2:
                        head_importance[head_key] = aggregate_metrics(
                            imp[:, h], method=aggregation_method, token_mask=token_mask
                        )
                    else:
                        head_importance[head_key] = float(imp[h])
            except Exception as e:
                logger.warning(f"Error calculating head importance for layer {layer_name}: {e}")
        
        # Calculate cross-head agreement if requested
        if include_cross_head_agreement:
            try:
                # Calculate agreement between heads within this layer
                head_agreement = calculate_cross_head_agreement(
                    attention_probs,
                    agreement_metric=agreement_metric,
                    token_mask=token_mask
                )
                
                # Store results
                if head_agreement:
                    cross_head_agreement[layer_name] = head_agreement
            except Exception as e:
                logger.warning(f"Error calculating cross-head agreement for layer {layer_name}: {e}")
    
    # Prepare token info
    input_tokens = None
    if token_info and "token_strings" in token_info:
        input_tokens = token_info["token_strings"]
    
    # Aggregate metrics across all layers
    all_entropy = np.array(list(layer_entropy.values()))
    all_sparsity = np.array(list(layer_sparsity.values()))
    
    overall_entropy = float(all_entropy.mean())
    overall_sparsity = float(all_sparsity.mean())
    
    # Create the result
    result = AttentionMetricsResult(
        entropy=overall_entropy,
        sparsity=overall_sparsity,
        head_importance=head_importance,
        layer_entropy=layer_entropy,
        layer_sparsity=layer_sparsity,
        head_entropy=head_entropy,
        head_sparsity=head_sparsity,
        input_tokens=input_tokens,
        aggregation_method=aggregation_method,
        layers_analyzed=layers_analyzed,
        heads_analyzed=heads_analyzed,
        cross_head_agreement=cross_head_agreement
    )
    
    return result


def calculate_activation_statistics(
    activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    per_token: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for activations across layers.
    
    Args:
        activations: Dictionary mapping layer names to activation tensors
                   [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
        token_mask: Optional mask for padding tokens [batch_size, seq_len]
        per_token: Whether to compute statistics per token
        
    Returns:
        Dictionary mapping layer names to statistics dictionaries containing:
        - mean: Mean activation value
        - std: Standard deviation of activations
        - sparsity: Activation sparsity (fraction of near-zero values)
        - l2_norm: Average L2 norm of activation vectors
        - max_val: Maximum activation value
        - min_val: Minimum activation value
        - token_stats: Per-token statistics if per_token=True
    """
    stats = {}
    
    for layer_name, layer_activations in activations.items():
        # Convert to numpy for consistent processing
        if isinstance(layer_activations, torch.Tensor):
            layer_activations = layer_activations.detach().cpu().numpy()
        
        # Skip non-numeric or empty activations
        if not isinstance(layer_activations, np.ndarray) or layer_activations.size == 0:
            continue
        
        # Apply token mask if provided and shapes match
        masked_activations = layer_activations
        if token_mask is not None:
            if isinstance(token_mask, torch.Tensor):
                token_mask = token_mask.detach().cpu().numpy()
            
            # Check if shapes are compatible for masking
            if len(layer_activations.shape) == 3 and token_mask.shape[0] == layer_activations.shape[0]:
                # Expand mask to match activations
                expanded_mask = np.expand_dims(token_mask, -1)
                masked_activations = layer_activations * expanded_mask
        
        # Initialize statistics dictionary
        layer_stats = {}
        
        # Calculate overall statistics
        layer_stats["mean"] = float(np.mean(masked_activations))
        layer_stats["std"] = float(np.std(masked_activations))
        
        # Calculate sparsity (fraction of near-zero values)
        near_zero = np.abs(masked_activations) < 1e-6
        layer_stats["sparsity"] = float(np.mean(near_zero))
        
        # Calculate L2 norm
        if len(masked_activations.shape) == 3:  # [batch_size, seq_len, hidden_size]
            # Average L2 norm across batch and sequence
            norms = np.sqrt(np.sum(masked_activations**2, axis=-1))
            
            if token_mask is not None:
                # Only consider non-padding tokens
                masked_norms = norms * token_mask
                non_zero = token_mask.sum()
                if non_zero > 0:
                    layer_stats["l2_norm"] = float(masked_norms.sum() / non_zero)
                else:
                    layer_stats["l2_norm"] = 0.0
            else:
                layer_stats["l2_norm"] = float(np.mean(norms))
        else:
            # Average L2 norm across batch
            norms = np.sqrt(np.sum(masked_activations**2, axis=-1))
            layer_stats["l2_norm"] = float(np.mean(norms))
        
        # Min and max values
        layer_stats["max_val"] = float(np.max(masked_activations))
        layer_stats["min_val"] = float(np.min(masked_activations))
        
        # Calculate per-token statistics if requested
        if per_token and len(masked_activations.shape) == 3:
            token_stats = []
            
            batch_size, seq_len, hidden_size = masked_activations.shape
            
            for i in range(min(seq_len, 50)):  # Limit to first 50 tokens
                token_acts = masked_activations[:, i, :]
                
                if token_mask is not None:
                    # Check which batch elements have this token
                    valid_mask = token_mask[:, i] > 0
                    
                    if np.any(valid_mask):
                        # Only consider valid batch elements
                        token_acts = token_acts[valid_mask]
                    else:
                        # Skip tokens that are masked in all batch elements
                        continue
                
                # Calculate statistics for this token
                token_stat = {
                    "token_idx": i,
                    "mean": float(np.mean(token_acts)),
                    "std": float(np.std(token_acts)),
                    "l2_norm": float(np.mean(np.sqrt(np.sum(token_acts**2, axis=-1)))),
                    "sparsity": float(np.mean(np.abs(token_acts) < 1e-6))
                }
                
                token_stats.append(token_stat)
            
            layer_stats["token_stats"] = token_stats
        
        # Add to overall statistics
        stats[layer_name] = layer_stats
    
    return stats


def analyze_cross_attention_consistency(
    attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
    input_tokens: Optional[List[str]] = None,
    focus_tokens: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze the consistency of attention patterns across layers for specific tokens.
    
    This metric shows whether different layers consistently attend to the same tokens,
    which can reveal the model's focus on certain input elements.
    
    Args:
        attention_data: Dictionary mapping layer names to attention probabilities
                      Each tensor should have shape [batch_size, n_heads, seq_len, seq_len]
        input_tokens: List of input tokens (strings)
        focus_tokens: Optional list of tokens to focus the analysis on
        
    Returns:
        Dictionary with metrics about cross-layer attention consistency
    """
    # Prepare result dictionary
    result = {
        "layer_consistency": {},
        "head_consistency": {},
        "token_attention_profiles": {},
        "focused_tokens": []
    }
    
    # Skip if we have fewer than 2 layers
    if len(attention_data) < 2:
        return result
    
    # Sort layers by name
    sorted_layers = sorted(attention_data.keys())
    
    # Ensure all layers have compatible shapes
    shapes = [attention_data[layer].shape for layer in sorted_layers]
    if not all(len(s) in (3, 4) for s in shapes):
        logger.warning("Layers have incompatible shapes for cross-attention analysis")
        return result
    
    # Extract attention patterns for consistent processing
    processed_attention = {}
    batch_size = None
    seq_len = None
    
    for layer in sorted_layers:
        attn = attention_data[layer]
        
        # Convert to numpy if needed
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        
        if len(attn.shape) == 4:  # [batch_size, n_heads, seq_len, seq_len]
            batch_size = attn.shape[0]
            n_heads = attn.shape[1]
            seq_len = attn.shape[2]
            
            # Average across heads for layer-level analysis
            processed_attention[layer] = attn.mean(axis=1)  # [batch_size, seq_len, seq_len]
        else:  # [n_heads, seq_len, seq_len]
            n_heads = attn.shape[0]
            seq_len = attn.shape[1]
            
            # Average across heads for layer-level analysis
            processed_attention[layer] = attn.mean(axis=0)  # [seq_len, seq_len]
    
    # Identify focus tokens
    focus_indices = []
    
    # If focus_tokens are provided and we have input_tokens
    if focus_tokens and input_tokens and len(input_tokens) == seq_len:
        for token in focus_tokens:
            try:
                # Find indices of tokens that match focus tokens
                matches = [i for i, t in enumerate(input_tokens) if token in t]
                focus_indices.extend(matches)
            except:
                pass
    
    # If no focus tokens found, use tokens with highest attention
    if not focus_indices:
        # Calculate aggregate attention across all layers
        aggregate_attention = np.zeros((seq_len,))
        
        for layer, attn in processed_attention.items():
            if len(attn.shape) == 3:  # [batch_size, seq_len, seq_len]
                # Sum attention to each token (how much each token is attended to)
                token_attention = attn.sum(axis=(0, 1))
            else:  # [seq_len, seq_len]
                token_attention = attn.sum(axis=0)
            
            aggregate_attention += token_attention
        
        # Find top 5 attended tokens
        top_indices = np.argsort(aggregate_attention)[-5:]
        focus_indices = top_indices.tolist()
    
    # Record focused tokens
    result["focused_tokens"] = focus_indices
    if input_tokens:
        result["focused_token_strings"] = [input_tokens[i] for i in focus_indices if i < len(input_tokens)]
    
    # Calculate layer consistency
    for i, layer1 in enumerate(sorted_layers):
        for j, layer2 in enumerate(sorted_layers):
            if i >= j:
                continue  # Skip self-comparisons and avoid duplicates
            
            attn1 = processed_attention[layer1]
            attn2 = processed_attention[layer2]
            
            # Calculate Pearson correlation for attention patterns
            if len(attn1.shape) == 3 and len(attn2.shape) == 3:
                # Calculate correlation per batch element and average
                correlations = []
                
                for b in range(batch_size):
                    # Flatten attention matrices
                    flat1 = attn1[b].flatten()
                    flat2 = attn2[b].flatten()
                    
                    # Calculate correlation
                    correlation = np.corrcoef(flat1, flat2)[0, 1]
                    correlations.append(correlation)
                
                layer_correlation = np.mean(correlations)
            else:
                # Flatten attention matrices
                flat1 = attn1.flatten()
                flat2 = attn2.flatten()
                
                # Calculate correlation
                layer_correlation = np.corrcoef(flat1, flat2)[0, 1]
            
            result["layer_consistency"][(layer1, layer2)] = float(layer_correlation)
    
    # Generate attention profiles for focus tokens
    token_profiles = {}
    
    for idx in focus_indices:
        # Skip if index is out of bounds
        if idx >= seq_len:
            continue
            
        profile = {}
        
        # For each layer, calculate attention to and from this token
        for layer in sorted_layers:
            attn = processed_attention[layer]
            
            if len(attn.shape) == 3:  # [batch_size, seq_len, seq_len]
                # Average across batch
                avg_attn = attn.mean(axis=0)  # [seq_len, seq_len]
            else:
                avg_attn = attn
            
            # Attention from this token to others
            attn_from = avg_attn[idx, :]
            
            # Attention to this token from others
            attn_to = avg_attn[:, idx]
            
            profile[layer] = {
                "attention_from": attn_from.tolist(),
                "attention_to": attn_to.tolist(),
                "max_attended_idx": int(np.argmax(attn_from)),
                "max_attending_idx": int(np.argmax(attn_to))
            }
        
        token_name = f"token_{idx}"
        if input_tokens and idx < len(input_tokens):
            token_name = input_tokens[idx]
            
        token_profiles[token_name] = profile
    
    result["token_attention_profiles"] = token_profiles
    
    # Calculate overall consistency
    consistency_values = list(result["layer_consistency"].values())
    if consistency_values:
        result["overall_consistency"] = float(np.mean(consistency_values))
        result["min_consistency"] = float(np.min(consistency_values))
        result["max_consistency"] = float(np.max(consistency_values))
    else:
        result["overall_consistency"] = 0.0
        result["min_consistency"] = 0.0
        result["max_consistency"] = 0.0
    
    return result


def calculate_activation_sensitivity(
    base_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    perturbed_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    metric: str = "cosine"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate sensitivity of activations to input perturbations.
    
    This metric quantifies how much the internal activations change when
    inputs are slightly perturbed, revealing the stability of features.
    
    Args:
        base_activations: Dictionary mapping layer names to baseline activations
        perturbed_activations: Dictionary mapping layer names to perturbed activations
        metric: Metric to compare activations ('cosine', 'euclidean', 'l2')
        
    Returns:
        Dictionary mapping layer names to sensitivity metrics
    """
    sensitivity = {}
    
    for layer_name, base_acts in base_activations.items():
        # Skip if we don't have this layer in perturbed activations
        if layer_name not in perturbed_activations:
            continue
        
        perturbed_acts = perturbed_activations[layer_name]
        
        # Convert to numpy if needed
        if isinstance(base_acts, torch.Tensor):
            base_acts = base_acts.detach().cpu().numpy()
        
        if isinstance(perturbed_acts, torch.Tensor):
            perturbed_acts = perturbed_acts.detach().cpu().numpy()
        
        # Ensure shapes match
        if base_acts.shape != perturbed_acts.shape:
            logger.warning(f"Shapes don't match for layer {layer_name}: {base_acts.shape} vs {perturbed_acts.shape}")
            continue
        
        # Calculate sensitivity based on the chosen metric
        layer_sensitivity = {}
        
        if metric == "cosine":
            # Calculate cosine similarity for each example
            if len(base_acts.shape) == 3:  # [batch_size, seq_len, hidden_size]
                batch_size, seq_len, _ = base_acts.shape
                
                similarities = np.zeros((batch_size, seq_len))
                
                for b in range(batch_size):
                    for s in range(seq_len):
                        vec1 = base_acts[b, s]
                        vec2 = perturbed_acts[b, s]
                        
                        # Calculate cosine similarity
                        similarities[b, s] = 1 - cosine(vec1, vec2)
                
                # Convert similarity to sensitivity (1 - similarity)
                # Average across sequence and batch
                sensitivity_value = float(1 - np.mean(similarities))
            else:  # [batch_size, hidden_size]
                batch_size = base_acts.shape[0]
                
                similarities = np.zeros(batch_size)
                
                for b in range(batch_size):
                    vec1 = base_acts[b]
                    vec2 = perturbed_acts[b]
                    
                    # Calculate cosine similarity
                    similarities[b] = 1 - cosine(vec1, vec2)
                
                # Convert similarity to sensitivity (1 - similarity)
                # Average across batch
                sensitivity_value = float(1 - np.mean(similarities))
            
            layer_sensitivity["cosine"] = sensitivity_value
            
        elif metric == "euclidean" or metric == "l2":
            # Calculate Euclidean distance for each example
            if len(base_acts.shape) == 3:  # [batch_size, seq_len, hidden_size]
                # Calculate L2 distance between activations
                distances = np.sqrt(np.sum((base_acts - perturbed_acts)**2, axis=-1))
                
                # Average across sequence and batch
                sensitivity_value = float(np.mean(distances))
            else:  # [batch_size, hidden_size]
                # Calculate L2 distance between activations
                distances = np.sqrt(np.sum((base_acts - perturbed_acts)**2, axis=-1))
                
                # Average across batch
                sensitivity_value = float(np.mean(distances))
            
            layer_sensitivity["euclidean"] = sensitivity_value
            
            # Normalize by baseline L2 norm to get relative sensitivity
            if len(base_acts.shape) == 3:
                base_norms = np.sqrt(np.sum(base_acts**2, axis=-1))
                relative_sensitivity = float(np.mean(distances / (base_norms + 1e-10)))
            else:
                base_norms = np.sqrt(np.sum(base_acts**2, axis=-1))
                relative_sensitivity = float(np.mean(distances / (base_norms + 1e-10)))
            
            layer_sensitivity["relative"] = relative_sensitivity
        
        # Add to overall result
        sensitivity[layer_name] = layer_sensitivity
    
    return sensitivity


class TransformerMetricsCalculator:
    """
    Calculator for transformer-specific metrics.
    
    This class provides a unified interface for computing various
    metrics for transformer models, handling caching and aggregation.
    
    Attributes:
        use_cache: Whether to cache results
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the metrics calculator.
        
        Args:
            use_cache: Whether to cache results
        """
        self.use_cache = use_cache
        self._cache = {}
    
    def compute_attention_metrics(
        self,
        attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
        outputs: Optional[Dict[str, Union[torch.Tensor, np.ndarray]]] = None,
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        token_info: Optional[Dict[str, Any]] = None,
        aggregation_method: str = "mean",
        include_head_importance: bool = True,
        include_cross_head_agreement: bool = False,
        agreement_metric: str = "cosine",
        sparsity_threshold: float = 0.01,
        force_recompute: bool = False
    ) -> AttentionMetricsResult:
        """
        Compute comprehensive attention metrics.
        
        Args:
            attention_data: Dictionary mapping layer names to attention probs
            outputs: Optional dictionary mapping layer names to outputs
            token_mask: Optional mask for padding tokens
            token_info: Optional information about input tokens
            aggregation_method: Method for aggregating metrics
            include_head_importance: Whether to calculate head importance
            include_cross_head_agreement: Whether to calculate cross-head agreement
            agreement_metric: Metric for comparing heads ('cosine', 'correlation', 'kl')
            sparsity_threshold: Threshold for calculating sparsity
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            AttentionMetricsResult with computed metrics
        """
        # Generate cache key
        cache_key = f"attention_metrics_{aggregation_method}_{sparsity_threshold}"
        if include_cross_head_agreement:
            cache_key += f"_agreement_{agreement_metric}"
        
        # Check cache
        if self.use_cache and cache_key in self._cache and not force_recompute:
            return self._cache[cache_key]
        
        # Compute metrics
        result = analyze_attention_patterns(
            attention_data=attention_data,
            outputs=outputs,
            token_mask=token_mask,
            token_info=token_info,
            aggregation_method=aggregation_method,
            include_head_importance=include_head_importance,
            include_cross_head_agreement=include_cross_head_agreement,
            agreement_metric=agreement_metric,
            sparsity_threshold=sparsity_threshold
        )
        
        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def compute_activation_statistics(
        self,
        activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        per_token: bool = False,
        force_recompute: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for activations.
        
        Args:
            activations: Dictionary mapping layer names to activations
            token_mask: Optional mask for padding tokens
            per_token: Whether to compute statistics per token
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            Dictionary mapping layer names to statistics
        """
        # Generate cache key
        cache_key = f"activation_stats_{per_token}"
        
        # Check cache
        if self.use_cache and cache_key in self._cache and not force_recompute:
            return self._cache[cache_key]
        
        # Compute statistics
        result = calculate_activation_statistics(
            activations=activations,
            token_mask=token_mask,
            per_token=per_token
        )
        
        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def compute_cross_attention_consistency(
        self,
        attention_data: Dict[str, Union[torch.Tensor, np.ndarray]],
        input_tokens: Optional[List[str]] = None,
        focus_tokens: Optional[List[str]] = None,
        force_recompute: bool = False
    ) -> Dict[str, Any]:
        """
        Compute cross-layer attention consistency.
        
        Args:
            attention_data: Dictionary mapping layer names to attention probs
            input_tokens: List of input tokens
            focus_tokens: Optional list of tokens to focus on
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            Dictionary with metrics about cross-layer attention consistency
        """
        # Generate cache key
        focus_key = "_".join(focus_tokens) if focus_tokens else "none"
        cache_key = f"cross_attention_{focus_key}"
        
        # Check cache
        if self.use_cache and cache_key in self._cache and not force_recompute:
            return self._cache[cache_key]
        
        # Compute consistency
        result = analyze_cross_attention_consistency(
            attention_data=attention_data,
            input_tokens=input_tokens,
            focus_tokens=focus_tokens
        )
        
        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def compute_activation_sensitivity(
        self,
        base_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        perturbed_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
        metric: str = "cosine",
        force_recompute: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute sensitivity of activations to input perturbations.
        
        Args:
            base_activations: Dictionary mapping layer names to baseline activations
            perturbed_activations: Dictionary mapping layer names to perturbed activations
            metric: Metric to compare activations
            force_recompute: Whether to force recomputation ignoring cache
            
        Returns:
            Dictionary mapping layer names to sensitivity metrics
        """
        # Generate cache key
        cache_key = f"sensitivity_{metric}"
        
        # Check cache
        if self.use_cache and cache_key in self._cache and not force_recompute:
            return self._cache[cache_key]
        
        # Compute sensitivity
        result = calculate_activation_sensitivity(
            base_activations=base_activations,
            perturbed_activations=perturbed_activations,
            metric=metric
        )
        
        # Cache result
        if self.use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """
        Clear the cache.
        """
        self._cache = {}