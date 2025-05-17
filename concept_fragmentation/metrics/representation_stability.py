"""
Representation Stability (Δ-Norm) Metric for Concept Fragmentation analysis.

This module provides functions to compute the representation stability across layers,
which measures how much representations change between consecutive layers.
Higher Δ-Norm indicates more dramatic changes in representation (potential reorganization).
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, List, Tuple, OrderedDict
from collections import OrderedDict as OrderedDictClass
import warnings

from ..config import METRICS, RANDOM_SEED


def compute_representation_stability(
    activations_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]], 
    layer_order: Optional[List[str]] = None,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute the representation stability (Δ-Norm) between consecutive layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        layer_order: Optional list specifying the order of layers
                    (if None, sorted keys of activations_dict will be used)
        normalize: Whether to normalize the difference by the previous layer's norm
        
    Returns:
        Dictionary mapping layer transitions to stability scores (delta norms)
    """
    # Handle empty input
    if not activations_dict:
        return {}
    
    # Determine layer order
    if layer_order is None:
        if isinstance(activations_dict, OrderedDictClass):
            layer_order = list(activations_dict.keys())
        else:
            layer_order = sorted(activations_dict.keys())
    
    # Need at least two layers to compute stability
    if len(layer_order) < 2:
        warnings.warn("Need at least two layers to compute stability.")
        return {}
    
    # Convert tensors to numpy arrays
    np_activations = {}
    for layer in layer_order:
        if layer not in activations_dict:
            warnings.warn(f"Layer {layer} specified in layer_order not found in activations_dict.")
            continue
            
        if isinstance(activations_dict[layer], torch.Tensor):
            np_activations[layer] = activations_dict[layer].detach().cpu().numpy()
        else:
            np_activations[layer] = activations_dict[layer]
    
    # Compute stability for each consecutive layer pair
    stability_scores = {}
    
    for i in range(1, len(layer_order)):
        prev_layer = layer_order[i-1]
        curr_layer = layer_order[i]
        
        if prev_layer not in np_activations or curr_layer not in np_activations:
            continue
        
        prev_acts = np_activations[prev_layer]
        curr_acts = np_activations[curr_layer]
        
        # Ensure same number of samples
        if prev_acts.shape[0] != curr_acts.shape[0]:
            warnings.warn(
                f"Different number of samples in layers {prev_layer} and {curr_layer}. "
                f"Skipping stability computation for this pair."
            )
            continue

        # ------------------------------------------------------------------
        # Align feature dimensions if they differ (e.g. 64 → 32)
        # Strategy: find a least-squares linear map W such that
        #           prev_acts @ W  ≈  curr_acts.
        #           This works for any (d_prev, d_curr) combination.
        # ------------------------------------------------------------------
        if prev_acts.shape[1] != curr_acts.shape[1]:
            try:
                # Solve prev_acts * W = curr_acts  in the LS sense
                W, *_ = np.linalg.lstsq(prev_acts, curr_acts, rcond=None)
                prev_aligned = prev_acts @ W          # (n_samples, d_curr)
            except Exception as e:
                warnings.warn(f"Alignment failed for {prev_layer}->{curr_layer}: {e}")
                continue
                
            curr_aligned = curr_acts
        else:
            prev_aligned = prev_acts
            curr_aligned = curr_acts

        # Compute the Frobenius norm of the difference on aligned matrices
        diff_norm = np.linalg.norm(curr_aligned - prev_aligned, ord='fro')
        
        # Normalize by previous layer's norm if requested
        if normalize:
            prev_norm = np.linalg.norm(prev_aligned, ord='fro')
            if prev_norm > 1e-8:  # Avoid division by zero
                diff_norm = diff_norm / prev_norm
        
        # Store the result
        transition_name = f"{prev_layer}_to_{curr_layer}"
        stability_scores[transition_name] = float(diff_norm)
    
    return stability_scores


def compute_layer_stability_profile(
    activations_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]],
    layer_order: Optional[List[str]] = None,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute a stability profile for all layers, attributing each score to the target layer.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        layer_order: Optional list specifying the order of layers
        normalize: Whether to normalize the difference by the previous layer's norm
        
    Returns:
        Dictionary mapping layers to stability scores (attributed to target layer)
    """
    # Compute stability between consecutive layers
    transition_scores = compute_representation_stability(
        activations_dict, layer_order, normalize
    )
    
    # Map transition scores to target layers
    layer_scores = {}
    
    for transition, score in transition_scores.items():
        # Extract target layer from transition name
        parts = transition.split('_to_')
        if len(parts) == 2:
            target_layer = parts[1]
            layer_scores[target_layer] = score
    
    return layer_scores


def compute_average_stability(
    activations_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]],
    layer_order: Optional[List[str]] = None,
    normalize: bool = True
) -> float:
    """
    Compute the average stability score across all layer transitions.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        layer_order: Optional list specifying the order of layers
        normalize: Whether to normalize the difference by the previous layer's norm
        
    Returns:
        Average stability score across all transitions
    """
    # Compute stability between consecutive layers
    transition_scores = compute_representation_stability(
        activations_dict, layer_order, normalize
    )
    
    # Return average score
    if not transition_scores:
        return 0.0
        
    return float(np.mean(list(transition_scores.values())))


def compute_fragmentation_score(
    activations_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]],
    layer_order: Optional[List[str]] = None
) -> float:
    """
    Compute a single fragmentation score based on representation stability.
    Higher values indicate more fragmentation (more representation change across layers).
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        layer_order: Optional list specifying the order of layers
        
    Returns:
        Fragmentation score (average normalized delta norm)
    """
    return compute_average_stability(activations_dict, layer_order, normalize=True) 