"""
Intra-Class Pairwise Distance (ICPD) Metric for Concept Fragmentation analysis.

This module provides functions to compute the intra-class pairwise distance,
which measures the average distance between pairs of points in the same class.
Higher ICPD indicates more dispersion within a class (more fragmentation).
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, List
from tqdm import tqdm

from ..config import METRICS, RANDOM_SEED


def compute_intra_class_distance(
    activations: Union[torch.Tensor, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    layer_name: Optional[str] = None,
    normalize_dim: bool = True,
    use_cosine: bool = False,
    random_state: int = RANDOM_SEED,
    sample_size: Optional[int] = 1000
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute the intra-class pairwise distance metric for concept fragmentation.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
                    or dictionary mapping layer names to activations
        labels: Tensor of shape (n_samples,) containing class labels
        layer_name: Layer name to use when activations is a dictionary
        normalize_dim: Whether to normalize distances by feature dimension
        use_cosine: Whether to use cosine similarity instead of Euclidean distance
        random_state: Random seed for reproducibility
        sample_size: Subsample size for pairwise distance computation when n_samples is large
        
    Returns:
        Dictionary containing:
            - 'mean_distance': Mean intra-class distance across all classes
            - 'class_distances': Dictionary mapping class labels to distance values
    """
    # Handle dictionary input
    if isinstance(activations, dict):
        if layer_name is None:
            raise ValueError("layer_name must be provided when activations is a dictionary")
        activations = activations[layer_name]
    
    # Convert tensors to numpy arrays
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Get unique class labels
    unique_labels = np.unique(labels)
    
    # Compute distances for each class
    class_distances = {}
    
    for label in unique_labels:
        # Get activations for this class
        class_indices = np.where(labels == label)[0]
        
        # Skip if only one sample or none
        if len(class_indices) <= 1:
            class_distances[int(label)] = 0.0
            continue
        
        class_acts = activations[class_indices]
        n_samples = len(class_acts)
        
        # Subsample if needed for efficiency
        if sample_size is not None and n_samples > sample_size:
            np.random.seed(random_state)
            indices = np.random.choice(n_samples, min(sample_size, n_samples), replace=False)
            class_acts = class_acts[indices]
            n_samples = len(class_acts)
        
        # Compute all pairwise distances efficiently
        if use_cosine:
            # For cosine, normalize vectors and compute 1 - dot product
            norms = np.linalg.norm(class_acts, axis=1, keepdims=True)
            normalized = class_acts / (norms + 1e-8)  # Avoid division by zero
            similarity_matrix = np.dot(normalized, normalized.T)
            np.fill_diagonal(similarity_matrix, 1.0)  # Ensure diagonal is exactly 1
            
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix
            
            # Average distance
            total_distance = np.sum(distance_matrix) / (n_samples * (n_samples - 1))
            
            # No need to normalize by dimension for cosine
            distance = total_distance
        else:
            # For Euclidean distance
            # Use a more memory-efficient approach for large matrices
            if n_samples <= 5000:
                # Compute full distance matrix
                a_squared = np.sum(class_acts**2, axis=1, keepdims=True)  # Shape (n, 1)
                b_squared = np.sum(class_acts**2, axis=1)  # Shape (n,)
                b_squared = b_squared.reshape(1, -1)  # Shape (1, n) for proper broadcasting
                dot_product = np.dot(class_acts, class_acts.T)
                
                # Fix: Ensure non-negative values inside the square root to avoid NaN
                distance_matrix = np.sqrt(np.maximum(a_squared + b_squared - 2 * dot_product, 0) + 1e-8)
                
                # Compute mean (excluding diagonal)
                np.fill_diagonal(distance_matrix, 0)
                total_distance = np.sum(distance_matrix) / (n_samples * (n_samples - 1))
            else:
                # Compute pairwise distances in batches
                total_distance = 0
                count = 0
                
                for i in range(n_samples):
                    # Compute distance from point i to all points j > i
                    diff = class_acts[i:i+1] - class_acts[i+1:]
                    # Fix: Use safe square root operation for batch distances too
                    batch_distances = np.sqrt(np.maximum(np.sum(diff**2, axis=1), 0) + 1e-8)
                    total_distance += np.sum(batch_distances)
                    count += len(batch_distances)
                
                # Complete the symmetric matrix (upper + lower triangular)
                total_distance *= 2
                count *= 2
                
                if count > 0:
                    total_distance = total_distance / count
            
            # Normalize by feature dimension if requested
            distance = total_distance
            if normalize_dim:
                distance = distance / np.sqrt(class_acts.shape[1])
        
        # Store result
        class_distances[int(label)] = float(distance)
    
    # Compute mean distance across all classes
    # Filter out any NaN values that might have occurred despite our precautions
    valid_distances = [v for v in class_distances.values() if not (np.isnan(v) or np.isinf(v))]
    mean_distance = float(np.mean(valid_distances)) if valid_distances else 0.0
    
    # Create result dictionary
    result = {
        'mean_distance': mean_distance,
        'class_distances': class_distances,
        # For consistency with other metrics
        'mean': mean_distance,
        'per_class': class_distances,
        'max': float(max(valid_distances)) if valid_distances else 0.0
    }
    
    return result


def compute_fragmentation_score(
    activations: torch.Tensor,
    labels: torch.Tensor,
    normalize_dim: bool = True,
    use_cosine: bool = False,
    random_state: int = RANDOM_SEED
) -> float:
    """
    Compute a single fragmentation score based on intra-class pairwise distance.
    Higher values indicate more fragmentation.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
        labels: Tensor of shape (n_samples,) containing class labels
        normalize_dim: Whether to normalize distances by feature dimension
        use_cosine: Whether to use cosine similarity instead of Euclidean distance
        random_state: Random seed for reproducibility
        
    Returns:
        Fragmentation score (mean intra-class distance)
    """
    result = compute_intra_class_distance(
        activations, labels, normalize_dim=normalize_dim, 
        use_cosine=use_cosine, random_state=random_state
    )
    
    return result['mean_distance'] 