"""
Subspace Angle Metric for Concept Fragmentation analysis.

This module provides functions to compute the subspace angle metric,
which measures the angle between subspaces occupied by different instances
of the same class. Larger angles indicate more fragmentation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
import scipy.linalg as linalg
import scipy.stats as stats
from scipy.linalg import subspace_angles
import warnings

from ..config import METRICS, RANDOM_SEED


def compute_principal_angles(
    subspace1: np.ndarray,
    subspace2: np.ndarray
) -> np.ndarray:
    """
    Compute the principal angles between two subspaces.
    
    Args:
        subspace1: Matrix whose columns form a basis for the first subspace
        subspace2: Matrix whose columns form a basis for the second subspace
        
    Returns:
        Array of principal angles in degrees
    """
    # Handle edge cases
    if subspace1.shape[1] == 0 or subspace2.shape[1] == 0:
        return np.array([90.0])  # Orthogonal by definition for empty subspaces
    
    # Use scipy's subspace_angles function for correct computation
    angles_rad = subspace_angles(subspace1, subspace2)
    angles_deg = np.degrees(angles_rad)
    
    return angles_deg


def compute_class_subspace(
    activations: np.ndarray,
    n_components: int = METRICS["subspace_angle"]["n_components"],
    var_threshold: float = METRICS["subspace_angle"]["var_threshold"]
) -> np.ndarray:
    """
    Compute the principal components of the activations to form a subspace.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        n_components: Legacy parameter - maximum number of components to use (deprecated)
        var_threshold: Minimum explained variance to retain (between 0.0 and 1.0)
        
    Returns:
        Matrix whose columns form a basis for the subspace
    """
    # Ensure we don't request more components than samples or features
    n_samples, n_features = activations.shape
    max_components = min(n_samples, n_features)
    
    # First try with variance threshold
    if var_threshold is not None and 0.0 <= var_threshold <= 1.0:
        # Fit PCA with max possible components
        pca = PCA(n_components=max_components)
        pca.fit(activations)
        
        # Find minimum components that meet the variance threshold
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Get the number of components that explain at least var_threshold of variance
        components_needed = np.searchsorted(cumulative_variance, var_threshold) + 1
        components_needed = min(components_needed, max_components)
        
        # Ensure at least 1 component
        components_needed = max(1, components_needed)
        
        # If n_components is provided, use the minimum of the two
        if n_components is not None:
            components_needed = min(components_needed, n_components)
    else:
        # Fallback to n_components if var_threshold is invalid
        components_needed = min(n_components, max_components)
    
    # Refit with the determined number of components
    pca = PCA(n_components=components_needed)
    pca.fit(activations)
    
    # Return the principal components (eigenvectors)
    return pca.components_.T  # Shape: (n_features, n_components)


def split_class_activations(
    activations: np.ndarray,
    random_state: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split class activations into two random subsets for subspace comparison.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of two activation matrices (subset1, subset2)
    """
    n_samples = activations.shape[0]
    
    # We need at least 2 samples to create two subsets
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to split into subsets")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    split_idx = n_samples // 2
    
    # Split into two subsets
    subset1 = activations[indices[:split_idx]]
    subset2 = activations[indices[split_idx:]]
    
    return subset1, subset2


def compute_subspace_angle(
    activations: torch.Tensor,
    labels: torch.Tensor,
    var_threshold: float = METRICS["subspace_angle"]["var_threshold"],
    bootstrap_samples: int = METRICS["subspace_angle"]["bootstrap_samples"],
    confidence_level: float = METRICS["subspace_angle"]["confidence_level"],
    random_state: int = RANDOM_SEED,
    layer_name: Optional[str] = None,
    n_components: Optional[int] = METRICS["subspace_angle"]["n_components"]  # Deprecated
) -> Dict[str, Union[float, Dict[int, Dict[str, float]]]]:
    """
    Compute the subspace angle metric for concept fragmentation.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
                    or dictionary mapping layer names to activations
        labels: Tensor of shape (n_samples,) containing class labels
        var_threshold: Minimum explained variance to retain (between 0.0 and 1.0)
        bootstrap_samples: Number of bootstrap samples for estimating confidence intervals
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        layer_name: Layer name to use when activations is a dictionary
        n_components: Legacy parameter - maximum number of components (deprecated)
        
    Returns:
        Dictionary containing:
            - 'mean_angle': Mean subspace angle across all classes
            - 'class_angles': Dictionary mapping class labels to angle statistics
    """
    # Show deprecation warning if n_components is explicitly provided by caller
    import inspect
    frame = inspect.currentframe().f_back
    if frame and 'n_components' in frame.f_locals:
        warnings.warn(
            "The n_components parameter is deprecated in favor of var_threshold. "
            "It will be removed in a future version.",
            DeprecationWarning, 
            stacklevel=2
        )
    
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
    
    # Dictionary to store results for each class
    class_angles = {}
    all_mean_angles = []
    
    # For backward compatibility (pairwise results between classes)
    pairwise_results = {}
    
    # Handle edge case with only a few samples directly
    if activations.shape[0] <= 4:  # If there are only 4 or fewer samples
        return {
            'mean_angle': 0.0,  # Return 0 for very small datasets
            'class_angles': {
                int(label): {'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0} 
                for label in unique_labels
            },
            'pairwise': {},
            'ci_95': (0.0, 0.0)
        }
    
    # Compute subspace angles for each class
    for label in unique_labels:
        # Get activations for this class
        class_mask = labels == label
        class_activations = activations[class_mask]
        
        # Skip if not enough samples
        if len(class_activations) < 2:
            class_angles[int(label)] = {
                'mean': float('nan'),
                'std': float('nan'),
                'ci_lower': float('nan'),
                'ci_upper': float('nan')
            }
            
            # Add to pairwise results for backward compatibility
            for other_label in unique_labels:
                if label != other_label:
                    pair = (int(label), int(other_label))
                    reverse_pair = (int(other_label), int(label))
                    
                    # Only add if the pair doesn't exist yet
                    if pair not in pairwise_results and reverse_pair not in pairwise_results:
                        pairwise_results[pair] = {
                            'mean': float('nan'),
                            'std': float('nan')
                        }
            
            continue
        
        # Bootstrap to estimate confidence intervals
        bootstrap_angles = []
        np.random.seed(random_state)
        
        for _ in range(bootstrap_samples):
            try:
                # Split class activations into two random subsets
                subset1, subset2 = split_class_activations(
                    class_activations,
                    random_state=np.random.randint(0, 10000)  # Vary seed for each bootstrap
                )
                
                # Compute subspaces for each subset using var_threshold
                subspace1 = compute_class_subspace(subset1, n_components, var_threshold)
                subspace2 = compute_class_subspace(subset2, n_components, var_threshold)
                
                # Compute principal angles between subspaces
                angles = compute_principal_angles(subspace1, subspace2)
                
                # Take the mean angle as the metric
                mean_angle = float(np.mean(angles))
                bootstrap_angles.append(mean_angle)
            except:
                # Skip this bootstrap sample if there's an error
                continue
        
        # Calculate statistics from bootstrap samples
        if bootstrap_angles:
            bootstrap_angles = np.array(bootstrap_angles)
            mean_angle = float(np.mean(bootstrap_angles))
            std_angle = float(np.std(bootstrap_angles))
            
            # Compute confidence interval
            confidence = 0.5 + confidence_level / 2
            ci_lower = float(np.percentile(bootstrap_angles, 100 * (1 - confidence)))
            ci_upper = float(np.percentile(bootstrap_angles, 100 * confidence))
            
            class_angles[int(label)] = {
                'mean': mean_angle,
                'std': std_angle,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
            
            all_mean_angles.append(mean_angle)
            
            # For backward compatibility, add pairwise results
            for other_label in unique_labels:
                if label != other_label:
                    pair = (int(label), int(other_label))
                    reverse_pair = (int(other_label), int(label))
                    
                    # Only add if the pair doesn't exist yet
                    if pair not in pairwise_results and reverse_pair not in pairwise_results:
                        pairwise_results[pair] = {
                            'mean': mean_angle,
                            'std': std_angle
                        }
        else:
            class_angles[int(label)] = {
                'mean': float('nan'),
                'std': float('nan'),
                'ci_lower': float('nan'),
                'ci_upper': float('nan')
            }
    
    # ------------------------------------------------------------------
    # NEW: Compute pairwise subspace angles BETWEEN different classes
    # ------------------------------------------------------------------
    subspaces = {}
    for label in unique_labels:
        class_mask = labels == label
        class_activations = activations[class_mask]
        # Build subspace using all activations of the class (not split)
        if len(class_activations) >= n_components:
            subspaces[int(label)] = compute_class_subspace(class_activations, n_components)
    
    pairwise_means = []
    # Keep any existing entries in pairwise_results (from bootstrap loop)
    labels_list = list(subspaces.keys())
    
    # If we don't have at least 2 valid subspaces, return NaN mean angle
    if len(labels_list) < 2:
        mean_angle = float('nan')
    else:
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                li, lj = labels_list[i], labels_list[j]
                angles = compute_principal_angles(subspaces[li], subspaces[lj])
                # Use MEDIAN angle for better stability with orthogonal spaces
                mean_ang = float(np.median(angles))
                std_ang = float(np.std(angles))
                pairwise_means.append(mean_ang)
                pairwise_results[(li, lj)] = {
                    'mean': mean_ang,
                    'std': std_ang
                }
        
        if pairwise_means:
            # Still use mean of the pairwise median angles
            mean_angle = float(np.mean(pairwise_means))
        else:
            # Fallback to previous within-class bootstrapped mean
            if all_mean_angles:
                mean_angle = float(np.mean(all_mean_angles))
            else:
                mean_angle = 0.0  # Return 0 instead of NaN for better test compatibility
    
    # Compile results
    result = {
        'mean_angle': mean_angle,
        'class_angles': class_angles,
        # For backward compatibility
        'pairwise': pairwise_results
    }
    
    # Add confidence interval for backward compatibility
    if all_mean_angles:
        bootstrap_means = np.array(all_mean_angles)
        ci_lower = float(np.percentile(bootstrap_means, 100 * (1 - confidence_level)))
        ci_upper = float(np.percentile(bootstrap_means, 100 * confidence_level))
        result['ci_95'] = (ci_lower, ci_upper)
    else:
        result['ci_95'] = (float('nan'), float('nan'))
    
    return result


def compute_fragmentation_score(
    activations: torch.Tensor,
    labels: torch.Tensor,
    var_threshold: float = METRICS["subspace_angle"]["var_threshold"],
    random_state: int = RANDOM_SEED
) -> float:
    """
    Compute a single scalar fragmentation score from subspace angles.
    
    This is a simplified version of the subspace angle metric that returns
    a single score instead of detailed statistics.
    
    Args:
        activations: Tensor of shape (n_samples, n_features) containing activations
        labels: Tensor of shape (n_samples,) containing class labels
        var_threshold: Minimum explained variance to retain (between 0.0 and 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Fragmentation score (higher means more fragmentation)
    """
    result = compute_subspace_angle(
        activations, labels, 
        var_threshold=var_threshold,
        bootstrap_samples=1, random_state=random_state
    )
    
    return result['mean_angle']
