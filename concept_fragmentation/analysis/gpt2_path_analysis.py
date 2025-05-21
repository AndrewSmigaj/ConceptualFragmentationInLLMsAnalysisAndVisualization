"""
GPT-2 specific Archetypal Path Analysis for concept fragmentation.

This module adapts the Archetypal Path Analysis framework from the paper
"Foundations of Archetypal Path Analysis: Toward a Principled Geometry 
for Cluster-Based Interpretability" to GPT-2 transformer models.

It focuses on identifying layers with high concept fragmentation and
analyzing paths through 3-layer windows centered on these critical layers.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from collections import Counter, defaultdict
import warnings
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import existing APA functionality
from concept_fragmentation.analysis.cross_layer_metrics import (
    compute_trajectory_fragmentation,
    compute_inter_cluster_path_density,
    extract_paths,
    validate_layer_order
)

# Import transformer-specific functionality
from concept_fragmentation.analysis.transformer_dimensionality import (
    TransformerDimensionalityReducer
)

# Set up logger
logger = logging.getLogger(__name__)


def calculate_layer_fragmentation_scores(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    n_clusters: int = 8,
    random_state: int = 42,
    dimensionality_reduction: bool = True,
    n_components: int = 50,
    dim_reducer: Optional[TransformerDimensionalityReducer] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate fragmentation scores for each layer in a GPT-2 model.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
                          [batch_size, seq_len, hidden_dim]
        n_clusters: Number of clusters to use for path analysis
        random_state: Random seed for reproducibility
        dimensionality_reduction: Whether to apply dimensionality reduction
        n_components: Number of components for dimensionality reduction
        dim_reducer: Optional dimensionality reducer
        
    Returns:
        Dictionary mapping layer pairs to fragmentation metrics
    """
    logger.info("Calculating layer-wise fragmentation scores")
    
    # Initialize dimensionality reducer if not provided
    if dim_reducer is None and dimensionality_reduction:
        dim_reducer = TransformerDimensionalityReducer(
            random_state=random_state,
            use_cache=True
        )
    
    # Ensure layer order is consistent
    layer_names = validate_layer_order(list(layer_activations.keys()))
    
    # We need at least 2 layers to calculate fragmentation
    if len(layer_names) < 2:
        warnings.warn("Need at least 2 layers to calculate fragmentation")
        return {}
    
    # Cluster activations for each layer
    logger.info(f"Clustering {len(layer_names)} layers with {n_clusters} clusters each")
    layer_clusters = {}
    
    for layer_name in layer_names:
        # Get activations for this layer
        activations = layer_activations[layer_name]
        
        # Convert to numpy if it's a torch tensor
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Handle different shapes - we need [n_samples, n_features]
        if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activations.shape
            # Reshape to [batch_size * seq_len, hidden_dim]
            activations_2d = activations.reshape(-1, hidden_dim)
        else:
            activations_2d = activations
        
        # Apply dimensionality reduction if needed
        if dimensionality_reduction and activations_2d.shape[1] > n_components and dim_reducer:
            logger.info(f"Reducing dimensionality for layer {layer_name} from {activations_2d.shape[1]} to {n_components}")
            reduction_result = dim_reducer.reduce_dimensionality(
                activations=activations_2d,
                n_components=n_components,
                method="auto",
                layer_name=layer_name
            )
            
            if reduction_result.success:
                activations_2d = reduction_result.reduced_activations
            else:
                logger.warning(f"Dimensionality reduction failed for layer {layer_name}, using original")
        
        # Perform clustering
        logger.info(f"Clustering layer {layer_name}")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        # Fit the model and predict cluster labels
        cluster_labels = kmeans.fit_predict(activations_2d)
        
        # For 3D activations, reshape labels back to [batch_size, seq_len]
        if len(activations.shape) == 3:
            cluster_labels = cluster_labels.reshape(batch_size, seq_len)
        
        # Store cluster assignments
        layer_clusters[layer_name] = cluster_labels
        
        logger.info(f"Completed clustering for layer {layer_name}")
    
    # Calculate pairwise fragmentation scores between consecutive layers
    logger.info("Calculating fragmentation scores between consecutive layers")
    layer_fragmentation = {}
    
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Get cluster assignments for these two layers only
        two_layer_clusters = {
            layer1: layer_clusters[layer1],
            layer2: layer_clusters[layer2]
        }
        
        # Extract paths through these two layers
        paths, _ = extract_paths(two_layer_clusters)
        
        # Calculate fragmentation metrics
        fragmentation = compute_trajectory_fragmentation(
            paths=paths,
            layer_names=[layer1, layer2]
        )
        
        # Store results
        layer_pair = f"{layer1}_to_{layer2}"
        layer_fragmentation[layer_pair] = {
            "overall_entropy": fragmentation["overall_entropy"],
            "normalized_entropy": fragmentation["normalized_entropy"],
            "unique_paths": fragmentation["unique_paths"],
            "transition_entropies": fragmentation["transition_entropies"][0] if fragmentation["transition_entropies"] else 0.0
        }
        
        logger.info(f"Fragmentation for {layer_pair}: {layer_fragmentation[layer_pair]['normalized_entropy']:.4f}")
    
    # Calculate global fragmentation across all layers
    all_layer_clusters = layer_clusters
    all_paths, _ = extract_paths(all_layer_clusters)
    
    all_fragmentation = compute_trajectory_fragmentation(
        paths=all_paths,
        layer_names=layer_names
    )
    
    layer_fragmentation["all_layers"] = {
        "overall_entropy": all_fragmentation["overall_entropy"],
        "normalized_entropy": all_fragmentation["normalized_entropy"],
        "unique_paths": all_fragmentation["unique_paths"],
        "transition_entropies": all_fragmentation["transition_entropies"]
    }
    
    logger.info(f"Global fragmentation across all layers: {layer_fragmentation['all_layers']['normalized_entropy']:.4f}")
    
    return layer_fragmentation


def rank_layers_by_fragmentation(
    layer_fragmentation: Dict[str, Dict[str, float]],
    metric: str = "normalized_entropy",
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Rank layer pairs by fragmentation score.
    
    Args:
        layer_fragmentation: Dictionary mapping layer pairs to fragmentation metrics
        metric: Metric to use for ranking ('normalized_entropy', 'overall_entropy', etc.)
        top_k: Number of top layer pairs to return
        
    Returns:
        List of (layer_pair, score) tuples, sorted by score (descending)
    """
    # Filter out the 'all_layers' entry
    layer_pairs = [(pair, metrics[metric]) 
                  for pair, metrics in layer_fragmentation.items() 
                  if pair != "all_layers" and metric in metrics]
    
    # Sort by score in descending order
    ranked_layers = sorted(layer_pairs, key=lambda x: x[1], reverse=True)
    
    # Return top-k
    return ranked_layers[:top_k]


def get_window_layers(
    ranked_layers: List[Tuple[str, float]],
    layer_names: List[str],
    window_size: int = 3
) -> List[List[str]]:
    """
    Get n-layer windows centered on the highest fragmentation layers.
    
    Args:
        ranked_layers: List of (layer_pair, score) tuples from rank_layers_by_fragmentation
        layer_names: List of all layer names
        window_size: Size of window to create (default: 3 for prev, current, next)
        
    Returns:
        List of lists, where each inner list contains layer names for a window
    """
    windows = []
    
    for layer_pair, _ in ranked_layers:
        # Parse layer names from the pair (format: "layer1_to_layer2")
        parts = layer_pair.split("_to_")
        if len(parts) != 2:
            continue
        
        # Get the two layers and find their indices
        layer1, layer2 = parts
        try:
            idx1 = layer_names.index(layer1)
            idx2 = layer_names.index(layer2)
        except ValueError:
            continue
        
        # Focus on the transition point between these layers
        # We'll use the midpoint as our center
        center_idx = (idx1 + idx2) // 2
        
        # Create window around this center point
        half_width = window_size // 2
        window_start = max(0, center_idx - half_width)
        window_end = min(len(layer_names), center_idx + half_width + 1)
        
        # Extract the window layers
        window_layers = layer_names[window_start:window_end]
        
        # Only add if we haven't already included this window
        if window_layers not in windows:
            windows.append(window_layers)
    
    return windows


def analyze_layer_fragmentation(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    n_clusters: int = 8,
    top_k: int = 3,
    window_size: int = 3,
    random_state: int = 42,
    dimensionality_reduction: bool = True,
    n_components: int = 50
) -> Dict[str, Any]:
    """
    Analyze layer fragmentation and identify high-fragmentation windows.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
        n_clusters: Number of clusters for path analysis
        top_k: Number of top fragmentation points to identify
        window_size: Size of layer window to analyze around each point
        random_state: Random seed
        dimensionality_reduction: Whether to apply dimensionality reduction
        n_components: Number of components for dimensionality reduction
        
    Returns:
        Dictionary with analysis results
    """
    # Calculate fragmentation scores for all layer pairs
    fragmentation_scores = calculate_layer_fragmentation_scores(
        layer_activations=layer_activations,
        n_clusters=n_clusters,
        random_state=random_state,
        dimensionality_reduction=dimensionality_reduction,
        n_components=n_components
    )
    
    # Rank layers by fragmentation
    ranked_layers = rank_layers_by_fragmentation(
        layer_fragmentation=fragmentation_scores,
        metric="normalized_entropy",
        top_k=top_k
    )
    
    # Get layer names in order
    layer_names = validate_layer_order(list(layer_activations.keys()))
    
    # Get layer windows for analysis
    windows = get_window_layers(
        ranked_layers=ranked_layers,
        layer_names=layer_names,
        window_size=window_size
    )
    
    # Prepare results
    return {
        "fragmentation_scores": fragmentation_scores,
        "ranked_layers": ranked_layers,
        "windows": windows,
        "global_fragmentation": fragmentation_scores.get("all_layers", {})
    }


def extract_window_activations(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    window_layers: List[str]
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Extract activations for a specific window of layers.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
        window_layers: List of layer names to include in the window
        
    Returns:
        Dictionary with activations for only the specified layers
    """
    return {layer: layer_activations[layer] for layer in window_layers if layer in layer_activations}


def analyze_window_paths(
    window_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    n_clusters: int = 8,
    random_state: int = 42,
    dimensionality_reduction: bool = True,
    n_components: int = 50,
    dim_reducer: Optional[TransformerDimensionalityReducer] = None
) -> Dict[str, Any]:
    """
    Analyze paths through a specific window of layers.
    
    Args:
        window_activations: Dictionary mapping layer names to activation tensors
        n_clusters: Number of clusters for path analysis
        random_state: Random seed
        dimensionality_reduction: Whether to apply dimensionality reduction
        n_components: Number of components for dimensionality reduction
        dim_reducer: Optional dimensionality reducer
        
    Returns:
        Dictionary with window path analysis results
    """
    # Initialize dimensionality reducer if not provided
    if dim_reducer is None and dimensionality_reduction:
        dim_reducer = TransformerDimensionalityReducer(
            random_state=random_state,
            use_cache=True
        )
    
    # Ensure layer order is consistent
    layer_names = validate_layer_order(list(window_activations.keys()))
    
    # We need at least 2 layers to calculate paths
    if len(layer_names) < 2:
        warnings.warn("Need at least 2 layers to calculate paths")
        return {}
    
    # Cluster activations for each layer
    logger.info(f"Clustering {len(layer_names)} layers in window with {n_clusters} clusters each")
    layer_clusters = {}
    
    for layer_name in layer_names:
        # Get activations for this layer
        activations = window_activations[layer_name]
        
        # Convert to numpy if it's a torch tensor
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()
        
        # Handle different shapes - we need [n_samples, n_features]
        if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activations.shape
            # Reshape to [batch_size * seq_len, hidden_dim]
            activations_2d = activations.reshape(-1, hidden_dim)
        else:
            activations_2d = activations
        
        # Apply dimensionality reduction if needed
        if dimensionality_reduction and activations_2d.shape[1] > n_components and dim_reducer:
            logger.info(f"Reducing dimensionality for layer {layer_name} from {activations_2d.shape[1]} to {n_components}")
            reduction_result = dim_reducer.reduce_dimensionality(
                activations=activations_2d,
                n_components=n_components,
                method="auto",
                layer_name=layer_name
            )
            
            if reduction_result.success:
                activations_2d = reduction_result.reduced_activations
            else:
                logger.warning(f"Dimensionality reduction failed for layer {layer_name}, using original")
        
        # Perform clustering
        logger.info(f"Clustering layer {layer_name}")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        
        # Fit the model and predict cluster labels
        cluster_labels = kmeans.fit_predict(activations_2d)
        
        # For 3D activations, reshape labels back to [batch_size, seq_len]
        if len(activations.shape) == 3:
            cluster_labels = cluster_labels.reshape(batch_size, seq_len)
        
        # Store cluster assignments
        layer_clusters[layer_name] = cluster_labels
        
        logger.info(f"Completed clustering for layer {layer_name}")
    
    # Extract paths through these layers
    paths, used_layer_names = extract_paths(layer_clusters)
    
    # Calculate path metrics using existing functionality
    fragmentation = compute_trajectory_fragmentation(
        paths=paths,
        layer_names=used_layer_names
    )
    
    # Compute inter-cluster path density
    path_density = compute_inter_cluster_path_density(
        paths=paths,
        layer_names=used_layer_names,
        min_density=0.05,
        max_steps=len(used_layer_names) - 1
    )
    
    # Compute path counts and frequencies
    path_tuples = [tuple(path) for path in paths]
    path_counts = Counter(path_tuples)
    
    # Sort by frequency (descending)
    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create human-readable path strings
    top_paths = []
    for path_tuple, count in sorted_paths[:10]:  # Top 10 paths
        path_str = "→".join(f"L{i}C{c}" for i, c in enumerate(path_tuple))
        top_paths.append({
            "path": list(path_tuple),
            "path_str": path_str,
            "count": count,
            "percentage": 100 * count / len(paths)
        })
    
    # Return comprehensive analysis results
    return {
        "fragmentation": fragmentation,
        "path_density": path_density,
        "top_paths": top_paths,
        "paths": paths,
        "layer_clusters": layer_clusters,
        "layer_names": used_layer_names
    }


def compare_window_analyses(
    window_analyses: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare analysis results across different windows.
    
    Args:
        window_analyses: Dictionary mapping window names to analysis results
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # Extract fragmentation metrics for comparison
    fragmentation_comparison = {}
    for window_name, analysis in window_analyses.items():
        if "fragmentation" in analysis:
            fragmentation = analysis["fragmentation"]
            fragmentation_comparison[window_name] = {
                "overall_entropy": fragmentation.get("overall_entropy", 0.0),
                "normalized_entropy": fragmentation.get("normalized_entropy", 0.0),
                "unique_paths": fragmentation.get("unique_paths", 0)
            }
    
    # Extract path density metrics for comparison
    density_comparison = {}
    for window_name, analysis in window_analyses.items():
        if "path_density" in analysis:
            path_density = analysis["path_density"]
            # Average density across all layer pairs
            densities = []
            for layer_pair, density_info in path_density.items():
                if "matrix" in density_info:
                    density = np.mean(density_info["matrix"])
                    densities.append(density)
            
            if densities:
                density_comparison[window_name] = float(np.mean(densities))
            else:
                density_comparison[window_name] = 0.0
    
    # Compare top paths
    path_comparison = {}
    for window_name, analysis in window_analyses.items():
        if "top_paths" in analysis:
            top_paths = analysis["top_paths"]
            # Extract statistics about top paths
            path_comparison[window_name] = {
                "top_path_percentage": top_paths[0]["percentage"] if top_paths else 0.0,
                "top3_coverage": sum(p["percentage"] for p in top_paths[:3]) if len(top_paths) >= 3 else 0.0,
                "top_path": top_paths[0]["path_str"] if top_paths else "None"
            }
    
    # Store results
    comparison["fragmentation"] = fragmentation_comparison
    comparison["density"] = density_comparison
    comparison["paths"] = path_comparison
    
    return comparison


def analyze_all_windows(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    n_clusters: int = 8,
    top_k: int = 3,
    window_size: int = 3,
    random_state: int = 42,
    dimensionality_reduction: bool = True,
    n_components: int = 50
) -> Dict[str, Any]:
    """
    Analyze all high-fragmentation windows in a GPT-2 model.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
        n_clusters: Number of clusters for path analysis
        top_k: Number of top fragmentation points to identify
        window_size: Size of layer window to analyze around each point
        random_state: Random seed
        dimensionality_reduction: Whether to apply dimensionality reduction
        n_components: Number of components for dimensionality reduction
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    # First, identify high-fragmentation windows
    fragmentation_analysis = analyze_layer_fragmentation(
        layer_activations=layer_activations,
        n_clusters=n_clusters,
        top_k=top_k,
        window_size=window_size,
        random_state=random_state,
        dimensionality_reduction=dimensionality_reduction,
        n_components=n_components
    )
    
    # Get the windows to analyze
    windows = fragmentation_analysis["windows"]
    
    # Analyze each window
    window_analyses = {}
    
    for i, window_layers in enumerate(windows):
        window_name = f"window_{i+1}"
        logger.info(f"Analyzing {window_name}: {window_layers}")
        
        # Extract activations for this window
        window_activations = extract_window_activations(
            layer_activations=layer_activations,
            window_layers=window_layers
        )
        
        # Analyze paths through this window
        window_analysis = analyze_window_paths(
            window_activations=window_activations,
            n_clusters=n_clusters,
            random_state=random_state,
            dimensionality_reduction=dimensionality_reduction,
            n_components=n_components
        )
        
        # Store results
        window_analyses[window_name] = window_analysis
    
    # Compare analyses across windows
    window_comparison = compare_window_analyses(window_analyses)
    
    # Return comprehensive results
    return {
        "fragmentation_analysis": fragmentation_analysis,
        "window_analyses": window_analyses,
        "window_comparison": window_comparison
    }


if __name__ == "__main__":
    # Example usage
    import torch
    import numpy as np
    
    # Create dummy activation data
    n_layers = 12  # GPT-2 small has 12 layers
    batch_size = 4
    seq_len = 16
    hidden_dim = 768  # GPT-2 small hidden dimension
    
    # Generate random activations
    activations = {}
    for i in range(n_layers):
        layer_name = f"layer{i}"
        activations[layer_name] = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Analyze all windows
    results = analyze_all_windows(
        layer_activations=activations,
        n_clusters=8,
        top_k=3,
        window_size=3,
        random_state=42,
        dimensionality_reduction=True,
        n_components=50
    )
    
    # Print results
    print("\nLayer Fragmentation Analysis Results:")
    print("-" * 40)
    
    print("\nTop fragmentation points:")
    for layer_pair, score in results["fragmentation_analysis"]["ranked_layers"]:
        print(f"  {layer_pair}: {score:.4f}")
    
    print("\nAnalysis windows:")
    for i, window in enumerate(results["fragmentation_analysis"]["windows"]):
        print(f"  Window {i+1}: {window}")
    
    print("\nWindow comparisons:")
    for window_name, metrics in results["window_comparison"]["fragmentation"].items():
        print(f"  {window_name}: Normalized Entropy = {metrics['normalized_entropy']:.4f}, Unique Paths = {metrics['unique_paths']}")
    
    print("\nTop path in each window:")
    for window_name, path_info in results["window_comparison"]["paths"].items():
        print(f"  {window_name}: {path_info['top_path']} ({path_info['top_path_percentage']:.2f}%)")