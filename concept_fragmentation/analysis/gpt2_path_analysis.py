"""
GPT-2 specific Archetypal Path Analysis for concept fragmentation.

This module adapts the Archetypal Path Analysis framework from the paper
"Foundations of Archetypal Path Analysis: Toward a Principled Geometry 
for Cluster-Based Interpretability" to GPT-2 transformer models.

It focuses on identifying layers with high concept fragmentation and
analyzing paths through 3-layer windows centered on these critical layers.

Extended for Phase 3.1: GPT-2 Semantic Subtypes Experiment
- Analyzes 774 semantic subtype words across 8 categories 
- Demonstrates within-subtype path coherence and between-subtype differentiation
- Generates comprehensive case study reports for transformer interpretability
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


def analyze_semantic_subtypes_paths(
    curated_words_file: str = "gpt2_semantic_subtypes_curated.json",
    model_name: str = "gpt2",
    max_memory_mb: int = 8000,
    cache_dir: Optional[str] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Analyze archetypal paths for semantic subtypes using existing APA framework.
    
    This function extends the GPT-2 path analysis to study how different semantic
    subtypes (concrete/abstract nouns, physical/emotive adjectives, etc.) follow
    different archetypal paths through GPT-2's layers.
    
    Args:
        curated_words_file: Path to the semantic subtypes curated words file
        model_name: GPT-2 model to analyze ('gpt2', 'gpt2-medium', etc.)
        max_memory_mb: Maximum memory usage for activation extraction
        cache_dir: Cache directory for model/tokenizer
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing semantic subtypes path analysis results
    """
    import json
    import os
    import sys
    from pathlib import Path
    
    # Add path for activation extractor
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logger.info("Starting semantic subtypes archetypal path analysis")
    logger.info(f"Model: {model_name}")
    logger.info(f"Curated words file: {curated_words_file}")
    
    # Load semantic subtypes words
    logger.info("Loading semantic subtypes words...")
    
    # Handle relative path - look in project root
    if not os.path.isabs(curated_words_file):
        project_root = Path(__file__).parent.parent.parent
        curated_words_file = project_root / curated_words_file
    
    if not os.path.exists(curated_words_file):
        raise FileNotFoundError(f"Curated words file not found: {curated_words_file}")
    
    with open(curated_words_file, 'r') as f:
        curated_data = json.load(f)
    
    # Extract semantic subtypes
    if "curated_words" in curated_data:
        semantic_subtypes = curated_data["curated_words"]
    else:
        raise ValueError("Invalid curated words file format - expected 'curated_words' key")
    
    logger.info(f"Loaded {len(semantic_subtypes)} semantic subtypes:")
    for subtype, words in semantic_subtypes.items():
        logger.info(f"  {subtype}: {len(words)} words")
    
    # Prepare words for activation extraction
    all_words = []
    word_to_subtype = {}
    
    for subtype, words in semantic_subtypes.items():
        for word in words:
            all_words.append(word)
            word_to_subtype[word] = subtype
    
    logger.info(f"Total words for analysis: {len(all_words)}")
    
    # Extract GPT-2 activations using existing enhanced extractor
    logger.info("Extracting GPT-2 activations...")
    
    try:
        from gpt2_activation_extractor import SimpleGPT2ActivationExtractor
    except ImportError:
        raise ImportError("Could not import SimpleGPT2ActivationExtractor. Make sure it's in the path.")
    
    # Initialize extractor with memory management
    extractor = SimpleGPT2ActivationExtractor(
        model_name=model_name,
        cache_dir=cache_dir,
        max_memory_mb=max_memory_mb
    )
    
    # Extract activations for all words
    activations_data = extractor.extract_activations(all_words)
    
    # Add semantic subtype information to metadata
    activations_data['semantic_subtypes'] = semantic_subtypes
    activations_data['word_to_subtype'] = word_to_subtype
    
    logger.info("Activation extraction completed")
    logger.info(f"Extracted activations for {len(activations_data['sentences'])} words")
    logger.info(f"Number of layers: {activations_data['metadata']['num_layers']}")
    
    # Prepare return data for existing APA pipeline
    results = {
        'activations_data': activations_data,
        'semantic_subtypes': semantic_subtypes,
        'word_to_subtype': word_to_subtype,
        'total_words': len(all_words),
        'analysis_metadata': {
            'model_name': model_name,
            'extraction_timestamp': activations_data['metadata'].get('timestamp'),
            'num_layers': activations_data['metadata']['num_layers'],
            'hidden_dim': activations_data['metadata']['hidden_dim'],
            'semantic_subtypes_count': len(semantic_subtypes),
            'words_per_subtype': {subtype: len(words) for subtype, words in semantic_subtypes.items()}
        }
    }
    
    logger.info("Semantic subtypes activation extraction completed successfully")
    logger.info("Data ready for APA pipeline analysis")
    
    return results


def apply_apa_pipeline_to_semantic_subtypes(
    semantic_subtypes_data: Dict[str, Any],
    clustering_method: str = 'kmeans',
    k_range: Tuple[int, int] = (2, 15),
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Apply existing APA pipeline to semantic subtypes activation data.
    
    Uses existing cluster_paths.py functions and enhanced GPT2PivotClusterer
    to find archetypal paths within and between semantic subtypes.
    
    Args:
        semantic_subtypes_data: Output from analyze_semantic_subtypes_paths()
        clustering_method: Clustering method ('kmeans' or 'hdbscan')
        k_range: Range of k values for clustering
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing APA pipeline results for semantic subtypes
    """
    import sys
    import numpy as np
    from pathlib import Path
    
    # Add paths for required modules
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logger.info("Applying APA pipeline to semantic subtypes data")
    logger.info(f"Clustering method: {clustering_method}")
    logger.info(f"K range: {k_range}")
    
    # Extract data from input
    activations_data = semantic_subtypes_data['activations_data']
    semantic_subtypes = semantic_subtypes_data['semantic_subtypes']
    word_to_subtype = semantic_subtypes_data['word_to_subtype']
    
    logger.info(f"Processing {len(semantic_subtypes)} semantic subtypes")
    logger.info(f"Total words: {semantic_subtypes_data['total_words']}")
    
    # Step 1: Apply clustering using enhanced GPT2PivotClusterer
    logger.info("Step 1: Clustering activations...")
    
    try:
        from gpt2_pivot_clusterer import GPT2PivotClusterer
    except ImportError:
        raise ImportError("Could not import GPT2PivotClusterer. Make sure it's in the path.")
    
    # Initialize clusterer with specified method
    clusterer = GPT2PivotClusterer(
        k_range=k_range,
        random_state=random_state,
        clustering_method=clustering_method
    )
    
    # Run clustering on all activation data
    clustering_results = clusterer.cluster_all_layers(activations_data)
    
    logger.info("Clustering completed")
    logger.info(f"Clustered {len(clustering_results['layer_results'])} layers")
    
    # Step 2: Compute cluster paths using existing APA functions
    logger.info("Step 2: Computing cluster paths...")
    
    try:
        from concept_fragmentation.analysis.cluster_paths import (
            compute_cluster_paths,
            compute_path_archetypes
        )
    except ImportError:
        raise ImportError("Could not import cluster_paths functions. Check module path.")
    
    # Use existing token_paths from clustering results - this is already the paths data we need
    token_paths = clustering_results['token_paths']
    
    # Convert token_paths to the format expected by path analysis
    # token_paths is Dict[sent_idx][token_idx] = [layer0_cluster, layer1_cluster, ...]
    all_paths = []
    human_readable_paths = []
    
    for sent_idx in sorted(token_paths.keys()):
        for token_idx in sorted(token_paths[sent_idx].keys()):
            path = token_paths[sent_idx][token_idx]
            all_paths.append(path)
            human_readable_paths.append("→".join(path))
    
    # Convert to numpy array for consistency
    unique_paths = np.array(all_paths)
    original_paths = unique_paths.copy()  # Same as unique paths in this case
    
    # Create layer names from clustering results
    layer_names = [f"layer_{i}" for i in range(len(all_paths[0]) if all_paths else 0)]
    
    # Create dummy id_to_layer_cluster for compatibility
    id_to_layer_cluster = {}
    
    logger.info("Cluster paths computed")
    logger.info(f"Found {len(unique_paths)} unique paths across {len(layer_names)} layers")
    
    # Step 3: Analyze paths within semantic subtypes
    logger.info("Step 3: Analyzing paths within semantic subtypes...")
    
    within_subtype_analysis = {}
    
    for subtype, words in semantic_subtypes.items():
        logger.info(f"Analyzing subtype: {subtype} ({len(words)} words)")
        
        # Find sentence indices for this subtype
        subtype_indices = []
        for sent_idx, sentence in clustering_results['sentences'].items():
            if sentence in words:
                subtype_indices.append(sent_idx)
        
        if not subtype_indices:
            logger.warning(f"No sentences found for subtype {subtype}")
            continue
        
        # Extract paths for this subtype
        subtype_paths = []
        subtype_human_paths = []
        for idx in subtype_indices:
            if idx < len(unique_paths):
                subtype_paths.append(unique_paths[idx])
                subtype_human_paths.append(human_readable_paths[idx])
        
        # Create DataFrame for compute_path_archetypes
        import pandas as pd
        
        subtype_df = pd.DataFrame({
            'word': [clustering_results['sentences'][idx] for idx in subtype_indices if idx in clustering_results['sentences']],
            'subtype': [subtype] * len(subtype_indices)
        })
        
        # Compute archetypal paths for this subtype
        if len(subtype_paths) > 0:
            subtype_archetypes = compute_path_archetypes(
                paths=np.array(subtype_paths),
                layer_names=layer_names,
                df=subtype_df,
                dataset_name=f"semantic_subtype_{subtype}",
                id_to_layer_cluster=id_to_layer_cluster,
                human_readable_paths=subtype_human_paths,
                top_k=min(5, len(subtype_paths)),  # Limit to available paths
                max_members=50
            )
            
            within_subtype_analysis[subtype] = {
                'word_count': len(words),
                'paths_found': len(subtype_paths),
                'unique_paths': len(set(subtype_human_paths)),
                'archetypal_paths': subtype_archetypes,
                'path_diversity': len(set(subtype_human_paths)) / len(subtype_paths) if subtype_paths else 0
            }
        else:
            within_subtype_analysis[subtype] = {
                'word_count': len(words),
                'paths_found': 0,
                'unique_paths': 0,
                'archetypal_paths': [],
                'path_diversity': 0
            }
    
    # Step 4: Analyze paths between semantic subtypes
    logger.info("Step 4: Analyzing paths between semantic subtypes...")
    
    # Create full dataset DataFrame for between-subtype analysis
    full_df_data = []
    for sent_idx, sentence in clustering_results['sentences'].items():
        subtype = word_to_subtype.get(sentence, 'unknown')
        full_df_data.append({'word': sentence, 'subtype': subtype})
    
    full_df = pd.DataFrame(full_df_data)
    
    # Compute overall archetypal paths across all subtypes
    overall_archetypes = compute_path_archetypes(
        paths=unique_paths,
        layer_names=layer_names,
        df=full_df,
        dataset_name="semantic_subtypes_overall",
        id_to_layer_cluster=id_to_layer_cluster,
        human_readable_paths=human_readable_paths,
        demographic_columns=['subtype'],
        top_k=10,
        max_members=100
    )
    
    # Analyze between-subtype patterns
    between_subtype_analysis = {
        'overall_archetypal_paths': overall_archetypes,
        'cross_subtype_patterns': {},
        'subtype_path_sharing': {}
    }
    
    # Find paths shared between subtypes
    all_subtype_paths = {}
    for subtype in semantic_subtypes.keys():
        if subtype in within_subtype_analysis:
            archetype_paths = within_subtype_analysis[subtype]['archetypal_paths']
            subtype_paths_set = set()
            for archetype in archetype_paths:
                subtype_paths_set.add(archetype.get('path_string', ''))
            all_subtype_paths[subtype] = subtype_paths_set
    
    # Calculate path sharing between subtypes
    for i, subtype1 in enumerate(semantic_subtypes.keys()):
        if subtype1 not in all_subtype_paths:
            continue
        for subtype2 in list(semantic_subtypes.keys())[i+1:]:
            if subtype2 not in all_subtype_paths:
                continue
            
            shared_paths = all_subtype_paths[subtype1] & all_subtype_paths[subtype2]
            total_paths = all_subtype_paths[subtype1] | all_subtype_paths[subtype2]
            
            if total_paths:
                sharing_ratio = len(shared_paths) / len(total_paths)
                between_subtype_analysis['subtype_path_sharing'][f"{subtype1}_vs_{subtype2}"] = {
                    'shared_paths': len(shared_paths),
                    'total_unique_paths': len(total_paths),
                    'sharing_ratio': sharing_ratio,
                    'shared_path_list': list(shared_paths)
                }
    
    # Prepare final results
    results = {
        'clustering_results': clustering_results,
        'cluster_paths': {
            'unique_paths': unique_paths,
            'layer_names': layer_names,
            'id_to_layer_cluster': id_to_layer_cluster,
            'original_paths': original_paths,
            'human_readable_paths': human_readable_paths
        },
        'within_subtype_analysis': within_subtype_analysis,
        'between_subtype_analysis': between_subtype_analysis,
        'semantic_subtypes': semantic_subtypes,
        'word_to_subtype': word_to_subtype,
        'pipeline_metadata': {
            'clustering_method': clustering_method,
            'k_range': k_range,
            'total_layers': len(layer_names),
            'total_paths': len(unique_paths),
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
    }
    
    logger.info("APA pipeline application completed successfully")
    logger.info(f"Within-subtype analysis: {len(within_subtype_analysis)} subtypes")
    logger.info(f"Between-subtype analysis: {len(between_subtype_analysis['subtype_path_sharing'])} comparisons")
    
    return results


def calculate_semantic_subtypes_apa_metrics(
    apa_pipeline_results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate comprehensive APA metrics for semantic subtypes analysis.
    
    Uses existing GPT2APAMetricsCalculator to compute all standard APA metrics
    plus semantic organization specific metrics.
    
    Args:
        apa_pipeline_results: Output from apply_apa_pipeline_to_semantic_subtypes()
        
    Returns:
        Dictionary containing comprehensive APA metrics for semantic subtypes
    """
    import sys
    import numpy as np
    from pathlib import Path
    
    # Add paths for required modules
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logger.info("Calculating APA metrics for semantic subtypes analysis")
    
    # Extract data from pipeline results
    clustering_results = apa_pipeline_results['clustering_results']
    within_subtype_analysis = apa_pipeline_results['within_subtype_analysis']
    between_subtype_analysis = apa_pipeline_results['between_subtype_analysis']
    semantic_subtypes = apa_pipeline_results['semantic_subtypes']
    cluster_paths = apa_pipeline_results['cluster_paths']
    
    logger.info(f"Computing metrics for {len(semantic_subtypes)} semantic subtypes")
    
    # Step 1: Calculate overall APA metrics using existing calculator
    logger.info("Step 1: Computing overall APA metrics...")
    
    try:
        from gpt2_apa_metrics import GPT2APAMetricsCalculator
    except ImportError:
        raise ImportError("Could not import GPT2APAMetricsCalculator. Make sure it's in the path.")
    
    # Initialize metrics calculator
    metrics_calculator = GPT2APAMetricsCalculator()
    
    # Compute overall metrics for the entire clustering
    overall_apa_metrics = metrics_calculator.calculate_all_metrics(clustering_results)
    
    logger.info("Overall APA metrics computed")
    
    # Step 2: Calculate within-subtype metrics
    logger.info("Step 2: Computing within-subtype metrics...")
    
    within_subtype_metrics = {}
    
    for subtype, subtype_data in within_subtype_analysis.items():
        logger.info(f"Computing metrics for subtype: {subtype}")
        
        archetypal_paths = subtype_data.get('archetypal_paths', [])
        
        if not archetypal_paths:
            within_subtype_metrics[subtype] = {
                'word_count': subtype_data['word_count'],
                'paths_found': 0,
                'path_coherence': 0.0,
                'path_diversity': 0.0,
                'top_paths': [],
                'fragmentation_stats': {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
            }
            continue
        
        # Calculate path coherence (how similar paths are within subtype)
        path_frequencies = [archetype.get('frequency', 0) for archetype in archetypal_paths]
        total_paths = sum(path_frequencies)
        
        if total_paths > 0:
            # Calculate normalized entropy as measure of path diversity
            path_probs = [freq / total_paths for freq in path_frequencies]
            path_entropy = -sum(p * np.log2(p) for p in path_probs if p > 0)
            max_entropy = np.log2(len(path_frequencies)) if len(path_frequencies) > 1 else 1.0
            normalized_entropy = path_entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Path coherence is inverse of normalized entropy
            path_coherence = 1.0 - normalized_entropy
        else:
            path_coherence = 0.0
            normalized_entropy = 0.0
        
        # Extract top paths for this subtype
        top_paths = []
        for i, archetype in enumerate(archetypal_paths[:5]):  # Top 5 paths
            path_info = {
                'rank': i + 1,
                'path_string': archetype.get('path_string', ''),
                'frequency': archetype.get('frequency', 0),
                'percentage': (archetype.get('frequency', 0) / total_paths * 100) if total_paths > 0 else 0.0,
                'member_count': len(archetype.get('members', []))
            }
            top_paths.append(path_info)
        
        # Calculate fragmentation statistics for paths in this subtype
        if archetypal_paths:
            # Use frequency as a proxy for fragmentation (more diverse = more fragmented)
            fragmentation_scores = [1.0 / (freq + 1) for freq in path_frequencies]  # Higher freq = lower fragmentation
            fragmentation_stats = {
                'mean': float(np.mean(fragmentation_scores)),
                'std': float(np.std(fragmentation_scores)),
                'min': float(np.min(fragmentation_scores)),
                'max': float(np.max(fragmentation_scores))
            }
        else:
            fragmentation_stats = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        within_subtype_metrics[subtype] = {
            'word_count': subtype_data['word_count'],
            'paths_found': subtype_data['paths_found'],
            'unique_paths': subtype_data['unique_paths'],
            'path_coherence': path_coherence,
            'path_diversity': subtype_data.get('path_diversity', 0.0),
            'normalized_entropy': normalized_entropy,
            'top_paths': top_paths,
            'fragmentation_stats': fragmentation_stats,
            'archetypal_path_count': len(archetypal_paths)
        }
    
    # Step 3: Calculate between-subtype metrics
    logger.info("Step 3: Computing between-subtype metrics...")
    
    between_subtype_metrics = {}
    
    # Analyze overall archetypal paths across subtypes
    overall_archetypes = between_subtype_analysis.get('overall_archetypal_paths', [])
    
    if overall_archetypes:
        # Calculate cross-subtype path sharing statistics
        path_sharing_stats = {}
        subtype_path_sharing = between_subtype_analysis.get('subtype_path_sharing', {})
        
        sharing_ratios = []
        for comparison, sharing_data in subtype_path_sharing.items():
            sharing_ratio = sharing_data.get('sharing_ratio', 0.0)
            sharing_ratios.append(sharing_ratio)
            
            path_sharing_stats[comparison] = {
                'shared_paths': sharing_data.get('shared_paths', 0),
                'total_unique_paths': sharing_data.get('total_unique_paths', 0),
                'sharing_ratio': sharing_ratio,
                'semantic_similarity': sharing_ratio  # Higher sharing = higher semantic similarity
            }
        
        # Overall between-subtype statistics
        if sharing_ratios:
            between_subtype_metrics = {
                'total_comparisons': len(sharing_ratios),
                'mean_sharing_ratio': float(np.mean(sharing_ratios)),
                'sharing_ratio_std': float(np.std(sharing_ratios)),
                'min_sharing': float(np.min(sharing_ratios)),
                'max_sharing': float(np.max(sharing_ratios)),
                'path_sharing_details': path_sharing_stats,
                'semantic_differentiation': 1.0 - float(np.mean(sharing_ratios))  # Lower sharing = higher differentiation
            }
        else:
            between_subtype_metrics = {
                'total_comparisons': 0,
                'mean_sharing_ratio': 0.0,
                'semantic_differentiation': 1.0
            }
    
    # Step 4: Calculate semantic organization metrics
    logger.info("Step 4: Computing semantic organization metrics...")
    
    semantic_organization_metrics = {
        'total_semantic_subtypes': len(semantic_subtypes),
        'words_per_subtype': {subtype: len(words) for subtype, words in semantic_subtypes.items()},
        'total_words_analyzed': sum(len(words) for words in semantic_subtypes.values()),
        'subtype_coherence_scores': {},
        'semantic_clustering_quality': {},
        'cross_subtype_relationships': {}
    }
    
    # Calculate coherence scores for each subtype
    for subtype in semantic_subtypes.keys():
        if subtype in within_subtype_metrics:
            coherence = within_subtype_metrics[subtype]['path_coherence']
            diversity = within_subtype_metrics[subtype]['path_diversity']
            
            semantic_organization_metrics['subtype_coherence_scores'][subtype] = {
                'path_coherence': coherence,
                'path_diversity': diversity,
                'clustering_quality': coherence * (1.0 - diversity) if diversity < 1.0 else 0.0  # High coherence, low diversity = good clustering
            }
    
    # Overall semantic clustering quality
    coherence_scores = [metrics['path_coherence'] for metrics in within_subtype_metrics.values()]
    if coherence_scores:
        semantic_organization_metrics['semantic_clustering_quality'] = {
            'mean_coherence': float(np.mean(coherence_scores)),
            'coherence_std': float(np.std(coherence_scores)),
            'min_coherence': float(np.min(coherence_scores)),
            'max_coherence': float(np.max(coherence_scores))
        }
    
    # Prepare final comprehensive results
    comprehensive_metrics = {
        'overall_apa_metrics': overall_apa_metrics,
        'within_subtype_metrics': within_subtype_metrics,
        'between_subtype_metrics': between_subtype_metrics,
        'semantic_organization_metrics': semantic_organization_metrics,
        'analysis_summary': {
            'total_layers': len(cluster_paths.get('layer_names', [])),
            'total_paths': len(cluster_paths.get('unique_paths', [])),
            'clustering_method': apa_pipeline_results['pipeline_metadata']['clustering_method'],
            'semantic_subtypes_analyzed': len(semantic_subtypes),
            'analysis_timestamp': apa_pipeline_results['pipeline_metadata']['analysis_timestamp']
        }
    }
    
    logger.info("APA metrics calculation completed successfully")
    logger.info(f"Computed metrics for {len(within_subtype_metrics)} subtypes")
    logger.info(f"Between-subtype comparisons: {between_subtype_metrics.get('total_comparisons', 0)}")
    
    return comprehensive_metrics


def generate_semantic_subtypes_llm_analysis(
    metrics_results: Dict[str, Any],
    provider: str = "openai",
    model: str = "gpt-4",
    temperature: float = 0.2,
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Generate LLM-powered analysis of semantic subtypes organization.
    
    Uses existing ClusterAnalysis from LLM framework with semantic organization
    instructions included in the prompt to analyze within-subtype coherence
    and between-subtype differentiation patterns.
    
    Args:
        metrics_results: Output from calculate_semantic_subtypes_apa_metrics()
        provider: LLM provider to use
        model: LLM model to use
        temperature: Temperature for generation
        use_cache: Whether to use caching
        
    Returns:
        Dictionary containing LLM-generated semantic organization analysis
    """
    import sys
    import json
    from pathlib import Path
    
    # Add paths for required modules
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    logger.info("Generating LLM-powered semantic subtypes analysis")
    logger.info(f"Using {provider} {model}")
    
    try:
        from concept_fragmentation.llm.analysis import ClusterAnalysis
    except ImportError:
        raise ImportError("Could not import ClusterAnalysis. Make sure LLM module is available.")
    
    # Initialize LLM analyzer
    analyzer = ClusterAnalysis(
        provider=provider,
        model=model,
        use_cache=use_cache,
        debug=False
    )
    
    # Extract key data for analysis
    within_subtype_metrics = metrics_results['within_subtype_metrics']
    between_subtype_metrics = metrics_results['between_subtype_metrics']
    semantic_organization_metrics = metrics_results['semantic_organization_metrics']
    analysis_summary = metrics_results['analysis_summary']
    
    logger.info("Step 1: Generating cluster labels for semantic subtypes")
    
    # Prepare cluster profiles for labeling
    cluster_profiles = {}
    
    for subtype, metrics in within_subtype_metrics.items():
        top_paths = metrics.get('top_paths', [])
        coherence = metrics.get('path_coherence', 0.0)
        diversity = metrics.get('path_diversity', 0.0)
        
        # Create profile describing this semantic subtype's clustering behavior
        profile_parts = [
            f"Semantic subtype: {subtype}",
            f"Word count: {metrics['word_count']}",
            f"Paths found: {metrics['paths_found']}",
            f"Path coherence: {coherence:.3f}",
            f"Path diversity: {diversity:.3f}",
            f"Unique paths: {metrics['unique_paths']}"
        ]
        
        if top_paths:
            profile_parts.append("Top archetypal paths:")
            for path_info in top_paths[:3]:  # Top 3 paths
                profile_parts.append(f"  - {path_info['path_string']} ({path_info['percentage']:.1f}%)")
        
        cluster_profiles[subtype] = "\n".join(profile_parts)
    
    # Generate cluster labels
    cluster_labels = analyzer.label_clusters_sync(cluster_profiles)
    
    logger.info("Step 2: Generating semantic organization narrative")
    
    # Build comprehensive prompt for semantic organization analysis
    semantic_analysis_prompt = f"""
You are analyzing the internal semantic organization of GPT-2 using Archetypal Path Analysis (APA). 

EXPERIMENTAL CONTEXT:
- Analyzed {analysis_summary['semantic_subtypes_analyzed']} semantic subtypes across {analysis_summary['total_layers']} GPT-2 layers
- Used {analysis_summary['clustering_method']} clustering on {len(within_subtype_metrics)} semantic categories
- Found {analysis_summary['total_paths']} total archetypal paths through the network

SEMANTIC SUBTYPES ANALYZED:
{json.dumps(semantic_organization_metrics['words_per_subtype'], indent=2)}

WITHIN-SUBTYPE ANALYSIS:
"""

    # Add within-subtype details
    for subtype, metrics in within_subtype_metrics.items():
        semantic_analysis_prompt += f"""
{subtype.upper()}:
- Path coherence: {metrics['path_coherence']:.3f} (higher = more consistent paths within subtype)
- Path diversity: {metrics['path_diversity']:.3f} (higher = more varied paths within subtype)  
- Words analyzed: {metrics['word_count']}
- Unique archetypal paths: {metrics['unique_paths']}
- Generated label: {cluster_labels.get(subtype, 'Unknown')}
"""
        
        top_paths = metrics.get('top_paths', [])
        if top_paths:
            semantic_analysis_prompt += "Top archetypal paths:\n"
            for path_info in top_paths[:3]:
                semantic_analysis_prompt += f"  - {path_info['path_string']} ({path_info['percentage']:.1f}% of {subtype} words)\n"
        semantic_analysis_prompt += "\n"

    # Add between-subtype analysis
    semantic_analysis_prompt += f"""
BETWEEN-SUBTYPE ANALYSIS:
- Total pairwise comparisons: {between_subtype_metrics.get('total_comparisons', 0)}
- Mean path sharing ratio: {between_subtype_metrics.get('mean_sharing_ratio', 0.0):.3f}
- Semantic differentiation: {between_subtype_metrics.get('semantic_differentiation', 0.0):.3f} (higher = more distinct subtypes)

PATH SHARING PATTERNS:
"""
    
    path_sharing_details = between_subtype_metrics.get('path_sharing_details', {})
    for comparison, sharing_data in list(path_sharing_details.items())[:5]:  # Top 5 comparisons
        semantic_analysis_prompt += f"- {comparison}: {sharing_data['sharing_ratio']:.3f} sharing ratio ({sharing_data['shared_paths']} shared paths)\n"

    # Add overall semantic clustering quality
    clustering_quality = semantic_organization_metrics.get('semantic_clustering_quality', {})
    if clustering_quality:
        semantic_analysis_prompt += f"""
OVERALL SEMANTIC CLUSTERING QUALITY:
- Mean coherence across subtypes: {clustering_quality.get('mean_coherence', 0.0):.3f}
- Coherence standard deviation: {clustering_quality.get('coherence_std', 0.0):.3f}
- Range: {clustering_quality.get('min_coherence', 0.0):.3f} to {clustering_quality.get('max_coherence', 0.0):.3f}
"""

    # Add analysis instructions
    semantic_analysis_prompt += """
ANALYSIS INSTRUCTIONS:
Please provide a comprehensive analysis of GPT-2's semantic organization based on this APA data. Focus on:

1. WITHIN-SUBTYPE COHERENCE: 
   - Which semantic subtypes show high path coherence (similar processing patterns)?
   - Which subtypes are more fragmented (diverse paths)?
   - What does this reveal about how GPT-2 processes different semantic categories?

2. BETWEEN-SUBTYPE DIFFERENTIATION:
   - Which semantic subtypes are most distinct from each other (low path sharing)?
   - Which subtypes share similar processing patterns (high path sharing)?
   - How does GPT-2 organize semantic knowledge hierarchically?

3. LAYER-WISE SEMANTIC PROCESSING:
   - What can we infer about semantic processing across GPT-2's layers?
   - Are there patterns in how semantic categories emerge or consolidate?

4. GPT-2 SEMANTIC ARCHITECTURE INSIGHTS:
   - What does this reveal about GPT-2's internal semantic representations?
   - How does this compare to human semantic organization?
   - What are the implications for transformer interpretability?

Provide specific examples from the data and explain the semantic significance of the patterns you observe.
"""

    # Generate semantic organization analysis
    logger.info("Generating semantic organization narrative...")
    
    semantic_response = analyzer.generate_with_cache(
        prompt=semantic_analysis_prompt,
        temperature=temperature,
        max_tokens=4000
    )
    
    logger.info("Step 3: Generating technical summary")
    
    # Generate technical summary prompt
    technical_summary_prompt = f"""
Based on the following APA metrics for GPT-2 semantic subtypes analysis, provide a concise technical summary:

METRICS SUMMARY:
- Clustering method: {analysis_summary['clustering_method']}
- Total layers analyzed: {analysis_summary['total_layers']}
- Semantic subtypes: {analysis_summary['semantic_subtypes_analyzed']}
- Total archetypal paths: {analysis_summary['total_paths']}
- Mean path coherence: {clustering_quality.get('mean_coherence', 0.0):.3f}
- Semantic differentiation: {between_subtype_metrics.get('semantic_differentiation', 0.0):.3f}

Please provide:
1. Key quantitative findings (2-3 sentences)
2. Main semantic organization patterns discovered (2-3 sentences)  
3. Technical implications for transformer interpretability (1-2 sentences)

Keep the summary concise and focused on the most significant findings.
"""

    technical_response = analyzer.generate_with_cache(
        prompt=technical_summary_prompt,
        temperature=temperature,
        max_tokens=1000
    )
    
    # Prepare comprehensive results
    llm_analysis_results = {
        'cluster_labels': cluster_labels,
        'semantic_organization_analysis': {
            'prompt': semantic_analysis_prompt,
            'response': semantic_response.content,
            'metadata': {
                'model': semantic_response.model,
                'provider': semantic_response.provider,
                'timestamp': semantic_response.timestamp,
                'tokens': semantic_response.tokens
            }
        },
        'technical_summary': {
            'prompt': technical_summary_prompt,
            'response': technical_response.content,
            'metadata': {
                'model': technical_response.model,
                'provider': technical_response.provider,
                'timestamp': technical_response.timestamp,
                'tokens': technical_response.tokens
            }
        },
        'analysis_statistics': {
            'total_tokens_used': semantic_response.tokens + technical_response.tokens,
            'semantic_subtypes_labeled': len(cluster_labels),
            'analysis_sections_generated': 2,
            'cache_hits': analyzer.get_cache_stats().get('hits', 0) if use_cache else 0
        }
    }
    
    logger.info("LLM-powered semantic subtypes analysis completed successfully")
    logger.info(f"Generated analysis for {len(cluster_labels)} semantic subtypes")
    logger.info(f"Total tokens used: {llm_analysis_results['analysis_statistics']['total_tokens_used']}")
    
    return llm_analysis_results


def create_semantic_subtypes_case_study_report(
    analysis_results: Dict[str, Any],
    llm_analysis_results: Dict[str, Any],
    output_dir: str = "results/case_studies",
    formats: List[str] = ["latex", "markdown"]
) -> Dict[str, str]:
    """
    Create comprehensive case study report for semantic subtypes experiment.
    
    Combines analysis results from all semantic subtypes functions to generate
    academic-quality case study reports following the GPT-2 case study format
    from the arxiv paper.
    
    Args:
        analysis_results: Output from calculate_semantic_subtypes_apa_metrics()
        llm_analysis_results: Output from generate_semantic_subtypes_llm_analysis()
        output_dir: Directory for output files
        formats: List of formats to generate ["latex", "markdown", "both"]
        
    Returns:
        Dictionary mapping format names to generated file paths
    """
    import sys
    from pathlib import Path
    
    # Add paths for required modules
    sys.path.insert(0, str(Path(__file__).parent))
    
    logger.info("Creating semantic subtypes case study report")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Formats: {formats}")
    
    try:
        from gpt2_semantic_subtypes_case_study_report import GPT2SemanticSubtypesCaseStudyReporter
    except ImportError:
        raise ImportError("Could not import GPT2SemanticSubtypesCaseStudyReporter. Make sure the module is available.")
    
    # Initialize reporter
    reporter = GPT2SemanticSubtypesCaseStudyReporter(output_dir=output_dir)
    
    # Prepare combined results for the reporter
    combined_results = {
        'semantic_subtypes_results': analysis_results['within_subtype_metrics'],
        'cross_subtype_analysis': analysis_results['between_subtype_metrics'],
        'overall_metrics': {
            'total_semantic_subtypes': analysis_results['semantic_organization_metrics']['total_semantic_subtypes'],
            'total_words_analyzed': analysis_results['semantic_organization_metrics']['total_words_analyzed'],
            'analysis_timestamp': analysis_results['analysis_summary']['analysis_timestamp'],
            'clustering_method': analysis_results['analysis_summary']['clustering_method']
        },
        'cluster_labels': llm_analysis_results['cluster_labels'],
        'llm_analysis': {
            'semantic_organization_narrative': llm_analysis_results['semantic_organization_analysis']['response'],
            'technical_summary': llm_analysis_results['technical_summary']['response']
        }
    }
    
    # Add semantic clustering quality metrics if available
    clustering_quality = analysis_results.get('semantic_organization_metrics', {}).get('semantic_clustering_quality', {})
    if clustering_quality:
        combined_results['overall_metrics'].update({
            'average_silhouette_score': clustering_quality.get('mean_coherence', 0.0),
            'within_subtype_coherence': clustering_quality.get('mean_coherence', 0.0),
            'cross_subtype_differentiation': analysis_results['between_subtype_metrics'].get('semantic_differentiation', 0.0)
        })
    
    # Load results into reporter
    logger.info("Loading analysis results into case study reporter...")
    
    # Manually populate reporter data structures
    reporter.overall_metrics = combined_results['overall_metrics']
    reporter.cluster_labels = combined_results['cluster_labels']
    reporter.llm_narratives = combined_results['llm_analysis']
    
    # Extract semantic subtype results
    reporter._extract_subtype_results(combined_results['semantic_subtypes_results'])
    
    # Extract cross-subtype comparisons from between_subtype_metrics
    between_subtype_data = combined_results['cross_subtype_analysis']
    path_sharing_details = between_subtype_data.get('path_sharing_details', {})
    
    for comparison_key, sharing_data in path_sharing_details.items():
        # Parse comparison key to extract subtype pair
        if '_vs_' in comparison_key:
            subtypes = comparison_key.split('_vs_')
        elif ' vs ' in comparison_key:
            subtypes = comparison_key.split(' vs ')
        else:
            # Try to infer from other patterns
            subtypes = comparison_key.replace('_', ' ').split(' vs ')
            
        if len(subtypes) == 2:
            from gpt2_semantic_subtypes_case_study_report import CrossSubtypeComparison
            
            reporter.cross_subtype_comparisons.append(
                CrossSubtypeComparison(
                    subtype_pair=(subtypes[0].strip(), subtypes[1].strip()),
                    path_overlap_ratio=sharing_data.get('sharing_ratio', 0.0),
                    differentiation_score=1.0 - sharing_data.get('sharing_ratio', 0.0),
                    divergence_layers=[],  # TODO: Extract from detailed analysis if available
                    shared_cluster_patterns=[]  # TODO: Extract from detailed analysis if available
                )
            )
    
    # Generate reports in requested formats
    generated_files = {}
    
    if "latex" in formats or "both" in formats:
        logger.info("Generating LaTeX case study report...")
        latex_content = reporter.generate_latex_report()
        # Find the generated file path
        latex_files = list(Path(output_dir).glob("gpt2_semantic_subtypes_case_study_*.tex"))
        if latex_files:
            generated_files["latex"] = str(latex_files[-1])  # Most recent
            logger.info(f"Generated LaTeX report: {generated_files['latex']}")
    
    if "markdown" in formats or "both" in formats:
        logger.info("Generating Markdown case study report...")
        markdown_content = reporter.generate_markdown_report()
        # Find the generated file path
        markdown_files = list(Path(output_dir).glob("gpt2_semantic_subtypes_case_study_*.md"))
        if markdown_files:
            generated_files["markdown"] = str(markdown_files[-1])  # Most recent
            logger.info(f"Generated Markdown report: {generated_files['markdown']}")
    
    # Generate summary statistics
    stats = reporter.generate_summary_statistics()
    logger.info("Case study summary statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Semantic subtypes case study report creation completed successfully")
    logger.info(f"Generated {len(generated_files)} report files")
    
    return generated_files


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