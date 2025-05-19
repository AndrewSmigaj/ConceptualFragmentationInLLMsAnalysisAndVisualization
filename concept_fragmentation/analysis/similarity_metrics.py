"""
Similarity metrics for cluster analysis across neural network layers.

This module provides functions to compute, analyze, and visualize similarities
between clusters across different layers of neural networks.
"""

import numpy as np
import warnings
from scipy.spatial.distance import cosine, euclidean
from typing import Dict, List, Tuple, Any, Optional, Union


def compute_centroid_similarity(
    layer_clusters: Dict[str, Dict[str, Any]],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]],
    metric: str = 'cosine',
    min_similarity: float = 0.0,
    same_layer: bool = False
) -> Dict[Tuple[int, int], float]:
    """
    Compute similarity between cluster centroids across all layers.
    
    Args:
        layer_clusters: Dictionary of cluster information per layer
        id_to_layer_cluster: Mapping from unique ID to layer information
        metric: Similarity metric to use ('cosine' or 'euclidean')
        min_similarity: Minimum similarity threshold to include in results
        same_layer: Whether to compute similarity between clusters in the same layer
        
    Returns:
        Dictionary mapping (cluster1_id, cluster2_id) to similarity score
    """
    similarity_matrix = {}
    
    # Extract centroids for each cluster
    centers_by_id = {}
    
    # First try to get pre-computed centers
    for unique_id, (layer_name, original_id, _) in id_to_layer_cluster.items():
        if "centers" in layer_clusters[layer_name]:
            centers = layer_clusters[layer_name]["centers"]
            if original_id < len(centers):
                centers_by_id[unique_id] = centers[original_id]
    
    # If we don't have centroids, try to compute them from the activations
    for unique_id, (layer_name, original_id, _) in id_to_layer_cluster.items():
        if unique_id not in centers_by_id:
            if "activations" in layer_clusters[layer_name] and "labels" in layer_clusters[layer_name]:
                activations = layer_clusters[layer_name]["activations"]
                labels = layer_clusters[layer_name]["labels"]
                mask = (labels == original_id)
                if np.sum(mask) > 0:
                    centers_by_id[unique_id] = np.mean(activations[mask], axis=0)
    
    # Compute similarity between all pairs of centroids across different layers
    cluster_ids = list(centers_by_id.keys())
    
    for i, id1 in enumerate(cluster_ids):
        for j, id2 in enumerate(cluster_ids):
            # Skip if it's the same cluster
            if id1 == id2:
                continue
            
            # Skip if we've already computed this pair
            if (id1, id2) in similarity_matrix or (id2, id1) in similarity_matrix:
                continue
            
            # Skip same-layer comparisons if requested
            if not same_layer:
                _, _, layer_idx1 = id_to_layer_cluster[id1]
                _, _, layer_idx2 = id_to_layer_cluster[id2]
                if layer_idx1 == layer_idx2:
                    continue
            
            # Get centroids
            c1 = centers_by_id[id1]
            c2 = centers_by_id[id2]
            
            # Skip if dimensions don't match (this can happen with different layer sizes)
            if c1.shape != c2.shape:
                continue
            
            # Compute similarity based on selected metric
            if metric == 'cosine':
                # Handle zero vectors
                if np.all(c1 == 0) or np.all(c2 == 0):
                    sim = 0.0
                else:
                    # Calculate cosine similarity directly
                    dot_product = np.dot(c1, c2)
                    norm1 = np.linalg.norm(c1)
                    norm2 = np.linalg.norm(c2)
                    
                    # Avoid division by zero
                    if norm1 == 0 or norm2 == 0:
                        sim = 0.0
                    else:
                        sim = dot_product / (norm1 * norm2)
                        
                        # Numerical precision issues can result in |sim| > 1
                        if sim > 1.0:
                            sim = 1.0
                        elif sim < -1.0:
                            sim = -1.0
            elif metric == 'euclidean':
                # Convert distance to similarity using Gaussian kernel
                dist = np.linalg.norm(c1 - c2)
                
                # Use adaptive sigma based on dimensionality
                sigma = np.sqrt(c1.shape[0])
                
                sim = np.exp(-dist**2 / (2 * sigma**2))
            else:
                raise ValueError(f"Unsupported similarity metric: {metric}")
            
            # Only include similarities above threshold
            if sim >= min_similarity:
                similarity_matrix[(id1, id2)] = float(sim)
                similarity_matrix[(id2, id1)] = float(sim)  # Ensure symmetry
    
    return similarity_matrix


def normalize_similarity_matrix(
    similarity_matrix: Dict[Tuple[int, int], float]
) -> Dict[Tuple[int, int], float]:
    """
    Normalize similarity scores to [0, 1] range.
    
    Args:
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        
    Returns:
        Normalized similarity matrix
    """
    if not similarity_matrix:
        return {}
        
    # Find min and max similarity
    min_sim = min(similarity_matrix.values())
    max_sim = max(similarity_matrix.values())
    
    # Handle case where all similarities are the same
    if min_sim == max_sim:
        return {pair: 1.0 for pair in similarity_matrix}
    
    # Normalize to [0, 1] range
    normalized = {}
    for pair, sim in similarity_matrix.items():
        normalized[pair] = (sim - min_sim) / (max_sim - min_sim)
    
    return normalized


def compute_layer_similarity_matrix(
    similarity_matrix: Dict[Tuple[int, int], float],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]]
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Compute aggregated similarity between layers.
    
    Args:
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        id_to_layer_cluster: Mapping from unique ID to layer information
        
    Returns:
        Dictionary mapping (layer_idx1, layer_idx2) to statistics about similarity
    """
    # Group similarities by layer pair
    layer_pairs = {}
    for (id1, id2), sim in similarity_matrix.items():
        if id1 in id_to_layer_cluster and id2 in id_to_layer_cluster:
            _, _, layer_idx1 = id_to_layer_cluster[id1]
            _, _, layer_idx2 = id_to_layer_cluster[id2]
            
            # Ensure consistent ordering (smaller index first)
            layer_pair = (min(layer_idx1, layer_idx2), max(layer_idx1, layer_idx2))
            if layer_pair not in layer_pairs:
                layer_pairs[layer_pair] = []
            layer_pairs[layer_pair].append(sim)
    
    # Compute statistics for each layer pair
    layer_similarity = {}
    for layer_pair, similarities in layer_pairs.items():
        if not similarities:
            continue
            
        layer_similarity[layer_pair] = {
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "max": float(np.max(similarities)),
            "min": float(np.min(similarities)),
            "std": float(np.std(similarities)),
            "count": len(similarities)
        }
    
    return layer_similarity


def get_top_similar_clusters(
    similarity_matrix: Dict[Tuple[int, int], float],
    id_to_layer_cluster: Dict[int, Tuple[str, int, int]],
    layer_idx: Optional[int] = None,
    top_k: int = 5,
    min_similarity: float = 0.5
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Get top-k most similar clusters for each cluster.
    
    Args:
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        id_to_layer_cluster: Mapping from unique ID to layer information
        layer_idx: If provided, only consider clusters in this layer
        top_k: Number of top similar clusters to return
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dictionary mapping cluster_id to list of (similar_cluster_id, similarity)
    """
    # Group similarities by cluster
    cluster_similarities = {}
    for cluster_id in id_to_layer_cluster:
        # Skip if not in specified layer
        if layer_idx is not None:
            _, _, cluster_layer_idx = id_to_layer_cluster[cluster_id]
            if cluster_layer_idx != layer_idx:
                continue
        
        similarities = []
        for other_id in id_to_layer_cluster:
            # Skip same cluster
            if cluster_id == other_id:
                continue
                
            # Skip if not in similarity matrix
            if (cluster_id, other_id) not in similarity_matrix:
                continue
                
            sim = similarity_matrix[(cluster_id, other_id)]
            if sim >= min_similarity:
                similarities.append((other_id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top-k
        cluster_similarities[cluster_id] = similarities[:top_k]
    
    return cluster_similarities


def serialize_similarity_matrix(
    similarity_matrix: Dict[Tuple[int, int], float]
) -> Dict[str, float]:
    """
    Convert similarity matrix to a JSON-serializable format.
    
    Args:
        similarity_matrix: Dictionary mapping (cluster1_id, cluster2_id) to similarity score
        
    Returns:
        Dictionary with string keys "id1,id2" mapping to similarity scores
    """
    serialized = {}
    for (id1, id2), sim in similarity_matrix.items():
        key = f"{id1},{id2}"
        serialized[key] = float(sim)  # Ensure float for JSON compatibility
    return serialized


def deserialize_similarity_matrix(
    serialized_matrix: Dict[str, float]
) -> Dict[Tuple[int, int], float]:
    """
    Convert serialized similarity matrix back to tuple-keyed dictionary.
    
    Args:
        serialized_matrix: Dictionary with string keys "id1,id2" mapping to similarity scores
        
    Returns:
        Dictionary mapping (cluster1_id, cluster2_id) to similarity score
    """
    similarity_matrix = {}
    for key, sim in serialized_matrix.items():
        if "," in key:
            try:
                id1, id2 = map(int, key.split(","))
                similarity_matrix[(id1, id2)] = float(sim)
            except (ValueError, TypeError):
                warnings.warn(f"Invalid key in serialized similarity matrix: {key}")
    return similarity_matrix