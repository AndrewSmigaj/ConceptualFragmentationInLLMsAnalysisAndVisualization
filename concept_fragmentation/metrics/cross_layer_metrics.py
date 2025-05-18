"""
Cross-Layer Metrics for Concept Fragmentation analysis.

This module implements various metrics that analyze the relationships
between clusters across different layers of a neural network.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import scipy.spatial.distance as distance
from sklearn.metrics import pairwise_distances
import networkx as nx

def compute_centroid_similarity(
    layer_clusters: Dict[str, Dict[str, Any]],
    similarity_metric: str = "cosine"
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute similarity between cluster centroids across different layers.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries,
                       Each containing 'centers' key with centroid coordinates
        similarity_metric: Metric to use for similarity calculation
                          (e.g., 'cosine', 'euclidean', 'correlation')
    
    Returns:
        Dictionary mapping layer pairs to similarity matrices between their centroids
    """
    similarity_matrices = {}
    
    # Get list of layers with valid cluster centers
    valid_layers = []
    for layer_name, layer_info in layer_clusters.items():
        if "centers" in layer_info and layer_info["centers"] is not None:
            if len(layer_info["centers"]) > 0:
                valid_layers.append(layer_name)
    
    # Compute similarity between each pair of layers
    for i, layer1 in enumerate(valid_layers):
        centers1 = layer_clusters[layer1]["centers"]
        
        for j, layer2 in enumerate(valid_layers):
            if i >= j:  # Only compute for i < j (upper triangle)
                continue
                
            centers2 = layer_clusters[layer2]["centers"]
            
            # If dimensions don't match, skip this pair
            if centers1.shape[1] != centers2.shape[1]:
                continue
            
            # Compute similarity matrix between centroids
            if similarity_metric == "cosine":
                # Convert to similarity (1 - distance)
                sim_matrix = 1 - pairwise_distances(centers1, centers2, metric="cosine")
            elif similarity_metric == "correlation":
                # Convert to similarity (1 - distance)
                sim_matrix = 1 - pairwise_distances(centers1, centers2, metric="correlation")
            elif similarity_metric == "euclidean":
                # Compute Euclidean distance
                dist_matrix = pairwise_distances(centers1, centers2, metric="euclidean")
                # Convert to similarity using exp(-dist)
                sim_matrix = np.exp(-dist_matrix)
            else:
                # Default to cosine similarity
                sim_matrix = 1 - pairwise_distances(centers1, centers2, metric="cosine")
            
            # Store in dictionary
            similarity_matrices[(layer1, layer2)] = sim_matrix
    
    return similarity_matrices

def compute_membership_overlap(
    layer_clusters: Dict[str, Dict[str, Any]],
    normalize: bool = True
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute overlap in cluster membership across different layers.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries,
                       Each containing 'labels' key with cluster assignments
        normalize: Whether to normalize overlap by cluster sizes
    
    Returns:
        Dictionary mapping layer pairs to overlap matrices between their clusters
    """
    overlap_matrices = {}
    
    # Get list of layers with valid cluster labels
    valid_layers = []
    for layer_name, layer_info in layer_clusters.items():
        if "labels" in layer_info and layer_info["labels"] is not None:
            if len(layer_info["labels"]) > 0:
                valid_layers.append(layer_name)
    
    # Compute overlap between each pair of layers
    for i, layer1 in enumerate(valid_layers):
        labels1 = layer_clusters[layer1]["labels"]
        unique_clusters1 = np.unique(labels1)
        n_clusters1 = len(unique_clusters1)
        
        for j, layer2 in enumerate(valid_layers):
            if i >= j:  # Only compute for i < j (upper triangle)
                continue
                
            labels2 = layer_clusters[layer2]["labels"]
            
            # If labels arrays have different lengths, skip this pair
            if len(labels1) != len(labels2):
                continue
                
            unique_clusters2 = np.unique(labels2)
            n_clusters2 = len(unique_clusters2)
            
            # Initialize overlap matrix
            overlap = np.zeros((n_clusters1, n_clusters2))
            
            # Compute contingency matrix
            for idx1, c1 in enumerate(unique_clusters1):
                mask1 = (labels1 == c1)
                size1 = np.sum(mask1)
                
                for idx2, c2 in enumerate(unique_clusters2):
                    mask2 = (labels2 == c2)
                    size2 = np.sum(mask2)
                    
                    # Count samples in both clusters
                    overlap_count = np.sum(mask1 & mask2)
                    
                    if normalize:
                        # Normalize by minimum cluster size
                        denominator = min(size1, size2)
                        if denominator > 0:
                            overlap[idx1, idx2] = overlap_count / denominator
                    else:
                        overlap[idx1, idx2] = overlap_count
            
            # Store in dictionary
            overlap_matrices[(layer1, layer2)] = overlap
    
    return overlap_matrices

def compute_trajectory_fragmentation(
    layer_clusters: Dict[str, Dict[str, Any]],
    class_labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute trajectory fragmentation score for each layer.
    
    This measures how much samples from the same class are split across different clusters.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries,
                       Each containing 'labels' key with cluster assignments
        class_labels: Ground truth class labels for samples
    
    Returns:
        Dictionary mapping layer names to fragmentation scores
    """
    fragmentation_scores = {}
    
    # Get unique classes
    unique_classes = np.unique(class_labels)
    
    for layer_name, layer_info in layer_clusters.items():
        if "labels" not in layer_info or layer_info["labels"] is None:
            continue
            
        cluster_labels = layer_info["labels"]
        
        # If labels and class_labels have different lengths, skip this layer
        if len(cluster_labels) != len(class_labels):
            continue
        
        # Calculate entropy-based fragmentation for each class
        class_fragmentations = []
        
        for class_id in unique_classes:
            # Get indices of samples in this class
            class_mask = (class_labels == class_id)
            class_clusters = cluster_labels[class_mask]
            
            # Count samples in each cluster
            unique_clusters, counts = np.unique(class_clusters, return_counts=True)
            
            # Skip if no samples in this class
            if len(counts) == 0:
                continue
                
            # Calculate proportions
            proportions = counts / np.sum(counts)
            
            # Calculate entropy
            entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
            
            # Normalize by log(k) where k is number of clusters
            max_entropy = np.log2(len(unique_clusters)) if len(unique_clusters) > 1 else 1
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0
                
            class_fragmentations.append(normalized_entropy)
        
        # Average fragmentation across classes
        if class_fragmentations:
            fragmentation_scores[layer_name] = np.mean(class_fragmentations)
    
    return fragmentation_scores

def compute_path_density(
    layer_clusters: Dict[str, Dict[str, Any]],
    min_overlap: float = 0.1
) -> Tuple[Dict[Tuple[str, str], float], nx.Graph]:
    """
    Compute inter-cluster path density between adjacent layers.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries
        min_overlap: Minimum overlap required to consider clusters connected
    
    Returns:
        Tuple of (density_scores, path_graph) where:
        - density_scores maps layer pairs to density scores
        - path_graph is a NetworkX graph representing the cluster connectivity
    """
    # Compute membership overlap matrices
    overlap_matrices = compute_membership_overlap(layer_clusters, normalize=True)
    
    # Build graph of cluster connections
    G = nx.Graph()
    
    # Sort layer names to ensure proper ordering
    layer_names = sorted(layer_clusters.keys())
    
    # Add nodes for each cluster in each layer
    for layer_name in layer_names:
        if "labels" not in layer_clusters[layer_name]:
            continue
            
        unique_clusters = np.unique(layer_clusters[layer_name]["labels"])
        
        for cluster_id in unique_clusters:
            node_id = f"{layer_name}_cluster{cluster_id}"
            G.add_node(node_id, layer=layer_name, cluster_id=cluster_id)
    
    # Add edges between clusters in adjacent layers based on overlap
    density_scores = {}
    
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        # Skip if this layer pair isn't in overlap matrices
        if (layer1, layer2) not in overlap_matrices:
            continue
            
        overlap = overlap_matrices[(layer1, layer2)]
        
        # Count connections above threshold
        connections = 0
        potential_connections = 0
        
        for c1_idx, c1 in enumerate(np.unique(layer_clusters[layer1]["labels"])):
            for c2_idx, c2 in enumerate(np.unique(layer_clusters[layer2]["labels"])):
                potential_connections += 1
                
                if c1_idx < overlap.shape[0] and c2_idx < overlap.shape[1]:
                    overlap_value = overlap[c1_idx, c2_idx]
                    
                    if overlap_value >= min_overlap:
                        connections += 1
                        
                        # Add edge to graph
                        node1 = f"{layer1}_cluster{c1}"
                        node2 = f"{layer2}_cluster{c2}"
                        G.add_edge(node1, node2, weight=overlap_value)
        
        # Calculate density
        if potential_connections > 0:
            density = connections / potential_connections
        else:
            density = 0
            
        density_scores[(layer1, layer2)] = density
    
    return density_scores, G

def analyze_cross_layer_metrics(
    layer_clusters: Dict[str, Dict[str, Any]], 
    activations: Dict[str, np.ndarray] = None,
    class_labels: Optional[np.ndarray] = None,
    config: Dict = None
) -> Dict:
    """
    Analyze cross-layer relationships using multiple metrics.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster info dictionaries
        activations: Dictionary mapping layer names to activation matrices (optional)
        class_labels: Ground truth class labels for samples (optional)
        config: Configuration dictionary with parameters for each metric
    
    Returns:
        Dictionary with results for each metric
    """
    # Initialize default config
    default_config = {
        "similarity_metric": "cosine",
        "min_overlap": 0.1,
    }
    
    # Update with user config
    if config:
        default_config.update(config)
    
    # Initialize results dictionary
    results = {}
    
    # Compute centroid similarity
    try:
        results["centroid_similarity"] = compute_centroid_similarity(
            layer_clusters, 
            similarity_metric=default_config["similarity_metric"]
        )
    except Exception as e:
        results["centroid_similarity_error"] = str(e)
    
    # Compute membership overlap
    try:
        results["membership_overlap"] = compute_membership_overlap(
            layer_clusters, 
            normalize=True
        )
    except Exception as e:
        results["membership_overlap_error"] = str(e)
    
    # Compute trajectory fragmentation if class labels provided
    if class_labels is not None:
        try:
            results["trajectory_fragmentation"] = compute_trajectory_fragmentation(
                layer_clusters, 
                class_labels
            )
        except Exception as e:
            results["trajectory_fragmentation_error"] = str(e)
    
    # Compute path density
    try:
        density_scores, path_graph = compute_path_density(
            layer_clusters, 
            min_overlap=default_config["min_overlap"]
        )
        results["path_density"] = density_scores
        results["path_graph"] = path_graph
    except Exception as e:
        results["path_density_error"] = str(e)
    
    return results