"""
Cross-Layer Path Analysis Metrics for Concept Fragmentation analysis.

This module implements the cross-layer metrics described in the paper
"Foundations of Archetypal Path Analysis: Toward a Principled Geometry 
for Cluster-Based Interpretability".

The metrics include:
1. Centroid Similarity (ρᶜ): Measures similarity between cluster centroids across layers
2. Membership Overlap (J): Quantifies how datapoints from one cluster are distributed in another layer
3. Trajectory Fragmentation Score (F): Measures how coherent or dispersed a datapoint's path is
4. Inter-Cluster Path Density (ICPD): Analyzes higher-order patterns in concept flow
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Set
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.stats import entropy
from collections import Counter
import warnings

def validate_layer_order(
    layer_names: List[str]
) -> List[str]:
    """
    Validate and sort layer names in the correct order.
    
    Args:
        layer_names: List of layer names to validate
        
    Returns:
        Sorted list of layer names
    
    Raises:
        ValueError: If layer names do not follow a consistent pattern
    """
    # Try to extract numeric part from layer names
    numeric_parts = []
    for name in layer_names:
        if name.startswith("layer"):
            try:
                num = int(name[5:])  # Extract number after "layer"
                numeric_parts.append((num, name))
            except ValueError:
                pass
    
    # If we successfully extracted numbers, sort by them
    if len(numeric_parts) == len(layer_names):
        return [name for _, name in sorted(numeric_parts)]
    
    # If we have standard layer names like "input", "hidden1", "output"
    standard_layer_order = {
        "input": 0,
        "output": 999  # Place at the end
    }
    
    # Try to sort by standard names and any numeric parts
    try:
        def layer_sort_key(name):
            # Standard layer names
            if name in standard_layer_order:
                return standard_layer_order[name]
            
            # Look for hidden layers with numbers
            if name.startswith("hidden"):
                try:
                    return int(name[6:]) + 100  # Add offset to place after input
                except ValueError:
                    pass
            
            # Try to extract any number in the name
            for i, char in enumerate(name):
                if char.isdigit():
                    try:
                        return int(name[i:])
                    except ValueError:
                        pass
            
            # Default to alphabetical
            return name
        
        return sorted(layer_names, key=layer_sort_key)
    except:
        # If all else fails, just sort alphabetically
        warnings.warn("Could not determine layer order, sorting alphabetically")
        return sorted(layer_names)

def project_to_common_space(
    activations: Dict[str, np.ndarray],
    projection_method: str = 'pca',
    projection_dims: int = 10
) -> Dict[str, np.ndarray]:
    """
    Project activations from different layers to a common space.
    
    Args:
        activations: Dictionary mapping layer names to activation matrices
        projection_method: Method to use for projection ('pca', 'none')
        projection_dims: Number of dimensions to project to
        
    Returns:
        Dictionary mapping layer names to projected activations
    """
    if projection_method.lower() == 'none':
        return activations
    
    projected_activations = {}
    
    if projection_method.lower() == 'pca':
        # For each layer, apply PCA to reduce dimensions
        for layer_name, layer_activations in activations.items():
            n_samples, n_features = layer_activations.shape
            
            # If dimensionality is already low, don't reduce
            if n_features <= projection_dims:
                projected_activations[layer_name] = layer_activations
                continue
            
            # Apply PCA
            pca = PCA(n_components=projection_dims)
            projected_activations[layer_name] = pca.fit_transform(layer_activations)
    else:
        raise ValueError(f"Unsupported projection method: {projection_method}")
    
    return projected_activations

def compute_centroids(
    activations: np.ndarray,
    cluster_labels: np.ndarray
) -> np.ndarray:
    """
    Compute centroids for each cluster.
    
    Args:
        activations: Activation matrix (n_samples, n_features)
        cluster_labels: Cluster assignments (n_samples,)
        
    Returns:
        Array of centroids (n_clusters, n_features)
    """
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    n_features = activations.shape[1]
    
    centroids = np.zeros((n_clusters, n_features))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = (cluster_labels == cluster_id)
        
        # Avoid empty clusters
        if np.sum(mask) > 0:
            centroids[i] = np.mean(activations[mask], axis=0)
    
    return centroids, unique_clusters

def compute_centroid_similarity(
    layer_clusters: Dict[str, np.ndarray],
    activations: Dict[str, np.ndarray],
    similarity_metric: str = 'cosine',
    projection_method: str = 'pca',
    projection_dims: int = 10,
    batch_size: Optional[int] = None
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute centroid similarity between clusters across different layers.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        activations: Dictionary mapping layer names to activation matrices
        similarity_metric: Metric to use for similarity ('cosine', 'euclidean')
        projection_method: Method for projecting to common space ('pca', 'none')
        projection_dims: Dimensions for projection
        batch_size: Optional batch size for large datasets
        
    Returns:
        Dictionary mapping layer pairs to similarity matrices
    """
    # Validate inputs
    if len(layer_clusters) != len(activations):
        raise ValueError("layer_clusters and activations must have the same layers")
    
    # Validate and sort layer names
    layer_names = validate_layer_order(list(layer_clusters.keys()))
    
    # Project activations to common space if needed
    if projection_method.lower() != 'none':
        projected_activations = project_to_common_space(
            activations, projection_method, projection_dims
        )
    else:
        projected_activations = activations
    
    # Compute centroids for each layer
    layer_centroids = {}
    layer_cluster_ids = {}
    
    for layer_name in layer_names:
        centroids, cluster_ids = compute_centroids(
            projected_activations[layer_name],
            layer_clusters[layer_name]
        )
        layer_centroids[layer_name] = centroids
        layer_cluster_ids[layer_name] = cluster_ids
    
    # Compute similarity matrices for all layer pairs
    similarities = {}
    
    for i, layer1 in enumerate(layer_names):
        for j, layer2 in enumerate(layer_names):
            if i == j:
                continue  # Skip self-comparisons
            
            centroids1 = layer_centroids[layer1]
            centroids2 = layer_centroids[layer2]
            cluster_ids1 = layer_cluster_ids[layer1]
            cluster_ids2 = layer_cluster_ids[layer2]
            
            # Compute similarity based on selected metric
            if similarity_metric == 'cosine':
                # Compute pairwise cosine similarities
                similarity_matrix = np.zeros((len(cluster_ids1), len(cluster_ids2)))
                
                # Process in batches if specified
                if batch_size is not None and len(cluster_ids1) * len(cluster_ids2) > batch_size:
                    # Batch processing for large clusters
                    for b1 in range(0, len(cluster_ids1), batch_size):
                        b1_end = min(b1 + batch_size, len(cluster_ids1))
                        for b2 in range(0, len(cluster_ids2), batch_size):
                            b2_end = min(b2 + batch_size, len(cluster_ids2))
                            
                            for i1 in range(b1, b1_end):
                                for i2 in range(b2, b2_end):
                                    if np.any(np.isnan(centroids1[i1])) or np.any(np.isnan(centroids2[i2])):
                                        similarity_matrix[i1, i2] = 0
                                    else:
                                        # 1 - cosine distance = cosine similarity
                                        similarity_matrix[i1, i2] = 1 - cosine(centroids1[i1], centroids2[i2])
                else:
                    # Vectorized computation for small clusters
                    for i1 in range(len(cluster_ids1)):
                        for i2 in range(len(cluster_ids2)):
                            if np.any(np.isnan(centroids1[i1])) or np.any(np.isnan(centroids2[i2])):
                                similarity_matrix[i1, i2] = 0
                            else:
                                # 1 - cosine distance = cosine similarity
                                similarity_matrix[i1, i2] = 1 - cosine(centroids1[i1], centroids2[i2])
            
            elif similarity_metric == 'euclidean':
                # Compute normalized Euclidean distances
                distances = pairwise_distances(centroids1, centroids2, metric='euclidean')
                
                # Convert distances to similarities (1 / (1 + distance))
                similarity_matrix = 1 / (1 + distances)
            
            else:
                raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
            
            # Store similarity matrix
            similarities[(layer1, layer2)] = {
                'matrix': similarity_matrix,
                'source_clusters': cluster_ids1,
                'target_clusters': cluster_ids2
            }
    
    return similarities

def compute_membership_overlap(
    layer_clusters: Dict[str, np.ndarray],
    overlap_type: str = 'jaccard',
    batch_size: Optional[int] = None
) -> Dict[Tuple[str, str], Dict]:
    """
    Compute membership overlap between clusters in different layers.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        overlap_type: Type of overlap measure ('jaccard' or 'containment')
        batch_size: Optional batch size for large datasets
        
    Returns:
        Dictionary mapping layer pairs to overlap matrices
    """
    # Validate and sort layer names
    layer_names = validate_layer_order(list(layer_clusters.keys()))
    
    # Compute overlap matrices for all layer pairs
    overlaps = {}
    
    for i, layer1 in enumerate(layer_names):
        for j, layer2 in enumerate(layer_names):
            if i == j:
                continue  # Skip self-comparisons
            
            labels1 = layer_clusters[layer1]
            labels2 = layer_clusters[layer2]
            
            # Get unique cluster IDs
            cluster_ids1 = np.unique(labels1)
            cluster_ids2 = np.unique(labels2)
            
            # Initialize overlap matrix
            overlap_matrix = np.zeros((len(cluster_ids1), len(cluster_ids2)))
            
            # Compute sample indices for each cluster
            cluster_samples1 = {
                cluster_id: set(np.where(labels1 == cluster_id)[0])
                for cluster_id in cluster_ids1
            }
            
            cluster_samples2 = {
                cluster_id: set(np.where(labels2 == cluster_id)[0])
                for cluster_id in cluster_ids2
            }
            
            # Compute overlaps
            for i1, c1 in enumerate(cluster_ids1):
                samples1 = cluster_samples1[c1]
                
                for i2, c2 in enumerate(cluster_ids2):
                    samples2 = cluster_samples2[c2]
                    
                    # Compute intersection and union
                    intersection = len(samples1.intersection(samples2))
                    
                    if overlap_type == 'jaccard':
                        # Jaccard similarity: |A ∩ B| / |A ∪ B|
                        union = len(samples1.union(samples2))
                        if union > 0:
                            overlap_matrix[i1, i2] = intersection / union
                    
                    elif overlap_type == 'containment':
                        # Containment: |A ∩ B| / |A|
                        if len(samples1) > 0:
                            overlap_matrix[i1, i2] = intersection / len(samples1)
                    
                    else:
                        raise ValueError(f"Unsupported overlap type: {overlap_type}")
            
            # Store overlap matrix
            overlaps[(layer1, layer2)] = {
                'matrix': overlap_matrix,
                'source_clusters': cluster_ids1,
                'target_clusters': cluster_ids2,
                'type': overlap_type
            }
    
    return overlaps

def extract_paths(
    layer_clusters: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Extract paths from layer clusters.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        
    Returns:
        Array of paths (n_samples, n_layers)
    """
    # Validate and sort layer names
    layer_names = validate_layer_order(list(layer_clusters.keys()))
    
    # Get number of samples (assuming all layers have the same samples)
    n_samples = len(layer_clusters[layer_names[0]])
    n_layers = len(layer_names)
    
    # Create paths array
    paths = np.zeros((n_samples, n_layers), dtype=int)
    
    # Fill in paths
    for i, layer in enumerate(layer_names):
        paths[:, i] = layer_clusters[layer]
    
    return paths, layer_names

def compute_trajectory_fragmentation(
    paths: np.ndarray,
    layer_names: List[str],
    labels: Optional[np.ndarray] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Compute trajectory fragmentation scores.
    
    Args:
        paths: Array of paths (n_samples, n_layers)
        layer_names: Names of layers in order
        labels: Optional class labels for class-specific fragmentation
        
    Returns:
        Dictionary of fragmentation metrics
    """
    # Validate layer ordering
    layer_names = validate_layer_order(layer_names)
    
    n_samples, n_layers = paths.shape
    
    # Convert paths to tuples for counting
    path_tuples = [tuple(path) for path in paths]
    
    # Count occurrences of each unique path
    path_counts = Counter(path_tuples)
    
    # Compute overall entropy of paths
    path_probs = np.array([count / n_samples for count in path_counts.values()])
    overall_entropy = -np.sum(path_probs * np.log2(path_probs + 1e-10))
    
    # Normalize by maximum possible entropy (log2 of number of unique paths)
    max_entropy = np.log2(min(n_samples, len(path_counts)))
    normalized_entropy = overall_entropy / max_entropy if max_entropy > 0 else 0
    
    # Compute transition entropies between consecutive layers
    transition_entropies = []
    
    for i in range(n_layers - 1):
        # Get clusters in consecutive layers
        source_clusters = paths[:, i]
        target_clusters = paths[:, i+1]
        
        # Count transitions
        transitions = {}
        for src, tgt in zip(source_clusters, target_clusters):
            if src not in transitions:
                transitions[src] = Counter()
            transitions[src][tgt] += 1
        
        # Compute entropy for each source cluster
        cluster_entropies = []
        for src, counts in transitions.items():
            total = sum(counts.values())
            probs = np.array([count / total for count in counts.values()])
            
            if len(probs) > 0:
                ent = -np.sum(probs * np.log2(probs + 1e-10))
                cluster_entropies.append(ent)
        
        # Average entropy for this transition
        if cluster_entropies:
            avg_entropy = np.mean(cluster_entropies)
            transition_entropies.append(avg_entropy)
    
    # Class-specific fragmentation
    class_fragmentation = {}
    if labels is not None:
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            # Get samples with this label
            label_mask = (labels == label)
            label_samples = np.where(label_mask)[0]
            
            if len(label_samples) > 0:
                # Get unique paths for this class
                label_paths = [tuple(paths[i]) for i in label_samples]
                unique_label_paths = set(label_paths)
                
                # Compute fragmentation rate: proportion of samples following different paths
                fragmentation_rate = len(unique_label_paths) / len(label_samples)
                
                # Compute entropy of class-specific paths
                label_path_counts = Counter(label_paths)
                label_path_probs = np.array([count / len(label_samples) for count in label_path_counts.values()])
                label_entropy = -np.sum(label_path_probs * np.log2(label_path_probs + 1e-10))
                
                class_fragmentation[int(label)] = {
                    'fragmentation_rate': fragmentation_rate,
                    'path_entropy': label_entropy,
                    'unique_paths': len(unique_label_paths),
                    'samples': len(label_samples)
                }
    
    return {
        'overall_entropy': float(overall_entropy),
        'normalized_entropy': float(normalized_entropy),
        'transition_entropies': transition_entropies,
        'unique_paths': len(path_counts),
        'class_fragmentation': class_fragmentation
    }

def compute_inter_cluster_path_density(
    paths: np.ndarray,
    layer_names: List[str],
    min_density: float = 0.05,
    max_steps: int = 3,
    batch_size: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute inter-cluster path density for multi-step transitions.
    
    Args:
        paths: Array of paths (n_samples, n_layers)
        layer_names: Names of layers in paths
        min_density: Minimum density threshold to include in results
        max_steps: Maximum number of steps for path analysis
        batch_size: Optional batch size for large datasets
        
    Returns:
        Dictionary mapping path patterns to density matrices
    """
    # Validate layer ordering
    layer_names = validate_layer_order(layer_names)
    
    n_samples, n_layers = paths.shape
    
    # Ensure max_steps doesn't exceed the number of layers
    max_steps = min(max_steps, n_layers - 1)
    
    # Initialize results
    path_densities = {}
    
    # For each path length (1 to max_steps)
    for step_size in range(1, max_steps + 1):
        # For each valid starting layer
        for start_idx in range(n_layers - step_size):
            start_layer = layer_names[start_idx]
            end_idx = start_idx + step_size
            end_layer = layer_names[end_idx]
            
            # Get clusters in start and end layers
            start_clusters = paths[:, start_idx]
            end_clusters = paths[:, end_idx]
            
            # Get unique clusters
            unique_start = np.unique(start_clusters)
            unique_end = np.unique(end_clusters)
            
            # Initialize density matrix
            density_matrix = np.zeros((len(unique_start), len(unique_end)))
            
            # Count transitions
            for i, src_cluster in enumerate(unique_start):
                # Get samples starting in this cluster
                src_samples = (start_clusters == src_cluster)
                src_count = np.sum(src_samples)
                
                if src_count > 0:
                    for j, tgt_cluster in enumerate(unique_end):
                        # Count samples that end in target cluster
                        joint_count = np.sum((start_clusters == src_cluster) & (end_clusters == tgt_cluster))
                        
                        # Compute density as proportion of source samples
                        density_matrix[i, j] = joint_count / src_count
            
            # Filter out weak connections
            density_matrix[density_matrix < min_density] = 0
            
            # For paths longer than 1 step, include intermediate steps
            if step_size > 1:
                # Get all possible intermediate paths
                intermediate_paths = {}
                
                for sample_idx in range(n_samples):
                    sample_path = tuple(paths[sample_idx, start_idx:end_idx+1])
                    src_cluster = sample_path[0]
                    tgt_cluster = sample_path[-1]
                    
                    key = (src_cluster, tgt_cluster)
                    if key not in intermediate_paths:
                        intermediate_paths[key] = []
                    
                    intermediate_paths[key].append(sample_path[1:-1])  # Exclude start and end
                
                # Count frequency of each intermediate path
                for (src, tgt), intermediates in intermediate_paths.items():
                    counter = Counter(intermediates)
                    total = len(intermediates)
                    
                    # Map src, tgt to indices
                    src_idx = np.where(unique_start == src)[0][0]
                    tgt_idx = np.where(unique_end == tgt)[0][0]
                    
                    # Only include if density is above threshold
                    if density_matrix[src_idx, tgt_idx] >= min_density:
                        intermediate_densities = {
                            path: count / total for path, count in counter.items()
                        }
                        
                        # Store intermediate path data
                        intermediate_paths[(src, tgt)] = {
                            'paths': intermediate_densities,
                            'density': density_matrix[src_idx, tgt_idx]
                        }
            
            # Store results
            path_key = f"{start_layer}_to_{end_layer}"
            path_densities[path_key] = {
                'matrix': density_matrix,
                'source_clusters': unique_start,
                'target_clusters': unique_end,
                'intermediate_paths': intermediate_paths if step_size > 1 else None
            }
    
    return path_densities

def analyze_cross_layer_metrics(
    layer_clusters: Dict[str, np.ndarray],
    activations: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    config: Dict = None
) -> Dict:
    """
    Comprehensive analysis of cross-layer metrics.
    
    Args:
        layer_clusters: Dictionary mapping layer names to cluster assignments
        activations: Dictionary mapping layer names to activation matrices
        labels: Optional ground truth labels
        config: Optional configuration dictionary
        
    Returns:
        Dictionary of all cross-layer metrics
    """
    # Set default configuration
    if config is None:
        config = {
            'centroid_similarity': {
                'similarity_metric': 'cosine',
                'projection_method': 'pca',
                'projection_dims': 10
            },
            'membership_overlap': {
                'overlap_type': 'jaccard'
            },
            'trajectory_fragmentation': {},
            'path_density': {
                'min_density': 0.05,
                'max_steps': 3
            }
        }
    
    # Extract paths
    paths, layer_names = extract_paths(layer_clusters)
    
    # Compute metrics
    centroid_similarity = compute_centroid_similarity(
        layer_clusters=layer_clusters,
        activations=activations,
        **config.get('centroid_similarity', {})
    )
    
    membership_overlap = compute_membership_overlap(
        layer_clusters=layer_clusters,
        **config.get('membership_overlap', {})
    )
    
    fragmentation = compute_trajectory_fragmentation(
        paths=paths,
        layer_names=layer_names,
        labels=labels,
        **config.get('trajectory_fragmentation', {})
    )
    
    path_density = compute_inter_cluster_path_density(
        paths=paths,
        layer_names=layer_names,
        **config.get('path_density', {})
    )
    
    return {
        'centroid_similarity': centroid_similarity,
        'membership_overlap': membership_overlap,
        'fragmentation': fragmentation,
        'path_density': path_density,
        'paths': paths,
        'layer_names': layer_names
    }