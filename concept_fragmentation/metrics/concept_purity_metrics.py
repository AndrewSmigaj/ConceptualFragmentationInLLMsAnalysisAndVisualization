"""
Concept purity metrics for transformer models.

This module provides specialized metrics for analyzing the purity and 
distinctiveness of concept representations within transformer models.
These metrics help quantify how clearly separated different concepts
are in the model's activation space, revealing insights about the
model's internal abstraction mechanisms.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
from dataclasses import dataclass
from collections import defaultdict
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform, cosine

# Setup logger
logger = logging.getLogger(__name__)


@dataclass
class ConceptPurityResult:
    """
    Results from concept purity metrics calculations.
    
    Attributes:
        intra_cluster_coherence: Dictionary mapping layers to coherence scores
        cross_layer_stability: Dictionary mapping layer pairs to stability scores
        concept_separability: Dictionary mapping layers to separability metrics
        concept_entropy: Dictionary mapping layers to entropy measurements
        layer_concept_purity: Dictionary mapping layers to overall purity scores
        concept_contamination: Dictionary mapping concepts to contamination scores
        concept_overlap: Dictionary mapping concept pairs to overlap scores
        concepts_analyzed: List of concept identifiers that were analyzed
        layers_analyzed: List of layer names that were analyzed
        aggregated_metrics: Dictionary with aggregated metrics across all concepts
    """
    intra_cluster_coherence: Dict[str, float]
    cross_layer_stability: Dict[Tuple[str, str], float]
    concept_separability: Dict[str, Dict[str, float]]
    concept_entropy: Dict[str, float]
    layer_concept_purity: Dict[str, float]
    concept_contamination: Dict[str, float]
    concept_overlap: Dict[Tuple[str, str], float]
    concepts_analyzed: List[str]
    layers_analyzed: List[str]
    aggregated_metrics: Dict[str, float]


def calculate_intra_cluster_coherence(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    metric: str = "cosine"
) -> Dict[str, float]:
    """
    Calculate coherence within concept clusters for each layer.
    
    Intra-cluster coherence measures how tightly grouped activations are
    within each concept cluster. Higher coherence indicates that members
    of the same concept are represented similarly.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
                         [n_samples, n_features]
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping layer names to coherence scores
    """
    # Ensure all inputs are numpy arrays
    processed_activations = {}
    for layer_name, activations in layer_activations.items():
        if isinstance(activations, torch.Tensor):
            processed_activations[layer_name] = activations.detach().cpu().numpy()
        else:
            processed_activations[layer_name] = activations
    
    processed_labels = {}
    for layer_name, labels in cluster_labels.items():
        if isinstance(labels, torch.Tensor):
            processed_labels[layer_name] = labels.detach().cpu().numpy()
        elif isinstance(labels, list):
            processed_labels[layer_name] = np.array(labels)
        else:
            processed_labels[layer_name] = labels
    
    # Calculate coherence for each layer
    coherence_scores = {}
    
    for layer_name in processed_activations.keys():
        # Skip if we don't have labels for this layer
        if layer_name not in processed_labels:
            logger.warning(f"No cluster labels for layer {layer_name}")
            continue
        
        activations = processed_activations[layer_name]
        labels = processed_labels[layer_name]
        
        # Ensure activations are 2D [n_samples, n_features]
        if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            # Reshape to [batch_size * seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activations.shape
            activations = activations.reshape(-1, hidden_dim)
            
            # Reshape labels if needed
            if len(labels.shape) == 2:  # [batch_size, seq_len]
                labels = labels.reshape(-1)
        
        # Skip if shapes don't match
        if activations.shape[0] != labels.shape[0]:
            logger.warning(f"Shape mismatch for layer {layer_name}: "
                          f"activations {activations.shape}, labels {labels.shape}")
            continue
        
        # Get unique clusters
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)
        
        # Skip if only one cluster or too few samples
        if n_clusters <= 1 or activations.shape[0] < 10:
            logger.warning(f"Too few clusters or samples for layer {layer_name}")
            coherence_scores[layer_name] = 0.0
            continue
        
        try:
            # For coherence, we use the silhouette score
            # Higher value means better defined clusters
            if n_clusters < activations.shape[0]:
                silhouette = silhouette_score(activations, labels, metric=metric)
                coherence_scores[layer_name] = float(silhouette)
            else:
                # Too many clusters for silhouette score, use alternative
                # Calculate average pairwise similarity within clusters
                similarities = []
                
                for cluster_id in unique_clusters:
                    # Get activations for this cluster
                    cluster_mask = (labels == cluster_id)
                    
                    # Skip if cluster has fewer than 2 samples
                    if np.sum(cluster_mask) < 2:
                        continue
                        
                    cluster_acts = activations[cluster_mask]
                    
                    # Calculate pairwise distances
                    if metric == "cosine":
                        distances = pdist(cluster_acts, metric="cosine")
                        # Convert distances to similarities
                        cluster_similarities = 1 - distances
                    elif metric == "euclidean":
                        distances = pdist(cluster_acts, metric="euclidean")
                        # Normalize and convert to similarities
                        max_dist = np.max(distances) if len(distances) > 0 else 1.0
                        cluster_similarities = 1 - (distances / max_dist)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                    
                    # Add to overall similarities
                    if len(cluster_similarities) > 0:
                        similarities.extend(cluster_similarities)
                
                # Average similarity as coherence
                if similarities:
                    coherence_scores[layer_name] = float(np.mean(similarities))
                else:
                    coherence_scores[layer_name] = 0.0
        
        except Exception as e:
            logger.warning(f"Error calculating coherence for layer {layer_name}: {e}")
            coherence_scores[layer_name] = 0.0
    
    return coherence_scores


def calculate_cross_layer_stability(
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    item_ids: Optional[Union[torch.Tensor, np.ndarray, List[Any]]] = None
) -> Dict[Tuple[str, str], float]:
    """
    Calculate stability of concept clusters across consecutive layers.
    
    Cross-layer stability measures how consistently items are assigned
    to the same or similar clusters as they progress through layers.
    Higher stability indicates more consistent concept representations.
    
    Args:
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        item_ids: Optional identifiers for each item, used to track items across layers
                If None, assumes items are in the same order in all layers
        
    Returns:
        Dictionary mapping layer pairs to stability scores
    """
    # Ensure all inputs are numpy arrays
    processed_labels = {}
    for layer_name, labels in cluster_labels.items():
        if isinstance(labels, torch.Tensor):
            processed_labels[layer_name] = labels.detach().cpu().numpy()
        elif isinstance(labels, list):
            processed_labels[layer_name] = np.array(labels)
        else:
            processed_labels[layer_name] = labels
    
    if item_ids is not None:
        if isinstance(item_ids, torch.Tensor):
            item_ids = item_ids.detach().cpu().numpy()
        elif isinstance(item_ids, list):
            item_ids = np.array(item_ids)
    
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(processed_labels.keys())
    
    # Skip if fewer than 2 layers
    if len(layer_names) < 2:
        logger.warning("Need at least 2 layers to calculate cross-layer stability")
        return {}
    
    # Calculate stability between consecutive layers
    stability_scores = {}
    
    for i in range(len(layer_names) - 1):
        layer1 = layer_names[i]
        layer2 = layer_names[i + 1]
        
        labels1 = processed_labels[layer1]
        labels2 = processed_labels[layer2]
        
        # Flatten labels if needed
        if len(labels1.shape) > 1:
            labels1 = labels1.flatten()
        if len(labels2.shape) > 1:
            labels2 = labels2.flatten()
        
        # Skip if different number of items
        if labels1.shape[0] != labels2.shape[0]:
            # Check if we can use item_ids to match items
            if item_ids is not None:
                logger.warning(f"Different number of items in layers {layer1} and {layer2}. "
                              f"Using item_ids to match items.")
                
                # TODO: Implement matching with item_ids if needed
                # This would require more complex logic to match items across layers
                continue
            else:
                logger.warning(f"Different number of items in layers {layer1} and {layer2}, "
                              f"and no item_ids provided for matching.")
                continue
        
        try:
            # Calculate adjusted mutual information score
            # This measures how well the clusters in one layer predict the clusters in the next
            ami_score = adjusted_mutual_info_score(labels1, labels2)
            
            # Calculate normalized mutual information score
            # This variant is normalized to [0, 1] range
            nmi_score = normalized_mutual_info_score(labels1, labels2)
            
            # Calculate adjusted Rand index
            # This measures the similarity of the two clusterings
            ari_score = adjusted_rand_score(labels1, labels2)
            
            # Combine scores (weighted average)
            stability = 0.4 * ami_score + 0.3 * nmi_score + 0.3 * ari_score
            
            stability_scores[(layer1, layer2)] = float(stability)
        
        except Exception as e:
            logger.warning(f"Error calculating stability for layers {layer1} and {layer2}: {e}")
            stability_scores[(layer1, layer2)] = 0.0
    
    return stability_scores


def calculate_concept_separability(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    metric: str = "cosine"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate separability of concept clusters for each layer.
    
    Concept separability measures how well-separated different concept
    clusters are in the activation space. Well-separated clusters indicate
    that the model represents different concepts distinctly.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
                         [n_samples, n_features]
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping layer names to separability metrics
    """
    # Ensure all inputs are numpy arrays
    processed_activations = {}
    for layer_name, activations in layer_activations.items():
        if isinstance(activations, torch.Tensor):
            processed_activations[layer_name] = activations.detach().cpu().numpy()
        else:
            processed_activations[layer_name] = activations
    
    processed_labels = {}
    for layer_name, labels in cluster_labels.items():
        if isinstance(labels, torch.Tensor):
            processed_labels[layer_name] = labels.detach().cpu().numpy()
        elif isinstance(labels, list):
            processed_labels[layer_name] = np.array(labels)
        else:
            processed_labels[layer_name] = labels
    
    # Calculate separability for each layer
    separability_scores = {}
    
    for layer_name in processed_activations.keys():
        # Skip if we don't have labels for this layer
        if layer_name not in processed_labels:
            logger.warning(f"No cluster labels for layer {layer_name}")
            continue
        
        activations = processed_activations[layer_name]
        labels = processed_labels[layer_name]
        
        # Ensure activations are 2D [n_samples, n_features]
        if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
            # Reshape to [batch_size * seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = activations.shape
            activations = activations.reshape(-1, hidden_dim)
            
            # Reshape labels if needed
            if len(labels.shape) == 2:  # [batch_size, seq_len]
                labels = labels.reshape(-1)
        
        # Skip if shapes don't match
        if activations.shape[0] != labels.shape[0]:
            logger.warning(f"Shape mismatch for layer {layer_name}: "
                          f"activations {activations.shape}, labels {labels.shape}")
            continue
        
        # Get unique clusters
        unique_clusters = np.unique(labels)
        n_clusters = len(unique_clusters)
        
        # Skip if only one cluster or too few samples
        if n_clusters <= 1 or activations.shape[0] < 10:
            logger.warning(f"Too few clusters or samples for layer {layer_name}")
            separability_scores[layer_name] = {"calinski_harabasz": 0.0, "davies_bouldin": 1.0, "between_within_ratio": 0.0}
            continue
        
        try:
            # Calculate different separability metrics
            metrics = {}
            
            # 1. Calinski-Harabasz Index (higher is better)
            # Ratio of between-cluster variance to within-cluster variance
            if n_clusters < activations.shape[0]:
                ch_score = calinski_harabasz_score(activations, labels)
                metrics["calinski_harabasz"] = float(ch_score)
            else:
                metrics["calinski_harabasz"] = 0.0
            
            # 2. Davies-Bouldin Index (lower is better)
            # Average similarity between clusters
            if n_clusters < activations.shape[0]:
                db_score = davies_bouldin_score(activations, labels)
                # Invert so higher is better, and normalize to [0, 1]
                db_normalized = max(0.0, min(1.0, 1.0 / (1.0 + db_score)))
                metrics["davies_bouldin"] = float(db_normalized)
            else:
                metrics["davies_bouldin"] = 0.0
            
            # 3. Between-to-within cluster distance ratio
            # Calculate cluster centroids
            centroids = {}
            for cluster_id in unique_clusters:
                cluster_mask = (labels == cluster_id)
                if np.sum(cluster_mask) > 0:
                    centroids[cluster_id] = np.mean(activations[cluster_mask], axis=0)
            
            # Calculate average between-cluster distances
            between_distances = []
            for c1 in centroids:
                for c2 in centroids:
                    if c1 < c2:  # Avoid duplicates and self-comparisons
                        if metric == "cosine":
                            dist = cosine(centroids[c1], centroids[c2])
                        elif metric == "euclidean":
                            dist = np.linalg.norm(centroids[c1] - centroids[c2])
                        else:
                            raise ValueError(f"Unsupported metric: {metric}")
                        between_distances.append(dist)
            
            # Calculate average within-cluster distances
            within_distances = []
            for cluster_id in unique_clusters:
                cluster_mask = (labels == cluster_id)
                if np.sum(cluster_mask) > 1:  # Need at least 2 samples
                    cluster_acts = activations[cluster_mask]
                    centroid = centroids[cluster_id]
                    
                    # Calculate distances to centroid
                    for i in range(cluster_acts.shape[0]):
                        if metric == "cosine":
                            dist = cosine(cluster_acts[i], centroid)
                        elif metric == "euclidean":
                            dist = np.linalg.norm(cluster_acts[i] - centroid)
                        else:
                            raise ValueError(f"Unsupported metric: {metric}")
                        within_distances.append(dist)
            
            # Calculate ratio
            if within_distances and np.mean(within_distances) > 0:
                bw_ratio = np.mean(between_distances) / np.mean(within_distances)
                metrics["between_within_ratio"] = float(bw_ratio)
            else:
                metrics["between_within_ratio"] = 0.0
            
            separability_scores[layer_name] = metrics
        
        except Exception as e:
            logger.warning(f"Error calculating separability for layer {layer_name}: {e}")
            separability_scores[layer_name] = {"calinski_harabasz": 0.0, "davies_bouldin": 0.0, "between_within_ratio": 0.0}
    
    return separability_scores


def calculate_concept_entropy(
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]]
) -> Dict[str, float]:
    """
    Calculate entropy of concept cluster distributions for each layer.
    
    Concept entropy measures how evenly distributed items are across
    different clusters. Lower entropy indicates that some clusters
    dominate, while higher entropy indicates more balanced clusters.
    
    Args:
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        
    Returns:
        Dictionary mapping layer names to entropy values
    """
    # Ensure all inputs are numpy arrays
    processed_labels = {}
    for layer_name, labels in cluster_labels.items():
        if isinstance(labels, torch.Tensor):
            processed_labels[layer_name] = labels.detach().cpu().numpy()
        elif isinstance(labels, list):
            processed_labels[layer_name] = np.array(labels)
        else:
            processed_labels[layer_name] = labels
    
    # Calculate entropy for each layer
    entropy_scores = {}
    
    for layer_name, labels in processed_labels.items():
        # Flatten labels if needed
        if len(labels.shape) > 1:
            labels = labels.flatten()
        
        # Count occurrences of each cluster
        unique_clusters, counts = np.unique(labels, return_counts=True)
        
        # Skip if no clusters
        if len(unique_clusters) == 0:
            entropy_scores[layer_name] = 0.0
            continue
        
        # Calculate probabilities
        total_samples = np.sum(counts)
        probabilities = counts / total_samples
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(unique_clusters))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        entropy_scores[layer_name] = float(normalized_entropy)
    
    return entropy_scores


def calculate_layer_concept_purity(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    ground_truth_labels: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    metric: str = "cosine"
) -> Dict[str, float]:
    """
    Calculate overall concept purity for each layer.
    
    Layer concept purity combines multiple metrics to provide a holistic
    assessment of how purely concepts are represented in each layer.
    Higher purity indicates that concepts are well-separated and coherent.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
                         [n_samples, n_features]
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        ground_truth_labels: Optional ground truth labels for supervised evaluation
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping layer names to overall purity scores
    """
    # Calculate component metrics
    coherence = calculate_intra_cluster_coherence(layer_activations, cluster_labels, metric)
    separability = calculate_concept_separability(layer_activations, cluster_labels, metric)
    entropy = calculate_concept_entropy(cluster_labels)
    
    # Process ground truth labels if provided
    processed_ground_truth = None
    if ground_truth_labels is not None:
        if isinstance(ground_truth_labels, torch.Tensor):
            processed_ground_truth = ground_truth_labels.detach().cpu().numpy()
        elif isinstance(ground_truth_labels, list):
            processed_ground_truth = np.array(ground_truth_labels)
        else:
            processed_ground_truth = ground_truth_labels
        
        # Flatten if needed
        if len(processed_ground_truth.shape) > 1:
            processed_ground_truth = processed_ground_truth.flatten()
    
    # Calculate purity for each layer
    purity_scores = {}
    
    for layer_name in coherence.keys():
        # Skip if we don't have all metrics for this layer
        if layer_name not in separability or layer_name not in entropy:
            logger.warning(f"Missing metrics for layer {layer_name}")
            continue
        
        # Get individual metrics
        coherence_score = coherence[layer_name]
        
        # For separability, use the average of available metrics
        sep_metrics = separability[layer_name]
        # Convert davies_bouldin so higher is better
        sep_scores = [
            sep_metrics.get("calinski_harabasz", 0.0) / 100.0,  # Normalize large values
            sep_metrics.get("davies_bouldin", 0.0),  # Already normalized
            sep_metrics.get("between_within_ratio", 0.0)  # Already normalized
        ]
        separability_score = np.mean([s for s in sep_scores if s >= 0])
        
        # Get entropy score
        entropy_score = entropy[layer_name]
        
        # Add supervised evaluation if ground truth is available
        supervised_score = 0.0
        if processed_ground_truth is not None:
            try:
                # Get cluster labels for this layer
                labels = cluster_labels[layer_name]
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()
                elif isinstance(labels, list):
                    labels = np.array(labels)
                
                # Flatten labels if needed
                if len(labels.shape) > 1:
                    labels = labels.flatten()
                
                # Skip if shapes don't match
                if labels.shape[0] != processed_ground_truth.shape[0]:
                    logger.warning(f"Shape mismatch for layer {layer_name}: "
                                  f"labels {labels.shape}, ground truth {processed_ground_truth.shape}")
                else:
                    # Calculate adjusted mutual information score
                    ami_score = adjusted_mutual_info_score(labels, processed_ground_truth)
                    
                    # Calculate normalized mutual information score
                    nmi_score = normalized_mutual_info_score(labels, processed_ground_truth)
                    
                    # Calculate adjusted Rand index
                    ari_score = adjusted_rand_score(labels, processed_ground_truth)
                    
                    # Combine scores (weighted average)
                    supervised_score = 0.4 * ami_score + 0.3 * nmi_score + 0.3 * ari_score
            
            except Exception as e:
                logger.warning(f"Error calculating supervised metrics for layer {layer_name}: {e}")
        
        # Calculate weighted combination
        # Weights depend on whether ground truth is available
        if processed_ground_truth is not None:
            # With ground truth, give more weight to supervised_score
            purity = (0.3 * coherence_score + 
                      0.3 * separability_score + 
                      0.1 * (1 - entropy_score) +  # Lower entropy is better for purity
                      0.3 * supervised_score)
        else:
            # Without ground truth, rely more on coherence and separability
            purity = (0.4 * coherence_score + 
                     0.4 * separability_score + 
                     0.2 * (1 - entropy_score))  # Lower entropy is better for purity
        
        # Normalize to [0, 1] range
        purity = max(0.0, min(1.0, purity))
        
        purity_scores[layer_name] = float(purity)
    
    return purity_scores


def calculate_concept_contamination(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    ground_truth_labels: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None
) -> Dict[str, float]:
    """
    Calculate contamination of concept clusters.
    
    Concept contamination measures how much concepts mix with unrelated
    items. Lower contamination indicates purer concepts, while higher
    contamination indicates concepts that include unrelated elements.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
                         [n_samples, n_features]
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        ground_truth_labels: Optional ground truth labels for supervised evaluation
        
    Returns:
        Dictionary mapping concept identifiers to contamination scores
    """
    # Skip if no ground truth is available
    if ground_truth_labels is None:
        logger.warning("Ground truth labels required for contamination calculation")
        return {}
    
    # Process ground truth labels
    if isinstance(ground_truth_labels, torch.Tensor):
        ground_truth = ground_truth_labels.detach().cpu().numpy()
    elif isinstance(ground_truth_labels, list):
        ground_truth = np.array(ground_truth_labels)
    else:
        ground_truth = ground_truth_labels
    
    # Flatten if needed
    if len(ground_truth.shape) > 1:
        ground_truth = ground_truth.flatten()
    
    # Get unique classes in ground truth
    unique_classes = np.unique(ground_truth)
    
    # Process cluster labels (focus on last layer for concept analysis)
    last_layer = sorted(cluster_labels.keys())[-1]
    labels = cluster_labels[last_layer]
    
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
    
    # Flatten if needed
    if len(labels.shape) > 1:
        labels = labels.flatten()
    
    # Skip if shapes don't match
    if labels.shape[0] != ground_truth.shape[0]:
        logger.warning(f"Shape mismatch: labels {labels.shape}, ground truth {ground_truth.shape}")
        return {}
    
    # Calculate contamination for each ground truth class
    contamination_scores = {}
    
    for class_id in unique_classes:
        # Get indices of samples in this class
        class_mask = (ground_truth == class_id)
        class_samples = np.sum(class_mask)
        
        if class_samples == 0:
            continue
        
        # Get cluster assignments for these samples
        class_clusters = labels[class_mask]
        
        # Count occurrences of each cluster
        cluster_counts = {}
        for cluster_id in np.unique(class_clusters):
            cluster_counts[cluster_id] = np.sum(class_clusters == cluster_id)
        
        # Find the most common cluster for this class
        if cluster_counts:
            most_common_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate what percentage of this class is in other clusters
            samples_in_most_common = cluster_counts[most_common_cluster]
            contamination = 1.0 - (samples_in_most_common / class_samples)
        else:
            contamination = 1.0
        
        # Store result
        contamination_scores[f"concept_{class_id}"] = float(contamination)
    
    return contamination_scores


def calculate_concept_overlap(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    ground_truth_labels: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    metric: str = "cosine"
) -> Dict[Tuple[str, str], float]:
    """
    Calculate overlap between different concept clusters.
    
    Concept overlap measures how much different concepts share the
    same activation space. Lower overlap indicates well-separated
    concepts, while higher overlap indicates concepts that are
    difficult to distinguish.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
                         [n_samples, n_features]
        cluster_labels: Dictionary mapping layer names to cluster assignments
                      [n_samples]
        ground_truth_labels: Optional ground truth labels for supervised evaluation
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        Dictionary mapping concept pairs to overlap scores
    """
    # Skip if no ground truth is available
    if ground_truth_labels is None:
        logger.warning("Ground truth labels required for overlap calculation")
        return {}
    
    # Process ground truth labels
    if isinstance(ground_truth_labels, torch.Tensor):
        ground_truth = ground_truth_labels.detach().cpu().numpy()
    elif isinstance(ground_truth_labels, list):
        ground_truth = np.array(ground_truth_labels)
    else:
        ground_truth = ground_truth_labels
    
    # Flatten if needed
    if len(ground_truth.shape) > 1:
        ground_truth = ground_truth.flatten()
    
    # Get unique classes in ground truth
    unique_classes = np.unique(ground_truth)
    
    # Process activations (focus on last layer for concept analysis)
    last_layer = sorted(layer_activations.keys())[-1]
    activations = layer_activations[last_layer]
    
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()
    
    # Ensure activations are 2D [n_samples, n_features]
    if len(activations.shape) == 3:  # [batch_size, seq_len, hidden_dim]
        # Reshape to [batch_size * seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = activations.shape
        activations = activations.reshape(-1, hidden_dim)
        
        # Reshape ground truth if needed
        if len(ground_truth.shape) == 2:  # [batch_size, seq_len]
            ground_truth = ground_truth.reshape(-1)
    
    # Skip if shapes don't match
    if activations.shape[0] != ground_truth.shape[0]:
        logger.warning(f"Shape mismatch: activations {activations.shape}, ground truth {ground_truth.shape}")
        return {}
    
    # Calculate centroids for each class
    centroids = {}
    for class_id in unique_classes:
        class_mask = (ground_truth == class_id)
        # Skip if no samples in this class
        if np.sum(class_mask) == 0:
            continue
        centroids[class_id] = np.mean(activations[class_mask], axis=0)
    
    # Calculate overlap between each pair of concepts
    overlap_scores = {}
    
    for c1 in centroids:
        for c2 in centroids:
            if c1 < c2:  # Avoid duplicates and self-comparisons
                # Calculate distance between centroids
                if metric == "cosine":
                    # Cosine similarity (convert distance to similarity)
                    distance = cosine(centroids[c1], centroids[c2])
                    similarity = 1 - distance
                    
                    # Higher similarity = more overlap
                    overlap = similarity
                    
                elif metric == "euclidean":
                    # Euclidean distance (normalize and convert to overlap)
                    distance = np.linalg.norm(centroids[c1] - centroids[c2])
                    
                    # Convert distance to overlap score [0, 1]
                    # Smaller distance = higher overlap
                    overlap = 1 / (1 + distance)
                    
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                # Store result
                overlap_scores[(f"concept_{c1}", f"concept_{c2}")] = float(overlap)
    
    return overlap_scores


def analyze_concept_purity(
    layer_activations: Dict[str, Union[torch.Tensor, np.ndarray]],
    cluster_labels: Dict[str, Union[torch.Tensor, np.ndarray, List[int]]],
    ground_truth_labels: Optional[Union[torch.Tensor, np.ndarray, List[int]]] = None,
    metric: str = "cosine"
) -> ConceptPurityResult:
    """
    Comprehensively analyze concept purity across transformer layers.
    
    This function combines multiple purity metrics to provide a holistic
    view of how concepts are represented in the network.
    
    Args:
        layer_activations: Dictionary mapping layer names to activation tensors
        cluster_labels: Dictionary mapping layer names to cluster assignments
        ground_truth_labels: Optional ground truth labels for supervised evaluation
        metric: Distance metric to use ('cosine', 'euclidean')
        
    Returns:
        ConceptPurityResult with comprehensive concept purity metrics
    """
    # Sort layers by name to ensure consistent processing
    layer_names = sorted(layer_activations.keys())
    
    # Get concepts to analyze
    concepts_analyzed = []
    if ground_truth_labels is not None:
        # Use ground truth concepts if available
        processed_gt = None
        if isinstance(ground_truth_labels, torch.Tensor):
            processed_gt = ground_truth_labels.detach().cpu().numpy()
        elif isinstance(ground_truth_labels, list):
            processed_gt = np.array(ground_truth_labels)
        else:
            processed_gt = ground_truth_labels
        
        # Flatten if needed
        if len(processed_gt.shape) > 1:
            processed_gt = processed_gt.flatten()
        
        # Get unique concepts
        unique_concepts = np.unique(processed_gt)
        concepts_analyzed = [f"concept_{c}" for c in unique_concepts]
    else:
        # Use cluster IDs from last layer
        last_layer = layer_names[-1]
        if last_layer in cluster_labels:
            labels = cluster_labels[last_layer]
            
            if isinstance(labels, torch.Tensor):
                labels = labels.detach().cpu().numpy()
            elif isinstance(labels, list):
                labels = np.array(labels)
            
            # Flatten if needed
            if len(labels.shape) > 1:
                labels = labels.flatten()
            
            # Get unique clusters
            unique_clusters = np.unique(labels)
            concepts_analyzed = [f"cluster_{c}" for c in unique_clusters]
    
    # Calculate intra-cluster coherence
    logger.info("Calculating intra-cluster coherence")
    intra_cluster_coherence = calculate_intra_cluster_coherence(
        layer_activations=layer_activations,
        cluster_labels=cluster_labels,
        metric=metric
    )
    
    # Calculate cross-layer stability
    logger.info("Calculating cross-layer stability")
    cross_layer_stability = calculate_cross_layer_stability(
        cluster_labels=cluster_labels
    )
    
    # Calculate concept separability
    logger.info("Calculating concept separability")
    concept_separability = calculate_concept_separability(
        layer_activations=layer_activations,
        cluster_labels=cluster_labels,
        metric=metric
    )
    
    # Calculate concept entropy
    logger.info("Calculating concept entropy")
    concept_entropy = calculate_concept_entropy(
        cluster_labels=cluster_labels
    )
    
    # Calculate layer concept purity
    logger.info("Calculating layer concept purity")
    layer_concept_purity = calculate_layer_concept_purity(
        layer_activations=layer_activations,
        cluster_labels=cluster_labels,
        ground_truth_labels=ground_truth_labels,
        metric=metric
    )
    
    # Calculate concept contamination
    logger.info("Calculating concept contamination")
    concept_contamination = {}
    if ground_truth_labels is not None:
        concept_contamination = calculate_concept_contamination(
            layer_activations=layer_activations,
            cluster_labels=cluster_labels,
            ground_truth_labels=ground_truth_labels
        )
    
    # Calculate concept overlap
    logger.info("Calculating concept overlap")
    concept_overlap = {}
    if ground_truth_labels is not None:
        concept_overlap = calculate_concept_overlap(
            layer_activations=layer_activations,
            cluster_labels=cluster_labels,
            ground_truth_labels=ground_truth_labels,
            metric=metric
        )
    
    # Calculate aggregated metrics
    avg_coherence = np.mean(list(intra_cluster_coherence.values())) if intra_cluster_coherence else 0.0
    avg_stability = np.mean(list(cross_layer_stability.values())) if cross_layer_stability else 0.0
    
    # Calculate average separability using the between_within_ratio
    separability_values = []
    for layer_metrics in concept_separability.values():
        if "between_within_ratio" in layer_metrics:
            separability_values.append(layer_metrics["between_within_ratio"])
    avg_separability = np.mean(separability_values) if separability_values else 0.0
    
    avg_purity = np.mean(list(layer_concept_purity.values())) if layer_concept_purity else 0.0
    avg_contamination = np.mean(list(concept_contamination.values())) if concept_contamination else 0.0
    avg_overlap = np.mean(list(concept_overlap.values())) if concept_overlap else 0.0
    
    # Create aggregated metrics dictionary
    aggregated_metrics = {
        "average_coherence": float(avg_coherence),
        "average_stability": float(avg_stability),
        "average_separability": float(avg_separability),
        "average_purity": float(avg_purity),
        "average_contamination": float(avg_contamination),
        "average_overlap": float(avg_overlap)
    }
    
    # Create result
    result = ConceptPurityResult(
        intra_cluster_coherence=intra_cluster_coherence,
        cross_layer_stability=cross_layer_stability,
        concept_separability=concept_separability,
        concept_entropy=concept_entropy,
        layer_concept_purity=layer_concept_purity,
        concept_contamination=concept_contamination,
        concept_overlap=concept_overlap,
        concepts_analyzed=concepts_analyzed,
        layers_analyzed=layer_names,
        aggregated_metrics=aggregated_metrics
    )
    
    return result