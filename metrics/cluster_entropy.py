import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import torch

def cluster_entropy(activations, labels, n_clusters=5, layer_name=None, 
                    use_pca=True, pca_components=0.95, return_clusters=False):
    """
    Calculate entropy-based fragmentation metric for each class.
    
    Parameters:
    -----------
    activations : dict or torch.Tensor
        If dict: {layer_name: activation_tensor}, use layer_name param to select layer
        If tensor: Directly use as activations
    labels : torch.Tensor or numpy.ndarray
        Class labels corresponding to each activation
    n_clusters : int, default=5
        Number of clusters for k-means
    layer_name : str, default=None
        Layer name to use if activations is a dict
    use_pca : bool, default=True
        Whether to apply PCA before clustering
    pca_components : float or int, default=0.95
        If float (0-1): Minimum explained variance
        If int: Number of components
    return_clusters : bool, default=False
        If True, return cluster assignments alongside metrics
        
    Returns:
    --------
    dict
        {
            'per_class': {class_idx: entropy_value, ...},
            'mean': mean_entropy_across_classes,
            'max': max_entropy_across_classes
        }
        If return_clusters=True: Also includes 'cluster_assignments'
    """
    # Extract activations from dict if needed
    if isinstance(activations, dict):
        if layer_name is None:
            raise ValueError("layer_name must be provided when activations is a dictionary")
        act = activations[layer_name]
    else:
        act = activations
    
    # Convert to numpy if needed
    if isinstance(act, torch.Tensor):
        act = act.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Apply PCA if requested
    if use_pca and act.shape[1] > 10:  # Only apply if dimension is large enough
        pca = PCA(n_components=pca_components)
        act = pca.fit_transform(act)
    
    # Calculate entropy for each class
    results = {'per_class': {}}
    all_cluster_assignments = {}
    
    for class_idx in unique_classes:
        # Extract activations for this class
        class_mask = (labels == class_idx)
        if np.sum(class_mask) <= 1:  # Skip if only one sample or none
            results['per_class'][int(class_idx)] = 0.0
            continue
            
        class_activations = act[class_mask]
        
        # Apply k-means clustering
        # Adjust number of clusters if fewer samples than requested clusters
        effective_n_clusters = min(n_clusters, len(class_activations) - 1)
        if effective_n_clusters <= 1:
            results['per_class'][int(class_idx)] = 0.0
            continue
            
        kmeans = KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(class_activations)
        
        # Save cluster assignments if requested
        if return_clusters:
            all_cluster_assignments[int(class_idx)] = cluster_labels
        
        # Calculate proportions and entropy
        counts = np.bincount(cluster_labels, minlength=effective_n_clusters)
        proportions = counts / len(cluster_labels)
        # Filter out zero proportions to avoid log(0)
        proportions = proportions[proportions > 0]
        entropy = -np.sum(proportions * np.log2(proportions))
        
        # Normalize by maximum possible entropy (uniform distribution)
        max_entropy = np.log2(effective_n_clusters)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        results['per_class'][int(class_idx)] = float(normalized_entropy)
    
    # Calculate aggregate metrics
    if results['per_class']:
        class_entropies = list(results['per_class'].values())
        results['mean'] = float(np.mean(class_entropies))
        results['max'] = float(np.max(class_entropies))
    else:
        results['mean'] = 0.0
        results['max'] = 0.0
    
    if return_clusters:
        results['cluster_assignments'] = all_cluster_assignments
        
    return results 