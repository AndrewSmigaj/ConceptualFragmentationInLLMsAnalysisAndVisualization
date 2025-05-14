import numpy as np
from sklearn.decomposition import PCA
import torch
from scipy.linalg import svd
import warnings

def principal_angles(subspace1, subspace2):
    """
    Calculate principal angles between two linear subspaces.
    
    Parameters:
    -----------
    subspace1, subspace2 : numpy.ndarray
        Orthonormal bases for the subspaces
        
    Returns:
    --------
    numpy.ndarray
        Principal angles in radians (sorted in ascending order)
    """
    # SVD of the inner product of the two subspaces
    Y1_T_Y2 = subspace1.T @ subspace2
    
    # Calculate SVD
    U, s, Vh = svd(Y1_T_Y2, full_matrices=False)
    
    # Clamp values to [-1, 1] to avoid numerical issues
    s = np.clip(s, -1.0, 1.0)
    
    # Calculate angles from singular values
    angles = np.arccos(s)
    
    return angles

def bootstrap_subspace(activations, n_bootstrap=10, random_state=42):
    """
    Generate bootstrapped versions of the subspace.
    
    Parameters:
    -----------
    activations : numpy.ndarray
        Activation matrix for a class
    n_bootstrap : int, default=10
        Number of bootstrap samples
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    list
        List of bootstrapped orthonormal bases
    """
    rng = np.random.RandomState(random_state)
    n_samples = activations.shape[0]
    bases = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = activations[indices]
        
        # Skip if not enough unique samples
        if np.unique(indices).size < 2:
            continue
            
        # Compute PCA
        try:
            pca = PCA(n_components=min(bootstrap_sample.shape[0]-1, bootstrap_sample.shape[1]))
            pca.fit(bootstrap_sample)
            bases.append(pca.components_.T)  # Use components as basis
        except Exception as e:
            warnings.warn(f"PCA failed on bootstrap sample: {str(e)}")
            continue
    
    return bases

def subspace_angle(activations, labels, layer_name=None, variance_threshold=0.9, 
                   n_bootstrap=10, random_state=42):
    """
    Calculate pairwise principal angles between class subspaces.
    
    Parameters:
    -----------
    activations : dict or torch.Tensor
        If dict: {layer_name: activation_tensor}, use layer_name param to select layer
        If tensor: Directly use as activations
    labels : torch.Tensor or numpy.ndarray
        Class labels corresponding to each activation
    layer_name : str, default=None
        Layer name to use if activations is a dict
    variance_threshold : float, default=0.9
        Minimum explained variance for PCA
    n_bootstrap : int, default=10
        Number of bootstrap iterations
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        {
            'pairwise': {(class1, class2): angle_stats, ...},
            'mean_angle': mean of all pairwise angles,
            'std_angle': standard deviation of all pairwise angles,
            'ci_95': (lower, upper) 95% confidence interval
        }
        where angle_stats contains 'mean', 'std', 'min', 'max', 'median'
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
    
    # Prepare results dictionary
    results = {'pairwise': {}}
    all_angles = []
    
    # For each pair of classes
    for i, class1 in enumerate(unique_classes):
        for j, class2 in enumerate(unique_classes):
            if i >= j:  # Only calculate for unique pairs (and skip same class)
                continue
                
            # Extract activations for each class
            class1_mask = (labels == class1)
            class2_mask = (labels == class2)
            
            if np.sum(class1_mask) < 3 or np.sum(class2_mask) < 3:
                # Skip if not enough samples
                results['pairwise'][(int(class1), int(class2))] = {
                    'mean': np.nan, 'std': np.nan, 'min': np.nan, 
                    'max': np.nan, 'median': np.nan, 'ci_95': (np.nan, np.nan)
                }
                continue
                
            class1_activations = act[class1_mask]
            class2_activations = act[class2_mask]
            
            # For small networks, we can use a simpler approach (tiny nets)
            if act.shape[1] <= 10:  # Small activation dimension
                # Compute principal components for each class
                pca1 = PCA(n_components=min(class1_activations.shape[0]-1, class1_activations.shape[1]))
                pca1.fit(class1_activations)
                
                pca2 = PCA(n_components=min(class2_activations.shape[0]-1, class2_activations.shape[1]))
                pca2.fit(class2_activations)
                
                # Take components that explain variance_threshold of variance
                var1 = np.cumsum(pca1.explained_variance_ratio_)
                n_comp1 = np.argmax(var1 >= variance_threshold) + 1 if any(var1 >= variance_threshold) else len(var1)
                
                var2 = np.cumsum(pca2.explained_variance_ratio_)
                n_comp2 = np.argmax(var2 >= variance_threshold) + 1 if any(var2 >= variance_threshold) else len(var2)
                
                # Get bases
                basis1 = pca1.components_[:n_comp1].T
                basis2 = pca2.components_[:n_comp2].T
                
                # Calculate angles
                angles = principal_angles(basis1, basis2)
                
                # Convert to degrees for easier interpretation
                angles_degrees = np.degrees(angles)
                
                # Store results
                pair_results = {
                    'mean': float(np.mean(angles_degrees)),
                    'std': float(np.std(angles_degrees)),
                    'min': float(np.min(angles_degrees)),
                    'max': float(np.max(angles_degrees)),
                    'median': float(np.median(angles_degrees)),
                    'ci_95': (float(np.nan), float(np.nan))  # No bootstrap for tiny nets
                }
                
                results['pairwise'][(int(class1), int(class2))] = pair_results
                all_angles.extend(angles_degrees)
            
            else:
                # For larger networks, use bootstrap approach
                # Generate bootstrapped subspaces
                bases1 = bootstrap_subspace(class1_activations, n_bootstrap, random_state)
                bases2 = bootstrap_subspace(class2_activations, n_bootstrap, random_state)
                
                if not bases1 or not bases2:
                    results['pairwise'][(int(class1), int(class2))] = {
                        'mean': np.nan, 'std': np.nan, 'min': np.nan, 
                        'max': np.nan, 'median': np.nan, 'ci_95': (np.nan, np.nan)
                    }
                    continue
                
                # Calculate angles for all bootstrap samples
                bootstrap_angles = []
                
                for basis1 in bases1:
                    for basis2 in bases2:
                        angles = principal_angles(basis1, basis2)
                        angles_degrees = np.degrees(angles)
                        bootstrap_angles.extend(angles_degrees)
                
                # Calculate statistics
                bootstrap_angles = np.array(bootstrap_angles)
                
                # 95% confidence interval
                sorted_angles = np.sort(bootstrap_angles)
                lower_ci = sorted_angles[int(0.025 * len(sorted_angles))] if len(sorted_angles) > 0 else np.nan
                upper_ci = sorted_angles[int(0.975 * len(sorted_angles))] if len(sorted_angles) > 0 else np.nan
                
                pair_results = {
                    'mean': float(np.mean(bootstrap_angles)),
                    'std': float(np.std(bootstrap_angles)),
                    'min': float(np.min(bootstrap_angles)),
                    'max': float(np.max(bootstrap_angles)),
                    'median': float(np.median(bootstrap_angles)),
                    'ci_95': (float(lower_ci), float(upper_ci))
                }
                
                results['pairwise'][(int(class1), int(class2))] = pair_results
                all_angles.extend(bootstrap_angles)
    
    # Calculate aggregate statistics across all pairs
    if all_angles:
        all_angles = np.array(all_angles)
        sorted_angles = np.sort(all_angles)
        lower_ci = sorted_angles[int(0.025 * len(sorted_angles))] if len(sorted_angles) > 0 else np.nan
        upper_ci = sorted_angles[int(0.975 * len(sorted_angles))] if len(sorted_angles) > 0 else np.nan
        
        results['mean_angle'] = float(np.mean(all_angles))
        results['std_angle'] = float(np.std(all_angles))
        results['ci_95'] = (float(lower_ci), float(upper_ci))
    else:
        results['mean_angle'] = np.nan
        results['std_angle'] = np.nan
        results['ci_95'] = (np.nan, np.nan)
    
    return results 