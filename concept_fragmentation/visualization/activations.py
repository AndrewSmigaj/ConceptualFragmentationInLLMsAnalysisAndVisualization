"""
Visualization utilities for neural network activations.

This module provides functions for visualizing activations from neural networks,
specifically for the Concept Fragmentation project. It includes:
- 2D/3D scatter plots of activations using PCA and UMAP
- Visualization of cluster assignments from k-means
- Layer comparison visualizations

These utilities work with both torch.Tensor and numpy.ndarray inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Union, Tuple, Optional, Any
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. UMAP visualizations will not be available.")

# Import configuration
from ..config import VISUALIZATION, RANDOM_SEED


def _ensure_numpy(data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert torch.Tensor to numpy.ndarray if needed.
    
    Args:
        data: Input data as torch.Tensor or numpy.ndarray
        
    Returns:
        Data as numpy.ndarray
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data


def reduce_dimensions(
    activations: Union[torch.Tensor, np.ndarray],
    method: str = 'pca',
    n_components: int = 2,
    random_state: int = RANDOM_SEED,
    standardize: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Reduce dimensions of activations using PCA or UMAP.
    
    Args:
        activations: Activation data as torch.Tensor or numpy.ndarray
        method: Dimensionality reduction method ('pca' or 'umap')
        n_components: Number of components to reduce to (2 or 3 recommended for visualization)
        random_state: Random seed for reproducibility
        standardize: Whether to standardize the data before reduction
        **kwargs: Additional arguments passed to the dimensionality reduction method
    
    Returns:
        Reduced activations as numpy.ndarray of shape (n_samples, n_components)
    """
    # Convert to numpy if needed
    act = _ensure_numpy(activations)
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        act = scaler.fit_transform(act)
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state, **kwargs)
        reduced_act = reducer.fit_transform(act)
        explained_variance = reducer.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_variance:.2%}")
        
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not available. Install it with 'pip install umap-learn'")
        
        # Set default UMAP parameters from config if not provided
        umap_kwargs = {
            'n_neighbors': VISUALIZATION['umap']['n_neighbors'],
            'min_dist': VISUALIZATION['umap']['min_dist'],
            'metric': VISUALIZATION['umap']['metric'],
            'random_state': random_state
        }
        # Update with user-provided kwargs
        umap_kwargs.update(kwargs)
        
        reducer = umap.UMAP(n_components=n_components, **umap_kwargs)
        reduced_act = reducer.fit_transform(act)
        
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return reduced_act


def plot_activations_2d(
    activations: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    title: str = "Activation Visualization",
    layer_name: Optional[str] = None,
    method: str = 'pca',
    figsize: Tuple[int, int] = VISUALIZATION['plot']['figsize'],
    markersize: int = VISUALIZATION['plot']['markersize'],
    alpha: float = VISUALIZATION['plot']['alpha'],
    cmap: str = VISUALIZATION['plot']['cmap'],
    cluster_labels: Optional[Union[Dict[int, np.ndarray], np.ndarray]] = None,
    save_path: Optional[str] = None,
    show_legend: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Create a 2D scatter plot of activations colored by class labels.
    
    Args:
        activations: Activation data as torch.Tensor, numpy.ndarray, or dict of layer activations
        labels: Class labels corresponding to activations
        title: Plot title
        layer_name: Layer name to use if activations is a dict
        method: Dimensionality reduction method ('pca' or 'umap')
        figsize: Figure size as (width, height)
        markersize: Size of markers in scatter plot
        alpha: Transparency of markers
        cmap: Colormap for class labels
        cluster_labels: Optional cluster labels from k-means
        save_path: Path to save the figure (None to not save)
        show_legend: Whether to show the legend
        **kwargs: Additional arguments passed to reduce_dimensions
        
    Returns:
        Matplotlib Figure object
    """
    # Extract activations from dict if needed
    if isinstance(activations, dict):
        if layer_name is None:
            raise ValueError("layer_name must be provided when activations is a dictionary")
        act = activations[layer_name]
    else:
        act = activations
    
    # Convert to numpy if needed
    act = _ensure_numpy(act)
    labels = _ensure_numpy(labels)
    
    # Reduce dimensions
    reduced_act = reduce_dimensions(act, method=method, n_components=2, **kwargs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique classes and color mapping
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Create custom color palette for consistent colors
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    else:
        palette = sns.color_palette("hsv", n_classes)
    
    # If we have cluster labels, we'll use different markers for different clusters
    if cluster_labels is not None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        
        # Handle different formats of cluster_labels
        if isinstance(cluster_labels, dict):
            # This format comes from cluster_entropy with return_clusters=True
            for class_idx in unique_classes:
                class_mask = (labels == class_idx)
                class_activations = reduced_act[class_mask]
                
                if int(class_idx) in cluster_labels:
                    # Get cluster assignments for this class
                    clusters = cluster_labels[int(class_idx)]
                    
                    # For each cluster in this class
                    for cluster_idx in np.unique(clusters):
                        cluster_mask = (clusters == cluster_idx)
                        
                        # Plot points in this cluster
                        ax.scatter(
                            class_activations[cluster_mask, 0],
                            class_activations[cluster_mask, 1],
                            s=markersize,
                            color=palette[int(class_idx % n_classes)],
                            marker=markers[int(cluster_idx) % len(markers)],
                            alpha=alpha,
                            label=f"Class {class_idx}, Cluster {cluster_idx}" if cluster_idx == 0 else None
                        )
        else:
            # Assume cluster_labels is an array parallel to activations
            cluster_labels = _ensure_numpy(cluster_labels)
            unique_clusters = np.unique(cluster_labels)
            
            for class_idx in unique_classes:
                class_mask = (labels == class_idx)
                
                for cluster_idx in unique_clusters:
                    mask = class_mask & (cluster_labels == cluster_idx)
                    if np.any(mask):
                        ax.scatter(
                            reduced_act[mask, 0],
                            reduced_act[mask, 1],
                            s=markersize,
                            color=palette[int(class_idx % n_classes)],
                            marker=markers[int(cluster_idx) % len(markers)],
                            alpha=alpha,
                            label=f"Class {class_idx}, Cluster {cluster_idx}" if cluster_idx == 0 else None
                        )
    else:
        # Simple plot colored by class
        for i, class_idx in enumerate(unique_classes):
            mask = (labels == class_idx)
            ax.scatter(
                reduced_act[mask, 0],
                reduced_act[mask, 1],
                s=markersize,
                color=palette[i % n_classes],
                alpha=alpha,
                label=f"Class {class_idx}"
            )
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    
    # Add legend if requested
    if show_legend:
        plt.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig


def plot_activations_3d(
    activations: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    title: str = "3D Activation Visualization",
    layer_name: Optional[str] = None,
    method: str = 'pca',
    figsize: Tuple[int, int] = VISUALIZATION['plot']['figsize'],
    markersize: int = VISUALIZATION['plot']['markersize'],
    alpha: float = VISUALIZATION['plot']['alpha'],
    cmap: str = VISUALIZATION['plot']['cmap'],
    save_path: Optional[str] = None,
    show_legend: bool = True,
    elev: int = 30,
    azim: int = 45,
    **kwargs
) -> plt.Figure:
    """
    Create a 3D scatter plot of activations colored by class labels.
    
    Args:
        activations: Activation data as torch.Tensor, numpy.ndarray, or dict of layer activations
        labels: Class labels corresponding to activations
        title: Plot title
        layer_name: Layer name to use if activations is a dict
        method: Dimensionality reduction method ('pca' or 'umap')
        figsize: Figure size as (width, height)
        markersize: Size of markers in scatter plot
        alpha: Transparency of markers
        cmap: Colormap for class labels
        save_path: Path to save the figure (None to not save)
        show_legend: Whether to show the legend
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        **kwargs: Additional arguments passed to reduce_dimensions
        
    Returns:
        Matplotlib Figure object
    """
    # Extract activations from dict if needed
    if isinstance(activations, dict):
        if layer_name is None:
            raise ValueError("layer_name must be provided when activations is a dictionary")
        act = activations[layer_name]
    else:
        act = activations
    
    # Convert to numpy if needed
    act = _ensure_numpy(act)
    labels = _ensure_numpy(labels)
    
    # Reduce dimensions
    reduced_act = reduce_dimensions(act, method=method, n_components=3, **kwargs)
    
    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique classes and color mapping
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Create custom color palette for consistent colors
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    else:
        palette = sns.color_palette("hsv", n_classes)
    
    # Plot each class
    for i, class_idx in enumerate(unique_classes):
        mask = (labels == class_idx)
        ax.scatter(
            reduced_act[mask, 0],
            reduced_act[mask, 1],
            reduced_act[mask, 2],
            s=markersize,
            color=palette[i % n_classes],
            alpha=alpha,
            label=f"Class {class_idx}"
        )
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Add title and labels
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_zlabel(f"{method.upper()} Component 3")
    
    # Add legend if requested
    if show_legend:
        plt.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig


def plot_layer_comparison(
    activations_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    layer_names: Optional[List[str]] = None,
    method: str = 'pca',
    figsize: Optional[Tuple[int, int]] = None,
    markersize: int = VISUALIZATION['plot']['markersize'] // 2,  # Smaller markers for grid
    alpha: float = VISUALIZATION['plot']['alpha'],
    cmap: str = VISUALIZATION['plot']['cmap'],
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a grid of scatter plots comparing activations across different layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        labels: Class labels corresponding to activations
        layer_names: List of layer names to include (None for all)
        method: Dimensionality reduction method ('pca' or 'umap')
        figsize: Figure size as (width, height)
        markersize: Size of markers in scatter plot
        alpha: Transparency of markers
        cmap: Colormap for class labels
        save_path: Path to save the figure (None to not save)
        **kwargs: Additional arguments passed to reduce_dimensions
        
    Returns:
        Matplotlib Figure object
    """
    # Determine which layers to plot
    if layer_names is None:
        layer_names = list(activations_dict.keys())
    
    # Skip non-existent layers
    layer_names = [layer for layer in layer_names if layer in activations_dict]
    
    # Need at least one layer
    if not layer_names:
        raise ValueError("No valid layers to plot")
    
    # Convert labels to numpy if needed
    labels = _ensure_numpy(labels)
    
    # Determine grid size
    n_layers = len(layer_names)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    # Determine figure size if not provided
    if figsize is None:
        figsize = (n_cols * 4, n_rows * 4)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    
    # Get unique classes and color mapping
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Create custom color palette for consistent colors
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    else:
        palette = sns.color_palette("hsv", n_classes)
    
    # Plot each layer
    for i, layer_name in enumerate(layer_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Reduce dimensions for this layer
        act = activations_dict[layer_name]
        act = _ensure_numpy(act)
        reduced_act = reduce_dimensions(act, method=method, n_components=2, **kwargs)
        
        # Plot each class
        for j, class_idx in enumerate(unique_classes):
            mask = (labels == class_idx)
            ax.scatter(
                reduced_act[mask, 0],
                reduced_act[mask, 1],
                s=markersize,
                color=palette[j % n_classes],
                alpha=alpha,
                label=f"Class {class_idx}" if i == 0 else None  # Only add label in first plot
            )
        
        # Add title and labels
        ax.set_title(f"Layer: {layer_name}")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
    
    # Remove unused subplots
    for i in range(n_layers, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    # Add legend to the first subplot
    if n_layers > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(5, n_classes))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the legend
    
    # Add overall title
    fig.suptitle(f"Layer Comparison using {method.upper()}", fontsize=16)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig


def plot_topk_neuron_activations(
    activations: Union[torch.Tensor, np.ndarray, Dict[str, Union[torch.Tensor, np.ndarray]]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    k: int = 3,
    layer_name: Optional[str] = None,
    figsize: Tuple[int, int] = VISUALIZATION['plot']['figsize'],
    markersize: int = VISUALIZATION['plot']['markersize'],
    alpha: float = VISUALIZATION['plot']['alpha'],
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a scatter plot matrix of the top-k neurons with highest activation variance.
    
    Args:
        activations: Activation data as torch.Tensor, numpy.ndarray, or dict of layer activations
        labels: Class labels corresponding to activations
        k: Number of top neurons to visualize
        layer_name: Layer name to use if activations is a dict
        figsize: Figure size as (width, height)
        markersize: Size of markers in scatter plot
        alpha: Transparency of markers
        save_path: Path to save the figure (None to not save)
        **kwargs: Additional arguments for plotting
        
    Returns:
        Matplotlib Figure object
    """
    # Extract activations from dict if needed
    if isinstance(activations, dict):
        if layer_name is None:
            raise ValueError("layer_name must be provided when activations is a dictionary")
        act = activations[layer_name]
    else:
        act = activations
    
    # Convert to numpy if needed
    act = _ensure_numpy(act)
    labels = _ensure_numpy(labels)
    
    # Find top-k neurons by variance
    variances = np.var(act, axis=0)
    top_k_indices = np.argsort(variances)[-k:][::-1]  # Indices of top-k neurons
    
    # Extract activations for top-k neurons
    top_k_act = act[:, top_k_indices]
    
    # Create scatter plot matrix
    fig = plt.figure(figsize=figsize)
    
    # Get unique classes and color mapping
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Create custom color palette for consistent colors
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    else:
        palette = sns.color_palette("hsv", n_classes)
    
    # Create a scatter plot matrix using a nested loop
    for i in range(k):
        for j in range(k):
            # Skip if i == j (would be histogram, not scatter)
            if i == j:
                continue
                
            # Add subplot
            ax = fig.add_subplot(k, k, i*k + j + 1)
            
            # Plot each class
            for c, class_idx in enumerate(unique_classes):
                mask = (labels == class_idx)
                ax.scatter(
                    top_k_act[mask, j],
                    top_k_act[mask, i],
                    s=markersize // 2,  # Smaller markers for grid
                    color=palette[c % n_classes],
                    alpha=alpha,
                    label=f"Class {class_idx}" if i == 0 and j == 1 else None
                )
            
            # Only add labels for border plots
            if i == k-1:
                ax.set_xlabel(f"Neuron {top_k_indices[j]}")
            if j == 0:
                ax.set_ylabel(f"Neuron {top_k_indices[i]}")
            
            # Remove ticks for cleaner look
            ax.tick_params(labelbottom=False, labelleft=False)
    
    # Add overall title
    layer_str = f" in {layer_name}" if layer_name else ""
    plt.suptitle(f"Top {k} neurons by variance{layer_str}", fontsize=16)
    
    # Add a legend
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=min(5, n_classes))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the title
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig
