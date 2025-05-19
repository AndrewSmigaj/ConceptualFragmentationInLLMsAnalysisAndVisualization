"""
Visualization utilities for neural network activation trajectories.

This module provides functions for visualizing how activations evolve across
layers of a neural network, specifically for the Concept Fragmentation project.
It includes:
- Direct visualization of sample activations across layers for tiny networks
- PCA-compressed trajectory paths for larger networks
- Visualization of class-specific activation patterns

These utilities work with both torch.Tensor and numpy.ndarray inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
import matplotlib.colors as mcolors
import warnings

# Import configuration and utility functions
from ..config import VISUALIZATION, RANDOM_SEED
from .activations import _ensure_numpy, reduce_dimensions


def plot_sample_trajectory(
    activations_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    sample_idx: Union[int, List[int]],
    layer_names: Optional[List[str]] = None,
    title: str = "Activation Trajectory",
    figsize: Tuple[int, int] = VISUALIZATION['plot']['figsize'],
    alpha: float = VISUALIZATION['plot']['alpha'],
    cmap: str = VISUALIZATION['plot']['cmap'],
    line_style: str = '-',
    linewidth: float = 2.0,
    marker: str = 'o',
    markersize: int = 8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot activation values of selected neurons for specific samples across layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        sample_idx: Index or list of indices of samples to visualize
        layer_names: List of layer names to include (None for all)
        title: Plot title
        figsize: Figure size as (width, height)
        alpha: Transparency of lines
        cmap: Colormap
        line_style: Style of trajectory lines
        linewidth: Width of trajectory lines
        marker: Marker style for layer points
        markersize: Size of markers
        save_path: Path to save the figure (None to not save)
        
    Returns:
        Matplotlib Figure object
    """
    # Determine which layers to plot
    if layer_names is None:
        layer_names = list(activations_dict.keys())
    
    # Skip non-existent layers
    layer_names = [layer for layer in layer_names if layer in activations_dict]
    
    # Need at least two layers
    if len(layer_names) < 2:
        raise ValueError("Need at least two layers to plot trajectories")
    
    # Convert sample_idx to list if it's a single integer
    if isinstance(sample_idx, int):
        sample_idx = [sample_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get a colormap
    if len(sample_idx) <= 10:
        colors = sns.color_palette("tab10", len(sample_idx))
    else:
        colors = [plt.cm.get_cmap(cmap)(i / len(sample_idx)) for i in range(len(sample_idx))]
    
    # Check dimensions
    first_layer = layer_names[0]
    act = _ensure_numpy(activations_dict[first_layer])
    
    # For each sample
    for i, idx in enumerate(sample_idx):
        # Check if the sample index is valid
        if idx >= act.shape[0]:
            warnings.warn(f"Sample index {idx} is out of bounds (max: {act.shape[0]-1}). Skipping.")
            continue
        
        color = colors[i % len(colors)]
        
        # For tiny networks (3×3×3), plot direct activations
        if act.shape[1] <= 10:  # Small activation dimension
            # Get activations for this sample across all layers
            layer_acts = []
            for layer in layer_names:
                layer_act = _ensure_numpy(activations_dict[layer])
                layer_acts.append(layer_act[idx])
            
            # Find the maximum number of neurons in any layer
            max_neurons = max(act.shape[1] for act in [_ensure_numpy(activations_dict[layer]) for layer in layer_names])
            
            # Plot activations as heatmap
            layer_acts_padded = []
            for act in layer_acts:
                # Pad with NaNs if needed
                padded = np.full(max_neurons, np.nan)
                padded[:len(act)] = act
                layer_acts_padded.append(padded)
            
            # Convert to array
            layer_acts_array = np.array(layer_acts_padded)
            
            # Create heatmap
            sns.heatmap(
                layer_acts_array,
                ax=ax,
                cmap='viridis',
                xticklabels=[f"Neuron {i+1}" for i in range(max_neurons)],
                yticklabels=layer_names,
                cbar_kws={"label": "Activation Value"}
            )
            
            ax.set_title(f"{title} - Sample {idx}")
        
        else:
            # For larger networks, plot the mean activation value
            mean_acts = []
            for layer in layer_names:
                layer_act = _ensure_numpy(activations_dict[layer])
                mean_acts.append(np.mean(layer_act[idx]))
            
            # Plot mean activation
            ax.plot(
                range(len(layer_names)),
                mean_acts,
                linestyle=line_style,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                color=color,
                alpha=alpha,
                label=f"Sample {idx}"
            )
            
            # Set x-tick labels
            ax.set_xticks(range(len(layer_names)))
            ax.set_xticklabels(layer_names, rotation=45)
            
            ax.set_title(title)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Mean Activation Value")
            
            # Add legend
            ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig


def compute_compressed_paths(
    activations_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    method: str = 'pca',
    n_components: int = 2,
    random_state: int = RANDOM_SEED,
    standardize: bool = True,
    **kwargs
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute compressed representations of activations across layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        method: Dimensionality reduction method ('pca' or 'umap')
        n_components: Number of components to reduce to
        random_state: Random seed for reproducibility
        standardize: Whether to standardize the data before reduction
        **kwargs: Additional arguments passed to reduce_dimensions
    
    Returns:
        Dictionary mapping layer names to dictionaries containing:
            'reduced': Reduced activations as numpy.ndarray
            'reducer': PCA or UMAP object used for reduction
    """
    result = {}
    
    for layer_name, activations in activations_dict.items():
        act = _ensure_numpy(activations)
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            # Create and fit PCA
            reducer = PCA(n_components=n_components, random_state=random_state)
            
            # Standardize if requested
            if standardize:
                scaler = StandardScaler()
                act = scaler.fit_transform(act)
            
            # Fit and transform
            reduced_act = reducer.fit_transform(act)
            
            # Store result
            result[layer_name] = {
                'reduced': reduced_act,
                'reducer': reducer,
                'scaler': scaler if standardize else None
            }
        
        elif method.lower() == 'umap':
            # Import UMAP if available
            try:
                import umap
                
                # Set default UMAP parameters
                umap_kwargs = {
                    'n_neighbors': VISUALIZATION['umap']['n_neighbors'],
                    'min_dist': VISUALIZATION['umap']['min_dist'],
                    'metric': VISUALIZATION['umap']['metric'],
                    'random_state': random_state
                }
                # Update with user-provided kwargs
                umap_kwargs.update(kwargs)
                
                # Standardize if requested
                if standardize:
                    scaler = StandardScaler()
                    act = scaler.fit_transform(act)
                
                # Create and fit UMAP
                reducer = umap.UMAP(n_components=n_components, **umap_kwargs)
                reduced_act = reducer.fit_transform(act)
                
                # Store result
                result[layer_name] = {
                    'reduced': reduced_act,
                    'reducer': reducer,
                    'scaler': scaler if standardize else None
                }
                
            except ImportError:
                raise ImportError("UMAP is not available. Install it with 'pip install umap-learn'")
        
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
    
    return result


def plot_activation_flow(
    activations_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    highlight_samples: Optional[List[int]] = None,
    layer_names: Optional[List[str]] = None,
    method: str = 'pca',
    n_components: int = 2,
    title: str = "Activation Flow Across Layers",
    figsize: Optional[Tuple[int, int]] = None,
    alpha_bg: float = 0.1,
    alpha_fg: float = 0.8,
    linewidth_bg: float = 0.5,
    linewidth_fg: float = 2.0,
    markersize: int = 8,
    n_samples_bg: Optional[int] = 100,
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Visualize the flow of activations across network layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        labels: Class labels corresponding to activations
        highlight_samples: List of sample indices to highlight (None for random)
        layer_names: List of layer names to include (None for all)
        method: Dimensionality reduction method ('pca' or 'umap')
        n_components: Number of components to reduce to
        title: Plot title
        figsize: Figure size as (width, height)
        alpha_bg: Transparency of background lines
        alpha_fg: Transparency of highlighted lines
        linewidth_bg: Width of background lines
        linewidth_fg: Width of highlighted lines
        markersize: Size of markers
        n_samples_bg: Number of background samples to show (None for all)
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
    
    # Need at least two layers
    if len(layer_names) < 2:
        raise ValueError("Need at least two layers to plot trajectories")
    
    # Convert to numpy if needed
    labels = _ensure_numpy(labels)
    
    # Determine figure size
    if figsize is None:
        figsize = (max(10, len(layer_names) * 3), 8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique classes and color mapping
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Create custom color palette for consistent colors
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    else:
        palette = sns.color_palette("hsv", n_classes)
    
    # Compute compressed representations for each layer
    compressed = compute_compressed_paths(
        {layer: activations_dict[layer] for layer in layer_names},
        method=method,
        n_components=n_components,
        **kwargs
    )
    
    # Get total number of samples
    n_samples = _ensure_numpy(activations_dict[layer_names[0]]).shape[0]
    
    # Get samples to highlight if not provided
    if highlight_samples is None:
        # Select a few random samples from each class
        highlight_samples = []
        np.random.seed(RANDOM_SEED)
        samples_per_class = 3
        
        for class_idx in unique_classes:
            class_mask = (labels == class_idx)
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) <= samples_per_class:
                highlight_samples.extend(class_indices.tolist())
            else:
                highlight_samples.extend(np.random.choice(class_indices, samples_per_class, replace=False).tolist())
    
    # Set background sample indices
    if n_samples_bg is not None and n_samples_bg < n_samples:
        # Get indices not in highlight_samples
        all_indices = set(range(n_samples))
        highlight_set = set(highlight_samples)
        bg_candidates = list(all_indices - highlight_set)
        
        # Randomly select background samples
        np.random.seed(RANDOM_SEED)
        bg_indices = list(np.random.choice(bg_candidates, min(n_samples_bg, len(bg_candidates)), replace=False))
    else:
        # Use all non-highlighted samples
        all_indices = set(range(n_samples))
        highlight_set = set(highlight_samples)
        bg_indices = list(all_indices - highlight_set)
    
    # Set x-coordinates for each layer
    layer_x = np.linspace(0, 1, len(layer_names))
    
    # First, plot background samples
    y_min, y_max = float('inf'), float('-inf')
    
    for idx in bg_indices:
        class_idx = labels[idx]
        color = palette[int(np.where(unique_classes == class_idx)[0][0])]
        
        # Get path coordinates
        y_coords = []
        for i, layer in enumerate(layer_names):
            reduced_act = compressed[layer]['reduced']
            # Use first component as y-coordinate
            y_coords.append(reduced_act[idx, 0])
            
            # Update y limits
            y_min = min(y_min, reduced_act[idx, 0])
            y_max = max(y_max, reduced_act[idx, 0])
        
        # Plot path
        ax.plot(
            layer_x,
            y_coords,
            color=color,
            alpha=alpha_bg,
            linewidth=linewidth_bg,
            marker=None
        )
    
    # Then, plot highlighted samples
    legend_handles = []
    
    for idx in highlight_samples:
        if idx >= n_samples:
            warnings.warn(f"Sample index {idx} is out of bounds (max: {n_samples-1}). Skipping.")
            continue
            
        class_idx = labels[idx]
        color = palette[int(np.where(unique_classes == class_idx)[0][0])]
        
        # Get path coordinates
        y_coords = []
        for i, layer in enumerate(layer_names):
            reduced_act = compressed[layer]['reduced']
            # Use first component as y-coordinate
            y_coords.append(reduced_act[idx, 0])
            
            # Update y limits
            y_min = min(y_min, reduced_act[idx, 0])
            y_max = max(y_max, reduced_act[idx, 0])
        
        # Plot path
        ax.plot(
            layer_x,
            y_coords,
            color=color,
            alpha=alpha_fg,
            linewidth=linewidth_fg,
            marker='o',
            markersize=markersize,
            label=f"Sample {idx} (Class {class_idx})"
        )
    
    # Create legend entries for classes
    for i, class_idx in enumerate(unique_classes):
        legend_handles.append(
            mpatches.Patch(color=palette[i], label=f"Class {class_idx}")
        )
    
    # Add vertical lines for each layer
    for i, (x, layer) in enumerate(zip(layer_x, layer_names)):
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
        
        # Add text labels
        ax.text(x, y_max + 0.05 * (y_max - y_min), layer, 
                ha='center', va='bottom', rotation=45)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"First {method.upper()} Component")
    
    # Remove x-ticks (we have custom layer labels)
    ax.set_xticks([])
    
    # Set y-limits with some padding
    padding = 0.1 * (y_max - y_min)
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Set x-limits with some padding
    ax.set_xlim(-0.05, 1.05)
    
    # Add legend
    ax.legend(handles=legend_handles, loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig


def plot_class_trajectories(
    activations_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    labels: Union[torch.Tensor, np.ndarray, List[int]],
    layer_names: Optional[List[str]] = None,
    class_indices: Optional[List[int]] = None,
    method: str = 'pca',
    title: str = "Class Activation Trajectories",
    figsize: Tuple[int, int] = VISUALIZATION['plot']['figsize'],
    alpha: float = VISUALIZATION['plot']['alpha'] + 0.2,  # Slightly higher alpha for class means
    line_style: str = '-',
    linewidth: float = 3.0,
    marker: str = 'o',
    markersize: int = 10,
    error_bars: bool = True,
    error_alpha: float = 0.2,
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot mean activation trajectories for each class across layers.
    
    Args:
        activations_dict: Dictionary mapping layer names to activation tensors
        labels: Class labels corresponding to activations
        layer_names: List of layer names to include (None for all)
        class_indices: List of class indices to include (None for all)
        method: Method for trajectory y-values: 'pca', 'mean', or 'norm'
        title: Plot title
        figsize: Figure size as (width, height)
        alpha: Transparency of lines
        line_style: Style of trajectory lines
        linewidth: Width of trajectory lines
        marker: Marker style for layer points
        markersize: Size of markers
        error_bars: Whether to show class activation standard deviations
        error_alpha: Transparency of error regions
        save_path: Path to save the figure (None to not save)
        **kwargs: Additional arguments passed to compute_compressed_paths
        
    Returns:
        Matplotlib Figure object
    """
    # Determine which layers to plot
    if layer_names is None:
        layer_names = list(activations_dict.keys())
    
    # Skip non-existent layers
    layer_names = [layer for layer in layer_names if layer in activations_dict]
    
    # Need at least two layers
    if len(layer_names) < 2:
        raise ValueError("Need at least two layers to plot trajectories")
    
    # Convert to numpy if needed
    labels = _ensure_numpy(labels)
    
    # Get unique classes
    unique_classes = np.unique(labels)
    
    # Filter classes if needed
    if class_indices is not None:
        unique_classes = np.array([c for c in unique_classes if c in class_indices])
    
    n_classes = len(unique_classes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom color palette for consistent colors
    if n_classes <= 10:
        palette = sns.color_palette("tab10", n_classes)
    else:
        palette = sns.color_palette("hsv", n_classes)
    
    # Set x-coordinates for each layer
    layer_x = np.linspace(0, 1, len(layer_names))
    
    if method.lower() == 'pca':
        # Compute compressed representations for each layer
        compressed = compute_compressed_paths(
            {layer: activations_dict[layer] for layer in layer_names},
            method='pca',
            n_components=2,
            **kwargs
        )
        
        # Plot trajectories for each class
        for i, class_idx in enumerate(unique_classes):
            class_mask = (labels == class_idx)
            color = palette[i % len(palette)]
            
            # Get mean and std for each layer
            means = []
            stds = []
            
            for layer in layer_names:
                reduced_act = compressed[layer]['reduced']
                class_act = reduced_act[class_mask]
                
                means.append(np.mean(class_act[:, 0]))  # First PCA component mean
                stds.append(np.std(class_act[:, 0]))    # First PCA component std
            
            # Plot mean trajectory
            ax.plot(
                layer_x,
                means,
                color=color,
                linestyle=line_style,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                label=f"Class {class_idx}"
            )
            
            # Plot error region if requested
            if error_bars:
                ax.fill_between(
                    layer_x,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=color,
                    alpha=error_alpha
                )
    
    elif method.lower() == 'mean':
        # Plot mean activation values for each class across layers
        for i, class_idx in enumerate(unique_classes):
            class_mask = (labels == class_idx)
            color = palette[i % len(palette)]
            
            # Get mean and std for each layer
            means = []
            stds = []
            
            for layer in layer_names:
                act = _ensure_numpy(activations_dict[layer])
                class_act = act[class_mask]
                
                means.append(np.mean(class_act))  # Mean of all activations
                stds.append(np.std(class_act))    # Std of all activations
            
            # Plot mean trajectory
            ax.plot(
                layer_x,
                means,
                color=color,
                linestyle=line_style,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                label=f"Class {class_idx}"
            )
            
            # Plot error region if requested
            if error_bars:
                ax.fill_between(
                    layer_x,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=color,
                    alpha=error_alpha
                )
    
    elif method.lower() == 'norm':
        # Plot L2 norm of activations for each class across layers
        for i, class_idx in enumerate(unique_classes):
            class_mask = (labels == class_idx)
            color = palette[i % len(palette)]
            
            # Get mean and std of L2 norms for each layer
            means = []
            stds = []
            
            for layer in layer_names:
                act = _ensure_numpy(activations_dict[layer])
                class_act = act[class_mask]
                
                # Compute L2 norm for each sample
                norms = np.linalg.norm(class_act, axis=1)
                
                means.append(np.mean(norms))  # Mean L2 norm
                stds.append(np.std(norms))    # Std of L2 norms
            
            # Plot mean trajectory
            ax.plot(
                layer_x,
                means,
                color=color,
                linestyle=line_style,
                linewidth=linewidth,
                marker=marker,
                markersize=markersize,
                alpha=alpha,
                label=f"Class {class_idx}"
            )
            
            # Plot error region if requested
            if error_bars:
                ax.fill_between(
                    layer_x,
                    np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=color,
                    alpha=error_alpha
                )
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'pca', 'mean', or 'norm'")
    
    # Add vertical lines for each layer
    for i, (x, layer) in enumerate(zip(layer_x, layer_names)):
        ax.axvline(x=x, color='gray', linestyle='--', alpha=0.3)
    
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("Layer")
    
    if method.lower() == 'pca':
        ax.set_ylabel("First PCA Component")
    elif method.lower() == 'mean':
        ax.set_ylabel("Mean Activation")
    elif method.lower() == 'norm':
        ax.set_ylabel("Mean L2 Norm")
    
    # Set custom x-ticks at layer positions
    ax.set_xticks(layer_x)
    ax.set_xticklabels(layer_names, rotation=45)
    
    # Add legend
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=VISUALIZATION['plot']['dpi'], bbox_inches='tight')
    
    return fig
