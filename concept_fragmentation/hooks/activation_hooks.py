"""
Activation hooks module for neural networks.

This module provides tools for registering forward hooks on PyTorch models
to capture and store layer activations. It focuses on memory-efficient
storage and standardized output formats.

Features:
- Context manager for automatic hook registration/removal
- Dictionary-based activation storage keyed by layer names
- Support for filtering layers by name or pattern
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Union, Any, Set
import numpy as np
import contextlib
import warnings

# Import random seed from config
from ..config import RANDOM_SEED


class ActivationHook:
    """
    Manages activation hooks for a PyTorch model.
    
    Attributes:
        model: The PyTorch model to attach hooks to
        layer_names: Names of layers to hook (None for all)
        include_patterns: List of patterns to match layer names (None for all)
        exclude_patterns: List of patterns to exclude from layer names (None for none)
        device: Device to store activations on ('cpu' or 'cuda')
        activations: Dictionary storing layer activations
        handles: List of hook handles for cleanup
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the activation hook.
        
        Args:
            model: PyTorch model to attach hooks to
            layer_names: Specific layer names to track (None for all named modules)
            include_patterns: List of patterns to match layer names (None for all)
            exclude_patterns: List of patterns to exclude from layer names (None for none)
            device: Device to store activations ('cpu' or 'cuda')
        """
        self.model = model
        self.layer_names = layer_names
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.device = device
        self.activations = {}
        self.handles = []
        
        # Validate device
        if device != 'cpu' and not (device.startswith('cuda') and torch.cuda.is_available()):
            warnings.warn(f"Device {device} not available, falling back to CPU")
            self.device = 'cpu'
    
    def _layer_filter(self, name: str) -> bool:
        """Check if a layer should be hooked based on filters."""
        # Skip the model itself
        if name == '':
            return False
            
        # Skip final classification layer unless explicitly requested
        if name == 'output' and self.layer_names is None and self.include_patterns is None:
            return False
            
        # If specific layer names provided, only include those
        if self.layer_names is not None:
            return name in self.layer_names
        
        # Apply include patterns
        if self.include_patterns is not None:
            if not any(pattern in name for pattern in self.include_patterns):
                return False
        
        # Apply exclude patterns
        if self.exclude_patterns is not None:
            if any(pattern in name for pattern in self.exclude_patterns):
                return False
        
        return True
    
    def _hook_fn(self, name: str) -> Callable:
        """Create a forward hook function for the given layer name."""
        def hook(module, input, output):
            # Convert output to tensor if needed
            if isinstance(output, tuple):
                output = output[0]
            
            # Move to device and convert to float for consistent storage
            activation = output.to(self.device).float()
            
            # Store activation
            self.activations[name] = activation.detach()
        
        return hook
    
    def register_hooks(self) -> None:
        """Register forward hooks on the model."""
        # Clear any existing activations and handles
        self.activations = {}
        self.remove_hooks()
        
        # Register hooks for each named module
        for name, module in self.model.named_modules():
            if self._layer_filter(name):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def clear_activations(self) -> None:
        """Clear stored activations."""
        self.activations = {}
    
    @property
    def layer_activations(self) -> Dict[str, torch.Tensor]:
        """Get the stored activations."""
        return self.activations
    
    def numpy_activations(self) -> Dict[str, np.ndarray]:
        """
        Convert all stored activations to NumPy arrays.
        
        Returns:
            Dictionary mapping layer names to activation arrays
        """
        return {name: tensor.detach().cpu().numpy() for name, tensor in self.activations.items()}


@contextlib.contextmanager
def capture_activations(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Context manager to capture activations from a model's forward pass.
    
    Usage:
    ```
    with capture_activations(model, layer_names=['layer1', 'layer2']) as activations:
        output = model(input_data)
    # activations now contains the layer outputs
    ```
    
    Args:
        model: PyTorch model to capture activations from
        layer_names: List of specific layer names to capture (None for all)
        include_patterns: List of patterns to match layer names (None for all)
        exclude_patterns: List of patterns to exclude from layer names (None for none)
        device: Device to store activations ('cpu' or 'cuda')
        
    Returns:
        Dictionary mapping layer names to activation tensors
    """
    hook = ActivationHook(
        model=model,
        layer_names=layer_names,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        device=device
    )
    
    hook.register_hooks()
    try:
        yield hook.activations
    finally:
        hook.remove_hooks()


def get_activation_hooks(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    device: str = 'cpu'
) -> ActivationHook:
    """
    Create and register activation hooks for a model.
    
    Args:
        model: PyTorch model to attach hooks to
        layer_names: Specific layer names to track (None for all named modules)
        include_patterns: List of patterns to match layer names (None for all)
        exclude_patterns: List of patterns to exclude from layer names (None for none)
        device: Device to store activations ('cpu' or 'cuda')
        
    Returns:
        ActivationHook object with registered hooks
    """
    hook = ActivationHook(
        model=model,
        layer_names=layer_names,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        device=device
    )
    hook.register_hooks()
    return hook


def get_all_named_layers(model: nn.Module) -> List[str]:
    """
    Get a list of all named layers in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of layer names
    """
    return [name for name, _ in model.named_modules()]


def get_activation_statistics(
    activations: Dict[str, torch.Tensor],
    percentiles: List[float] = [0, 25, 50, 75, 100]
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for layer activations.
    
    Args:
        activations: Dictionary of layer activations
        percentiles: List of percentiles to compute
        
    Returns:
        Dictionary mapping layer names to activation statistics
    """
    stats = {}
    
    for name, activation in activations.items():
        # Convert to numpy for easier statistics computation
        if isinstance(activation, torch.Tensor):
            act_np = activation.detach().cpu().numpy()
        else:
            act_np = activation
        
        # Compute basic statistics
        layer_stats = {
            'mean': float(np.mean(act_np)),
            'std': float(np.std(act_np)),
            'min': float(np.min(act_np)),
            'max': float(np.max(act_np)),
            'sparsity': float((act_np == 0).sum() / act_np.size),
        }
        
        # Add percentiles
        for p in percentiles:
            layer_stats[f'p{p}'] = float(np.percentile(act_np, p))
        
        stats[name] = layer_stats
    
    return stats


def get_neuron_importance(
    activations: Dict[str, torch.Tensor],
    method: str = 'mean_abs',
) -> Dict[str, np.ndarray]:
    """
    Calculate importance scores for each neuron in each layer.
    
    Parameters:
    -----------
    activations : Dict[str, torch.Tensor]
        Dictionary of layer activations
    method : str, default='mean_abs'
        Method to calculate importance:
        - 'mean_abs': Mean of absolute activations
        - 'max_abs': Maximum absolute activation
        - 'variance': Variance of activation
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary mapping layer names to neuron importance scores
    """
    importance = {}
    
    for layer_name, activation in activations.items():
        # Convert to numpy if needed
        if isinstance(activation, torch.Tensor):
            activation = activation.detach().cpu().numpy()
        
        # Reshape if needed
        if len(activation.shape) > 2:  # For conv layers, reshape
            batch_size = activation.shape[0]
            activation = activation.reshape(batch_size, -1)
        
        # Calculate importance based on method
        if method == 'mean_abs':
            scores = np.mean(np.abs(activation), axis=0)
        elif method == 'max_abs':
            scores = np.max(np.abs(activation), axis=0)
        elif method == 'variance':
            scores = np.var(activation, axis=0)
        else:
            raise ValueError(f"Unknown importance method: {method}")
        
        importance[layer_name] = scores
    
    return importance
