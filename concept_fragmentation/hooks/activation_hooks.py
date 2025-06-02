"""
Activation hooks module for neural networks.

This module provides tools for registering forward hooks on PyTorch models
to capture and store layer activations. It focuses on memory-efficient
storage and standardized output formats.

Features:
- Context manager for automatic hook registration/removal
- Dictionary-based activation storage keyed by layer names
- Support for filtering layers by name or pattern
- Dimension tracking with diagnostic logging
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Union, Any, Set
import numpy as np
import contextlib
import warnings
import logging

# Import from config
from ..config import RANDOM_SEED, LOG_LEVEL

# Setup module-level logger
logger = logging.getLogger(__name__)

# Global flags to control activation behavior
ACTIVATION_DIMENSION_LOGGING = False
DIMENSION_MISMATCH_STRATEGY = 'warn'  # Options: 'warn', 'error', 'truncate', 'pad'

def set_dimension_logging(enabled: bool = True) -> None:
    """
    Enable or disable detailed activation dimension logging.
    
    Args:
        enabled: Whether to enable detailed dimension logging
    """
    global ACTIVATION_DIMENSION_LOGGING
    ACTIVATION_DIMENSION_LOGGING = enabled
    logger.info(f"Activation dimension logging {'enabled' if enabled else 'disabled'}")
    
def get_dimension_logging() -> bool:
    """
    Get the current state of dimension logging.
    
    Returns:
        Whether detailed dimension logging is enabled
    """
    return ACTIVATION_DIMENSION_LOGGING


def set_dimension_mismatch_strategy(strategy: str) -> None:
    """
    Set the strategy for handling dimension mismatches between train and test activations.
    
    Args:
        strategy: Strategy to use ('warn', 'error', 'truncate', 'pad')
    """
    global DIMENSION_MISMATCH_STRATEGY
    valid_strategies = ['warn', 'error', 'truncate', 'pad']
    
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid dimension mismatch strategy: {strategy}. "
                        f"Valid options are: {valid_strategies}")
    
    DIMENSION_MISMATCH_STRATEGY = strategy
    logger.info(f"Dimension mismatch strategy set to: {strategy}")


def get_dimension_mismatch_strategy() -> str:
    """
    Get the current strategy for handling dimension mismatches.
    
    Returns:
        Current dimension mismatch strategy
    """
    return DIMENSION_MISMATCH_STRATEGY


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
            
            # Log dimensions if enabled
            if ACTIVATION_DIMENSION_LOGGING:
                # Log input dimensions
                if isinstance(input, tuple) and len(input) > 0:
                    input_shape = tuple(input[0].shape) if isinstance(input[0], torch.Tensor) else "unknown"
                    logger.info(f"Layer {name}: Input shape = {input_shape}")
                
                # Log output dimensions 
                output_shape = tuple(activation.shape)
                logger.info(f"Layer {name}: Output shape = {output_shape}, dtype = {activation.dtype}")
                
                # Log module info
                module_info = f"Module type: {type(module).__name__}"
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    module_info += f", in_features: {module.in_features}, out_features: {module.out_features}"
                elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                    module_info += f", in_channels: {module.in_channels}, out_channels: {module.out_channels}"
                logger.info(f"Layer {name}: {module_info}")
            
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
        numpy_acts = {}
        for name, tensor in self.activations.items():
            # Log tensor shape before conversion if dimension logging is enabled
            if ACTIVATION_DIMENSION_LOGGING:
                logger.info(f"Converting {name} tensor to NumPy: shape = {tuple(tensor.shape)}, dtype = {tensor.dtype}")
            
            # Convert to numpy
            numpy_array = tensor.detach().cpu().numpy()
            
            # Log numpy array shape after conversion if dimension logging is enabled
            if ACTIVATION_DIMENSION_LOGGING:
                logger.info(f"Converted {name} array: shape = {numpy_array.shape}, dtype = {numpy_array.dtype}")
                
                # Check for NaN or inf values
                if np.isnan(numpy_array).any() or np.isinf(numpy_array).any():
                    nan_count = np.isnan(numpy_array).sum()
                    inf_count = np.isinf(numpy_array).sum()
                    logger.warning(f"Layer {name} contains {nan_count} NaN and {inf_count} inf values")
                
                # Log statistics about the activations
                logger.info(f"Layer {name} stats: min = {numpy_array.min():.4f}, max = {numpy_array.max():.4f}, " 
                           f"mean = {numpy_array.mean():.4f}, std = {numpy_array.std():.4f}")
            
            numpy_acts[name] = numpy_array
        
        return numpy_acts


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


def track_train_test_dimensions(
    train_activations: Dict[str, torch.Tensor],
    test_activations: Dict[str, torch.Tensor],
    phase: str = "unknown",
    dataset: str = "unknown",
    handle_mismatch: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Track and log dimensions between train and test activations.
    
    Args:
        train_activations: Dictionary of training activations by layer
        test_activations: Dictionary of test activations by layer
        phase: Name of the current phase (e.g., "training", "evaluation")
        dataset: Name of the dataset being processed
        
    Returns:
        Dictionary with dimension analysis results
    """
    # Enable dimension logging temporarily if not already enabled
    was_enabled = get_dimension_logging()
    if not was_enabled:
        set_dimension_logging(True)
    
    results = {
        "dataset": dataset,
        "phase": phase,
        "timestamp": np.datetime_as_string(np.datetime64('now')),
        "layers": {}
    }
    
    logger.info(f"Dimension analysis for {dataset} dataset during {phase} phase:")
    
    # Get common layers
    train_layers = set(train_activations.keys())
    test_layers = set(test_activations.keys())
    common_layers = train_layers.intersection(test_layers)
    missing_in_train = test_layers - train_layers
    missing_in_test = train_layers - test_layers
    
    # Log layer coverage
    logger.info(f"Common layers: {len(common_layers)}, Missing in train: {len(missing_in_train)}, Missing in test: {len(missing_in_test)}")
    if missing_in_train:
        logger.warning(f"Layers in test but not in train: {missing_in_train}")
    if missing_in_test:
        logger.warning(f"Layers in train but not in test: {missing_in_test}")
    
    # Check dimensions for common layers
    for layer in common_layers:
        train_tensor = train_activations[layer]
        test_tensor = test_activations[layer]
        
        # Convert to numpy if needed
        train_array = train_tensor.detach().cpu().numpy() if isinstance(train_tensor, torch.Tensor) else train_tensor
        test_array = test_tensor.detach().cpu().numpy() if isinstance(test_tensor, torch.Tensor) else test_tensor
        
        # Compare dimensions
        train_shape = train_array.shape
        test_shape = test_array.shape
        
        # Calculate shape difference excluding batch dimension
        shape_match = train_shape[1:] == test_shape[1:]
        
        layer_results = {
            "train_shape": train_shape,
            "test_shape": test_shape,
            "shape_match": shape_match,
            "train_dtype": str(train_array.dtype),
            "test_dtype": str(test_array.dtype),
            "original_arrays": {"train": train_array, "test": test_array}
        }
        
        # Handle dimension mismatch if needed
        if not shape_match and handle_mismatch:
            strategy = get_dimension_mismatch_strategy()
            logger.warning(f"Using '{strategy}' strategy for dimension mismatch in layer {layer}")
            
            if strategy == 'error':
                raise ValueError(f"Dimension mismatch in layer {layer}: Train shape {train_shape}, Test shape {test_shape}")
            
            elif strategy == 'truncate' or strategy == 'pad':
                # Get feature dimensions (excluding batch size)
                train_features = train_shape[1:]
                test_features = test_shape[1:]
                
                if strategy == 'truncate':
                    # Find minimum feature dimensions
                    min_dims = [min(t, e) for t, e in zip(train_features, test_features)]
                    
                    # Truncate train array if needed
                    if train_features != min_dims:
                        # Create slicing for all dimensions
                        slices = [slice(None)] + [slice(0, d) for d in min_dims]
                        new_train_array = train_array[tuple(slices)]
                        logger.info(f"Truncated train array from {train_shape} to {new_train_array.shape}")
                        layer_results["adjusted_train"] = new_train_array
                    
                    # Truncate test array if needed
                    if test_features != min_dims:
                        # Create slicing for all dimensions
                        slices = [slice(None)] + [slice(0, d) for d in min_dims]
                        new_test_array = test_array[tuple(slices)]
                        logger.info(f"Truncated test array from {test_shape} to {new_test_array.shape}")
                        layer_results["adjusted_test"] = new_test_array
                
                elif strategy == 'pad':
                    # Find maximum feature dimensions
                    max_dims = [max(t, e) for t, e in zip(train_features, test_features)]
                    
                    # Pad train array if needed
                    if train_features != max_dims:
                        # Create padding configuration ((0,0) for batch dimension, then pad each feature dimension)
                        pad_width = [(0, 0)] + [(0, max_d - cur_d) for cur_d, max_d in zip(train_features, max_dims)]
                        new_train_array = np.pad(train_array, pad_width, mode='constant', constant_values=0)
                        logger.info(f"Padded train array from {train_shape} to {new_train_array.shape}")
                        layer_results["adjusted_train"] = new_train_array
                    
                    # Pad test array if needed
                    if test_features != max_dims:
                        # Create padding configuration ((0,0) for batch dimension, then pad each feature dimension)
                        pad_width = [(0, 0)] + [(0, max_d - cur_d) for cur_d, max_d in zip(test_features, max_dims)]
                        new_test_array = np.pad(test_array, pad_width, mode='constant', constant_values=0)
                        logger.info(f"Padded test array from {test_shape} to {new_test_array.shape}")
                        layer_results["adjusted_test"] = new_test_array
        
        # Log dimension comparison
        if shape_match:
            logger.info(f"Layer {layer}: Shapes match - Train: {train_shape}, Test: {test_shape}")
        else:
            logger.warning(f"Layer {layer}: Shape mismatch - Train: {train_shape}, Test: {test_shape}")
            
            # Compute detailed statistics for mismatched dimensions
            layer_results["train_stats"] = {
                "min": float(np.min(train_array)),
                "max": float(np.max(train_array)),
                "mean": float(np.mean(train_array)),
                "std": float(np.std(train_array)),
                "nan_count": int(np.isnan(train_array).sum()),
                "inf_count": int(np.isinf(train_array).sum())
            }
            
            layer_results["test_stats"] = {
                "min": float(np.min(test_array)),
                "max": float(np.max(test_array)),
                "mean": float(np.mean(test_array)),
                "std": float(np.std(test_array)),
                "nan_count": int(np.isnan(test_array).sum()),
                "inf_count": int(np.isinf(test_array).sum())
            }
            
            # Log potential remedies for dimension mismatch
            logger.info(f"Potential fixes for layer {layer} dimension mismatch:")
            logger.info(f"1. Reshape train: {train_shape} to match test feature dimension: {test_shape[1:]}")
            logger.info(f"2. Reshape test: {test_shape} to match train feature dimension: {train_shape[1:]}")
            logger.info(f"3. Project both to common dimension with PCA")
        
        results["layers"][layer] = layer_results
    
    # Restore previous logging state
    if not was_enabled:
        set_dimension_logging(False)
    
    return results


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
