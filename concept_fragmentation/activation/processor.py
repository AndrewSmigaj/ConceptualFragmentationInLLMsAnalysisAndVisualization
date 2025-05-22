"""
Activation processing module.

This module provides tools for processing and transforming neural network activations,
including dimensionality reduction, filtering, and feature extraction. It works with
both streaming and in-memory activation data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any, Tuple, Generator
import logging
from enum import Enum, auto
from dataclasses import dataclass
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection

from .collector import ActivationFormat

# Setup module-level logger
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Enumeration of processing modes."""
    MEMORY = auto()
    STREAMING = auto()
    DISTRIBUTED = auto()

class ProcessingOperation(Enum):
    """Enumeration of processing operations."""
    DIMENSIONALITY_REDUCTION = auto()
    NORMALIZATION = auto()
    FILTERING = auto()
    FEATURE_EXTRACTION = auto()
    POOLING = auto()
    TRANSFORMATION = auto()
    AGGREGATION = auto()
    CUSTOM = auto()

@dataclass
class ProcessorConfig:
    """Configuration for activation processing."""
    mode: ProcessingMode = ProcessingMode.MEMORY
    output_format: ActivationFormat = ActivationFormat.NUMPY
    batch_size: Optional[int] = None
    precision: str = 'float32'
    preserve_metadata: bool = True


class ActivationProcessor:
    """
    Processes neural network activations with support for streaming operations.
    
    This processor can handle both in-memory and streaming activation data,
    with a focus on memory efficiency for large models.
    
    Attributes:
        config: Configuration for processing operations
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """
        Initialize the activation processor.
        
        Args:
            config: Configuration for processing operations
        """
        self.config = config or ProcessorConfig()
        self._operations = []
        self._dim_reduction_models = {}
    
    def add_operation(
        self,
        operation_type: ProcessingOperation,
        op_fn: Callable,
        layers: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add a processing operation to the pipeline.
        
        Args:
            operation_type: Type of operation
            op_fn: Function implementing the operation
            layers: Specific layers to apply to (None for all)
            name: Name for this operation
        """
        operation = {
            "type": operation_type,
            "function": op_fn,
            "layers": layers,
            "name": name or f"{operation_type.name.lower()}_{len(self._operations)}"
        }
        self._operations.append(operation)
        logger.info(f"Added {operation_type.name} operation: {operation['name']}")
    
    def dimensionality_reduction(
        self,
        method: str = 'pca',
        n_components: int = 50,
        layers: Optional[List[str]] = None,
        fit_data: Optional[Dict[str, np.ndarray]] = None,
        name: Optional[str] = None,
        random_state: int = 42
    ) -> None:
        """
        Add dimensionality reduction operation.
        
        Args:
            method: Reduction method ('pca', 'ipca', 'random_projection')
            n_components: Number of components to reduce to
            layers: Specific layers to apply to (None for all)
            fit_data: Data to fit the reduction model (required for PCA)
            name: Name for this operation
            random_state: Random state for reproducibility
        """
        operation_name = name or f"dim_reduction_{method}_{n_components}"
        
        if method == 'pca' and fit_data is None and self.config.mode == ProcessingMode.MEMORY:
            raise ValueError("PCA requires fit_data when operating in memory mode")
        
        # Create dimensionality reduction model per layer
        if fit_data and method == 'pca':
            for layer, data in fit_data.items():
                if layers is None or layer in layers:
                    # Reshape if needed
                    if len(data.shape) > 2:
                        data = data.reshape(data.shape[0], -1)
                    
                    # Create and fit PCA model
                    pca = PCA(n_components=min(n_components, data.shape[1]), random_state=random_state)
                    pca.fit(data)
                    self._dim_reduction_models[f"{operation_name}_{layer}"] = pca
                    
                    logger.info(f"Fitted PCA model for layer {layer} with {pca.n_components_} components")
                    logger.info(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        def reduce_dimension(activations):
            """Apply dimensionality reduction to activations."""
            result = {}
            
            for layer, data in activations.items():
                if layers is not None and layer not in layers:
                    # Pass through layers that don't need reduction
                    result[layer] = data
                    continue
                
                # Skip metadata
                if layer == 'metadata':
                    result[layer] = data
                    continue
                
                # Convert to numpy if needed
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                
                # Reshape data if needed
                orig_shape = data.shape
                if len(orig_shape) > 2:
                    data = data.reshape(orig_shape[0], -1)
                
                # Apply reduction
                if method == 'pca':
                    # Check if we have a fitted model
                    model_key = f"{operation_name}_{layer}"
                    if model_key in self._dim_reduction_models:
                        pca = self._dim_reduction_models[model_key]
                        reduced = pca.transform(data)
                    else:
                        # Fit and transform
                        pca = PCA(n_components=min(n_components, data.shape[1]), random_state=random_state)
                        reduced = pca.fit_transform(data)
                        self._dim_reduction_models[model_key] = pca
                
                elif method == 'ipca':
                    # Incremental PCA
                    ipca = IncrementalPCA(n_components=min(n_components, data.shape[1]))
                    reduced = ipca.fit_transform(data)
                    self._dim_reduction_models[f"{operation_name}_{layer}"] = ipca
                
                elif method == 'random_projection':
                    # Random projection
                    rp = GaussianRandomProjection(n_components=min(n_components, data.shape[1]), random_state=random_state)
                    reduced = rp.fit_transform(data)
                    self._dim_reduction_models[f"{operation_name}_{layer}"] = rp
                
                # Store result
                result[layer] = reduced
                
                logger.debug(f"Reduced layer {layer} from {orig_shape} to {reduced.shape}")
            
            return result
        
        # Add the operation
        self.add_operation(
            operation_type=ProcessingOperation.DIMENSIONALITY_REDUCTION,
            op_fn=reduce_dimension,
            layers=layers,
            name=operation_name
        )
    
    def normalize(
        self,
        method: str = 'standard',
        layers: Optional[List[str]] = None,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        axis: int = 0,
        name: Optional[str] = None
    ) -> None:
        """
        Add normalization operation.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
            layers: Specific layers to apply to (None for all)
            stats: Pre-computed statistics for normalization
            axis: Axis to normalize along (0 for per-feature, 1 for per-sample)
            name: Name for this operation
        """
        operation_name = name or f"normalize_{method}"
        
        def normalize_activations(activations):
            """Apply normalization to activations."""
            result = {}
            
            for layer, data in activations.items():
                if layers is not None and layer not in layers:
                    # Pass through layers that don't need normalization
                    result[layer] = data
                    continue
                
                # Skip metadata
                if layer == 'metadata':
                    result[layer] = data
                    continue
                
                # Convert to numpy if needed
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                
                # Apply normalization
                if method == 'standard':
                    if stats and layer in stats and 'mean' in stats[layer] and 'std' in stats[layer]:
                        # Use pre-computed stats
                        mean = stats[layer]['mean']
                        std = stats[layer]['std']
                    else:
                        # Compute stats
                        mean = np.mean(data, axis=axis, keepdims=True)
                        std = np.std(data, axis=axis, keepdims=True)
                        std = np.where(std == 0, 1.0, std)  # Avoid division by zero
                    
                    normalized = (data - mean) / std
                
                elif method == 'minmax':
                    if stats and layer in stats and 'min' in stats[layer] and 'max' in stats[layer]:
                        # Use pre-computed stats
                        min_val = stats[layer]['min']
                        max_val = stats[layer]['max']
                    else:
                        # Compute stats
                        min_val = np.min(data, axis=axis, keepdims=True)
                        max_val = np.max(data, axis=axis, keepdims=True)
                        data_range = max_val - min_val
                        data_range = np.where(data_range == 0, 1.0, data_range)  # Avoid division by zero
                    
                    normalized = (data - min_val) / (max_val - min_val)
                
                elif method == 'robust':
                    if stats and layer in stats and 'p25' in stats[layer] and 'p75' in stats[layer]:
                        # Use pre-computed stats
                        p25 = stats[layer]['p25']
                        p75 = stats[layer]['p75']
                    else:
                        # Compute stats
                        p25 = np.percentile(data, 25, axis=axis, keepdims=True)
                        p75 = np.percentile(data, 75, axis=axis, keepdims=True)
                        iqr = p75 - p25
                        iqr = np.where(iqr == 0, 1.0, iqr)  # Avoid division by zero
                    
                    normalized = (data - p25) / (p75 - p25)
                
                # Store result
                result[layer] = normalized
            
            return result
        
        # Add the operation
        self.add_operation(
            operation_type=ProcessingOperation.NORMALIZATION,
            op_fn=normalize_activations,
            layers=layers,
            name=operation_name
        )
    
    def filter(
        self,
        filter_fn: Callable,
        layers: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add filtering operation.
        
        Args:
            filter_fn: Function to apply to each activation tensor
            layers: Specific layers to apply to (None for all)
            name: Name for this operation
        """
        operation_name = name or f"filter_{len(self._operations)}"
        
        def filter_activations(activations):
            """Apply filtering to activations."""
            result = {}
            
            for layer, data in activations.items():
                if layers is not None and layer not in layers:
                    # Pass through layers that don't need filtering
                    result[layer] = data
                    continue
                
                # Skip metadata
                if layer == 'metadata':
                    result[layer] = data
                    continue
                
                # Apply filter
                result[layer] = filter_fn(data)
            
            return result
        
        # Add the operation
        self.add_operation(
            operation_type=ProcessingOperation.FILTERING,
            op_fn=filter_activations,
            layers=layers,
            name=operation_name
        )
    
    def extract_features(
        self,
        method: str = 'mean',
        layers: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add feature extraction operation.
        
        Args:
            method: Extraction method ('mean', 'max', 'std', 'entropy')
            layers: Specific layers to apply to (None for all)
            name: Name for this operation
        """
        operation_name = name or f"extract_{method}"
        
        def extract_activations(activations):
            """Extract features from activations."""
            result = {}
            
            for layer, data in activations.items():
                if layers is not None and layer not in layers:
                    # Pass through layers that don't need extraction
                    result[layer] = data
                    continue
                
                # Skip metadata
                if layer == 'metadata':
                    result[layer] = data
                    continue
                
                # Convert to numpy if needed
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                
                # Extract features
                if method == 'mean':
                    extracted = np.mean(data, axis=1, keepdims=True)
                elif method == 'max':
                    extracted = np.max(data, axis=1, keepdims=True)
                elif method == 'std':
                    extracted = np.std(data, axis=1, keepdims=True)
                elif method == 'entropy':
                    # Compute entropy for probability-like activations
                    # Normalize to probabilities
                    probs = np.abs(data)
                    probs = probs / np.sum(probs, axis=1, keepdims=True)
                    # Calculate entropy
                    epsilon = 1e-10  # Small value to avoid log(0)
                    entropy = -np.sum(probs * np.log2(probs + epsilon), axis=1, keepdims=True)
                    extracted = entropy
                
                # Store result
                result[layer] = extracted
            
            return result
        
        # Add the operation
        self.add_operation(
            operation_type=ProcessingOperation.FEATURE_EXTRACTION,
            op_fn=extract_activations,
            layers=layers,
            name=operation_name
        )
    
    def pool(
        self,
        method: str = 'mean',
        window_size: Optional[Tuple[int, ...]] = None,
        layers: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add pooling operation for convolutional layers.
        
        Args:
            method: Pooling method ('mean', 'max', 'min')
            window_size: Size of pooling window
            layers: Specific layers to apply to (None for all)
            name: Name for this operation
        """
        operation_name = name or f"pool_{method}"
        
        def pool_activations(activations):
            """Apply pooling to activations."""
            result = {}
            
            for layer, data in activations.items():
                if layers is not None and layer not in layers:
                    # Pass through layers that don't need pooling
                    result[layer] = data
                    continue
                
                # Skip metadata
                if layer == 'metadata':
                    result[layer] = data
                    continue
                
                # Convert to numpy if needed
                if isinstance(data, torch.Tensor):
                    is_tensor = True
                    data_np = data.detach().cpu().numpy()
                else:
                    is_tensor = False
                    data_np = data
                
                # Only apply pooling to tensors with enough dimensions
                if len(data_np.shape) < 3:
                    result[layer] = data
                    continue
                
                # Apply pooling
                # For simplicity, we'll implement a basic pooling
                # A more sophisticated implementation would use PyTorch's pooling layers
                
                # Determine pooling dimensions
                spatial_dims = data_np.shape[2:]
                
                if method == 'mean':
                    # Mean over spatial dimensions
                    pooled = np.mean(data_np, axis=tuple(range(2, len(data_np.shape))))
                elif method == 'max':
                    # Max over spatial dimensions
                    pooled = np.max(data_np, axis=tuple(range(2, len(data_np.shape))))
                elif method == 'min':
                    # Min over spatial dimensions
                    pooled = np.min(data_np, axis=tuple(range(2, len(data_np.shape))))
                
                # Convert back to tensor if needed
                if is_tensor:
                    pooled = torch.tensor(pooled)
                
                # Store result
                result[layer] = pooled
            
            return result
        
        # Add the operation
        self.add_operation(
            operation_type=ProcessingOperation.POOLING,
            op_fn=pool_activations,
            layers=layers,
            name=operation_name
        )
    
    def transform(
        self,
        transform_fn: Callable,
        layers: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Add custom transformation operation.
        
        Args:
            transform_fn: Function to apply to each activation tensor
            layers: Specific layers to apply to (None for all)
            name: Name for this operation
        """
        operation_name = name or f"transform_{len(self._operations)}"
        
        def transform_activations(activations):
            """Apply transformation to activations."""
            result = {}
            
            for layer, data in activations.items():
                if layers is not None and layer not in layers:
                    # Pass through layers that don't need transformation
                    result[layer] = data
                    continue
                
                # Skip metadata
                if layer == 'metadata':
                    result[layer] = data
                    continue
                
                # Apply transform
                result[layer] = transform_fn(data)
            
            return result
        
        # Add the operation
        self.add_operation(
            operation_type=ProcessingOperation.TRANSFORMATION,
            op_fn=transform_activations,
            layers=layers,
            name=operation_name
        )
    
    def process(
        self,
        activations: Dict[str, Any],
        streaming: bool = False
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Process activations through all registered operations.
        
        Args:
            activations: Dictionary of activations or generator of batched activations
            streaming: Whether to process in streaming mode
            
        Returns:
            Processed activations
        """
        if streaming or isinstance(activations, Generator):
            return self._process_streaming(activations)
        else:
            return self._process_memory(activations)
    
    def _process_memory(self, activations: Dict[str, Any]) -> Dict[str, Any]:
        """Process activations in memory."""
        result = activations
        
        # Apply all operations in sequence
        for operation in self._operations:
            operation_fn = operation["function"]
            result = operation_fn(result)
        
        return result
    
    def _process_streaming(
        self,
        activations: Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Process activations in streaming mode."""
        # Convert to generator if not already
        activation_gen = activations
        if not isinstance(activations, Generator):
            # Create single-item generator
            def single_item_gen():
                yield activations
            activation_gen = single_item_gen()
        
        # Process each batch
        for batch in activation_gen:
            result = batch
            
            # Apply all operations in sequence
            for operation in self._operations:
                operation_fn = operation["function"]
                result = operation_fn(result)
            
            yield result