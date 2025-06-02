"""
Activation collection module with streaming support.

This module provides tools for efficiently collecting activations from neural networks,
with a focus on memory efficiency through streaming operations and flexible persistence options.
It is designed to work with both small MLPs and large transformer models like GPT-2.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Generator, Any, Callable, Tuple, Set
import contextlib
import tempfile
import os
import logging
import pickle
import time
from enum import Enum, auto
import warnings
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from ..hooks.activation_hooks import (
    ActivationHook, get_all_named_layers, 
    get_dimension_logging, set_dimension_logging
)
from ..utils.path_utils import ensure_directory
from ..models.model_interfaces import ModelInterface, ActivationPoint

# Setup module-level logger
logger = logging.getLogger(__name__)

class ActivationFormat(Enum):
    """Enumeration of activation storage formats."""
    TORCH = auto()
    NUMPY = auto()
    DISK = auto()

@dataclass
class CollectionConfig:
    """Configuration for activation collection."""
    device: str = 'cpu'
    format: ActivationFormat = ActivationFormat.NUMPY
    temp_dir: Optional[str] = None
    batch_size: Optional[int] = None
    dtype: torch.dtype = torch.float32
    log_dimensions: bool = False
    stream_to_disk: bool = False
    include_metadata: bool = True


class ActivationCollector:
    """
    Manages activation collection with memory efficiency and streaming support.
    
    This collector improves upon the original ActivationHook by:
    1. Supporting streaming activations (yield instead of storing all in memory)
    2. Providing disk-based temporary storage for large models
    3. Allowing batch processing for memory-constrained environments
    4. Supporting different storage formats (torch, numpy, disk)
    
    Attributes:
        config: Configuration for activation collection
        hooks: Dictionary of ActivationHooks for different models
    """
    
    def __init__(
        self,
        config: Optional[CollectionConfig] = None
    ):
        """
        Initialize the activation collector.
        
        Args:
            config: Configuration for activation collection
        """
        self.config = config or CollectionConfig()
        self.hooks = {}
        
        # Create temp directory if using disk mode and not specified
        self._temp_dir_obj = None
        if self.config.format == ActivationFormat.DISK and not self.config.temp_dir:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.config.temp_dir = self._temp_dir_obj.name
            logger.info(f"Created temporary directory for activations: {self.config.temp_dir}")

        # Set dimension logging based on config
        current_logging = get_dimension_logging()
        if current_logging != self.config.log_dimensions:
            set_dimension_logging(self.config.log_dimensions)
    
    def __del__(self):
        """Clean up resources on deletion."""
        # Clean up hooks
        for hook in self.hooks.values():
            hook.remove_hooks()
            
        # Clean up temp directory
        if self._temp_dir_obj:
            self._temp_dir_obj.cleanup()
    
    def register_model(
        self, 
        model: Union[nn.Module, ModelInterface],
        model_id: str = 'default',
        activation_points: Optional[List[Union[str, ActivationPoint]]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> None:
        """
        Register a model for activation collection.
        
        Args:
            model: PyTorch model or ModelInterface to collect activations from
            model_id: Identifier for this model
            activation_points: Specific activation points or layer names to track
            include_patterns: Patterns to match layer names
            exclude_patterns: Patterns to exclude from layer names
        """
        # If model has a .model attribute (ModelInterface), get the underlying model
        if hasattr(model, 'model') and isinstance(getattr(model, 'model'), nn.Module):
            underlying_model = model.model
        else:
            underlying_model = model
            
        # Convert activation points to layer names if needed
        layer_names = None
        if activation_points:
            layer_names = []
            for point in activation_points:
                if isinstance(point, str):
                    layer_names.append(point)
                elif isinstance(point, ActivationPoint):
                    layer_names.append(point.layer_name)
        
        # Remove existing hook if present
        if model_id in self.hooks:
            self.hooks[model_id].remove_hooks()
            
        # Create new hook
        hook = ActivationHook(
            model=underlying_model,
            layer_names=layer_names,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            device=self.config.device
        )
        hook.register_hooks()
        self.hooks[model_id] = hook
        
        logger.info(f"Registered model '{model_id}' for activation collection")
        
        # Log which layers will be captured
        all_layers = get_all_named_layers(underlying_model)
        filtered_layers = [name for name in all_layers if hook._layer_filter(name)]
        if filtered_layers:
            logger.info(f"Will capture activations from {len(filtered_layers)} layers")
            if len(filtered_layers) <= 10 or self.config.log_dimensions:
                logger.info(f"Layers: {filtered_layers}")
        else:
            logger.warning(f"No layers matched the filter criteria")
    
    def collect(
        self,
        model: Union[nn.Module, ModelInterface],
        inputs: Any,
        model_id: str = 'default',
        activation_points: Optional[List[Union[str, ActivationPoint]]] = None,
        streaming: bool = False,
        process_fn: Optional[Callable] = None,
        to_device: Optional[str] = None,
        to_format: Optional[ActivationFormat] = None
    ) -> Union[Dict[str, Union[torch.Tensor, np.ndarray]], Generator]:
        """
        Collect activations from a model forward pass.
        
        Args:
            model: PyTorch model or ModelInterface to collect from
            inputs: Input data for the model
            model_id: Identifier for the model
            activation_points: Specific activation points to collect
            streaming: Whether to stream activations or return all at once
            process_fn: Optional function to process activations before returning
            to_device: Device to move activations to (None for config default)
            to_format: Format to convert activations to (None for config default)
            
        Returns:
            Either a dictionary of activations or a generator of activations
        """
        # Register model if not already registered
        if model_id not in self.hooks:
            self.register_model(
                model=model,
                model_id=model_id,
                activation_points=activation_points
            )
            
        # Set output format
        output_format = to_format or self.config.format
        
        # Set output device
        device = to_device or self.config.device
        
        # Get the hook
        hook = self.hooks[model_id]
        
        # Handle streaming mode
        if streaming:
            return self._collect_streaming(
                model=model,
                inputs=inputs,
                hook=hook,
                output_format=output_format,
                device=device,
                process_fn=process_fn
            )
        
        # Non-streaming mode: collect all at once
        with torch.no_grad():
            # Forward pass to collect activations
            logger.debug("About to perform forward pass")
            try:
                if hasattr(model, 'forward') and hasattr(model, 'model'):
                    model.forward(inputs)
                else:
                    _ = model(inputs)
                logger.debug("Forward pass completed")
            except Exception as e:
                logger.error(f"Error during forward pass: {e}")
                logger.debug("Traceback for forward pass error:", exc_info=True)
                raise
            
            # Get activations
            logger.debug("About to process activations")
            try:
                activations = self._process_activations(
                    hook=hook,
                    output_format=output_format,
                    device=device,
                    process_fn=process_fn
                )
                logger.debug(f"Activations processed, returning {len(activations) if isinstance(activations, dict) else 'non-dict'} activations")
            except Exception as e:
                logger.error(f"Error processing activations: {e}")
                logger.debug("Traceback for activation processing error:", exc_info=True)
                raise
            
            return activations
    
    def _process_activations(
        self, 
        hook: ActivationHook,
        output_format: ActivationFormat,
        device: str,
        process_fn: Optional[Callable] = None
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Process activations based on format and apply processing function."""
        # Get activations
        if output_format == ActivationFormat.NUMPY:
            logger.debug("About to call hook.numpy_activations()")
            try:
                activations = hook.numpy_activations()
                logger.debug(f"After numpy_activations, got type: {type(activations)}, keys: {activations.keys() if isinstance(activations, dict) else 'Not a dict'}")
                
                # DEBUG: Check what's in activations
                if isinstance(activations, dict):
                    for k, v in activations.items():
                        logger.debug(f"activations[{k}] type: {type(v)}, shape: {v.shape if hasattr(v, 'shape') else 'NO SHAPE'}")
            except Exception as e:
                logger.error(f"Error in numpy_activations: {e}")
                logger.debug("Traceback for numpy_activations error:", exc_info=True)
                raise
        else:
            activations = {
                k: v.to(device) for k, v in hook.layer_activations.items()
            }
        
        # Apply processing function if provided
        if process_fn:
            logger.debug(f"About to apply process_fn: {process_fn}")
            try:
                activations = process_fn(activations)
                logger.debug(f"After process_fn, activations type: {type(activations)}")
            except Exception as e:
                logger.error(f"Error in process_fn: {e}")
                logger.debug("Traceback for process_fn error:", exc_info=True)
                raise
            
        return activations
    
    def _collect_streaming(
        self,
        model: Union[nn.Module, ModelInterface],
        inputs: Any,
        hook: ActivationHook,
        output_format: ActivationFormat,
        device: str,
        process_fn: Optional[Callable] = None
    ) -> Generator:
        """
        Generator for streaming activations.
        
        Args:
            model: PyTorch model or ModelInterface
            inputs: Input data
            hook: Activation hook
            output_format: Format to output activations in
            device: Device to move activations to
            process_fn: Processing function
            
        Yields:
            Activations for each batch
        """
        # Check if inputs is a DataLoader or similar iterable
        if hasattr(inputs, '__iter__') and not isinstance(inputs, (torch.Tensor, np.ndarray)):
            # Process each batch from the DataLoader
            for batch_idx, batch_inputs in enumerate(inputs):
                # Clear previous activations
                hook.clear_activations()
                
                # Forward pass to collect activations
                with torch.no_grad():
                    if hasattr(model, 'forward') and hasattr(model, 'model'):
                        model.forward(batch_inputs)
                    else:
                        _ = model(batch_inputs)
                
                # Process activations
                batch_activations = self._process_activations(
                    hook=hook,
                    output_format=output_format,
                    device=device,
                    process_fn=process_fn
                )
                
                # Add batch index to metadata
                batch_activations = {
                    'metadata': {'batch_idx': batch_idx},
                    'activations': batch_activations
                }
                
                yield batch_activations
        else:
            # Handle tensor inputs by splitting into batches
            batch_size = self.config.batch_size or len(inputs)
            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs)
                
            num_samples = len(inputs)
            for batch_start in range(0, num_samples, batch_size):
                # Clear previous activations
                hook.clear_activations()
                
                # Get batch
                batch_end = min(batch_start + batch_size, num_samples)
                if isinstance(inputs, tuple) and len(inputs) == 2:
                    # Handle (X, y) tuple
                    batch_X = inputs[0][batch_start:batch_end]
                    batch_y = inputs[1][batch_start:batch_end]
                    batch_inputs = (batch_X, batch_y)
                else:
                    # Handle single tensor
                    batch_inputs = inputs[batch_start:batch_end]
                
                # Forward pass to collect activations
                with torch.no_grad():
                    if hasattr(model, 'forward') and hasattr(model, 'model'):
                        model.forward(batch_inputs)
                    else:
                        _ = model(batch_inputs)
                
                # Process activations
                batch_activations = self._process_activations(
                    hook=hook,
                    output_format=output_format,
                    device=device,
                    process_fn=process_fn
                )
                
                # Add batch metadata
                batch_activations = {
                    'metadata': {
                        'batch_idx': batch_start // batch_size,
                        'batch_start': batch_start,
                        'batch_end': batch_end,
                        'batch_size': batch_end - batch_start
                    },
                    'activations': batch_activations
                }
                
                yield batch_activations
    
    def collect_and_store(
        self,
        model: Union[nn.Module, ModelInterface],
        inputs: Any,
        output_path: Optional[str] = None,
        model_id: str = 'default',
        split_name: str = 'train',
        activation_points: Optional[List[Union[str, ActivationPoint]]] = None,
        streaming: bool = False,
        process_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Collect activations and store them to disk.
        
        Args:
            model: PyTorch model or ModelInterface
            inputs: Input data
            output_path: Path to save activations (None for temp file)
            model_id: Model identifier
            split_name: Name of the data split (e.g., 'train', 'test')
            activation_points: Specific activation points to collect
            streaming: Whether to use streaming mode
            process_fn: Processing function
            metadata: Additional metadata to store
            
        Returns:
            Path to the saved file
        """
        # Create output path if not provided
        if not output_path:
            if not self.config.temp_dir:
                self._temp_dir_obj = tempfile.TemporaryDirectory()
                self.config.temp_dir = self._temp_dir_obj.name
            
            filename = f"{model_id}_{split_name}_activations.pkl"
            output_path = os.path.join(self.config.temp_dir, filename)
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_path))
        
        # Create basic metadata
        save_metadata = {
            "model_id": model_id,
            "split": split_name,
            "timestamp": np.datetime64('now').astype(str),
            "streaming": streaming,
            "format": self.config.format.name
        }
        
        # Add custom metadata if provided
        if metadata:
            save_metadata.update(metadata)
        
        if streaming:
            # Use streaming collection and direct disk storage
            output_path = self._collect_streaming_to_file(
                model=model,
                inputs=inputs,
                output_path=output_path,
                model_id=model_id,
                activation_points=activation_points,
                process_fn=process_fn,
                metadata=save_metadata
            )
        else:
            # Collect all activations at once
            activations = self.collect(
                model=model,
                inputs=inputs,
                model_id=model_id,
                activation_points=activation_points,
                streaming=False,
                process_fn=process_fn
            )
            
            # Update metadata
            save_metadata["layer_count"] = len(activations)
            save_metadata["input_shape"] = self._get_input_shape(inputs)
            
            # Store data with metadata
            data_to_save = {
                "metadata": save_metadata,
                "activations": activations
            }
            
            # Save to disk
            with open(output_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            logger.info(f"Saved {len(activations)} layer activations to {output_path}")
        
        return output_path
    
    def _collect_streaming_to_file(
        self,
        model: Union[nn.Module, ModelInterface],
        inputs: Any,
        output_path: str,
        model_id: str,
        activation_points: Optional[List[Union[str, ActivationPoint]]] = None,
        process_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Collect activations in streaming mode and save directly to file.
        
        Args:
            model: PyTorch model or ModelInterface
            inputs: Input data
            output_path: Path to save activations
            model_id: Model identifier
            activation_points: Specific activation points to collect
            process_fn: Processing function
            metadata: Additional metadata to include
            
        Returns:
            Path to the saved file
        """
        # Register model if not already registered
        if model_id not in self.hooks:
            self.register_model(
                model=model,
                model_id=model_id,
                activation_points=activation_points
            )
        
        # Create streaming generator
        activations_gen = self.collect(
            model=model,
            inputs=inputs,
            model_id=model_id,
            activation_points=activation_points,
            streaming=True,
            process_fn=process_fn
        )
        
        # Open file for writing
        with open(output_path, 'wb') as f:
            # Initialize layer structure
            layer_names = []
            batch_count = 0
            header_written = False
            
            # Process batches
            start_time = time.time()
            for batch_idx, batch_data in enumerate(activations_gen):
                batch_count += 1
                batch_metadata = batch_data["metadata"]
                batch_activations = batch_data["activations"]
                
                # Record layer names on first batch
                if batch_idx == 0:
                    layer_names = list(batch_activations.keys())
                    
                    # Create header with metadata
                    header_metadata = metadata.copy() if metadata else {}
                    header_metadata.update({
                        "layer_count": len(layer_names),
                        "layer_names": layer_names,
                        "input_shape": self._get_input_shape(inputs),
                        "batch_count": 0  # Will be updated at end
                    })
                    
                    # Write header
                    header = {
                        "metadata": header_metadata,
                        "format": "streaming"
                    }
                    pickle.dump(header, f)
                    header_written = True
                
                # Pickle batch data with batch index
                pickle.dump({
                    "metadata": batch_metadata,
                    "activations": batch_activations
                }, f)
                
                # Periodically report progress
                if batch_idx % 10 == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {batch_idx} batches in {elapsed:.2f}s ({batch_idx/elapsed:.2f} batches/s)")
                
                # Ensure file is flushed periodically
                if batch_idx % 100 == 0:
                    f.flush()
            
            # Update header with final batch count
            if header_written:
                # Go back to beginning of file
                f.seek(0)
                
                # Read header
                header = pickle.load(f)
                
                # Update batch count
                header["metadata"]["batch_count"] = batch_count
                header["metadata"]["timestamp"] = np.datetime64('now').astype(str)
                
                # Write updated header
                f.seek(0)
                pickle.dump(header, f)
            
            elapsed = time.time() - start_time
            logger.info(f"Saved streaming activations from {batch_count} batches to {output_path} in {elapsed:.2f}s")
        
        return output_path
    
    def collect_train_test(
        self,
        model: Union[nn.Module, ModelInterface],
        train_data: Any,
        test_data: Any,
        output_dir: Optional[str] = None,
        model_id: str = 'default',
        activation_points: Optional[List[Union[str, ActivationPoint]]] = None,
        streaming: bool = False,
        process_fn: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Collect activations for both training and test data.
        
        Args:
            model: PyTorch model or ModelInterface
            train_data: Training data
            test_data: Test data
            output_dir: Directory to save activations
            model_id: Model identifier
            activation_points: Specific activation points to collect
            streaming: Whether to use streaming mode
            process_fn: Processing function
            metadata: Additional metadata to include
            
        Returns:
            Dictionary with paths to saved activation files
        """
        # Create output directory if needed
        if output_dir:
            ensure_directory(output_dir)
        else:
            if not self.config.temp_dir:
                self._temp_dir_obj = tempfile.TemporaryDirectory()
                self.config.temp_dir = self._temp_dir_obj.name
            output_dir = self.config.temp_dir
        
        results = {}
        
        # Create common metadata
        common_metadata = {
            "model_id": model_id,
            "timestamp": np.datetime64('now').astype(str),
            "streaming": streaming,
        }
        
        if metadata:
            common_metadata.update(metadata)
        
        # Collect train activations
        train_metadata = common_metadata.copy()
        train_metadata["split"] = "train"
        
        train_file = self.collect_and_store(
            model=model,
            inputs=train_data,
            output_path=os.path.join(output_dir, f"{model_id}_train_activations.pkl"),
            model_id=model_id,
            split_name='train',
            activation_points=activation_points,
            streaming=streaming,
            process_fn=process_fn,
            metadata=train_metadata
        )
        results['train'] = train_file
        
        # Collect test activations
        test_metadata = common_metadata.copy()
        test_metadata["split"] = "test"
        
        test_file = self.collect_and_store(
            model=model,
            inputs=test_data,
            output_path=os.path.join(output_dir, f"{model_id}_test_activations.pkl"),
            model_id=model_id,
            split_name='test',
            activation_points=activation_points,
            streaming=streaming,
            process_fn=process_fn,
            metadata=test_metadata
        )
        results['test'] = test_file
        
        # Create index file with references to both files
        index_data = {
            "metadata": {
                "model_id": model_id,
                "timestamp": np.datetime64('now').astype(str),
                "streaming": streaming,
            },
            "files": {
                "train": os.path.basename(train_file),
                "test": os.path.basename(test_file)
            }
        }
        
        if metadata:
            index_data["metadata"].update(metadata)
        
        index_path = os.path.join(output_dir, f"{model_id}_activations_index.json")
        with open(index_path, 'w') as f:
            import json
            json.dump(index_data, f, indent=2)
        
        results['index'] = index_path
        
        return results
    
    def _get_input_shape(self, inputs: Any) -> Union[List[int], Dict[str, List[int]], str]:
        """Extract input shape information for metadata."""
        try:
            if isinstance(inputs, torch.Tensor):
                return list(inputs.shape)
            elif isinstance(inputs, np.ndarray):
                return list(inputs.shape)
            elif isinstance(inputs, tuple) and len(inputs) > 0:
                shapes = {}
                for i, item in enumerate(inputs):
                    if hasattr(item, 'shape'):
                        shapes[f"input_{i}"] = list(item.shape)
                return shapes
            elif hasattr(inputs, '__len__'):
                return [len(inputs)]
        except Exception as e:
            logger.warning(f"Could not determine input shape: {e}")
        
        return "unknown"
    
    @staticmethod
    def load_activations(
        file_path: str,
        layers: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
        concat_batches: bool = True
    ) -> Dict[str, Any]:
        """
        Load activations from a saved file.
        
        Args:
            file_path: Path to the saved activations
            layers: Specific layers to load (None for all)
            max_batches: Maximum number of batches to load
            concat_batches: Whether to concatenate batches for streaming files
            
        Returns:
            Dictionary with loaded activations
        """
        with open(file_path, 'rb') as f:
            # Load header
            data = pickle.load(f)
            
            # Check if this is a streaming file
            if isinstance(data, dict) and data.get("format") == "streaming":
                # Handle streaming format
                metadata = data["metadata"]
                all_layer_names = metadata.get("layer_names", [])
                
                # Filter layers if specified
                if layers:
                    layer_names = [l for l in all_layer_names if l in layers]
                else:
                    layer_names = all_layer_names
                
                # Initialize result structure
                result = {
                    "metadata": metadata,
                    "batches": []
                }
                
                # Initialize layer data if concatenating
                if concat_batches:
                    result["activations"] = {layer: [] for layer in layer_names}
                
                # Load batch data
                batch_count = 0
                try:
                    while True:
                        if max_batches and batch_count >= max_batches:
                            break
                            
                        # Load next batch
                        batch_data = pickle.load(f)
                        batch_activations = batch_data["activations"]
                        
                        # Filter layers if needed
                        if layers:
                            batch_activations = {k: v for k, v in batch_activations.items() if k in layers}
                        
                        # Store batch if not concatenating
                        if not concat_batches:
                            result["batches"].append(batch_data)
                        
                        # Concatenate activations if requested
                        if concat_batches:
                            for layer_name, activation in batch_activations.items():
                                if layer_name in result["activations"]:
                                    result["activations"][layer_name].append(activation)
                        
                        batch_count += 1
                except EOFError:
                    # End of file
                    pass
                
                # Concatenate activations if possible and requested
                if concat_batches:
                    for layer_name, activations in result["activations"].items():
                        if activations:
                            # Try concatenating numpy arrays
                            if all(isinstance(a, np.ndarray) for a in activations):
                                try:
                                    result["activations"][layer_name] = np.concatenate(activations, axis=0)
                                except Exception as e:
                                    logger.warning(f"Could not concatenate activations for layer {layer_name}: {e}")
                            # Try concatenating torch tensors
                            elif all(isinstance(a, torch.Tensor) for a in activations):
                                try:
                                    result["activations"][layer_name] = torch.cat(activations, dim=0)
                                except Exception as e:
                                    logger.warning(f"Could not concatenate activations for layer {layer_name}: {e}")
                
                return result
            else:
                # Handle regular format (all data in memory)
                if isinstance(data, dict) and "activations" in data:
                    if layers:
                        # Filter for specific layers
                        data["activations"] = {k: v for k, v in data["activations"].items() if k in layers}
                    
                    return data
                else:
                    # Legacy format or unexpected format
                    logger.warning("Found legacy or unexpected activation format")
                    activations = data
                    
                    if layers:
                        # Filter for specific layers
                        activations = {k: v for k, v in activations.items() if k in layers}
                    
                    # Wrap in standard format
                    return {
                        "metadata": {
                            "timestamp": np.datetime64('now').astype(str),
                            "format": "legacy"
                        },
                        "activations": activations
                    }


@contextlib.contextmanager
def collect_activations(
    model: Union[nn.Module, ModelInterface],
    layer_names: Optional[List[str]] = None,
    device: str = 'cpu',
    to_numpy: bool = False
) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
    """
    Context manager for activation collection, compatible with original API.
    
    Args:
        model: PyTorch model to collect activations from
        layer_names: Specific layer names to track
        device: Device to store activations
        to_numpy: Whether to convert results to NumPy arrays
        
    Returns:
        Dictionary of activations
    """
    config = CollectionConfig(device=device)
    collector = ActivationCollector(config=config)
    collector.register_model(model, activation_points=layer_names)
    
    hook = collector.hooks['default']
    try:
        if to_numpy:
            yield lambda: hook.numpy_activations()
        else:
            yield hook.activations
    finally:
        hook.remove_hooks()