"""
Activation storage module.

This module provides flexible storage mechanisms for neural network activations,
with support for different storage formats, compression, and efficient retrieval.
It is designed to work with both small MLPs and large transformer models like GPT-2.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, Generator, BinaryIO
import os
import pickle
import json
import logging
import tempfile
import time
import shutil
import zlib
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import h5py

from ..utils.path_utils import ensure_directory
from .collector import ActivationFormat

# Setup module-level logger
logger = logging.getLogger(__name__)

class StorageBackend(Enum):
    """Enumeration of storage backends."""
    PICKLE = auto()
    HDF5 = auto()
    MEMORY = auto()
    NUMPY = auto()
    CUSTOM = auto()

class CompressionLevel(Enum):
    """Enumeration of compression levels."""
    NONE = 0
    FAST = 1
    BALANCED = 5
    MAX = 9

@dataclass
class StorageConfig:
    """Configuration for activation storage."""
    backend: StorageBackend = StorageBackend.PICKLE
    format: ActivationFormat = ActivationFormat.NUMPY
    compression: CompressionLevel = CompressionLevel.NONE
    chunk_size: Optional[int] = None  # For HDF5
    metadata_only: bool = False
    base_dir: Optional[str] = None
    temp_dir: Optional[str] = None


class ActivationStorage:
    """
    Manages storage and retrieval of neural network activations.
    
    This class provides a flexible interface for storing activations in various
    formats and backends, with options for compression and chunking to handle
    large models efficiently.
    
    Attributes:
        config: Configuration for storage operations
    """
    
    def __init__(self, config: Optional[StorageConfig] = None):
        """
        Initialize the activation storage.
        
        Args:
            config: Configuration for storage operations
        """
        self.config = config or StorageConfig()
        
        # Create base directory if needed
        if self.config.base_dir:
            ensure_directory(self.config.base_dir)
        
        # Create temp directory if needed
        self._temp_dir_obj = None
        if self.config.temp_dir is None:
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            self.config.temp_dir = self._temp_dir_obj.name
            ensure_directory(self.config.temp_dir)
    
    def __del__(self):
        """Clean up resources on deletion."""
        if self._temp_dir_obj:
            self._temp_dir_obj.cleanup()
    
    def save(
        self,
        activations: Dict[str, Any],
        file_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        streaming: bool = False
    ) -> str:
        """
        Save activations to storage.
        
        Args:
            activations: Dictionary of activations or generator of batched activations
            file_path: Path to save to (None for auto-generated)
            metadata: Additional metadata to store
            streaming: Whether the input is a stream of batches
            
        Returns:
            Path to the saved file
        """
        # Generate file path if not provided
        if not file_path:
            timestamp = int(time.time())
            file_path = os.path.join(
                self.config.base_dir or self.config.temp_dir,
                f"activations_{timestamp}.act"
            )
        
        # Ensure directory exists
        ensure_directory(os.path.dirname(file_path))
        
        # Select save method based on backend
        if self.config.backend == StorageBackend.PICKLE:
            return self._save_pickle(activations, file_path, metadata, streaming)
        elif self.config.backend == StorageBackend.HDF5:
            return self._save_hdf5(activations, file_path, metadata, streaming)
        elif self.config.backend == StorageBackend.NUMPY:
            return self._save_numpy(activations, file_path, metadata, streaming)
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.backend}")
    
    def _save_pickle(
        self,
        activations: Dict[str, Any],
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        streaming: bool = False
    ) -> str:
        """Save activations using pickle format."""
        # Create metadata
        meta = {
            "timestamp": np.datetime64('now').astype(str),
            "backend": "pickle",
            "format": self.config.format.name,
            "compression": self.config.compression.name,
            "streaming": streaming
        }
        
        if metadata:
            meta.update(metadata)
        
        # Handle different modes
        if streaming:
            # Open file for streaming writes
            with open(file_path, 'wb') as f:
                # Write header
                header = {
                    "metadata": meta,
                    "format": "streaming"
                }
                
                # Compress if needed
                if self.config.compression != CompressionLevel.NONE:
                    pickled_header = pickle.dumps(header)
                    compressed_header = zlib.compress(pickled_header, self.config.compression.value)
                    f.write(len(compressed_header).to_bytes(8, byteorder='little'))
                    f.write(compressed_header)
                else:
                    pickle.dump(header, f)
                
                # Write batches
                batch_count = 0
                
                # Iterate through generator if provided
                if isinstance(activations, Generator):
                    for batch_data in activations:
                        self._write_batch(f, batch_data, batch_count)
                        batch_count += 1
                else:
                    # Single batch
                    self._write_batch(f, activations, 0)
                    batch_count = 1
                
                # Update header with batch count
                f.seek(0)
                header["metadata"]["batch_count"] = batch_count
                
                # Write updated header
                if self.config.compression != CompressionLevel.NONE:
                    pickled_header = pickle.dumps(header)
                    compressed_header = zlib.compress(pickled_header, self.config.compression.value)
                    f.write(len(compressed_header).to_bytes(8, byteorder='little'))
                    f.write(compressed_header)
                else:
                    pickle.dump(header, f)
        else:
            # In-memory mode
            data_to_save = {
                "metadata": meta,
                "activations": activations
            }
            
            # Save to file
            with open(file_path, 'wb') as f:
                if self.config.compression != CompressionLevel.NONE:
                    pickled_data = pickle.dumps(data_to_save)
                    compressed_data = zlib.compress(pickled_data, self.config.compression.value)
                    f.write(len(compressed_data).to_bytes(8, byteorder='little'))
                    f.write(compressed_data)
                else:
                    pickle.dump(data_to_save, f)
        
        logger.info(f"Saved activations to {file_path} using pickle backend")
        return file_path
    
    def _write_batch(
        self,
        file: BinaryIO,
        batch_data: Dict[str, Any],
        batch_idx: int
    ) -> None:
        """Write a batch of activations to a file."""
        # Ensure batch has metadata
        if 'metadata' not in batch_data:
            batch_data['metadata'] = {}
        
        # Add batch index
        batch_data['metadata']['batch_idx'] = batch_idx
        
        # Write batch
        if self.config.compression != CompressionLevel.NONE:
            pickled_batch = pickle.dumps(batch_data)
            compressed_batch = zlib.compress(pickled_batch, self.config.compression.value)
            file.write(len(compressed_batch).to_bytes(8, byteorder='little'))
            file.write(compressed_batch)
        else:
            pickle.dump(batch_data, file)
    
    def _save_hdf5(
        self,
        activations: Dict[str, Any],
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        streaming: bool = False
    ) -> str:
        """Save activations using HDF5 format."""
        # Create metadata
        meta = {
            "timestamp": np.datetime64('now').astype(str),
            "backend": "hdf5",
            "format": self.config.format.name,
            "compression": self.config.compression.name,
            "streaming": streaming
        }
        
        if metadata:
            meta.update(metadata)
        
        # Convert metadata to json-compatible format
        json_meta = {k: v if isinstance(v, (int, float, str, bool, list, dict)) else str(v) 
                   for k, v in meta.items()}
        
        # Set compression options
        compression = None
        compression_opts = None
        if self.config.compression != CompressionLevel.NONE:
            compression = 'gzip'
            compression_opts = self.config.compression.value
        
        # Create HDF5 file
        with h5py.File(file_path, 'w') as f:
            # Store metadata
            f.attrs['metadata'] = json.dumps(json_meta)
            
            # Create metadata group
            meta_group = f.create_group('metadata')
            for k, v in json_meta.items():
                if isinstance(v, (int, float, str, bool)):
                    meta_group.attrs[k] = v
                elif isinstance(v, (list, dict)):
                    meta_group.attrs[k] = json.dumps(v)
            
            # Handle different modes
            if streaming:
                # Create activations group
                act_group = f.create_group('activations')
                batch_count = 0
                
                # Iterate through generator if provided
                if isinstance(activations, Generator):
                    for batch_idx, batch_data in enumerate(activations):
                        # Create batch group
                        batch_group = act_group.create_group(f'batch_{batch_idx}')
                        
                        # Save batch metadata
                        if 'metadata' in batch_data:
                            batch_meta = batch_data['metadata']
                            for k, v in batch_meta.items():
                                if isinstance(v, (int, float, str, bool)):
                                    batch_group.attrs[k] = v
                                elif isinstance(v, (list, dict)):
                                    batch_group.attrs[k] = json.dumps(v)
                        
                        # Save batch activations
                        if 'activations' in batch_data:
                            batch_acts = batch_data['activations']
                        else:
                            # Assume the whole batch is activations if no 'activations' key
                            batch_acts = {k: v for k, v in batch_data.items() if k != 'metadata'}
                        
                        for layer, data in batch_acts.items():
                            # Convert to numpy if needed
                            if isinstance(data, torch.Tensor):
                                data = data.detach().cpu().numpy()
                            
                            # Save activation
                            if data is not None and isinstance(data, np.ndarray):
                                batch_group.create_dataset(
                                    layer, 
                                    data=data,
                                    compression=compression,
                                    compression_opts=compression_opts,
                                    chunks=True if self.config.chunk_size else None
                                )
                        
                        batch_count += 1
                
                # Update metadata with batch count
                meta_group.attrs['batch_count'] = batch_count
            else:
                # In-memory mode
                # Create activations group
                act_group = f.create_group('activations')
                
                # Save activations
                for layer, data in activations.items():
                    # Skip metadata
                    if layer == 'metadata':
                        continue
                    
                    # Convert to numpy if needed
                    if isinstance(data, torch.Tensor):
                        data = data.detach().cpu().numpy()
                    
                    # Save activation
                    if data is not None and isinstance(data, np.ndarray):
                        act_group.create_dataset(
                            layer, 
                            data=data,
                            compression=compression,
                            compression_opts=compression_opts,
                            chunks=True if self.config.chunk_size else None
                        )
        
        logger.info(f"Saved activations to {file_path} using HDF5 backend")
        return file_path
    
    def _save_numpy(
        self,
        activations: Dict[str, Any],
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        streaming: bool = False
    ) -> str:
        """Save activations using NumPy format."""
        # For NumPy backend, we'll create a directory to store each layer separately
        # This is more efficient for large datasets and allows partial loading
        
        # Ensure file_path doesn't have an extension
        file_path = os.path.splitext(file_path)[0]
        
        # Create directory
        ensure_directory(file_path)
        
        # Create metadata
        meta = {
            "timestamp": np.datetime64('now').astype(str),
            "backend": "numpy",
            "format": self.config.format.name,
            "compression": self.config.compression.name,
            "streaming": streaming,
            "directory": file_path
        }
        
        if metadata:
            meta.update(metadata)
        
        # Save metadata
        with open(os.path.join(file_path, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2, default=str)
        
        # Handle different modes
        if streaming:
            # Create subdirectories for batches
            batches_dir = os.path.join(file_path, 'batches')
            ensure_directory(batches_dir)
            
            batch_count = 0
            
            # Iterate through generator if provided
            if isinstance(activations, Generator):
                for batch_idx, batch_data in enumerate(activations):
                    # Create batch directory
                    batch_dir = os.path.join(batches_dir, f'batch_{batch_idx}')
                    ensure_directory(batch_dir)
                    
                    # Save batch metadata
                    if 'metadata' in batch_data:
                        batch_meta = batch_data['metadata']
                        with open(os.path.join(batch_dir, 'metadata.json'), 'w') as f:
                            json.dump(batch_meta, f, indent=2, default=str)
                    
                    # Save batch activations
                    if 'activations' in batch_data:
                        batch_acts = batch_data['activations']
                    else:
                        # Assume the whole batch is activations if no 'activations' key
                        batch_acts = {k: v for k, v in batch_data.items() if k != 'metadata'}
                    
                    for layer, data in batch_acts.items():
                        # Convert to numpy if needed
                        if isinstance(data, torch.Tensor):
                            data = data.detach().cpu().numpy()
                        
                        # Save activation
                        if data is not None and isinstance(data, np.ndarray):
                            np.save(os.path.join(batch_dir, f'{layer}.npy'), data)
                    
                    batch_count += 1
            
            # Update metadata with batch count
            meta['batch_count'] = batch_count
            with open(os.path.join(file_path, 'metadata.json'), 'w') as f:
                json.dump(meta, f, indent=2, default=str)
        else:
            # In-memory mode
            # Create layers directory
            layers_dir = os.path.join(file_path, 'layers')
            ensure_directory(layers_dir)
            
            # Save activations
            for layer, data in activations.items():
                # Skip metadata
                if layer == 'metadata':
                    continue
                
                # Convert to numpy if needed
                if isinstance(data, torch.Tensor):
                    data = data.detach().cpu().numpy()
                
                # Save activation
                if data is not None and isinstance(data, np.ndarray):
                    np.save(os.path.join(layers_dir, f'{layer}.npy'), data)
        
        logger.info(f"Saved activations to {file_path} using NumPy backend")
        return file_path
    
    def load(
        self,
        file_path: str,
        layers: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
        metadata_only: bool = False,
        concat_batches: bool = True
    ) -> Dict[str, Any]:
        """
        Load activations from storage.
        
        Args:
            file_path: Path to load from
            layers: Specific layers to load (None for all)
            max_batches: Maximum number of batches to load
            metadata_only: Whether to load only metadata
            concat_batches: Whether to concatenate batches for streaming files
            
        Returns:
            Dictionary with loaded activations
        """
        # Determine backend based on file extension or directory
        if os.path.isdir(file_path):
            # Check if this is a NumPy storage directory
            if os.path.exists(os.path.join(file_path, 'metadata.json')):
                return self._load_numpy(file_path, layers, max_batches, metadata_only, concat_batches)
            else:
                raise ValueError(f"Unknown directory format: {file_path}")
        elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
            return self._load_hdf5(file_path, layers, max_batches, metadata_only, concat_batches)
        else:
            # Assume pickle by default
            return self._load_pickle(file_path, layers, max_batches, metadata_only, concat_batches)
    
    def _load_pickle(
        self,
        file_path: str,
        layers: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
        metadata_only: bool = False,
        concat_batches: bool = True
    ) -> Dict[str, Any]:
        """Load activations from pickle format."""
        with open(file_path, 'rb') as f:
            # Try to detect if the file is compressed
            is_compressed = False
            try:
                # Read first 8 bytes as length
                length_bytes = f.read(8)
                if len(length_bytes) == 8:
                    length = int.from_bytes(length_bytes, byteorder='little')
                    compressed_data = f.read(length)
                    if len(compressed_data) == length:
                        # Try to decompress
                        try:
                            decompressed_data = zlib.decompress(compressed_data)
                            header = pickle.loads(decompressed_data)
                            is_compressed = True
                        except:
                            # Not compressed or invalid compression
                            f.seek(0)
                            header = pickle.load(f)
                    else:
                        # Not enough data for compression
                        f.seek(0)
                        header = pickle.load(f)
                else:
                    # Not enough data for length
                    f.seek(0)
                    header = pickle.load(f)
            except:
                # Reset and try normal pickle
                f.seek(0)
                header = pickle.load(f)
            
            # Check if this is a streaming file
            is_streaming = False
            if isinstance(header, dict) and header.get("format") == "streaming":
                is_streaming = True
                metadata = header["metadata"]
                
                # Return early if only metadata requested
                if metadata_only:
                    return {"metadata": metadata}
                
                # Initialize result structure
                result = {
                    "metadata": metadata,
                    "batches": []
                }
                
                if concat_batches:
                    result["activations"] = {}
                
                # Load batch data
                batch_count = 0
                max_batch_count = max_batches or float('inf')
                
                try:
                    while batch_count < max_batch_count:
                        # Load next batch
                        if is_compressed:
                            try:
                                length_bytes = f.read(8)
                                if len(length_bytes) != 8:
                                    break  # End of file
                                    
                                length = int.from_bytes(length_bytes, byteorder='little')
                                compressed_batch = f.read(length)
                                if len(compressed_batch) != length:
                                    break  # Incomplete data
                                    
                                decompressed_batch = zlib.decompress(compressed_batch)
                                batch_data = pickle.loads(decompressed_batch)
                            except:
                                break  # Error reading batch
                        else:
                            try:
                                batch_data = pickle.load(f)
                            except EOFError:
                                break  # End of file
                        
                        # Filter layers if needed
                        if layers and 'activations' in batch_data:
                            batch_data['activations'] = {
                                k: v for k, v in batch_data['activations'].items() 
                                if k in layers
                            }
                        
                        # Store batch
                        if not concat_batches:
                            result["batches"].append(batch_data)
                        
                        # Concatenate activations if requested
                        if concat_batches and 'activations' in batch_data:
                            for layer, activation in batch_data['activations'].items():
                                if layer not in result["activations"]:
                                    result["activations"][layer] = []
                                result["activations"][layer].append(activation)
                        
                        batch_count += 1
                except Exception as e:
                    logger.warning(f"Error loading batch {batch_count}: {e}")
                
                # Convert lists to arrays if possible
                if concat_batches:
                    for layer, activations in result["activations"].items():
                        if activations:
                            # Try to concatenate numpy arrays
                            if all(isinstance(a, np.ndarray) for a in activations):
                                try:
                                    result["activations"][layer] = np.concatenate(activations, axis=0)
                                except Exception as e:
                                    logger.warning(f"Could not concatenate activations for layer {layer}: {e}")
                            # Try to concatenate torch tensors
                            elif all(isinstance(a, torch.Tensor) for a in activations):
                                try:
                                    result["activations"][layer] = torch.cat(activations, dim=0)
                                except Exception as e:
                                    logger.warning(f"Could not concatenate activations for layer {layer}: {e}")
                
                return result
            else:
                # Regular file (all data in memory)
                if is_compressed:
                    # Already loaded the header
                    data = header
                else:
                    # Reset and load full data
                    f.seek(0)
                    data = pickle.load(f)
                
                # Handle legacy format
                if not isinstance(data, dict) or "metadata" not in data:
                    # Legacy format or unexpected format
                    logger.warning("Found legacy or unexpected activation format")
                    
                    if isinstance(data, dict) and not ("activations" in data):
                        # Assume the whole dict is activations
                        activations = data
                    else:
                        activations = data.get("activations", data)
                    
                    # Filter layers if needed
                    if layers:
                        activations = {k: v for k, v in activations.items() if k in layers}
                    
                    # Wrap in standard format
                    return {
                        "metadata": {
                            "timestamp": np.datetime64('now').astype(str),
                            "format": "legacy",
                            "backend": "pickle"
                        },
                        "activations": activations
                    }
                
                # Return metadata if requested
                if metadata_only:
                    return {"metadata": data["metadata"]}
                
                # Filter layers if needed
                if layers and "activations" in data:
                    data["activations"] = {
                        k: v for k, v in data["activations"].items() 
                        if k in layers
                    }
                
                return data
    
    def _load_hdf5(
        self,
        file_path: str,
        layers: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
        metadata_only: bool = False,
        concat_batches: bool = True
    ) -> Dict[str, Any]:
        """Load activations from HDF5 format."""
        with h5py.File(file_path, 'r') as f:
            # Load metadata
            metadata = {}
            
            # Load from attributes
            if 'metadata' in f.attrs:
                try:
                    metadata = json.loads(f.attrs['metadata'])
                except:
                    metadata = dict(f.attrs.items())
            
            # Load from metadata group
            if 'metadata' in f:
                meta_group = f['metadata']
                for k, v in meta_group.attrs.items():
                    # Try to parse JSON values
                    if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                        try:
                            metadata[k] = json.loads(v)
                        except:
                            metadata[k] = v
                    else:
                        metadata[k] = v
            
            # Return early if only metadata requested
            if metadata_only:
                return {"metadata": metadata}
            
            # Check if this is a streaming file
            is_streaming = metadata.get('streaming', False)
            
            if is_streaming and 'activations' in f and 'batch_0' in f['activations']:
                # Streaming file
                result = {
                    "metadata": metadata,
                    "batches": []
                }
                
                if concat_batches:
                    result["activations"] = {}
                
                # Determine batch count
                batch_count = metadata.get('batch_count', 0)
                if batch_count == 0:
                    # Count batches
                    batch_count = sum(1 for k in f['activations'].keys() if k.startswith('batch_'))
                
                # Load batches
                max_batch = min(batch_count, max_batches or float('inf'))
                
                for batch_idx in range(int(max_batch)):
                    batch_key = f'batch_{batch_idx}'
                    if batch_key not in f['activations']:
                        continue
                    
                    batch_group = f['activations'][batch_key]
                    
                    # Load batch metadata
                    batch_meta = {}
                    for k, v in batch_group.attrs.items():
                        if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                            try:
                                batch_meta[k] = json.loads(v)
                            except:
                                batch_meta[k] = v
                        else:
                            batch_meta[k] = v
                    
                    # Load batch activations
                    batch_acts = {}
                    for layer in batch_group.keys():
                        if layers is None or layer in layers:
                            batch_acts[layer] = batch_group[layer][()]
                    
                    # Store batch
                    batch_data = {
                        "metadata": batch_meta,
                        "activations": batch_acts
                    }
                    
                    if not concat_batches:
                        result["batches"].append(batch_data)
                    
                    # Concatenate activations if requested
                    if concat_batches:
                        for layer, activation in batch_acts.items():
                            if layer not in result["activations"]:
                                result["activations"][layer] = []
                            result["activations"][layer].append(activation)
                
                # Convert lists to arrays if possible
                if concat_batches:
                    for layer, activations in result["activations"].items():
                        if activations:
                            try:
                                result["activations"][layer] = np.concatenate(activations, axis=0)
                            except Exception as e:
                                logger.warning(f"Could not concatenate activations for layer {layer}: {e}")
                
                return result
            else:
                # Regular file
                result = {
                    "metadata": metadata,
                    "activations": {}
                }
                
                # Load activations
                if 'activations' in f:
                    act_group = f['activations']
                    for layer in act_group.keys():
                        if layers is None or layer in layers:
                            result["activations"][layer] = act_group[layer][()]
                
                return result
    
    def _load_numpy(
        self,
        directory: str,
        layers: Optional[List[str]] = None,
        max_batches: Optional[int] = None,
        metadata_only: bool = False,
        concat_batches: bool = True
    ) -> Dict[str, Any]:
        """Load activations from NumPy format."""
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise ValueError(f"No metadata found in directory: {directory}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Return early if only metadata requested
        if metadata_only:
            return {"metadata": metadata}
        
        # Check if this is a streaming file
        is_streaming = metadata.get('streaming', False)
        
        if is_streaming and os.path.exists(os.path.join(directory, 'batches')):
            # Streaming file
            result = {
                "metadata": metadata,
                "batches": []
            }
            
            if concat_batches:
                result["activations"] = {}
            
            # List batch directories
            batches_dir = os.path.join(directory, 'batches')
            batch_dirs = [d for d in os.listdir(batches_dir) 
                        if os.path.isdir(os.path.join(batches_dir, d)) and d.startswith('batch_')]
            batch_dirs.sort(key=lambda d: int(d.split('_')[1]))
            
            # Limit batch count
            if max_batches:
                batch_dirs = batch_dirs[:max_batches]
            
            # Load batches
            for batch_dir_name in batch_dirs:
                batch_dir = os.path.join(batches_dir, batch_dir_name)
                
                # Load batch metadata
                batch_meta = {}
                batch_meta_path = os.path.join(batch_dir, 'metadata.json')
                if os.path.exists(batch_meta_path):
                    with open(batch_meta_path, 'r') as f:
                        batch_meta = json.load(f)
                
                # Load batch activations
                batch_acts = {}
                for file_name in os.listdir(batch_dir):
                    if file_name.endswith('.npy'):
                        layer = os.path.splitext(file_name)[0]
                        if layers is None or layer in layers:
                            batch_acts[layer] = np.load(os.path.join(batch_dir, file_name))
                
                # Store batch
                batch_data = {
                    "metadata": batch_meta,
                    "activations": batch_acts
                }
                
                if not concat_batches:
                    result["batches"].append(batch_data)
                
                # Concatenate activations if requested
                if concat_batches:
                    for layer, activation in batch_acts.items():
                        if layer not in result["activations"]:
                            result["activations"][layer] = []
                        result["activations"][layer].append(activation)
            
            # Convert lists to arrays if possible
            if concat_batches:
                for layer, activations in result["activations"].items():
                    if activations:
                        try:
                            result["activations"][layer] = np.concatenate(activations, axis=0)
                        except Exception as e:
                            logger.warning(f"Could not concatenate activations for layer {layer}: {e}")
            
            return result
        else:
            # Regular file
            result = {
                "metadata": metadata,
                "activations": {}
            }
            
            # Load activations from layers directory
            layers_dir = os.path.join(directory, 'layers')
            if os.path.exists(layers_dir):
                for file_name in os.listdir(layers_dir):
                    if file_name.endswith('.npy'):
                        layer = os.path.splitext(file_name)[0]
                        if layers is None or layer in layers:
                            result["activations"][layer] = np.load(os.path.join(layers_dir, file_name))
            
            return result
    
    def list_activations(
        self,
        directory: Optional[str] = None,
        pattern: str = "*"
    ) -> List[Dict[str, Any]]:
        """
        List available activation files.
        
        Args:
            directory: Directory to search in (None for default)
            pattern: Pattern to match files
            
        Returns:
            List of activation file information
        """
        # Set search directory
        search_dir = directory or self.config.base_dir or self.config.temp_dir
        if not search_dir:
            return []
        
        result = []
        
        # Search for files
        for root, dirs, files in os.walk(search_dir):
            # Check for HDF5 and pickle files
            for file in files:
                if file.endswith('.h5') or file.endswith('.hdf5') or file.endswith('.pkl') or file.endswith('.act'):
                    # Load metadata only
                    file_path = os.path.join(root, file)
                    try:
                        metadata = self.load(file_path, metadata_only=True).get("metadata", {})
                        result.append({
                            "path": file_path,
                            "filename": file,
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.warning(f"Error loading metadata from {file_path}: {e}")
            
            # Check for NumPy directories
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                metadata_path = os.path.join(dir_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        result.append({
                            "path": dir_path,
                            "filename": dir_name,
                            "metadata": metadata
                        })
                    except Exception as e:
                        logger.warning(f"Error loading metadata from {dir_path}: {e}")
        
        return result
    
    def merge(
        self,
        files: List[str],
        output_path: Optional[str] = None,
        output_backend: Optional[StorageBackend] = None
    ) -> str:
        """
        Merge multiple activation files.
        
        Args:
            files: List of files to merge
            output_path: Path for merged file (None for auto-generated)
            output_backend: Backend for merged file (None for config default)
            
        Returns:
            Path to the merged file
        """
        if not files:
            raise ValueError("No files provided for merging")
        
        # Use first file's metadata as base
        base_metadata = self.load(files[0], metadata_only=True).get("metadata", {})
        
        # Generate output path if not provided
        if not output_path:
            timestamp = int(time.time())
            file_suffix = ".act"
            if output_backend == StorageBackend.HDF5:
                file_suffix = ".h5"
            
            output_path = os.path.join(
                self.config.base_dir or self.config.temp_dir,
                f"merged_activations_{timestamp}{file_suffix}"
            )
        
        # Set backend
        backend = output_backend or self.config.backend
        
        # Load and merge activations
        merged_activations = {}
        
        for file_path in files:
            data = self.load(file_path)
            if "activations" in data:
                for layer, activation in data["activations"].items():
                    if layer not in merged_activations:
                        merged_activations[layer] = []
                    merged_activations[layer].append(activation)
        
        # Concatenate activations
        for layer, activations in merged_activations.items():
            if activations:
                if all(isinstance(a, np.ndarray) for a in activations):
                    try:
                        merged_activations[layer] = np.concatenate(activations, axis=0)
                    except Exception as e:
                        logger.warning(f"Could not concatenate activations for layer {layer}: {e}")
                elif all(isinstance(a, torch.Tensor) for a in activations):
                    try:
                        merged_activations[layer] = torch.cat(activations, dim=0)
                    except Exception as e:
                        logger.warning(f"Could not concatenate activations for layer {layer}: {e}")
        
        # Update metadata
        merged_metadata = {
            **base_metadata,
            "timestamp": np.datetime64('now').astype(str),
            "merged_from": [os.path.basename(f) for f in files],
            "merged_count": len(files)
        }
        
        # Save merged data
        saved_path = self.save(
            activations=merged_activations,
            file_path=output_path,
            metadata=merged_metadata
        )
        
        return saved_path