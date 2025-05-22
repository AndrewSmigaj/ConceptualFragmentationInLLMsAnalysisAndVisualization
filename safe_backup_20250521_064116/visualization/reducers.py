"""
Dimensionality reduction module for the concept fragmentation visualization.

This module contains functionality for reducing the dimensionality of
neural network layer activations using UMAP, with disk caching to
avoid recomputing embeddings.
"""

import os
import sys
import hashlib
import pickle
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

try:
    import umap
    from umap import UMAP
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
    UMAP = None

# Add parent directory to path to import data_interface
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualization.data_interface import load_activations, load_stats, get_best_config, get_baseline_config

class Embedder:
    """
    Class for dimensionality reduction of layer activations.
    Handles UMAP embedding with consistent parameters and disk caching.
    """
    
    def __init__(self, 
                 n_components: int = 3, 
                 n_neighbors: int = 15, 
                 min_dist: float = 0.1,
                 metric: str = "euclidean",
                 random_state: int = 42,
                 cache_dir: Optional[str] = None):
        """
        Initialize the embedder.
        
        Args:
            n_components: Number of dimensions to reduce to.
            n_neighbors: Number of neighbors for UMAP.
            min_dist: Minimum distance for UMAP.
            metric: Distance metric for UMAP.
            random_state: Random state for reproducibility.
            cache_dir: Directory to store cached embeddings.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        self.cache_dir = cache_dir
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize UMAP
        if UMAP is not None:
            self.umap = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )
        else:
            self.umap = None
    
    def _get_cache_path(self, data: np.ndarray) -> str:
        """
        Get the cache path for the given data.
        
        Args:
            data: Input data array.
            
        Returns:
            Path to the cache file.
        """
        # Add diagnostic prints to identify problematic data
        print(f"DEBUG _get_cache_path: type={type(data)}")
        if not isinstance(data, np.ndarray):
            print(f"WARNING: data is not ndarray but {type(data)}")
            if isinstance(data, list):
                print(f"  List length: {len(data)}")
                if data:
                    print(f"  First element type: {type(data[0])}")
        
        # Ensure we have a proper numeric array that supports .size
        if not isinstance(data, np.ndarray):
            print(f"Warning: Converting non-ndarray of type {type(data)} to numpy array")
            data = np.asarray(data)
        
        # Handle object-dtype arrays that might contain lists
        if data.dtype == object:
            print(f"Warning: Converting object-dtype array to numeric array")
            try:
                # Try to stack into a 2D array if all elements are same length
                data = np.vstack(data)
            except Exception as e:
                # Fall back: flatten everything into 1D array
                try:
                    data = np.hstack([np.asarray(x).ravel() for x in data])
                except Exception as e2:
                    print(f"Warning: Could not convert object array: {e2}")
                    # Last resort - just use a small sample that won't crash
                    data = np.zeros((10, 10))
        
        # Create a hash of the data and parameters
        # We only hash a subset of the data to avoid memory issues with large arrays
        # and still have a reasonable chance of detecting changes
        data_sample = data
        if data.size > 1000:
            indices = np.linspace(0, data.size - 1, 1000, dtype=int)
            data_sample = data.ravel()[indices]
        
        params_str = f"{self.n_components}_{self.n_neighbors}_{self.min_dist}_{self.metric}_{self.random_state}"
        hash_input = str(data_sample.shape) + str(data_sample.mean()) + str(data_sample.std()) + params_str
        
        # Create a hash of the input
        hash_obj = hashlib.md5(hash_input.encode())
        hash_str = hash_obj.hexdigest()
        
        # Return the cache path
        return os.path.join(self.cache_dir, f"embedding_{hash_str}.npz")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to the data.
        
        Args:
            data: Input data array.
            
        Returns:
            Embedded data in lower-dimensional space. Should be a 2D NumPy array.
        """
        if self.umap is None:
            raise ImportError("UMAP is not installed. Install with: pip install umap-learn")
        
        cache_path = self._get_cache_path(data)
        embedding_result: Any = None # Temporary holder for type flexibility

        if os.path.exists(cache_path):
            print(f"Loading cached embedding from {cache_path}")
            try:
                # Deny pickles from cache for safety and to ensure ndarray loading
                with np.load(cache_path, allow_pickle=False) as npz:
                    embedding_result = npz["embedding"]
                print(f"Loaded embedding from cache. Type: {type(embedding_result)}, Shape: {getattr(embedding_result, 'shape', 'N/A')}, Dtype: {getattr(embedding_result, 'dtype', 'N/A')}")
            except Exception as e:
                print(f"Error loading cache (or pickle denied): {e}. Recomputing embedding...")
                embedding_result = None # Ensure recomputation
        
        if embedding_result is None: # Not loaded from cache or cache loading failed
            print(f"Computing UMAP embedding for data with shape {data.shape}...")
            embedding_result = self.umap.fit_transform(data)
            # source_for_log = "UMAP computation" # Removed as it's not used after this block
            print(f"UMAP computation finished. Type: {type(embedding_result)}, Shape: {getattr(embedding_result, 'shape', 'N/A')}, Dtype: {getattr(embedding_result, 'dtype', 'N/A')}")
            
            embedding_to_cache = embedding_result
            if not isinstance(embedding_to_cache, np.ndarray):
                print(f"Warning: UMAP computed embedding is type {type(embedding_to_cache)}. Attempting to convert to np.ndarray before caching.")
                try:
                    embedding_to_cache = np.array(embedding_to_cache)
                except Exception as conversion_err:
                    raise TypeError(f"Failed to convert UMAP output of type {type(embedding_to_cache)} to np.ndarray for caching: {conversion_err}") from conversion_err
            
            if not isinstance(embedding_to_cache, np.ndarray): 
                 raise TypeError(f"Internal error: UMAP output (after potential conversion) is not an np.ndarray for caching. Type: {type(embedding_to_cache)}")

            print(f"Caching embedding (shape {embedding_to_cache.shape}, dtype {embedding_to_cache.dtype}) to {cache_path}")
            np.savez_compressed(cache_path, embedding=embedding_to_cache)
        
        # Final conversion and validation step for the returned embedding
        if not isinstance(embedding_result, np.ndarray):
            original_type = type(embedding_result)
            # Corrected source_description logic
            source_description = "cache" if os.path.exists(cache_path) and 'npz' in locals() and embedding_result is npz.get("embedding", None) else "UMAP computation"
            if embedding_result is None and os.path.exists(cache_path): # If cache load failed and embedding_result is None, it will be from UMAP
                 source_description = "UMAP computation (after failed cache load)"

            print(f"Warning: Embedding result (from {source_description}) is type {original_type}, not np.ndarray. Attempting conversion.")
            try:
                embedding_result = np.array(embedding_result)
            except Exception as final_conv_err:
                raise TypeError(f"The embedding result (type {original_type} from {source_description}) could not be converted to np.ndarray before returning: {final_conv_err}") from final_conv_err

        if not isinstance(embedding_result, np.ndarray): # Must be an ndarray now
            raise TypeError(f"Critical: Final embedding is not an np.ndarray after all checks. Type: {type(embedding_result)}. Problematic source was likely cache or UMAP output.")

        if embedding_result.ndim != 2:
            raise ValueError(f"Final embedding must be 2D, but got {embedding_result.ndim} dimensions. Shape: {embedding_result.shape}, Dtype: {embedding_result.dtype}")
        
        if embedding_result.dtype == object:
            raise ValueError(f"Final embedding has dtype 'object', which is invalid for numeric embeddings. Shape: {embedding_result.shape}")
            
        return embedding_result

def embed_layer_activations(
    dataset: str, 
    config: Dict[str, Any], 
    seed: int,
    layers: Optional[List[str]] = None,
    embedder: Optional[Embedder] = None,
    use_test_set: bool = True,  # Whether to use test or train set
    epoch_idx: int = -1  # Which epoch to use (-1 means last epoch)
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """
    Embed layer activations for the given dataset, configuration, and seed.
    
    Args:
        dataset: Name of the dataset.
        config: Configuration dictionary.
        seed: Random seed.
        layers: List of layer names to embed. If None, embed all layers.
        embedder: Embedder instance. If None, create a new one.
        use_test_set: Whether to use test set (True) or train set (False).
        epoch_idx: Which epoch to use (-1 for last epoch).
        
    Returns:
        Tuple containing:
        - Dictionary mapping layer names to embedded activations
        - NumPy array of true class labels for the samples (or None if not available)
    """
    # Add pre-load detailed diagnostic logging
    print(f"\n=== Embedding Layer Activations ===")
    print(f"Dataset: {dataset}")
    if isinstance(config, dict):
        print(f"Config: {config}")
    else:
        print(f"Config: {config}")
    print(f"Seed: {seed}")
    print(f"Using {'test' if use_test_set else 'train'} set")
    
    # Load activations with detailed info
    print(f"Loading activations from dataset {dataset}, config {config}, seed {seed}...")
    try:
        activations = load_activations(dataset, config, seed)
        print(f"Successfully loaded activations with {len(activations)} keys")
    except Exception as e:
        import traceback
        print(f"Error loading activations: {e}")
        print(traceback.format_exc())
        raise
    
    # Determine the number of epochs from 'epoch' key or layer dimensions
    n_epochs = None
    if 'epoch' in activations and isinstance(activations['epoch'], np.ndarray):
        n_epochs = len(activations['epoch'])
    
    # Determine which epoch to use
    actual_epoch_idx = None
    if n_epochs is not None:
        actual_epoch_idx = epoch_idx if epoch_idx >= 0 else n_epochs + epoch_idx
        print(f"Using epoch {actual_epoch_idx} of {n_epochs}")
    
    print(f"\nAvailable keys:")
    for key, value in activations.items():
        if isinstance(value, np.ndarray):
            print(f"  '{key}': numpy array with shape {value.shape}, dtype {value.dtype}")
        elif isinstance(value, dict) and 'test' in value and 'train' in value:
            test_val = value['test']
            train_val = value['train']
            print(f"  '{key}': dict with test/train, test shape: {type(test_val)}")
        else:
            print(f"  '{key}': {type(value)}")
    
    # Create embedder if not provided
    if embedder is None:
        embedder = Embedder()
    
    # Filter layers if specified
    layer_keys = [k for k in activations.keys() if (k.startswith('layer') or k == 'input') and isinstance(activations[k], dict)]
    if layers is not None:
        layer_keys = [k for k in layer_keys if k in layers]
    
    print(f"\nProcessing layer keys: {layer_keys}")
    
    # Extract the appropriate data (test/train) from each layer
    valid_activations = {}
    for layer in layer_keys:
        layer_data = activations[layer]
        if not isinstance(layer_data, dict) or 'test' not in layer_data or 'train' not in layer_data:
            print(f"Skipping layer '{layer}': unexpected format (not a dict with test/train)")
            continue
        
        # Get the appropriate split
        split_key = 'test' if use_test_set else 'train'
        data = layer_data[split_key]
        
        # Convert list to numpy array if needed
        if isinstance(data, list):
            print(f"DEBUG: Attempting to convert {layer} {split_key} list to numpy array...")
            print(f"DEBUG: Outer list length for {layer} {split_key}: {len(data)}")
            
            if not data:
                print(f"DEBUG: {layer} {split_key} data list is empty. Skipping.")
                continue

            # Detailed inspection of list elements
            is_homogeneous_based_on_inspection = True
            first_element_for_comparison = data[0]
            first_elem_type = type(first_element_for_comparison)
            expected_inner_shape = None
            print(f"DEBUG: {layer} {split_key} - First element type: {first_elem_type}")

            if isinstance(first_element_for_comparison, (list, np.ndarray)):
                try:
                    # Convert first element to np.array to get its shape reliably
                    first_elem_array = np.asarray(first_element_for_comparison)
                    expected_inner_shape = first_elem_array.shape
                    print(f"DEBUG: {layer} {split_key} - First element shape: {expected_inner_shape}, dtype: {first_elem_array.dtype}")
                except Exception as e:
                    print(f"DEBUG: {layer} {split_key} - Could not determine shape/dtype of first element: {e}")
                    is_homogeneous_based_on_inspection = False # Can't proceed with shape check

            for i, item in enumerate(data):
                current_item_type = type(item)
                if current_item_type != first_elem_type:
                    print(f"DEBUG: {layer} {split_key} - Type inhomogeneity at index {i}.")
                    print(f"  Expected type: {first_elem_type}, Got type: {current_item_type}")
                    print(f"  Item (first 100 chars): {str(item)[:100]}...")
                    is_homogeneous_based_on_inspection = False
                    break
                
                if expected_inner_shape is not None and isinstance(item, (list, np.ndarray)):
                    try:
                        current_item_array = np.asarray(item)
                        current_inner_shape = current_item_array.shape
                        if current_inner_shape != expected_inner_shape:
                            print(f"DEBUG: {layer} {split_key} - Shape inhomogeneity at index {i}.")
                            print(f"  Expected shape: {expected_inner_shape}, Got shape: {current_inner_shape}")
                            print(f"  Item (type {type(item)}, first 100 chars): {str(item)[:100]}...")
                            is_homogeneous_based_on_inspection = False
                            break
                        # Optionally, check dtype consistency too:
                        # if current_item_array.dtype != np.asarray(first_element_for_comparison).dtype:
                        #     print(f"DEBUG: {layer} {split_key} - Dtype inhomogeneity at index {i}...")
                        #     is_homogeneous_based_on_inspection = False
                        #     break
                    except Exception as e:
                        print(f"DEBUG: {layer} {split_key} - Could not determine shape of element at index {i}: {e}")
                        # This item might be problematic for np.array()
                        is_homogeneous_based_on_inspection = False # Treat as inhomogeneous if shape check fails
                        break
                elif expected_inner_shape is not None and not isinstance(item, (list, np.ndarray)):
                    # First element was array-like, but this one is not
                    print(f"DEBUG: {layer} {split_key} - Structural inhomogeneity at index {i}.")
                    print(f"  First element was list/array, but item {i} is type: {current_item_type}")
                    print(f"  Item (first 100 chars): {str(item)[:100]}...")
                    is_homogeneous_based_on_inspection = False
                    break
            
            if not is_homogeneous_based_on_inspection:
                print(f"DEBUG: {layer} {split_key} list appears inhomogeneous. np.array conversion below might fail or result in an object array.")

            try:
                original_data_list_ref = data # Keep ref for potential fallback or further debugging
                data = np.array(data)
                print(f"Successfully converted {layer} {split_key} list to numpy array with shape {data.shape} and dtype {data.dtype}")
                
                if data.dtype == object:
                    print(f"WARNING: {layer} {split_key} converted to numpy array with dtype 'object'. Shape: {data.shape}.")
                    print(f"  This often indicates an underlying inhomogeneous structure (e.g., ragged array).")
                    print(f"  UMAP requires a 2D numeric array and will likely fail with this object array.")
                    # Consider adding 'continue' here if object arrays are strictly disallowed
                    # For example:
                    # print(f"  Skipping layer {layer} {split_key} due to object dtype array.")
                    # continue
                    
            except Exception as e:
                print(f"Failed to convert {layer} {split_key} list to array during np.array(data) call: {e}")
                print(f"  This confirms the list structure is problematic for direct conversion to a standard NumPy numerical array.")
                print(f"  Please inspect the data generation process for '{layer}' activations, '{split_key}' split.")
                continue
        
        # Validate the data
        if not isinstance(data, np.ndarray):
            print(f"Skipping layer '{layer}': {split_key} data is not a numpy array (type: {type(data)})")
            continue
        
        if data.dtype == object:
            print(f"Skipping layer '{layer}': {split_key} data has object dtype")
            continue
        
        # Handle 3D data (epochs, samples, features)
        if data.ndim == 3:
            if actual_epoch_idx is None:
                # If we couldn't determine epoch from 'epoch' key, use the first dimension size
                n_epochs = data.shape[0]
                actual_epoch_idx = epoch_idx if epoch_idx >= 0 else n_epochs + epoch_idx
                print(f"Determined epochs from data: {n_epochs}, using epoch {actual_epoch_idx}")
            
            if actual_epoch_idx < 0 or actual_epoch_idx >= data.shape[0]:
                print(f"Invalid epoch index {actual_epoch_idx} for data with {data.shape[0]} epochs")
                continue
                
            print(f"Extracting epoch {actual_epoch_idx} from 3D data with shape {data.shape}")
            data = data[actual_epoch_idx]
            print(f"Extracted 2D data with shape {data.shape}")
        
        if data.ndim != 2:
            print(f"Skipping layer '{layer}': data is not 2D after processing (shape: {data.shape})")
            continue
        
        print(f"Using '{layer}' {split_key} data with shape {data.shape}")
        valid_activations[layer] = data
    
    # If no valid activations, try to use output
    if not valid_activations and 'output' in activations and isinstance(activations['output'], dict):
        split_key = 'test' if use_test_set else 'train'
        if split_key in activations['output']:
            data = activations['output'][split_key]
            
            # Convert list to numpy array if needed
            if isinstance(data, list):
                print(f"Converting output {split_key} list to numpy array...")
                try:
                    data = np.array(data)
                    print(f"Successfully converted to shape {data.shape}")
                except Exception as e:
                    print(f"Failed to convert output list to array: {e}")
                    data = None
            
            # Handle 3D data for output too
            if isinstance(data, np.ndarray) and data.ndim == 3:
                if actual_epoch_idx is None:
                    n_epochs = data.shape[0]
                    actual_epoch_idx = epoch_idx if epoch_idx >= 0 else n_epochs + epoch_idx
                
                if 0 <= actual_epoch_idx < data.shape[0]:
                    print(f"Extracting epoch {actual_epoch_idx} from output 3D data with shape {data.shape}")
                    data = data[actual_epoch_idx]
                    print(f"Extracted 2D data with shape {data.shape}")
            
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype != object:
                print(f"No layer data found, using 'output' {split_key} as fallback (shape: {data.shape})")
                valid_activations['output'] = data
    
    if not valid_activations:
        raise ValueError(f"No valid layer activations found for {dataset}, {config}, seed {seed}")
    
    # Extract true labels from activations if available
    true_labels = None
    split_key = 'test' if use_test_set else 'train'
    
    # Try to get labels from activations
    if "labels" in activations:
        if isinstance(activations["labels"], np.ndarray):
            true_labels = activations["labels"]
            print(f"Extracted true labels from activations array with shape {true_labels.shape}")
        elif isinstance(activations["labels"], dict) and split_key in activations["labels"]:
            labels_data = activations["labels"][split_key]
            
            # Handle 3D epochs data for labels
            if isinstance(labels_data, np.ndarray) and labels_data.ndim > 1:
                if actual_epoch_idx is not None and 0 <= actual_epoch_idx < labels_data.shape[0]:
                    true_labels = labels_data[actual_epoch_idx]
                    print(f"Extracted true labels for epoch {actual_epoch_idx} with shape {true_labels.shape}")
                else:
                    # Just take the last epoch if we couldn't determine actual_epoch_idx
                    true_labels = labels_data[-1]
                    print(f"Using labels from last epoch with shape {true_labels.shape}")
            elif isinstance(labels_data, list) and labels_data:
                # Handle labels as a list
                try:
                    if isinstance(labels_data[0], list) or isinstance(labels_data[0], np.ndarray):
                        # List of arrays/lists (epochs)
                        epoch_to_use = actual_epoch_idx if actual_epoch_idx is not None else -1
                        if epoch_to_use < 0:
                            epoch_to_use = len(labels_data) + epoch_to_use
                        if 0 <= epoch_to_use < len(labels_data):
                            true_labels = np.array(labels_data[epoch_to_use])
                            print(f"Extracted true labels from list for epoch {epoch_to_use} with shape {true_labels.shape}")
                    else:
                        # Simple list of labels
                        true_labels = np.array(labels_data)
                        print(f"Converted labels list to array with shape {true_labels.shape}")
                except Exception as e:
                    print(f"Error extracting labels from list: {e}")
            else:
                true_labels = labels_data
                print(f"Using labels data of type {type(true_labels)}")
    
    # Validate true_labels shape matches the data
    if true_labels is not None:
        # Get expected number of samples from the first layer
        first_layer = next(iter(valid_activations.values()))
        expected_samples = first_layer.shape[0]
        
        if len(true_labels) != expected_samples:
            print(f"WARNING: True labels shape ({len(true_labels)}) doesn't match data shape ({expected_samples})")
            print("Labels will not be used due to shape mismatch.")
            true_labels = None  # Don't use mismatched labels
    
    # Embed each layer
    embeddings = {}
    for layer, layer_activations in valid_activations.items():
        # Double-check before passing to fit_transform
        assert isinstance(layer_activations, np.ndarray), f"Layer {layer} is not ndarray: {type(layer_activations)}"
        assert layer_activations.dtype != object, f"Layer {layer} has object dtype"
        assert layer_activations.ndim == 2, f"Layer {layer} is not 2D: shape={layer_activations.shape}"
        
        print(f"Embedding layer {layer}...")
        embeddings[layer] = embedder.fit_transform(layer_activations)
    
    return embeddings, true_labels

def embed_all_configs(dataset: str, seeds: List[int] = [0, 1, 2], use_test_set: bool = True) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Embed all configurations (baseline and best) for the given dataset.
    
    Args:
        dataset: Name of the dataset.
        seeds: List of random seeds.
        use_test_set: Whether to use the test set (True) or train set (False)
        
    Returns:
        Nested dictionary mapping: config_name -> seed -> layer -> embedding.
    """
    # Get configurations
    baseline_config = get_baseline_config(dataset)
    best_config = get_best_config(dataset)
    
    # Create embedder
    embedder = Embedder()
    
    # Embed all configurations
    all_embeddings = {}
    
    # Baseline
    baseline_embeddings = {}
    for seed in seeds:
        print(f"Embedding baseline configuration for seed {seed}...")
        baseline_embeddings[seed] = embed_layer_activations(
            dataset, baseline_config, seed, embedder=embedder, use_test_set=use_test_set)
    all_embeddings["baseline"] = baseline_embeddings
    
    # Best config
    best_embeddings = {}
    for seed in seeds:
        print(f"Embedding best configuration for seed {seed}...")
        best_embeddings[seed] = embed_layer_activations(
            dataset, best_config, seed, embedder=embedder, use_test_set=use_test_set)
    all_embeddings["best"] = best_embeddings
    
    return all_embeddings

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Embed layer activations for visualization.")
    parser.add_argument("datasets", nargs="+", help="Datasets to process (e.g., titanic, heart)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to process")
    parser.add_argument("--cache-dir", type=str, help="Cache directory for embeddings")
    
    args = parser.parse_args()
    
    # Process each dataset
    for dataset in args.datasets:
        try:
            print(f"Processing dataset: {dataset}")
            
            # Create embedder with specified cache directory
            embedder_kwargs = {}
            if args.cache_dir is not None:
                embedder_kwargs["cache_dir"] = args.cache_dir
            
            # Embed all configurations
            embed_all_configs(dataset, seeds=args.seeds)
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")

if __name__ == "__main__":
    main() 