"""
Data interface module for the concept fragmentation visualization project.

This module contains functions for loading and processing data from the concept
fragmentation experiments, including statistics and layer activations.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define the best configurations as specified in the plan
BEST_CONFIGS = {
    "titanic": {"weight": 0.1, "temperature": 0.07, "threshold": 0.0, "layers": "L3"},
    "heart": {"weight": 0.1, "temperature": 0.07, "threshold": 0.0, "layers": "L3"},
}

# Will be updated from config.py
DATA_ROOT = "D:/concept_fragmentation_results"

def get_config_path() -> str:
    """Look for config.py in the repo to get the data root path."""
    # TEMPORARY: Hardcode the correct path for debugging
    hardcoded_path = "D:/concept_fragmentation_results"
    print(f"DEBUG: Using hardcoded data_root path: {hardcoded_path}")
    return hardcoded_path
    
    # Original implementation below - currently bypassed
    try:
        import sys
        import os
        import importlib
        
        # Add the repo root to sys.path if needed
        repo_root = Path(__file__).resolve().parents[1]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        
        # 1) Try to import config from repo root
        try:
            config_spec = importlib.util.spec_from_file_location(
                "config", os.path.join(repo_root_str, "config.py"))
            if config_spec:
                config_module = importlib.util.module_from_spec(config_spec)
                config_spec.loader.exec_module(config_module)
                if hasattr(config_module, "RESULTS_PATH"):
                    return config_module.RESULTS_PATH
        except Exception:
            pass
            
        # 2) Try to import from regular Python path
        try:
            import config
            if hasattr(config, "RESULTS_PATH"):
                return config.RESULTS_PATH
        except (ImportError, AttributeError):
            pass
            
        # 3) Try to get RESULTS_DIR from concept_fragmentation.config
        try:
            from concept_fragmentation.config import RESULTS_DIR
            return RESULTS_DIR
        except (ImportError, AttributeError):
            # Use the default path
            print("Warning: Could not find RESULTS_PATH in config.py. Using default.")
            return DATA_ROOT
    except Exception as e:
        print(f"Error loading config: {e}")
        return DATA_ROOT

def load_stats(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the statistics from the CSV file.
    
    Args:
        csv_path: Path to the CSV file. If None, use the default path.
        
    Returns:
        DataFrame containing the statistics.
    """
    if csv_path is None:
        # Try the local repo file first
        local_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 "results", "cohesion_stats.csv")
        if os.path.exists(local_path):
            csv_path = local_path
        else:
            # Use the production server path
            data_root = get_config_path()
            csv_path = os.path.join(data_root, "analysis", "cohesion_summary.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Statistics file not found: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if we have the necessary columns
    required_cols = ["dataset", "experiment_type", "weight", "reg_id", 
                    "final_entropy_layer1", "final_entropy_layer2", "final_entropy_layer3",
                    "final_angle_layer1", "final_angle_layer2", "final_angle_layer3"]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in statistics file: {missing_cols}")
    
    return df

def get_best_config(dataset: str) -> Dict[str, Any]:
    """
    Get the best configuration for the given dataset.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic", "heart").
        
    Returns:
        Dictionary containing the configuration parameters.
    """
    dataset = dataset.lower()
    if dataset not in BEST_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(BEST_CONFIGS.keys())}")
    
    return BEST_CONFIGS[dataset]

def get_config_id(config: Dict[str, Any]) -> str:
    """
    Generate a configuration ID from the configuration parameters.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        String identifier for the configuration.
    """
    weight = config.get("weight", 0.0)
    temp = config.get("temperature", 0.07)
    threshold = config.get("threshold", 0.0)
    layers = config.get("layers", "")
    
    # Check if this is a baseline configuration (weight=0 and no layers specified)
    if weight == 0.0 and not layers:
        return "baseline"
    elif layers:  # regularized model with layer specification
        # Handle different layer formats:
        # Convert "L3" format directly
        if layers.startswith("L"):
            layer_part = layers
        # Convert ["layer2", "layer3"] or "layer2,layer3" to "L2-3"
        elif isinstance(layers, list):
            # Extract numbers from layer names
            layer_nums = [l.replace("layer", "") for l in layers]
            layer_part = f"L{'-'.join(layer_nums)}"
        else:
            # Handle comma-separated string format
            if "," in layers:
                layer_list = layers.split(",")
                layer_nums = [l.strip().replace("layer", "") for l in layer_list]
                layer_part = f"L{'-'.join(layer_nums)}"
            else:
                # Single layer as string
                layer_part = f"L{layers.replace('layer', '')}"
        
        return f"w{weight}_t{temp}_thr{threshold}_{layer_part}"
    else:  # other non-baseline model without layers
        return f"w{weight}_t{temp}_thr{threshold}"

def inspect_activation_file(dataset: str, config: Dict[str, Any], seed: int) -> None:
    """
    Inspect the content of activation file for debugging.
    Print detailed information about each key and value in the activation file.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic", "heart").
        config: Configuration dictionary or configuration ID string.
        seed: Random seed used for the experiment.
    """
    data_root = get_config_path()
    
    # Handle config as string or dict
    if isinstance(config, str):
        config_id = config
    else:
        config_id = get_config_id(config)
    
    # Construct the path to the activations file
    path = os.path.join(data_root, "cohesion", dataset.lower(), config_id, 
                        f"seed_{seed}", "layer_activations.pkl")
    
    if not os.path.exists(path):
        print(f"ERROR: Activations file not found: {path}")
        return
    
    print(f"\n=== Inspecting activation file ===")
    print(f"Dataset: {dataset}")
    print(f"Config ID: {config_id}")
    print(f"Seed: {seed}")
    print(f"Path: {path}")
    
    # Load the pickle file
    try:
        with open(path, "rb") as f:
            activations = pickle.load(f)
        
        print(f"\nFile loaded successfully.")
        print(f"Keys in activation file: {list(activations.keys())}")
        
        for key, value in activations.items():
            print(f"\n== Key: {key} ==")
            print(f"  Type: {type(value)}")
            
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
                
                if value.dtype == object:
                    print(f"  Contains object data (possibly Python objects):")
                    for i, x in enumerate(value.flat[:3]):
                        print(f"    Element {i}: type={type(x)}, value={x}")
                    
                elif value.size > 0:
                    print(f"  Min: {np.min(value)}")
                    print(f"  Max: {np.max(value)}")
                    print(f"  Mean: {np.mean(value)}")
                
            elif isinstance(value, list):
                print(f"  Length: {len(value)}")
                if value:
                    print(f"  First element type: {type(value[0])}")
                    print(f"  First element value: {value[0]}")
                    if len(value) > 1:
                        print(f"  Second element type: {type(value[1])}")
                    if len(value) > 2:
                        print(f"  Third element type: {type(value[2])}")
            
            elif isinstance(value, dict):
                print(f"  Dictionary with keys: {list(value.keys())}")
            
            else:
                print(f"  Value: {value}")
    
    except Exception as e:
        print(f"Error inspecting file: {e}")

def load_activations(dataset: str, config: Dict[str, Any], seed: int) -> Dict[str, np.ndarray]:
    """
    Load the layer activations for the given dataset, configuration, and seed.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic", "heart").
        config: Configuration dictionary or configuration ID string.
        seed: Random seed used for the experiment.
        
    Returns:
        Dictionary mapping layer names to activation arrays.
    """
    data_root = get_config_path()
    print(f"DEBUG load_activations: data_root={data_root}")
    
    # Handle config as string or dict
    if isinstance(config, str):
        config_id = config
    else:
        config_id = get_config_id(config)
    print(f"DEBUG load_activations: config_id={config_id}")
    
    # Construct the path to the activations file
    path = os.path.join(data_root, "cohesion", dataset.lower(), config_id, 
                        f"seed_{seed}", "layer_activations.pkl")
    print(f"DEBUG load_activations: Attempting to load from {path}")
    
    if not os.path.exists(path):
        # Let's look for alternate paths
        print(f"DEBUG load_activations: Path not found, trying alternate locations")
        alternate_paths = [
            os.path.join(data_root, "cohesion", dataset.lower(), config_id, 
                         f"seed{seed}", "layer_activations.pkl"),
            os.path.join(data_root, "cohesion", dataset.lower(), 
                         f"{dataset.lower()}_{config_id}_seed{seed}", "layer_activations.pkl"),
            os.path.join(data_root, "cohesion", dataset.lower(), 
                         f"{dataset.lower()}_{config_id}_seed{seed}", "activations.pkl"),
            os.path.join(data_root, "baselines", dataset.lower(), 
                         f"{dataset.lower()}_baseline_seed{seed}", "layer_activations.pkl"),
            os.path.join(data_root, "baselines", dataset.lower(), 
                         f"{dataset.lower()}_baseline_seed{seed}", "activations.pkl")
        ]
        
        for alt_path in alternate_paths:
            print(f"DEBUG load_activations: Trying {alt_path}")
            if os.path.exists(alt_path):
                print(f"DEBUG load_activations: Found alternate path: {alt_path}")
                path = alt_path
                break
        
        if not os.path.exists(path):
            print(f"DEBUG load_activations: Listing all files in data_root:")
            try:
                import glob
                top_level_dirs = glob.glob(os.path.join(data_root, "*"))
                print(f"DEBUG load_activations: Top level dirs: {top_level_dirs}")
                if os.path.exists(os.path.join(data_root, "cohesion")):
                    print(f"DEBUG load_activations: Cohesion dir exists, checking contents:")
                    cohesion_contents = glob.glob(os.path.join(data_root, "cohesion", "*"))
                    print(f"DEBUG load_activations: Cohesion contents: {cohesion_contents}")
                    if os.path.exists(os.path.join(data_root, "cohesion", dataset.lower())):
                        dataset_contents = glob.glob(os.path.join(data_root, "cohesion", dataset.lower(), "*"))
                        print(f"DEBUG load_activations: Dataset dir contents: {dataset_contents}")
            except Exception as e:
                print(f"DEBUG load_activations: Error listing directories: {e}")
            
            raise FileNotFoundError(f"Activations file not found: {path}")
    
    # Load the pickle file
    try:
        print(f"DEBUG load_activations: Loading pickle file from {path}")
        with open(path, "rb") as f:
            activations = pickle.load(f)
        print(f"DEBUG load_activations: Successfully loaded pickle with {len(activations)} keys")
        
        # List all keys for debugging
        print(f"DEBUG load_activations: Keys in loaded pickle: {list(activations.keys())}")
        
        # Check for layer keys specifically
        layer_keys = [k for k in activations.keys() if k.startswith('layer')]
        print(f"DEBUG load_activations: Found {len(layer_keys)} layer keys: {layer_keys}")
        
        # Look for non-standard layer formats
        if not layer_keys:
            # Check if activation data is structured differently
            print(f"DEBUG load_activations: No standard layer keys found, looking for alternate formats")
            for key, value in activations.items():
                print(f"DEBUG load_activations: Key {key} is type {type(value)}")
                if isinstance(value, dict):
                    print(f"DEBUG load_activations: Dict key {key} has keys: {list(value.keys())}")
        
        # Convert Python lists to NumPy arrays where possible
        activations = clean_activations(activations)
        print(f"DEBUG load_activations: After cleaning, have {len(activations)} keys")
        
        return activations
    except Exception as e:
        import traceback
        print(f"DEBUG load_activations: Error during pickle load or processing: {e}")
        print(traceback.format_exc())
        raise

def clean_activations(activations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean up activations dictionary by converting lists to NumPy arrays where possible.
    
    Args:
        activations: Dictionary of activations, potentially containing lists.
        
    Returns:
        Dictionary with lists converted to NumPy arrays where possible.
    """
    cleaned = {}
    for key, value in activations.items():
        # Handle nested dictionaries with train/test keys
        if isinstance(value, dict) and 'train' in value and 'test' in value:
            cleaned_dict = {}
            for split in ['train', 'test']:
                split_data = value[split]
                if isinstance(split_data, list):
                    try:
                        # Try to convert list to array
                        arr = np.array(split_data)
                        if arr.dtype != object:
                            print(f"Converted '{key}.{split}' from list to numpy array with shape {arr.shape}")
                            cleaned_dict[split] = arr
                        else:
                            # Keep as is if conversion created object array
                            cleaned_dict[split] = split_data
                    except Exception as e:
                        print(f"Warning: Could not convert '{key}.{split}' list to array: {e}")
                        cleaned_dict[split] = split_data
                else:
                    # Keep as is if not a list
                    cleaned_dict[split] = split_data
            cleaned[key] = cleaned_dict
            continue
        
        # Keep as is if already a NumPy array
        if isinstance(value, np.ndarray):
            cleaned[key] = value
            continue
            
        # Try to convert lists to arrays
        if isinstance(value, list):
            try:
                # For uniform lists (like lists of floats/ints)
                arr = np.array(value)
                if arr.dtype != object:  # Only keep if conversion worked well
                    print(f"Converted '{key}' from list to numpy array with shape {arr.shape}")
                    cleaned[key] = arr
                    continue
                else:
                    # For lists that converted to object arrays, try more approaches
                    if len(value) > 0 and all(isinstance(item, (list, tuple)) for item in value):
                        # Try to make a 2D array if all elements are lists/tuples of same length
                        if all(len(item) == len(value[0]) for item in value):
                            try:
                                arr_2d = np.array([np.array(item) for item in value])
                                print(f"Converted '{key}' list-of-lists to 2D array with shape {arr_2d.shape}")
                                cleaned[key] = arr_2d
                                continue
                            except:
                                pass
            except Exception as e:
                print(f"Warning: Could not convert '{key}' list to array: {e}")
                
            # Keep the original list if conversion failed
            print(f"Keeping '{key}' as original list")
            cleaned[key] = value
        else:
            # Keep other types as they are
            cleaned[key] = value
            
    return cleaned

def select_samples(
    dataset: str,
    config: Union[Dict[str, Any], str],
    seed: int,
    k_frag: int = 20,
    k_norm: int = 20,
    layer_clusters: Optional[Dict[str, Dict[str, Any]]] = None,
    actual_class_labels: Optional[np.ndarray] = None
) -> List[int]:
    """
    Select sample indices with high and low fragmentation.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string
        seed: Random seed used
        k_frag: Number of high-fragmentation samples to select
        k_norm: Number of normal (low-fragmentation) samples to select
        layer_clusters: Pre-computed embedded cluster information for a specific config.
                        If provided, compute_fractured_off_scores_embedded is used.
                        Structure: {layer_name: {"k": k, "centers": centers, "labels": labels}}
        actual_class_labels: True class labels for the samples (if available)
    
    Returns:
        List of sample indices (high fragmentation first, then low)
    """
    print(f"DEBUG: select_samples called with k_frag={k_frag}, k_norm={k_norm}")
    print(f"DEBUG: actual_class_labels type: {type(actual_class_labels)}")
    if isinstance(actual_class_labels, np.ndarray):
        print(f"DEBUG: actual_class_labels shape: {actual_class_labels.shape}")
        print(f"DEBUG: Sample of actual_class_labels: {actual_class_labels[:5]} (first 5)")

    # For embedded clusters, use the new embedded approach
    if layer_clusters is not None:
        print("DEBUG: Using embedded layer clusters approach")
        try:
            # Compute scores using the embedded space clusters
            frag_scores = compute_fractured_off_scores_embedded(
                dataset, config, seed, layer_clusters, actual_class_labels
            )
            print(f"DEBUG: Got frag_scores: {list(frag_scores.keys())[:5]} (first 5 keys)")
            
            if not frag_scores:
                print("DEBUG: Empty frag_scores, returning empty list")
                return []
            
            # Sort samples by fragmentation score (highest first)
            sorted_samples = sorted(frag_scores.keys(), key=lambda s: frag_scores[s], reverse=True)
            
            # Select top k_frag samples with highest scores
            high_frag_samples = sorted_samples[:k_frag]
            
            # Select k_norm samples with lowest scores
            low_frag_samples = sorted_samples[-k_norm:] if k_norm > 0 else []
            
            selected = high_frag_samples + low_frag_samples
            print(f"DEBUG: Returning {len(selected)} selected samples")
            return selected
        
        except Exception as e:
            import traceback
            print(f"DEBUG: Error in embedded cluster approach: {e}")
            print(f"DEBUG: {traceback.format_exc()}")
            return []
            
    # Previous raw activations approach
    print("DEBUG: Using raw activations approach (fallback)")
    try:
        # Compute clusters using raw activations
        layer_clusters_raw = compute_layer_clusters(dataset, config, seed)
        
        # Compute fragmentation scores
        from concept_fragmentation.utils.metrics import compute_fractured_off_scores
        frag_scores = compute_fractured_off_scores(dataset, config, seed, layer_clusters_raw)
        
        if not frag_scores:
            return []
        
        # Sort samples by fragmentation score
        sorted_samples = sorted(frag_scores.keys(), key=lambda s: frag_scores[s], reverse=True)
        
        # Select top k_frag samples with highest scores
        high_frag_samples = sorted_samples[:k_frag]
        
        # Select k_norm samples with lowest scores
        low_frag_samples = sorted_samples[-k_norm:] if k_norm > 0 else []
        
        return high_frag_samples + low_frag_samples
    
    except Exception as e:
        print(f"Error in raw activation approach: {e}")
        return []

def compute_layer_clusters(dataset: str, config: Union[Dict[str, Any], str], seed: int, max_k: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Compute optimal k-means clusters for each layer based on silhouette score.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic", "heart")
        config: Configuration dictionary or config_id string
        seed: Random seed used for the experiment
        max_k: Maximum number of clusters to try (default: 10)
        
    Returns:
        Dictionary mapping layer names to dictionaries containing:
            - k: optimal number of clusters
            - centers: array of cluster centers (shape: k x 3)
            - labels: array of cluster assignments for each sample
    """
    # Check cache first
    cache_path = _get_clusters_cache_path(dataset, config, seed, max_k)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                layer_clusters = pickle.load(f)
            print(f"Loaded cached clusters from {cache_path}")
            return layer_clusters
        except Exception as e:
            print(f"Error loading cached clusters: {e}")
    
    # Load activations
    activations = load_activations(dataset, config, seed)
    if not activations:
        raise ValueError(f"No activations found for {dataset}, config {config}, seed {seed}")
    
    layer_clusters = {}
    
    # Process each layer
    for layer_name, layer_data in activations.items():
        # Handle train/test split in activations if present
        if isinstance(layer_data, dict) and "test" in layer_data:
            # Use test set for clustering
            layer_data = layer_data["test"]
        
        # Skip if layer data is empty
        if layer_data.shape[0] == 0:
            continue
            
        # Ensure layer_data is 2D for KMeans
        if len(layer_data.shape) == 1:
            # Reshape 1D array to 2D
            layer_data = layer_data.reshape(-1, 1)
            print(f"Reshaped 1D layer data to shape {layer_data.shape}")
        elif len(layer_data.shape) > 2:
            # Reshape higher-dimensional array to 2D by flattening all dimensions after the first
            original_shape = layer_data.shape
            layer_data = layer_data.reshape(original_shape[0], -1)
            print(f"Reshaped {original_shape} array to 2D shape {layer_data.shape}")
        
        # Find optimal number of clusters using silhouette score
        best_score = -1
        best_k = 2  # Minimum is 2 clusters
        best_kmeans = None
        
        max_possible_k = min(max_k, int(np.sqrt(layer_data.shape[0])))
        
        for k in range(2, max_possible_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(layer_data)
            
            # If we have only one point in a cluster, silhouette score will fail
            # Check for this case and skip this k
            counts = np.bincount(cluster_labels)
            if np.any(counts < 2):
                continue
            
            try:
                score = silhouette_score(layer_data, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_kmeans = kmeans
            except Exception as e:
                print(f"Error computing silhouette score for k={k}: {e}")
                continue
        
        # If we found a valid clustering, save it
        if best_kmeans is not None:
            layer_clusters[layer_name] = {
                "k": best_k,
                "centers": best_kmeans.cluster_centers_,
                "labels": best_kmeans.labels_,
                "silhouette_score": best_score
            }
        else:
            # Fallback to k=2 if no valid clustering found
            # Ensure layer_data is still 2D for KMeans (in case it was modified after initial check)
            if len(layer_data.shape) == 1:
                layer_data = layer_data.reshape(-1, 1)
                print(f"Reshaped 1D layer data to shape {layer_data.shape} in fallback clustering")
            elif len(layer_data.shape) > 2:
                original_shape = layer_data.shape
                layer_data = layer_data.reshape(original_shape[0], -1)
                print(f"Reshaped {original_shape} array to 2D shape {layer_data.shape} in fallback clustering")
                
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(layer_data)
            
            layer_clusters[layer_name] = {
                "k": 2,
                "centers": kmeans.cluster_centers_,
                "labels": cluster_labels,
                "silhouette_score": 0.0  # Indicate this is a fallback with a poor score
            }
    
    # Cache the results
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(layer_clusters, f)
        print(f"Cached clusters to {cache_path}")
    except Exception as e:
        print(f"Error caching clusters: {e}")
    
    return layer_clusters

def _get_clusters_cache_path(dataset: str, config: Union[Dict[str, Any], str], seed: int, max_k: int) -> str:
    """Generate the cache path for the clusters."""
    # Handle config as string or dict
    if isinstance(config, str):
        config_id = config
    else:
        config_id = get_config_id(config)
    
    cache_dir = os.path.join(get_config_path(), "cache", "clusters")
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, f"{dataset}_{config_id}_seed{seed}_maxk{max_k}.pkl")

def get_baseline_config(dataset: str, seed: int = 0) -> Dict[str, Any]:
    """
    Get the baseline configuration for the given dataset.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic", "heart").
        seed: Random seed to use.
        
    Returns:
        Dictionary containing the baseline configuration parameters.
    """
    # Baseline has no regularization
    return {
        "weight": 0.0,
        "temperature": 0.0,
        "threshold": 0.0,
        "layers": "",
        "seed": seed
    }

def load_experiment_history(dataset: str, config: Union[Dict[str, Any], str], seed: int) -> Dict[str, Any]:
    """
    Load the training history for a specific experiment.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string
        seed: Random seed used
        
    Returns:
        Dictionary containing the training history
    """
    data_root = get_config_path()
    print(f"DEBUG: data_root = {data_root}")
    
    # Handle config as string or dict
    if isinstance(config, str):
        config_id = config
    else:
        config_id = get_config_id(config)
    print(f"DEBUG: config_id = {config_id}")
    
    # Determine if this is a baseline or regularized experiment
    if config_id == "baseline":
        # Look in baseline directory
        baseline_dir = os.path.join(data_root, "baselines", dataset.lower())
        print(f"DEBUG: Looking in baseline dir: {baseline_dir}")
        
        if os.path.exists(baseline_dir):
            # Find all directories for this seed
            all_dirs = os.listdir(baseline_dir)
            print(f"DEBUG: All dirs in {baseline_dir}: {all_dirs}")
            
            # Filter to match the pattern and specific seed
            seed_dirs = [d for d in all_dirs if f"{dataset.lower()}_baseline_seed{seed}_" in d]
            print(f"DEBUG: Seed dirs for seed {seed}: {seed_dirs}")
            
            if not seed_dirs:
                print(f"WARNING: No baseline experiment found for {dataset} with seed {seed}")
                # Try to use any available seed if specific seed not found
                any_seed_dirs = [d for d in all_dirs if f"{dataset.lower()}_baseline_seed" in d]
                if not any_seed_dirs:
                    raise FileNotFoundError(f"No baseline experiments found for {dataset}")
                seed_dirs = any_seed_dirs
                print(f"Using alternative seed directory instead: {seed_dirs[-1]}")
            
            # Sort by name (timestamp at the end will put newest last)
            seed_dirs.sort()
            # Use the most recent experiment
            exp_dir = os.path.join(baseline_dir, seed_dirs[-1])
            print(f"DEBUG: Using most recent experiment dir: {exp_dir}")
        else:
            raise FileNotFoundError(f"Baseline directory not found for {dataset}")
    else:
        # Look in cohesion directory
        exp_dir = os.path.join(data_root, "cohesion", dataset.lower(), config_id, f"seed_{seed}")
        print(f"DEBUG: Looking in cohesion dir: {exp_dir}")
        if not os.path.exists(exp_dir):
            raise FileNotFoundError(f"Cohesion experiment directory not found: {exp_dir}")
    
    history_path = os.path.join(exp_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history

def load_layer_class_metrics(dataset: str, config: Union[Dict[str, Any], str], seed: int) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Load per-layer, per-class fragmentation metrics from training history.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string
        seed: Random seed used
        
    Returns:
        Nested dictionary: {layer_name: {class_label: {"entropy": value, "angle": value}}}
    """
    history = load_experiment_history(dataset, config, seed)
    
    # Initialize the metrics dictionary
    metrics = {}
    
    # Get class labels for this dataset
    class_labels = load_dataset_metadata(dataset)["class_labels"]
    
    # Extract final metrics for each layer
    for layer in ["layer1", "layer2", "layer3"]:  # Assuming 3-layer network
        metrics[layer] = {}
        
        # Get the final epoch's metrics for this layer
        entropy_key = f"entropy_fragmentation_{layer}"
        angle_key = f"angle_fragmentation_{layer}"
        
        if entropy_key in history and angle_key in history:
            for class_idx, class_label in enumerate(class_labels):
                metrics[layer][str(class_label)] = {
                    "entropy": history[entropy_key][-1][class_idx],  # Assuming metrics are per-class lists
                    "angle": history[angle_key][-1][class_idx]
                }
    
    return metrics

def load_dataset_metadata(dataset: str) -> Dict[str, Any]:
    """
    Load metadata about the dataset, including actual class labels from the data.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic", "heart")
        
    Returns:
        Dictionary containing dataset metadata and actual class labels
    """
    # First, try to load actual class labels from the baseline experiment
    try:
        data_root = get_config_path()
        print(f"DEBUG: data_root for metadata = {data_root}")
        baseline_dir = os.path.join(data_root, "baselines", dataset.lower())
        print(f"DEBUG: Looking for metadata in baseline dir: {baseline_dir}")
        
        if os.path.exists(baseline_dir):
            # Find all experiment directories
            all_dirs = os.listdir(baseline_dir)
            print(f"DEBUG: All dirs in {baseline_dir}: {all_dirs}")
            
            # Get all seed directories, regardless of seed number
            seed_dirs = [d for d in all_dirs if f"{dataset.lower()}_baseline_seed" in d]
            print(f"DEBUG: Found seed dirs for metadata: {seed_dirs}")
            
            if seed_dirs:
                # Sort to get newest experiment last
                seed_dirs.sort()
                # Use the most recent experiment
                exp_dir = os.path.join(baseline_dir, seed_dirs[-1])
                print(f"DEBUG: Using most recent experiment for metadata: {exp_dir}")
                
                # Try to load class labels from metadata or activations
                metadata_path = os.path.join(exp_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        if "class_labels" in metadata:
                            actual_labels = metadata["class_labels"]
                            print(f"DEBUG: Loaded class_labels from metadata.json: {actual_labels}")
                        else:
                            # If no explicit class_labels, try to infer from the data
                            try:
                                history = load_experiment_history(dataset, "baseline", int(seed_dirs[-1].split("seed")[1].split("_")[0]))
                                if "class_names" in history:
                                    actual_labels = history["class_names"]
                                    print(f"DEBUG: Loaded class_names from history: {actual_labels}")
                                else:
                                    # Default to binary classification if we can't find explicit labels
                                    actual_labels = [0, 1]
                                    print(f"DEBUG: Using default binary class labels: {actual_labels}")
                            except Exception as e:
                                print(f"ERROR loading history for class labels: {e}")
                                actual_labels = [0, 1]
                else:
                    try:
                        # Extract seed from directory name
                        seed_num = int(exp_dir.split("seed")[1].split("_")[0])
                        print(f"DEBUG: Extracted seed {seed_num} from {exp_dir}")
                        
                        # Try to get from history
                        history = load_experiment_history(dataset, "baseline", seed_num)
                        if "class_names" in history:
                            actual_labels = history["class_names"]
                            print(f"DEBUG: Loaded class_names from history: {actual_labels}")
                        else:
                            # Default to binary classification if we can't find explicit labels
                            actual_labels = [0, 1]
                            print(f"DEBUG: Using default binary class labels: {actual_labels}")
                    except Exception as e:
                        print(f"ERROR loading history for class labels: {e}")
                        actual_labels = [0, 1]
    except Exception as e:
        print(f"Warning: Could not load actual class labels: {e}")
        actual_labels = None
    
    # Dataset-specific metadata (now including actual labels)
    metadata = {
        "titanic": {
            "class_names": ["Died", "Survived"],
            "key_features": ["Pclass", "Sex", "Age", "Fare"],
            "high_interest_feature": "Fare"
        },
        "heart": {
            "class_names": ["No Disease", "Disease"],
            "key_features": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol"],
            "high_interest_feature": "Cholesterol"
        }
    }
    
    dataset = dataset.lower()
    if dataset not in metadata:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(metadata.keys())}")
    
    # Add actual labels to the metadata if we found them
    if actual_labels is not None:
        metadata[dataset]["class_labels"] = actual_labels
    else:
        # Use the predefined class names as a fallback
        metadata[dataset]["class_labels"] = metadata[dataset]["class_names"]
    
    return metadata[dataset]

def compute_fractured_off_scores(
    dataset: str,
    config: Union[Dict[str, Any], str],
    seed: int,
    n_clusters: Optional[int] = None,
    layer_clusters: Optional[Dict[str, Dict[str, Any]]] = None,
    max_k: int = 10
) -> Dict[int, float]:
    """
    Compute "fractured off" scores for each sample based on their cluster assignments.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string
        seed: Random seed used
        n_clusters: Number of clusters to use for all layers (if None, will be determined per layer)
        layer_clusters: Pre-computed cluster information (if None, will compute internally)
        max_k: Maximum number of clusters to try if computing clusters
        
    Returns:
        Dictionary mapping sample indices to their "fractured off" scores
    """
    # Determine the number of samples based on the first layer's cluster labels
    if layer_clusters:
        first_layer = next(iter(layer_clusters.values()))
        num_samples = len(first_layer["labels"])
        
        # Load the metadata class labels
        metadata = load_dataset_metadata(dataset)
        class_labels = metadata["class_labels"]
        print(f"Using class labels from metadata: {class_labels}")
        
        # If we have a mismatch in the number of samples, create dummy labels
        if len(class_labels) != num_samples:
            print(f"WARNING: Length mismatch between class labels ({len(class_labels)}) and samples ({num_samples})")
            if len(class_labels) >= 2:
                # Use first two classes and repeat to match sample count
                class_labels = np.array([class_labels[i % 2] for i in range(num_samples)])
                print(f"Created synthetic labels with alternating classes, shape {class_labels.shape}")
    else:
        # Load activations to get the actual data
        activations = load_activations(dataset, config, seed)
        
        # Check if there are class labels in the activations
        if "labels" in activations and isinstance(activations["labels"], np.ndarray):
            class_labels = activations["labels"]
            print(f"Using class labels from activations with shape {class_labels.shape}")
        elif "labels" in activations and isinstance(activations["labels"], dict) and "test" in activations["labels"]:
            class_labels = activations["labels"]["test"]
            print(f"Using class labels from test set with shape {class_labels.shape}")
        else:
            # Fallback to metadata
            metadata = load_dataset_metadata(dataset)
            class_labels = metadata["class_labels"]
            print(f"Using fallback class labels from metadata: {class_labels}")
            
            # If we have a mismatch in the number of samples, create dummy labels
            # This is a workaround and should be addressed in the data loading pipeline
            first_layer = next(iter(activations.values()))
            if isinstance(first_layer, dict) and "test" in first_layer:
                num_samples = first_layer["test"].shape[0]
            else:
                num_samples = first_layer.shape[0]
                
            if len(class_labels) != num_samples:
                print(f"WARNING: Length mismatch between class labels ({len(class_labels)}) and samples ({num_samples})")
                if len(class_labels) >= 2:
                    # Use first two classes and repeat to match sample count
                    class_labels = np.array([class_labels[i % 2] for i in range(num_samples)])
                    print(f"Created synthetic labels with alternating classes, shape {class_labels.shape}")
    
    # Use pre-computed clusters if provided, otherwise compute them
    if layer_clusters is None:
        # If n_clusters is provided, we won't use the silhouette search
        if n_clusters is not None:
            # Load activations and manually cluster with fixed k
            activations = load_activations(dataset, config, seed)
            layer_clusters = {}
            
            for layer_name, layer_activations in activations.items():
                if isinstance(layer_activations, dict) and "test" in layer_activations:
                    layer_data = layer_activations["test"]
                else:
                    layer_data = layer_activations
                
                # Skip if layer data is empty
                if layer_data.shape[0] == 0:
                    continue
                
                # Ensure layer_data is 2D for KMeans
                if len(layer_data.shape) == 1:
                    # Reshape 1D array to 2D
                    layer_data = layer_data.reshape(-1, 1)
                    print(f"Reshaped 1D layer data to shape {layer_data.shape}")
                elif len(layer_data.shape) > 2:
                    # Reshape higher-dimensional array to 2D by flattening all dimensions after the first
                    original_shape = layer_data.shape
                    layer_data = layer_data.reshape(original_shape[0], -1)
                    print(f"Reshaped {original_shape} array to 2D shape {layer_data.shape}")
                
                # Perform k-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(layer_data)
                
                layer_clusters[layer_name] = {
                    "k": n_clusters,
                    "centers": kmeans.cluster_centers_,
                    "labels": cluster_labels
                }
        else:
            # Compute optimal clusters per layer
            layer_clusters = compute_layer_clusters(dataset, config, seed, max_k)
    
    scores = {}
    
    # Process each layer
    for layer_name, cluster_info in layer_clusters.items():
        cluster_labels = cluster_info["labels"]
        
        # For each class, identify its majority cluster(s)
        class_cluster_counts = {}
        for class_label in np.unique(class_labels):
            class_mask = (class_labels == class_label)
            class_clusters = cluster_labels[class_mask]
            counts = np.bincount(class_clusters)
            majority_threshold = np.max(counts) * 0.8  # Consider clusters with at least 80% of max count
            class_cluster_counts[class_label] = set(np.where(counts >= majority_threshold)[0])
        
        # For each sample, check if it's in a minority cluster for its class
        for i, (cluster, true_class) in enumerate(zip(cluster_labels, class_labels)):
            if i not in scores:
                scores[i] = 0.0
            
            # If sample is not in any of its class's majority clusters, increment its score
            if cluster not in class_cluster_counts[true_class]:
                scores[i] += 1.0 / len(layer_clusters)  # Normalize by number of layers
    
    return scores

# New function to be added
def compute_fractured_off_scores_embedded(
    dataset: str,
    config: Union[Dict[str, Any], str], # Config can be dict or config_id string
    seed: int, # Seed is used for consistency if we need to load other metadata
    layer_clusters: Dict[str, Dict[str, Any]], # Expects {layer_name: cluster_info} for ONE config
    actual_class_labels: Optional[np.ndarray] = None # Directly provide true class labels
) -> Dict[int, float]:
    """
    Compute "fractured off" scores based on embedded space cluster assignments.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string (used for context, e.g., loading metadata)
        seed: Random seed used (for context)
        layer_clusters: Embedded cluster information for a specific config
        actual_class_labels: Directly provide true class labels (optional)
        
    Returns:
        Dictionary mapping sample indices to fractured-off scores
    """
    print(f"DEBUG: compute_fractured_off_scores_embedded called")
    print(f"DEBUG: actual_class_labels type: {type(actual_class_labels)}")
    
    if isinstance(actual_class_labels, np.ndarray):
        print(f"DEBUG: actual_class_labels shape: {actual_class_labels.shape}")
    
    # First, validate the layer_clusters
    if not layer_clusters:
        print("DEBUG: layer_clusters is empty")
        return {}
    
    # Get a list of the layers we have cluster information for
    layers = list(layer_clusters.keys())
    print(f"DEBUG: Have clusters for these layers: {layers}")
    
    if not layers:
        print("DEBUG: No layers found in layer_clusters")
        return {}
    
    # Get class labels either from provided actual_class_labels or from metadata
    if actual_class_labels is not None:
        print("DEBUG: Using actual_class_labels provided")
        # Note: Already checked type above
        class_labels = actual_class_labels
    else:
        print("DEBUG: Getting class labels from metadata")
        try:
            metadata = load_dataset_metadata(dataset)
            class_labels = metadata["class_labels"]
            print(f"DEBUG: Loaded metadata class labels: {len(class_labels)} labels")
        except Exception as e:
            print(f"DEBUG: Error loading class labels from metadata: {e}")
            return {}
    
    # Check if we have a reasonable number of class labels
    if len(class_labels) < 2:
        print(f"DEBUG: Not enough class labels: {len(class_labels)}")
        return {}
    
    # Get the number of samples by checking the first layer's cluster labels
    first_layer = layers[0]
    if "labels" not in layer_clusters[first_layer]:
        print(f"DEBUG: No 'labels' in layer_clusters[{first_layer}]")
        return {}
    
    num_samples = len(layer_clusters[first_layer]["labels"])
    print(f"DEBUG: num_samples: {num_samples}")
    
    # Check if class labels array is compatible with the number of samples
    if len(class_labels) != num_samples:
        print(f"DEBUG: WARNING: Metadata class labels ({len(class_labels)}) don't match samples ({num_samples})")
        if len(class_labels) < num_samples:
            # Not enough class labels - can't compute fractured off score
            print("DEBUG: Not enough class labels, returning empty scores")
            return {}
        else:
            # More class labels than samples - truncate
            print(f"DEBUG: Truncating class labels from {len(class_labels)} to {num_samples}")
            class_labels = class_labels[:num_samples]
    
    # Calculate the "fractured off" scores for each sample
    # For each sample, we check how often it gets assigned to a cluster that's dominated by another class
    fractured_off_scores = {}
    
    # Process each layer
    for layer_name in layers:
        # Get cluster info for this layer
        cluster_info = layer_clusters[layer_name]
        
        # Skip if k < 2 (not meaningful for fracture analysis)
        if cluster_info.get("k", 0) < 2:
            print(f"DEBUG: Skipping layer {layer_name} because k < 2")
            continue
        
        cluster_labels = cluster_info["labels"]
        
        # Skip if labels don't match sample count
        if len(cluster_labels) != num_samples:
            print(f"DEBUG: Skipping layer {layer_name} because labels don't match sample count: {len(cluster_labels)} vs {num_samples}")
            continue
        
        # Calculate dominant class for each cluster
        clusters = np.unique(cluster_labels)
        cluster_to_dominant_class = {}
        for cluster_id in clusters:
            # Find samples in this cluster
            cluster_mask = cluster_labels == cluster_id
            
            # Get class labels for these samples
            class_labels_in_cluster = class_labels[cluster_mask]
            
            # Count occurrences of each class in this cluster
            unique_classes, class_counts = np.unique(class_labels_in_cluster, return_counts=True)
            
            # Dominant class is the one with the highest count
            if len(unique_classes) > 0:
                dominant_class = unique_classes[np.argmax(class_counts)]
                cluster_to_dominant_class[cluster_id] = dominant_class
            else:
                print(f"DEBUG: Empty cluster found for cluster_id {cluster_id}")
        
        # Update fractured off scores for each sample
        for sample_idx in range(num_samples):
            # Get cluster assignment for this sample
            cluster_id = cluster_labels[sample_idx]
            
            # Get dominant class for this cluster
            if cluster_id not in cluster_to_dominant_class:
                print(f"DEBUG: Cluster ID {cluster_id} not found in cluster_to_dominant_class. Keys: {list(cluster_to_dominant_class.keys())}")
                continue
                
            dominant_class = cluster_to_dominant_class[cluster_id]
            
            # Get true class for this sample
            true_class = class_labels[sample_idx]
            
            # If sample's true class doesn't match dominant class, it's "fractured off"
            if true_class != dominant_class:
                if sample_idx not in fractured_off_scores:
                    fractured_off_scores[sample_idx] = 0.0
                
                # Increment the score (we'll normalize by layer count later)
                fractured_off_scores[sample_idx] += 1.0
    
    # Normalize scores by number of layers processed
    num_layers_processed = len(layers)
    if num_layers_processed > 0:
        for sample_idx in fractured_off_scores:
            fractured_off_scores[sample_idx] /= num_layers_processed
    
    # Return scores for all samples, defaulting to 0.0 for samples that were never "fractured off"
    result = {i: fractured_off_scores.get(i, 0.0) for i in range(num_samples)}
    print(f"DEBUG: Returning {len(result)} scores")
    return result

# When run as a script, test the functionality
if __name__ == "__main__":
    try:
        print("Loading statistics...")
        stats = load_stats()
        print(f"Loaded statistics with {len(stats)} rows.")
        
        print("\nTesting get_best_config...")
        for ds in ["titanic", "heart"]:
            config = get_best_config(ds)
            print(f"Best config for {ds}: {config}")
            
        print("\nTesting select_samples...")
        for ds in ["titanic", "heart"]:
            sample_indices = select_samples(stats, ds)
            print(f"Selected {len(sample_indices)} samples for {ds}")
            
    except Exception as e:
        print(f"Error testing data_interface.py: {e}") 