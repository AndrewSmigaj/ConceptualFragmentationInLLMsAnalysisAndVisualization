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
    
    # Handle config as string or dict
    if isinstance(config, str):
        config_id = config
    else:
        config_id = get_config_id(config)
    
    # Construct the path to the activations file
    path = os.path.join(data_root, "cohesion", dataset.lower(), config_id, 
                        f"seed_{seed}", "layer_activations.pkl")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Activations file not found: {path}")
    
    # Load the pickle file
    with open(path, "rb") as f:
        activations = pickle.load(f)
    
    # Convert Python lists to NumPy arrays where possible
    activations = clean_activations(activations)
    
    return activations

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

def select_samples(df_stats: pd.DataFrame, dataset: str, k_frag: int = 20, k_norm: int = 20) -> List[int]:
    """
    Select sample indices with high and low fragmentation.
    
    Args:
        df_stats: DataFrame containing the statistics.
        dataset: Name of the dataset (e.g., "titanic", "heart").
        k_frag: Number of high-fragmentation samples to select.
        k_norm: Number of normal (low-fragmentation) samples to select.
        
    Returns:
        List of sample indices.
    """
    # Filter for the dataset
    df = df_stats[df_stats["dataset"] == dataset.lower()]
    
    if df.empty:
        raise ValueError(f"No data found for dataset: {dataset}")
    
    # Use entropy in layer 3 as the fragmentation metric
    # Higher entropy = higher fragmentation
    entropy_col = "final_entropy_layer3"
    
    # Sort by entropy
    df_sorted = df.sort_values(by=entropy_col, ascending=False)
    
    # Select top k_frag samples with highest entropy
    high_frag_indices = df_sorted.iloc[:k_frag].index.tolist()
    
    # Select k_norm samples with lowest entropy
    low_frag_indices = df_sorted.iloc[-k_norm:].index.tolist()
    
    # Combine and return
    return high_frag_indices + low_frag_indices

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
    n_clusters: Optional[int] = None
) -> Dict[int, float]:
    """
    Compute "fractured off" scores for each sample based on their cluster assignments.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string
        seed: Random seed used
        n_clusters: Number of clusters to use (if None, will be determined by silhouette score)
        
    Returns:
        Dictionary mapping sample indices to their "fractured off" scores
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Load activations and class labels
    activations = load_activations(dataset, config, seed)
    metadata = load_dataset_metadata(dataset)
    class_labels = metadata["class_labels"]
    
    scores = {}
    
    # Process each layer
    for layer_name, layer_activations in activations.items():
        if isinstance(layer_activations, dict) and "test" in layer_activations:
            # If we have train/test split in activations
            layer_data = layer_activations["test"]
        else:
            layer_data = layer_activations
        
        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            best_score = -1
            best_k = 2
            for k in range(2, min(13, int(np.sqrt(len(layer_data))) + 1)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(layer_data)
                score = silhouette_score(layer_data, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_k = k
            n_clusters = best_k
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(layer_data)
        
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
                scores[i] += 1.0 / len(activations)  # Normalize by number of layers
    
    return scores

def select_samples(
    dataset: str,
    config: Union[Dict[str, Any], str],
    seed: int,
    k_frag: int = 20,
    k_norm: int = 20
) -> List[int]:
    """
    Select sample indices with high and low fragmentation based on "fractured off" scores.
    
    Args:
        dataset: Name of the dataset
        config: Configuration dictionary or config_id string
        seed: Random seed used
        k_frag: Number of high-fragmentation samples to select
        k_norm: Number of normal (low-fragmentation) samples to select
        
    Returns:
        List of sample indices
    """
    # Compute fractured off scores
    scores = compute_fractured_off_scores(dataset, config, seed)
    
    # Convert to Series for easy sorting
    scores_series = pd.Series(scores)
    
    # Select samples
    high_frag_indices = scores_series.nlargest(k_frag).index.tolist()
    low_frag_indices = scores_series.nsmallest(k_norm).index.tolist()
    
    return high_frag_indices + low_frag_indices

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