"""
Visualize concept spaces from baseline and cohesion experiments.

This script:
1. Loads activations from baseline and cohesion experiments
2. Uses dimensionality reduction (UMAP or PCA) to visualize concept spaces
3. Creates side-by-side comparisons of baseline vs. regularized models
4. Saves visualization figures
"""

import os
import sys
import json
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_fragmentation.config import (
    DATASETS, RESULTS_DIR, VISUALIZATION, METRICS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import UMAP, fallback to PCA if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    logger.warning("UMAP not available, will use PCA instead")
    UMAP_AVAILABLE = False

def load_activations(experiment_dir: str, layer_name: str = "layer3", epoch: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load activations from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        layer_name: Name of the layer to visualize
        epoch: Epoch to visualize (-1 for last epoch)
        
    Returns:
        Tuple of (activations, labels)
    """
    activations_path = os.path.join(experiment_dir, "layer_activations.pkl")
    
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"Activations file not found: {activations_path}")
    
    try:
        with open(activations_path, "rb") as f:
            activations_data = pickle.load(f)
        
        # Get test activations for the specified layer and epoch
        layer_activations = activations_data[layer_name]["test"]
        labels = activations_data["labels"]["test"]
        
        if not layer_activations:
            raise ValueError(f"No activations found for layer {layer_name}")
        
        # Adjust epoch index (using last epoch if epoch is -1)
        if epoch == -1:
            epoch = len(layer_activations) - 1
        
        if epoch >= len(layer_activations):
            raise ValueError(f"Epoch {epoch} not available (max: {len(layer_activations) - 1})")
        
        # Get activations and labels for the specified epoch
        activations = layer_activations[epoch]
        epoch_labels = labels[epoch]
        
        return activations.numpy(), epoch_labels.numpy()
    
    except Exception as e:
        logger.error(f"Error loading activations from {experiment_dir}: {str(e)}")
        raise

def get_best_epoch_from_history(experiment_dir: str) -> int:
    """
    Get the best epoch from training history.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Index of the best epoch
    """
    history_path = os.path.join(experiment_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        logger.warning(f"Training history not found: {history_path}")
        return -1
    
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        
        # Find epoch with best test accuracy
        if "test_accuracy" in history and len(history["test_accuracy"]) > 0:
            best_epoch = np.argmax(history["test_accuracy"])
            return best_epoch
        else:
            logger.warning(f"No test accuracy found in history: {history_path}")
            return -1
    
    except Exception as e:
        logger.error(f"Error reading history from {experiment_dir}: {str(e)}")
        return -1

def reduce_dimensions(
    activations: np.ndarray, 
    method: str = "umap",
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce dimensionality of activations data.
    
    Args:
        activations: Activation data to reduce
        method: Dimensionality reduction method ('umap' or 'pca')
        n_components: Number of components in the reduced space
        random_state: Random seed for reproducibility
        
    Returns:
        Reduced dimensional data
    """
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(activations)
    
    # Apply dimensionality reduction
    if method.lower() == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_neighbors=VISUALIZATION["umap"]["n_neighbors"],
            min_dist=VISUALIZATION["umap"]["min_dist"],
            metric=VISUALIZATION["umap"]["metric"],
            n_components=n_components,
            random_state=random_state
        )
    else:
        # Fallback to PCA
        reducer = PCA(
            n_components=n_components,
            svd_solver=VISUALIZATION["pca"]["svd_solver"],
            random_state=random_state
        )
    
    # Fit and transform
    reduced_data = reducer.fit_transform(scaled_data)
    
    return reduced_data

def plot_concept_space(
    activations: np.ndarray,
    labels: np.ndarray,
    title: str,
    dataset_name: str,
    ax=None,
    convex_hull: bool = True,
    method: str = "umap",
    random_state: int = 42
) -> plt.Axes:
    """
    Plot a 2D concept space visualization.
    
    Args:
        activations: Activation data to visualize
        labels: Class labels
        title: Plot title
        dataset_name: Name of the dataset (for class names)
        ax: Matplotlib axes (optional)
        convex_hull: Whether to draw convex hulls around class clusters
        method: Dimensionality reduction method
        random_state: Random seed for reproducibility
        
    Returns:
        Matplotlib axes object
    """
    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(
            figsize=VISUALIZATION["plot"]["figsize"],
            dpi=VISUALIZATION["plot"]["dpi"]
        )
    
    # Reduce dimensionality to 2D
    reduced_data = reduce_dimensions(
        activations, 
        method=method,
        n_components=2,
        random_state=random_state
    )
    
    # Get unique labels
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Use a good colormap for the number of classes
    if num_classes <= 10:
        cmap = plt.cm.get_cmap(VISUALIZATION["plot"]["cmap"], num_classes)
    else:
        cmap = plt.cm.get_cmap("viridis", num_classes)
    
    # Get class names if available
    class_names = None
    if dataset_name == "fashion_mnist" and "classes" in DATASETS[dataset_name]:
        class_names = DATASETS[dataset_name]["classes"]
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = cmap(i)
        
        ax.scatter(
            reduced_data[mask, 0],
            reduced_data[mask, 1],
            c=[color],
            s=VISUALIZATION["plot"]["markersize"],
            alpha=VISUALIZATION["plot"]["alpha"],
            label=class_names[label] if class_names is not None else f"Class {label}"
        )
        
        # Draw convex hull if requested
        if convex_hull and np.sum(mask) > 2:
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(reduced_data[mask, :2])
                for simplex in hull.simplices:
                    ax.plot(
                        reduced_data[mask, 0][simplex],
                        reduced_data[mask, 1][simplex],
                        c=color,
                        alpha=0.5
                    )
            except Exception as e:
                logger.warning(f"Could not draw convex hull: {str(e)}")
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    
    # Add a legend outside the plot if there are many classes
    if num_classes > 5:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.legend()
    
    return ax

def find_best_cohesion_experiment(
    dataset_name: str,
    metric: str = "final_entropy",
    cohesion_dir: str = None,
    seed: int = 0
) -> str:
    """
    Find the best cohesion experiment based on a metric.
    
    Args:
        dataset_name: Name of the dataset
        metric: Metric to optimize ('final_entropy', 'final_angle', 'best_test_accuracy')
        cohesion_dir: Path to cohesion experiments directory
        seed: Seed to use
        
    Returns:
        Path to the best experiment directory
    """
    if cohesion_dir is None:
        cohesion_dir = os.path.join(RESULTS_DIR, "cohesion")
    
    dataset_dir = os.path.join(cohesion_dir, dataset_name)
    
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return None
    
    # Find all parameter directories that are not baseline
    param_dirs = [
        d for d in os.listdir(dataset_dir) 
        if os.path.isdir(os.path.join(dataset_dir, d)) and d != "baseline"
    ]
    
    if not param_dirs:
        logger.error(f"No cohesion experiments found for dataset: {dataset_name}")
        return None
    
    # Look for aggregated results first
    analysis_dir = os.path.join(RESULTS_DIR, "analysis")
    summary_path = os.path.join(analysis_dir, "cohesion_summary.csv")
    
    if os.path.exists(summary_path):
        # Use aggregated results to find the best experiment
        df = pd.read_csv(summary_path)
        
        # Filter by dataset and seed
        mask = (df["dataset"] == dataset_name) & (df["seed"] == seed) & (df["experiment_type"] == "cohesion")
        
        if not np.any(mask):
            logger.warning(f"No cohesion experiments found for {dataset_name} seed {seed} in summary")
            return None
        
        # Determine whether lower or higher is better
        if metric in ["final_entropy", "final_angle"]:
            # Lower is better
            best_row = df.loc[mask].sort_values(metric).iloc[0]
        else:
            # Higher is better
            best_row = df.loc[mask].sort_values(metric, ascending=False).iloc[0]
        
        if "experiment_dir" in best_row:
            experiment_dir = best_row["experiment_dir"]
            if os.path.exists(experiment_dir):
                logger.info(f"Found best experiment for {dataset_name}: {experiment_dir}")
                return experiment_dir
    
    # Fallback to manual search
    logger.warning("Using manual search to find best experiment")
    
    best_value = float("inf") if metric in ["final_entropy", "final_angle"] else float("-inf")
    best_dir = None
    
    for param_dir in param_dirs:
        seed_dir = os.path.join(dataset_dir, param_dir, f"seed_{seed}")
        
        if not os.path.exists(seed_dir):
            continue
        
        history_path = os.path.join(seed_dir, "training_history.json")
        
        if not os.path.exists(history_path):
            continue
        
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
            
            if metric in history and len(history[metric]) > 0:
                value = history[metric][-1]  # Use final value
                
                # Check if this is better
                if metric in ["final_entropy", "final_angle"]:
                    if value < best_value:
                        best_value = value
                        best_dir = seed_dir
                else:
                    if value > best_value:
                        best_value = value
                        best_dir = seed_dir
        
        except Exception as e:
            logger.warning(f"Error reading history from {history_path}: {str(e)}")
    
    if best_dir:
        logger.info(f"Found best experiment for {dataset_name}: {best_dir}")
        return best_dir
    else:
        logger.error(f"No valid experiments found for {dataset_name}")
        return None

def find_baseline_experiment(dataset_name: str, seed: int = 0) -> str:
    """
    Find the baseline experiment for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        seed: Seed to use
        
    Returns:
        Path to the baseline experiment directory
    """
    baseline_dir = os.path.join(RESULTS_DIR, "baselines", dataset_name)
    
    if not os.path.exists(baseline_dir):
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return None
    
    # Find all experiment directories with this seed
    seed_dirs = [
        d for d in os.listdir(baseline_dir) 
        if os.path.isdir(os.path.join(baseline_dir, d)) and f"seed{seed}" in d
    ]
    
    if not seed_dirs:
        logger.error(f"No baseline experiment found for {dataset_name} with seed {seed}")
        return None
    
    # Use the first matching directory
    return os.path.join(baseline_dir, seed_dirs[0])

def plot_dataset_comparison(
    dataset_name: str,
    layer_name: str = "layer3",
    baseline_seed: int = 0,
    cohesion_seed: int = 0,
    metric: str = "final_entropy",
    dimred_method: str = "umap",
    output_dir: str = None,
    use_best_epoch: bool = True,
    convex_hull: bool = True,
    figsize: Tuple[int, int] = None,
    dpi: int = None
) -> str:
    """
    Create and save a side-by-side comparison of baseline vs. cohesion concept spaces.
    
    Args:
        dataset_name: Name of the dataset
        layer_name: Layer to visualize
        baseline_seed: Seed for baseline experiment
        cohesion_seed: Seed for cohesion experiment
        metric: Metric to use for finding best cohesion experiment
        dimred_method: Dimensionality reduction method
        output_dir: Directory to save output (default: RESULTS_DIR/visualization)
        use_best_epoch: Whether to use the best epoch or final epoch
        convex_hull: Whether to draw convex hulls around class clusters
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        Path to the saved figure
    """
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find baseline experiment
    baseline_dir = find_baseline_experiment(dataset_name, baseline_seed)
    if baseline_dir is None:
        logger.error(f"Could not find baseline experiment for {dataset_name}")
        return None
    
    # Find best cohesion experiment
    cohesion_dir = find_best_cohesion_experiment(dataset_name, metric, seed=cohesion_seed)
    if cohesion_dir is None:
        logger.error(f"Could not find cohesion experiment for {dataset_name}")
        return None
    
    # Get best epoch if requested
    baseline_epoch = get_best_epoch_from_history(baseline_dir) if use_best_epoch else -1
    cohesion_epoch = get_best_epoch_from_history(cohesion_dir) if use_best_epoch else -1
    
    # Load baseline activations
    try:
        baseline_activations, baseline_labels = load_activations(
            baseline_dir, 
            layer_name=layer_name, 
            epoch=baseline_epoch
        )
    except Exception as e:
        logger.error(f"Error loading baseline activations: {str(e)}")
        return None
    
    # Load cohesion activations
    try:
        cohesion_activations, cohesion_labels = load_activations(
            cohesion_dir, 
            layer_name=layer_name, 
            epoch=cohesion_epoch
        )
    except Exception as e:
        logger.error(f"Error loading cohesion activations: {str(e)}")
        return None
    
    # Get regularization parameters from directory name
    reg_params = os.path.basename(os.path.dirname(cohesion_dir))
    
    # Create figure for side-by-side comparison
    if figsize is None:
        figsize = (VISUALIZATION["plot"]["figsize"][0] * 2, VISUALIZATION["plot"]["figsize"][1])
    
    if dpi is None:
        dpi = VISUALIZATION["plot"]["dpi"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Plot baseline concept space
    baseline_title = f"{dataset_name.capitalize()} - Baseline"
    if use_best_epoch:
        baseline_title += f" (Epoch {baseline_epoch})"
    
    plot_concept_space(
        baseline_activations,
        baseline_labels,
        baseline_title,
        dataset_name,
        ax=ax1,
        convex_hull=convex_hull,
        method=dimred_method
    )
    
    # Plot cohesion concept space
    cohesion_title = f"{dataset_name.capitalize()} - Cohesion ({reg_params})"
    if use_best_epoch:
        cohesion_title += f" (Epoch {cohesion_epoch})"
    
    plot_concept_space(
        cohesion_activations,
        cohesion_labels,
        cohesion_title,
        dataset_name,
        ax=ax2,
        convex_hull=convex_hull,
        method=dimred_method
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_file = f"concept_space_{dataset_name}.png"
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, bbox_inches="tight")
    
    logger.info(f"Saved concept space visualization to {output_path}")
    
    return output_path

def main():
    """Main function to create concept space visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize concept spaces from baseline and cohesion experiments."
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Datasets to visualize"
    )
    
    parser.add_argument(
        "--layer",
        type=str,
        default="layer3",
        help="Layer to visualize"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to use for experiments"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="final_entropy",
        choices=["final_entropy", "final_angle", "best_test_accuracy"],
        help="Metric to use for finding best cohesion experiment"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "pca"],
        help="Dimensionality reduction method"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "visualization"),
        help="Directory to save output"
    )
    
    parser.add_argument(
        "--final_epoch",
        action="store_true",
        help="Use final epoch instead of best epoch"
    )
    
    parser.add_argument(
        "--no_hull",
        action="store_true",
        help="Disable drawing convex hulls"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizations for each dataset
    for dataset_name in args.datasets:
        logger.info(f"Creating visualization for dataset: {dataset_name}")
        
        output_path = plot_dataset_comparison(
            dataset_name=dataset_name,
            layer_name=args.layer,
            baseline_seed=args.seed,
            cohesion_seed=args.seed,
            metric=args.metric,
            dimred_method=args.method,
            output_dir=args.output_dir,
            use_best_epoch=not args.final_epoch,
            convex_hull=not args.no_hull
        )
        
        if output_path:
            logger.info(f"Visualization for {dataset_name} saved to {output_path}")
        else:
            logger.error(f"Failed to create visualization for {dataset_name}")
    
    logger.info("Visualization complete")

if __name__ == "__main__":
    main() 