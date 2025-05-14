"""
Visualize metric trajectories from baseline and cohesion experiments.

This script:
1. Loads training histories from experiments
2. Creates plots of metrics (accuracy, loss, entropy, angle) over training epochs
3. Compares baseline vs. cohesion regularization
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_fragmentation.config import (
    DATASETS, RESULTS_DIR, VISUALIZATION, COHESION_GRID
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

def load_training_history(experiment_dir: str) -> Dict[str, List[float]]:
    """
    Load training history from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary of training metrics
    """
    history_path = os.path.join(experiment_dir, "training_history.json")
    
    if not os.path.exists(history_path):
        logger.error(f"Training history not found: {history_path}")
        return {}
    
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        
        # Keep only metrics that are lists (trajectories)
        filtered_history = {
            k: v for k, v in history.items() 
            if isinstance(v, list) and len(v) > 0 and k not in ["epoch"]
        }
        
        return filtered_history
    
    except Exception as e:
        logger.error(f"Error loading history from {experiment_dir}: {str(e)}")
        return {}

def find_experiment_dirs(
    dataset_name: str,
    seed: int = 0,
    include_baselines: bool = True,
    include_cohesion: bool = True
) -> Dict[str, str]:
    """
    Find all experiment directories for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        seed: Seed to use
        include_baselines: Whether to include baseline experiments
        include_cohesion: Whether to include cohesion experiments
        
    Returns:
        Dictionary mapping experiment names to directories
    """
    experiment_dirs = {}
    
    # Find baseline experiments
    if include_baselines:
        baseline_dir = os.path.join(RESULTS_DIR, "baselines", dataset_name)
        
        if os.path.exists(baseline_dir):
            # Find all experiment directories with this seed
            for exp_dir in os.listdir(baseline_dir):
                if os.path.isdir(os.path.join(baseline_dir, exp_dir)) and f"seed{seed}" in exp_dir:
                    experiment_dirs["baseline"] = os.path.join(baseline_dir, exp_dir)
    
    # Find cohesion experiments
    if include_cohesion:
        cohesion_dir = os.path.join(RESULTS_DIR, "cohesion", dataset_name)
        
        if os.path.exists(cohesion_dir):
            # Find all parameter directories
            for param_dir in os.listdir(cohesion_dir):
                if os.path.isdir(os.path.join(cohesion_dir, param_dir)):
                    # Find seed directory
                    seed_dir = os.path.join(cohesion_dir, param_dir, f"seed_{seed}")
                    
                    if os.path.exists(seed_dir):
                        experiment_dirs[param_dir] = seed_dir
    
    return experiment_dirs

def plot_metric_trajectories(
    dataset_name: str,
    seed: int = 0,
    metrics: List[str] = None,
    output_dir: str = None,
    figsize: Tuple[int, int] = None,
    dpi: int = None
) -> List[str]:
    """
    Plot metric trajectories for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        seed: Seed to use
        metrics: List of metrics to plot (default: accuracy, loss, entropy, angle)
        output_dir: Directory to save output
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        List of paths to saved figures
    """
    if metrics is None:
        metrics = [
            "test_accuracy",
            "train_loss",
            "entropy_fragmentation",
            "angle_fragmentation"
        ]
    
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if figsize is None:
        figsize = VISUALIZATION["plot"]["figsize"]
    
    if dpi is None:
        dpi = VISUALIZATION["plot"]["dpi"]
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(dataset_name, seed)
    
    if not experiment_dirs:
        logger.error(f"No experiments found for {dataset_name} with seed {seed}")
        return []
    
    # Load training histories
    histories = {}
    for experiment_name, exp_dir in experiment_dirs.items():
        history = load_training_history(exp_dir)
        if history:
            histories[experiment_name] = history
    
    if not histories:
        logger.error(f"No training histories found for {dataset_name}")
        return []
    
    # Create plots for each metric
    output_paths = []
    
    for metric in metrics:
        # Check if metric exists in all histories
        if not all(metric in history for history in histories.values()):
            logger.warning(f"Metric {metric} not found in all histories, skipping")
            continue
        
        # Create figure
        plt.figure(figsize=figsize, dpi=dpi)
        
        # Plot metric for each experiment
        for experiment_name, history in histories.items():
            if metric in history:
                # Format experiment name
                if experiment_name == "baseline":
                    label = "Baseline"
                    linestyle = "-"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    # Extract parameters from experiment name
                    weight = 0.0
                    if experiment_name.startswith("w"):
                        parts = experiment_name.split("_")
                        for part in parts:
                            if part.startswith("w"):
                                try:
                                    weight = float(part[1:])
                                except:
                                    pass
                    
                    label = f"Cohesion (w={weight})"
                    linestyle = "--"
                    linewidth = 2.0
                    alpha = 0.8
                
                # Plot trajectory
                plt.plot(
                    history[metric],
                    label=label,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha
                )
        
        # Format metric name for title and labels
        if metric == "test_accuracy":
            metric_name = "Test Accuracy"
            ylabel = "Accuracy"
        elif metric == "train_accuracy":
            metric_name = "Train Accuracy"
            ylabel = "Accuracy"
        elif metric == "test_loss":
            metric_name = "Test Loss"
            ylabel = "Loss"
        elif metric == "train_loss":
            metric_name = "Train Loss"
            ylabel = "Loss"
        elif metric == "entropy_fragmentation":
            metric_name = "Entropy Fragmentation"
            ylabel = "Entropy"
        elif metric == "angle_fragmentation":
            metric_name = "Angle Fragmentation"
            ylabel = "Angle (degrees)"
        else:
            metric_name = metric.replace("_", " ").title()
            ylabel = metric_name
        
        # Add title and labels
        plt.title(f"{dataset_name.capitalize()} - {metric_name} vs. Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_file = f"{dataset_name}_{metric}_trajectory.png"
        output_path = os.path.join(output_dir, output_file)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved {metric} trajectory to {output_path}")
        output_paths.append(output_path)
    
    return output_paths

def plot_multi_metric_grid(
    dataset_name: str,
    seed: int = 0,
    metrics: List[str] = None,
    output_dir: str = None,
    figsize: Tuple[int, int] = None,
    dpi: int = None
) -> str:
    """
    Create a grid of metric trajectories.
    
    Args:
        dataset_name: Name of the dataset
        seed: Seed to use
        metrics: List of metrics to plot
        output_dir: Directory to save output
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        Path to saved figure
    """
    if metrics is None:
        metrics = [
            "test_accuracy",
            "train_loss",
            "entropy_fragmentation",
            "angle_fragmentation"
        ]
    
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if figsize is None:
        figsize = (12, 10)  # Larger figure for grid
    
    if dpi is None:
        dpi = VISUALIZATION["plot"]["dpi"]
    
    # Find experiment directories
    experiment_dirs = find_experiment_dirs(dataset_name, seed)
    
    if not experiment_dirs:
        logger.error(f"No experiments found for {dataset_name} with seed {seed}")
        return None
    
    # Load training histories
    histories = {}
    for experiment_name, exp_dir in experiment_dirs.items():
        history = load_training_history(exp_dir)
        if history:
            histories[experiment_name] = history
    
    if not histories:
        logger.error(f"No training histories found for {dataset_name}")
        return None
    
    # Create grid of plots
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi)
    
    # Flatten axes array if grid
    if n_metrics > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if i >= len(axes_flat):
            break
        
        ax = axes_flat[i]
        
        # Check if metric exists in all histories
        if not all(metric in history for history in histories.values()):
            logger.warning(f"Metric {metric} not found in all histories, skipping")
            ax.text(0.5, 0.5, f"Metric {metric} not available", 
                    ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot metric for each experiment
        for experiment_name, history in histories.items():
            if metric in history:
                # Format experiment name
                if experiment_name == "baseline":
                    label = "Baseline"
                    linestyle = "-"
                    linewidth = 2.5
                    alpha = 1.0
                else:
                    # Extract parameters from experiment name
                    weight = 0.0
                    if experiment_name.startswith("w"):
                        parts = experiment_name.split("_")
                        for part in parts:
                            if part.startswith("w"):
                                try:
                                    weight = float(part[1:])
                                except:
                                    pass
                    
                    label = f"Cohesion (w={weight})"
                    linestyle = "--"
                    linewidth = 2.0
                    alpha = 0.8
                
                # Plot trajectory
                ax.plot(
                    history[metric],
                    label=label,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha
                )
        
        # Format metric name for title and labels
        if metric == "test_accuracy":
            metric_name = "Test Accuracy"
            ylabel = "Accuracy"
        elif metric == "train_accuracy":
            metric_name = "Train Accuracy"
            ylabel = "Accuracy"
        elif metric == "test_loss":
            metric_name = "Test Loss"
            ylabel = "Loss"
        elif metric == "train_loss":
            metric_name = "Train Loss"
            ylabel = "Loss"
        elif metric == "entropy_fragmentation":
            metric_name = "Entropy Fragmentation"
            ylabel = "Entropy"
        elif metric == "angle_fragmentation":
            metric_name = "Angle Fragmentation"
            ylabel = "Angle (degrees)"
        else:
            metric_name = metric.replace("_", " ").title()
            ylabel = metric_name
        
        # Add title and labels
        ax.set_title(metric_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend()
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    # Add overall title
    plt.suptitle(f"{dataset_name.capitalize()} - Training Trajectories", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure
    output_file = f"{dataset_name}_trajectory_grid.png"
    output_path = os.path.join(output_dir, output_file)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved trajectory grid to {output_path}")
    
    return output_path

def main():
    """Main function to create metric trajectory visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize metric trajectories from baseline and cohesion experiments."
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
        "--seed",
        type=int,
        default=0,
        help="Seed to use for experiments"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["test_accuracy", "train_loss", "entropy_fragmentation", "angle_fragmentation"],
        help="Metrics to plot"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "visualization"),
        help="Directory to save output"
    )
    
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Create separate plots for each metric instead of a grid"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizations for each dataset
    for dataset_name in args.datasets:
        logger.info(f"Creating trajectory visualizations for dataset: {dataset_name}")
        
        if args.separate:
            # Create separate plots for each metric
            paths = plot_metric_trajectories(
                dataset_name=dataset_name,
                seed=args.seed,
                metrics=args.metrics,
                output_dir=args.output_dir
            )
            
            if paths:
                logger.info(f"Created {len(paths)} trajectory plots for {dataset_name}")
            else:
                logger.error(f"Failed to create trajectory plots for {dataset_name}")
        else:
            # Create grid of plots
            path = plot_multi_metric_grid(
                dataset_name=dataset_name,
                seed=args.seed,
                metrics=args.metrics,
                output_dir=args.output_dir
            )
            
            if path:
                logger.info(f"Created trajectory grid for {dataset_name}: {path}")
            else:
                logger.error(f"Failed to create trajectory grid for {dataset_name}")
    
    logger.info("Trajectory visualization complete")

if __name__ == "__main__":
    main() 