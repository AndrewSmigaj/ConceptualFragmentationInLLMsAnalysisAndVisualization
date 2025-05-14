"""
Visualization script for the Concept Fragmentation project.

This script:
1. Generates paper-ready figures using visualization utilities
2. Creates comparative visualizations of baseline vs. regularized models
3. Produces trajectory visualizations across training epochs
"""

import os
import torch
import numpy as np
import pandas as pd
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import glob
from datetime import datetime

from concept_fragmentation.config import (
    RESULTS_DIR, VISUALIZATION, RANDOM_SEED
)
from concept_fragmentation.visualization import (
    plot_activations_2d,
    plot_activations_3d,
    plot_layer_comparison,
    plot_topk_neuron_activations,
    reduce_dimensions,
    plot_sample_trajectory,
    plot_activation_flow,
    plot_class_trajectories,
    compute_compressed_paths
)


# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "visualization.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_experiment_data(experiment_name: str) -> Dict[str, Any]:
    """
    Load data from a trained experiment.
    
    Args:
        experiment_name: Name of the experiment to load
        
    Returns:
        Dictionary containing experiment data
    """
    experiment_dir = os.path.join(RESULTS_DIR, experiment_name)
    
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory {experiment_dir} not found")
    
    # Load training history
    history_path = os.path.join(experiment_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            training_history = json.load(f)
    else:
        training_history = None
        logger.warning(f"Training history not found for experiment {experiment_name}")
    
    # Load activations if available
    activations_path = os.path.join(experiment_dir, "layer_activations.pkl")
    if os.path.exists(activations_path):
        with open(activations_path, "rb") as f:
            activations = pickle.load(f)
    else:
        activations = None
        logger.warning(f"Layer activations not found for experiment {experiment_name}")
    
    # Load detailed metrics if available
    metrics_path = os.path.join(experiment_dir, "detailed_metrics_best.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = None
        logger.warning(f"Detailed metrics not found for experiment {experiment_name}")
    
    # Extract dataset name and regularization info from experiment name
    parts = experiment_name.split("_")
    dataset_name = parts[0]
    is_regularized = "regularized" in experiment_name
    
    return {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "is_regularized": is_regularized,
        "training_history": training_history,
        "activations": activations,
        "metrics": metrics,
        "experiment_dir": experiment_dir
    }


def visualize_activations(
    experiment_name: str,
    method: str = "pca",
    layer_name: str = "layer3",
    output_dir: Optional[str] = None,
    show_clusters: bool = True,
    epoch: int = -1  # -1 means the last epoch
) -> None:
    """
    Generate activation visualizations for a specific layer and epoch.
    
    Args:
        experiment_name: Name of the experiment to visualize
        method: Visualization method ("pca" or "umap")
        layer_name: Name of the layer to visualize
        output_dir: Directory to save visualizations
        show_clusters: Whether to show cluster assignments
        epoch: Epoch to visualize (-1 for last epoch)
    """
    # Load experiment data
    experiment_data = load_experiment_data(experiment_name)
    activations = experiment_data["activations"]
    
    if activations is None:
        logger.error(f"No activations found for experiment {experiment_name}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(experiment_data["experiment_dir"], "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get epoch activations
    epoch_indices = activations["epoch"]
    if epoch == -1:
        epoch = epoch_indices[-1]  # Last epoch
    
    epoch_idx = epoch_indices.index(epoch)
    
    # Get activations and labels for test data
    test_activations = activations[layer_name]["test"][epoch_idx]
    test_labels = activations["labels"]["test"][epoch_idx]
    
    # Generate the visualizations
    logger.info(f"Generating {method} visualization for {experiment_name}, layer {layer_name}, epoch {epoch}")
    
    # Define custom-named output files
    base_filename = f"{experiment_name}_{method}_{layer_name}_epoch{epoch}"
    
    if method == "pca":
        # PCA visualization
        fig, _ = plot_activations_pca(
            activations=test_activations,
            labels=test_labels,
            title=f"{experiment_name} - {layer_name} (Epoch {epoch})",
            n_components=VISUALIZATION["pca"]["n_components"],
            random_state=RANDOM_SEED
        )
        fig.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=VISUALIZATION["plot"]["dpi"])
        plt.close(fig)
        
        # PCA with clusters if requested
        if show_clusters:
            fig, _ = plot_activation_clusters(
                activations=test_activations,
                labels=test_labels,
                title=f"{experiment_name} - {layer_name} (Epoch {epoch}) - Clusters",
                n_components=VISUALIZATION["pca"]["n_components"],
                random_state=RANDOM_SEED
            )
            fig.savefig(os.path.join(output_dir, f"{base_filename}_clusters.png"), dpi=VISUALIZATION["plot"]["dpi"])
            plt.close(fig)
    
    elif method == "umap":
        # UMAP visualization
        fig, _ = plot_activations_umap(
            activations=test_activations,
            labels=test_labels,
            title=f"{experiment_name} - {layer_name} (Epoch {epoch})",
            n_neighbors=VISUALIZATION["umap"]["n_neighbors"],
            min_dist=VISUALIZATION["umap"]["min_dist"],
            n_components=VISUALIZATION["umap"]["n_components"],
            metric=VISUALIZATION["umap"]["metric"],
            random_state=RANDOM_SEED
        )
        fig.savefig(os.path.join(output_dir, f"{base_filename}.png"), dpi=VISUALIZATION["plot"]["dpi"])
        plt.close(fig)
    
    logger.info(f"Saved visualization to {output_dir}")


def visualize_trajectory(
    experiment_name: str,
    layer_name: str = "layer3",
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    stride: int = 1  # Visualize every `stride` epochs
) -> None:
    """
    Generate trajectory visualizations across training epochs.
    
    Args:
        experiment_name: Name of the experiment to visualize
        layer_name: Name of the layer to visualize
        output_dir: Directory to save visualizations
        num_epochs: Number of epochs to visualize (None for all)
        stride: Stride for epoch selection
    """
    # Load experiment data
    experiment_data = load_experiment_data(experiment_name)
    activations = experiment_data["activations"]
    
    if activations is None:
        logger.error(f"No activations found for experiment {experiment_name}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(experiment_data["experiment_dir"], "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get epoch indices
    epoch_indices = activations["epoch"]
    
    # Limit to requested number of epochs
    if num_epochs is not None:
        epoch_indices = epoch_indices[:num_epochs]
    
    # Apply stride
    epoch_indices = epoch_indices[::stride]
    
    # Get test activations and labels for all epochs
    test_activations = []
    test_labels = None
    
    for i, epoch in enumerate(epoch_indices):
        epoch_idx = activations["epoch"].index(epoch)
        test_activations.append(activations[layer_name]["test"][epoch_idx])
        
        # Use labels from the last epoch (they're the same for all epochs)
        if i == len(epoch_indices) - 1:
            test_labels = activations["labels"]["test"][epoch_idx]
    
    # Generate the trajectory visualization
    logger.info(f"Generating trajectory visualization for {experiment_name}, layer {layer_name}")
    
    fig, _ = plot_training_trajectory(
        activations_list=test_activations,
        labels=test_labels,
        epoch_indices=epoch_indices,
        title=f"{experiment_name} - {layer_name} Trajectory",
        n_components=VISUALIZATION["pca"]["n_components"],
        random_state=RANDOM_SEED
    )
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"{experiment_name}_trajectory_{layer_name}.png")
    fig.savefig(output_file, dpi=VISUALIZATION["plot"]["dpi"])
    plt.close(fig)
    
    logger.info(f"Saved trajectory visualization to {output_file}")


def visualize_comparative_trajectories(
    baseline_exp_name: str,
    regularized_exp_name: str,
    layer_name: str = "layer3",
    output_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    stride: int = 1
) -> None:
    """
    Generate comparative trajectory visualizations for baseline vs. regularized models.
    
    Args:
        baseline_exp_name: Name of the baseline experiment
        regularized_exp_name: Name of the regularized experiment
        layer_name: Name of the layer to visualize
        output_dir: Directory to save visualizations
        num_epochs: Number of epochs to visualize (None for all)
        stride: Stride for epoch selection
    """
    # Load experiment data
    baseline_data = load_experiment_data(baseline_exp_name)
    regularized_data = load_experiment_data(regularized_exp_name)
    
    baseline_activations = baseline_data["activations"]
    regularized_activations = regularized_data["activations"]
    
    if baseline_activations is None or regularized_activations is None:
        logger.error("Missing activations for comparative trajectory visualization")
        return
    
    # Extract dataset name
    dataset_name = baseline_data["dataset_name"]
    
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, f"{dataset_name}_comparison", "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get epoch indices
    baseline_epochs = baseline_activations["epoch"]
    regularized_epochs = regularized_activations["epoch"]
    
    # Find common epochs to compare
    common_epochs = sorted(set(baseline_epochs).intersection(set(regularized_epochs)))
    
    # Limit to requested number of epochs
    if num_epochs is not None:
        common_epochs = common_epochs[:num_epochs]
    
    # Apply stride
    common_epochs = common_epochs[::stride]
    
    # Get test activations and labels
    baseline_test_activations = []
    regularized_test_activations = []
    baseline_labels = None
    regularized_labels = None
    
    for i, epoch in enumerate(common_epochs):
        baseline_idx = baseline_activations["epoch"].index(epoch)
        regularized_idx = regularized_activations["epoch"].index(epoch)
        
        baseline_test_activations.append(baseline_activations[layer_name]["test"][baseline_idx])
        regularized_test_activations.append(regularized_activations[layer_name]["test"][regularized_idx])
        
        # Use labels from the last epoch (they're the same for all epochs)
        if i == len(common_epochs) - 1:
            baseline_labels = baseline_activations["labels"]["test"][baseline_idx]
            regularized_labels = regularized_activations["labels"]["test"][regularized_idx]
    
    # Generate the comparative trajectory visualization
    logger.info(f"Generating comparative trajectory visualization for {dataset_name}, layer {layer_name}")
    
    fig, _ = plot_comparative_trajectories(
        baseline_activations_list=baseline_test_activations,
        regularized_activations_list=regularized_test_activations,
        baseline_labels=baseline_labels,
        regularized_labels=regularized_labels,
        epoch_indices=common_epochs,
        title=f"{dataset_name} - {layer_name} Trajectory Comparison",
        n_components=VISUALIZATION["pca"]["n_components"],
        random_state=RANDOM_SEED
    )
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"{dataset_name}_comparative_trajectory_{layer_name}.png")
    fig.savefig(output_file, dpi=VISUALIZATION["plot"]["dpi"])
    plt.close(fig)
    
    logger.info(f"Saved comparative trajectory visualization to {output_file}")


def visualize_fragmentation_metrics(
    experiment_name: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Generate visualizations of fragmentation metrics over training.
    
    Args:
        experiment_name: Name of the experiment to visualize
        output_dir: Directory to save visualizations
    """
    # Load experiment data
    experiment_data = load_experiment_data(experiment_name)
    history = experiment_data["training_history"]
    
    if history is None:
        logger.error(f"No training history found for experiment {experiment_name}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(experiment_data["experiment_dir"], "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate fragmentation metrics visualization
    logger.info(f"Generating fragmentation metrics visualization for {experiment_name}")
    
    plt.figure(figsize=VISUALIZATION["plot"]["figsize"])
    plt.plot(history["entropy_fragmentation"], label="Entropy Fragmentation")
    plt.plot(history["angle_fragmentation"], label="Angle Fragmentation")
    plt.title(f"{experiment_name} - Fragmentation Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Fragmentation Score")
    plt.legend()
    plt.grid(True)
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"{experiment_name}_fragmentation_metrics.png")
    plt.savefig(output_file, dpi=VISUALIZATION["plot"]["dpi"])
    plt.close()
    
    logger.info(f"Saved fragmentation metrics visualization to {output_file}")
    
    # Generate accuracy and loss visualization
    logger.info(f"Generating accuracy and loss visualization for {experiment_name}")
    
    fig, ax1 = plt.subplots(figsize=VISUALIZATION["plot"]["figsize"])
    
    # Plot accuracy
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(history["train_accuracy"], label="Train Accuracy", color="tab:blue", linestyle="--")
    ax1.plot(history["test_accuracy"], label="Test Accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    
    # Create second y-axis for loss
    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="tab:red")
    ax2.plot(history["train_loss"], label="Train Loss", color="tab:red", linestyle="--")
    ax2.plot(history["test_loss"], label="Test Loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    plt.title(f"{experiment_name} - Accuracy and Loss")
    plt.grid(True)
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"{experiment_name}_accuracy_loss.png")
    plt.savefig(output_file, dpi=VISUALIZATION["plot"]["dpi"])
    plt.close()
    
    logger.info(f"Saved accuracy and loss visualization to {output_file}")
    
    # If regularization was used, visualize regularization loss
    if "regularized" in experiment_name:
        logger.info(f"Generating regularization loss visualization for {experiment_name}")
        
        plt.figure(figsize=VISUALIZATION["plot"]["figsize"])
        plt.plot(history["train_regularization_loss"], label="Regularization Loss")
        plt.title(f"{experiment_name} - Regularization Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save the visualization
        output_file = os.path.join(output_dir, f"{experiment_name}_regularization_loss.png")
        plt.savefig(output_file, dpi=VISUALIZATION["plot"]["dpi"])
        plt.close()
        
        logger.info(f"Saved regularization loss visualization to {output_file}")


def visualize_layer_comparison(
    experiment_name: str,
    output_dir: Optional[str] = None,
    epoch: int = -1  # -1 means the last epoch
) -> None:
    """
    Generate visualizations comparing activations across different layers.
    
    Args:
        experiment_name: Name of the experiment to visualize
        output_dir: Directory to save visualizations
        epoch: Epoch to visualize (-1 for last epoch)
    """
    # Load experiment data
    experiment_data = load_experiment_data(experiment_name)
    activations = experiment_data["activations"]
    
    if activations is None:
        logger.error(f"No activations found for experiment {experiment_name}")
        return
    
    if output_dir is None:
        output_dir = os.path.join(experiment_data["experiment_dir"], "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get epoch activations
    epoch_indices = activations["epoch"]
    if epoch == -1:
        epoch = epoch_indices[-1]  # Last epoch
    
    epoch_idx = epoch_indices.index(epoch)
    
    # Get test activations and labels for all layers
    test_activations = {}
    for layer_name in ["layer1", "layer2", "layer3"]:
        test_activations[layer_name] = activations[layer_name]["test"][epoch_idx]
    
    test_labels = activations["labels"]["test"][epoch_idx]
    
    # Generate multi-layer visualization
    logger.info(f"Generating layer comparison visualization for {experiment_name}, epoch {epoch}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (layer_name, layer_activations) in enumerate(test_activations.items()):
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        reduced_activations = pca.fit_transform(layer_activations)
        
        # Get unique labels
        unique_labels = torch.unique(test_labels).cpu().numpy()
        
        # Plot each class with a different color
        for label in unique_labels:
            mask = (test_labels.cpu().numpy() == label)
            axes[i].scatter(
                reduced_activations[mask, 0],
                reduced_activations[mask, 1],
                label=f"Class {label}",
                alpha=VISUALIZATION["plot"]["alpha"],
                s=VISUALIZATION["plot"]["markersize"]
            )
        
        axes[i].set_title(f"{layer_name}")
        axes[i].grid(True)
        
        if i == 0:
            axes[i].legend()
    
    plt.suptitle(f"{experiment_name} - Layer Comparison (Epoch {epoch})")
    plt.tight_layout()
    
    # Save the visualization
    output_file = os.path.join(output_dir, f"{experiment_name}_layer_comparison_epoch{epoch}.png")
    plt.savefig(output_file, dpi=VISUALIZATION["plot"]["dpi"])
    plt.close()
    
    logger.info(f"Saved layer comparison visualization to {output_file}")


def visualize_experiment(
    experiment_name: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Generate all visualizations for a single experiment.
    
    Args:
        experiment_name: Name of the experiment to visualize
        output_dir: Directory to save visualizations
    """
    # Set output directory
    if output_dir is None:
        experiment_data = load_experiment_data(experiment_name)
        output_dir = os.path.join(experiment_data["experiment_dir"], "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating visualizations for experiment: {experiment_name}")
    
    # Generate various visualizations
    try:
        # 1. Activation visualizations (PCA)
        for layer_name in ["layer1", "layer2", "layer3"]:
            visualize_activations(
                experiment_name=experiment_name,
                method="pca",
                layer_name=layer_name,
                output_dir=output_dir
            )
        
        # 2. Activation visualizations (UMAP)
        for layer_name in ["layer1", "layer2", "layer3"]:
            visualize_activations(
                experiment_name=experiment_name,
                method="umap",
                layer_name=layer_name,
                output_dir=output_dir
            )
        
        # 3. Trajectory visualizations
        for layer_name in ["layer1", "layer2", "layer3"]:
            visualize_trajectory(
                experiment_name=experiment_name,
                layer_name=layer_name,
                output_dir=output_dir,
                stride=2  # Visualize every 2 epochs
            )
        
        # 4. Layer comparison
        visualize_layer_comparison(
            experiment_name=experiment_name,
            output_dir=output_dir
        )
        
        # 5. Fragmentation metrics
        visualize_fragmentation_metrics(
            experiment_name=experiment_name,
            output_dir=output_dir
        )
        
        logger.info(f"All visualizations completed for experiment: {experiment_name}")
        
    except Exception as e:
        logger.error(f"Error generating visualizations for {experiment_name}: {str(e)}")


def compare_experiments(
    baseline_exp_name: str,
    regularized_exp_name: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Generate comparative visualizations for baseline vs. regularized models.
    
    Args:
        baseline_exp_name: Name of the baseline experiment
        regularized_exp_name: Name of the regularized experiment
        output_dir: Directory to save visualizations
    """
    # Load experiment data
    baseline_data = load_experiment_data(baseline_exp_name)
    regularized_data = load_experiment_data(regularized_exp_name)
    
    # Extract dataset name
    dataset_name = baseline_data["dataset_name"]
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, f"{dataset_name}_comparison", "visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating comparative visualizations for {dataset_name} (baseline vs. regularized)")
    
    try:
        # 1. Comparative trajectory visualizations
        for layer_name in ["layer1", "layer2", "layer3"]:
            visualize_comparative_trajectories(
                baseline_exp_name=baseline_exp_name,
                regularized_exp_name=regularized_exp_name,
                layer_name=layer_name,
                output_dir=output_dir,
                stride=2  # Visualize every 2 epochs
            )
        
        # 2. Comparative metrics visualization
        baseline_history = baseline_data["training_history"]
        regularized_history = regularized_data["training_history"]
        
        if baseline_history is not None and regularized_history is not None:
            # Accuracy comparison
            plt.figure(figsize=VISUALIZATION["plot"]["figsize"])
            plt.plot(baseline_history["test_accuracy"], label="Baseline")
            plt.plot(regularized_history["test_accuracy"], label="Regularized")
            plt.title(f"{dataset_name} - Test Accuracy Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_accuracy_comparison.png"), dpi=VISUALIZATION["plot"]["dpi"])
            plt.close()
            
            # Fragmentation metrics comparison
            plt.figure(figsize=VISUALIZATION["plot"]["figsize"])
            plt.plot(baseline_history["entropy_fragmentation"], label="Baseline (Entropy)")
            plt.plot(regularized_history["entropy_fragmentation"], label="Regularized (Entropy)")
            plt.title(f"{dataset_name} - Entropy Fragmentation Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Entropy Fragmentation Score")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_entropy_comparison.png"), dpi=VISUALIZATION["plot"]["dpi"])
            plt.close()
            
            plt.figure(figsize=VISUALIZATION["plot"]["figsize"])
            plt.plot(baseline_history["angle_fragmentation"], label="Baseline (Angle)")
            plt.plot(regularized_history["angle_fragmentation"], label="Regularized (Angle)")
            plt.title(f"{dataset_name} - Angle Fragmentation Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Angle Fragmentation Score")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{dataset_name}_angle_comparison.png"), dpi=VISUALIZATION["plot"]["dpi"])
            plt.close()
        
        logger.info(f"All comparative visualizations completed for {dataset_name}")
        
    except Exception as e:
        logger.error(f"Error generating comparative visualizations for {dataset_name}: {str(e)}")


def visualize_all_experiments():
    """Generate visualizations for all experiments in the results directory."""
    # List all experiment directories
    experiments = [d for d in os.listdir(RESULTS_DIR) 
                   if os.path.isdir(os.path.join(RESULTS_DIR, d)) 
                   and not d.endswith("_comparison")]
    
    logger.info(f"Found {len(experiments)} experiments to visualize")
    
    # First, visualize individual experiments
    for exp in experiments:
        try:
            visualize_experiment(exp)
        except Exception as e:
            logger.error(f"Error visualizing experiment {exp}: {str(e)}")
    
    # Group experiments by dataset
    dataset_experiments = {}
    for exp in experiments:
        # Skip non-experiment directories
        if exp.startswith(".") or not any(x in exp for x in ["baseline", "regularized"]):
            continue
            
        # Extract dataset name
        dataset = exp.split("_")[0]
        is_regularized = "regularized" in exp
        
        if dataset not in dataset_experiments:
            dataset_experiments[dataset] = {"baseline": [], "regularized": []}
        
        if is_regularized:
            dataset_experiments[dataset]["regularized"].append(exp)
        else:
            dataset_experiments[dataset]["baseline"].append(exp)
    
    logger.info(f"Grouped into {len(dataset_experiments)} datasets")
    
    # Then, create comparative visualizations
    for dataset, exps in dataset_experiments.items():
        for baseline_exp in exps["baseline"]:
            for regularized_exp in exps["regularized"]:
                try:
                    compare_experiments(baseline_exp, regularized_exp)
                except Exception as e:
                    logger.error(f"Error comparing {baseline_exp} vs {regularized_exp}: {str(e)}")


if __name__ == "__main__":
    visualize_all_experiments()
