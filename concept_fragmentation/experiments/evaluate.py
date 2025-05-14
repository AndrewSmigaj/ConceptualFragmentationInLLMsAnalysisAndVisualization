"""
Evaluation script for the Concept Fragmentation project.

This script:
1. Computes fragmentation metrics on test data
2. Analyzes correlation between fragmentation and generalization
3. Compares baseline vs. regularized models
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
from scipy.stats import pearsonr, spearmanr
import glob
from datetime import datetime
import sys

from concept_fragmentation.config import (
    MODELS, TRAINING, REGULARIZATION, RANDOM_SEED, RESULTS_DIR
)
from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.data import get_dataset_loader
from concept_fragmentation.metrics import (
    compute_cluster_entropy,
    compute_entropy_fragmentation_score,
    compute_subspace_angle,
    compute_angle_fragmentation_score
)
from concept_fragmentation.experiments.train import convert_tensors_to_python


# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "evaluation.log")),
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
        "experiment_dir": experiment_dir
    }


def load_model(experiment_name: str, model_type: str = "best") -> Tuple[FeedforwardNetwork, Dict[str, Any]]:
    """
    Load a trained model from an experiment.
    
    Args:
        experiment_name: Name of the experiment
        model_type: Type of model to load ("best" or "final")
        
    Returns:
        Tuple containing the loaded model and experiment data
    """
    # Load experiment data
    experiment_data = load_experiment_data(experiment_name)
    experiment_dir = experiment_data["experiment_dir"]
    dataset_name = experiment_data["dataset_name"]
    
    # Determine model path
    if model_type == "best":
        model_path = os.path.join(experiment_dir, "best_model.pt")
    else:
        model_path = os.path.join(experiment_dir, "final_model.pt")
    
    if not os.path.exists(model_path):
        raise ValueError(f"Model file {model_path} not found")
    
    # Get dataset info to determine model input/output dimensions
    from concept_fragmentation.experiments.train import prepare_dataset
    
    _, _, input_dim, num_classes = prepare_dataset(dataset_name)
    
    # Initialize the model
    model = FeedforwardNetwork(
        input_dim=input_dim, 
        output_dim=num_classes,
        hidden_layer_sizes=MODELS["feedforward"]["hidden_dims"][dataset_name],
        dropout_rate=MODELS["feedforward"]["dropout"],
        activation=MODELS["feedforward"]["activation"],
        seed=RANDOM_SEED
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    return model, experiment_data


def compute_detailed_metrics(
    model: FeedforwardNetwork,
    dataset_name: str
) -> Dict[str, Any]:
    """
    Compute detailed metrics for a model on a dataset.
    
    Args:
        model: Trained model
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing detailed metrics
    """
    from concept_fragmentation.experiments.train import prepare_dataset
    
    # Prepare dataset
    _, test_data, _, _ = prepare_dataset(dataset_name)
    
    # Compute metrics on test data
    model.eval()
    
    # Initialize metrics
    metrics = {
        "accuracy": 0.0,
        "loss": 0.0,
        "entropy_fragmentation": {},
        "angle_fragmentation": {},
        "layer_metrics": {}
    }
    
    # Get test data
    if isinstance(test_data, tuple):
        X_test, y_test = test_data
        
        # Compute accuracy and loss
        with torch.no_grad():
            outputs = model(X_test)
            loss = torch.nn.functional.cross_entropy(outputs, y_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).sum().item() / len(y_test)
        
        # Update metrics
        metrics["accuracy"] = accuracy
        metrics["loss"] = loss.item()
        
        # Compute fragmentation metrics for each layer
        for layer_name in ["layer1", "layer2", "layer3"]:
            # Get activations for this layer
            layer_activations = model.activations[layer_name]
            
            # Compute entropy fragmentation
            entropy_results = compute_cluster_entropy(layer_activations, y_test)
            
            # Compute angle fragmentation
            angle_results = compute_subspace_angle(layer_activations, y_test)
            
            # Store in metrics
            metrics["layer_metrics"][layer_name] = {
                "entropy": entropy_results,
                "angle": angle_results
            }
        
        # Overall fragmentation scores
        metrics["entropy_fragmentation"] = metrics["layer_metrics"]["layer3"]["entropy"]
        metrics["angle_fragmentation"] = metrics["layer_metrics"]["layer3"]["angle"]
    
    else:
        # Handle DataLoader case (e.g., Fashion MNIST)
        correct = 0
        total = 0
        running_loss = 0.0
        all_activations = {layer: [] for layer in ["layer1", "layer2", "layer3"]}
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_data:
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Store activations and labels
                for layer_name in ["layer1", "layer2", "layer3"]:
                    all_activations[layer_name].append(model.activations[layer_name].detach().cpu())
                all_labels.append(labels.detach().cpu())
        
        # Update metrics
        metrics["accuracy"] = correct / total
        metrics["loss"] = running_loss / total
        
        # Concatenate activations and labels
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute fragmentation metrics for each layer
        for layer_name in ["layer1", "layer2", "layer3"]:
            # Concatenate activations for this layer
            layer_activations = torch.cat(all_activations[layer_name], dim=0)
            
            # Compute entropy fragmentation
            entropy_results = compute_cluster_entropy(layer_activations, all_labels)
            
            # Compute angle fragmentation
            angle_results = compute_subspace_angle(layer_activations, all_labels)
            
            # Store in metrics
            metrics["layer_metrics"][layer_name] = {
                "entropy": entropy_results,
                "angle": angle_results
            }
        
        # Overall fragmentation scores
        metrics["entropy_fragmentation"] = metrics["layer_metrics"]["layer3"]["entropy"]
        metrics["angle_fragmentation"] = metrics["layer_metrics"]["layer3"]["angle"]
    
    return metrics


def evaluate_model(experiment_name: str, model_type: str = "best") -> Dict[str, Any]:
    """
    Evaluate a trained model and compute fragmentation metrics.
    
    Args:
        experiment_name: Name of the experiment to evaluate
        model_type: Type of model to load ("best" or "final")
        
    Returns:
        Dictionary containing evaluation metrics
    """
    logger.info(f"Evaluating experiment: {experiment_name} (model: {model_type})")
    
    # Load model and experiment data
    model, experiment_data = load_model(experiment_name, model_type)
    dataset_name = experiment_data["dataset_name"]
    experiment_dir = experiment_data["experiment_dir"]
    
    # Compute detailed metrics
    metrics = compute_detailed_metrics(model, dataset_name)
    
    # Add experiment info to metrics
    metrics["experiment_name"] = experiment_name
    metrics["dataset_name"] = dataset_name
    metrics["is_regularized"] = experiment_data["is_regularized"]
    metrics["model_type"] = model_type
    
    # Save metrics to file
    metrics_path = os.path.join(experiment_dir, f"detailed_metrics_{model_type}.json")
    
    # Convert metrics to JSON-serializable format using the helper function
    serializable_metrics = convert_tensors_to_python(metrics)
    
    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)
    
    logger.info(f"Saved detailed metrics to {metrics_path}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Loss: {metrics['loss']:.4f}, "
                f"Entropy Fragmentation: {metrics['entropy_fragmentation']['mean_entropy']:.4f}, "
                f"Angle Fragmentation: {metrics['angle_fragmentation']['mean_angle']:.4f}")
    
    return metrics


def compare_models(baseline_exp_name: str, regularized_exp_name: str, model_type: str = "best") -> Dict[str, Any]:
    """
    Compare baseline and regularized models.
    
    Args:
        baseline_exp_name: Name of the baseline experiment
        regularized_exp_name: Name of the regularized experiment
        model_type: Type of model to load ("best" or "final")
        
    Returns:
        Dictionary containing comparison metrics
    """
    logger.info(f"Comparing baseline ({baseline_exp_name}) vs regularized ({regularized_exp_name})")
    
    # Evaluate both models
    baseline_metrics = evaluate_model(baseline_exp_name, model_type)
    regularized_metrics = evaluate_model(regularized_exp_name, model_type)
    
    # Extract dataset name
    dataset_name = baseline_metrics["dataset_name"]
    
    # Create comparison directory
    comparison_dir = os.path.join(RESULTS_DIR, f"{dataset_name}_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compute differences
    comparison = {
        "dataset_name": dataset_name,
        "baseline_experiment": baseline_exp_name,
        "regularized_experiment": regularized_exp_name,
        "model_type": model_type,
        "accuracy": {
            "baseline": baseline_metrics["accuracy"],
            "regularized": regularized_metrics["accuracy"],
            "difference": regularized_metrics["accuracy"] - baseline_metrics["accuracy"],
            "relative_improvement": (regularized_metrics["accuracy"] - baseline_metrics["accuracy"]) / baseline_metrics["accuracy"] if baseline_metrics["accuracy"] > 0 else 0
        },
        "loss": {
            "baseline": baseline_metrics["loss"],
            "regularized": regularized_metrics["loss"],
            "difference": baseline_metrics["loss"] - regularized_metrics["loss"],
            "relative_improvement": (baseline_metrics["loss"] - regularized_metrics["loss"]) / baseline_metrics["loss"] if baseline_metrics["loss"] > 0 else 0
        },
        "entropy_fragmentation": {
            "baseline": baseline_metrics["entropy_fragmentation"]["mean_entropy"],
            "regularized": regularized_metrics["entropy_fragmentation"]["mean_entropy"],
            "difference": baseline_metrics["entropy_fragmentation"]["mean_entropy"] - regularized_metrics["entropy_fragmentation"]["mean_entropy"],
            "relative_improvement": (baseline_metrics["entropy_fragmentation"]["mean_entropy"] - regularized_metrics["entropy_fragmentation"]["mean_entropy"]) / baseline_metrics["entropy_fragmentation"]["mean_entropy"] if baseline_metrics["entropy_fragmentation"]["mean_entropy"] > 0 else 0
        },
        "angle_fragmentation": {
            "baseline": baseline_metrics["angle_fragmentation"]["mean_angle"],
            "regularized": regularized_metrics["angle_fragmentation"]["mean_angle"],
            "difference": baseline_metrics["angle_fragmentation"]["mean_angle"] - regularized_metrics["angle_fragmentation"]["mean_angle"],
            "relative_improvement": (baseline_metrics["angle_fragmentation"]["mean_angle"] - regularized_metrics["angle_fragmentation"]["mean_angle"]) / baseline_metrics["angle_fragmentation"]["mean_angle"] if baseline_metrics["angle_fragmentation"]["mean_angle"] > 0 else 0
        },
        "layer_metrics": {}
    }
    
    # Compare layer-wise metrics
    for layer_name in ["layer1", "layer2", "layer3"]:
        comparison["layer_metrics"][layer_name] = {
            "entropy": {
                "baseline": baseline_metrics["layer_metrics"][layer_name]["entropy"]["mean_entropy"],
                "regularized": regularized_metrics["layer_metrics"][layer_name]["entropy"]["mean_entropy"],
                "difference": baseline_metrics["layer_metrics"][layer_name]["entropy"]["mean_entropy"] - regularized_metrics["layer_metrics"][layer_name]["entropy"]["mean_entropy"]
            },
            "angle": {
                "baseline": baseline_metrics["layer_metrics"][layer_name]["angle"]["mean_angle"],
                "regularized": regularized_metrics["layer_metrics"][layer_name]["angle"]["mean_angle"],
                "difference": baseline_metrics["layer_metrics"][layer_name]["angle"]["mean_angle"] - regularized_metrics["layer_metrics"][layer_name]["angle"]["mean_angle"]
            }
        }
    
    # Save comparison results
    comparison_path = os.path.join(comparison_dir, f"model_comparison_{model_type}.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=4)
    
    logger.info(f"Saved comparison results to {comparison_path}")
    
    # Log key findings
    logger.info(f"Accuracy improvement: {comparison['accuracy']['difference']:.4f} "
                f"({comparison['accuracy']['relative_improvement']*100:.2f}%)")
    logger.info(f"Entropy fragmentation reduction: {comparison['entropy_fragmentation']['difference']:.4f} "
                f"({comparison['entropy_fragmentation']['relative_improvement']*100:.2f}%)")
    logger.info(f"Angle fragmentation reduction: {comparison['angle_fragmentation']['difference']:.4f} "
                f"({comparison['angle_fragmentation']['relative_improvement']*100:.2f}%)")
    
    # Generate comparison plots
    generate_comparison_plots(baseline_exp_name, regularized_exp_name, comparison_dir)
    
    return comparison


def generate_comparison_plots(
    baseline_exp_name: str,
    regularized_exp_name: str,
    output_dir: str
):
    """
    Generate comparison plots between baseline and regularized models.
    
    Args:
        baseline_exp_name: Name of the baseline experiment
        regularized_exp_name: Name of the regularized experiment
        output_dir: Directory to save the plots
    """
    # Load experiment data
    baseline_data = load_experiment_data(baseline_exp_name)
    regularized_data = load_experiment_data(regularized_exp_name)
    
    # Get training histories
    baseline_history = baseline_data["training_history"]
    regularized_history = regularized_data["training_history"]
    
    if baseline_history is None or regularized_history is None:
        logger.warning("Training history not available for comparison plots")
        return
    
    # Create subplots for accuracy, loss, and fragmentation
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Accuracy comparison
    axes[0].plot(baseline_history["train_accuracy"], label="Baseline (Train)")
    axes[0].plot(baseline_history["test_accuracy"], label="Baseline (Test)")
    axes[0].plot(regularized_history["train_accuracy"], label="Regularized (Train)")
    axes[0].plot(regularized_history["test_accuracy"], label="Regularized (Test)")
    axes[0].set_title("Accuracy Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss comparison
    axes[1].plot(baseline_history["train_loss"], label="Baseline (Train)")
    axes[1].plot(baseline_history["test_loss"], label="Baseline (Test)")
    axes[1].plot(regularized_history["train_loss"], label="Regularized (Train)")
    axes[1].plot(regularized_history["test_loss"], label="Regularized (Test)")
    axes[1].set_title("Loss Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    # Fragmentation comparison
    axes[2].plot(baseline_history["entropy_fragmentation"], label="Baseline (Entropy)")
    axes[2].plot(baseline_history["angle_fragmentation"], label="Baseline (Angle)")
    axes[2].plot(regularized_history["entropy_fragmentation"], label="Regularized (Entropy)")
    axes[2].plot(regularized_history["angle_fragmentation"], label="Regularized (Angle)")
    axes[2].set_title("Fragmentation Metrics Comparison")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Fragmentation Score")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Saved comparison plots to {plot_path}")


def analyze_fragmentation_generalization_correlation(
    experiment_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze correlation between fragmentation metrics and generalization performance.
    
    Args:
        experiment_results: List of dictionaries containing experiment results
        
    Returns:
        Dictionary containing correlation analysis results
    """
    # Extract metrics for correlation analysis
    accuracies = []
    entropy_scores = []
    angle_scores = []
    
    for result in experiment_results:
        accuracies.append(result["accuracy"])
        entropy_scores.append(result["entropy_fragmentation"]["mean_entropy"])
        angle_scores.append(result["angle_fragmentation"]["mean_angle"])
    
    # Calculate correlations
    entropy_accuracy_corr, entropy_accuracy_pval = pearsonr(entropy_scores, accuracies)
    angle_accuracy_corr, angle_accuracy_pval = pearsonr(angle_scores, accuracies)
    
    # Calculate Spearman rank correlations
    entropy_accuracy_spearman, entropy_accuracy_spearman_pval = spearmanr(entropy_scores, accuracies)
    angle_accuracy_spearman, angle_accuracy_spearman_pval = spearmanr(angle_scores, accuracies)
    
    correlation_results = {
        "pearson": {
            "entropy_accuracy": {
                "correlation": entropy_accuracy_corr,
                "p_value": entropy_accuracy_pval
            },
            "angle_accuracy": {
                "correlation": angle_accuracy_corr,
                "p_value": angle_accuracy_pval
            }
        },
        "spearman": {
            "entropy_accuracy": {
                "correlation": entropy_accuracy_spearman,
                "p_value": entropy_accuracy_spearman_pval
            },
            "angle_accuracy": {
                "correlation": angle_accuracy_spearman,
                "p_value": angle_accuracy_spearman_pval
            }
        }
    }
    
    logger.info("Correlation Analysis:")
    logger.info(f"Entropy-Accuracy Pearson correlation: {entropy_accuracy_corr:.4f} (p-value: {entropy_accuracy_pval:.4f})")
    logger.info(f"Angle-Accuracy Pearson correlation: {angle_accuracy_corr:.4f} (p-value: {angle_accuracy_pval:.4f})")
    logger.info(f"Entropy-Accuracy Spearman correlation: {entropy_accuracy_spearman:.4f} (p-value: {entropy_accuracy_spearman_pval:.4f})")
    logger.info(f"Angle-Accuracy Spearman correlation: {angle_accuracy_spearman:.4f} (p-value: {angle_accuracy_spearman_pval:.4f})")
    
    # Generate scatter plots
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(entropy_scores, accuracies)
    plt.title(f"Entropy vs. Accuracy (r={entropy_accuracy_corr:.4f})")
    plt.xlabel("Entropy Fragmentation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(angle_scores, accuracies)
    plt.title(f"Angle vs. Accuracy (r={angle_accuracy_corr:.4f})")
    plt.xlabel("Angle Fragmentation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "fragmentation_correlation.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Saved correlation plots to {plot_path}")
    
    return correlation_results


def evaluate_all_experiments(model_type: str = "best"):
    """
    Evaluate all experiments in the results directory.
    
    Args:
        model_type: Type of model to load ("best" or "final")
    """
    # List all experiment directories
    experiments = [d for d in os.listdir(RESULTS_DIR) 
                   if os.path.isdir(os.path.join(RESULTS_DIR, d)) 
                   and not d.endswith("_comparison")]
    
    logger.info(f"Found {len(experiments)} experiments to evaluate")
    
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
    
    # Store all evaluation results for correlation analysis
    all_results = []
    
    # Evaluate each experiment
    for dataset, exps in dataset_experiments.items():
        logger.info(f"Evaluating experiments for dataset: {dataset}")
        
        # Evaluate each experiment individually
        for exp_type in ["baseline", "regularized"]:
            for exp in exps[exp_type]:
                try:
                    result = evaluate_model(exp, model_type)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating experiment {exp}: {str(e)}")
        
        # Compare pairs of baseline and regularized experiments
        for baseline_exp in exps["baseline"]:
            for regularized_exp in exps["regularized"]:
                try:
                    compare_models(baseline_exp, regularized_exp, model_type)
                except Exception as e:
                    logger.error(f"Error comparing {baseline_exp} vs {regularized_exp}: {str(e)}")
    
    # Perform correlation analysis
    if all_results:
        analyze_fragmentation_generalization_correlation(all_results)
    else:
        logger.warning("No results available for correlation analysis")


if __name__ == "__main__":
    # Evaluate all experiments using the best model
    evaluate_all_experiments(model_type="best")
