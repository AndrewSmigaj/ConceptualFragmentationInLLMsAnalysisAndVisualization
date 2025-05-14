"""
Training script for the Concept Fragmentation project.

This script:
1. Loads datasets (Titanic, Adult Income, Heart Disease, Fashion-MNIST subset)
2. Trains models with and without cohesion regularization
3. Captures activations during training
4. Logs metrics including fragmentation measures
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json
import pickle
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

from concept_fragmentation.config import (
    MODELS, TRAINING, REGULARIZATION, RANDOM_SEED, RESULTS_DIR, DATASETS
)
from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.models.regularizers import CohesionRegularizer
from concept_fragmentation.data import get_dataset_loader, DataPreprocessor
from concept_fragmentation.metrics import (
    compute_cluster_entropy,
    compute_entropy_fragmentation_score,
    compute_subspace_angle,
    compute_angle_fragmentation_score
)


# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_dataset(
    dataset_name: str
) -> Tuple[
    Union[torch.utils.data.DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    Union[torch.utils.data.DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    int,
    int
]:
    """
    Prepare dataset for training.
    
    Args:
        dataset_name: Name of the dataset to prepare
        
    Returns:
        Tuple containing:
            - Training data (dataloader or tensors)
            - Testing data (dataloader or tensors)
            - Input dimension
            - Number of classes
    """
    logger.info(f"Preparing dataset: {dataset_name}")
    
    if dataset_name == "fashion_mnist":
        # Fashion MNIST is already loaded as DataLoader
        train_loader, test_loader = get_dataset_loader(dataset_name).load_data()
        
        # Get input dimension from a sample
        sample_data, _ = next(iter(train_loader))
        input_dim = sample_data[0].flatten().shape[0]
        
        # Get number of classes
        num_classes = len(DATASETS["fashion_mnist"]["classes"])
        
        return train_loader, test_loader, input_dim, num_classes
    
    elif dataset_name in ["titanic", "adult", "heart"]:
        # Load the tabular dataset
        DatasetClass = get_dataset_loader(dataset_name)
        train_df, test_df = DatasetClass.load_data()
        
        # Get dataset configuration
        dataset_config = DATASETS[dataset_name]
        
        # Create a generic preprocessor
        preprocessor = DataPreprocessor(
            categorical_cols=dataset_config["categorical_features"],
            numerical_cols=dataset_config["numerical_features"],
            target_col=dataset_config["target"]
        )
        
        # Preprocess the data
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_test, y_test = preprocessor.transform(test_df)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)
        
        # Get dimensions
        input_dim = X_train_tensor.shape[1]
        num_classes = len(np.unique(y_train))
        
        return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), input_dim, num_classes
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_model(
    dataset_name: str,
    use_regularization: bool = False,
    save_activations: bool = True,
    save_model: bool = True,
    save_fragmentation_metrics: bool = True,
    experiment_name: Optional[str] = None,
    experiment_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    early_stopping_patience: Optional[int] = None,
    random_seed: int = RANDOM_SEED,
    device: Optional[str] = None,
    reg_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a feedforward neural network on the specified dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        use_regularization: Whether to use cohesion regularization
        save_activations: Whether to save activations during training
        save_model: Whether to save the trained model
        save_fragmentation_metrics: Whether to compute and save fragmentation metrics
        experiment_name: Custom name for the experiment (default: auto-generated)
        experiment_dir: Custom directory for storing experiment results (default: auto-generated)
        epochs: Number of epochs to train (default: from config)
        batch_size: Batch size for training (default: from config)
        early_stopping_patience: Number of epochs to wait for improvement (default: from config)
        random_seed: Random seed for reproducibility (default: from config)
        device: Device to run model on ("cuda" or "cpu")
        reg_params: Dictionary of regularization parameters (default: from config)
        
    Returns:
        Dictionary containing training metrics and results
    """
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create experiment name if not provided
    if experiment_name is None:
        reg_suffix = "_regularized" if use_regularization else "_baseline"
        seed_suffix = f"_seed{random_seed}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{dataset_name}{reg_suffix}{seed_suffix}_{timestamp}"
    
    # Create results directory
    if experiment_dir is None:
        experiment_dir = os.path.join(RESULTS_DIR, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Configure logging for this experiment
    file_handler = logging.FileHandler(os.path.join(experiment_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Results will be saved to: {experiment_dir}")
    logger.info(f"Using regularization: {use_regularization}")
    if use_regularization and reg_params:
        logger.info(f"Regularization parameters: {reg_params}")
    logger.info(f"Using random seed: {random_seed}")
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Get training parameters
    if epochs is None:
        epochs = TRAINING["epochs"][dataset_name]
    if batch_size is None:
        batch_size = TRAINING["batch_size"][dataset_name]
    if early_stopping_patience is None:
        early_stopping_patience = TRAINING["early_stopping"]["patience"]
    
    logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Prepare dataset
    dataset_data = prepare_dataset(dataset_name)
    train_data, test_data, input_dim, num_classes = dataset_data
    
    logger.info(f"Dataset prepared: input_dim={input_dim}, num_classes={num_classes}")
    
    # Initialize model
    model = FeedforwardNetwork(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_layer_sizes=MODELS["feedforward"]["hidden_dims"][dataset_name],
        dropout_rate=MODELS["feedforward"]["dropout"],
        activation=MODELS["feedforward"]["activation"],
        seed=random_seed
    )
    
    # Move model to device
    model = model.to(device)
    
    logger.info(f"Model initialized: {model}")
    
    # Initialize optimizer
    if TRAINING["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAINING["lr"],
            weight_decay=TRAINING["weight_decay"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=TRAINING["lr"],
            momentum=0.9,
            weight_decay=TRAINING["weight_decay"]
        )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize regularizer if needed
    regularizer = None
    if use_regularization:
        # Use provided reg_params if available, otherwise use defaults from config
        if reg_params is None:
            reg_params = {
                "weight": REGULARIZATION["cohesion"]["weight"],
                "temperature": REGULARIZATION["cohesion"]["temperature"],
                "threshold": REGULARIZATION["cohesion"]["similarity_threshold"],
                "minibatch_size": REGULARIZATION["cohesion"]["minibatch_size"]
            }
        
        regularizer = CohesionRegularizer(
            weight=reg_params.get("weight", REGULARIZATION["cohesion"]["weight"]),
            temperature=reg_params.get("temperature", REGULARIZATION["cohesion"]["temperature"]),
            threshold=reg_params.get("similarity_threshold", REGULARIZATION["cohesion"]["similarity_threshold"]),
            minibatch_size=reg_params.get("minibatch_size", REGULARIZATION["cohesion"]["minibatch_size"]),
            seed=random_seed
        )
        logger.info(f"Regularizer initialized: {regularizer.__class__.__name__} with params: {reg_params}")
    
    # Initialize storage for training history
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_regularization_loss": [],
        "test_loss": [],
        "test_accuracy": [],
        "entropy_fragmentation": [],
        "angle_fragmentation": []
    }
    
    # Store experiment configuration
    training_history["config"] = {
        "dataset": dataset_name,
        "model": "feedforward",
        "hidden_dims": MODELS["feedforward"]["hidden_dims"][dataset_name],
        "dropout": MODELS["feedforward"]["dropout"],
        "optimizer": TRAINING["optimizer"],
        "learning_rate": TRAINING["lr"],
        "weight_decay": TRAINING["weight_decay"],
        "batch_size": batch_size,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "random_seed": random_seed,
        "use_regularization": use_regularization
    }
    
    # Store regularization parameters if used
    if use_regularization and reg_params:
        training_history["config"]["regularization"] = reg_params
    
    # Initialize storage for activations
    if save_activations:
        layer_activations = {
            "epoch": [],
            "layer1": {"train": [], "test": []},
            "layer2": {"train": [], "test": []},
            "layer3": {"train": [], "test": []},
            "output": {"train": [], "test": []},
            "labels": {"train": [], "test": []}
        }
    else:
        layer_activations = None
    
    # Training loop
    best_test_accuracy = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_reg_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Store activations and labels for this epoch
        epoch_activations = {
            "layer1": [],
            "layer2": [],
            "layer3": [],
            "output": []
        }
        epoch_labels = []
        
        # Handle different data formats (DataLoader vs. tensors)
        if isinstance(train_data, tuple):
            # Data is already in tensors (X_train, y_train)
            X_train, y_train = train_data
            
            # Split into batches
            num_samples = len(X_train)
            indices = torch.randperm(num_samples)
            
            for i in range(0, num_samples, batch_size):
                # Get batch indices
                batch_indices = indices[i:min(i+batch_size, num_samples)]
                
                # Get batch data
                inputs = X_train[batch_indices]
                labels = y_train[batch_indices]
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Add regularization loss if enabled
                reg_loss = 0.0
                if use_regularization:
                    activations_dict = model.get_activations()
                    # If layers are specified in reg_params, use them
                    if reg_params and "layers" in reg_params and reg_params["layers"]:
                        reg_loss = regularizer.get_total_loss(activations_dict, labels, layer_names=reg_params["layers"])
                    else:
                        reg_loss = regularizer(activations_dict, labels)
                    loss += reg_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * len(labels)
                train_reg_loss += reg_loss * len(labels) if use_regularization else 0.0
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += len(labels)
                
                # Store activations and labels for later analysis
                if save_activations:
                    activations_dict = model.get_activations()
                    for layer_name, activations in activations_dict.items():
                        epoch_activations[layer_name].append(activations.detach().cpu())
                    epoch_labels.append(labels.detach().cpu())
                
        else:
            # Data is in a DataLoader
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_data, desc="Training")):
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Add regularization loss if enabled
                reg_loss = 0.0
                if use_regularization:
                    activations_dict = model.get_activations()
                    # If layers are specified in reg_params, use them
                    if reg_params and "layers" in reg_params and reg_params["layers"]:
                        reg_loss = regularizer.get_total_loss(activations_dict, labels, layer_names=reg_params["layers"])
                    else:
                        reg_loss = regularizer(activations_dict, labels)
                    loss += reg_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * len(labels)
                train_reg_loss += reg_loss * len(labels) if use_regularization else 0.0
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += len(labels)
                
                # Store activations and labels for later analysis
                if save_activations:
                    activations_dict = model.get_activations()
                    for layer_name, activations in activations_dict.items():
                        epoch_activations[layer_name].append(activations.detach().cpu())
                    epoch_labels.append(labels.detach().cpu())
        
        # Calculate average loss and accuracy
        train_loss /= train_total
        train_reg_loss /= train_total
        train_accuracy = train_correct / train_total
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.4f}")
        
        # Concatenate activations and labels from all batches
        if save_activations:
            for layer_name in epoch_activations.keys():
                if epoch_activations[layer_name]:
                    layer_activations["layer1"]["train"].append(torch.cat(epoch_activations[layer_name], dim=0))
            layer_activations["labels"]["train"].append(torch.cat(epoch_labels, dim=0))
            layer_activations["epoch"].append(epoch)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Store test activations and labels for this epoch
        test_epoch_activations = {
            "layer1": [],
            "layer2": [],
            "layer3": [],
            "output": []
        }
        test_epoch_labels = []
        
        with torch.no_grad():
            # Handle different data formats (DataLoader vs. tensors)
            if isinstance(test_data, tuple):
                # Data is already in tensors (X_test, y_test)
                X_test, y_test = test_data
                
                # Forward pass
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                
                # Update statistics
                test_loss += loss.item() * len(y_test)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == y_test).sum().item()
                test_total += len(y_test)
                
                # Store activations and labels for later analysis
                if save_activations:
                    activations_dict = model.get_activations()
                    for layer_name, activations in activations_dict.items():
                        test_epoch_activations[layer_name].append(activations.detach().cpu())
                    test_epoch_labels.append(y_test.detach().cpu())
                
            else:
                # Data is in a DataLoader
                for batch_idx, (inputs, labels) in enumerate(tqdm(test_data, desc="Testing")):
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Update statistics
                    test_loss += loss.item() * len(labels)
                    _, predicted = torch.max(outputs.data, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_total += len(labels)
                    
                    # Store activations and labels for later analysis
                    if save_activations:
                        activations_dict = model.get_activations()
                        for layer_name, activations in activations_dict.items():
                            test_epoch_activations[layer_name].append(activations.detach().cpu())
                        test_epoch_labels.append(labels.detach().cpu())
        
        # Calculate average loss and accuracy
        test_loss /= test_total
        test_accuracy = test_correct / test_total
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Test Loss: {test_loss:.4f}, "
                   f"Test Accuracy: {test_accuracy:.4f}")
        
        # Concatenate test activations and labels from all batches
        if save_activations:
            for layer_name in test_epoch_activations.keys():
                if test_epoch_activations[layer_name]:
                    layer_activations[layer_name]["test"].append(torch.cat(test_epoch_activations[layer_name], dim=0))
            layer_activations["labels"]["test"].append(torch.cat(test_epoch_labels, dim=0))
        
        # Compute fragmentation metrics on test data
        if save_fragmentation_metrics:
            layer_name = "layer3"  # Use the last hidden layer
            
            if isinstance(test_data, tuple):
                # Get activations and labels from model
                model.eval()
                with torch.no_grad():
                    _ = model(X_test)
                test_activations = model.activations[layer_name]
                test_labels = y_test
            else:
                # Concatenate activations from all batches
                test_activations = torch.cat(test_epoch_activations[layer_name], dim=0)
                test_labels = torch.cat(test_epoch_labels, dim=0)
            
            # Compute cluster entropy
            entropy_score = compute_entropy_fragmentation_score(
                test_activations, test_labels
            )
            
            # Compute subspace angle
            angle_score = compute_angle_fragmentation_score(
                test_activations, test_labels
            )
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Fragmentation Metrics: "
                       f"Entropy Score: {entropy_score:.4f}, "
                       f"Angle Score: {angle_score:.4f}")
        else:
            entropy_score = 0.0
            angle_score = 0.0
        
        # Update training history
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["train_regularization_loss"].append(train_reg_loss)
        training_history["test_loss"].append(test_loss)
        training_history["test_accuracy"].append(test_accuracy)
        training_history["entropy_fragmentation"].append(entropy_score)
        training_history["angle_fragmentation"].append(angle_score)
        
        # Check for early stopping
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            patience_counter = 0
            
            # Save the best model
            if save_model:
                best_model_path = os.path.join(experiment_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs "
                       f"(patience: {early_stopping_patience})")
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break
    
    # Save the final model
    if save_model:
        final_model_path = os.path.join(experiment_dir, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")
    
    # Save the activations if requested
    if save_activations and layer_activations:
        activations_path = os.path.join(experiment_dir, "layer_activations.pkl")
        with open(activations_path, "wb") as f:
            pickle.dump(layer_activations, f)
        logger.info(f"Saved layer activations to {activations_path}")
    
    # Plot training curves
    plot_training_curves(training_history, experiment_dir)
    
    # Remove the file handler for this experiment
    logger.removeHandler(file_handler)
    
    return {
        "model": model,
        "history": training_history,
        "activations": layer_activations if save_activations else None,
        "experiment_name": experiment_name
    }


def plot_training_curves(history: Dict[str, List[float]], output_dir: str):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save the plots
    """
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["test_accuracy"], label="Test Accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()
    
    # Plot fragmentation metrics
    plt.figure(figsize=(10, 5))
    plt.plot(history["entropy_fragmentation"], label="Entropy Fragmentation")
    plt.plot(history["angle_fragmentation"], label="Angle Fragmentation")
    plt.title("Fragmentation Metrics over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Fragmentation Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "fragmentation.png"))
    plt.close()


def run_all_experiments(seeds=[RANDOM_SEED], use_regularization=False):
    """
    Run experiments for all datasets with specified seeds.
    
    Args:
        seeds: List of random seeds to use for each experiment
        use_regularization: Whether to use cohesion regularization
    """
    datasets = ["titanic", "adult", "heart", "fashion_mnist"]
    
    for dataset_name in datasets:
        logger.info(f"Running experiments for dataset: {dataset_name}")
        
        # Run with each seed
        for seed in seeds:
            logger.info(f"Using seed: {seed}")
            
            # Run model with the specified configuration
            train_model(
                dataset_name=dataset_name, 
                use_regularization=use_regularization,
                random_seed=seed
            )
        
        logger.info(f"Completed experiments for dataset: {dataset_name}")


if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train neural networks for concept fragmentation analysis')
    parser.add_argument('--dataset', type=str, choices=['titanic', 'adult', 'heart', 'fashion_mnist'], 
                        help='Dataset to use for training')
    parser.add_argument('--seeds', type=int, nargs='+', default=[RANDOM_SEED],
                        help='Random seeds for reproducibility (can provide multiple)')
    parser.add_argument('--regularization', action='store_true',
                        help='Use cohesion regularization during training')
    
    args = parser.parse_args()
    
    # Run experiments based on arguments
    if args.dataset:
        # Run for a specific dataset with specified seeds
        for seed in args.seeds:
            logger.info(f"Running experiment for dataset: {args.dataset} with seed: {seed}")
            train_model(
                dataset_name=args.dataset,
                use_regularization=args.regularization,
                random_seed=seed
            )
    else:
        # Run for all datasets with default seed
        run_all_experiments(seeds=args.seeds, use_regularization=args.regularization)
