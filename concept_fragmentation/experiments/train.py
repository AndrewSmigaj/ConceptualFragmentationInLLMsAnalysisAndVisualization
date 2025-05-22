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
import sys

from concept_fragmentation.config import (
    MODELS, TRAINING, REGULARIZATION, RANDOM_SEED, RESULTS_DIR, DATASETS
)
from concept_fragmentation.utils.path_utils import build_experiment_dir
from concept_fragmentation.models.feedforward import FeedforwardNetwork
from concept_fragmentation.models.regularizers import CohesionRegularizer
from concept_fragmentation.data import get_dataset_loader, DataPreprocessor
from concept_fragmentation.metrics import (
    compute_cluster_entropy,
    compute_entropy_fragmentation_score,
    compute_subspace_angle,
    compute_angle_fragmentation_score,
    compute_intra_class_distance,
    compute_icpd_fragmentation_score,
    compute_optimal_k,
    compute_kstar_fragmentation_score,
    compute_representation_stability,
    compute_stability_fragmentation_score,
    compute_layer_stability_profile
)
from concept_fragmentation.utils.cluster_paths import (
    prepare_cluster_path_data,
    save_cluster_paths as save_paths
)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# Helper function to convert PyTorch tensors to Python types for JSON serialization
def convert_tensors_to_python(obj):
    """
    Recursively convert PyTorch tensors to Python lists or scalars.
    
    Args:
        obj: Object that may contain PyTorch tensors
        
    Returns:
        Object with all PyTorch tensors converted to Python types
    """
    if isinstance(obj, torch.Tensor):
        # Convert tensor to CPU and then to list or scalar
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else obj.item()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_python(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    else:
        return obj


def prepare_dataset(
    dataset_name: str
) -> Tuple[
    Union[torch.utils.data.DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    Union[torch.utils.data.DataLoader, Tuple[torch.Tensor, torch.Tensor]],
    int,
    int,
    Optional[pd.DataFrame],  # Added return type for test_df_raw
    Optional[Dict[str, Any]]  # Added return type for metadata
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
            - Raw test dataframe (for preserving original features)
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
        
        return train_loader, test_loader, input_dim, num_classes, None
    
    elif dataset_name in ["titanic", "adult", "heart"]:
        # Load the tabular dataset
        DatasetClass = get_dataset_loader(dataset_name)
        train_df, test_df = DatasetClass.load_data()
        
        # Keep a copy of the original test dataframe before preprocessing
        test_df_raw = test_df.copy()
        
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
        
        # Validate dimensions between train and test
        metadata = {
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
            "dimensions_match": X_train.shape[1] == X_test.shape[1]
        }
        
        # Log warning if dimensions don't match
        if not metadata["dimensions_match"]:
            logger.warning(f"Dimension mismatch in {dataset_name} dataset: ")
            logger.warning(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            logger.warning(f"  This may cause issues during activation collection and analysis")
        
        return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor), input_dim, num_classes, test_df_raw, metadata
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_model(
    dataset_name: str,
    use_regularization: bool = False,
    save_activations: bool = True,
    save_model: bool = True,
    save_fragmentation_metrics: bool = True,
    save_cluster_paths: bool = True,  # New parameter
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
        save_cluster_paths: Whether to save cluster path data
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
        config_id = "regularized" if use_regularization else "baseline"
        experiment_dir = build_experiment_dir(dataset_name, config_id, random_seed)
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
    train_data, test_data, input_dim, num_classes, test_df_raw, dataset_metadata = dataset_data
    
    # Validate dataset dimensions
    if dataset_metadata and not dataset_metadata.get("dimensions_match", True):
        logger.warning(f"Dataset {dataset_name} has dimension mismatch between train and test sets")
        logger.warning(f"Train shape: {dataset_metadata['train_shape']}, Test shape: {dataset_metadata['test_shape']}")
        logger.warning("This might cause issues when analyzing activations later")
        
        # Add indication to experiment name if dimensions don't match
        if experiment_name is None:
            experiment_name += "_dim_mismatch"
    
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
        "angle_fragmentation": [],
        "icpd_fragmentation": [],
        "kstar_fragmentation": [],
        "stability_fragmentation": [],
        "layer_fragmentation": {}
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
            "input": {"train": [], "test": []},
            "layer1": {"train": [], "test": []},
            "layer2": {"train": [], "test": []},
            "layer3": {"train": [], "test": []},
            "output": {"train": [], "test": []},
            "labels": {"train": [], "test": []}
        }
    else:
        layer_activations = None
    
    # Initialize layer fragmentation dict outside epoch loop
    layer_fragmentation: Dict[str, Dict[str, float]] = {}
    
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
            "input": [],
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
                
                # Store input activations before forward pass
                if save_activations:
                    epoch_activations["input"].append(inputs.detach().cpu())
                
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
                        if layer_name != "input":  # Skip input since we stored it before forward pass
                            epoch_activations[layer_name].append(activations.detach().cpu())
                    epoch_labels.append(labels.detach().cpu())
                
        else:
            # Data is in a DataLoader
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_data, desc="Training")):
                # Store input activations before forward pass
                if save_activations:
                    epoch_activations["input"].append(inputs.detach().cpu())
                
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
                        if layer_name != "input":  # Skip input since we stored it before forward pass
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
                    layer_activations[layer_name]["train"].append(torch.cat(epoch_activations[layer_name], dim=0))
            layer_activations["labels"]["train"].append(torch.cat(epoch_labels, dim=0))
            layer_activations["epoch"].append(epoch)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        # Store test activations and labels for this epoch
        test_epoch_activations = {
            "input": [],
            "layer1": [],
            "layer2": [],
            "layer3": [],
            "output": []
        }
        test_epoch_labels = []
        
        with torch.no_grad():
            if isinstance(test_data, tuple):
                # Data is already in tensors
                X_test, y_test = test_data
                
                # Store input activations before forward pass
                if save_activations:
                    test_epoch_activations["input"].append(X_test.detach().cpu())
                
                # Forward pass
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                
                # Update statistics
                test_loss += loss.item() * len(y_test)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == y_test).sum().item()
                test_total += len(y_test)
                
                # Store activations and labels
                if save_activations:
                    activations_dict = model.get_activations()
                    for layer_name, activations in activations_dict.items():
                        if layer_name != "input":  # Skip input since we stored it before forward pass
                            test_epoch_activations[layer_name].append(activations.detach().cpu())
                    test_epoch_labels.append(y_test.detach().cpu())
            else:
                # Data is in a DataLoader
                for inputs, labels in test_data:
                    # Store input activations before forward pass
                    if save_activations:
                        test_epoch_activations["input"].append(inputs.detach().cpu())
                    
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Update statistics
                    test_loss += loss.item() * len(labels)
                    _, predicted = torch.max(outputs.data, 1)
                    test_correct += (predicted == labels).sum().item()
                    test_total += len(labels)
                    
                    # Store activations and labels
                    if save_activations:
                        activations_dict = model.get_activations()
                        for layer_name, activations in activations_dict.items():
                            if layer_name != "input":  # Skip input since we stored it before forward pass
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
        
        # Compute fragmentation metrics on test data ONLY on the final epoch
        if save_fragmentation_metrics and epoch == epochs - 1:
            # Create dictionaries to store metrics per layer
            entropy_scores_per_layer = {}
            angle_scores_per_layer = {}
            icpd_scores_per_layer = {}
            kstar_scores_per_layer = {}
            all_layer_activations = {}
            
            # For cluster paths
            layer_cluster_labels = {}
            
            # Get list of layers to evaluate
            layers_to_evaluate = [name for name in model.activations.keys() if name.startswith("layer")]
            
            # Compute fragmentation for each layer
            model.eval()
            with torch.no_grad():
                # Get test data activations
                if isinstance(test_data, tuple):
                    X_test, y_test = test_data
                    _ = model(X_test)
                    test_labels = y_test
                    
                    for layer_name in layers_to_evaluate:
                        # Get activations for this layer
                        test_activations = model.activations[layer_name]
                        all_layer_activations[layer_name] = test_activations
                        
                        # Compute fragmentation metrics
                        entropy_score = compute_entropy_fragmentation_score(test_activations, test_labels)
                        angle_score = compute_angle_fragmentation_score(test_activations, test_labels)
                        icpd_score = compute_icpd_fragmentation_score(test_activations, test_labels)
                        kstar_score = compute_kstar_fragmentation_score(test_activations, test_labels)
                        
                        # Store in dictionaries
                        entropy_scores_per_layer[layer_name] = entropy_score
                        angle_scores_per_layer[layer_name] = angle_score
                        icpd_scores_per_layer[layer_name] = icpd_score
                        kstar_scores_per_layer[layer_name] = kstar_score
                        
                        # Also update the per-epoch metrics with final layer scores
                        if layer_name == "layer3":
                            training_history["entropy_fragmentation"][-1] = entropy_score
                            training_history["angle_fragmentation"][-1] = angle_score
                            training_history["icpd_fragmentation"][-1] = icpd_score
                            training_history["kstar_fragmentation"][-1] = kstar_score
                        
                        # For cluster paths, store the cluster assignments for each layer
                        if save_cluster_paths:
                            # Get cluster assignments from the entropy computation
                            entropy_result = compute_cluster_entropy(
                                test_activations, test_labels, 
                                return_clusters=True, k_selection='auto'
                            )
                            # Get global assignments
                            global_assignments = np.zeros(len(test_labels), dtype=int)
                            for label, indices in entropy_result.get('cluster_assignments', {}).items():
                                label_indices = np.where(test_labels.cpu().numpy() == label)[0]
                                for i, cluster_idx in enumerate(indices):
                                    if i < len(label_indices):
                                        global_assignments[label_indices[i]] = cluster_idx
                            
                            # Store for cluster paths
                            layer_cluster_labels[layer_name] = global_assignments.tolist()
                    
                    # Compute stability metric
                    stability_score = compute_stability_fragmentation_score(all_layer_activations)
                    stability_scores_per_layer = compute_layer_stability_profile(all_layer_activations)
                    training_history["stability_fragmentation"][-1] = stability_score
                    
                    # Save cluster paths if requested
                    if save_cluster_paths and dataset_name in ["titanic", "adult", "heart"]:
                        # Create cluster paths data
                        ordered_layers = sorted(layer_cluster_labels.keys())
                        
                        # Get final layer activations for finding representative samples
                        final_layer = ordered_layers[-1] if ordered_layers else None
                        final_activations = all_layer_activations.get(final_layer, None)
                        if final_activations is not None:
                            final_activations = final_activations.cpu().numpy()
                        
                        # Prepare and save cluster paths data
                        cluster_paths_data = prepare_cluster_path_data(
                            layer_cluster_labels=layer_cluster_labels,
                            test_df_raw=test_df_raw,
                            y_test=y_test.cpu().numpy(),
                            ordered_layers=ordered_layers,
                            final_layer_activations=final_activations
                        )
                        
                        # Save to file
                        cluster_paths_path = os.path.join(experiment_dir, "cluster_paths.json")
                        save_paths(cluster_paths_data, cluster_paths_path)
                        logger.info(f"Saved cluster paths data to {cluster_paths_path}")
                else:
                    # For DataLoader, we need to process in batches and concatenate
                    all_activations = {layer: [] for layer in layers_to_evaluate}
                    all_labels = []
                    
                    for inputs, labels in test_data:
                        _ = model(inputs)
                        for layer_name in layers_to_evaluate:
                            all_activations[layer_name].append(model.activations[layer_name].detach().cpu())
                        all_labels.append(labels.detach().cpu())
                    
                    # Process each layer
                    for layer_name in layers_to_evaluate:
                        test_activations = torch.cat(all_activations[layer_name], dim=0)
                        test_labels = torch.cat(all_labels, dim=0)
                        all_layer_activations[layer_name] = test_activations
                        
                        # Compute fragmentation metrics
                        entropy_score = compute_entropy_fragmentation_score(test_activations, test_labels)
                        angle_score = compute_angle_fragmentation_score(test_activations, test_labels)
                        icpd_score = compute_icpd_fragmentation_score(test_activations, test_labels)
                        kstar_score = compute_kstar_fragmentation_score(test_activations, test_labels)
                        
                        # Store in dictionaries
                        entropy_scores_per_layer[layer_name] = entropy_score
                        angle_scores_per_layer[layer_name] = angle_score
                        icpd_scores_per_layer[layer_name] = icpd_score
                        kstar_scores_per_layer[layer_name] = kstar_score
                        
                        # Also update the per-epoch metrics with final layer scores
                        if layer_name == "layer3":
                            training_history["entropy_fragmentation"][-1] = entropy_score
                            training_history["angle_fragmentation"][-1] = angle_score
                            training_history["icpd_fragmentation"][-1] = icpd_score
                            training_history["kstar_fragmentation"][-1] = kstar_score
                    
                    # Compute stability metric
                    stability_score = compute_stability_fragmentation_score(all_layer_activations)
                    stability_scores_per_layer = compute_layer_stability_profile(all_layer_activations)
                    training_history["stability_fragmentation"][-1] = stability_score
            
            # Save to layer_fragmentation dictionary
            layer_fragmentation["entropy"] = entropy_scores_per_layer
            layer_fragmentation["angle"] = angle_scores_per_layer
            layer_fragmentation["icpd"] = icpd_scores_per_layer
            layer_fragmentation["kstar"] = kstar_scores_per_layer
            layer_fragmentation["stability"] = stability_scores_per_layer
        else:
            # Skip expensive computation on intermediate epochs
            entropy_score = 0.0
            angle_score = 0.0
            icpd_score = 0.0
            kstar_score = 0.0
            stability_score = 0.0
        
        # Update training history (per-epoch)
        training_history["train_loss"].append(train_loss)
        training_history["train_accuracy"].append(train_accuracy)
        training_history["train_regularization_loss"].append(train_reg_loss)
        training_history["test_loss"].append(test_loss)
        training_history["test_accuracy"].append(test_accuracy)
        training_history["entropy_fragmentation"].append(entropy_score)
        training_history["angle_fragmentation"].append(angle_score)
        training_history["icpd_fragmentation"].append(icpd_score)
        training_history["kstar_fragmentation"].append(kstar_score)
        training_history["stability_fragmentation"].append(stability_score)
        
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
    
    # After training loop finishes, attach per-layer fragmentation metrics (final epoch)
    training_history["layer_fragmentation"] = layer_fragmentation
    
    # If we didn't compute fragmentation in the last epoch (due to early stopping),
    # compute it here on the final model
    if save_fragmentation_metrics and (not layer_fragmentation or not layer_fragmentation.get("entropy")):
        # Create dictionaries to store metrics per layer
        entropy_scores_per_layer = {}
        angle_scores_per_layer = {}
        icpd_scores_per_layer = {}
        kstar_scores_per_layer = {}
        all_layer_activations = {}
        
        # Get list of layers to evaluate
        layers_to_evaluate = [name for name in model.activations.keys() if name.startswith("layer")]
        
        # Compute fragmentation for each layer
        model.eval()
        with torch.no_grad():
            # Get test data activations
            if isinstance(test_data, tuple):
                X_test, y_test = test_data
                _ = model(X_test)
                test_labels = y_test
                
                for layer_name in layers_to_evaluate:
                    # Get activations for this layer
                    test_activations = model.activations[layer_name]
                    all_layer_activations[layer_name] = test_activations
                    
                    # Compute fragmentation metrics
                    entropy_score = compute_entropy_fragmentation_score(test_activations, test_labels)
                    angle_score = compute_angle_fragmentation_score(test_activations, test_labels)
                    icpd_score = compute_icpd_fragmentation_score(test_activations, test_labels)
                    kstar_score = compute_kstar_fragmentation_score(test_activations, test_labels)
                    
                    # Store in dictionaries
                    entropy_scores_per_layer[layer_name] = entropy_score
                    angle_scores_per_layer[layer_name] = angle_score
                    icpd_scores_per_layer[layer_name] = icpd_score
                    kstar_scores_per_layer[layer_name] = kstar_score
                    
                    # Also update the per-epoch metrics with final layer scores
                    if layer_name == "layer3":
                        training_history["entropy_fragmentation"][-1] = entropy_score
                        training_history["angle_fragmentation"][-1] = angle_score
                        training_history["icpd_fragmentation"][-1] = icpd_score
                        training_history["kstar_fragmentation"][-1] = kstar_score
                
                # Compute stability metric
                stability_score = compute_stability_fragmentation_score(all_layer_activations)
                stability_scores_per_layer = compute_layer_stability_profile(all_layer_activations)
                training_history["stability_fragmentation"][-1] = stability_score
            else:
                # For DataLoader, we need to process in batches and concatenate
                all_activations = {layer: [] for layer in layers_to_evaluate}
                all_labels = []
                
                for inputs, labels in test_data:
                    _ = model(inputs)
                    for layer_name in layers_to_evaluate:
                        all_activations[layer_name].append(model.activations[layer_name].detach().cpu())
                    all_labels.append(labels.detach().cpu())
                
                # Process each layer
                for layer_name in layers_to_evaluate:
                    test_activations = torch.cat(all_activations[layer_name], dim=0)
                    test_labels = torch.cat(all_labels, dim=0)
                    all_layer_activations[layer_name] = test_activations
                    
                    # Compute fragmentation metrics
                    entropy_score = compute_entropy_fragmentation_score(test_activations, test_labels)
                    angle_score = compute_angle_fragmentation_score(test_activations, test_labels)
                    icpd_score = compute_icpd_fragmentation_score(test_activations, test_labels)
                    kstar_score = compute_kstar_fragmentation_score(test_activations, test_labels)
                    
                    # Store in dictionaries
                    entropy_scores_per_layer[layer_name] = entropy_score
                    angle_scores_per_layer[layer_name] = angle_score
                    icpd_scores_per_layer[layer_name] = icpd_score
                    kstar_scores_per_layer[layer_name] = kstar_score
                    
                    # Also update the per-epoch metrics with final layer scores
                    if layer_name == "layer3":
                        training_history["entropy_fragmentation"][-1] = entropy_score
                        training_history["angle_fragmentation"][-1] = angle_score
                        training_history["icpd_fragmentation"][-1] = icpd_score
                        training_history["kstar_fragmentation"][-1] = kstar_score
                
                # Compute stability metric
                stability_score = compute_stability_fragmentation_score(all_layer_activations)
                stability_scores_per_layer = compute_layer_stability_profile(all_layer_activations)
                training_history["stability_fragmentation"][-1] = stability_score
            
            # Save to layer_fragmentation dictionary
            layer_fragmentation["entropy"] = entropy_scores_per_layer
            layer_fragmentation["angle"] = angle_scores_per_layer
            layer_fragmentation["icpd"] = icpd_scores_per_layer
            layer_fragmentation["kstar"] = kstar_scores_per_layer
            layer_fragmentation["stability"] = stability_scores_per_layer
        
        logger.info(f"Computed fragmentation metrics on final model")
        logger.info(f"Entropy Score: {training_history['entropy_fragmentation'][-1]:.4f}, "
                   f"Angle Score: {training_history['angle_fragmentation'][-1]:.4f}")
    
    # Save the final model
    if save_model:
        final_model_path = os.path.join(experiment_dir, "final_model.pt")
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(experiment_dir, "training_history.json")
    with open(history_path, "w") as f:
        # Convert tensors to Python types before serialization
        serializable_history = convert_tensors_to_python(training_history)
        json.dump(serializable_history, f, indent=4)
    logger.info(f"Saved training history to {history_path}")
    
    # ----------------- SANITIZE LAYER ACTIVATIONS BEFORE SAVING -----------------
    # Ensure every layer's 'train' and 'test' entries are plain 2-D float32 numpy
    # arrays instead of lists of tensors.  This guarantees downstream loaders
    # (e.g., Dash visualisation) see all layers, including deeper ones like
    # 'layer3'.
    if save_activations and layer_activations:
        for layer_key, splits in layer_activations.items():
            # Expect splits to be a dict with "train"/"test" keys; skip others
            if not isinstance(splits, dict):
                continue
            for split_name in ["train", "test"]:
                if split_name not in splits:
                    continue
                data_obj = splits[split_name]

                # Case 1: list of per-epoch tensors / arrays
                if isinstance(data_obj, list) and len(data_obj) > 0:
                    try:
                        # Stack along epoch dimension then keep last epoch
                        stacked = torch.stack(data_obj, dim=0)  # (epochs, samples, feats)
                        cleaned = stacked[-1].cpu().numpy().astype(np.float32)
                    except Exception:
                        # Fallback: try to use the last element directly
                        last = data_obj[-1]
                        if torch.is_tensor(last):
                            cleaned = last.cpu().numpy().astype(np.float32)
                        else:
                            cleaned = np.asarray(last, dtype=np.float32)
                    splits[split_name] = cleaned

                # Case 2: single tensor
                elif torch.is_tensor(data_obj):
                    splits[split_name] = data_obj.cpu().numpy().astype(np.float32)

                # Case 3: already ndarray – cast to float32 to be safe
                elif isinstance(data_obj, np.ndarray):
                    splits[split_name] = data_obj.astype(np.float32)

            # Convert labels, which live under layer_activations["labels"], too
            if "labels" in layer_activations and isinstance(layer_activations["labels"], dict):
                for split_name in ["train", "test"]:
                    lbl_obj = layer_activations["labels"].get(split_name)
                    if isinstance(lbl_obj, list) and len(lbl_obj) > 0:
                        last = lbl_obj[-1]
                        if torch.is_tensor(last):
                            layer_activations["labels"][split_name] = last.cpu().numpy()
                        else:
                            layer_activations["labels"][split_name] = np.asarray(last)
                    elif torch.is_tensor(lbl_obj):
                        layer_activations["labels"][split_name] = lbl_obj.cpu().numpy()
    # --------------------------------------------------------------------------
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
    
    # Plot original fragmentation metrics
    plt.figure(figsize=(10, 5))
    plt.plot(history["entropy_fragmentation"], label="Entropy Fragmentation")
    plt.plot(history["angle_fragmentation"], label="Angle Fragmentation")
    plt.title("Original Fragmentation Metrics over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Fragmentation Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "original_fragmentation.png"))
    plt.close()
    
    # Plot new fragmentation metrics
    plt.figure(figsize=(10, 5))
    plt.plot(history["icpd_fragmentation"], label="ICPD", color='blue')
    plt.title("Intra-Class Pairwise Distance over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("ICPD Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "icpd_fragmentation.png"))
    plt.close()
    
    # Plot optimal k
    plt.figure(figsize=(10, 5))
    plt.plot(history["kstar_fragmentation"], label="k*", color='green')
    plt.title("Optimal Number of Clusters (k*) over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("k* Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "kstar_fragmentation.png"))
    plt.close()
    
    # Plot representation stability
    plt.figure(figsize=(10, 5))
    plt.plot(history["stability_fragmentation"], label="Δ-Norm", color='red')
    plt.title("Representation Stability (Δ-Norm) over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Δ-Norm")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "stability_fragmentation.png"))
    plt.close()
    
    # Combined new metrics plot
    plt.figure(figsize=(12, 6))
    
    # Primary y-axis for ICPD
    ax1 = plt.gca()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ICPD Score", color='blue')
    ax1.plot(history["icpd_fragmentation"], label="ICPD", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Secondary y-axis for k*
    ax2 = ax1.twinx()
    ax2.set_ylabel("k* Value", color='green')
    ax2.plot(history["kstar_fragmentation"], label="k*", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Third y-axis for stability
    ax3 = ax1.twinx()
    # Offset the third axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel("Δ-Norm", color='red')
    ax3.plot(history["stability_fragmentation"], label="Δ-Norm", color='red')
    ax3.tick_params(axis='y', labelcolor='red')
    
    # Add title
    plt.title("Combined Fragmentation Metrics over epochs")
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_fragmentation.png"))
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
