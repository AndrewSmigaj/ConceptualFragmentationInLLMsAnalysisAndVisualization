"""
Utilities and helper functions for the Concept Fragmentation project.

This module provides utility functions for:
- Setting random seeds for reproducibility
- Logging setup
- Command-line interface wrappers
- Common file I/O operations
"""

import os
import random
import numpy as np
import torch
import logging
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import argparse
import datetime
import sys
import matplotlib.pyplot as plt
from contextlib import contextmanager
import time
from pathlib import Path

# Import project configuration
from ..config import RANDOM_SEED, RESULTS_DIR, LOG_LEVEL

logger = logging.getLogger(__name__)

def set_global_seed(seed: int) -> None:
    """
    Set random seed for all libraries to ensure reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global random seed set to {seed}")

def set_random_seed(seed: int = RANDOM_SEED, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic: If True, set PyTorch to be deterministic
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(
    log_dir: str = RESULTS_DIR,
    log_name: str = "concept_fragmentation",
    console_level: int = LOG_LEVEL,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up logging to file and console.
    
    Args:
        log_dir: Directory to save log files
        log_name: Base name for log files
        console_level: Logging level for console output
        file_level: Logging level for file output
        
    Returns:
        Configured logger object
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)  # Capture all, filter at handler level
    
    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    return logger


def save_json(data: Dict, filepath: str, indent: int = 2) -> None:
    """
    Save dictionary data to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
        indent: Indentation level for pretty printing
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy/torch types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        else:
            return obj
    
    # Write JSON with conversion
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=convert_to_serializable)

    logger.info(f"Data saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Load dictionary data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save the pickle file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_experiment_dir(
    experiment_name: str,
    base_dir: str = RESULTS_DIR,
    create_subdirs: bool = True
) -> str:
    """
    Create a directory for experiment results.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for all experiments
        create_subdirs: Whether to create standard subdirectories
        
    Returns:
        Path to the created experiment directory
    """
    # Create timestamp-based experiment name if none provided
    if not experiment_name:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Create experiment directory
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create standard subdirectories if requested
    if create_subdirs:
        subdirs = ['checkpoints', 'visualizations', 'metrics', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir


def get_cli_parser(description: str = "Concept Fragmentation Experiment") -> argparse.ArgumentParser:
    """
    Create a standard command-line argument parser for experiment scripts.
    
    Args:
        description: Description of the script
        
    Returns:
        ArgumentParser object with standard arguments
    """
    parser = argparse.ArgumentParser(description=description)
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='titanic',
                       choices=['titanic', 'adult', 'heart', 'fashion_mnist', 'xor'],
                       help='Dataset to use for experiment')
    
    # Model configuration
    parser.add_argument('--hidden_dims', type=str, default='64,32,16',
                       help='Comma-separated list of hidden layer dimensions')
    
    # Regularization
    parser.add_argument('--regularize', action='store_true',
                       help='Apply cohesion regularization')
    parser.add_argument('--reg_weight', type=float, default=0.1,
                       help='Weight for cohesion regularization')
    parser.add_argument('--reg_temp', type=float, default=0.07,
                       help='Temperature for cohesion regularization')
    parser.add_argument('--reg_threshold', type=float, default=0.0,
                       help='Similarity threshold for cohesion regularization')
    parser.add_argument('--reg_layers', type=str, default='layer3',
                       help='Comma-separated list of layers to apply regularization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                       help='Random seed for reproducibility')
    
    # Output options
    parser.add_argument('--experiment_name', type=str, default='',
                       help='Name for the experiment (default: auto-generated)')
    parser.add_argument('--output_dir', type=str, default=RESULTS_DIR,
                       help='Directory to save results')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save model checkpoints')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for training (cuda or cpu)')
    
    return parser


@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None) -> None:
    """
    Context manager for timing code execution.
    
    Args:
        description: Description of the operation being timed
        logger: Logger to use for output (None for print)
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    
    message = f"{description} completed in {elapsed:.2f} seconds"
    if logger:
        logger.info(message)
    else:
        print(message)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_checkpoint_name(
    experiment_name: str,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    extension: str = '.pt'
) -> str:
    """
    Create a standardized checkpoint filename.
    
    Args:
        experiment_name: Name of the experiment
        epoch: Current epoch number
        metrics: Dictionary of metrics to include in filename
        extension: File extension
        
    Returns:
        Checkpoint filename
    """
    if metrics and 'val_acc' in metrics:
        val_acc = metrics['val_acc']
        return f"{experiment_name}_epoch{epoch:03d}_val_acc{val_acc:.4f}{extension}"
    else:
        return f"{experiment_name}_epoch{epoch:03d}{extension}"


def find_all_checkpoints(experiment_dir: str, pattern: str = "*.pt") -> List[str]:
    """
    Find all checkpoint files in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        pattern: File pattern to match
        
    Returns:
        List of checkpoint file paths
    """
    # Check in main directory and checkpoints subdirectory
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    
    checkpoints = []
    if os.path.exists(experiment_dir):
        checkpoints.extend(list(Path(experiment_dir).glob(pattern)))
    if os.path.exists(checkpoint_dir):
        checkpoints.extend(list(Path(checkpoint_dir).glob(pattern)))
    
    return sorted([str(p) for p in checkpoints])


def find_best_checkpoint(
    experiment_dir: str,
    metric: str = "val_acc",
    higher_better: bool = True
) -> Optional[str]:
    """
    Find the best checkpoint based on a metric in the filename.
    
    Args:
        experiment_dir: Path to experiment directory
        metric: Metric to use for selection
        higher_better: Whether higher values are better
        
    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoints = find_all_checkpoints(experiment_dir)
    
    best_score = float('-inf') if higher_better else float('inf')
    best_checkpoint = None
    
    for checkpoint in checkpoints:
        filename = os.path.basename(checkpoint)
        
        # Extract metric value from filename
        if metric in filename:
            try:
                # Find the metric name in the filename and extract the value after it
                parts = filename.split(metric)
                if len(parts) >= 2:
                    # Extract a number from the second part
                    import re
                    match = re.search(r'(\d+\.\d+)', parts[1])
                    if match:
                        score = float(match.group(1))
                        if (higher_better and score > best_score) or (not higher_better and score < best_score):
                            best_score = score
                            best_checkpoint = checkpoint
            except (ValueError, IndexError):
                continue
    
    return best_checkpoint


def ensure_dir(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path to the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)


def get_latest_checkpoint(experiment_dir):
    """
    Get the path to the latest checkpoint in an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Path to the latest checkpoint, or None if no checkpoint exists
    """
    checkpoint_path = os.path.join(experiment_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    return None


def generate_experiment_summary(experiment_dir):
    """
    Generate a summary of an experiment from its log files.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        Dictionary with experiment summary data
    """
    history_path = os.path.join(experiment_dir, "training_history.json")
    if not os.path.exists(history_path):
        logger.warning(f"No training history found at {history_path}")
        return None
    
    history = load_json(history_path)
    
    # Extract final metrics
    final_metrics = {
        "test_accuracy": history["test_accuracy"][-1],
        "train_accuracy": history["train_accuracy"][-1],
        "test_loss": history["test_loss"][-1],
        "train_loss": history["train_loss"][-1],
        "entropy_fragmentation": history["entropy_fragmentation"][-1],
        "angle_fragmentation": history["angle_fragmentation"][-1],
        "epochs_trained": len(history["test_accuracy"]),
        "best_test_accuracy": max(history["test_accuracy"]),
        "best_epoch": history["test_accuracy"].index(max(history["test_accuracy"])) + 1
    }
    
    return final_metrics
