"""
Configuration module for Concept Fragmentation project.

This module provides a flexible, hierarchical configuration system for managing
all settings related to datasets, models, training, metrics, visualization, and analysis.

It maintains backward compatibility with the legacy configuration system while
providing a more structured, type-safe approach to configuration management.
"""

import os
import logging
from typing import Dict, Any, List

from .config_manager import ConfigManager
from .config_classes import (
    Config, 
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    RegularizationConfig,
    MetricsConfig,
    VisualizationConfig,
    AnalysisConfig,
    ClusterEntropyConfig,
    SubspaceAngleConfig,
    ETSConfig,
    CrossLayerMetricsConfig,
    PlotConfig,
    UMAPConfig,
    PCAConfig,
    TrajectoryConfig,
    ETSVisConfig,
    PathArchetypesConfig,
    TransitionMatrixConfig
)

# Export the ConfigManager instance
config_manager = ConfigManager()

# Try to load legacy config if available
try:
    config_manager.load_legacy_config()
except (ImportError, Exception) as e:
    print(f"Warning: Could not load legacy config: {e}")
    print("Using default configuration.")

# Export the get_config function for easy access
get_config = config_manager.get_config

# Create legacy-style variables for backward compatibility
config = config_manager.get_config()

# General settings
RANDOM_SEED = config.random_seed
RESULTS_DIR = config.results_dir
LOG_LEVEL = config.log_level
DEVICE = config.device

# Component configurations
DATASETS = {name: dataset_config.__dict__ for name, dataset_config in config.datasets.items()}
MODELS = {"feedforward": {"hidden_dims": {}, "dropout": 0.0, "activation": "", "final_activation": None}}

# Fill in hidden_dims for each dataset from the model configs
for model_name, model_config in config.models.items():
    if model_name.startswith("feedforward_"):
        dataset_name = model_name.replace("feedforward_", "")
        MODELS["feedforward"]["hidden_dims"][dataset_name] = model_config.hidden_dims
        MODELS["feedforward"]["dropout"] = model_config.dropout
        MODELS["feedforward"]["activation"] = model_config.activation
        MODELS["feedforward"]["final_activation"] = model_config.final_activation

# Training settings
TRAINING = {
    "batch_size": {},
    "lr": config.training.get(next(iter(config.training), "titanic")).lr,
    "epochs": {},
    "early_stopping": {
        "patience": config.training.get(next(iter(config.training), "titanic")).early_stopping_patience,
        "min_delta": config.training.get(next(iter(config.training), "titanic")).early_stopping_min_delta
    },
    "optimizer": config.training.get(next(iter(config.training), "titanic")).optimizer,
    "weight_decay": config.training.get(next(iter(config.training), "titanic")).weight_decay,
    "clip_grad_norm": config.training.get(next(iter(config.training), "titanic")).clip_grad_norm
}

# Fill in batch_size and epochs for each dataset
for dataset_name, training_config in config.training.items():
    TRAINING["batch_size"][dataset_name] = training_config.batch_size
    TRAINING["epochs"][dataset_name] = training_config.epochs

# Regularization settings
REGULARIZATION = {"cohesion": {}}
if "cohesion_medium" in config.regularization:
    cohesion_config = config.regularization["cohesion_medium"]
    REGULARIZATION["cohesion"] = {
        "weight": cohesion_config.weight,
        "temperature": cohesion_config.temperature,
        "similarity_threshold": cohesion_config.similarity_threshold,
        "layers": cohesion_config.layers,
        "minibatch_size": cohesion_config.minibatch_size
    }

# Metrics settings
METRICS = {
    "cluster_entropy": config.metrics.cluster_entropy.__dict__,
    "subspace_angle": config.metrics.subspace_angle.__dict__,
    "explainable_threshold_similarity": config.metrics.explainable_threshold_similarity.__dict__
}

# Visualization settings
VISUALIZATION = {
    "plot": config.visualization.plot.__dict__,
    "umap": config.visualization.umap.__dict__,
    "pca": config.visualization.pca.__dict__,
    "trajectory": config.visualization.trajectory.__dict__,
    "ets": config.visualization.ets.__dict__
}

# Analysis settings
ANALYSIS = {
    "path_archetypes": config.analysis.path_archetypes.__dict__,
    "transition_matrix": config.analysis.transition_matrix.__dict__
}

# Cohesion grid for experiments
COHESION_GRID = [
    {"weight": 0.0, "temperature": 0.07, "similarity_threshold": 0.0, "layers": []},
    {"weight": 0.1, "temperature": 0.07, "similarity_threshold": 0.0, "layers": ["layer3"]},
    {"weight": 0.1, "temperature": 0.07, "similarity_threshold": 0.3, "layers": ["layer3"]},
    {"weight": 0.1, "temperature": 0.07, "similarity_threshold": 0.3, "layers": ["layer2", "layer3"]},
    {"weight": 0.5, "temperature": 0.07, "similarity_threshold": 0.3, "layers": ["layer2", "layer3"]}
]

# For each configuration object created from the ConfigManager,
# ensure directories exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR, exist_ok=True)

__all__ = [
    # New configuration classes
    'Config',
    'DatasetConfig',
    'ModelConfig',
    'TrainingConfig',
    'RegularizationConfig',
    'MetricsConfig',
    'VisualizationConfig',
    'AnalysisConfig',
    'ClusterEntropyConfig',
    'SubspaceAngleConfig',
    'ETSConfig',
    'CrossLayerMetricsConfig',
    'PlotConfig',
    'UMAPConfig',
    'PCAConfig',
    'TrajectoryConfig',
    'ETSVisConfig',
    'PathArchetypesConfig',
    'TransitionMatrixConfig',
    
    # Configuration manager
    'ConfigManager',
    'config_manager',
    'get_config',
    
    # Legacy compatibility variables
    'RANDOM_SEED',
    'RESULTS_DIR',
    'LOG_LEVEL',
    'DEVICE',
    'DATASETS',
    'MODELS',
    'TRAINING',
    'REGULARIZATION',
    'METRICS',
    'VISUALIZATION',
    'ANALYSIS',
    'COHESION_GRID'
]