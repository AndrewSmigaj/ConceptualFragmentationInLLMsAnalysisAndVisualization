"""
Configuration settings for the Concept Fragmentation project.

This file contains all hyperparameters, paths, and settings used across the project.
"""

import os
import logging
from pathlib import Path

###########################################
# General Settings
###########################################

# Random seed for reproducibility
RANDOM_SEED = 42

# Base directory for results - store on D drive to save space
RESULTS_DIR = "D:/concept_fragmentation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Logging level
LOG_LEVEL = logging.INFO

# Device configuration (will be automatically determined at runtime)
DEVICE = "cuda"  # Will be overridden if CUDA not available

###########################################
# Dataset Settings
###########################################

DATASETS = {
    "titanic": {
        "path": "data/titanic/titanic.csv",
        "test_size": 0.2,
        "val_size": 0.2,
        "categorical_features": [
            "Pclass", "Sex", "SibSp", "Parch", "Embarked"
        ],
        "numerical_features": [
            "Age", "Fare"
        ],
        "target": "Survived",
        "drop_columns": ["PassengerId", "Name", "Ticket", "Cabin"],
        "impute_strategy": {
            "categorical": "most_frequent",
            "numerical": "median"
        }
    },
    
    "adult": {
        "path": "data/adult/adult.csv",
        "test_size": 0.2,
        "val_size": 0.2,
        "categorical_features": [
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        "numerical_features": [
            "age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"
        ],
        "target": "income",
        "drop_columns": [],
        "impute_strategy": {
            "categorical": "most_frequent",
            "numerical": "median"
        }
    },
    
    "heart": {
        "path": "data/heart/heart.csv",
        "test_size": 0.2,
        "val_size": 0.2,
        "categorical_features": [
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ],
        "numerical_features": [
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
        "target": "target",
        "drop_columns": [],
        "impute_strategy": {
            "categorical": "most_frequent",
            "numerical": "median"
        }
    },
    
    "fashion_mnist": {
        "path": None,  # Will be downloaded by torchvision
        "test_size": 0.2,
        "val_size": 0.2,
        "n_samples_per_class": 100,  # Use subset for faster training
        "image_size": 28,
        "num_classes": 10,
        "classes": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
    }
}

###########################################
# Model Settings
###########################################

MODELS = {
    "feedforward": {
        # Default dimensions for each dataset
        "hidden_dims": {
            "titanic": [64, 32, 16],
            "adult": [128, 64, 32],
            "heart": [64, 32, 16],
            "fashion_mnist": [256, 128, 64]
        },
        "dropout": 0.2,
        "activation": "relu",  # Options: relu, tanh, sigmoid, leaky_relu
        "final_activation": None  # None for linear output
    }
}

###########################################
# Training Settings
###########################################

TRAINING = {
    "batch_size": {
        "titanic": 32,
        "adult": 64,
        "heart": 32,
        "fashion_mnist": 128
    },
    "lr": 0.001,
    "epochs": {
        "titanic": 50,
        "adult": 50,
        "heart": 50,
        "fashion_mnist": 20
    },
    "early_stopping": {
        "patience": 10,
        "min_delta": 0.001
    },
    "optimizer": "adam",  # Options: adam, sgd, rmsprop
    "weight_decay": 0.0001,
    "clip_grad_norm": None,  # None to disable gradient clipping
}

###########################################
# Regularization Settings
###########################################

REGULARIZATION = {
    "cohesion": {
        "weight": 0.1,
        "temperature": 0.07,
        "similarity_threshold": 0.0,
        "layers": ["layer3"],
        "minibatch_size": 1024  # Added for regularizers.py
    }
}

###########################################
# Metric Settings
###########################################

METRICS = {
    "cluster_entropy": {
        "k_values": [2, 3, 5, 8],  # Number of clusters to try
        "default_k": 3,
        "n_init": 10,  # Number of k-means runs
        "max_iter": 300,  # Max k-means iterations
        "random_state": RANDOM_SEED,
        "k_selection": "auto"  # Method to select k ('auto' or 'fixed')
    },
    
    "subspace_angle": {
        "var_threshold": 0.9,  # PCA variance threshold (retain ≥ 90% variance)
        "n_components": 10,    # Legacy parameter - kept for backward compatibility
        "bootstrap_samples": 10,
        "confidence_level": 0.95,
        "random_state": RANDOM_SEED
    },
    
    "explainable_threshold_similarity": {
        "threshold_percentile": 0.1,  # Percentile for automatic threshold calculation
        "min_threshold": 1e-5,       # Minimum threshold to avoid numerical issues
        "batch_size": 1000,          # Batch size for processing large datasets
        "verbose": False             # Whether to print progress information
    }
}

###########################################
# Visualization Settings
###########################################

VISUALIZATION = {
    "plot": {
        "figsize": (10, 8),
        "dpi": 150,
        "markersize": 60,
        "alpha": 0.7,
        "cmap": "tab10"
    },
    
    "umap": {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
        "n_components": 2
    },
    
    "pca": {
        "n_components": 2,
        "svd_solver": "auto"
    },
    
    "trajectory": {
        "line_width": 2.0,
        "marker_size": 8,
        "alpha_fg": 0.8,
        "alpha_bg": 0.2
    },
    
    "ets": {
        "dimension_color_threshold": 0.8,  # Color dimensions that are close to threshold
        "similarity_matrix_colormap": "viridis",
        "cluster_colormap": "tab10"
    }
}

###########################################
# Analysis Settings
###########################################

ANALYSIS = {
    "path_archetypes": {
        "top_k": 5,           # Number of top paths to analyze
        "max_members": 50,    # Maximum number of member indices to include
        "min_path_size": 3    # Minimum number of samples in a path to include in analysis
    },
    
    "transition_matrix": {
        "entropy_normalization": True,  # Whether to normalize entropy by log2(n_clusters)
        "max_steps": 3                  # Maximum number of steps for multi-step transitions
    }
}

###########################################
# Experiment Settings
###########################################

# These hyperparameter combinations will be used for the grid search
COHESION_GRID = [
    {"weight": 0.0, "temperature": 0.07, "similarity_threshold": 0.0, "layers": []},  # Baseline
    {"weight": 0.1, "temperature": 0.07, "similarity_threshold": 0.0, "layers": ["layer3"]},
    {"weight": 0.1, "temperature": 0.07, "similarity_threshold": 0.3, "layers": ["layer3"]},
    {"weight": 0.1, "temperature": 0.07, "similarity_threshold": 0.3, "layers": ["layer2", "layer3"]},
    {"weight": 0.5, "temperature": 0.07, "similarity_threshold": 0.3, "layers": ["layer2", "layer3"]}
]