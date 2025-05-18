"""
Default configurations for the Concept Fragmentation project.

This module provides functions to create default configurations for different
components of the project.
"""

import logging
from .config_classes import (
    Config,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    RegularizationConfig,
    ClusterEntropyConfig,
    SubspaceAngleConfig,
    ETSConfig,
    CrossLayerMetricsConfig,
    MetricsConfig,
    PlotConfig,
    UMAPConfig,
    PCAConfig,
    TrajectoryConfig,
    ETSVisConfig,
    VisualizationConfig,
    PathArchetypesConfig,
    TransitionMatrixConfig,
    AnalysisConfig
)


def create_default_titanic_config() -> DatasetConfig:
    """Create default configuration for Titanic dataset."""
    return DatasetConfig(
        path="data/titanic/titanic.csv",
        test_size=0.2,
        val_size=0.2,
        categorical_features=[
            "Pclass", "Sex", "SibSp", "Parch", "Embarked"
        ],
        numerical_features=[
            "Age", "Fare"
        ],
        target="Survived",
        drop_columns=["PassengerId", "Name", "Ticket", "Cabin"],
        impute_strategy={
            "categorical": "most_frequent",
            "numerical": "median"
        }
    )


def create_default_adult_config() -> DatasetConfig:
    """Create default configuration for Adult dataset."""
    return DatasetConfig(
        path="data/adult/adult.csv",
        test_size=0.2,
        val_size=0.2,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        numerical_features=[
            "age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"
        ],
        target="income",
        drop_columns=[],
        impute_strategy={
            "categorical": "most_frequent",
            "numerical": "median"
        }
    )


def create_default_heart_config() -> DatasetConfig:
    """Create default configuration for Heart Disease dataset."""
    return DatasetConfig(
        path="data/heart/heart.csv",
        test_size=0.2,
        val_size=0.2,
        categorical_features=[
            "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ],
        numerical_features=[
            "age", "trestbps", "chol", "thalach", "oldpeak"
        ],
        target="target",
        drop_columns=[],
        impute_strategy={
            "categorical": "most_frequent",
            "numerical": "median"
        }
    )


def create_default_fashion_mnist_config() -> DatasetConfig:
    """Create default configuration for Fashion MNIST dataset."""
    return DatasetConfig(
        path="",  # Will be downloaded by torchvision
        test_size=0.2,
        val_size=0.2,
        categorical_features=[],
        numerical_features=[],
        target="label",
        drop_columns=[],
        impute_strategy={}
    )


def create_default_model_configs() -> dict:
    """Create default model configurations."""
    return {
        "feedforward_titanic": ModelConfig(
            hidden_dims=[64, 32, 16],
            dropout=0.2,
            activation="relu",
            final_activation=None
        ),
        "feedforward_adult": ModelConfig(
            hidden_dims=[128, 64, 32],
            dropout=0.2,
            activation="relu",
            final_activation=None
        ),
        "feedforward_heart": ModelConfig(
            hidden_dims=[64, 32, 16],
            dropout=0.2,
            activation="relu",
            final_activation=None
        ),
        "feedforward_fashion_mnist": ModelConfig(
            hidden_dims=[256, 128, 64],
            dropout=0.2,
            activation="relu",
            final_activation=None
        )
    }


def create_default_training_configs() -> dict:
    """Create default training configurations."""
    return {
        "titanic": TrainingConfig(
            batch_size=32,
            lr=0.001,
            epochs=50,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            optimizer="adam",
            weight_decay=0.0001,
            clip_grad_norm=None
        ),
        "adult": TrainingConfig(
            batch_size=64,
            lr=0.001,
            epochs=50,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            optimizer="adam",
            weight_decay=0.0001,
            clip_grad_norm=None
        ),
        "heart": TrainingConfig(
            batch_size=32,
            lr=0.001,
            epochs=50,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            optimizer="adam",
            weight_decay=0.0001,
            clip_grad_norm=None
        ),
        "fashion_mnist": TrainingConfig(
            batch_size=128,
            lr=0.001,
            epochs=20,
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
            optimizer="adam",
            weight_decay=0.0001,
            clip_grad_norm=None
        )
    }


def create_default_regularization_configs() -> dict:
    """Create default regularization configurations."""
    return {
        "none": RegularizationConfig(
            weight=0.0,
            temperature=0.07,
            similarity_threshold=0.0,
            layers=[],
            minibatch_size=1024
        ),
        "cohesion_light": RegularizationConfig(
            weight=0.1,
            temperature=0.07,
            similarity_threshold=0.0,
            layers=["layer3"],
            minibatch_size=1024
        ),
        "cohesion_medium": RegularizationConfig(
            weight=0.1,
            temperature=0.07,
            similarity_threshold=0.3,
            layers=["layer3"],
            minibatch_size=1024
        ),
        "cohesion_full": RegularizationConfig(
            weight=0.1,
            temperature=0.07,
            similarity_threshold=0.3,
            layers=["layer2", "layer3"],
            minibatch_size=1024
        ),
        "cohesion_strong": RegularizationConfig(
            weight=0.5,
            temperature=0.07,
            similarity_threshold=0.3,
            layers=["layer2", "layer3"],
            minibatch_size=1024
        )
    }


def create_default_metrics_config() -> MetricsConfig:
    """Create default metrics configuration."""
    return MetricsConfig(
        cluster_entropy=ClusterEntropyConfig(
            k_values=[2, 3, 5, 8],
            default_k=3,
            n_init=10,
            max_iter=300,
            random_state=42,
            k_selection="auto"
        ),
        subspace_angle=SubspaceAngleConfig(
            var_threshold=0.9,
            n_components=10,
            bootstrap_samples=10,
            confidence_level=0.95,
            random_state=42
        ),
        explainable_threshold_similarity=ETSConfig(
            threshold_percentile=0.1,
            min_threshold=1e-5,
            batch_size=1000,
            verbose=False
        ),
        cross_layer_metrics=CrossLayerMetricsConfig(
            similarity_metric="cosine",
            min_overlap=0.1,
            batch_size=1000,
            projection_method="pca",
            projection_dims=10
        )
    )


def create_default_visualization_config() -> VisualizationConfig:
    """Create default visualization configuration."""
    return VisualizationConfig(
        plot=PlotConfig(
            figsize=(10, 8),
            dpi=150,
            markersize=60,
            alpha=0.7,
            cmap="tab10"
        ),
        umap=UMAPConfig(
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            n_components=2
        ),
        pca=PCAConfig(
            n_components=2,
            svd_solver="auto"
        ),
        trajectory=TrajectoryConfig(
            line_width=2.0,
            marker_size=8,
            alpha_fg=0.8,
            alpha_bg=0.2
        ),
        ets=ETSVisConfig(
            dimension_color_threshold=0.8,
            similarity_matrix_colormap="viridis",
            cluster_colormap="tab10"
        )
    )


def create_default_analysis_config() -> AnalysisConfig:
    """Create default analysis configuration."""
    return AnalysisConfig(
        path_archetypes=PathArchetypesConfig(
            top_k=5,
            max_members=50,
            min_path_size=3
        ),
        transition_matrix=TransitionMatrixConfig(
            entropy_normalization=True,
            max_steps=3
        )
    )


def create_default_config() -> Config:
    """
    Create a default configuration with standard settings.
    
    Returns:
        Config object with default settings
    """
    # Create dataset configurations
    datasets = {
        "titanic": create_default_titanic_config(),
        "adult": create_default_adult_config(),
        "heart": create_default_heart_config(),
        "fashion_mnist": create_default_fashion_mnist_config()
    }
    
    # Create the Config object
    return Config(
        random_seed=42,
        results_dir="D:/concept_fragmentation_results",
        device="cuda",
        log_level=logging.INFO,
        datasets=datasets,
        models=create_default_model_configs(),
        training=create_default_training_configs(),
        regularization=create_default_regularization_configs(),
        metrics=create_default_metrics_config(),
        visualization=create_default_visualization_config(),
        analysis=create_default_analysis_config()
    )