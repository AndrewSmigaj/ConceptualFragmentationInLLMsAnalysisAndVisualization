"""
ConfigManager for the Concept Fragmentation project.

This module provides a singleton ConfigManager class for accessing and
managing configuration across the project.
"""

import os
import json
import yaml
import logging
import re
from typing import Dict, Any, Optional, List, Union
import copy
from pathlib import Path

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
    PlotConfig,
    UMAPConfig,
    PCAConfig,
    TrajectoryConfig,
    ETSVisConfig,
    PathArchetypesConfig,
    TransitionMatrixConfig
)
from .defaults import create_default_config
from .dependencies import check_dependencies


class ConfigManager:
    """Singleton class to manage configuration across the project."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Create a singleton instance of ConfigManager."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = create_default_config()
            cls._instance._experiment_configs = {}
            cls._instance._env_prefix = "CF_"  # Concept Fragmentation environment variable prefix
            cls._instance._logger = logging.getLogger(__name__)
            
            # Check dependencies
            cls._instance._check_dependencies()
            
            # Apply environment variable overrides
            cls._instance._apply_env_overrides()
        return cls._instance
    
    def _check_dependencies(self):
        """Check for required dependencies."""
        dependency_results = check_dependencies()
        
        # Log the results of dependency checks
        for package_name, (is_available, error_message) in dependency_results.items():
            if not is_available:
                self._logger.warning(f"Dependency warning: {error_message}")
                if package_name == "PyYAML":
                    self._logger.warning("YAML support is required for configuration import/export")
                elif package_name == "torch":
                    self._logger.warning("PyTorch not available, will use CPU for computations")
    
    def _apply_env_overrides(self):
        """Apply configuration overrides from environment variables."""
        # Get all environment variables with the specified prefix
        env_overrides = {}
        env_var_pattern = re.compile(f"^{self._env_prefix}(.+)$")
        
        for env_var, value in os.environ.items():
            match = env_var_pattern.match(env_var)
            if match:
                # Convert environment variable name to config key
                key = match.group(1).lower()
                
                # Replace double underscore with dot for nested keys
                key = key.replace("__", ".")
                
                # Parse the value
                if value.lower() == "true":
                    parsed_value = True
                elif value.lower() == "false":
                    parsed_value = False
                else:
                    try:
                        # Try to parse as number
                        if "." in value:
                            parsed_value = float(value)
                        else:
                            parsed_value = int(value)
                    except ValueError:
                        # Keep as string
                        parsed_value = value
                
                env_overrides[key] = parsed_value
                self._logger.debug(f"Found environment override: {key}={parsed_value}")
        
        # Apply the overrides if any were found
        if env_overrides:
            self.override(env_overrides)
            self._logger.info(f"Applied {len(env_overrides)} environment variable overrides")
    
    def get_config(self) -> Config:
        """
        Get the current configuration.
        
        Returns:
            Current Config object
        """
        return self._config
    
    def set_config(self, config: Config) -> None:
        """
        Set a new configuration.
        
        Args:
            config: New Config object to use
        """
        self._config = config
    
    def load_from_file(self, filepath: str) -> Config:
        """
        Load configuration from a file.
        
        Args:
            filepath: Path to the configuration file (JSON or YAML)
            
        Returns:
            The loaded Config object
        """
        self._config = Config.from_file(filepath)
        return self._config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """
        Load configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            The loaded Config object
        """
        self._config = Config.from_dict(config_dict)
        return self._config
    
    def override(self, overrides: Dict[str, Any]) -> Config:
        """
        Override specific configuration values.
        
        Args:
            overrides: Dictionary of key-value pairs to override
            
        Returns:
            The updated Config object
        """
        config_dict = self._config.to_dict()
        
        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys (e.g., "metrics.cluster_entropy.default_k")
                parts = key.split('.')
                d = config_dict
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
            else:
                # Simple key
                config_dict[key] = value
        
        # Create new Config from the updated dictionary
        self._config = Config.from_dict(config_dict)
        return self._config
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        """
        Get configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            DatasetConfig for the specified dataset, or None if not found
        """
        return self._config.datasets.get(dataset_name)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig for the specified model, or None if not found
        """
        return self._config.models.get(model_name)
    
    def get_training_config(self, dataset_name: str) -> Optional[TrainingConfig]:
        """
        Get training configuration for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            TrainingConfig for the specified dataset, or None if not found
        """
        return self._config.training.get(dataset_name)
    
    def get_experiment_config(self, dataset: str, experiment_id: str) -> Config:
        """
        Get configuration for a specific experiment.
        
        Creates a new Config object with specific settings for the given
        dataset and experiment ID.
        
        Args:
            dataset: Dataset name
            experiment_id: Experiment identifier
            
        Returns:
            Config object for the specified experiment
        """
        cache_key = f"{dataset}_{experiment_id}"
        
        # Return cached config if available
        if cache_key in self._experiment_configs:
            return self._experiment_configs[cache_key]
        
        # Create a new Config based on the current one
        exp_config = copy.deepcopy(self._config)
        
        # Update settings based on dataset and experiment
        # For now, just update results_dir
        exp_config.results_dir = os.path.join(
            self._config.results_dir, 
            dataset, 
            experiment_id
        )
        
        # Cache and return the config
        self._experiment_configs[cache_key] = exp_config
        return exp_config
    
    def save_config(self, filepath: str, format: str = 'json') -> None:
        """
        Save current configuration to a file.
        
        Args:
            filepath: Path to save the configuration file
            format: Format to use ('json' or 'yaml')
        """
        if format.lower() == 'json':
            self._config.to_json(filepath)
        elif format.lower() in {'yaml', 'yml'}:
            self._config.to_yaml(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def validate(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if the configuration is valid, False otherwise
        """
        try:
            # Simply recreate the Config object, which will run all validations
            Config.from_dict(self._config.to_dict())
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
    
    def convert_legacy_config(self, legacy_config: Dict[str, Any]) -> Config:
        """
        Convert a legacy configuration dictionary to a Config object.
        
        Args:
            legacy_config: Dictionary with the legacy configuration structure
            
        Returns:
            Config object
        """
        # Create a new Config with default values
        config = create_default_config()
        
        # Map legacy fields to new structure
        if "RANDOM_SEED" in legacy_config:
            config.random_seed = legacy_config["RANDOM_SEED"]
        
        if "RESULTS_DIR" in legacy_config:
            config.results_dir = legacy_config["RESULTS_DIR"]
        
        if "DEVICE" in legacy_config:
            config.device = legacy_config["DEVICE"]
        
        # Map DATASETS
        if "DATASETS" in legacy_config:
            for dataset_name, dataset_config in legacy_config["DATASETS"].items():
                # Create a new DatasetConfig
                config.datasets[dataset_name] = DatasetConfig(
                    path=dataset_config.get("path", ""),
                    test_size=dataset_config.get("test_size", 0.2),
                    val_size=dataset_config.get("val_size", 0.2),
                    categorical_features=dataset_config.get("categorical_features", []),
                    numerical_features=dataset_config.get("numerical_features", []),
                    target=dataset_config.get("target", None),
                    drop_columns=dataset_config.get("drop_columns", []),
                    impute_strategy=dataset_config.get("impute_strategy", {})
                )
        
        # Map MODELS
        if "MODELS" in legacy_config:
            for model_name, model_config in legacy_config["MODELS"].items():
                if model_name == "feedforward":
                    # Handle the nested hidden_dims structure
                    hidden_dims = model_config.get("hidden_dims", {})
                    
                    # Create a new ModelConfig for each dataset
                    for dataset_name, dims in hidden_dims.items():
                        config.models[f"{model_name}_{dataset_name}"] = ModelConfig(
                            hidden_dims=dims,
                            dropout=model_config.get("dropout", 0.2),
                            activation=model_config.get("activation", "relu"),
                            final_activation=model_config.get("final_activation", None)
                        )
        
        # Map TRAINING
        if "TRAINING" in legacy_config:
            training_config = legacy_config["TRAINING"]
            
            # Handle batch_size as a dictionary
            batch_sizes = training_config.get("batch_size", {})
            for dataset_name, batch_size in batch_sizes.items():
                # Get epochs for this dataset
                epochs = training_config.get("epochs", {}).get(dataset_name, 50)
                
                # Create a new TrainingConfig
                config.training[dataset_name] = TrainingConfig(
                    batch_size=batch_size,
                    lr=training_config.get("lr", 0.001),
                    epochs=epochs,
                    early_stopping_patience=training_config.get("early_stopping", {}).get("patience", 10),
                    early_stopping_min_delta=training_config.get("early_stopping", {}).get("min_delta", 0.001),
                    optimizer=training_config.get("optimizer", "adam"),
                    weight_decay=training_config.get("weight_decay", 0.0001),
                    clip_grad_norm=training_config.get("clip_grad_norm", None)
                )
        
        # Map REGULARIZATION
        if "REGULARIZATION" in legacy_config:
            reg_config = legacy_config["REGULARIZATION"]
            
            if "cohesion" in reg_config:
                cohesion_config = reg_config["cohesion"]
                
                config.regularization["cohesion"] = RegularizationConfig(
                    weight=cohesion_config.get("weight", 0.1),
                    temperature=cohesion_config.get("temperature", 0.07),
                    similarity_threshold=cohesion_config.get("similarity_threshold", 0.0),
                    layers=cohesion_config.get("layers", []),
                    minibatch_size=cohesion_config.get("minibatch_size", 1024)
                )
        
        # Map METRICS
        if "METRICS" in legacy_config:
            metrics_config = legacy_config["METRICS"]
            
            # Map cluster_entropy config
            if "cluster_entropy" in metrics_config:
                ce_config = metrics_config["cluster_entropy"]
                config.metrics.cluster_entropy = ClusterEntropyConfig(
                    k_values=ce_config.get("k_values", [2, 3, 5, 8]),
                    default_k=ce_config.get("default_k", 3),
                    n_init=ce_config.get("n_init", 10),
                    max_iter=ce_config.get("max_iter", 300),
                    random_state=ce_config.get("random_state", 42),
                    k_selection=ce_config.get("k_selection", "auto")
                )
            
            # Map subspace_angle config
            if "subspace_angle" in metrics_config:
                sa_config = metrics_config["subspace_angle"]
                config.metrics.subspace_angle = SubspaceAngleConfig(
                    var_threshold=sa_config.get("var_threshold", 0.9),
                    n_components=sa_config.get("n_components", 10),
                    bootstrap_samples=sa_config.get("bootstrap_samples", 10),
                    confidence_level=sa_config.get("confidence_level", 0.95),
                    random_state=sa_config.get("random_state", 42)
                )
            
            # Map ETS config
            if "explainable_threshold_similarity" in metrics_config:
                ets_config = metrics_config["explainable_threshold_similarity"]
                config.metrics.explainable_threshold_similarity = ETSConfig(
                    threshold_percentile=ets_config.get("threshold_percentile", 0.1),
                    min_threshold=ets_config.get("min_threshold", 1e-5),
                    batch_size=ets_config.get("batch_size", 1000),
                    verbose=ets_config.get("verbose", False)
                )
        
        # Map VISUALIZATION
        if "VISUALIZATION" in legacy_config:
            vis_config = legacy_config["VISUALIZATION"]
            
            # Map plot config
            if "plot" in vis_config:
                plot_config = vis_config["plot"]
                config.visualization.plot = PlotConfig(
                    figsize=plot_config.get("figsize", (10, 8)),
                    dpi=plot_config.get("dpi", 150),
                    markersize=plot_config.get("markersize", 60),
                    alpha=plot_config.get("alpha", 0.7),
                    cmap=plot_config.get("cmap", "tab10")
                )
            
            # Map UMAP config
            if "umap" in vis_config:
                umap_config = vis_config["umap"]
                config.visualization.umap = UMAPConfig(
                    n_neighbors=umap_config.get("n_neighbors", 15),
                    min_dist=umap_config.get("min_dist", 0.1),
                    metric=umap_config.get("metric", "euclidean"),
                    n_components=umap_config.get("n_components", 2)
                )
            
            # Map PCA config
            if "pca" in vis_config:
                pca_config = vis_config["pca"]
                config.visualization.pca = PCAConfig(
                    n_components=pca_config.get("n_components", 2),
                    svd_solver=pca_config.get("svd_solver", "auto")
                )
            
            # Map trajectory config
            if "trajectory" in vis_config:
                traj_config = vis_config["trajectory"]
                config.visualization.trajectory = TrajectoryConfig(
                    line_width=traj_config.get("line_width", 2.0),
                    marker_size=traj_config.get("marker_size", 8),
                    alpha_fg=traj_config.get("alpha_fg", 0.8),
                    alpha_bg=traj_config.get("alpha_bg", 0.2)
                )
            
            # Map ETS visualization config
            if "ets" in vis_config:
                ets_vis_config = vis_config["ets"]
                config.visualization.ets = ETSVisConfig(
                    dimension_color_threshold=ets_vis_config.get("dimension_color_threshold", 0.8),
                    similarity_matrix_colormap=ets_vis_config.get("similarity_matrix_colormap", "viridis"),
                    cluster_colormap=ets_vis_config.get("cluster_colormap", "tab10")
                )
        
        # Map ANALYSIS
        if "ANALYSIS" in legacy_config:
            analysis_config = legacy_config["ANALYSIS"]
            
            # Map path_archetypes config
            if "path_archetypes" in analysis_config:
                pa_config = analysis_config["path_archetypes"]
                config.analysis.path_archetypes = PathArchetypesConfig(
                    top_k=pa_config.get("top_k", 5),
                    max_members=pa_config.get("max_members", 50),
                    min_path_size=pa_config.get("min_path_size", 3)
                )
            
            # Map transition_matrix config
            if "transition_matrix" in analysis_config:
                tm_config = analysis_config["transition_matrix"]
                config.analysis.transition_matrix = TransitionMatrixConfig(
                    entropy_normalization=tm_config.get("entropy_normalization", True),
                    max_steps=tm_config.get("max_steps", 3)
                )
        
        return config
    
    def load_legacy_config(self) -> Config:
        """
        Load the legacy configuration from the concept_fragmentation/config.py file.
        
        This method attempts to import the legacy configuration from the old config.py file.
        If successful, it converts it to the new Config structure. If not, it falls back
        to the default configuration.
        
        Returns:
            Config object created from the legacy configuration
        """
        try:
            # Import the legacy configuration
            from importlib import import_module
            
            # Try to import the legacy config module
            config_module = import_module('concept_fragmentation.config')
            
            # Extract all uppercase variables (these are the config constants)
            legacy_config = {}
            for key in dir(config_module):
                if key.isupper():
                    legacy_config[key] = getattr(config_module, key)
            
            # Also check for the root config.py file
            try:
                root_config = import_module('config')
                for key in dir(root_config):
                    if key.isupper() and key not in legacy_config:
                        legacy_config[key] = getattr(root_config, key)
            except (ImportError, ModuleNotFoundError):
                # If root config doesn't exist, just continue
                pass
            
            # Convert to new Config object
            self._config = self.convert_legacy_config(legacy_config)
            
            # Refresh the experiment configs cache since we changed the base config
            self._experiment_configs = {}
            
            return self._config
            
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Legacy configuration could not be loaded: {e}")
            print("Using default configuration.")
            self._config = create_default_config()
            return self._config
        except Exception as e:
            print(f"Error loading legacy configuration: {e}")
            print("Using default configuration.")
            self._config = create_default_config()
            return self._config