"""
Configuration classes for the Concept Fragmentation project.

This module defines dataclasses that represent the configuration structure
for different aspects of the project including datasets, models, training,
regularization, metrics, visualization, and analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Set
import os
import json
import yaml
import logging
from pathlib import Path
import copy


@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    path: str
    test_size: float = 0.2
    val_size: float = 0.2
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    target: Optional[str] = None
    drop_columns: List[str] = field(default_factory=list)
    impute_strategy: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and default impute strategy if not provided."""
        if not self.impute_strategy:
            self.impute_strategy = {
                "categorical": "most_frequent",
                "numerical": "median"
            }


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    hidden_dims: List[int]
    dropout: float = 0.2
    activation: str = "relu"
    final_activation: Optional[str] = None
    
    def __post_init__(self):
        """Validate the model configuration."""
        if not isinstance(self.hidden_dims, list):
            raise ValueError("hidden_dims must be a list of integers")
        
        valid_activations = {"relu", "tanh", "sigmoid", "leaky_relu", None}
        if self.activation not in valid_activations:
            raise ValueError(f"activation must be one of {valid_activations}")
        
        valid_final_activations = {"sigmoid", "softmax", "tanh", None}
        if self.final_activation not in valid_final_activations:
            raise ValueError(f"final_activation must be one of {valid_final_activations}")


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int
    lr: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0001
    clip_grad_norm: Optional[float] = None
    
    def __post_init__(self):
        """Validate the training configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        
        valid_optimizers = {"adam", "sgd", "rmsprop"}
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}")


@dataclass
class RegularizationConfig:
    """Configuration for regularization techniques."""
    weight: float = 0.0
    temperature: float = 0.07
    similarity_threshold: float = 0.0
    layers: List[str] = field(default_factory=list)
    minibatch_size: int = 1024
    
    def __post_init__(self):
        """Validate the regularization configuration."""
        if self.weight < 0:
            raise ValueError("weight must be non-negative")
        
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass
class ClusterEntropyConfig:
    """Configuration for cluster entropy metrics."""
    k_values: List[int] = field(default_factory=lambda: [2, 3, 5, 8])
    default_k: int = 3
    n_init: int = 10
    max_iter: int = 300
    random_state: int = 42
    k_selection: str = "auto"


@dataclass
class SubspaceAngleConfig:
    """Configuration for subspace angle metrics."""
    var_threshold: float = 0.9
    n_components: int = 10
    bootstrap_samples: int = 10
    confidence_level: float = 0.95
    random_state: int = 42


@dataclass
class ETSConfig:
    """Configuration for explainable threshold similarity metrics."""
    threshold_percentile: float = 0.1
    min_threshold: float = 1e-5
    batch_size: int = 1000
    verbose: bool = False


@dataclass
class CrossLayerMetricsConfig:
    """Configuration for cross-layer metrics."""
    similarity_metric: str = "cosine"
    min_overlap: float = 0.1
    batch_size: int = 1000
    projection_method: str = "pca"
    projection_dims: int = 10


@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    cluster_entropy: ClusterEntropyConfig = field(default_factory=ClusterEntropyConfig)
    subspace_angle: SubspaceAngleConfig = field(default_factory=SubspaceAngleConfig)
    explainable_threshold_similarity: ETSConfig = field(default_factory=ETSConfig)
    cross_layer_metrics: CrossLayerMetricsConfig = field(default_factory=CrossLayerMetricsConfig)


@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    figsize: tuple = field(default_factory=lambda: (10, 8))
    dpi: int = 150
    markersize: int = 60
    alpha: float = 0.7
    cmap: str = "tab10"


@dataclass
class UMAPConfig:
    """Configuration for UMAP dimensionality reduction."""
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = 2


@dataclass
class PCAConfig:
    """Configuration for PCA dimensionality reduction."""
    n_components: int = 2
    svd_solver: str = "auto"


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory visualization."""
    line_width: float = 2.0
    marker_size: int = 8
    alpha_fg: float = 0.8
    alpha_bg: float = 0.2


@dataclass
class ETSVisConfig:
    """Configuration for ETS visualization."""
    dimension_color_threshold: float = 0.8
    similarity_matrix_colormap: str = "viridis"
    cluster_colormap: str = "tab10"


@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    plot: PlotConfig = field(default_factory=PlotConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    ets: ETSVisConfig = field(default_factory=ETSVisConfig)


@dataclass
class PathArchetypesConfig:
    """Configuration for path archetypes analysis."""
    top_k: int = 5
    max_members: int = 50
    min_path_size: int = 3


@dataclass
class TransitionMatrixConfig:
    """Configuration for transition matrix analysis."""
    entropy_normalization: bool = True
    max_steps: int = 3


@dataclass
class AnalysisConfig:
    """Configuration for analysis components."""
    path_archetypes: PathArchetypesConfig = field(default_factory=PathArchetypesConfig)
    transition_matrix: TransitionMatrixConfig = field(default_factory=TransitionMatrixConfig)


@dataclass
class Config:
    """Root configuration class for the Concept Fragmentation project."""
    # General settings
    random_seed: int = 42
    results_dir: str = field(default_factory=lambda: os.environ.get("RESULTS_DIR", "D:/concept_fragmentation_results"))
    device: str = "cuda"
    log_level: int = logging.INFO
    
    # Component configurations - use empty dictionaries as defaults
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: Dict[str, TrainingConfig] = field(default_factory=dict)
    regularization: Dict[str, RegularizationConfig] = field(default_factory=dict)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        self._initialize_device()
        self._normalize_paths()
    
    def _initialize_device(self):
        """Initialize the device based on availability."""
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
                    print("CUDA not available, using CPU instead.")
            except ImportError:
                self.device = "cpu"
                print("PyTorch not installed, using CPU instead.")
    
    def _normalize_paths(self):
        """Normalize file paths for cross-platform compatibility."""
        # Convert results_dir to platform-specific path
        if self.results_dir:
            self.results_dir = str(Path(self.results_dir))
        
        # Normalize dataset paths
        for dataset_name, dataset_config in self.datasets.items():
            if dataset_config.path:
                dataset_config.path = str(Path(dataset_config.path))
        
        # Ensure results directory exists
        if self.results_dir and not os.path.exists(self.results_dir):
            try:
                os.makedirs(self.results_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                # Don't fail initialization if we can't create the directory,
                # just log a warning
                print(f"Warning: Could not create results directory {self.results_dir}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Convert configuration to JSON and optionally save to file.
        
        Args:
            filepath: Optional path to save the JSON file
            
        Returns:
            JSON string representation of the configuration
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """
        Convert configuration to YAML and optionally save to file.
        
        Args:
            filepath: Optional path to save the YAML file
            
        Returns:
            YAML string representation of the configuration
        """
        yaml_str = yaml.dump(self.to_dict(), default_flow_style=False)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Config object
        """
        # Create a deep copy to avoid modifying the input
        config_dict = copy.deepcopy(config_dict)
        
        # Process nested configurations
        if "datasets" in config_dict:
            for dataset_name, dataset_config in config_dict["datasets"].items():
                config_dict["datasets"][dataset_name] = DatasetConfig(**dataset_config)
        
        if "models" in config_dict:
            for model_name, model_config in config_dict["models"].items():
                config_dict["models"][model_name] = ModelConfig(**model_config)
        
        if "training" in config_dict:
            for dataset_name, training_config in config_dict["training"].items():
                config_dict["training"][dataset_name] = TrainingConfig(**training_config)
        
        if "regularization" in config_dict:
            for reg_name, reg_config in config_dict["regularization"].items():
                config_dict["regularization"][reg_name] = RegularizationConfig(**reg_config)
        
        # Process metrics, visualization, and analysis
        if "metrics" in config_dict:
            metrics_config = config_dict["metrics"]
            # Process nested configs within metrics
            if "cluster_entropy" in metrics_config:
                metrics_config["cluster_entropy"] = ClusterEntropyConfig(**metrics_config["cluster_entropy"])
            if "subspace_angle" in metrics_config:
                metrics_config["subspace_angle"] = SubspaceAngleConfig(**metrics_config["subspace_angle"])
            if "explainable_threshold_similarity" in metrics_config:
                metrics_config["explainable_threshold_similarity"] = ETSConfig(**metrics_config["explainable_threshold_similarity"])
            if "cross_layer_metrics" in metrics_config:
                metrics_config["cross_layer_metrics"] = CrossLayerMetricsConfig(**metrics_config["cross_layer_metrics"])
            config_dict["metrics"] = MetricsConfig(**metrics_config)
        
        if "visualization" in config_dict:
            vis_config = config_dict["visualization"]
            # Process nested configs within visualization
            if "plot" in vis_config:
                vis_config["plot"] = PlotConfig(**vis_config["plot"])
            if "umap" in vis_config:
                vis_config["umap"] = UMAPConfig(**vis_config["umap"])
            if "pca" in vis_config:
                vis_config["pca"] = PCAConfig(**vis_config["pca"])
            if "trajectory" in vis_config:
                vis_config["trajectory"] = TrajectoryConfig(**vis_config["trajectory"])
            if "ets" in vis_config:
                vis_config["ets"] = ETSVisConfig(**vis_config["ets"])
            config_dict["visualization"] = VisualizationConfig(**vis_config)
        
        if "analysis" in config_dict:
            analysis_config = config_dict["analysis"]
            # Process nested configs within analysis
            if "path_archetypes" in analysis_config:
                analysis_config["path_archetypes"] = PathArchetypesConfig(**analysis_config["path_archetypes"])
            if "transition_matrix" in analysis_config:
                analysis_config["transition_matrix"] = TransitionMatrixConfig(**analysis_config["transition_matrix"])
            config_dict["analysis"] = AnalysisConfig(**analysis_config)
        
        # Create the Config object
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Config':
        """
        Create configuration from a JSON string.
        
        Args:
            json_str: JSON string containing configuration
            
        Returns:
            Config object
        """
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Config':
        """
        Create configuration from a YAML string.
        
        Args:
            yaml_str: YAML string containing configuration
            
        Returns:
            Config object
        """
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """
        Load configuration from a file (JSON or YAML).
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            Config object
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Determine file type based on extension
        if filepath.suffix.lower() in {'.yaml', '.yml'}:
            return cls.from_yaml(content)
        elif filepath.suffix.lower() == '.json':
            return cls.from_json(content)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    
    def merge(self, other: 'Config') -> 'Config':
        """
        Merge another configuration into this one.
        
        Args:
            other: Another Config object to merge with
            
        Returns:
            A new Config object with merged values
        """
        # Convert both configs to dictionaries
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        
        # Define a recursive merge function
        def _merge_dicts(d1, d2):
            """Recursively merge d2 into d1."""
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    _merge_dicts(d1[k], v)
                else:
                    d1[k] = v
            return d1
        
        # Merge the dictionaries
        merged_dict = _merge_dicts(copy.deepcopy(self_dict), other_dict)
        
        # Create a new Config from the merged dictionary
        return Config.from_dict(merged_dict)
    
    def update(self, **kwargs) -> 'Config':
        """
        Update specific configuration values.
        
        Args:
            **kwargs: Key-value pairs to update
            
        Returns:
            A new Config object with updated values
        """
        # Create a dictionary from the current config
        config_dict = self.to_dict()
        
        # Update with the provided values
        for key, value in kwargs.items():
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
        
        # Create a new Config from the updated dictionary
        return Config.from_dict(config_dict)