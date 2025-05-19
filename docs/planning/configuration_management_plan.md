# Configuration Management Plan

## 1. Overview

This document outlines the plan for implementing a comprehensive configuration management system for the Concept Fragmentation project. The goal is to create a unified, flexible, and maintainable configuration framework that supports all components of the project while maintaining backward compatibility.

## 2. Current State Analysis

### 2.1 Existing Configuration

The project currently has two main configuration files:
1. `/config.py` - A simple file with a single path variable
2. `/concept_fragmentation/config.py` - A comprehensive configuration with nested dictionaries for various components

The main issues with the current configuration approach are:
- Duplication between the two files
- No clear hierarchy or inheritance
- Limited runtime configuration modification
- Lack of environment-specific configuration
- No validation for configuration values
- Limited support for user-provided configuration overrides

### 2.2 Configuration Requirements

Based on the project structure and the foundations paper, we need a configuration system that supports:

1. **Hierarchical configuration** - Support for nested settings with inheritance
2. **Environment-specific settings** - Development, testing, production
3. **Per-experiment settings** - Each experiment should have its own configuration
4. **Validation** - Ensure configuration values are valid before use
5. **Serializability** - Save and load configurations in JSON/YAML format
6. **Overrides** - Allow command-line and runtime overrides
7. **Documentation** - Self-documenting configuration with descriptions

## 3. Implementation Plan

### 3.1 Configuration Structure

We will implement a class-based configuration system with the following components:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import os
import yaml
import json

@dataclass
class DatasetConfig:
    """Configuration for dataset processing."""
    path: str
    test_size: float = 0.2
    val_size: float = 0.2
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    target: str = None
    drop_columns: List[str] = field(default_factory=list)
    impute_strategy: Dict[str, str] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    hidden_dims: List[int]
    dropout: float = 0.2
    activation: str = "relu"
    final_activation: Optional[str] = None

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

@dataclass
class RegularizationConfig:
    """Configuration for regularization techniques."""
    weight: float = 0.0
    temperature: float = 0.07
    similarity_threshold: float = 0.0
    layers: List[str] = field(default_factory=list)
    minibatch_size: int = 1024

@dataclass
class MetricsConfig:
    """Configuration for evaluation metrics."""
    cluster_entropy: Dict[str, Any] = field(default_factory=dict)
    subspace_angle: Dict[str, Any] = field(default_factory=dict)
    explainable_threshold_similarity: Dict[str, Any] = field(default_factory=dict)
    cross_layer_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VisualizationConfig:
    """Configuration for visualization components."""
    plot: Dict[str, Any] = field(default_factory=dict)
    umap: Dict[str, Any] = field(default_factory=dict)
    pca: Dict[str, Any] = field(default_factory=dict)
    trajectory: Dict[str, Any] = field(default_factory=dict)
    ets: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisConfig:
    """Configuration for analysis components."""
    path_archetypes: Dict[str, Any] = field(default_factory=dict)
    transition_matrix: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Config:
    """Root configuration class for the Concept Fragmentation project."""
    # General settings
    random_seed: int = 42
    results_dir: str = field(default_factory=lambda: os.environ.get("RESULTS_DIR", "results"))
    device: str = "cuda"
    
    # Component configurations
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: Dict[str, TrainingConfig] = field(default_factory=dict)
    regularization: Dict[str, RegularizationConfig] = field(default_factory=dict)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # Convenience methods
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        # Implementation...
        
    def to_json(self, filepath: str = None) -> str:
        """Save configuration as JSON."""
        # Implementation...
        
    def to_yaml(self, filepath: str = None) -> str:
        """Save configuration as YAML."""
        # Implementation...
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from a dictionary."""
        # Implementation...
        
    @classmethod
    def from_json(cls, json_str: str) -> 'Config':
        """Create configuration from a JSON string."""
        # Implementation...
        
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Config':
        """Create configuration from a YAML string."""
        # Implementation...
        
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from a file (JSON or YAML)."""
        # Implementation...
```

### 3.2 Configuration Manager

We will implement a singleton ConfigManager to handle configuration loading, validation, and access:

```python
class ConfigManager:
    """Singleton class to manage configuration across the project."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._config = Config()  # Initialize with defaults
        return cls._instance
    
    def get_config(self) -> Config:
        """Get the current configuration."""
        return self._config
    
    def set_config(self, config: Config) -> None:
        """Set a new configuration."""
        self._config = config
    
    def load_from_file(self, filepath: str) -> Config:
        """Load configuration from a file."""
        self._config = Config.from_file(filepath)
        return self._config
    
    def load_from_dict(self, config_dict: Dict[str, Any]) -> Config:
        """Load configuration from a dictionary."""
        self._config = Config.from_dict(config_dict)
        return self._config
    
    def override(self, overrides: Dict[str, Any]) -> Config:
        """Override specific configuration values."""
        # Implementation...
        return self._config
    
    def get_experiment_config(self, dataset: str, experiment_id: str) -> Config:
        """Get configuration for a specific experiment."""
        # Implementation...
        return Config()
    
    def validate(self) -> bool:
        """Validate the current configuration."""
        # Implementation...
        return True
```

### 3.3 Configuration Defaults

We will provide a comprehensive set of default configurations in:
- `concept_fragmentation/config/defaults.py` - Default configuration values
- `concept_fragmentation/config/datasets/` - Dataset-specific configurations
- `concept_fragmentation/config/experiments/` - Pre-defined experiment configurations

### 3.4 Configuration CLI

We will enhance the command-line interface to support configuration operations:

```python
import argparse
import sys
from concept_fragmentation.config import ConfigManager

def main():
    parser = argparse.ArgumentParser(description="Concept Fragmentation Configuration Tool")
    
    # Subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # View configuration
    view_parser = subparsers.add_parser("view", help="View configuration")
    view_parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    view_parser.add_argument("--output", help="Output file (if not specified, print to stdout)")
    
    # Export configuration
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    export_parser.add_argument("--output", required=True, help="Output file")
    export_parser.add_argument("--dataset", help="Dataset to export configuration for")
    export_parser.add_argument("--experiment", help="Experiment to export configuration for")
    
    # Import configuration
    import_parser = subparsers.add_parser("import", help="Import configuration")
    import_parser.add_argument("input", help="Input file (JSON or YAML)")
    import_parser.add_argument("--validate", action="store_true", help="Validate configuration")
    
    # Validate configuration
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("input", help="Input file (JSON or YAML)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    cm = ConfigManager()
    
    if args.command == "view":
        # Implementation...
        pass
    elif args.command == "export":
        # Implementation...
        pass
    elif args.command == "import":
        # Implementation...
        pass
    elif args.command == "validate":
        # Implementation...
        pass
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 4. Integration Plan

### 4.1 Migration Strategy

1. **Create the new configuration system** alongside the existing one
2. **Implement ConfigManager** with conversion to/from the current structure
3. **Update core modules** to use the new configuration system
4. **Adapt existing scripts** to work with both the old and new systems
5. **Update documentation** to reflect the new configuration approach
6. **Create migration guide** for users of the old configuration

### 4.2 Integration Points

The following components will need to be updated to use the new configuration system:

1. **Data loaders** - Update to use dataset-specific configuration
2. **Model builders** - Use model configuration for architecture
3. **Training scripts** - Use training and regularization configuration
4. **Metric computation** - Use metrics configuration
5. **Visualization** - Use visualization configuration
6. **Analysis tools** - Use analysis configuration

### 4.3 Backward Compatibility

To maintain backward compatibility:

1. Keep supporting the old configuration format with warning messages
2. Provide conversion utilities from old to new format
3. Document changes and migration path for users

## 5. Testing Strategy

### 5.1 Unit Tests

Create unit tests for:
1. **Configuration classes** - Test initialization, validation, serialization
2. **ConfigManager** - Test singleton behavior, loading, overriding
3. **CLI** - Test command-line interface functions

### 5.2 Integration Tests

Create integration tests for:
1. **End-to-end tests** with configuration files
2. **Cross-module tests** to ensure configuration is properly shared
3. **Migration tests** to verify compatibility with old configuration

## 6. Documentation Plan

### 6.1 User Documentation

1. **Configuration Guide** - Explain the structure and options
2. **Examples** - Provide realistic examples for different use cases
3. **Migration Guide** - Help users transition from the old system

### 6.2 API Documentation

1. **Class documentation** - Document configuration classes and methods
2. **Module documentation** - Document configuration module structure
3. **CLI documentation** - Document command-line interface

## 7. Implementation Timeline

1. **Week 1: Core Implementation**
   - Implement configuration classes
   - Implement ConfigManager
   - Write unit tests

2. **Week 2: Integration**
   - Integrate with core modules
   - Implement CLI
   - Write integration tests

3. **Week 3: Documentation & Finalization**
   - Write user documentation
   - Write API documentation
   - Finalize migration strategy

## 8. Extensions and Future Work

1. **Schema validation** using JSON Schema or Pydantic
2. **Web UI** for configuration editing
3. **Version control** for configurations
4. **Experiment tracking** integration (MLflow, Weights & Biases)
5. **Remote configuration** storage and sharing