# Configuration System Migration Guide

This guide explains how to migrate from the legacy configuration system (using direct imports from `concept_fragmentation.config`) to the new hierarchical, class-based configuration system.

## Overview

The new configuration system offers several advantages:
- **Type safety** with dataclasses
- **Hierarchical organization** of related settings
- **Validation** of configuration values
- **Serialization/deserialization** to/from JSON and YAML
- **Environment-specific configurations**
- **Per-experiment configurations**

While maintaining complete backward compatibility with the legacy system.

## Backward Compatibility

For existing code, **no changes are required**. The new system maintains full backward compatibility:

```python
# This still works exactly as before
from concept_fragmentation.config import RANDOM_SEED, MODELS, TRAINING
```

All legacy configuration constants continue to be available through the same imports.

## Migrating to the New Configuration System

While backward compatibility is maintained, we recommend migrating to the new API for new code:

### Step 1: Import the Configuration Manager

Instead of importing individual configuration variables, import the configuration manager:

```python
# Legacy approach
from concept_fragmentation.config import RANDOM_SEED, MODELS, TRAINING

# New approach
from concept_fragmentation.config import config_manager, get_config
```

### Step 2: Access Configuration Values

Use the configuration manager to access configuration values:

```python
# Legacy approach
batch_size = TRAINING["batch_size"]["titanic"]
hidden_dims = MODELS["feedforward"]["hidden_dims"]["titanic"]

# New approach
config = get_config()
batch_size = config.training["titanic"].batch_size
hidden_dims = config.models["feedforward_titanic"].hidden_dims
```

### Step 3: Use Typed Configuration Classes

The new system provides typed configuration classes:

```python
from concept_fragmentation.config import (
    DatasetConfig, ModelConfig, TrainingConfig, RegularizationConfig
)

# Create a custom model configuration
model_config = ModelConfig(
    hidden_dims=[128, 64, 32],
    dropout=0.3,
    activation="relu",
    final_activation="sigmoid"
)
```

### Step 4: Override Configuration Values

Override specific configuration values at runtime:

```python
# Create a modified configuration
custom_config = config_manager.get_config().update(
    random_seed=123,
    "metrics.cluster_entropy.default_k": 5
)

# Use this configuration for the current operation
with config_manager.use_config(custom_config):
    # Run code with custom configuration
    pass
```

### Step 5: Load and Save Configurations

Load configuration from files or save it to files:

```python
# Load from file
config = config_manager.load_from_file("experiment_config.yaml")

# Save to file
config_manager.save_config("experiment_config.json", format="json")
```

### Step 6: Get Experiment-Specific Configurations

Create experiment-specific configurations:

```python
# Get configuration for a specific experiment
exp_config = config_manager.get_experiment_config(
    dataset="titanic",
    experiment_id="cohesion_regularized_001"
)

# The experiment config has dataset-specific paths
results_dir = exp_config.results_dir  # e.g., "results/titanic/cohesion_regularized_001"
```

## Configuration Classes

The new system provides the following configuration classes:

- `Config`: Root configuration class
- `DatasetConfig`: Dataset-specific configuration
- `ModelConfig`: Model architecture configuration
- `TrainingConfig`: Training hyperparameters
- `RegularizationConfig`: Regularization settings
- `MetricsConfig`: Metrics computation settings
- `VisualizationConfig`: Visualization settings
- `AnalysisConfig`: Analysis settings

And more specialized classes for specific features.

## Best Practices

1. **Use typed configuration classes** for better IDE support and error checking
2. **Validate configurations** using the `config_manager.validate()` method
3. **Create experiment-specific configurations** with `get_experiment_config()`
4. **Save configurations** with experiments for reproducibility
5. **Use environment variables** for machine-specific settings

## Command-Line Interface

The new system includes a command-line interface for managing configurations:

```bash
# Export configuration to a file
python -m concept_fragmentation.config.cli export --format yaml config.yaml

# View current configuration
python -m concept_fragmentation.config.cli view

# Validate a configuration file
python -m concept_fragmentation.config.cli validate config.yaml
```

## Legacy Import Equivalents

| Legacy Import                      | New Equivalent                                        |
|------------------------------------|------------------------------------------------------|
| `RANDOM_SEED`                      | `get_config().random_seed`                           |
| `RESULTS_DIR`                      | `get_config().results_dir`                           |
| `DEVICE`                           | `get_config().device`                                |
| `DATASETS["titanic"]`              | `get_config().datasets["titanic"]`                   |
| `MODELS["feedforward"]`            | `{name.replace("feedforward_", ""): config for name, config in get_config().models.items() if name.startswith("feedforward_")}` |
| `TRAINING["batch_size"]["titanic"]`| `get_config().training["titanic"].batch_size`        |
| `METRICS["cluster_entropy"]`       | `get_config().metrics.cluster_entropy`               |

## Getting Help

If you encounter any issues or have questions about the configuration system, please refer to the unit tests in `tests/config/` or open an issue on the project repository.