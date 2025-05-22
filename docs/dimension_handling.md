# Dimension Handling in Neural Network Activations

This document describes the dimension handling mechanisms implemented in the Conceptual Fragmentation project to address dimension mismatches between training and test activations.

## Problem Description

When running neural network analysis, particularly with tabular datasets like the Heart Disease dataset, dimension mismatches can occur between training and test data. This leads to inconsistent activation shapes, which can cause errors during analysis, particularly when:

- Concatenating train and test activations
- Computing similarity metrics across layers
- Visualizing activations and generating cluster paths

The most common scenario is when the preprocessor creates different feature dimensions for train and test sets, as observed in the Heart Disease dataset: train shape (216, 2) vs test shape (54, 64).

## Solution Components

We've implemented a comprehensive solution with the following components:

### 1. Dimension Validation in Train.py

The `prepare_dataset` function now validates dimensions between train and test sets and provides metadata about any mismatches. This helps catch issues early in the pipeline.

```python
metadata = {
    "train_shape": X_train.shape,
    "test_shape": X_test.shape,
    "dimensions_match": X_train.shape[1] == X_test.shape[1]
}
```

### 2. Configurable Fallback Behavior in Activation Hooks

The activation_hooks module now supports several strategies for handling dimension mismatches:

- **warn**: Log warnings but don't modify activations (default)
- **error**: Raise an exception when a mismatch is detected
- **truncate**: Automatically truncate to the smallest common dimensions
- **pad**: Automatically pad to the largest common dimensions

Control this behavior using the `set_dimension_mismatch_strategy` function:

```python
from concept_fragmentation.hooks.activation_hooks import set_dimension_mismatch_strategy

# Choose a strategy
set_dimension_mismatch_strategy('truncate')
```

### 3. Dimension Tracking Between Train and Test

The new `track_train_test_dimensions` function provides detailed information about dimension mismatches and can apply the configured strategy:

```python
results = track_train_test_dimensions(
    train_activations, 
    test_activations,
    phase="training",
    dataset="heart",
    handle_mismatch=True
)
```

### 4. Consistent Preprocessing with DataPreprocessor

The `DataPreprocessor` class now includes dimension consistency options:

```python
preprocessor = DataPreprocessor(
    categorical_cols=["sex", "cp", ...],
    numerical_cols=["age", "trestbps", ...],
    target_col="target",
    ensure_dimension_consistency=True,
    dimension_handling='truncate'
)
```

This ensures that the same feature dimensions are maintained between training and test data, using strategies like truncation or padding.

## Usage Guide

### Enabling Dimension Checks

Use the `enable_dimension_checks.py` script to enable dimension validation and set the handling strategy:

```bash
# Enable with default 'warn' strategy
python enable_dimension_checks.py

# Enable with 'truncate' strategy and verbose logging
python enable_dimension_checks.py truncate --verbose
```

### In Analysis Scripts

For existing scripts, add the following code at the beginning:

```python
from concept_fragmentation.hooks.activation_hooks import (
    set_dimension_logging,
    set_dimension_mismatch_strategy
)

# Enable dimension logging for diagnostics
set_dimension_logging(True)

# Set strategy for dimension mismatches
set_dimension_mismatch_strategy('truncate')
```

### For the Heart Dataset

The Heart dataset specifically benefits from the 'truncate' strategy, which ensures consistent dimensions by truncating to the common feature set (dimension 2) across train and test sets.

## Troubleshooting

### Diagnosing Dimension Issues

1. Enable verbose dimension logging:
   ```python
   set_dimension_logging(True)
   ```

2. Run a test track of dimensions:
   ```python
   results = track_train_test_dimensions(train_activations, test_activations)
   ```

3. Check the logs for detailed dimension information.

### Common Issues

1. **Inconsistent concatenation**: If you see "ValueError: all input arrays must have the same shape", this indicates dimension mismatches during concatenation. Apply the 'truncate' strategy.

2. **NaN or inf values**: These can appear when dimensions are inconsistent. Check activation statistics with:
   ```python
   stats = get_activation_statistics(activations)
   ```

3. **Different cluster counts**: If clusters differ dramatically between runs, inconsistent dimensions may be causing it. Ensure dimensions are consistent.

## Running Tests

Unit tests for dimension handling are available in the `/concept_fragmentation/hooks/tests/` directory:

```bash
# Run all dimension tests
python -m unittest concept_fragmentation.hooks.tests.test_activation_dimensions

# Run a specific test
python -m unittest concept_fragmentation.hooks.tests.test_activation_dimensions.TestActivationDimensions.test_track_dimensions_truncate_strategy
```