# Custom Configuration Schema

This document describes the JSON schema for custom model configurations.

## Required Fields

- `lr` (float): Learning rate (e.g., 0.001)
- `batch_size` (int): Batch size for training (e.g., 32)

## Optional Fields

### Model Architecture
- `model_type` (string): Type of model - "standard", "bottleneck", or "overparameterized" (default: "standard")
- `hidden_sizes` (array of int): List of hidden layer sizes for standard models (e.g., [64, 32])
- `bottleneck_size` (int): Size of bottleneck layer for bottleneck models
- `expansion_factor` (int): Expansion factor after bottleneck
- `n_layers` (int): Number of layers for overparameterized models
- `width_multiplier` (int): Width multiplier for overparameterized models

### Model Configuration
- `activation` (string): Activation function - "relu", "elu", "leaky_relu", "tanh", "sigmoid", "swish", "gelu" (default: "relu")
- `dropout` (float): Dropout rate between 0 and 1 (default: 0.0)
- `batch_norm` (bool): Whether to use batch normalization (default: false)
- `init_method` (string): Weight initialization - "xavier", "kaiming", "normal" (default: "xavier")

### Training Configuration
- `optimizer` (string): Optimizer - "adam", "adamw", "sgd", "rmsprop" (default: "adam")
- `weight_decay` (float): L2 regularization weight (default: 0.0)
- `momentum` (float): Momentum for SGD optimizer (default: 0.9)
- `epochs` (int): Maximum training epochs (default: 100)
- `early_stopping_patience` (int): Epochs to wait before early stopping (default: 10)
- `seed` (int or null): Random seed for reproducibility (default: 42)

### Metadata
- `description` (string): Description of the configuration

## Examples

### Standard Network
```json
{
  "model_type": "standard",
  "hidden_sizes": [128, 64, 32],
  "activation": "elu",
  "dropout": 0.2,
  "lr": 0.001,
  "batch_size": 32,
  "optimizer": "adam",
  "weight_decay": 0.0001
}
```

### Bottleneck Network
```json
{
  "model_type": "bottleneck",
  "bottleneck_size": 8,
  "expansion_factor": 4,
  "activation": "relu",
  "lr": 0.0005,
  "batch_size": 64
}
```

### Overparameterized Network
```json
{
  "model_type": "overparameterized",
  "n_layers": 5,
  "width_multiplier": 20,
  "activation": "relu",
  "lr": 0.01,
  "batch_size": 16
}
```