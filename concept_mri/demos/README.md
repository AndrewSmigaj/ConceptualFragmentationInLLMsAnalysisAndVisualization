# Concept MRI Demo Models

This directory contains pre-trained feedforward neural networks for demonstrating Concept MRI's analysis capabilities. These models have been trained on classic tabular datasets with various architectures and training configurations to showcase different neural network behaviors.

## Available Datasets

### 1. Titanic Survival Prediction
- **Input Features**: 11 (passenger class, sex, age, fare, etc.)
- **Output Classes**: 2 (survived/not survived)
- **Dataset Size**: ~890 samples

### 2. Heart Disease Prediction
- **Input Features**: 13 (age, chest pain type, blood pressure, etc.)
- **Output Classes**: 2 (disease/no disease)
- **Dataset Size**: ~300 samples

## Model Variants

Each dataset includes multiple model variants designed to demonstrate different aspects of neural network behavior:

### optimal
- **Purpose**: Best performing model found via hyperparameter optimization
- **Method**: Optuna-based Bayesian optimization over architecture and training parameters
- **Use Case**: Baseline for comparison, demonstrates good concept organization
- **Expected Behavior**: Clean clustering, stable concept paths

### bottleneck
- **Architecture**: Wide → Narrow → Wide (e.g., 128 → 8 → 128)
- **Purpose**: Forces concept compression in middle layer
- **Use Case**: Demonstrates how architectural constraints affect concept organization
- **Expected Behavior**: Strong concept compression, potentially clearer clustering

### overfit
- **Architecture**: Very deep and wide (e.g., 5 layers, 500+ neurons each)
- **Purpose**: Intentionally overparameterized for small dataset
- **Training**: No regularization, trained to convergence
- **Use Case**: Shows concept fragmentation from overfitting
- **Expected Behavior**: Fragmented concepts, unstable paths, poor generalization

### underfit
- **Architecture**: Too small (e.g., single layer with 4 neurons)
- **Purpose**: Insufficient capacity for the task
- **Training**: Poor hyperparameters (low learning rate, high dropout)
- **Use Case**: Demonstrates failure to form meaningful concepts
- **Expected Behavior**: Poor clustering, random-like patterns

### unstable
- **Architecture**: Erratic layer sizes (e.g., 32 → 64 → 16 → 32)
- **Training**: Very high learning rate, no momentum
- **Purpose**: Shows effects of poor training dynamics
- **Expected Behavior**: Inconsistent concepts, high variance in paths

### fragmented
- **Architecture**: Many redundant neurons (e.g., 100 → 100 → 100)
- **Purpose**: Demonstrates distributed/fragmented concept representation
- **Training**: Poor initialization, no fixed seed
- **Expected Behavior**: Same concepts spread across multiple clusters

### regularized (Heart Disease only)
- **Architecture**: Moderate size with strong regularization
- **Training**: L2 weight decay, dropout, batch normalization
- **Purpose**: Shows effects of heavy regularization
- **Expected Behavior**: Sparse, well-separated concepts

### multipath (Heart Disease only)
- **Architecture**: Expansion then contraction (e.g., 64 → 128 → 64 → 32)
- **Purpose**: Creates multiple processing paths
- **Expected Behavior**: Complex concept flow patterns

## Performance Metrics

Each model includes metadata with:
- **Training Accuracy**: Performance on training set
- **Validation Accuracy**: Performance on held-out validation set
- **Test Accuracy**: Final performance on test set
- **Architecture Details**: Layer sizes, activation functions, etc.
- **Training Configuration**: Learning rate, optimizer, epochs, etc.

## Usage in Concept MRI

These models are automatically discovered by Concept MRI and can be:
1. Loaded through the model upload interface
2. Selected from the demo models dropdown (if implemented)
3. Analyzed for concept organization and fragmentation
4. Compared to understand how architecture affects concepts

## File Structure

```
dataset_name/
├── model_variant.pt          # PyTorch model file
├── metadata_variant.json     # Model metadata and performance
└── data_info.json           # Dataset information
```

## Model Format

Models are saved in PyTorch format with embedded architecture information:
```python
{
    'model_state_dict': state_dict,     # Model weights
    'input_size': int,                  # Number of input features
    'output_size': int,                 # Number of output classes  
    'architecture': [int, ...],         # Hidden layer sizes
    'activation': str,                  # Activation function
    'dropout_rate': float              # Dropout rate
}
```

## Training Scripts

These models were created using the training scripts in `scripts/prepare_demo_models/`. To retrain or create new variants:

```bash
cd scripts/prepare_demo_models
python train_demo_models.py --dataset titanic --variant optimal
```

## Expected Analysis Results

When analyzing these models in Concept MRI, you should observe:

1. **Optimal models**: Clear concept organization, stable paths
2. **Bottleneck models**: Strong compression effects, fewer but clearer concepts
3. **Overfit models**: Many fragmented clusters, unstable paths
4. **Underfit models**: Poor concept formation, near-random patterns
5. **Regularized models**: Sparse, well-separated concepts

These observations help validate that Concept MRI correctly identifies and visualizes different types of neural network behaviors.