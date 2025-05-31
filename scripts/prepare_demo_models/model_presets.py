"""
Predefined model configurations for different demonstration purposes.
"""
from typing import Dict, Any, Optional


# Base configurations that can be overridden
BASE_TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'optimizer': 'adam',
    'early_stopping_patience': 10,
    'seed': 42
}


def get_preset_config(dataset: str, variant: str) -> Dict[str, Any]:
    """
    Get a predefined configuration for a specific dataset and variant.
    
    Args:
        dataset: Dataset name ('titanic' or 'heart_disease')
        variant: Model variant name
        
    Returns:
        Configuration dictionary
    """
    if dataset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    if variant not in PRESET_CONFIGS[dataset]:
        raise ValueError(f"Unknown variant '{variant}' for dataset '{dataset}'")
    
    # Merge base config with specific config
    config = BASE_TRAINING_CONFIG.copy()
    config.update(PRESET_CONFIGS[dataset][variant])
    
    return config


# Predefined configurations for different model variants
PRESET_CONFIGS = {
    'titanic': {
        # Optimal will be found via hyperparameter search
        'optimal': None,
        
        # Severe bottleneck to force concept compression
        'bottleneck': {
            'model_type': 'bottleneck',
            'bottleneck_size': 4,  # Very small bottleneck
            'expansion_factor': 8,
            'activation': 'relu',
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'description': 'Bottleneck architecture forcing concept compression'
        },
        
        # Overparameterized model likely to overfit
        'overfit': {
            'model_type': 'overparameterized',
            'n_layers': 4,
            'width_multiplier': 15,  # Very wide layers for small dataset
            'activation': 'relu',
            'dropout': 0.0,  # No regularization
            'lr': 0.01,  # High learning rate
            'batch_size': 8,  # Small batches
            'weight_decay': 0.0,
            'epochs': 200,  # Train for longer to ensure overfitting
            'early_stopping_patience': 0,  # No early stopping
            'description': 'Overparameterized model designed to overfit'
        },
        
        # Underparameterized model
        'underfit': {
            'model_type': 'standard',
            'hidden_sizes': [4, 2],  # Too small for the task
            'activation': 'sigmoid',  # Less effective for deep networks
            'dropout': 0.5,  # Too much dropout
            'lr': 0.0001,  # Too low learning rate
            'batch_size': 128,  # Large batches
            'epochs': 20,  # Too few epochs
            'description': 'Underparameterized model with poor training setup'
        },
        
        # Unstable training configuration
        'unstable': {
            'model_type': 'standard',
            'hidden_sizes': [32, 64, 16, 32],  # Erratic layer sizes
            'activation': 'tanh',
            'dropout': 0.3,
            'lr': 0.1,  # Way too high
            'batch_size': 16,
            'optimizer': 'sgd',  # No momentum
            'weight_decay': 0.01,  # High weight decay with high LR
            'description': 'Unstable training configuration'
        },
        
        # Model designed to create fragmented concepts
        'fragmented': {
            'model_type': 'standard',
            'hidden_sizes': [100, 100, 100],  # Many redundant neurons
            'activation': 'relu',
            'dropout': 0.0,
            'lr': 0.005,
            'batch_size': 32,
            'init_method': 'normal',  # Poor initialization
            'seed': None,  # Random seed each time
            'description': 'Architecture prone to concept fragmentation'
        },
        
        # Well-regularized model
        'regularized': {
            'model_type': 'standard',
            'hidden_sizes': [32, 16],
            'activation': 'elu',
            'dropout': 0.3,
            'lr': 0.001,
            'batch_size': 32,
            'weight_decay': 0.001,
            'batch_norm': True,
            'description': 'Well-regularized model with good practices'
        }
    },
    
    'heart_disease': {
        # Optimal will be found via hyperparameter search
        'optimal': None,
        
        # Bottleneck for heart disease (more features than Titanic)
        'bottleneck': {
            'model_type': 'bottleneck',
            'bottleneck_size': 6,  # Slightly larger than Titanic
            'expansion_factor': 6,
            'activation': 'relu',
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'description': 'Bottleneck architecture for heart disease features'
        },
        
        # Overfit configuration
        'overfit': {
            'model_type': 'overparameterized',
            'n_layers': 5,
            'width_multiplier': 10,
            'activation': 'relu',
            'dropout': 0.0,
            'lr': 0.01,
            'batch_size': 8,
            'weight_decay': 0.0,
            'epochs': 200,
            'early_stopping_patience': 0,
            'description': 'Overparameterized model for heart disease'
        },
        
        # Underfit configuration
        'underfit': {
            'model_type': 'standard',
            'hidden_sizes': [8, 4],  # Too small
            'activation': 'sigmoid',
            'dropout': 0.4,
            'lr': 0.0001,
            'batch_size': 128,
            'epochs': 15,
            'description': 'Underparameterized model for heart disease'
        },
        
        # Multi-path architecture (interesting for fragmentation)
        'multipath': {
            'model_type': 'standard',
            'hidden_sizes': [64, 128, 64, 32],  # Expansion then contraction
            'activation': 'leaky_relu',
            'dropout': 0.1,
            'lr': 0.001,
            'batch_size': 32,
            'description': 'Multi-path architecture with expansion and contraction'
        },
        
        # Sparse architecture
        'sparse': {
            'model_type': 'standard',
            'hidden_sizes': [256, 16, 256],  # Very sparse middle layer
            'activation': 'relu',
            'dropout': 0.2,
            'lr': 0.001,
            'batch_size': 32,
            'weight_decay': 0.01,  # High L2 to encourage sparsity
            'description': 'Sparse architecture with high regularization'
        }
    }
}


def get_search_space(dataset: str) -> Dict[str, Any]:
    """
    Get hyperparameter search space for Optuna optimization.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary defining the search space
    """
    # Base search space
    base_space = {
        'n_layers': {'type': 'int', 'low': 1, 'high': 3},
        'lr': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True},
        'dropout': {'type': 'float', 'low': 0.0, 'high': 0.4},
        'activation': {'type': 'categorical', 'choices': ['relu', 'elu', 'leaky_relu']},
        'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
        'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw']},
        'weight_decay': {'type': 'float', 'low': 0.0, 'high': 0.01}
    }
    
    # Dataset-specific adjustments
    if dataset == 'titanic':
        # Smaller layers for smaller dataset
        base_space['layer_sizes'] = {
            'type': 'categorical',
            'choices': [16, 32, 64, 128]
        }
    elif dataset == 'heart_disease':
        # Can support larger layers
        base_space['layer_sizes'] = {
            'type': 'categorical',
            'choices': [32, 64, 128, 256]
        }
    
    return base_space


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if not
    """
    required_fields = ['lr', 'batch_size']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ranges
    if config.get('lr', 0) <= 0:
        raise ValueError("Learning rate must be positive")
    
    if config.get('batch_size', 0) <= 0:
        raise ValueError("Batch size must be positive")
    
    if 'dropout' in config and not (0 <= config['dropout'] <= 1):
        raise ValueError("Dropout must be between 0 and 1")
    
    return True