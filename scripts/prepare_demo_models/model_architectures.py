"""
Flexible feedforward neural network architectures for demo models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class OptimizableFFNet(nn.Module):
    """
    Flexible feedforward network with configurable depth and width.
    Supports variable number of layers unlike the fixed 3-layer version.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        init_method: str = 'xavier'
    ):
        """
        Initialize flexible feedforward network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes (can be any length)
            output_size: Number of output classes
            activation: Activation function name
            dropout_rate: Dropout probability (0 = no dropout)
            batch_norm: Whether to use batch normalization
            init_method: Weight initialization method
        """
        super().__init__()
        
        # Store architecture info
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation_name = activation
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization (before activation)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer (no activation, dropout, or batch norm)
        layers.append(nn.Linear(prev_size, output_size))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights(init_method)
        
        # Dictionary to store activations for analysis
        self.activations = {}
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function module by name."""
        activations = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'swish': nn.SiLU(),  # Swish activation
            'gelu': nn.GELU()
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
        
        return activations[name]
    
    def _initialize_weights(self, method: str):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'normal':
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                else:
                    # Default PyTorch initialization
                    pass
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Store input
        self.activations['input'] = x.detach()
        
        # Pass through each layer and store activations
        current = x
        layer_idx = 0
        
        for i, module in enumerate(self.network):
            current = module(current)
            
            # Store activations after each linear layer
            if isinstance(module, nn.Linear) and i < len(self.network) - 1:
                self.activations[f'layer{layer_idx + 1}'] = current.detach()
                layer_idx += 1
        
        # Store output
        self.activations['output'] = current.detach()
        
        return current
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get stored activations for all layers."""
        return self.activations
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of network architecture."""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'n_layers': len(self.hidden_sizes),
            'total_params': sum(p.numel() for p in self.parameters()),
            'activation': self.activation_name,
            'dropout_rate': self.dropout_rate
        }


class BottleneckFFNet(OptimizableFFNet):
    """
    Specialized architecture with bottleneck for concept compression.
    """
    
    def __init__(
        self,
        input_size: int,
        bottleneck_size: int,
        expansion_factor: int = 4,
        output_size: int = 2,
        **kwargs
    ):
        """
        Create bottleneck architecture.
        
        Args:
            input_size: Number of input features
            bottleneck_size: Size of bottleneck layer
            expansion_factor: How much to expand after bottleneck
            output_size: Number of output classes
            **kwargs: Additional arguments for OptimizableFFNet
        """
        # Create bottleneck architecture: expand -> compress -> expand
        first_layer = input_size * 2  # Initial expansion
        hidden_sizes = [
            first_layer,
            bottleneck_size,  # Compression
            bottleneck_size * expansion_factor  # Expansion
        ]
        
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            **kwargs
        )


class OverparameterizedFFNet(OptimizableFFNet):
    """
    Intentionally overparameterized network for demonstrating fragmentation.
    """
    
    def __init__(
        self,
        input_size: int,
        n_layers: int = 5,
        width_multiplier: int = 20,
        output_size: int = 2,
        **kwargs
    ):
        """
        Create overparameterized architecture.
        
        Args:
            input_size: Number of input features
            n_layers: Number of hidden layers
            width_multiplier: Multiplier for layer width
            output_size: Number of output classes
            **kwargs: Additional arguments for OptimizableFFNet
        """
        # Create very wide layers
        hidden_sizes = [input_size * width_multiplier] * n_layers
        
        # Force no regularization
        kwargs['dropout_rate'] = 0.0
        kwargs['batch_norm'] = False
        
        super().__init__(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            **kwargs
        )


def create_model(model_type: str, input_size: int, output_size: int, 
                 config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create models based on type and config.
    
    Args:
        model_type: Type of model ('standard', 'bottleneck', 'overparameterized')
        input_size: Number of input features
        output_size: Number of output classes
        config: Model configuration
        
    Returns:
        PyTorch model instance
    """
    if model_type == 'standard':
        return OptimizableFFNet(
            input_size=input_size,
            hidden_sizes=config.get('hidden_sizes', [64, 32]),
            output_size=output_size,
            activation=config.get('activation', 'relu'),
            dropout_rate=config.get('dropout', 0.0),
            batch_norm=config.get('batch_norm', False),
            init_method=config.get('init_method', 'xavier')
        )
    
    elif model_type == 'bottleneck':
        return BottleneckFFNet(
            input_size=input_size,
            bottleneck_size=config.get('bottleneck_size', 8),
            expansion_factor=config.get('expansion_factor', 4),
            output_size=output_size,
            activation=config.get('activation', 'relu'),
            dropout_rate=config.get('dropout', 0.1)
        )
    
    elif model_type == 'overparameterized':
        return OverparameterizedFFNet(
            input_size=input_size,
            n_layers=config.get('n_layers', 5),
            width_multiplier=config.get('width_multiplier', 20),
            output_size=output_size,
            activation=config.get('activation', 'relu')
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")