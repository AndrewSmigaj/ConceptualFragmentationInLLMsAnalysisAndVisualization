"""
Feedforward neural network implementation for concept fragmentation analysis.
This module provides a 3-layer feedforward network with configurable layer sizes,
fixed seed initialization, ReLU activations, and named layers for hook registration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

from ..config import MODELS, RANDOM_SEED

class FeedforwardNetwork(nn.Module):
    """
    A 3-layer feedforward neural network with configurable layer sizes.
    
    Features:
    - Configurable layer sizes
    - Fixed seed initialization
    - ReLU activations
    - Named layers for hook registration
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_layer_sizes: Optional[List[int]] = None,
        dropout_rate: float = MODELS["feedforward"]["dropout"],
        activation: str = MODELS["feedforward"]["activation"],
        seed: int = RANDOM_SEED
    ):
        """
        Initialize the feedforward network.
        
        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output (number of classes)
            hidden_layer_sizes: List of hidden layer sizes. If None, uses config values.
            dropout_rate: Probability of dropout. Defaults to config value.
            activation: Activation function to use. Defaults to config value.
            seed: Random seed for weight initialization. Defaults to RANDOM_SEED.
        """
        super(FeedforwardNetwork, self).__init__()
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Use default hidden layer sizes from config if not provided
        if hidden_layer_sizes is None:
            # This needs dataset-specific values, so we'll need to handle this differently
            # Default to first dataset's hidden dimensions if not provided
            # TODO: Callers should normally provide dataset-specific hidden_layer_sizes
            # rather than relying on this default, which just picks the first available config
            hidden_layer_sizes = list(MODELS["feedforward"]["hidden_dims"].values())[0]
        
        # Ensure we have exactly 3 hidden layers
        if len(hidden_layer_sizes) != 3:
            raise ValueError(f"Expected 3 hidden layers, got {len(hidden_layer_sizes)}")
        
        # Define network layers with named modules for hook registration
        self.fc1 = nn.Linear(input_dim, hidden_layer_sizes[0])
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1])
        self.fc3 = nn.Linear(hidden_layer_sizes[1], hidden_layer_sizes[2])
        self.output = nn.Linear(hidden_layer_sizes[2], output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Store layer dimensions for reference
        self.layer_dims = [
            input_dim, 
            hidden_layer_sizes[0], 
            hidden_layer_sizes[1], 
            hidden_layer_sizes[2], 
            output_dim
        ]
        
        # Map activation function name to function
        self.activation_name = activation
        self.activation_fn = self._get_activation(activation)
        
        # Dictionary to store activations for each layer (used with hooks)
        self.activations: Dict[str, torch.Tensor] = {}
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> callable:
        """
        Get the activation function based on name.
        
        Args:
            activation: Name of the activation function
            
        Returns:
            Activation function
        """
        activations = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu
        }
        
        if activation not in activations:
            raise ValueError(f"Activation {activation} not supported. Choose from {list(activations.keys())}")
        
        return activations[activation]
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # First hidden layer
        x = self.fc1(x)
        x = self.activation_fn(x)
        self.activations['layer1'] = x.detach()  # Store activations for hooks
        x = self.dropout(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.activation_fn(x)
        self.activations['layer2'] = x.detach()  # Store activations for hooks
        x = self.dropout(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.activation_fn(x)
        self.activations['layer3'] = x.detach()  # Store activations for hooks
        x = self.dropout(x)
        
        # Output layer
        x = self.output(x)
        self.activations['output'] = x.detach()  # Store raw logits
        
        return x
    
    def get_layer_by_name(self, name: str) -> nn.Module:
        """
        Get a layer by its name for hook registration.
        
        Args:
            name: Name of the layer ('layer1', 'layer2', 'layer3', 'output')
            
        Returns:
            The requested layer module
        """
        name_to_layer = {
            'layer1': self.fc1,
            'layer2': self.fc2,
            'layer3': self.fc3,
            'output': self.output
        }
        
        if name not in name_to_layer:
            raise ValueError(f"Layer name {name} not found. Available layers: {list(name_to_layer.keys())}")
        
        return name_to_layer[name]
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """
        Get stored activations for all layers.
        
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        return self.activations
    
    def get_layer_dims(self) -> List[int]:
        """
        Get dimensions of all layers.
        
        Returns:
            List of layer dimensions [input_dim, hidden1, hidden2, hidden3, output_dim]
        """
        return self.layer_dims
