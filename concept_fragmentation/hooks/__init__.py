"""
Hooks package for the Concept Fragmentation project.

This package provides hooks for capturing activations in neural networks.
"""

from .activation_hooks import (
    ActivationHook,
    get_activation_hooks,
    capture_activations,
    get_neuron_importance
)

__all__ = [
    "ActivationHook",
    "get_activation_hooks",
    "capture_activations",
    "get_neuron_importance"
]
