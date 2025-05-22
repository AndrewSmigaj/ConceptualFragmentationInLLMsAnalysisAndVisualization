"""
Activation collection and processing module.

This module provides tools for efficient collection, streaming, and processing
of activations from neural network models. It focuses on memory efficiency
through streaming operations and flexible persistence options.

Key components:
- ActivationCollector: Manages collection of activations with streaming support
- ActivationProcessor: Processes and transforms collected activations
- ActivationStorage: Handles persistence of activations to disk/memory
"""

from .collector import ActivationCollector, collect_activations
from .processor import ActivationProcessor
from .storage import ActivationStorage, ActivationFormat

__all__ = [
    'ActivationCollector',
    'collect_activations',
    'ActivationProcessor',
    'ActivationStorage',
    'ActivationFormat',
]