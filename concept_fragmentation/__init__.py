"""
Concept Fragmentation Analysis Library

A comprehensive library for analyzing concept trajectories in neural networks,
with a focus on transformer models.
"""

__version__ = "2.0.0"
__author__ = "CTA Team"

# Import key classes for easier access - but only if they exist
__all__ = ["__version__", "__author__"]

# Optional imports with error handling
try:
    from .clustering.base import BaseClusterer
    __all__.append("BaseClusterer")
except ImportError:
    pass

try:
    from .labeling.base import BaseLabeler
    __all__.append("BaseLabeler")
except ImportError:
    pass

try:
    from .visualization.base import BaseVisualizer
    __all__.append("BaseVisualizer")
except ImportError:
    pass

try:
    from .experiments.base import BaseExperiment
    __all__.append("BaseExperiment")
except ImportError:
    pass