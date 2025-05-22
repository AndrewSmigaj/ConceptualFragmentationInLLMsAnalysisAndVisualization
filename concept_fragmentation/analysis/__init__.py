"""
Analysis module for the Conceptual Fragmentation project.

This package contains modules for analyzing activation clusters, paths, and
similarity metrics across neural network layers.
"""

# Re-export key classes
from .transformer_dimensionality import (
    TransformerDimensionalityReducer,
    DimensionalityReductionResult,
    DimensionalityReductionPipelineStage
)

from .transformer_cross_layer import (
    TransformerCrossLayerAnalyzer,
    CrossLayerTransformerAnalysisResult
)

# Making the analysis directory a proper Python package