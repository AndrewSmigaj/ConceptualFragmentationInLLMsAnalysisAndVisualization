"""
Metrics package for the Concept Fragmentation project.

This package provides metrics to quantify concept fragmentation in neural networks.
"""

from .cluster_entropy import (
    compute_cluster_entropy,
    compute_fragmentation_score as compute_entropy_fragmentation_score
)

from .subspace_angle import (
    compute_subspace_angle,
    compute_fragmentation_score as compute_angle_fragmentation_score
)

__all__ = [
    'compute_cluster_entropy',
    'compute_entropy_fragmentation_score',
    'compute_subspace_angle',
    'compute_angle_fragmentation_score'
]
