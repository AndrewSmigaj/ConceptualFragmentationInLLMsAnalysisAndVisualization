"""
Visualization module for Concept Fragmentation project.

This module provides utilities for visualizing neural network activations
and activation trajectories, with a focus on concept fragmentation analysis.
"""

from .activations import (
    plot_activations_2d,
    plot_activations_3d,
    plot_layer_comparison,
    plot_topk_neuron_activations,
    reduce_dimensions
)

from .trajectories import (
    plot_sample_trajectory,
    plot_activation_flow,
    plot_class_trajectories,
    compute_compressed_paths
)

__all__ = [
    'plot_activations_2d',
    'plot_activations_3d',
    'plot_layer_comparison',
    'plot_topk_neuron_activations',
    'reduce_dimensions',
    'plot_sample_trajectory',
    'plot_activation_flow',
    'plot_class_trajectories',
    'compute_compressed_paths'
]
