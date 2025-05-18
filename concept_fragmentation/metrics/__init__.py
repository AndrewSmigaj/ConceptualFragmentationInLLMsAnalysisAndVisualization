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

from .intra_class_distance import (
    compute_intra_class_distance,
    compute_fragmentation_score as compute_icpd_fragmentation_score
)

from .optimal_num_clusters import (
    compute_optimal_k,
    compute_fragmentation_score as compute_kstar_fragmentation_score
)

from .representation_stability import (
    compute_representation_stability,
    compute_layer_stability_profile,
    compute_average_stability,
    compute_fragmentation_score as compute_stability_fragmentation_score
)

from .explainable_threshold_similarity import (
    compute_ets_clustering,
    compute_dimension_thresholds,
    explain_ets_similarity,
    compute_ets_statistics
)

__all__ = [
    'compute_cluster_entropy',
    'compute_entropy_fragmentation_score',
    'compute_subspace_angle',
    'compute_angle_fragmentation_score',
    'compute_intra_class_distance',
    'compute_icpd_fragmentation_score',
    'compute_optimal_k',
    'compute_kstar_fragmentation_score',
    'compute_representation_stability',
    'compute_layer_stability_profile',
    'compute_average_stability',
    'compute_stability_fragmentation_score',
    'compute_ets_clustering',
    'compute_dimension_thresholds',
    'explain_ets_similarity',
    'compute_ets_statistics'
]