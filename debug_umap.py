#!/usr/bin/env python
"""Debug script to test UMAP trajectory visualization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from concept_mri.components.visualizations.umap_trajectory import UMAPTrajectoryVisualization
import json

# Create mock model data with activations
mock_activations = {
    'layer_0': np.random.randn(100, 32),
    'layer_1': np.random.randn(100, 16),
    'layer_2': np.random.randn(100, 8)
}

mock_model_data = {
    'model_loaded': True,
    'activations': mock_activations,
    'architecture': [32, 16, 8]
}

# Create mock clustering data
mock_clustering_data = {
    'completed': True,
    'algorithm': 'kmeans',
    'clusters_per_layer': {
        'layer_0': {
            'n_clusters': 3,
            'labels': np.random.randint(0, 3, 100).tolist()
        },
        'layer_1': {
            'n_clusters': 3,
            'labels': np.random.randint(0, 3, 100).tolist()
        },
        'layer_2': {
            'n_clusters': 3,
            'labels': np.random.randint(0, 3, 100).tolist()
        }
    },
    'metrics': {
        'unique_paths': 25,
        'total_samples': 100
    }
}

# Create UMAP trajectory visualizer
visualizer = UMAPTrajectoryVisualization()

# Generate figure
try:
    fig_dict = visualizer.generate_umap_trajectory(
        model_data=mock_model_data,
        clustering_data=mock_clustering_data,
        config={
            'n_samples': 20,
            'n_neighbors': 10,
            'color_by': 'cluster',
            'show_options': ['paths', 'arrows']
        }
    )
    
    # Print debug info
    print("Generated UMAP figure dictionary:")
    print(f"Keys: {list(fig_dict.keys())}")
    print(f"Data type: {type(fig_dict.get('data'))}")
    if 'data' in fig_dict and fig_dict['data']:
        print(f"Number of traces: {len(fig_dict['data'])}")
        for i, trace in enumerate(fig_dict['data'][:3]):  # Show first 3 traces
            print(f"Trace {i}: type={trace.get('type')}, mode={trace.get('mode')}")
            if 'x' in trace:
                print(f"  Points: {len(trace['x'])}")
    
    # Save to file
    with open('debug_umap_output.json', 'w') as f:
        json.dump(fig_dict, f, indent=2)
    print("\nFigure saved to debug_umap_output.json")
    
except Exception as e:
    print(f"Error generating UMAP visualization: {e}")
    import traceback
    traceback.print_exc()