#!/usr/bin/env python
"""Debug script to test Sankey visualization generation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
import json

# Create mock clustering data
mock_clustering_data = {
    'completed': True,
    'algorithm': 'kmeans',
    'hierarchy': 'meso',
    'paths': {
        '0': ['L0_C0', 'L1_C1', 'L2_C2'],
        '1': ['L0_C0', 'L1_C2', 'L2_C1'],
        '2': ['L0_C1', 'L1_C1', 'L2_C2'],
        '3': ['L0_C1', 'L1_C2', 'L2_C0'],
        '4': ['L0_C2', 'L1_C0', 'L2_C1']
    },
    'cluster_labels': {
        'L0_C0': 'Layer 0 - Input Processing',
        'L0_C1': 'Layer 0 - Feature Extraction',
        'L0_C2': 'Layer 0 - Noise Filter',
        'L1_C0': 'Layer 1 - Pattern Recognition',
        'L1_C1': 'Layer 1 - Combination',
        'L1_C2': 'Layer 1 - Transformation',
        'L2_C0': 'Layer 2 - Final Decision A',
        'L2_C1': 'Layer 2 - Final Decision B',
        'L2_C2': 'Layer 2 - Final Decision C'
    },
    'clusters_per_layer': {
        'layer_0': {'n_clusters': 3, 'labels': [0, 0, 1, 1, 2]},
        'layer_1': {'n_clusters': 3, 'labels': [1, 2, 1, 2, 0]},
        'layer_2': {'n_clusters': 3, 'labels': [2, 1, 2, 0, 1]}
    }
}

# Create sankey wrapper
wrapper = SankeyWrapper()

# Generate figure
fig_dict = wrapper.generate_sankey(
    clustering_data=mock_clustering_data,
    path_data=mock_clustering_data['paths'],
    cluster_labels=mock_clustering_data['cluster_labels'],
    window_config=None,
    config={'top_n': 10, 'color_by': 'cluster'}
)

# Print debug info
print("Generated figure dictionary:")
print(f"Keys: {list(fig_dict.keys())}")
print(f"Data type: {type(fig_dict.get('data'))}")
if 'data' in fig_dict and fig_dict['data']:
    print(f"Number of traces: {len(fig_dict['data'])}")
    if len(fig_dict['data']) > 0:
        trace = fig_dict['data'][0]
        print(f"Trace type: {trace.get('type')}")
        if 'node' in trace:
            print(f"Number of nodes: {len(trace['node'].get('label', []))}")
            print(f"Node labels: {trace['node'].get('label', [])}")
        if 'link' in trace:
            print(f"Number of links: {len(trace['link'].get('source', []))}")
            print(f"Links: source={trace['link'].get('source', [])}, target={trace['link'].get('target', [])}")

# Save to file for inspection
with open('debug_sankey_output.json', 'w') as f:
    json.dump(fig_dict, f, indent=2)
    
print("\nFigure saved to debug_sankey_output.json")

# Also test the empty figure
empty_fig = wrapper._create_empty_figure()
print(f"\nEmpty figure type: {type(empty_fig)}")
print(f"Empty figure keys: {list(empty_fig.keys())}")