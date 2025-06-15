#!/usr/bin/env python
"""Debug why visualizations show white/blank."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from concept_mri.components.visualizations.sankey_wrapper import SankeyWrapper
import plotly.graph_objects as go
import json

# Create mock data
mock_clustering_data = {
    'completed': True,
    'paths': {
        '0': ['L0_C0', 'L1_C1', 'L2_C2'],
        '1': ['L0_C0', 'L1_C2', 'L2_C1']
    },
    'cluster_labels': {
        'L0_C0': 'Input',
        'L1_C1': 'Hidden 1',
        'L1_C2': 'Hidden 2', 
        'L2_C1': 'Output 1',
        'L2_C2': 'Output 2'
    }
}

# Test 1: Direct figure generation
wrapper = SankeyWrapper()
fig_dict = wrapper.generate_sankey(
    clustering_data=mock_clustering_data,
    path_data=mock_clustering_data['paths'],
    cluster_labels=mock_clustering_data['cluster_labels'],
    window_config=None,
    config={'top_n': 10}
)

print("Test 1 - Direct generation:")
print(f"Type: {type(fig_dict)}")
print(f"Is dict: {isinstance(fig_dict, dict)}")
print(f"Has data key: {'data' in fig_dict}")

# Test 2: What happens in the callback
from dash import dcc

# Simulate what happens in the callback
graph = dcc.Graph(
    figure=fig_dict,
    id="test-sankey-graph",
    config={'displayModeBar': False}
)

print("\nTest 2 - Graph component:")
print(f"Graph type: {type(graph)}")
print(f"Graph figure type: {type(graph.figure)}")

# Test 3: Check if it's a serialization issue
try:
    json_str = json.dumps(fig_dict)
    print("\nTest 3 - JSON serialization: SUCCESS")
except Exception as e:
    print(f"\nTest 3 - JSON serialization: FAILED - {e}")

# Test 4: Check the actual figure structure
if isinstance(fig_dict, dict) and 'data' in fig_dict:
    print("\nTest 4 - Figure structure:")
    print(f"Number of traces: {len(fig_dict['data'])}")
    if fig_dict['data']:
        trace = fig_dict['data'][0]
        print(f"Trace type: {trace.get('type')}")
        if 'node' in trace:
            print(f"Has nodes: {len(trace['node'].get('label', []))}")
        if 'link' in trace:
            print(f"Has links: {len(trace['link'].get('source', []))}")

# Test 5: Create a simple test figure to verify Plotly works
simple_fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
simple_dict = simple_fig.to_dict()
print("\nTest 5 - Simple figure:")
print(f"Type: {type(simple_dict)}")
print(f"Has data: {'data' in simple_dict}")