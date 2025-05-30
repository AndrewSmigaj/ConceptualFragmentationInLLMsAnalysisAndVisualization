#!/usr/bin/env python3
"""
Test script to understand Sankey coordinate systems and proper label placement.
"""

import plotly.graph_objects as go
import numpy as np

# Simple test with 3 paths through 3 layers
paths = [
    {'name': 'Path A', 'clusters': [0, 0, 0], 'count': 100},
    {'name': 'Path B', 'clusters': [0, 1, 1], 'count': 50},
    {'name': 'Path C', 'clusters': [1, 1, 1], 'count': 30}
]

# Layer positions
layer_x = [0.1, 0.5, 0.9]
y_positions = [0.8, 0.5, 0.2]  # Vertical positions for the 3 paths

# Create nodes
nodes = []
node_x = []
node_y = []
node_idx = 0
node_map = {}

for layer_idx in range(3):
    for path_idx, path in enumerate(paths):
        key = f"L{layer_idx}_P{path_idx}"
        node_map[key] = node_idx
        nodes.append(f"{path['name']} L{layer_idx}")
        node_x.append(layer_x[layer_idx])
        node_y.append(y_positions[path_idx])
        node_idx += 1

# Create links
links = {'source': [], 'target': [], 'value': []}
for layer_idx in range(2):
    for path_idx, path in enumerate(paths):
        source_key = f"L{layer_idx}_P{path_idx}"
        target_key = f"L{layer_idx + 1}_P{path_idx}"
        links['source'].append(node_map[source_key])
        links['target'].append(node_map[target_key])
        links['value'].append(path['count'])

# Create figure with Sankey
fig = go.Figure(data=[go.Sankey(
    arrangement='fixed',
    node=dict(
        pad=20,
        thickness=20,
        label=nodes,
        x=node_x,
        y=node_y
    ),
    link=dict(
        source=links['source'],
        target=links['target'],
        value=links['value']
    )
)])

# Try different annotation positioning approaches
annotations = []

# Approach 1: Direct y-coordinate mapping (what we were doing)
for layer_idx in range(3):
    x = layer_x[layer_idx]
    
    # Find clusters at this layer
    clusters_at_layer = {}
    for path_idx, path in enumerate(paths):
        cluster = path['clusters'][layer_idx]
        if cluster not in clusters_at_layer:
            clusters_at_layer[cluster] = []
        clusters_at_layer[cluster].append(y_positions[path_idx])
    
    # Add labels
    for cluster_id, cluster_y_positions in clusters_at_layer.items():
        y = np.mean(cluster_y_positions)
        
        annotations.append(dict(
            x=x,
            y=y,
            text=f"<b>Cluster {cluster_id}</b>",
            showarrow=False,
            xanchor='center',
            yanchor='middle',
            bgcolor='yellow',
            bordercolor='red',
            borderwidth=2
        ))

# Also add reference markers at exact node positions
for i, (x, y) in enumerate(zip(node_x, node_y)):
    annotations.append(dict(
        x=x,
        y=y,
        text=f"•",
        showarrow=False,
        xanchor='center',
        yanchor='middle',
        font=dict(size=20, color='red')
    ))

fig.update_layout(
    title="Sankey Coordinate Test",
    annotations=annotations,
    height=600,
    width=1000,
    showlegend=False
)

# Save to see the result
fig.write_html("test_sankey_coordinates.html")
print("Saved test_sankey_coordinates.html")

# Now let's try a different approach - using domain-relative coordinates
fig2 = go.Figure(data=[go.Sankey(
    arrangement='fixed',
    domain=dict(x=[0.1, 0.9], y=[0.1, 0.9]),  # Explicitly set the domain
    node=dict(
        pad=20,
        thickness=20,
        label=nodes,
        x=node_x,
        y=node_y
    ),
    link=dict(
        source=links['source'],
        target=links['target'],
        value=links['value']
    )
)])

# Adjust annotations for domain
domain_annotations = []
for ann in annotations:
    # Scale annotation positions to match domain
    new_ann = ann.copy()
    new_ann['x'] = 0.1 + ann['x'] * 0.8  # Map to domain x range
    new_ann['y'] = 0.1 + ann['y'] * 0.8  # Map to domain y range
    new_ann['xref'] = 'paper'
    new_ann['yref'] = 'paper'
    domain_annotations.append(new_ann)

fig2.update_layout(
    title="Sankey with Domain-Adjusted Coordinates",
    annotations=domain_annotations,
    height=600,
    width=1000
)

fig2.write_html("test_sankey_coordinates_domain.html")
print("Saved test_sankey_coordinates_domain.html")