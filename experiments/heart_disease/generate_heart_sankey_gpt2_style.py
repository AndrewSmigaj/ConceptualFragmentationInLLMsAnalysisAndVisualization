#!/usr/bin/env python3
"""
Generate heart disease Sankey diagram in GPT-2 style.
Uses inline cluster labels and structured path approach.
"""

import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Define cluster labels based on the analysis
CLUSTER_LABELS = {
    'L1C0': 'High-Risk Older Males',
    'L1C1': 'Lower-Risk Younger',
    'L2C0': 'Low CV Stress',
    'L2C1': 'Controlled High-Risk',
    'L3C0': 'Stress-Induced Risk',
    'L3C1': 'Moderate-Risk Active',
    'OutputC0': 'No Heart Disease',
    'OutputC1': 'Heart Disease Present'
}

# Define the 5 archetypal paths with their statistics
PATHS = [
    {
        'path': ['L1C1', 'L2C0', 'L3C0', 'OutputC0'],
        'count': 117,  # 43.3% of 270
        'percentage': 43.3,
        'name': 'Conservative Low-Risk',
        'color': 'rgba(46, 204, 113, 0.7)'  # Green
    },
    {
        'path': ['L1C0', 'L2C1', 'L3C1', 'OutputC1'],
        'count': 95,   # 35.2% of 270
        'percentage': 35.2,
        'name': 'Classic High-Risk',
        'color': 'rgba(231, 76, 60, 0.7)'   # Red
    },
    {
        'path': ['L1C1', 'L2C1', 'L3C1', 'OutputC1'],
        'count': 29,   # 10.7% of 270
        'percentage': 10.7,
        'name': 'Progressive Risk',
        'color': 'rgba(230, 126, 34, 0.7)'  # Orange
    },
    {
        'path': ['L1C1', 'L2C0', 'L3C1', 'OutputC1'],
        'count': 18,   # 6.7% of 270
        'percentage': 6.7,
        'name': 'Male-Biased',
        'color': 'rgba(155, 89, 182, 0.7)'  # Purple
    },
    {
        'path': ['L1C1', 'L2C0', 'L3C0', 'OutputC1'],
        'count': 6,    # 2.2% of 270
        'percentage': 2.2,
        'name': 'Misclassification',
        'color': 'rgba(241, 196, 15, 0.7)'  # Yellow
    }
]

def create_structured_sankey():
    """Create structured path Sankey diagram with inline cluster labels."""
    
    # Create nodes with cluster labels included
    nodes = []
    node_positions_x = []
    node_positions_y = []
    node_colors = []
    
    # Layer positions
    layers = ['L1', 'L2', 'L3', 'Output']
    layer_x = [0.1, 0.37, 0.63, 0.9]
    
    # Calculate vertical positions for paths
    n_paths = len(PATHS)
    y_positions = [0.85, 0.70, 0.50, 0.30, 0.15][:n_paths]
    
    # Create nodes for each unique cluster at each layer
    node_map = {}
    node_idx = 0
    
    # Track which clusters appear at each layer
    layer_clusters = {
        'L1': ['L1C0', 'L1C1'],
        'L2': ['L2C0', 'L2C1'],
        'L3': ['L3C0', 'L3C1'],
        'Output': ['OutputC0', 'OutputC1']
    }
    
    for layer_idx, layer in enumerate(layers):
        # Get unique clusters at this layer from paths
        clusters_at_layer = set()
        for path_info in PATHS:
            if layer_idx < len(path_info['path']):
                clusters_at_layer.add(path_info['path'][layer_idx])
        
        # Create a node for each unique cluster at this layer
        cluster_nodes = {}  # Map cluster_id to node index
        
        for cluster_id in sorted(clusters_at_layer):
            cluster_label = CLUSTER_LABELS.get(cluster_id, cluster_id)
            
            # Find y position based on paths that use this cluster
            y_positions_for_cluster = []
            for path_idx, path_info in enumerate(PATHS):
                if layer_idx < len(path_info['path']) and path_info['path'][layer_idx] == cluster_id:
                    y_positions_for_cluster.append(y_positions[path_idx])
            
            # Use average y position for this cluster
            avg_y = np.mean(y_positions_for_cluster) if y_positions_for_cluster else 0.5
            
            # Create node with cluster label
            nodes.append(f"<b>{cluster_label}</b>")
            node_positions_x.append(layer_x[layer_idx])
            node_positions_y.append(avg_y)
            
            # Color nodes based on layer type
            if layer == 'Output':
                if 'No' in cluster_label:
                    node_colors.append('rgba(46, 204, 113, 0.8)')  # Green for no disease
                else:
                    node_colors.append('rgba(231, 76, 60, 0.8)')   # Red for disease
            else:
                node_colors.append('rgba(200, 200, 200, 0.8)')  # Gray for intermediate
            
            cluster_nodes[cluster_id] = node_idx
            node_idx += 1
        
        # Store mapping for this layer
        for path_idx, path_info in enumerate(PATHS):
            if layer_idx < len(path_info['path']):
                cluster_id = path_info['path'][layer_idx]
                node_key = f"L{layer_idx}_P{path_idx}"
                node_map[node_key] = cluster_nodes[cluster_id]
    
    # Create links between consecutive layers
    links = {
        'source': [],
        'target': [],
        'value': [],
        'color': [],
        'label': []
    }
    
    for layer_idx in range(len(layers) - 1):
        for path_idx, path_info in enumerate(PATHS):
            source_key = f"L{layer_idx}_P{path_idx}"
            target_key = f"L{layer_idx + 1}_P{path_idx}"
            
            if source_key in node_map and target_key in node_map:
                links['source'].append(node_map[source_key])
                links['target'].append(node_map[target_key])
                links['value'].append(path_info['count'])
                links['color'].append(path_info['color'])
                links['label'].append(f"{path_info['name']}: {path_info['count']} patients ({path_info['percentage']}%)")
    
    # Create Sankey figure
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',  # Use snap arrangement for better control
        node=dict(
            pad=30,  # Padding between nodes
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            x=node_positions_x,
            y=node_positions_y
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color'],
            label=links['label']
        ),
        textfont=dict(size=11, color='black', family='Arial')
    )])
    
    # Add layer headers
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Output']
    layer_subtitles = ['Initial Risk', 'CV Health', 'Risk Profile', 'Diagnosis']
    
    for i, (x, name, subtitle) in enumerate(zip(layer_x, layer_names, layer_subtitles)):
        fig.add_annotation(
            x=x,
            y=1.02,
            text=f"<b>{name}</b><br>{subtitle}",
            showarrow=False,
            xanchor='center',
            font=dict(size=14, color='black'),
            xref='x',
            yref='paper'
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Heart Disease Model: Patient Flow Through Risk Stratification",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1000,
        width=1400,
        margin=dict(l=100, r=100, t=150, b=200),
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add path statistics at bottom
    stats_text = "<b>Path Distribution:</b> "
    stats_text += " | ".join([f"{p['name']} ({p['percentage']}%)" for p in PATHS[:3]])
    stats_text += f" | + {len(PATHS) - 3} more paths"
    
    fig.add_annotation(
        x=0.5,
        y=-0.08,
        text=stats_text,
        showarrow=False,
        xanchor='center',
        yanchor='top',
        font=dict(size=11),
        xref='paper',
        yref='paper'
    )
    
    # Add key finding
    fig.add_annotation(
        x=0.5,
        y=-0.12,
        text="<b>Key Finding:</b> Path 4 shows 83.3% male composition revealing gender bias in predictions",
        showarrow=False,
        xanchor='center',
        yanchor='top',
        font=dict(size=10, color='purple'),
        xref='paper',
        yref='paper'
    )
    
    return fig

def main():
    """Generate and save the heart disease Sankey diagram in GPT-2 style."""
    print("Generating heart disease Sankey diagram in GPT-2 style...")
    
    # Create the diagram
    fig = create_structured_sankey()
    
    # Save outputs
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save HTML
    html_path = output_dir / "heart_sankey_gpt2_style.html"
    fig.write_html(str(html_path))
    print(f"Saved interactive HTML to: {html_path}")
    
    # Save to arxiv figures directory
    arxiv_dir = Path(__file__).parent.parent.parent / "arxiv_submission" / "figures"
    arxiv_dir.mkdir(exist_ok=True, parents=True)
    
    arxiv_html = arxiv_dir / "heart_sankey.html"
    fig.write_html(str(arxiv_html))
    print(f"Saved arxiv HTML to: {arxiv_html}")
    
    # Save static image
    try:
        png_path = arxiv_dir / "heart_sankey.png"
        fig.write_image(str(png_path), width=1400, height=1000, scale=2)
        print(f"Saved static PNG to: {png_path}")
    except Exception as e:
        print(f"Could not save PNG: {e}")
        print("Note: Install kaleido to generate static images: pip install kaleido")

if __name__ == "__main__":
    main()