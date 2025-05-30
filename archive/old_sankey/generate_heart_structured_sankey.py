#!/usr/bin/env python3
"""
Generate structured path Sankey diagram for heart disease model.
Uses Approach B: Shows archetypal paths as continuous flows with cluster labels.
"""

import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Define cluster labels
CLUSTER_LABELS = {
    0: {  # Layer 1
        0: 'High-Risk Older Males',
        1: 'Lower-Risk Younger'
    },
    1: {  # Layer 2
        0: 'Low CV Stress',
        1: 'Controlled High-Risk'
    },
    2: {  # Layer 3
        0: 'Stress-Induced Risk',
        1: 'Moderate-Risk Active'
    },
    3: {  # Output
        0: 'No Heart Disease',
        1: 'Heart Disease Present'
    }
}

# Define the 7 archetypal paths with statistics
ARCHETYPAL_PATHS = [
    {
        'name': 'Conservative Low-Risk Path',
        'path': [1, 0, 0, 0],  # L1C1 -> L2C0 -> L3C0 -> OutputC0
        'count': 117,
        'percentage': 43.3,
        'color': 'rgba(46, 204, 113, 0.7)'  # Green
    },
    {
        'name': 'Classic High-Risk Path',
        'path': [0, 1, 1, 1],  # L1C0 -> L2C1 -> L3C1 -> OutputC1
        'count': 95,
        'percentage': 35.2,
        'color': 'rgba(231, 76, 60, 0.7)'  # Red
    },
    {
        'name': 'Progressive Risk Path',
        'path': [1, 1, 1, 1],  # L1C1 -> L2C1 -> L3C1 -> OutputC1
        'count': 29,
        'percentage': 10.7,
        'color': 'rgba(230, 126, 34, 0.7)'  # Orange
    },
    {
        'name': 'Male-Biased Path',
        'path': [1, 0, 1, 1],  # L1C1 -> L2C0 -> L3C1 -> OutputC1
        'count': 18,
        'percentage': 6.7,
        'color': 'rgba(155, 89, 182, 0.7)'  # Purple
    },
    {
        'name': 'Misclassification Path',
        'path': [1, 0, 0, 1],  # L1C1 -> L2C0 -> L3C0 -> OutputC1
        'count': 6,
        'percentage': 2.2,
        'color': 'rgba(241, 196, 15, 0.7)'  # Yellow
    },
    {
        'name': 'Resilient High-Risk Path',
        'path': [0, 1, 1, 0],  # L1C0 -> L2C1 -> L3C1 -> OutputC0
        'count': 3,
        'percentage': 1.1,
        'color': 'rgba(52, 152, 219, 0.7)'  # Blue
    },
    {
        'name': 'Unexpected Recovery Path',
        'path': [0, 1, 0, 0],  # L1C0 -> L2C1 -> L3C0 -> OutputC0
        'count': 2,
        'percentage': 0.7,
        'color': 'rgba(149, 165, 166, 0.7)'  # Gray
    }
]

def create_structured_sankey():
    """Create structured path Sankey diagram with cluster labels."""
    
    # Create nodes - one for each path at each layer
    nodes = []
    node_positions_x = []
    node_positions_y = []
    node_colors = []
    node_labels = []
    
    # Layer positions
    layer_x = [0.1, 0.37, 0.63, 0.9]
    
    # Calculate vertical positions for paths with better spacing
    n_paths = len(ARCHETYPAL_PATHS)
    # Group top 3 paths closer, spread out the rest
    y_positions = []
    # Top 3 paths: positions 0.9, 0.8, 0.7
    y_positions.extend([0.9, 0.8, 0.7])
    # Remaining paths: spread from 0.5 to 0.1
    if n_paths > 3:
        remaining_positions = np.linspace(0.5, 0.1, n_paths - 3)
        y_positions.extend(remaining_positions)
    
    # Create nodes for each path at each layer
    node_map = {}
    node_idx = 0
    
    for layer_idx in range(4):
        for path_idx, path_info in enumerate(ARCHETYPAL_PATHS):
            node_key = f"L{layer_idx}_P{path_idx}"
            node_map[node_key] = node_idx
            
            # Get cluster for this path at this layer
            cluster_id = path_info['path'][layer_idx]
            cluster_label = CLUSTER_LABELS[layer_idx][cluster_id]
            
            # Add node
            nodes.append(f"{path_info['name']}")
            node_positions_x.append(layer_x[layer_idx])
            node_positions_y.append(y_positions[path_idx])
            node_colors.append(path_info['color'])
            node_labels.append(cluster_label)
            
            node_idx += 1
    
    # Create links between consecutive layers
    links = {
        'source': [],
        'target': [],
        'value': [],
        'color': [],
        'label': []
    }
    
    for layer_idx in range(3):  # 0, 1, 2
        for path_idx, path_info in enumerate(ARCHETYPAL_PATHS):
            source_key = f"L{layer_idx}_P{path_idx}"
            target_key = f"L{layer_idx + 1}_P{path_idx}"
            
            links['source'].append(node_map[source_key])
            links['target'].append(node_map[target_key])
            links['value'].append(path_info['count'])
            links['color'].append(path_info['color'])
            links['label'].append(f"{path_info['name']}: {path_info['count']} patients ({path_info['percentage']}%)")
    
    # Create Sankey figure with explicit domain
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        domain=dict(x=[0.05, 0.95], y=[0.15, 0.85]),  # Leave space for labels
        node=dict(
            pad=20,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            x=node_positions_x,
            y=node_positions_y,
            hovertemplate='%{label}<br>Cluster: %{customdata}<extra></extra>',
            customdata=node_labels
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color'],
            label=links['label']
        ),
        textfont=dict(size=10, color='black')
    )])
    
    # Add annotations for cluster labels
    annotations = []
    
    # Define Sankey domain bounds
    domain_x_min, domain_x_max = 0.05, 0.95
    domain_y_min, domain_y_max = 0.15, 0.85
    
    # Layer headers (above the plot)
    layer_names = ['Initial Risk', 'CV Health', 'Risk Profile', 'Diagnosis']
    for i, (x, name) in enumerate(zip(layer_x, layer_names)):
        # Transform x coordinate to paper space
        paper_x = domain_x_min + x * (domain_x_max - domain_x_min)
        annotations.append(dict(
            x=paper_x, 
            y=domain_y_max + 0.05,  # Just above the plot
            text=f"<b>{name}</b>",
            showarrow=False,
            xanchor='center',
            font=dict(size=14),
            xref='paper',
            yref='paper'
        ))
    
    # Add cluster labels based on actual node positions
    # Transform coordinates from Sankey space to paper space
    for layer_idx in range(4):
        x = layer_x[layer_idx]
        
        # Group paths by their cluster at this layer
        cluster_groups = {}
        for path_idx, path_info in enumerate(ARCHETYPAL_PATHS):
            cluster_id = path_info['path'][layer_idx]
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            # Store the actual y position for this path
            cluster_groups[cluster_id].append(y_positions[path_idx])
        
        # Add label for each cluster group
        for cluster_id, path_y_positions in cluster_groups.items():
            # Use the actual positions of paths in this cluster
            if len(path_y_positions) == 1:
                label_y = path_y_positions[0]
            else:
                # For multiple paths, position label at the midpoint
                label_y = (min(path_y_positions) + max(path_y_positions)) / 2
            
            # Transform coordinates to paper space
            paper_x = domain_x_min + x * (domain_x_max - domain_x_min)
            paper_y = domain_y_min + label_y * (domain_y_max - domain_y_min)
            
            cluster_label = CLUSTER_LABELS[layer_idx][cluster_id]
            
            # Add cluster label annotation
            annotations.append(dict(
                x=paper_x,
                y=paper_y,
                text=f"<b>{cluster_label}</b>",
                showarrow=False,
                xanchor='center',
                yanchor='middle',
                font=dict(size=12, color='black', family='Arial, sans-serif'),
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                xref='paper',
                yref='paper'
            ))
    
    # Add title and layout
    fig.update_layout(
        title={
            'text': "Heart Disease Model: Archetypal Patient Pathways",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1200,
        width=1400,
        margin=dict(l=100, r=350, t=150, b=300),
        plot_bgcolor='white',
        font=dict(size=12),
        annotations=annotations
    )
    
    # Add legend for paths
    legend_text = "<b>Path Distribution:</b><br>"
    for path_info in ARCHETYPAL_PATHS:
        legend_text += f"{path_info['name']}: {path_info['count']} ({path_info['percentage']}%)<br>"
    
    fig.add_annotation(
        x=1.02,
        y=0.5,
        text=legend_text,
        showarrow=False,
        xanchor='left',
        yanchor='middle',
        font=dict(size=10),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=1,
        xref='paper',
        yref='paper'
    )
    
    return fig

def main():
    """Generate and save the structured sankey diagram."""
    print("Generating structured path Sankey diagram for heart disease...")
    
    # Create the diagram
    fig = create_structured_sankey()
    
    # Save to arxiv figures
    arxiv_dir = Path(__file__).parent.parent.parent / "arxiv_submission" / "figures"
    
    # Save as HTML
    html_path = arxiv_dir / "heart_sankey.html"
    fig.write_html(str(html_path))
    print(f"Saved HTML to: {html_path}")
    
    # Save as PNG
    try:
        png_path = arxiv_dir / "heart_sankey.png"
        fig.write_image(str(png_path), width=1400, height=900, scale=2)
        print(f"Saved PNG to: {png_path}")
    except Exception as e:
        print(f"Could not save PNG: {e}")
        print("Make sure kaleido is installed: pip install kaleido")

if __name__ == "__main__":
    main()