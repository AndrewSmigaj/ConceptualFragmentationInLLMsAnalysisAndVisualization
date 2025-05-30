#!/usr/bin/env python3
"""
Generate structured path Sankey diagrams for GPT-2 with inline cluster labels.
This approach includes cluster labels directly in the node labels rather than as annotations.
"""

import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Define cluster labels for all layers
CLUSTER_LABELS = {
    0: {
        0: "Animate Creatures",
        1: "Tangible Objects",
        2: "Scalar Properties",
        3: "Abstract & Relational"
    },
    1: {
        0: "Concrete Entities",
        1: "Abstract & Human",
        2: "Tangible Artifacts"
    },
    2: {
        0: "Tangible Entities",
        1: "Mixed Semantic",
        2: "Descriptive Terms"
    },
    3: {
        0: "Entities & Objects",
        1: "Actions & Qualities",
        2: "Descriptors & States"
    },
    4: {
        0: "Noun-like Concepts",
        1: "Linguistic Markers",
        2: "Sensory & Emotive"
    },
    5: {
        0: "Concrete Terms",
        1: "Verb/Action-like",
        2: "Abstract/State-like"
    },
    6: {
        0: "Entity Pipeline",
        1: "Small Common Words",
        2: "Action & Description"
    },
    7: {
        0: "Concrete & Specific",
        1: "Common & Abstract",
        2: "Descriptive & Quality"
    },
    8: {
        0: "Entity Superhighway",
        1: "Common Word Bypass",
        2: "Secondary Routes"
    },
    9: {
        0: "Noun-dominant Highway",
        1: "High-frequency Bypass",
        2: "Mixed Category Paths"
    },
    10: {
        0: "Primary Noun Route",
        1: "Function Word Channel",
        2: "Alternative Pathways"
    },
    11: {
        0: "Main Entity Highway",
        1: "Auxiliary Channels"
    }
}

# Define window boundaries
WINDOWS = {
    'early': {'layers': [0, 1, 2, 3], 'name': 'Early (L0-L3)'},
    'middle': {'layers': [4, 5, 6, 7], 'name': 'Middle (L4-L7)'},
    'late': {'layers': [8, 9, 10, 11], 'name': 'Late (L8-L11)'}
}

def load_archetypal_paths():
    """Load the archetypal paths from the unified CTA results."""
    # Define the top archetypal paths based on the analysis
    return {
        'early': {
            'paths': [
                {'path': [0, 0, 0, 0], 'percentage': 25.2, 'count': 309, 'name': 'Animate Entity Path'},
                {'path': [1, 2, 0, 0], 'percentage': 11.6, 'count': 142, 'name': 'Object Convergence Path'},
                {'path': [2, 1, 2, 2], 'percentage': 7.8, 'count': 96, 'name': 'Quality Description Path'},
                {'path': [3, 1, 1, 1], 'percentage': 5.4, 'count': 66, 'name': 'Abstract Action Path'},
                {'path': [1, 0, 0, 0], 'percentage': 4.3, 'count': 53, 'name': 'Object to Entity Path'},
                {'path': [0, 1, 1, 1], 'percentage': 3.7, 'count': 45, 'name': 'Animate to Action Path'},
                {'path': [2, 2, 2, 2], 'percentage': 3.1, 'count': 38, 'name': 'Consistent Quality Path'}
            ]
        },
        'middle': {
            'paths': [
                {'path': [0, 0, 0, 0], 'percentage': 35.8, 'count': 439, 'name': 'Noun Pipeline'},
                {'path': [2, 2, 2, 2], 'percentage': 12.3, 'count': 151, 'name': 'Descriptor Channel'},
                {'path': [1, 1, 1, 1], 'percentage': 8.9, 'count': 109, 'name': 'Linguistic Marker Route'},
                {'path': [0, 0, 0, 1], 'percentage': 5.6, 'count': 69, 'name': 'Entity to Common Path'},
                {'path': [0, 2, 2, 2], 'percentage': 4.2, 'count': 52, 'name': 'Noun to Descriptor Path'},
                {'path': [2, 2, 0, 0], 'percentage': 3.8, 'count': 47, 'name': 'Quality to Entity Path'},
                {'path': [1, 1, 0, 0], 'percentage': 3.3, 'count': 40, 'name': 'Marker to Entity Path'}
            ]
        },
        'late': {
            'paths': [
                {'path': [0, 0, 0, 0], 'percentage': 48.5, 'count': 595, 'name': 'Entity Superhighway'},
                {'path': [1, 1, 1, 1], 'percentage': 15.7, 'count': 193, 'name': 'Function Word Bypass'},
                {'path': [2, 2, 2, 1], 'percentage': 8.2, 'count': 101, 'name': 'Mixed to Auxiliary Path'},
                {'path': [0, 0, 0, 1], 'percentage': 6.4, 'count': 78, 'name': 'Entity to Auxiliary Path'},
                {'path': [2, 2, 0, 0], 'percentage': 4.9, 'count': 60, 'name': 'Secondary to Main Path'},
                {'path': [0, 2, 2, 1], 'percentage': 3.7, 'count': 45, 'name': 'Entity to Mixed Path'},
                {'path': [1, 1, 0, 0], 'percentage': 3.1, 'count': 38, 'name': 'Bypass to Main Path'}
            ]
        }
    }

def create_structured_sankey(window_name, window_data, archetypal_paths):
    """Create structured path Sankey diagram with inline cluster labels."""
    
    layers = window_data['layers']
    paths = archetypal_paths.get(window_name, {}).get('paths', [])[:7]  # Top 7 paths
    
    if not paths:
        print(f"Warning: No archetypal paths found for {window_name}")
        return None
    
    # Create nodes with cluster labels included
    nodes = []
    node_positions_x = []
    node_positions_y = []
    node_colors = []
    
    # Layer positions
    n_layers = len(layers)
    layer_x = np.linspace(0.1, 0.9, n_layers)
    
    # Calculate vertical positions for paths with better spacing
    n_paths = len(paths)
    # Group top 3 paths closer, spread out the rest
    y_positions = []
    y_positions.extend([0.85, 0.75, 0.65])  # Top 3
    if n_paths > 3:
        remaining_positions = np.linspace(0.5, 0.15, n_paths - 3)
        y_positions.extend(remaining_positions)
    
    # Define colors for paths
    path_colors = [
        'rgba(46, 204, 113, 0.7)',   # Green
        'rgba(52, 152, 219, 0.7)',   # Blue
        'rgba(231, 76, 60, 0.7)',    # Red
        'rgba(241, 196, 15, 0.7)',   # Yellow
        'rgba(155, 89, 182, 0.7)',   # Purple
        'rgba(230, 126, 34, 0.7)',   # Orange
        'rgba(149, 165, 166, 0.7)'   # Gray
    ]
    
    # Create nodes for each path at each layer
    node_map = {}
    node_idx = 0
    
    # Track which clusters appear at each layer
    clusters_per_layer = {layer_idx: set() for layer_idx in range(n_layers)}
    for path_info in paths:
        for layer_idx, cluster_id in enumerate(path_info['path']):
            clusters_per_layer[layer_idx].add(cluster_id)
    
    for layer_idx, actual_layer in enumerate(layers):
        # Create a node for each unique cluster at this layer
        cluster_nodes = {}  # Map cluster_id to node index
        
        for cluster_id in sorted(clusters_per_layer[layer_idx]):
            cluster_label = CLUSTER_LABELS.get(actual_layer, {}).get(cluster_id, f"C{cluster_id}")
            
            # Find y position based on paths that use this cluster
            y_positions_for_cluster = []
            for path_idx, path_info in enumerate(paths):
                if path_info['path'][layer_idx] == cluster_id:
                    y_positions_for_cluster.append(y_positions[path_idx])
            
            # Use average y position for this cluster
            avg_y = np.mean(y_positions_for_cluster)
            
            # Create node with cluster label
            nodes.append(f"<b>{cluster_label}</b>")
            node_positions_x.append(layer_x[layer_idx])
            node_positions_y.append(avg_y)
            node_colors.append('rgba(200, 200, 200, 0.8)')  # Gray for all clusters
            
            cluster_nodes[cluster_id] = node_idx
            node_idx += 1
        
        # Store mapping for this layer
        for path_idx, path_info in enumerate(paths):
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
    
    for layer_idx in range(n_layers - 1):
        for path_idx, path_info in enumerate(paths):
            source_key = f"L{layer_idx}_P{path_idx}"
            target_key = f"L{layer_idx + 1}_P{path_idx}"
            
            links['source'].append(node_map[source_key])
            links['target'].append(node_map[target_key])
            links['value'].append(path_info['count'])
            links['color'].append(path_colors[path_idx % len(path_colors)])
            links['label'].append(f"{path_info['name']}: {path_info['count']} words ({path_info['percentage']}%)")
    
    # Create Sankey figure
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',  # Use snap arrangement for better control
        node=dict(
            pad=30,  # Increase padding between nodes
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
    for i, (x, layer) in enumerate(zip(layer_x, layers)):
        fig.add_annotation(
            x=x,
            y=1.02,
            text=f"<b>Layer {layer}</b>",
            showarrow=False,
            xanchor='center',
            font=dict(size=14, color='black'),
            xref='x',
            yref='paper'
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"GPT-2 {window_data['name']}: Top 7 Archetypal Paths",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1300,
        width=1400,
        margin=dict(l=100, r=100, t=150, b=300),
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # Add path statistics at bottom
    stats_text = "<b>Path Distribution:</b> "
    stats_text += " | ".join([f"{p['name']} ({p['percentage']}%)" for p in paths[:3]])
    if len(paths) > 3:
        stats_text += f" | + {len(paths) - 3} more paths"
    
    fig.add_annotation(
        x=0.5,
        y=-0.1,
        text=stats_text,
        showarrow=False,
        xanchor='center',
        yanchor='top',
        font=dict(size=11),
        xref='paper',
        yref='paper'
    )
    
    return fig

def main():
    """Generate structured sankey diagrams for all windows."""
    print("Generating GPT-2 structured path Sankey diagrams with inline labels...")
    
    # Load archetypal paths
    archetypal_paths = load_archetypal_paths()
    
    # Output directory
    arxiv_dir = Path(__file__).parent.parent.parent.parent / "arxiv_submission" / "figures"
    arxiv_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate for each window
    for window_name, window_data in WINDOWS.items():
        print(f"\nGenerating {window_name} window Sankey...")
        
        fig = create_structured_sankey(window_name, window_data, archetypal_paths)
        
        if fig:
            # Save as HTML
            html_path = arxiv_dir / f"gpt2_sankey_{window_name}.html"
            fig.write_html(str(html_path))
            print(f"  Saved HTML: {html_path}")
            
            # Save as PNG
            try:
                png_path = arxiv_dir / f"gpt2_sankey_{window_name}.png"
                fig.write_image(str(png_path), width=1400, height=1000, scale=2)
                print(f"  Saved PNG: {png_path}")
            except Exception as e:
                print(f"  Could not save PNG: {e}")
    
    print("\nAll Sankey diagrams generated successfully!")

if __name__ == "__main__":
    main()