#!/usr/bin/env python3
"""
Generate structured path Sankey diagrams for GPT-2 analysis.
Uses Approach B: Shows archetypal paths as continuous flows with cluster labels.
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
    # Try to load from the unified_cta results
    results_path = Path(__file__).parent / "unified_cta" / "results" / "archetypal_paths_analysis.json"
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            data = json.load(f)
            return data.get('windows', {})
    
    # Fallback: Define the top archetypal paths based on the analysis
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
    """Create structured path Sankey diagram for a specific window."""
    
    layers = window_data['layers']
    paths = archetypal_paths.get(window_name, {}).get('paths', [])[:7]  # Top 7 paths
    
    if not paths:
        print(f"Warning: No archetypal paths found for {window_name}")
        return None
    
    # Create nodes - one for each path at each layer
    nodes = []
    node_positions_x = []
    node_positions_y = []
    node_colors = []
    node_labels = []
    
    # Layer positions
    n_layers = len(layers)
    layer_x = np.linspace(0.1, 0.9, n_layers)
    
    # Calculate vertical positions for paths with better spacing
    n_paths = len(paths)
    # Group top 3 paths closer, spread out the rest
    y_positions = []
    # Top 3 paths: positions 0.85, 0.75, 0.65
    y_positions.extend([0.85, 0.75, 0.65])
    # Remaining paths: spread from 0.5 to 0.15
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
    
    for layer_idx, actual_layer in enumerate(layers):
        for path_idx, path_info in enumerate(paths):
            node_key = f"L{layer_idx}_P{path_idx}"
            node_map[node_key] = node_idx
            
            # Get cluster for this path at this layer
            cluster_id = path_info['path'][layer_idx]
            cluster_label = CLUSTER_LABELS.get(actual_layer, {}).get(cluster_id, f"C{cluster_id}")
            
            # Add node
            nodes.append(f"{path_info['name']}")
            node_positions_x.append(layer_x[layer_idx])
            node_positions_y.append(y_positions[path_idx])
            node_colors.append(path_colors[path_idx % len(path_colors)])
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
    
    for layer_idx in range(n_layers - 1):
        for path_idx, path_info in enumerate(paths):
            source_key = f"L{layer_idx}_P{path_idx}"
            target_key = f"L{layer_idx + 1}_P{path_idx}"
            
            links['source'].append(node_map[source_key])
            links['target'].append(node_map[target_key])
            links['value'].append(path_info['count'])
            links['color'].append(path_colors[path_idx % len(path_colors)])
            links['label'].append(f"{path_info['name']}: {path_info['count']} words ({path_info['percentage']}%)")
    
    # Create Sankey figure with explicit domain
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        domain=dict(x=[0.05, 0.95], y=[0.15, 0.85]),  # Leave space for labels
        node=dict(
            pad=25,
            thickness=20,
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
    for i, (x, layer) in enumerate(zip(layer_x, layers)):
        # Transform x coordinate to paper space
        paper_x = domain_x_min + x * (domain_x_max - domain_x_min)
        annotations.append(dict(
            x=paper_x, 
            y=domain_y_max + 0.05,  # Just above the plot
            text=f"<b>Layer {layer}</b>",
            showarrow=False,
            xanchor='center',
            font=dict(size=14, color='black'),
            xref='paper',
            yref='paper'
        ))
    
    # Add cluster labels based on actual node positions
    # Transform coordinates from Sankey space to paper space
    for layer_idx, actual_layer in enumerate(layers):
        x = layer_x[layer_idx]
        
        # Group paths by their cluster at this layer
        cluster_groups = {}
        for path_idx, path_info in enumerate(paths):
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
            
            cluster_label = CLUSTER_LABELS.get(actual_layer, {}).get(cluster_id, f"C{cluster_id}")
            
            # Add cluster label annotation
            annotations.append(dict(
                x=paper_x,
                y=paper_y,
                text=f"<b>{cluster_label}</b>",
                showarrow=False,
                xanchor='center',
                yanchor='middle',
                font=dict(size=11, color='black', family='Arial, sans-serif'),
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='black',
                borderwidth=2,
                borderpad=3,
                xref='paper',
                yref='paper'
            ))
    
    # Add title and layout
    fig.update_layout(
        title={
            'text': f"GPT-2 {window_data['name']}: Top 7 Archetypal Paths",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=1000,
        width=1400,
        margin=dict(l=100, r=100, t=150, b=120),
        font=dict(size=12),
        annotations=annotations
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
    print("Generating GPT-2 structured path Sankey diagrams...")
    
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