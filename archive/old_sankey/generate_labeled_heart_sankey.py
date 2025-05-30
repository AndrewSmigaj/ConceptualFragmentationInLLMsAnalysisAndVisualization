#!/usr/bin/env python3
"""
Generate labeled Sankey diagram for heart disease model visualization.
Shows patient flow through risk stratification layers with meaningful cluster labels.
"""

import json
import plotly.graph_objects as go
from pathlib import Path

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
        'label': 'Conservative Low-Risk Path',
        'color': 'rgba(46, 204, 113, 0.8)'  # Green - mostly correct no disease
    },
    {
        'path': ['L1C0', 'L2C1', 'L3C1', 'OutputC1'],
        'count': 95,   # 35.2% of 270
        'label': 'Classic High-Risk Path',
        'color': 'rgba(231, 76, 60, 0.8)'   # Red - heart disease
    },
    {
        'path': ['L1C1', 'L2C1', 'L3C1', 'OutputC1'],
        'count': 29,   # 10.7% of 270
        'label': 'Progressive Risk Path',
        'color': 'rgba(230, 126, 34, 0.8)'  # Orange - evolving risk
    },
    {
        'path': ['L1C1', 'L2C0', 'L3C1', 'OutputC1'],
        'count': 18,   # 6.7% of 270
        'label': 'Male-Biased Path',
        'color': 'rgba(155, 89, 182, 0.8)'  # Purple - bias indicator
    },
    {
        'path': ['L1C1', 'L2C0', 'L3C0', 'OutputC1'],
        'count': 6,    # 2.2% of 270
        'label': 'Misclassification Path',
        'color': 'rgba(241, 196, 15, 0.8)'  # Yellow - errors
    }
]

def create_sankey_data():
    """Create nodes and links for Sankey diagram."""
    # Create unique nodes with labels
    nodes = []
    node_dict = {}
    node_idx = 0
    
    # Add nodes in layer order
    layers = [
        ['L1C0', 'L1C1'],
        ['L2C0', 'L2C1'],
        ['L3C0', 'L3C1'],
        ['OutputC0', 'OutputC1']
    ]
    
    for layer_nodes in layers:
        for node_id in layer_nodes:
            if node_id not in node_dict:
                nodes.append(CLUSTER_LABELS[node_id])
                node_dict[node_id] = node_idx
                node_idx += 1
    
    # Create links from paths
    links = {
        'source': [],
        'target': [],
        'value': [],
        'label': [],
        'color': []
    }
    
    # Process each path
    for path_info in PATHS:
        path = path_info['path']
        for i in range(len(path) - 1):
            source_node = path[i]
            target_node = path[i + 1]
            
            links['source'].append(node_dict[source_node])
            links['target'].append(node_dict[target_node])
            links['value'].append(path_info['count'])
            links['label'].append(f"{path_info['label']}: {path_info['count']} patients")
            links['color'].append(path_info['color'])
    
    return nodes, links

def create_sankey_diagram():
    """Create the Sankey diagram with labels."""
    nodes, links = create_sankey_data()
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',  # Force fixed positioning
        node=dict(
            pad=50,  # Much larger padding to force separation
            thickness=10,  # Thinner nodes
            line=dict(color="black", width=0.5),
            label=nodes,
            color=[
                'rgba(52, 152, 219, 0.8)',  # L1C0 - blue
                'rgba(52, 152, 219, 0.6)',  # L1C1 - light blue
                'rgba(46, 204, 113, 0.8)',  # L2C0 - green
                'rgba(231, 76, 60, 0.8)',   # L2C1 - red
                'rgba(241, 196, 15, 0.8)',  # L3C0 - yellow
                'rgba(155, 89, 182, 0.8)',  # L3C1 - purple
                'rgba(46, 204, 113, 1.0)',  # OutputC0 - green (no disease)
                'rgba(231, 76, 60, 1.0)',   # OutputC1 - red (disease)
            ],
            # Horizontal positions - evenly spaced
            x=[0.01, 0.01, 0.33, 0.33, 0.66, 0.66, 0.99, 0.99],
            # Vertical positions - more conservative to avoid cutoff
            y=[0.1, 0.9,     # Layer 1: 80% separation
               0.3, 0.7,      # Layer 2: 40% separation  
               0.1, 0.9,      # Layer 3: same as Layer 1
               0.3, 0.7],     # Output: same as Layer 2
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            label=links['label'],
            color=links['color']
        )
    )])
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Heart Disease Model: Patient Flow Through Risk Stratification Layers",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font=dict(size=12),
        height=800,  # Increased height for better spacing
        width=1200,
        margin=dict(l=50, r=50, t=120, b=100),  # Further increased margins
        annotations=[
            # Layer labels
            dict(x=0.01, y=-0.05, text="<b>Layer 1</b><br>Initial Risk", showarrow=False, xanchor='center'),
            dict(x=0.33, y=-0.05, text="<b>Layer 2</b><br>CV Health", showarrow=False, xanchor='center'),
            dict(x=0.66, y=-0.05, text="<b>Layer 3</b><br>Risk Profile", showarrow=False, xanchor='center'),
            dict(x=0.99, y=-0.05, text="<b>Output</b><br>Diagnosis", showarrow=False, xanchor='center'),
            
            # Path statistics
            dict(x=0.5, y=1.15, text="<b>Path Distribution:</b> Path 1 (43.3%) | Path 2 (35.2%) | Path 3 (10.7%) | Path 4 (6.7%) | Path 5 (2.2%)",
                 showarrow=False, xanchor='center', font=dict(size=10)),
            
            # Key insights
            dict(x=0.5, y=1.05, text="<b>Key Finding:</b> 83.3% male composition in Path 4 reveals gender bias in predictions",
                 showarrow=False, xanchor='center', font=dict(size=10, color='purple'))
        ]
    )
    
    return fig

def main():
    """Generate and save the labeled Sankey diagram."""
    # Create the diagram
    fig = create_sankey_diagram()
    
    # Save as HTML
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    html_path = output_dir / "heart_sankey_labeled.html"
    fig.write_html(str(html_path))
    print(f"Saved interactive diagram to: {html_path}")
    
    # Save as static image (requires kaleido)
    try:
        import kaleido
        png_path = output_dir / "heart_sankey_labeled.png"
        fig.write_image(str(png_path), width=1200, height=800, scale=2)
        print(f"Saved static image to: {png_path}")
        
        # Also save to arxiv figures directory
        arxiv_path = Path(__file__).parent.parent.parent / "arxiv_submission" / "figures" / "heart_membership_overlap_sankey_labeled.png"
        fig.write_image(str(arxiv_path), width=1200, height=800, scale=2)
        print(f"Saved to arxiv figures: {arxiv_path}")
    except ImportError:
        print("Note: Install kaleido to generate static images: pip install kaleido")
    except Exception as e:
        print(f"Could not generate static image: {e}")

if __name__ == "__main__":
    main()