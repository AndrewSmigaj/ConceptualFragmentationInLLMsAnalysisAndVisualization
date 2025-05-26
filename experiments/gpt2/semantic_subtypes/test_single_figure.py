#!/usr/bin/env python3
"""
Test script to generate a single GPT-2 figure with proper labels and UMAP.
"""

import json
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Test with just early window
def generate_test_sankey():
    """Generate a test Sankey diagram to verify labels are showing."""
    
    # Create simple test data
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="black", width=2),
            label=[
                "<b>L0: Animate Creatures</b>",
                "<b>L0: Tangible Objects</b>",
                "<b>L0: Scalar Properties</b>",
                "<b>L0: Abstract Relations</b>",
                "<b>L1: Modifier Space</b>",
                "<b>L1: Entity Space</b>",
                "<b>L2: Property Attractor</b>",
                "<b>L2: Object Attractor</b>",
                "<b>L3: Property Attractor</b>",
                "<b>L3: Object Attractor</b>"
            ],
            color=[
                'rgba(255, 127, 14, 0.8)',  # Orange
                'rgba(31, 119, 180, 0.8)',   # Blue
                'rgba(44, 160, 44, 0.8)',    # Green
                'rgba(214, 39, 40, 0.8)',    # Red
                'rgba(44, 160, 44, 0.8)',    # Green
                'rgba(31, 119, 180, 0.8)',   # Blue
                'rgba(44, 160, 44, 0.8)',    # Green
                'rgba(31, 119, 180, 0.8)',   # Blue
                'rgba(44, 160, 44, 0.8)',    # Green
                'rgba(31, 119, 180, 0.8)',   # Blue
            ],
            x=[0, 0, 0, 0, 0.33, 0.33, 0.66, 0.66, 1, 1],
            y=[0.1, 0.3, 0.5, 0.7, 0.3, 0.6, 0.3, 0.6, 0.3, 0.6]
        ),
        link=dict(
            source=[0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 4, 5, 6, 7],
            target=[4, 5, 4, 5, 5, 5, 4, 4, 6, 7, 8, 9, 8, 9],
            value=[50, 100, 75, 80, 20, 50, 15, 40, 100, 200, 50, 100, 150, 250],
            color='rgba(0,0,0,0.2)'
        ),
        textfont=dict(size=14, color="black", family="Arial Black")
    )])
    
    # Add layer labels as annotations
    for i, layer_name in enumerate(["Layer 0", "Layer 1", "Layer 2", "Layer 3"]):
        fig.add_annotation(
            x=i/3,
            y=1.1,
            text=f"<b>{layer_name}</b>",
            showarrow=False,
            font=dict(size=16, color="black"),
            xanchor="center",
            yanchor="bottom"
        )
    
    fig.update_layout(
        title={
            'text': "GPT-2 Early Window Test - Cluster Labels Visible<br>" +
                   "<sub>Testing label visibility and layout</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font=dict(size=14, family="Arial"),
        height=800,
        width=1200,
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    
    # Save
    output_dir = Path(__file__).parent.parent.parent.parent / "arxiv_submission" / "figures"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    html_path = output_dir / f"test_sankey_{timestamp}.html"
    fig.write_html(str(html_path))
    print(f"Saved HTML: {html_path}")
    
    png_path = output_dir / "test_sankey.png"
    fig.write_image(str(png_path), width=1200, height=800, scale=2)
    print(f"Saved PNG: {png_path}")

def main():
    print("Generating test figure...")
    generate_test_sankey()
    print("Done!")

if __name__ == "__main__":
    main()