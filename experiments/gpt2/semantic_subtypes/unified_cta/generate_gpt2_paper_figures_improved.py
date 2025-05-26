#!/usr/bin/env python3
"""
Improved GPT-2 Sankey diagrams with better layout and visualization.
- Better node positioning to prevent overlap
- Consistent color scheme
- Highlighted archetypal paths
- Clearer labels
"""

import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Define cluster labels based on the analysis
CLUSTER_LABELS = {
    # Layer 0 - Semantic Differentiation
    "L0_C0": "Animate Creatures",
    "L0_C1": "Tangible Objects", 
    "L0_C2": "Scalar Properties",
    "L0_C3": "Abstract & Relational",
    
    # Layers 1-3 - Binary Consolidation
    "L1_C0": "Modifier Space",
    "L1_C1": "Entity Space",
    "L2_C0": "Property Attractor",
    "L2_C1": "Object Attractor",
    "L3_C0": "Property Attractor",
    "L3_C1": "Object Attractor",
    
    # Layers 4-7 - Grammatical Highways
    "L4_C0": "Adjective Gateway",
    "L4_C1": "Noun Gateway",
    "L5_C0": "Entity Pipeline",
    "L5_C1": "Property Pipeline",
    "L6_C0": "Entity Pipeline",
    "L6_C1": "Property Pipeline",
    "L7_C0": "Modifier Hub",
    "L7_C1": "Entity Hub",
    
    # Layers 8-11 - Syntactic Superhighways
    "L8_C0": "Modifier Entry",
    "L8_C1": "Entity Entry",
    "L9_C0": "Entity Stream",
    "L9_C1": "Modifier Stream",
    "L10_C0": "Entity Stream",
    "L10_C1": "Modifier Stream",
    "L11_C0": "Terminal Modifiers",
    "L11_C1": "Terminal Entities"
}

# Define a professional color scheme
COLOR_SCHEME = {
    # Early window - semantic colors
    "L0_C0": "rgba(255, 107, 107, 0.9)",  # Red - Animals
    "L0_C1": "rgba(78, 205, 196, 0.9)",   # Turquoise - Objects
    "L0_C2": "rgba(255, 195, 0, 0.9)",    # Yellow - Properties
    "L0_C3": "rgba(155, 89, 182, 0.9)",   # Purple - Abstract
    
    # Middle layers - transitioning to grammatical
    "modifier": "rgba(52, 152, 219, 0.8)",   # Blue for modifiers
    "entity": "rgba(46, 204, 113, 0.8)",     # Green for entities
    
    # Late layers - final grammatical categories
    "modifier_final": "rgba(41, 128, 185, 0.9)",  # Dark blue
    "entity_final": "rgba(39, 174, 96, 0.9)",     # Dark green
}

def load_windowed_data():
    """Load the windowed path data from our analysis."""
    base_path = Path(__file__).parent / "results/unified_cta_config/unified_cta_20250524_073316/windowed_analysis"
    
    # Load the complete windowed paths data
    with open(base_path / "all_windowed_paths.json", 'r') as f:
        data = json.load(f)
    
    return data

def get_node_color(cluster_id):
    """Get appropriate color for a node based on its cluster ID."""
    if cluster_id in COLOR_SCHEME:
        return COLOR_SCHEME[cluster_id]
    
    # Determine if it's modifier or entity based on cluster number
    cluster_num = int(cluster_id.split('_C')[1])
    layer_num = int(cluster_id.split('_')[0][1:])
    
    if layer_num <= 3:
        # Early layers - use specific colors if defined
        return "rgba(200, 200, 200, 0.8)"  # Gray default
    elif layer_num <= 7:
        # Middle layers
        return COLOR_SCHEME["modifier"] if cluster_num == 0 else COLOR_SCHEME["entity"]
    else:
        # Late layers
        return COLOR_SCHEME["modifier_final"] if cluster_num == 0 else COLOR_SCHEME["entity_final"]

def create_improved_sankey_for_window(window_name, paths_data, window_layers):
    """Create an improved Sankey diagram with better layout."""
    
    # Count path frequencies
    path_counts = {}
    total_words = 0
    
    # Handle different data formats
    if isinstance(paths_data, dict) and 'paths' in paths_data:
        paths_data = paths_data['paths']
    
    for word, word_data in paths_data.items():
        if isinstance(word_data, dict) and window_name in word_data:
            path = tuple(word_data[window_name])
            path_counts[path] = path_counts.get(path, 0) + 1
            total_words += 1
    
    # Sort paths by frequency
    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create nodes with better organization
    nodes = []
    node_map = {}
    node_idx = 0
    node_positions_x = []
    node_positions_y = []
    node_colors = []
    
    # Organize nodes by layer
    layer_nodes = {}
    for layer in window_layers:
        layer_nodes[layer] = []
        for path, _ in sorted_paths:
            for cluster in path:
                if cluster.startswith(f"L{layer}_") and cluster not in node_map:
                    layer_nodes[layer].append(cluster)
                    node_map[cluster] = node_idx
                    node_idx += 1
    
    # Calculate positions for better layout
    x_positions = np.linspace(0.05, 0.95, len(window_layers))
    
    for i, layer in enumerate(window_layers):
        layer_clusters = sorted(set(layer_nodes[layer]))
        n_clusters = len(layer_clusters)
        
        if n_clusters == 0:
            continue
            
        # Distribute nodes vertically with good spacing
        if n_clusters == 1:
            y_positions = [0.5]
        elif n_clusters == 2:
            y_positions = [0.3, 0.7]
        elif n_clusters == 3:
            y_positions = [0.2, 0.5, 0.8]
        elif n_clusters == 4:
            y_positions = [0.15, 0.38, 0.62, 0.85]
        else:
            y_positions = np.linspace(0.1, 0.9, n_clusters)
        
        for j, cluster in enumerate(layer_clusters):
            label = f"{CLUSTER_LABELS.get(cluster, cluster)}"
            nodes.append(label)
            node_positions_x.append(x_positions[i])
            node_positions_y.append(y_positions[j])
            node_colors.append(get_node_color(cluster))
    
    # Create links with better styling
    links = {"source": [], "target": [], "value": [], "label": [], "color": []}
    
    # Process top paths (archetypal paths)
    top_paths_list = []
    for path, count in sorted_paths[:10]:  # Top 10 paths
        top_paths_list.append((path, count))
        
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if source in node_map and target in node_map:
                links["source"].append(node_map[source])
                links["target"].append(node_map[target])
                links["value"].append(count)
                percentage = (count / total_words) * 100
                links["label"].append(f"{count} words ({percentage:.1f}%)")
                
                # Color links based on their destination (entity vs modifier)
                if "Entity" in CLUSTER_LABELS.get(target, "") or "Noun" in CLUSTER_LABELS.get(target, ""):
                    links["color"].append("rgba(46, 204, 113, 0.4)")  # Green tint
                else:
                    links["color"].append("rgba(52, 152, 219, 0.4)")  # Blue tint
    
    return nodes, links, node_positions_x, node_positions_y, node_colors, top_paths_list

def create_improved_sankey_figures():
    """Create improved Sankey diagrams for all windows."""
    
    # Load data
    data = load_windowed_data()
    
    # Define windows
    windows = {
        "early": {"layers": list(range(0, 4)), "title": "Semantic Differentiation Phase"},
        "middle": {"layers": list(range(4, 8)), "title": "Grammatical Convergence Phase"},
        "late": {"layers": list(range(8, 12)), "title": "Syntactic Superhighways Phase"}
    }
    
    # Create figures for each window
    for window_name, window_info in windows.items():
        nodes, links, x_pos, y_pos, colors, top_paths = create_improved_sankey_for_window(
            window_name, data, window_info["layers"]
        )
        
        # Create Sankey figure with improved layout
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=30,  # Increased padding
                thickness=25,  # Slightly thicker nodes
                line=dict(color="black", width=1),
                label=nodes,
                color=colors,
                x=x_pos,
                y=y_pos
            ),
            link=dict(
                source=links["source"],
                target=links["target"],
                value=links["value"],
                label=links["label"],
                color=links["color"]
            )
        )])
        
        # Count unique paths
        unique_paths = len(set(path for path, _ in top_paths))
        
        # Update layout with better styling
        fig.update_layout(
            title={
                'text': f"<b>{window_info['title']}</b><br>" +
                       f"<span style='font-size: 14px'>Layers {window_info['layers'][0]}-{window_info['layers'][-1]} | " +
                       f"{unique_paths} dominant paths | " +
                       f"Window: {window_name.capitalize()}</span>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            font=dict(size=12),
            height=700,
            width=1200,
            margin=dict(l=50, r=50, t=100, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add annotations for key insights
        if window_name == "early":
            fig.add_annotation(
                x=0.5, y=-0.05,
                text="<b>Key Finding:</b> Words begin in 4 semantic clusters based on meaning",
                showarrow=False,
                xanchor='center',
                font=dict(size=12, color='#666')
            )
        elif window_name == "middle":
            fig.add_annotation(
                x=0.5, y=-0.05,
                text="<b>Key Finding:</b> Semantic clusters merge into grammatical categories (Noun/Modifier gateways)",
                showarrow=False,
                xanchor='center',
                font=dict(size=12, color='#666')
            )
        else:  # late
            fig.add_annotation(
                x=0.5, y=-0.05,
                text="<b>Key Finding:</b> 72.8% converge to Entity superhighway, 25.8% to Modifier highway",
                showarrow=False,
                xanchor='center',
                font=dict(size=12, color='#666')
            )
        
        # Save outputs
        output_path = Path(__file__).parent.parent.parent.parent.parent / "arxiv_submission" / "figures"
        output_path.mkdir(exist_ok=True)
        
        # Save as HTML
        html_path = output_path / f"gpt2_sankey_{window_name}_improved.html"
        fig.write_html(str(html_path))
        print(f"Saved improved {window_name} window to: {html_path}")
        
        # Try to save as PNG
        try:
            png_path = output_path / f"gpt2_sankey_{window_name}_improved.png"
            fig.write_image(str(png_path), width=1200, height=700, scale=2)
            print(f"Saved improved PNG to: {png_path}")
        except:
            print(f"Could not save PNG for {window_name} window (install kaleido)")

def create_combined_sankey():
    """Create a single combined diagram showing the complete transformation."""
    # This would create a single large diagram showing all 12 layers
    # Implementation depends on specific requirements
    pass

def main():
    """Generate improved GPT-2 figures."""
    print("Generating improved GPT-2 Sankey diagrams...")
    print("=" * 60)
    
    create_improved_sankey_figures()
    
    print("\n" + "=" * 60)
    print("Improved diagrams generated with:")
    print("- Better node positioning to prevent overlap")
    print("- Consistent color scheme (semantic → grammatical)")
    print("- Increased padding and thickness")
    print("- Key findings annotations")
    print("- Professional styling")

if __name__ == "__main__":
    main()