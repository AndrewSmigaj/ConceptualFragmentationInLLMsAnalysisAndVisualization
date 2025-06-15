#!/usr/bin/env python3
"""
Generate comprehensive GPT-2 Sankey diagrams for the paper with:
1. All three windows (Early, Middle, Late)
2. Archetypal paths highlighted
3. Legend with cluster labels
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import plotly.io as pio

# Load the cluster labels from our analysis
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

def load_windowed_data():
    """Load the windowed path data from our analysis."""
    base_path = Path(__file__).parent / "results/unified_cta_config/unified_cta_20250524_073316/windowed_analysis"
    
    # Load the complete windowed paths data
    with open(base_path / "all_windowed_paths.json", 'r') as f:
        data = json.load(f)
    
    return data

def create_sankey_for_window(window_name, paths_data, window_layers):
    """Create a Sankey diagram for a specific window."""
    
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
    
    # Create nodes
    nodes = []
    node_map = {}
    node_idx = 0
    
    # Add all unique clusters as nodes
    for layer in window_layers:
        for cluster in set(cluster for path, _ in sorted_paths for cluster in path if cluster.startswith(f"L{layer}_")):
            if cluster not in node_map:
                label = f"{cluster}: {CLUSTER_LABELS.get(cluster, cluster)}"
                nodes.append(label)
                node_map[cluster] = node_idx
                node_idx += 1
    
    # Create links
    links = {"source": [], "target": [], "value": [], "label": []}
    
    # Add links for top paths (showing archetypal paths)
    for path, count in sorted_paths[:10]:  # Top 10 paths
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            if source in node_map and target in node_map:
                links["source"].append(node_map[source])
                links["target"].append(node_map[target])
                links["value"].append(count)
                percentage = (count / total_words) * 100
                links["label"].append(f"{count} words ({percentage:.1f}%)")
    
    return nodes, links, sorted_paths[:5]  # Return top 5 archetypal paths

def create_combined_figure():
    """Create a figure with all three windows and archetypal paths."""
    
    # Load data
    data = load_windowed_data()
    
    # Define windows
    windows = {
        "early": list(range(0, 4)),
        "middle": list(range(4, 8)),
        "late": list(range(8, 12))
    }
    
    # Create three separate Sankey diagrams
    for window_name, layers in windows.items():
        nodes, links, top_paths = create_sankey_for_window(window_name, data, layers)
        
        # Create Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="lightblue" if window_name == "early" else 
                      "lightgreen" if window_name == "middle" else 
                      "lightcoral"
            ),
            link=dict(
                source=links["source"],
                target=links["target"],
                value=links["value"],
                label=links["label"]
            )
        )])
        
        # Update layout
        title_text = {
            "early": "Early Window (L0-L3): Semantic Differentiation<br>19 unique paths",
            "middle": "Middle Window (L4-L7): Grammatical Convergence<br>5 unique paths", 
            "late": "Late Window (L8-L11): Syntactic Superhighways<br>4 unique paths"
        }
        
        fig.update_layout(
            title={
                'text': title_text[window_name],
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            font=dict(size=10),
            height=600,
            width=1000,
            margin=dict(l=10, r=10, t=60, b=10)
        )
        
        # Save individual window figures
        output_path = Path(__file__).parent.parent.parent.parent.parent / "arxiv_submission" / "figures"
        output_path.mkdir(exist_ok=True)
        
        # Save as HTML
        html_path = output_path / f"gpt2_sankey_{window_name}.html"
        fig.write_html(str(html_path))
        print(f"Saved {window_name} window to: {html_path}")
        
        # Try to save as PNG
        try:
            png_path = output_path / f"gpt2_sankey_{window_name}.png"
            fig.write_image(str(png_path), width=1000, height=600, scale=2)
            print(f"Saved PNG to: {png_path}")
        except:
            print(f"Could not save PNG for {window_name} window (install kaleido)")
    
    # Create a comprehensive figure with archetypal paths
    create_archetypal_paths_figure(data)

def create_archetypal_paths_figure(data):
    """Create a figure showing the top archetypal paths across all windows."""
    
    # Count complete paths (all windows)
    complete_paths = {}
    
    # Handle different data formats
    if isinstance(data, dict) and 'paths' in data:
        data = data['paths']
    
    for word, word_data in data.items():
        if isinstance(word_data, dict) and all(window in word_data for window in ["early", "middle", "late"]):
            full_path = (
                tuple(word_data["early"]),
                tuple(word_data["middle"]), 
                tuple(word_data["late"])
            )
            complete_paths[full_path] = complete_paths.get(full_path, 0) + 1
    
    # Get top 5 complete paths
    top_paths = sorted(complete_paths.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Create text summary
    summary_text = "# Top 5 Archetypal Paths Through GPT-2\n\n"
    
    for i, (path, count) in enumerate(top_paths, 1):
        percentage = (count / len(data)) * 100
        summary_text += f"## Path {i}: {count} words ({percentage:.1f}%)\n"
        
        # Format each window
        early_path = " → ".join(f"{c} ({CLUSTER_LABELS.get(c, c)})" for c in path[0])
        middle_path = " → ".join(f"{c} ({CLUSTER_LABELS.get(c, c)})" for c in path[1])
        late_path = " → ".join(f"{c} ({CLUSTER_LABELS.get(c, c)})" for c in path[2])
        
        summary_text += f"- Early: {early_path}\n"
        summary_text += f"- Middle: {middle_path}\n"
        summary_text += f"- Late: {late_path}\n\n"
        
        # Identify path type
        if "Noun Gateway" in middle_path:
            summary_text += "**Type**: Noun Processing Pipeline\n\n"
        elif "Adjective Gateway" in middle_path:
            summary_text += "**Type**: Modifier Processing Pipeline\n\n"
        else:
            summary_text += "**Type**: Alternative Processing Route\n\n"
    
    # Save archetypal paths summary
    output_path = Path(__file__).parent.parent.parent.parent.parent / "arxiv_submission" / "figures"
    with open(output_path / "gpt2_archetypal_paths.txt", 'w') as f:
        f.write(summary_text)
    print(f"Saved archetypal paths to: {output_path / 'gpt2_archetypal_paths.txt'}")
    
    # Create legend figure
    create_legend_figure()

def create_legend_figure():
    """Create a legend figure explaining all cluster labels."""
    
    # Create a figure for the legend
    fig = go.Figure()
    
    # Add invisible scatter plot to create legend entries
    layers = ["Layer 0", "Layers 1-3", "Layers 4-7", "Layers 8-11"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#F7DC6F"]
    descriptions = [
        "Semantic Differentiation",
        "Binary Consolidation", 
        "Grammatical Highways",
        "Syntactic Superhighways"
    ]
    
    for layer, color, desc in zip(layers, colors, descriptions):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color=color),
            name=f"{layer}: {desc}",
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "GPT-2 Cluster Organization Legend",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=0.1,
            font=dict(size=14)
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=800,
        width=1000,
        plot_bgcolor='white'
    )
    
    # Add cluster label text
    cluster_text = ""
    y_pos = 0.8
    
    for cluster, label in CLUSTER_LABELS.items():
        layer_num = int(cluster.split('_')[0][1:])
        
        # Add section headers
        if cluster == "L0_C0":
            cluster_text += "<b>Layer 0: Semantic Differentiation</b><br>"
        elif cluster == "L1_C0":
            cluster_text += "<br><b>Layers 1-3: Binary Consolidation</b><br>"
        elif cluster == "L4_C0":
            cluster_text += "<br><b>Layers 4-7: Grammatical Highways</b><br>"
        elif cluster == "L8_C0":
            cluster_text += "<br><b>Layers 8-11: Syntactic Superhighways</b><br>"
        
        cluster_text += f"{cluster}: {label}<br>"
    
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=cluster_text,
        showarrow=False,
        xanchor='center',
        yanchor='middle',
        font=dict(size=12, family="monospace"),
        align='left'
    )
    
    # Save legend
    output_path = Path(__file__).parent.parent.parent.parent.parent / "arxiv_submission" / "figures"
    html_path = output_path / "gpt2_cluster_legend.html"
    fig.write_html(str(html_path))
    print(f"Saved legend to: {html_path}")
    
    try:
        png_path = output_path / "gpt2_cluster_legend.png"
        fig.write_image(str(png_path), width=1000, height=800, scale=2)
        print(f"Saved legend PNG to: {png_path}")
    except:
        print("Could not save legend PNG (install kaleido)")

def main():
    """Generate all GPT-2 figures for the paper."""
    print("Generating GPT-2 Concept MRI figures...")
    print("=" * 60)
    
    create_combined_figure()
    
    print("\n" + "=" * 60)
    print("GENERATED FILES:")
    print("=" * 60)
    print("1. Three window Sankey diagrams:")
    print("   - gpt2_sankey_early.html/png")
    print("   - gpt2_sankey_middle.html/png") 
    print("   - gpt2_sankey_late.html/png")
    print("\n2. Archetypal paths summary:")
    print("   - gpt2_archetypal_paths.txt")
    print("\n3. Cluster legend:")
    print("   - gpt2_cluster_legend.html/png")
    print("\nAll files saved to: arxiv_submission/figures/")

if __name__ == "__main__":
    main()