"""
Generate Sankey diagrams for Concept MRI visualization.

Creates three Sankey diagrams (early, middle, late windows) with:
- Grammatical category coloring
- Proportional path thickness
- Interactive tooltips
"""

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def create_windowed_sankey(window_data, path_data, window_name):
    """Create a Sankey diagram for a single window."""
    
    # Extract nodes and links from the existing Sankey data
    nodes = window_data['nodes']
    links = window_data['links']
    
    # Create node labels with cluster info
    node_labels = []
    node_colors = []
    
    for node in nodes:
        # Enhance label with layer info
        label = node['label'].replace('-', ' ')
        node_labels.append(label)
        # Default gray color (will be overridden by flows)
        node_colors.append('rgba(128, 128, 128, 0.8)')
    
    # Process links with grammatical coloring
    sources = []
    targets = []
    values = []
    link_colors = []
    link_labels = []
    
    # Get path information to determine colors
    path_info = path_data['all_paths']
    
    # Create a mapping of transitions to grammatical categories
    transition_to_grammar = {}
    
    for path in path_info:
        sequence = path['cluster_sequence']
        grammar_cat = path['category_analysis']['dominant_grammatical']
        color = path['category_analysis']['dominant_color']
        
        # Map each transition in the path
        for i in range(len(sequence) - 1):
            trans_key = (sequence[i], sequence[i+1])
            if trans_key not in transition_to_grammar:
                transition_to_grammar[trans_key] = {
                    'grammar': grammar_cat,
                    'color': color,
                    'count': 0,
                    'examples': []
                }
            transition_to_grammar[trans_key]['count'] += path['frequency']
            transition_to_grammar[trans_key]['examples'].extend(path['example_words'][:3])
    
    # Map node IDs to indices
    node_id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}
    
    # Process links
    for link in links:
        source_idx = link['source']
        target_idx = link['target']
        value = link['value']
        
        # Get transition key
        source_id = nodes[source_idx]['id']
        target_id = nodes[target_idx]['id']
        trans_key = (source_id, target_id)
        
        # Get grammatical info
        if trans_key in transition_to_grammar:
            grammar_info = transition_to_grammar[trans_key]
            # Convert hex to rgba with transparency
            hex_color = grammar_info['color']
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            color = f'rgba({r},{g},{b},0.5)'
            label = f"{grammar_info['grammar']}: {value} words"
        else:
            color = 'rgba(128, 128, 128, 0.4)'
            label = f"{value} words"
        
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(value)
        link_colors.append(color)
        link_labels.append(label)
    
    # Create Sankey trace
    sankey_trace = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors,
            hovertemplate='%{label}<br>%{value} words<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{label}<extra></extra>',
            label=link_labels
        ),
        textfont=dict(size=10, color='black')
    )
    
    return sankey_trace


def generate_all_sankeys(results_dir):
    """Generate all three Sankey diagrams for the MRI visualization."""
    
    # Load the prepared MRI data
    mri_data_path = Path(results_dir) / "concept_mri_data.json"
    with open(mri_data_path, 'r') as f:
        mri_data = json.load(f)
    
    # Create subplots for three windows
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Early Window (L0-L3)", "Middle Window (L4-L7)", "Late Window (L8-L11)"),
        specs=[[{"type": "sankey"}, {"type": "sankey"}, {"type": "sankey"}]],
        horizontal_spacing=0.05
    )
    
    # Generate Sankey for each window
    for idx, window in enumerate(['early', 'middle', 'late']):
        if window in mri_data['sankey'] and window in mri_data['paths']:
            sankey_trace = create_windowed_sankey(
                mri_data['sankey'][window],
                mri_data['paths'][window],
                window
            )
            fig.add_trace(sankey_trace, row=1, col=idx+1)
    
    # Update layout
    fig.update_layout(
        title={
            'text': "GPT-2 Concept MRI: Word Flow Through Neural Layers",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font=dict(size=12),
        height=600,
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    # Save as HTML
    output_path = Path(results_dir) / "sankey_all_windows.html"
    fig.write_html(
        str(output_path),
        include_plotlyjs='cdn',
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    print(f"Saved combined Sankey diagram to: {output_path}")
    
    # Also generate individual Sankey diagrams
    for window in ['early', 'middle', 'late']:
        if window not in mri_data['sankey'] or window not in mri_data['paths']:
            continue
            
        # Create individual figure
        fig_single = go.Figure()
        
        sankey_trace = create_windowed_sankey(
            mri_data['sankey'][window],
            mri_data['paths'][window],
            window
        )
        fig_single.add_trace(sankey_trace)
        
        # Get window info
        window_info = mri_data['paths'][window]['window_info']
        n_paths = len(mri_data['paths'][window]['all_paths'])
        
        fig_single.update_layout(
            title={
                'text': f"{window.capitalize()} Window ({window_info['layer_range']}) - {n_paths} Unique Paths",
                'x': 0.5,
                'xanchor': 'center'
            },
            font=dict(size=14),
            height=500,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Save individual file
        output_path = Path(results_dir) / f"sankey_{window}_enhanced.html"
        fig_single.write_html(
            str(output_path),
            include_plotlyjs='cdn',
            config={'displayModeBar': True, 'displaylogo': False}
        )
        
        print(f"Saved {window} window Sankey to: {output_path}")
    
    # Create a legend for grammatical categories
    create_grammar_legend(mri_data, results_dir)


def create_grammar_legend(mri_data, results_dir):
    """Create a legend showing grammatical category colors."""
    
    grammar_colors = {
        'noun': '#1f77b4',
        'adjective': '#ff7f0e',
        'adverb': '#2ca02c',
        'verb': '#9467bd',
        'unknown': '#7f7f7f'
    }
    
    # Create a simple bar chart as legend
    fig = go.Figure()
    
    categories = list(grammar_colors.keys())
    colors = [grammar_colors[cat] for cat in categories]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=[1] * len(categories),
        marker_color=colors,
        text=categories,
        textposition='inside',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Grammatical Category Colors",
        xaxis_title="Category",
        yaxis_visible=False,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    output_path = Path(results_dir) / "grammar_legend.html"
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    
    print(f"Saved grammar legend to: {output_path}")


def main(results_dir=None):
    """Main function to generate all Sankey visualizations."""
    
    if results_dir is None:
        results_dir = Path("results/unified_cta_config/unified_cta_20250524_073316")
    else:
        results_dir = Path(results_dir)
    
    print("Generating Sankey diagrams for Concept MRI...")
    generate_all_sankeys(results_dir)
    print("\nAll Sankey diagrams generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = None
    
    main(results_dir)