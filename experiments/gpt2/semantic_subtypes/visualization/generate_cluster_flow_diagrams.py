#!/usr/bin/env python3
"""
Generate cluster flow diagrams (Sankey diagrams) for interesting windows.
Shows how words flow between clusters across layers.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.offline as pyo

def load_results():
    """Load clustering results and curated words."""
    result_dir = Path("semantic_subtypes_optimal_experiment_20250523_182344")
    
    with open(result_dir / "semantic_subtypes_kmeans_optimal.pkl", 'rb') as f:
        kmeans_results = pickle.load(f)
    
    with open(result_dir / "semantic_subtypes_ets_optimal.pkl", 'rb') as f:
        ets_results = pickle.load(f)
    
    with open("data/gpt2_semantic_subtypes_curated.json", 'r') as f:
        curated_data = json.load(f)
    
    return kmeans_results, ets_results, curated_data

def create_sankey_diagram(clustering_results, start_layer, end_layer, curated_data, method_name, window_name):
    """Create Sankey diagram for cluster flow between layers."""
    
    sentences = clustering_results['sentences']
    
    # Create word to subtype mapping
    word_to_subtype = {}
    for subtype, words in curated_data['curated_words'].items():
        for word in words:
            word_to_subtype[word] = subtype
    
    # Build nodes and links
    nodes = []
    node_labels = []
    node_colors = []
    
    # Color palette for clusters
    cluster_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    # Create nodes for each layer and cluster
    node_map = {}
    node_idx = 0
    
    for layer in range(start_layer, end_layer + 1):
        layer_key = f"layer_{layer}"
        layer_data = clustering_results['layer_results'][layer_key]
        
        # Find unique clusters in this layer
        clusters_in_layer = set()
        for clusters in layer_data['cluster_labels'].values():
            if 0 in clusters:
                clusters_in_layer.add(clusters[0])
        
        for cluster_id in sorted(clusters_in_layer):
            node_key = f"L{layer}_C{cluster_id}"
            node_map[node_key] = node_idx
            node_labels.append(f"L{layer}:C{cluster_id}")
            node_colors.append(cluster_colors[cluster_id % len(cluster_colors)])
            node_idx += 1
    
    # Build links between consecutive layers
    links = defaultdict(lambda: {'count': 0, 'subtypes': Counter()})
    
    for sent_idx in range(len(sentences)):
        path = []
        valid = True
        
        # Build path for this word
        for layer in range(start_layer, end_layer + 1):
            layer_key = f"layer_{layer}"
            if (sent_idx in clustering_results['layer_results'][layer_key]['cluster_labels'] and
                0 in clustering_results['layer_results'][layer_key]['cluster_labels'][sent_idx]):
                cluster_id = clustering_results['layer_results'][layer_key]['cluster_labels'][sent_idx][0]
                path.append(f"L{layer}_C{cluster_id}")
            else:
                valid = False
                break
        
        if valid and len(path) == (end_layer - start_layer + 1):
            word = sentences[sent_idx]
            subtype = word_to_subtype.get(word, 'unknown')
            
            # Add links for consecutive layers
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i + 1]
                link_key = (source_node, target_node)
                links[link_key]['count'] += 1
                links[link_key]['subtypes'][subtype] += 1
    
    # Convert links to Plotly format
    source_indices = []
    target_indices = []
    values = []
    link_labels = []
    link_colors = []
    
    # Subtype colors for links
    subtype_colors = {
        'concrete_nouns': 'rgba(255, 107, 107, 0.4)',
        'abstract_nouns': 'rgba(78, 205, 196, 0.4)',
        'physical_adjectives': 'rgba(69, 183, 209, 0.4)',
        'emotive_adjectives': 'rgba(150, 206, 180, 0.4)',
        'manner_adverbs': 'rgba(254, 202, 87, 0.4)',
        'degree_adverbs': 'rgba(255, 159, 243, 0.4)',
        'action_verbs': 'rgba(155, 89, 182, 0.4)',
        'stative_verbs': 'rgba(52, 152, 219, 0.4)',
        'unknown': 'rgba(128, 128, 128, 0.4)'
    }
    
    for (source, target), link_data in links.items():
        if source in node_map and target in node_map:
            source_indices.append(node_map[source])
            target_indices.append(node_map[target])
            values.append(link_data['count'])
            
            # Determine dominant subtype for coloring
            if link_data['subtypes']:
                dominant_subtype = max(link_data['subtypes'].items(), key=lambda x: x[1])[0]
                link_colors.append(subtype_colors.get(dominant_subtype, subtype_colors['unknown']))
                
                # Create label showing subtype breakdown
                subtype_str = ", ".join([f"{st}: {cnt}" for st, cnt in link_data['subtypes'].most_common(3)])
                link_labels.append(f"{link_data['count']} words ({subtype_str})")
            else:
                link_colors.append(subtype_colors['unknown'])
                link_labels.append(f"{link_data['count']} words")
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            label=link_labels,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title=f"{method_name} Cluster Flow: {window_name} (Layers {start_layer}-{end_layer})",
        font_size=12,
        height=600,
        width=1000
    )
    
    # Save as HTML
    output_file = f"{method_name.lower()}_flow_{window_name}.html"
    pyo.plot(fig, filename=output_file, auto_open=False)
    print(f"Saved: {output_file}")
    
    return fig

def analyze_window_metrics(clustering_results, window_start, window_end):
    """Calculate metrics for a specific window."""
    
    # Count unique paths in this window
    paths = defaultdict(int)
    
    for sent_idx in clustering_results['sentences']:
        path = []
        valid = True
        
        for layer in range(window_start, window_end + 1):
            layer_key = f"layer_{layer}"
            if (sent_idx in clustering_results['layer_results'][layer_key]['cluster_labels'] and
                0 in clustering_results['layer_results'][layer_key]['cluster_labels'][sent_idx]):
                cluster_id = clustering_results['layer_results'][layer_key]['cluster_labels'][sent_idx][0]
                path.append(f"L{layer}C{cluster_id}")
            else:
                valid = False
                break
        
        if valid:
            path_str = " -> ".join(path)
            paths[path_str] += 1
    
    # Calculate metrics
    total_words = len(clustering_results['sentences'])
    unique_paths = len(paths)
    
    # Path entropy
    if paths:
        path_probs = np.array(list(paths.values())) / sum(paths.values())
        entropy = -np.sum(path_probs * np.log2(path_probs + 1e-10))
    else:
        entropy = 0
    
    # Fragmentation
    start_to_next = defaultdict(set)
    for path_str in paths:
        parts = path_str.split(" -> ")
        if len(parts) >= 2:
            start_to_next[parts[0]].add(parts[1])
    
    fragmentation = np.mean([len(nexts) for nexts in start_to_next.values()]) if start_to_next else 1.0
    
    return {
        'unique_paths': unique_paths,
        'entropy': entropy,
        'fragmentation': fragmentation,
        'coverage': sum(paths.values()) / total_words
    }

def main():
    """Generate cluster flow diagrams for interesting windows."""
    
    print("Loading data...")
    kmeans_results, ets_results, curated_data = load_results()
    
    # Define interesting windows based on our earlier analysis
    interesting_windows = [
        (0, 2, "early"),      # Early layers - highest diversity
        (5, 7, "middle"),     # Middle layers - semantic organization
        (9, 11, "late")       # Late layers - stable patterns
    ]
    
    print("\nGenerating cluster flow diagrams...")
    
    # Analyze windows first
    print("\nWindow Analysis:")
    print("="*60)
    
    for start, end, name in interesting_windows:
        print(f"\nWindow {name} (layers {start}-{end}):")
        
        # K-means analysis
        kmeans_metrics = analyze_window_metrics(kmeans_results, start, end)
        print(f"  K-means: {kmeans_metrics['unique_paths']} unique paths, "
              f"entropy={kmeans_metrics['entropy']:.2f}, "
              f"fragmentation={kmeans_metrics['fragmentation']:.2f}")
        
        # ETS analysis
        ets_metrics = analyze_window_metrics(ets_results, start, end)
        print(f"  ETS: {ets_metrics['unique_paths']} unique paths, "
              f"entropy={ets_metrics['entropy']:.2f}, "
              f"fragmentation={ets_metrics['fragmentation']:.2f}")
    
    # Generate Sankey diagrams
    print("\nGenerating Sankey diagrams...")
    
    for start, end, name in interesting_windows:
        # K-means diagram
        create_sankey_diagram(kmeans_results, start, end, curated_data, 
                            "K-means", f"window_{name}")
        
        # ETS diagram
        create_sankey_diagram(ets_results, start, end, curated_data, 
                            "ETS", f"window_{name}")
    
    # Also create a full-path diagram for a smaller subset of representative words
    print("\nGenerating representative word paths...")
    create_representative_paths_diagram(kmeans_results, ets_results, curated_data)
    
    print("\nAll diagrams generated!")

def create_representative_paths_diagram(kmeans_results, ets_results, curated_data):
    """Create diagram showing full paths for representative words from each subtype."""
    
    # Select 2 representative words from each subtype
    representatives = {}
    for subtype, words in curated_data['curated_words'].items():
        if len(words) >= 2:
            # Take first and middle word alphabetically
            sorted_words = sorted(words)
            representatives[subtype] = [sorted_words[0], sorted_words[len(sorted_words)//2]]
        else:
            representatives[subtype] = words[:2]
    
    # Build traces for visualization
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("K-means Paths", "ETS Paths"),
        horizontal_spacing=0.1
    )
    
    # Colors for subtypes
    subtype_colors = {
        'concrete_nouns': '#FF6B6B',
        'abstract_nouns': '#4ECDC4',
        'physical_adjectives': '#45B7D1',
        'emotive_adjectives': '#96CEB4',
        'manner_adverbs': '#FECA57',
        'degree_adverbs': '#FF9FF3',
        'action_verbs': '#9B59B6',
        'stative_verbs': '#3498DB'
    }
    
    # Plot paths for each method
    for col, (method_results, method_name) in enumerate([(kmeans_results, "K-means"), 
                                                          (ets_results, "ETS")], 1):
        sentences = method_results['sentences']
        
        for subtype, rep_words in representatives.items():
            for word in rep_words:
                if word in sentences:
                    word_idx = sentences.index(word)
                    
                    # Extract path
                    x_coords = []
                    y_coords = []
                    
                    for layer_idx in range(13):
                        layer_key = f"layer_{layer_idx}"
                        if (word_idx in method_results['layer_results'][layer_key]['cluster_labels'] and
                            0 in method_results['layer_results'][layer_key]['cluster_labels'][word_idx]):
                            cluster_id = method_results['layer_results'][layer_key]['cluster_labels'][word_idx][0]
                            x_coords.append(layer_idx)
                            y_coords.append(cluster_id)
                    
                    if x_coords:
                        fig.add_trace(
                            go.Scatter(
                                x=x_coords,
                                y=y_coords,
                                mode='lines+markers',
                                name=f"{word} ({subtype})",
                                line=dict(color=subtype_colors[subtype], width=2),
                                marker=dict(size=8),
                                showlegend=(col == 1),  # Only show legend for first subplot
                                hovertemplate=f"{word}<br>Layer: %{{x}}<br>Cluster: %{{y}}<extra></extra>"
                            ),
                            row=1, col=col
                        )
    
    fig.update_xaxes(title_text="Layer", row=1, col=1)
    fig.update_xaxes(title_text="Layer", row=1, col=2)
    fig.update_yaxes(title_text="Cluster ID", row=1, col=1)
    fig.update_yaxes(title_text="Cluster ID", row=1, col=2)
    
    fig.update_layout(
        title="Representative Word Paths Through Clusters",
        height=600,
        width=1200,
        hovermode='x unified'
    )
    
    pyo.plot(fig, filename="representative_word_paths.html", auto_open=False)
    print("Saved: representative_word_paths.html")

if __name__ == "__main__":
    main()