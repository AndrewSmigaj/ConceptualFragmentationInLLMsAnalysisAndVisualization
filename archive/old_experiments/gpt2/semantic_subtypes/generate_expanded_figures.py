"""
Generate all GPT-2 figures for the expanded dataset.
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def load_results():
    """Load the full CTA results."""
    results_path = Path("results/full_cta_expanded/full_cta_results.pkl")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def generate_sankey_diagrams(results):
    """Generate Sankey diagrams for each window."""
    window_analysis = results['window_analysis']
    paths = results['paths']
    
    windows = {
        'early': (0, 4, 'Early Window (L0-L3): Semantic Differentiation'),
        'middle': (4, 8, 'Middle Window (L4-L7): Transition Phase'),
        'late': (8, 12, 'Late Window (L8-L11): Grammatical Organization')
    }
    
    for window_name, (start, end, title) in windows.items():
        # Extract window paths
        window_paths = []
        for path in paths:
            window_path = path[start:end]
            window_paths.append(window_path)
        
        # Build Sankey data
        sources = []
        targets = []
        values = []
        labels = []
        label_set = set()
        
        # Count transitions
        transitions = defaultdict(int)
        for path in window_paths:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                transitions[(source, target)] += 1
                label_set.add(source)
                label_set.add(target)
        
        # Create label mapping
        label_list = sorted(list(label_set))
        label_to_idx = {label: idx for idx, label in enumerate(label_list)}
        
        # Build sankey data
        for (source, target), count in transitions.items():
            sources.append(label_to_idx[source])
            targets.append(label_to_idx[target])
            values.append(count)
        
        # Create colors based on cluster type
        colors = []
        for label in label_list:
            if 'C0' in label:
                colors.append('rgba(31, 119, 180, 0.8)')  # Blue
            elif 'C1' in label:
                colors.append('rgba(255, 127, 14, 0.8)')  # Orange
            elif 'C2' in label:
                colors.append('rgba(44, 160, 44, 0.8)')   # Green
            else:
                colors.append('rgba(214, 39, 40, 0.8)')   # Red
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=label_list,
                color=colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color='rgba(200, 200, 200, 0.4)'
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16)
            ),
            font=dict(size=12),
            width=800,
            height=600
        )
        
        # Save as HTML and PNG
        output_dir = Path(__file__).parent.parent.parent.parent / "arxiv_submission" / "figures"
        html_path = output_dir / f"gpt2_sankey_{window_name}_expanded.html"
        png_path = output_dir / f"gpt2_sankey_{window_name}_improved.png"
        
        fig.write_html(str(html_path))
        fig.write_image(str(png_path))
        
        print(f"Generated {window_name} sankey diagram")

def generate_trajectory_visualizations(results):
    """Generate 3D trajectory visualizations."""
    from sklearn.manifold import TSNE
    from umap import UMAP
    
    # Load activation data
    activation_path = Path("results/expanded_activations/gpt2_activations_expanded.pkl")
    with open(activation_path, 'rb') as f:
        activation_data = pickle.load(f)
    
    activations_by_layer = activation_data['activations_by_layer']
    categories = activation_data['categories']
    
    # Define windows
    windows = {
        'early': (0, 4, 'Early'),
        'middle': (4, 8, 'Middle'),
        'late': (8, 12, 'Late')
    }
    
    # Color mapping
    cat_colors = {
        'concrete_nouns': '#1f77b4',
        'abstract_nouns': '#ff7f0e',
        'physical_adjectives': '#2ca02c',
        'emotive_adjectives': '#d62728',
        'manner_adverbs': '#9467bd',
        'degree_adverbs': '#8c564b',
        'action_verbs': '#e377c2',
        'stative_verbs': '#7f7f7f'
    }
    
    for window_name, (start, end, title) in windows.items():
        # Average activations across window
        window_acts = []
        for i in range(len(categories)):
            acts = []
            for layer_idx in range(start, end):
                acts.append(activations_by_layer[layer_idx][i])
            window_acts.append(np.mean(acts, axis=0))
        
        X = np.array(window_acts)
        
        # Reduce to 3D
        reducer = UMAP(n_components=3, random_state=42, n_neighbors=15)
        X_3d = reducer.fit_transform(X)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot by category
        for cat in set(categories):
            indices = [i for i, c in enumerate(categories) if c == cat]
            color = cat_colors.get(cat, '#333333')
            
            ax.scatter(X_3d[indices, 0], 
                      X_3d[indices, 1], 
                      X_3d[indices, 2],
                      c=color, 
                      label=cat.replace('_', ' ').title(),
                      s=30, 
                      alpha=0.6, 
                      edgecolors='w', 
                      linewidth=0.5)
        
        # Customize plot
        ax.set_title(f"{title} Window Trajectories (n=1,228)", fontsize=16, pad=20)
        ax.set_xlabel("UMAP 1", fontsize=12)
        ax.set_ylabel("UMAP 2", fontsize=12)
        ax.set_zlabel("UMAP 3", fontsize=12)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
        
        # Adjust viewing angle
        ax.view_init(elev=20, azim=45)
        
        # Clean up axes
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Save figure
        output_path = Path(__file__).parent.parent.parent.parent / "arxiv_submission" / "figures" / f"gpt2_trajectories_{window_name}_3d.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Generated {window_name} trajectory visualization")

def generate_cluster_legend(results):
    """Generate improved cluster legend."""
    cluster_labels = results['cluster_labels']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = 0.95
    
    # Title
    ax.text(0.5, 0.98, "GPT-2 Cluster Organization (Expanded Dataset)", 
            fontsize=18, fontweight='bold', ha='center', transform=ax.transAxes)
    
    # Subtitle with dataset info
    ax.text(0.5, 0.94, "1,228 words: 275 nouns (22.4%), 280 adjectives (22.8%), 267 adverbs (21.7%), 406 verbs (33.1%)",
            fontsize=12, ha='center', transform=ax.transAxes, style='italic')
    
    y_pos -= 0.08
    
    # Group by early/middle/late
    layer_groups = {
        "Early Layers (0-3): Initial Processing": [0, 1, 2, 3],
        "Middle Layers (4-7): Transition Phase": [4, 5, 6, 7],
        "Late Layers (8-11): Final Organization": [8, 9, 10, 11]
    }
    
    for group_name, layers in layer_groups.items():
        # Group header
        ax.text(0.05, y_pos, group_name, fontsize=14, fontweight='bold', 
                transform=ax.transAxes)
        y_pos -= 0.06
        
        for layer_idx in layers:
            layer_key = f"layer_{layer_idx}"
            if layer_key in cluster_labels:
                layer_info = cluster_labels[layer_key]
                
                # Layer header
                ax.text(0.08, y_pos, f"Layer {layer_idx}:", fontsize=12, 
                        fontweight='bold', transform=ax.transAxes)
                y_pos -= 0.04
                
                # Cluster labels
                for cluster_id, info in sorted(layer_info.items()):
                    label_text = f"  {cluster_id}: {info['label']} (n={info['size']})"
                    ax.text(0.1, y_pos, label_text, fontsize=11, 
                            transform=ax.transAxes)
                    y_pos -= 0.035
                
                y_pos -= 0.01  # Extra space between layers
        
        y_pos -= 0.02  # Extra space between groups
    
    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Save
    output_path = Path(__file__).parent.parent.parent.parent / "arxiv_submission" / "figures" / "gpt2_cluster_legend.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Generated cluster legend")

def calculate_proper_stability(results):
    """Calculate stability metrics properly."""
    paths = results['paths']
    
    stability = {}
    
    for window_name, layer_range in [('early', range(0, 4)), 
                                     ('middle', range(4, 8)), 
                                     ('late', range(8, 12))]:
        # For each path, calculate how stable it is within the window
        stability_scores = []
        
        for path in paths:
            window_path = [path[i] for i in layer_range]
            
            # Count cluster changes
            changes = 0
            for i in range(1, len(window_path)):
                if window_path[i] != window_path[i-1]:
                    changes += 1
            
            # Stability is 1 - (fraction of possible changes that occurred)
            max_changes = len(window_path) - 1
            if max_changes > 0:
                stability_score = 1 - (changes / max_changes)
            else:
                stability_score = 1.0
                
            stability_scores.append(stability_score)
        
        stability[window_name] = np.mean(stability_scores)
    
    return stability

def generate_metrics_tables(results):
    """Generate updated metrics tables."""
    
    # Calculate proper stability
    stability = calculate_proper_stability(results)
    
    # Calculate fragmentation (based on path diversity)
    window_analysis = results['window_analysis']
    
    print("\n=== UPDATED METRICS FOR PAPER ===")
    
    print("\nStability Analysis:")
    for window in ['early', 'middle', 'late']:
        frag = 1 - (window_analysis[window]['dominant_count'] / window_analysis[window]['total_paths'])
        print(f"  {window.capitalize()}: Stability={stability[window]:.3f}, Fragmentation={frag:.3f}")
    
    print("\nPath Evolution:")
    for window in ['early', 'middle', 'late']:
        wa = window_analysis[window]
        print(f"  {window.capitalize()}: {wa['unique_paths']} paths, dominant={wa['dominant_percentage']:.1f}%")
    
    # Generate LaTeX table updates
    print("\n=== LaTeX Table Updates ===")
    
    print("\nStability and Fragmentation Table:")
    print("\\begin{tabular}{lccc}")
    print("\\toprule")
    print("Window & Stability & Fragmentation & Interpretation \\\\")
    print("\\midrule")
    
    for window in ['early', 'middle', 'late']:
        frag = 1 - (window_analysis[window]['dominant_count'] / window_analysis[window]['total_paths'])
        if window == 'early':
            interp = "Moderate stability, semantic diversity"
        elif window == 'middle':
            interp = "Transition phase: semantic to grammatical"
        else:
            interp = "Stabilized mixed organization"
        
        print(f"{window.capitalize()} & {stability[window]:.3f} & {frag:.3f} & {interp} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")

def main():
    """Generate all figures for expanded dataset."""
    print("=== GENERATING EXPANDED DATASET FIGURES ===")
    
    # Load results
    results = load_results()
    
    # Generate all visualizations
    print("\n1. Generating Sankey diagrams...")
    generate_sankey_diagrams(results)
    
    print("\n2. Generating trajectory visualizations...")
    generate_trajectory_visualizations(results)
    
    print("\n3. Generating cluster legend...")
    generate_cluster_legend(results)
    
    print("\n4. Calculating updated metrics...")
    generate_metrics_tables(results)
    
    print("\n=== ALL FIGURES GENERATED ===")

if __name__ == "__main__":
    main()