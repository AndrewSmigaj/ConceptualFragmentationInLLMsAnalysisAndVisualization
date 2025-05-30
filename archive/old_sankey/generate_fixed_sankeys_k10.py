#!/usr/bin/env python3
"""
Generate fixed Sankey diagrams for k=10 clustering.
Fixes the issue where 10 labels are shown but only 7 clusters are visible.
This is a temporary fix until the full refactor is complete.
"""

import json
import sys
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


class FixedSankeyGenerator:
    """Generate Sankey diagrams with proper label handling."""
    
    def __init__(self, base_dir: Path, k: int = 10):
        self.base_dir = base_dir
        self.k = k
        self.results_dir = base_dir / f"k{k}_analysis_results"
        
        # Load analysis results
        windowed_path = self.results_dir / f"windowed_analysis_k{k}.json"
        with open(windowed_path, 'r') as f:
            self.windowed_results = json.load(f)
        
        # Load semantic labels
        labels_path = base_dir / f"llm_labels_k{k}" / f"cluster_labels_k{k}.json"
        with open(labels_path, 'r') as f:
            label_data = json.load(f)
            self.semantic_labels = label_data['labels']
        
        # Load semantic purity data
        purity_path = base_dir / f"llm_labels_k{k}" / f"semantic_purity_k{k}.json"
        if purity_path.exists():
            with open(purity_path, 'r') as f:
                self.purity_data = json.load(f)
        else:
            self.purity_data = {}
        
        # Define colors for top 25 paths
        self.path_colors = [
            'rgba(255, 99, 71, 0.6)',    # Tomato
            'rgba(30, 144, 255, 0.6)',   # Dodger Blue
            'rgba(50, 205, 50, 0.6)',    # Lime Green
            'rgba(255, 215, 0, 0.6)',    # Gold
            'rgba(138, 43, 226, 0.6)',   # Blue Violet
            'rgba(255, 140, 0, 0.6)',    # Dark Orange
            'rgba(0, 206, 209, 0.6)',    # Dark Turquoise
            'rgba(255, 20, 147, 0.6)',   # Deep Pink
            'rgba(154, 205, 50, 0.6)',   # Yellow Green
            'rgba(219, 112, 147, 0.6)',  # Pale Violet Red
            'rgba(100, 149, 237, 0.6)',  # Cornflower Blue
            'rgba(255, 182, 193, 0.6)',  # Light Pink
            'rgba(144, 238, 144, 0.6)',  # Light Green
            'rgba(255, 160, 122, 0.6)',  # Light Salmon
            'rgba(176, 196, 222, 0.6)',  # Light Steel Blue
            'rgba(220, 20, 60, 0.6)',    # Crimson
            'rgba(75, 0, 130, 0.6)',     # Indigo
            'rgba(255, 127, 80, 0.6)',   # Coral
            'rgba(0, 128, 128, 0.6)',    # Teal
            'rgba(240, 128, 128, 0.6)',  # Light Coral
            'rgba(32, 178, 170, 0.6)',   # Light Sea Green
            'rgba(250, 128, 114, 0.6)',  # Salmon
            'rgba(0, 191, 255, 0.6)',    # Deep Sky Blue
            'rgba(127, 255, 0, 0.6)',    # Chartreuse
            'rgba(255, 0, 255, 0.6)'     # Magenta
        ]
    
    def generate_path_description(self, semantic_labels):
        """Generate a concise description of a path based on its semantic transitions."""
        # Extract primary labels
        primary_labels = []
        for label in semantic_labels:
            if '(' in label:
                primary = label.split('(')[0].strip()
            else:
                primary = label
            primary_labels.append(primary)
        
        # Identify key transitions
        transitions = []
        for i in range(len(primary_labels) - 1):
            if primary_labels[i] != primary_labels[i+1]:
                transitions.append(f"{primary_labels[i]}→{primary_labels[i+1]}")
        
        # Create description based on path pattern
        if len(set(primary_labels)) == 1:
            return f"Pure {primary_labels[0]}"
        elif "Function Words" in primary_labels and "Content Words" in primary_labels:
            return "Function-Content Bridge"
        elif "Pronouns" in primary_labels and "Auxiliaries" in primary_labels:
            return "Grammar Integration"
        elif "Punctuation" in primary_labels:
            if primary_labels[0] == "Punctuation":
                return "Punctuation Origin"
            elif primary_labels[-1] == "Punctuation":
                return "Punctuation Convergence"
            else:
                return "Punctuation Transit"
        elif len(transitions) == 1:
            return f"{transitions[0]} Shift"
        else:
            # Use first and last for complex paths
            return f"{primary_labels[0]}→{primary_labels[-1]}"
    
    def create_fixed_sankey(self, window_name: str, top_n: int = 25):
        """Create Sankey diagram with fixed label handling."""
        window_data = self.windowed_results[window_name]
        
        if 'archetypal_paths' not in window_data:
            logging.warning(f"No archetypal paths found for {window_name} window")
            return None
        
        # Get top N paths
        top_paths = window_data['archetypal_paths'][:top_n]
        
        # Extract layer range
        layer_map = {
            'early': [0, 1, 2, 3],
            'middle': [4, 5, 6, 7],
            'late': [8, 9, 10, 11]
        }
        layers = layer_map[window_name]
        
        # First, find which clusters actually appear in the top paths
        clusters_by_layer = defaultdict(set)
        for path_info in top_paths:
            path = path_info['path']
            for layer_idx, cluster in enumerate(path):
                layer = layers[layer_idx]
                clusters_by_layer[layer].add(cluster)
        
        # Build nodes - only for clusters that appear in top paths
        nodes = []
        node_map = {}  # (layer, cluster) -> node_index
        
        # Create nodes for each layer
        for layer_idx, layer in enumerate(layers):
            layer_clusters = sorted(clusters_by_layer[layer])
            
            for cluster in layer_clusters:
                node_idx = len(nodes)
                cluster_key = f"L{layer}_C{cluster}"
                layer_key = f"layer_{layer}"
                
                # Get semantic label with purity
                if layer_key in self.semantic_labels and cluster_key in self.semantic_labels[layer_key]:
                    label_info = self.semantic_labels[layer_key][cluster_key]
                    base_label = label_info['label']
                    
                    # Add purity percentage if available
                    if (layer_key in self.purity_data and 
                        cluster_key in self.purity_data[layer_key]):
                        purity = self.purity_data[layer_key][cluster_key]['purity']
                        label = f"{base_label} ({purity:.0f}%)" 
                    else:
                        label = base_label
                else:
                    label = f"C{cluster}"
                
                nodes.append(f"L{layer}: {label}")
                node_map[(layer, cluster)] = node_idx
        
        # Track which paths use which links
        path_links = {}  # link_key -> list of path indices
        for path_idx, path_info in enumerate(top_paths):
            path = path_info['path']
            for i in range(len(path) - 1):
                source_layer = layers[i]
                target_layer = layers[i + 1]
                source_cluster = path[i]
                target_cluster = path[i + 1]
                
                source_idx = node_map.get((source_layer, source_cluster))
                target_idx = node_map.get((target_layer, target_cluster))
                
                if source_idx is not None and target_idx is not None:
                    link_key = (source_idx, target_idx)
                    if link_key not in path_links:
                        path_links[link_key] = []
                    path_links[link_key].append(path_idx)
        
        # Build links with appropriate values and colors
        links = []
        for link_key, path_indices in path_links.items():
            # Sum frequencies from all paths using this link
            total_freq = sum(top_paths[idx]['frequency'] for idx in path_indices)
            
            # Use the color of the lowest-indexed (highest-ranked) path
            primary_path = min(path_indices)
            color = self.path_colors[primary_path % len(self.path_colors)]
            
            links.append({
                'source': link_key[0],
                'target': link_key[1],
                'value': total_freq,
                'color': color
            })
        
        # Identify which nodes are in the last layer
        last_layer_nodes = []
        last_layer = layers[-1]
        for (layer, cluster), node_idx in node_map.items():
            if layer == last_layer:
                last_layer_nodes.append((node_idx, cluster))
        last_layer_nodes.sort(key=lambda x: x[1])  # Sort by cluster number
        
        # Create custom labels (empty for last layer)
        custom_labels = nodes.copy()
        for node_idx, _ in last_layer_nodes:
            custom_labels[node_idx] = ""  # Empty label for last layer nodes
        
        # Create Sankey trace
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=custom_labels,  # Use custom labels with empty last layer
                color='rgba(100, 100, 100, 0.8)'  # All nodes same color
            ),
            link=dict(
                source=[l['source'] for l in links],
                target=[l['target'] for l in links],
                value=[l['value'] for l in links],
                color=[l['color'] for l in links]
            )
        )])
        
        # Add annotations for path legend and last layer labels
        annotations = []
        
        # Add path legend on the LEFT
        legend_y_start = 0.95
        for i, path_info in enumerate(top_paths):
            # Generate path description
            path_desc = self.generate_path_description(path_info['semantic_labels'])
            frequency = path_info['frequency']
            
            # Create legend entry
            annotations.append(dict(
                x=-0.02,  # Left side
                y=legend_y_start - i * 0.035,
                text=f"<b>Path {i+1}:</b> {path_desc} ({frequency} tokens)",
                showarrow=False,
                xref='paper',
                yref='paper',
                xanchor='right',  # Right-align text on the left side
                font=dict(
                    size=9,
                    color=self.path_colors[i % len(self.path_colors)]
                )
            ))
        
        # Add labels for the last layer nodes (on the right side)
        # Only for nodes that actually exist
        if last_layer_nodes:
            # Calculate y positions based on number of actual nodes
            n_last_layer = len(last_layer_nodes)
            last_layer_y_positions = np.linspace(0.9, 0.1, n_last_layer)
            
            for i, (node_idx, cluster) in enumerate(last_layer_nodes):
                # Get the original label
                label_text = nodes[node_idx]
                
                # Add annotation to the right of the last layer
                annotations.append(dict(
                    x=0.98,  # Position to the right of the diagram
                    y=last_layer_y_positions[i],
                    text=label_text,
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    xanchor='left',
                    font=dict(size=10)
                ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"K={self.k} Token Flow - {window_name.capitalize()} Window (Top {top_n} Paths)",
                'x': 0.5,
                'xanchor': 'center'
            },
            font_size=12,
            height=800,
            width=1600,  # Wider to accommodate left legend
            margin=dict(l=250, r=150, t=50, b=50),  # Adjust margins for left legend
            annotations=annotations
        )
        
        return fig
    
    def generate_all_fixed_sankeys(self, top_n=25):
        """Generate fixed Sankey diagrams for all windows."""
        for window in ['early', 'middle', 'late']:
            logging.info(f"Generating fixed Sankey for {window} window...")
            
            fig = self.create_fixed_sankey(window, top_n=top_n)
            if fig:
                # Save HTML
                output_path = self.results_dir / f"sankey_{window}_k{self.k}_fixed.html"
                fig.write_html(str(output_path))
                logging.info(f"Saved fixed {window} Sankey to {output_path}")
                
                # Also save static image if kaleido is installed
                try:
                    img_path = self.results_dir / f"sankey_{window}_k{self.k}_fixed.png"
                    fig.write_image(str(img_path), width=1600, height=800, scale=2)
                    logging.info(f"Saved static image to {img_path}")
                except:
                    pass


def main():
    """Generate fixed Sankey diagrams."""
    base_dir = Path(__file__).parent
    
    print("============================================================")
    print("GENERATING FIXED K=10 SANKEY DIAGRAMS")
    print("============================================================")
    print("This fixes the issue where 10 labels show for 7 clusters")
    print("============================================================")
    
    generator = FixedSankeyGenerator(base_dir, k=10)
    generator.generate_all_fixed_sankeys()
    
    print("\n============================================================")
    print("FIXED SANKEY GENERATION COMPLETE")
    print("============================================================")
    print(f"\nGenerated outputs in {generator.results_dir}:")
    print(f"  - sankey_early_k10_fixed.html")
    print(f"  - sankey_middle_k10_fixed.html")
    print(f"  - sankey_late_k10_fixed.html")


if __name__ == "__main__":
    main()