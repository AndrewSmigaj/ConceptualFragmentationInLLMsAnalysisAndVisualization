#!/usr/bin/env python3
"""
Generate Sankey diagrams for k-means clustering analysis.
Parameterized to work with any k value.
Usage: python generate_sankey_diagrams.py --k 10
"""

import json
import sys
import argparse
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SankeyGenerator:
    """Generate Sankey diagrams for k-means clustering results."""
    
    def __init__(self, base_dir: Path, k: int):
        self.base_dir = base_dir
        self.k = k
        self.results_dir = base_dir / f"k{k}_analysis_results"
        
        # Load analysis results
        windowed_path = self.results_dir / f"windowed_analysis_k{k}.json"
        if not windowed_path.exists():
            raise FileNotFoundError(f"Windowed analysis not found at {windowed_path}")
            
        with open(windowed_path, 'r') as f:
            self.windowed_results = json.load(f)
        
        # Load semantic labels
        labels_path = base_dir / f"llm_labels_k{k}" / f"cluster_labels_k{k}.json"
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                label_data = json.load(f)
                self.semantic_labels = label_data['labels']
        else:
            logging.warning(f"Semantic labels not found at {labels_path}")
            self.semantic_labels = {}
        
        # Load semantic purity data
        purity_path = base_dir / f"llm_labels_k{k}" / f"semantic_purity_k{k}.json"
        if purity_path.exists():
            with open(purity_path, 'r') as f:
                self.purity_data = json.load(f)
        else:
            logging.warning("Semantic purity data not found, labels will not include purity percentages")
            self.purity_data = {}
    
    def create_sankey_for_window(self, window_name: str, top_n: int = 15):
        """Create Sankey diagram for a specific window showing top N paths."""
        window_data = self.windowed_results[window_name]
        
        if 'archetypal_paths' not in window_data:
            logging.warning(f"No archetypal paths found for {window_name} window")
            return None
        
        # Get top N paths
        top_paths = window_data['archetypal_paths'][:top_n]
        
        # Extract layer range
        if window_name == 'early':
            layers = [0, 1, 2, 3]
        elif window_name == 'middle':
            layers = [4, 5, 6, 7]
        else:  # late
            layers = [8, 9, 10, 11]
        
        # Build nodes and links
        nodes = []
        node_map = {}  # (layer, cluster) -> node_index
        
        # Create nodes for each layer
        for layer in layers:
            layer_clusters = set()
            # Find all clusters used in top paths at this layer
            for path_info in top_paths:
                path = path_info['path']
                layer_idx = layer - layers[0]
                if layer_idx < len(path):
                    layer_clusters.add(path[layer_idx])
            
            # Create nodes for this layer
            for cluster in sorted(layer_clusters):
                node_idx = len(nodes)
                cluster_key = f"L{layer}_C{cluster}"
                layer_key = f"layer_{layer}"
                
                # Get semantic label with purity
                if layer_key in self.semantic_labels and cluster_key in self.semantic_labels[layer_key]:
                    base_label = self.semantic_labels[layer_key][cluster_key]['label']
                    
                    # Add purity percentage if available
                    if (layer_key in self.purity_data and 
                        cluster_key in self.purity_data[layer_key]):
                        purity = self.purity_data[layer_key][cluster_key]['purity']
                        label = f"{base_label} ({purity:.0f}%)" 
                    else:
                        label = base_label
                else:
                    label = f"C{cluster}"
                
                nodes.append({
                    'label': f"L{layer}: {label}",
                    'color': 'rgba(100, 100, 100, 0.8)'
                })
                node_map[(layer, cluster)] = node_idx
        
        # Build links from paths
        links = []
        link_counts = defaultdict(int)
        
        for path_info in top_paths:
            path = path_info['path']
            frequency = path_info['frequency']
            
            # Add links between consecutive layers
            for i in range(len(path) - 1):
                source_layer = layers[i]
                target_layer = layers[i + 1]
                source_cluster = path[i]
                target_cluster = path[i + 1]
                
                source_idx = node_map.get((source_layer, source_cluster))
                target_idx = node_map.get((target_layer, target_cluster))
                
                if source_idx is not None and target_idx is not None:
                    link_key = (source_idx, target_idx)
                    link_counts[link_key] += frequency
        
        # Convert to Sankey format
        for (source, target), value in link_counts.items():
            links.append({
                'source': source,
                'target': target,
                'value': value,
                'color': 'rgba(0, 0, 0, 0.2)'
            })
        
        # Create Sankey trace
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[n['label'] for n in nodes],
                color=[n['color'] for n in nodes]
            ),
            link=dict(
                source=[l['source'] for l in links],
                target=[l['target'] for l in links],
                value=[l['value'] for l in links],
                color=[l['color'] for l in links]
            )
        )])
        
        # Update layout
        fig.update_layout(
            title={
                'text': f"K={self.k} Token Flow - {window_name.capitalize()} Window (Top {top_n} Paths)",
                'x': 0.5,
                'xanchor': 'center'
            },
            font_size=12,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def generate_all_sankeys(self, top_n=15):
        """Generate Sankey diagrams for all windows."""
        for window in ['early', 'middle', 'late']:
            logging.info(f"Generating Sankey for {window} window...")
            
            fig = self.create_sankey_for_window(window, top_n=top_n)
            if fig:
                # Save HTML
                output_path = self.results_dir / f"sankey_{window}_k{self.k}.html"
                fig.write_html(str(output_path))
                logging.info(f"Saved {window} Sankey to {output_path}")
                
                # Also save static image if kaleido is installed
                try:
                    img_path = self.results_dir / f"sankey_{window}_k{self.k}.png"
                    fig.write_image(str(img_path), width=1200, height=600, scale=2)
                    logging.info(f"Saved static image to {img_path}")
                except:
                    pass
    
    def create_path_summary_table(self, top_n=15):
        """Create a summary table of top paths."""
        summary = [f"TOP {top_n} ARCHETYPAL PATHS BY WINDOW (K={self.k})", "=" * 60, ""]
        
        for window in ['early', 'middle', 'late']:
            window_data = self.windowed_results[window]
            summary.append(f"{window.upper()} WINDOW")
            summary.append("-" * 40)
            
            if 'archetypal_paths' in window_data:
                for i, path_info in enumerate(window_data['archetypal_paths'][:top_n]):
                    path = path_info['path']
                    freq = path_info['frequency']
                    examples = path_info.get('representative_words', path_info.get('example_words', []))[:5]
                    
                    summary.append(f"\nPath {i+1}: {' → '.join(map(str, path))}")
                    summary.append(f"Frequency: {freq} tokens")
                    
                    if 'semantic_labels' in path_info:
                        semantic = ' → '.join(path_info['semantic_labels'])
                        summary.append(f"Semantic: {semantic}")
                    
                    summary.append(f"Examples: {', '.join(examples)}")
            
            summary.append("")
        
        # Save summary
        summary_path = self.results_dir / f"top_paths_summary_k{self.k}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        logging.info(f"Saved path summary to {summary_path}")


def main():
    """Generate Sankey diagrams for specified k value."""
    parser = argparse.ArgumentParser(description='Generate Sankey diagrams for GPT-2 clustering analysis')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top paths to show')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    k = args.k
    
    print(f"\n{'='*60}")
    print(f"GENERATING SANKEY DIAGRAMS FOR K={k}")
    print(f"{'='*60}")
    
    # Check if windowed analysis exists
    results_dir = base_dir / f"k{k}_analysis_results"
    if not results_dir.exists():
        print(f"\nERROR: Analysis results not found at {results_dir}")
        print(f"Please run windowed analysis for k={k} first.")
        return
    
    # Initialize generator
    generator = SankeyGenerator(base_dir, k=k)
    
    # Generate all Sankey diagrams
    generator.generate_all_sankeys(top_n=args.top_n)
    
    # Create summary table
    generator.create_path_summary_table(top_n=args.top_n)
    
    print(f"\n{'='*60}")
    print(f"SANKEY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated outputs in {results_dir}:")
    print(f"  - sankey_early_k{k}.html")
    print(f"  - sankey_middle_k{k}.html")
    print(f"  - sankey_late_k{k}.html")
    print(f"  - top_paths_summary_k{k}.txt")


if __name__ == "__main__":
    main()