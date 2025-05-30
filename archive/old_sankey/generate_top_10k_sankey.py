#!/usr/bin/env python3
"""
Generate Sankey diagrams for all GPT-2 tokens trajectory analysis.
Adapts the structured path approach from the paper for the full vocabulary.
"""

import json
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Top10kSankeyGenerator:
    def __init__(self, base_dir: Path, k: int = None):
        self.base_dir = base_dir
        if k:
            self.results_dir = base_dir / f"trajectory_analysis_k{k}"
            self.output_dir = base_dir / f"sankey_visualizations_k{k}"
        else:
            self.results_dir = base_dir / "trajectory_analysis"
            self.output_dir = base_dir / "sankey_visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        # Define windows (same as paper)
        self.windows = {
            'early': {'layers': [0, 1, 2, 3], 'name': 'Early (L0-L3)'},
            'middle': {'layers': [4, 5, 6, 7], 'name': 'Middle (L4-L7)'},
            'late': {'layers': [8, 9, 10, 11], 'name': 'Late (L8-L11)'}
        }
        
        # Load token information for examples
        logging.info("Loading token information...")
        with open(base_dir / "top_10k_tokens_full.json", 'r', encoding='utf-8') as f:
            token_list = json.load(f)
        
        self.token_id_to_str = {}
        for token_data in token_list:
            self.token_id_to_str[token_data['token_id']] = token_data['token_str']
    
    def load_trajectory_results(self):
        """Load the most recent trajectory analysis results."""
        # Find most recent results file
        result_files = list(self.results_dir.glob("trajectory_analysis_*.json"))
        if not result_files:
            raise FileNotFoundError("No trajectory analysis results found. Run trajectory analysis first.")
        
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        logging.info(f"Loading trajectory results from {latest_file}")
        
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def load_cluster_labels(self):
        """Load cluster labels from per-layer analysis to generate names."""
        # Try to load detailed results for cluster characteristics
        results_path = self.base_dir / "clustering_results_per_layer" / "per_layer_adaptive_results.json"
        if not results_path.exists():
            results_path = self.base_dir / "clustering_results_per_layer" / "per_layer_results.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                return json.load(f)
        return None
    
    def generate_cluster_label(self, layer: int, cluster: int, cluster_info: dict = None) -> str:
        """Generate descriptive label for a cluster based on its contents."""
        # If we have cluster analysis info, use it
        if cluster_info:
            # Try to generate meaningful label based on dominant token types
            # This is simplified - in practice you might use LLM analysis
            return f"L{layer}_C{cluster}"
        
        # Default format
        return f"L{layer}_C{cluster}"
    
    def create_structured_sankey(self, window_name: str, window_data: dict, 
                               trajectory_results: dict, max_paths: int = 15):
        """Create structured path Sankey diagram for a specific window."""
        
        window_info = self.windows[window_name]
        layers = window_info['layers']
        
        # Get archetypal paths for this window
        if window_name not in trajectory_results['window_analysis']:
            logging.warning(f"No data for {window_name} window")
            return None
        
        window_results = trajectory_results['window_analysis'][window_name]
        paths = window_results['archetypal_paths'][:max_paths]
        
        if not paths:
            logging.warning(f"No archetypal paths found for {window_name}")
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
        
        # Calculate vertical positions for paths
        n_paths = len(paths)
        
        # Distribute paths vertically based on their percentage
        # Higher percentage paths get more space
        total_percentage = sum(p['percentage'] for p in paths)
        y_current = 0.9
        y_positions = []
        
        for i, path in enumerate(paths):
            # Allocate vertical space proportional to path percentage
            path_height = (path['percentage'] / total_percentage) * 0.7
            y_center = y_current - path_height / 2
            y_positions.append(y_center)
            y_current -= path_height + 0.02  # Small gap between paths
        
        # Define colors based on token type dominance
        def get_path_color(path_info):
            """Get color based on dominant token type."""
            type_dist = path_info.get('token_type_distribution', {})
            if not type_dist:
                return 'rgba(149, 165, 166, 0.7)'  # Gray for unknown
            
            dominant_type = max(type_dist.items(), key=lambda x: x[1])[0]
            
            # Color mapping for token types
            type_colors = {
                'word': 'rgba(46, 204, 113, 0.7)',          # Green
                'word_with_space': 'rgba(52, 152, 219, 0.7)', # Blue
                'punctuation': 'rgba(231, 76, 60, 0.7)',     # Red
                'number': 'rgba(241, 196, 15, 0.7)',         # Yellow
                'subword': 'rgba(155, 89, 182, 0.7)',        # Purple
                'mixed': 'rgba(230, 126, 34, 0.7)',          # Orange
            }
            
            return type_colors.get(dominant_type, 'rgba(149, 165, 166, 0.7)')
        
        # Create nodes for each path at each layer
        node_map = {}
        node_idx = 0
        
        for layer_idx, actual_layer in enumerate(layers):
            for path_idx, path_info in enumerate(paths):
                node_key = f"L{layer_idx}_P{path_idx}"
                node_map[node_key] = node_idx
                
                # Parse cluster from path string
                path_str = path_info['path']
                clusters = self._parse_path_string(path_str)
                cluster_id = clusters[layer_idx] if layer_idx < len(clusters) else 0
                
                # Create informative node label
                dominant_morph = path_info.get('morphological_distribution', {})
                if dominant_morph and dominant_morph != {'no_pattern': 100.0}:
                    morph_label = max(dominant_morph.items(), key=lambda x: x[1])[0]
                    if morph_label != 'no_pattern':
                        node_label = f"Path {path_idx + 1} ({morph_label})"
                    else:
                        node_label = f"Path {path_idx + 1}"
                else:
                    node_label = f"Path {path_idx + 1}"
                
                # Add node
                nodes.append(node_label)
                node_positions_x.append(layer_x[layer_idx])
                node_positions_y.append(y_positions[path_idx])
                node_colors.append(get_path_color(path_info))
                node_labels.append(f"{self.generate_cluster_label(actual_layer, cluster_id)}")
                
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
                links['color'].append(get_path_color(path_info))
                
                # Create hover label with examples
                examples = path_info.get('example_tokens', [])[:3]
                example_str = ', '.join(repr(t) for t in examples)
                links['label'].append(
                    f"{path_info['count']} tokens ({path_info['percentage']:.1f}%)<br>"
                    f"Examples: {example_str}"
                )
        
        # Create Sankey figure
        fig = go.Figure(data=[go.Sankey(
            arrangement='fixed',
            domain=dict(x=[0.05, 0.95], y=[0.1, 0.9]),
            node=dict(
                pad=20,
                thickness=15,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=node_colors,
                x=node_positions_x,
                y=node_positions_y,
                hovertemplate='Path: %{label}<br>Cluster: %{customdata}<extra></extra>',
                customdata=node_labels
            ),
            link=dict(
                source=links['source'],
                target=links['target'],
                value=links['value'],
                color=links['color'],
                label=links['label'],
                hovertemplate='%{label}<extra></extra>'
            ),
            textfont=dict(size=10, color='black')
        )])
        
        # Add annotations
        annotations = []
        
        # Layer headers
        for i, (x, layer) in enumerate(zip(layer_x, layers)):
            paper_x = 0.05 + x * 0.9
            annotations.append(dict(
                x=paper_x,
                y=0.95,
                text=f"<b>Layer {layer}</b>",
                showarrow=False,
                xanchor='center',
                font=dict(size=14, color='black'),
                xref='paper',
                yref='paper'
            ))
        
        # Add cluster labels for major paths
        # (Only for top 5 paths to avoid clutter)
        for layer_idx, actual_layer in enumerate(layers):
            x = layer_x[layer_idx]
            
            # Group top paths by cluster
            cluster_groups = defaultdict(list)
            for path_idx, path_info in enumerate(paths[:5]):
                clusters = self._parse_path_string(path_info['path'])
                cluster_id = clusters[layer_idx] if layer_idx < len(clusters) else 0
                cluster_groups[cluster_id].append((path_idx, y_positions[path_idx]))
            
            # Add label for each cluster group
            for cluster_id, path_positions in cluster_groups.items():
                # Calculate center position for cluster label
                y_coords = [y for _, y in path_positions]
                label_y = sum(y_coords) / len(y_coords)
                
                paper_x = 0.05 + x * 0.9
                paper_y = 0.1 + label_y * 0.8
                
                annotations.append(dict(
                    x=paper_x,
                    y=paper_y,
                    text=f"<b>{self.generate_cluster_label(actual_layer, cluster_id)}</b>",
                    showarrow=False,
                    xanchor='center',
                    yanchor='middle',
                    font=dict(size=11, color='black'),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='black',
                    borderwidth=1,
                    borderpad=2,
                    xref='paper',
                    yref='paper'
                ))
        
        # Add title and layout
        fig.update_layout(
            title={
                'text': f"GPT-2 Top 10k Tokens {window_info['name']}: Top {len(paths)} Archetypal Paths",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=1200,
            width=1600,
            margin=dict(l=100, r=100, t=150, b=150),
            font=dict(size=12),
            annotations=annotations
        )
        
        # Add statistics at bottom
        total_tokens = window_results['total_tokens']
        unique_paths = window_results['num_unique_paths']
        coverage = sum(p['percentage'] for p in paths)
        
        stats_text = (f"<b>Window Statistics:</b> {total_tokens:,} tokens | "
                     f"{unique_paths:,} unique paths | "
                     f"Top {len(paths)} paths cover {coverage:.1f}% of tokens")
        
        fig.add_annotation(
            x=0.5,
            y=-0.05,
            text=stats_text,
            showarrow=False,
            xanchor='center',
            yanchor='top',
            font=dict(size=11),
            xref='paper',
            yref='paper'
        )
        
        # Add legend for token types
        legend_items = [
            ("Word", 'rgba(46, 204, 113, 0.7)'),
            ("Word with space", 'rgba(52, 152, 219, 0.7)'),
            ("Punctuation", 'rgba(231, 76, 60, 0.7)'),
            ("Number", 'rgba(241, 196, 15, 0.7)'),
            ("Subword", 'rgba(155, 89, 182, 0.7)'),
        ]
        
        legend_text = "<b>Token Type Colors:</b> " + " | ".join(
            f'<span style="color:{color}">■</span> {name}' 
            for name, color in legend_items
        )
        
        fig.add_annotation(
            x=0.5,
            y=-0.1,
            text=legend_text,
            showarrow=False,
            xanchor='center',
            yanchor='top',
            font=dict(size=10),
            xref='paper',
            yref='paper'
        )
        
        return fig
    
    def _parse_path_string(self, path_str: str) -> list:
        """Parse path string like 'L0_C1 → L1_C0 → ...' into cluster IDs."""
        clusters = []
        parts = path_str.split(' → ')
        for part in parts:
            # Extract cluster number from format L{layer}_C{cluster}
            if '_C' in part:
                cluster_id = int(part.split('_C')[1])
                clusters.append(cluster_id)
        return clusters
    
    def generate_all_sankeys(self):
        """Generate Sankey diagrams for all windows."""
        logging.info("Generating Sankey diagrams for all GPT-2 tokens...")
        
        # Load trajectory results
        trajectory_results = self.load_trajectory_results()
        
        # Load cluster info if available
        cluster_info = self.load_cluster_labels()
        
        # Generate for each window
        for window_name, window_data in self.windows.items():
            logging.info(f"\nGenerating {window_name} window Sankey...")
            
            fig = self.create_structured_sankey(
                window_name, 
                window_data, 
                trajectory_results,
                max_paths=15  # Show top 15 paths
            )
            
            if fig:
                # Save as HTML
                html_path = self.output_dir / f"top_10k_sankey_{window_name}.html"
                fig.write_html(str(html_path))
                logging.info(f"  Saved HTML: {html_path}")
                
                # Save as PNG if kaleido is installed
                try:
                    png_path = self.output_dir / f"top_10k_sankey_{window_name}.png"
                    fig.write_image(str(png_path), width=1600, height=1200, scale=2)
                    logging.info(f"  Saved PNG: {png_path}")
                except Exception as e:
                    logging.warning(f"  Could not save PNG (install kaleido): {e}")
        
        logging.info("\nAll Sankey diagrams generated successfully!")
        
        # Also create a combined view showing all three windows
        self.create_combined_view(trajectory_results)
    
    def create_combined_view(self, trajectory_results):
        """Create a single HTML file with all three windows for easy comparison."""
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>GPT-2 All Tokens Trajectory Analysis - Sankey Diagrams</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .window-container {
            margin: 30px 0;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stats {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        iframe {
            width: 100%;
            height: 1200px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>GPT-2 Top 10k Tokens Trajectory Analysis</h1>
    <div class="stats">
        <h3>Overall Statistics</h3>
        <p><strong>Total Tokens:</strong> 10,000 (most common)</p>
        <p><strong>Analysis:</strong> Cross-layer trajectory patterns showing how different token types 
        (words, subwords, punctuation, numbers) flow through GPT-2's 12 layers.</p>
    </div>
"""
        
        for window_name in ['early', 'middle', 'late']:
            window_data = trajectory_results['window_analysis'].get(window_name, {})
            if window_data:
                html_content += f"""
    <div class="window-container">
        <h2>{window_name.capitalize()} Window (Layers {self.windows[window_name]['layers'][0]}-{self.windows[window_name]['layers'][-1]})</h2>
        <div class="stats">
            <p><strong>Unique Paths:</strong> {window_data['num_unique_paths']:,}</p>
            <p><strong>Path Diversity:</strong> {window_data['metrics']['diversity']:.3f}</p>
            <p><strong>Top Path Concentration:</strong> {window_data['metrics']['concentration']:.1%}</p>
            <p><strong>Entropy:</strong> {window_data['metrics']['entropy']:.3f}</p>
        </div>
        <iframe src="top_10k_sankey_{window_name}.html"></iframe>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        combined_path = self.output_dir / "top_10k_sankey_combined.html"
        with open(combined_path, 'w') as f:
            f.write(html_content)
        
        logging.info(f"Created combined view: {combined_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Sankey diagrams for trajectory analysis')
    parser.add_argument('--k', type=int, default=None, help='Number of clusters (default: use optimal)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    generator = Top10kSankeyGenerator(base_dir, k=args.k)
    generator.generate_all_sankeys()


if __name__ == "__main__":
    main()