"""
Unified visualization module for semantic subtypes analysis.
Creates key visualizations while respecting existing patterns in the codebase.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys
from collections import defaultdict, Counter

# Add parent directories to path for imports
unified_cta_dir = Path(__file__).parent.parent
if str(unified_cta_dir) not in sys.path:
    sys.path.insert(0, str(unified_cta_dir))

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from logging_config import setup_logging

logger = setup_logging(__name__)


class UnifiedVisualizer:
    """
    Unified visualization interface for semantic subtypes analysis.
    Creates visualizations following existing patterns in the codebase.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer with output directory.
        
        Args:
            output_dir: Directory for saving visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized UnifiedVisualizer with output dir: {output_dir}")
    
    def create_trajectory_sankey(self,
                               trajectories: Dict[str, List[int]],
                               word_subtypes: Optional[Dict[str, str]] = None,
                               title: str = "Word Trajectories Through Clusters",
                               max_words: int = 100) -> Dict[str, Any]:
        """
        Create Sankey diagram of word trajectories.
        
        Following the pattern from existing gpt2_token_sankey.py
        """
        # Select subset of words
        words = list(trajectories.keys())[:max_words]
        n_layers = len(next(iter(trajectories.values())))
        
        # Build flow data
        sources = []
        targets = []
        values = []
        labels = []
        
        # Create node labels for each layer-cluster combination
        node_map = {}
        node_counter = 0
        
        for layer in range(n_layers):
            clusters_in_layer = set()
            for word in words:
                cluster = trajectories[word][layer]
                clusters_in_layer.add(cluster)
            
            for cluster in sorted(clusters_in_layer):
                node_key = (layer, cluster)
                if node_key not in node_map:
                    node_map[node_key] = node_counter
                    labels.append(f"L{layer}_C{cluster}")
                    node_counter += 1
        
        # Count transitions
        for layer in range(n_layers - 1):
            transition_counts = defaultdict(int)
            
            for word in words:
                from_cluster = trajectories[word][layer]
                to_cluster = trajectories[word][layer + 1]
                from_node = node_map[(layer, from_cluster)]
                to_node = node_map[(layer + 1, to_cluster)]
                transition_counts[(from_node, to_node)] += 1
            
            for (from_node, to_node), count in transition_counts.items():
                sources.append(from_node)
                targets.append(to_node)
                values.append(count)
        
        # Create Sankey figure
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=labels,
                color="lightblue"
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color="rgba(100, 100, 100, 0.3)"
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=10,
            height=600,
            width=1200
        )
        
        # Save figure
        output_path = self.output_dir / "trajectory_sankey.html"
        fig.write_html(str(output_path))
        
        logger.info(f"Created trajectory Sankey diagram: {output_path}")
        
        return {
            'figure': fig,
            'output_path': output_path,
            'n_words': len(words),
            'n_layers': n_layers,
            'n_nodes': len(labels)
        }
    
    def create_cluster_evolution_plot(self,
                                    macro_labels: Dict[int, np.ndarray]) -> Dict[str, Any]:
        """
        Visualize how clusters evolve across layers.
        """
        layers = sorted(macro_labels.keys())
        
        # Calculate metrics per layer
        n_clusters = []
        avg_cluster_size = []
        max_cluster_size = []
        min_cluster_size = []
        
        for layer in layers:
            labels = macro_labels[layer]
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            n_clusters.append(len(unique_labels))
            avg_cluster_size.append(np.mean(counts))
            max_cluster_size.append(np.max(counts))
            min_cluster_size.append(np.min(counts))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot 1: Number of clusters
        ax1.plot(layers, n_clusters, 'o-', linewidth=2, markersize=8)
        ax1.set_ylabel('Number of Clusters')
        ax1.set_title('Cluster Evolution Across Layers')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cluster sizes
        ax2.plot(layers, avg_cluster_size, 'o-', label='Average', linewidth=2)
        ax2.fill_between(layers, min_cluster_size, max_cluster_size, 
                        alpha=0.3, label='Min-Max Range')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Cluster Size')
        ax2.set_title('Cluster Size Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "cluster_evolution.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created cluster evolution plot: {output_path}")
        
        return {
            'figure': fig,
            'output_path': output_path,
            'n_clusters_per_layer': n_clusters,
            'avg_sizes': avg_cluster_size
        }
    
    def create_path_diversity_heatmap(self,
                                    trajectories: Dict[str, List[int]],
                                    word_subtypes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Create heatmap showing path diversity between layers.
        """
        # Convert to matrix
        traj_matrix = np.array(list(trajectories.values()))
        n_words, n_layers = traj_matrix.shape
        
        # Calculate transition diversity between each layer pair
        diversity_matrix = np.zeros((n_layers - 1, n_layers - 1))
        
        for i in range(n_layers - 1):
            for j in range(i, n_layers - 1):
                # Count unique transitions from layer i to j+1
                if i == j:
                    # Direct transition
                    transitions = set(zip(traj_matrix[:, i], traj_matrix[:, i+1]))
                else:
                    # Multi-hop transition
                    transitions = set(zip(traj_matrix[:, i], traj_matrix[:, j+1]))
                
                # Diversity = number of unique transitions / max possible
                max_possible = len(np.unique(traj_matrix[:, i])) * len(np.unique(traj_matrix[:, j+1]))
                diversity = len(transitions) / max_possible if max_possible > 0 else 0
                
                diversity_matrix[i, j] = diversity
                if i != j:
                    diversity_matrix[j, i] = diversity
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(diversity_matrix, cmap='YlOrRd', aspect='auto')
        
        # Labels
        layer_labels = [f"L{i}→{i+1}" for i in range(n_layers - 1)]
        ax.set_xticks(range(n_layers - 1))
        ax.set_yticks(range(n_layers - 1))
        ax.set_xticklabels(layer_labels, rotation=45, ha='right')
        ax.set_yticklabels(layer_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transition Diversity', rotation=270, labelpad=15)
        
        # Add values to cells
        for i in range(n_layers - 1):
            for j in range(n_layers - 1):
                text = ax.text(j, i, f'{diversity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if diversity_matrix[i, j] < 0.5 else "white")
        
        ax.set_title('Path Transition Diversity Between Layers')
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "path_diversity_heatmap.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created path diversity heatmap: {output_path}")
        
        return {
            'figure': fig,
            'output_path': output_path,
            'diversity_matrix': diversity_matrix,
            'mean_diversity': np.mean(diversity_matrix)
        }
    
    def create_archetypal_paths_visualization(self,
                                            archetypal_paths: List[Dict],
                                            max_paths: int = 20) -> Dict[str, Any]:
        """
        Visualize the most common archetypal paths.
        """
        # Select top paths
        paths_to_show = archetypal_paths[:max_paths]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Path frequencies
        frequencies = [p['frequency'] for p in paths_to_show]
        path_labels = [f"Path {i+1}" for i in range(len(paths_to_show))]
        
        bars = ax1.barh(path_labels, frequencies)
        ax1.set_xlabel('Frequency')
        ax1.set_title(f'Top {len(paths_to_show)} Archetypal Paths')
        
        # Color by stability
        for i, (bar, path) in enumerate(zip(bars, paths_to_show)):
            if path.get('is_stable', False):
                bar.set_color('green')
            elif path.get('is_wandering', False):
                bar.set_color('red')
            else:
                bar.set_color('orange')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Stable'),
            Patch(facecolor='orange', label='Moderate'),
            Patch(facecolor='red', label='Wandering')
        ]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Path characteristics scatter
        stabilities = [p.get('stability', 0) for p in paths_to_show]
        n_unique = [p.get('n_unique_clusters', 1) for p in paths_to_show]
        sizes = [p['frequency'] * 10 for p in paths_to_show]
        
        scatter = ax2.scatter(stabilities, n_unique, s=sizes, alpha=0.6, c=frequencies, cmap='viridis')
        ax2.set_xlabel('Path Stability')
        ax2.set_ylabel('Number of Unique Clusters Visited')
        ax2.set_title('Path Characteristics')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Frequency', rotation=270, labelpad=15)
        
        # Add example words for top 5 paths
        for i, path in enumerate(paths_to_show[:5]):
            example = ', '.join(path['example_words'][:3])
            ax2.annotate(f"P{i+1}: {example}", 
                        (stabilities[i], n_unique[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "archetypal_paths.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created archetypal paths visualization: {output_path}")
        
        return {
            'figure': fig,
            'output_path': output_path,
            'n_paths_shown': len(paths_to_show),
            'total_frequency': sum(frequencies)
        }
    
    def create_summary_dashboard(self,
                               trajectories: Dict[str, List[int]],
                               path_metrics: Dict,
                               cluster_names: Dict[Tuple[int, int], str],
                               archetypal_paths: List[Dict]) -> Dict[str, Any]:
        """
        Create a comprehensive summary dashboard.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Path metrics summary
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Fragmentation', 'Stability', 'Convergence']
        values = [
            path_metrics.get('fragmentation', 0),
            path_metrics.get('stability', 0),
            path_metrics.get('convergence', 0)
        ]
        bars = ax1.bar(metrics, values)
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        ax1.set_title('Path Quality Metrics')
        
        # Color bars
        for bar, val in zip(bars, values):
            bar.set_color('green' if val > 0.7 else 'orange' if val > 0.3 else 'red')
        
        # 2. Cluster count evolution
        ax2 = fig.add_subplot(gs[0, 1])
        layers = sorted(set(layer for layer, _ in cluster_names.keys()))
        clusters_per_layer = []
        for layer in layers:
            n_clusters = len([1 for (l, _) in cluster_names.keys() if l == layer])
            clusters_per_layer.append(n_clusters)
        
        ax2.plot(layers, clusters_per_layer, 's-', markersize=8, linewidth=2)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Cluster Count Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Path type distribution
        ax3 = fig.add_subplot(gs[0, 2])
        path_types = {'Stable': 0, 'Dynamic': 0, 'Wandering': 0}
        for path in archetypal_paths:
            if path.get('is_stable', False):
                path_types['Stable'] += path['frequency']
            elif path.get('is_wandering', False):
                path_types['Wandering'] += path['frequency']
            else:
                path_types['Dynamic'] += path['frequency']
        
        if sum(path_types.values()) > 0:
            wedges, texts, autotexts = ax3.pie(path_types.values(), 
                                               labels=path_types.keys(), 
                                               autopct='%1.1f%%',
                                               colors=['green', 'orange', 'red'])
            ax3.set_title('Path Type Distribution')
        
        # 4. Layer transition heatmap (simplified)
        ax4 = fig.add_subplot(gs[1, :2])
        n_layers = len(layers)
        transition_matrix = np.zeros((n_layers - 1, n_layers - 1))
        
        # Fill with transition counts
        if 'transition_counts' in path_metrics:
            for i, counts in path_metrics['transition_counts'].items():
                if isinstance(i, int) and i < n_layers - 1:
                    unique_transitions = len(counts)
                    transition_matrix[i, i] = unique_transitions
        
        im = ax4.imshow(transition_matrix, cmap='Blues', aspect='auto')
        ax4.set_xlabel('To Layer')
        ax4.set_ylabel('From Layer')
        ax4.set_title('Layer Transition Complexity')
        plt.colorbar(im, ax=ax4, label='Unique Transitions')
        
        # 5. Summary statistics
        ax5 = fig.add_subplot(gs[1:, 2])
        ax5.axis('off')
        
        # Calculate statistics
        n_words = len(trajectories)
        n_unique_paths = len(set(tuple(t) for t in trajectories.values()))
        n_layers_total = len(next(iter(trajectories.values())))
        n_clusters_total = len(cluster_names)
        n_archetypal = len(archetypal_paths)
        
        # Most common path
        if archetypal_paths:
            top_path = archetypal_paths[0]
            top_freq = top_path['frequency']
            top_pct = top_path['percentage']
        else:
            top_freq = 0
            top_pct = 0
        
        summary_text = f"""📊 UNIFIED CTA SUMMARY REPORT

Dataset Statistics:
• Total Words: {n_words}
• Unique Trajectories: {n_unique_paths}
• Path Diversity: {n_unique_paths/n_words:.2%}
• Total Layers: {n_layers_total}

Clustering Results:
• Total Named Clusters: {n_clusters_total}
• Avg Clusters/Layer: {n_clusters_total/n_layers_total:.1f}
• Cluster Evolution: {'Expanding' if clusters_per_layer[-1] > clusters_per_layer[0] else 'Consolidating'}

Path Analysis:
• Archetypal Paths: {n_archetypal}
• Most Common Path: {top_freq} words ({top_pct:.1f}%)
• Path Stability: {path_metrics.get('stability', 0):.3f}
• Path Convergence: {path_metrics.get('convergence', 0):.3f}

Quality Metrics:
• Fragmentation: {path_metrics.get('fragmentation', 0):.3f}
• Semantic Coherence: {path_metrics.get('semantic_coherence', 'N/A')}

Key Insights:
• Words show {('high' if path_metrics.get('stability', 0) > 0.7 else 'moderate' if path_metrics.get('stability', 0) > 0.3 else 'low')} stability across layers
• Clustering reveals {('strong' if n_unique_paths < n_words * 0.3 else 'moderate' if n_unique_paths < n_words * 0.7 else 'weak')} semantic organization
• Processing shows {('convergent' if path_metrics.get('convergence', 0) > 0.5 else 'divergent')} behavior
"""
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Save dashboard
        output_path = self.output_dir / "summary_dashboard.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Created summary dashboard: {output_path}")
        
        return {
            'figure': fig,
            'output_path': output_path,
            'statistics': {
                'n_words': n_words,
                'n_unique_paths': n_unique_paths,
                'n_clusters': n_clusters_total,
                'n_archetypal_paths': n_archetypal
            }
        }