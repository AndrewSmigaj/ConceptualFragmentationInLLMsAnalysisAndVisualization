"""Unified Sankey diagram generator for concept trajectory analysis."""

from typing import Dict, List, Set, Tuple, Any, Optional, TypedDict, Union
from pathlib import Path
from collections import defaultdict, Counter
import logging
import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    
from .base import BaseVisualizer
from .configs import SankeyConfig
from .exceptions import VisualizationError, InvalidDataError

logger = logging.getLogger(__name__)


# Data structure definitions
class PathInfo(TypedDict):
    """Structure for path information."""
    path: List[int]
    frequency: int
    representative_words: List[str]
    semantic_labels: Optional[List[str]]
    percentage: Optional[float]


class WindowedAnalysis(TypedDict):
    """Structure for windowed analysis results."""
    layers: List[int]
    total_paths: int
    unique_paths: int
    archetypal_paths: List[PathInfo]


class SankeyData(TypedDict):
    """Structure for Sankey input data."""
    windowed_analysis: Dict[str, WindowedAnalysis]
    labels: Dict[str, Dict[str, Dict[str, Any]]]
    purity_data: Optional[Dict[str, Dict[str, Dict[str, float]]]]


class SankeyGenerator(BaseVisualizer):
    """Unified Sankey diagram generator for concept trajectory analysis.
    
    This class consolidates all Sankey functionality into a single configurable
    implementation, supporting colored paths, semantic labels, and various
    layout options.
    """
    
    def __init__(self, config: Optional[SankeyConfig] = None):
        """Initialize Sankey generator.
        
        Args:
            config: SankeyConfig object or None for defaults
        """
        if go is None:
            raise ImportError("plotly is required for Sankey visualization")
            
        super().__init__(config or SankeyConfig())
        
    def create_figure(self, 
                     data: Dict[str, Any],
                     window: str = 'early',
                     **kwargs) -> go.Figure:
        """Create Sankey diagram for specified window.
        
        Args:
            data: Dictionary containing windowed_analysis, labels, and purity_data
            window: Which window to visualize ('early', 'middle', 'late')
            **kwargs: Additional options to override config
            
        Returns:
            Plotly Figure object
            
        Raises:
            InvalidDataError: If input data is invalid
            VisualizationError: If creation fails
        """
        try:
            # Update config with any kwargs
            self.update_config(**kwargs)
            
            # Validate input data
            self._validate_sankey_data(data)
            
            # Check window validity
            if window not in data['windowed_analysis']:
                raise InvalidDataError(f"Window '{window}' not found in data")
                
            # Extract window data
            window_data = data['windowed_analysis'][window]
            labels = data.get('labels', {})
            purity_data = data.get('purity_data', {})
            
            # Get top paths
            if 'archetypal_paths' not in window_data:
                raise InvalidDataError(f"No archetypal paths found for {window} window")
                
            top_paths = window_data['archetypal_paths'][:self.config.top_n_paths]
            layers = window_data['layers']
            
            # Find visible clusters
            clusters_by_layer = self._find_visible_clusters(top_paths, layers)
            
            # Build nodes
            nodes, node_map = self._build_nodes(
                clusters_by_layer, layers, labels, purity_data
            )
            
            # Build links with colors
            links, path_colors = self._build_links(
                top_paths, layers, node_map
            )
            
            # Create figure
            fig = self._create_sankey_figure(
                nodes, links, window, len(top_paths)
            )
            
            # Add annotations (legend and labels)
            if self.config.legend_position != 'none':
                self._add_path_legend(fig, top_paths, path_colors)
                
            if self.config.last_layer_labels_position != 'none':
                self._add_last_layer_labels(
                    fig, nodes, node_map, layers[-1], clusters_by_layer[layers[-1]]
                )
                
            return fig
            
        except (InvalidDataError, VisualizationError):
            # Re-raise these specific exceptions without wrapping
            raise
        except Exception as e:
            logger.error(f"Failed to create Sankey: {e}")
            raise VisualizationError(f"Sankey creation failed: {e}")
            
    def _validate_sankey_data(self, data: Dict[str, Any]) -> None:
        """Validate input data structure."""
        if not isinstance(data, dict):
            raise InvalidDataError("Data must be a dictionary")
            
        if 'windowed_analysis' not in data:
            raise InvalidDataError("Missing 'windowed_analysis' in data")
            
        if not isinstance(data['windowed_analysis'], dict):
            raise InvalidDataError("'windowed_analysis' must be a dictionary")
            
    def _find_visible_clusters(self, 
                              paths: List[PathInfo],
                              layers: List[int]) -> Dict[int, Set[int]]:
        """Find which clusters actually appear in the paths.
        
        Args:
            paths: List of path information
            layers: List of layer indices
            
        Returns:
            Dictionary mapping layer to set of cluster IDs
        """
        clusters_by_layer = defaultdict(set)
        
        for path_info in paths:
            path = path_info['path']
            for layer_idx, cluster in enumerate(path):
                if layer_idx < len(layers):
                    layer = layers[layer_idx]
                    clusters_by_layer[layer].add(cluster)
                    
        return dict(clusters_by_layer)
        
    def _build_nodes(self,
                    clusters_by_layer: Dict[int, Set[int]],
                    layers: List[int],
                    labels: Dict[str, Dict[str, Dict[str, Any]]],
                    purity_data: Dict[str, Dict[str, Dict[str, float]]]) -> Tuple[List[str], Dict[Tuple[int, int], int]]:
        """Build node list and mapping.
        
        Returns:
            Tuple of (node_labels, node_map)
        """
        nodes = []
        node_map = {}  # (layer, cluster) -> node_index
        
        for layer in layers:
            layer_clusters = sorted(clusters_by_layer.get(layer, []))
            
            for cluster in layer_clusters:
                node_idx = len(nodes)
                cluster_key = f"L{layer}_C{cluster}"
                layer_key = f"L{layer}"
                
                # Get semantic label
                label = f"C{cluster}"  # Default
                
                if layer_key in labels and cluster_key in labels[layer_key]:
                    label_info = labels[layer_key][cluster_key]
                    label = label_info.get('label', label)
                    
                    # Add purity if available and configured
                    if (self.config.show_purity and 
                        purity_data is not None and
                        layer_key in purity_data and 
                        cluster_key in purity_data[layer_key]):
                        purity = purity_data[layer_key][cluster_key].get('purity', 0)
                        label = f"{label} ({purity:.0f}%)"
                        
                nodes.append(f"L{layer}: {label}")
                node_map[(layer, cluster)] = node_idx
                
        return nodes, node_map
        
    def _build_links(self,
                    paths: List[PathInfo],
                    layers: List[int],
                    node_map: Dict[Tuple[int, int], int]) -> Tuple[List[Dict], Dict[int, str]]:
        """Build links with appropriate colors.
        
        Returns:
            Tuple of (links, path_colors)
        """
        # Track which paths use which links
        path_links = defaultdict(list)  # link_key -> list of path indices
        
        for path_idx, path_info in enumerate(paths):
            path = path_info['path']
            for i in range(len(path) - 1):
                if i < len(layers) - 1:
                    source_layer = layers[i]
                    target_layer = layers[i + 1]
                    source_cluster = path[i]
                    target_cluster = path[i + 1]
                    
                    source_idx = node_map.get((source_layer, source_cluster))
                    target_idx = node_map.get((target_layer, target_cluster))
                    
                    if source_idx is not None and target_idx is not None:
                        link_key = (source_idx, target_idx)
                        path_links[link_key].append(path_idx)
                        
        # Assign colors to paths
        path_colors = self._assign_path_colors(len(paths))
        
        # Build links with colors and values
        links = []
        for link_key, path_indices in path_links.items():
            # Sum frequencies from all paths using this link
            total_freq = sum(paths[idx]['frequency'] for idx in path_indices)
            
            # Use color of the lowest-indexed (highest-ranked) path
            primary_path = min(path_indices)
            color = path_colors[primary_path]
            
            links.append({
                'source': link_key[0],
                'target': link_key[1],
                'value': total_freq,
                'color': color
            })
            
        return links, path_colors
        
    def _assign_path_colors(self, n_paths: int) -> Dict[int, str]:
        """Assign colors to paths.
        
        Args:
            n_paths: Number of paths to color
            
        Returns:
            Dictionary mapping path index to color
        """
        colors = {}
        palette = self.config.color_palette
        
        for i in range(n_paths):
            colors[i] = palette[i % len(palette)]
            
        return colors
        
    def _create_sankey_figure(self,
                             nodes: List[str],
                             links: List[Dict],
                             window: str,
                             n_paths: int) -> go.Figure:
        """Create the base Sankey figure."""
        # Prepare node labels (empty for last layer if configured)
        display_labels = nodes.copy()
        
        # Create Sankey trace
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=self.config.node_pad,
                thickness=self.config.node_thickness,
                line=dict(color="black", width=0.5),
                label=display_labels,
                color=self.config.node_color
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
                'text': f"Apple Quality Routing Through Neural Network - {window.capitalize()} View (Top {n_paths} Paths)",
                'x': 0.5,
                'xanchor': 'center'
            },
            font_size=12,
            height=self.config.height,
            width=self.config.width,
            margin=dict(
                l=self.config.margin[0],
                r=self.config.margin[1],
                t=self.config.margin[2],
                b=self.config.margin[3]
            )
        )
        
        return fig
        
    def _generate_path_description(self, semantic_labels: List[str]) -> str:
        """Generate concise description of a path.
        
        Args:
            semantic_labels: List of semantic labels for the path
            
        Returns:
            Path description string
        """
        if not semantic_labels:
            return "Unknown Path"
            
        # Extract primary labels (before parentheses)
        primary_labels = []
        for label in semantic_labels:
            if '(' in label:
                primary = label.split('(')[0].strip()
            else:
                primary = label
            primary_labels.append(primary)
            
        # Count occurrences of each label type
        label_counts = Counter(primary_labels)
        total_labels = len(primary_labels)
        
        # Check for pure paths (all same label)
        if len(label_counts) == 1:
            label = list(label_counts.keys())[0]
            return f"Pure {label}"
            
        # Check for dominant patterns
        most_common = label_counts.most_common(2)
        dominant_label, dominant_count = most_common[0]
        
        if dominant_count >= total_labels * 0.7:  # 70% threshold
            # Mostly one type with some transitions
            other_labels = [l for l, _ in most_common[1:]]
            if len(other_labels) == 1:
                return f"{dominant_label}-{other_labels[0]} Bridge"
            else:
                return f"Mostly {dominant_label}"
                
        # Analyze transitions
        transitions = []
        for i in range(len(primary_labels) - 1):
            if primary_labels[i] != primary_labels[i + 1]:
                transitions.append((primary_labels[i], primary_labels[i + 1]))
                
        # Identify key patterns
        if len(transitions) == 1:
            # Single transition
            return f"{transitions[0][0]}→{transitions[0][1]}"
            
        # Check for common patterns
        first_label = primary_labels[0]
        last_label = primary_labels[-1]
        
        # Function words patterns
        if "Function Words" in primary_labels:
            if "Content Words" in primary_labels:
                if first_label == "Function Words":
                    return "Function→Content"
                else:
                    return "Content→Function"
            elif "Technical/Foreign" in primary_labels:
                return "Function-Technical Bridge"
                
        # Multi-transition paths
        unique_sequence = []
        for label in primary_labels:
            if not unique_sequence or label != unique_sequence[-1]:
                unique_sequence.append(label)
                
        if len(unique_sequence) <= 3:
            return "→".join(unique_sequence)
        else:
            # Complex path - show start and end
            return f"{first_label}→...→{last_label}"
            
    def _add_path_legend(self,
                        fig: go.Figure,
                        paths: List[PathInfo],
                        path_colors: Dict[int, str]) -> None:
        """Add legend annotations for paths."""
        annotations = []
        
        # Determine position
        if self.config.legend_position == 'left':
            x_pos, x_anchor = -0.02, 'right'
        elif self.config.legend_position == 'right':
            x_pos, x_anchor = 1.02, 'left'
        else:
            return  # top/bottom not implemented yet
            
        y_start = 0.95
        y_step = 0.035
        
        for i, path_info in enumerate(paths):
            # Generate description
            desc = self._generate_path_description(
                path_info.get('semantic_labels', [])
            )
            freq = path_info['frequency']
            
            annotations.append(dict(
                x=x_pos,
                y=y_start - i * y_step,
                text=f"<b>Path {i+1}:</b> {desc} ({freq} samples)",
                showarrow=False,
                xref='paper',
                yref='paper',
                xanchor=x_anchor,
                font=dict(
                    size=12,
                    color=path_colors[i]
                )
            ))
            
        fig.update_layout(annotations=annotations)
        
    def _add_last_layer_labels(self,
                              fig: go.Figure,
                              nodes: List[str],
                              node_map: Dict[Tuple[int, int], int],
                              last_layer: int,
                              last_layer_clusters: Set[int]) -> None:
        """Add labels for last layer nodes."""
        if self.config.last_layer_labels_position == 'inline':
            return  # Labels already in nodes
            
        # Hide inline labels for last layer nodes
        label_list = list(fig.data[0].node.label)
        for cluster in last_layer_clusters:
            node_idx = node_map.get((last_layer, cluster))
            if node_idx is not None:
                label_list[node_idx] = ""
        
        # Update the figure with modified labels
        fig.data[0].node.label = label_list
                
        # Add annotations
        annotations = list(fig.layout.annotations) if fig.layout.annotations else []
        
        # Position based on config
        if self.config.last_layer_labels_position == 'right':
            x_pos, x_anchor = 0.98, 'left'
        else:  # left
            x_pos, x_anchor = 0.02, 'right'
            
        # Calculate y positions
        n_clusters = len(last_layer_clusters)
        y_positions = np.linspace(0.9, 0.1, n_clusters)
        
        for i, cluster in enumerate(sorted(last_layer_clusters)):
            node_idx = node_map.get((last_layer, cluster))
            if node_idx is not None:
                label_text = nodes[node_idx]
                
                annotations.append(dict(
                    x=x_pos,
                    y=y_positions[i],
                    text=label_text,
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    xanchor=x_anchor,
                    font=dict(size=10)
                ))
                
        fig.update_layout(annotations=annotations)
        
    def save_figure(self,
                   fig: go.Figure,
                   output_path: Union[str, Path],
                   format: str = 'html',
                   **kwargs) -> None:
        """Save figure to file.
        
        Args:
            fig: Plotly figure to save
            output_path: Output file path
            format: Output format ('html', 'png', 'pdf', 'svg')
            **kwargs: Additional save parameters
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'html':
            fig.write_html(str(output_path), **kwargs)
        elif format in ['png', 'pdf', 'svg', 'jpeg']:
            # Requires kaleido
            fig.write_image(str(output_path), format=format, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        logger.info(f"Saved Sankey diagram to {output_path}")
        
    def create_path_summary(self,
                           data: SankeyData,
                           output_format: str = 'markdown') -> str:
        """Create summary table of paths.
        
        Args:
            data: Sankey data
            output_format: Format for output ('markdown', 'text', 'html')
            
        Returns:
            Formatted summary string
        """
        lines = []
        
        if output_format == 'markdown':
            lines.append("# Archetypal Paths Summary\n")
            
        for window, window_data in data['windowed_analysis'].items():
            if output_format == 'markdown':
                lines.append(f"## {window.capitalize()} Window\n")
            else:
                lines.append(f"{window.upper()} WINDOW")
                lines.append("-" * 40)
                
            paths = window_data.get('archetypal_paths', [])[:self.config.top_n_paths]
            
            for i, path_info in enumerate(paths):
                path = path_info['path']
                freq = path_info['frequency']
                examples = path_info.get('representative_words', [])[:5]
                
                if output_format == 'markdown':
                    lines.append(f"### Path {i+1}: {' → '.join(map(str, path))}")
                    lines.append(f"- **Frequency**: {freq} tokens")
                    lines.append(f"- **Examples**: {', '.join(examples)}")
                    
                    if 'semantic_labels' in path_info:
                        labels = ' → '.join(path_info['semantic_labels'])
                        lines.append(f"- **Semantic**: {labels}")
                    lines.append("")
                else:
                    lines.append(f"\nPath {i+1}: {' → '.join(map(str, path))}")
                    lines.append(f"Frequency: {freq} tokens")
                    lines.append(f"Examples: {', '.join(examples)}")
                    
                    if 'semantic_labels' in path_info:
                        labels = ' → '.join(path_info['semantic_labels'])
                        lines.append(f"Semantic: {labels}")
                        
        return '\n'.join(lines)
        
    def _infer_k_value(self, nodes: List[str]) -> int:
        """Infer k value from the number of unique clusters in nodes.
        
        Args:
            nodes: List of node labels
            
        Returns:
            Inferred k value
        """
        # Extract cluster numbers from node labels
        clusters = set()
        for node in nodes:
            # Node format is "L{layer}: {label} ({purity}%)" or similar
            if 'C' in node:
                # Extract cluster number
                import re
                match = re.search(r'C(\d+)', node)
                if match:
                    clusters.add(int(match.group(1)))
                    
        # Return max cluster ID + 1 as k value
        return max(clusters) + 1 if clusters else 10
        
    def create_all_windows(self,
                          data: SankeyData,
                          output_dir: Union[str, Path],
                          format: str = 'html',
                          **kwargs) -> Dict[str, go.Figure]:
        """Create Sankey diagrams for all windows.
        
        Args:
            data: Sankey data with all windows
            output_dir: Directory to save outputs
            format: Output format
            **kwargs: Additional options
            
        Returns:
            Dictionary mapping window name to figure
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        for window in data['windowed_analysis'].keys():
            logger.info(f"Creating Sankey for {window} window")
            
            # Create figure
            fig = self.create_figure(data, window, **kwargs)
            figures[window] = fig
            
            # Save figure
            output_path = output_dir / f"sankey_{window}_k{self.config.top_n_paths}.{format}"
            self.save_figure(fig, output_path, format=format)
            
        # Save summary if requested
        if self.config.generate_summary:
            summary_path = output_dir / f"path_summary_k{self.config.top_n_paths}.md"
            summary = self.create_path_summary(data, output_format='markdown')
            summary_path.write_text(summary, encoding='utf-8')
            logger.info(f"Saved path summary to {summary_path}")
            
        return figures