"""Configuration classes for visualization components."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class SankeyConfig:
    """Configuration for Sankey diagram generation.
    
    Attributes:
        top_n_paths: Number of top paths to display
        show_purity: Whether to show semantic purity percentages
        colored_paths: Whether to color code paths
        legend_position: Position of the legend ('left', 'right', 'top', 'bottom')
        last_layer_labels_position: Position of last layer labels ('left', 'right', 'none')
        width: Figure width in pixels
        height: Figure height in pixels
        node_pad: Padding between nodes
        node_thickness: Thickness of nodes
        link_opacity: Opacity of links (0-1)
        color_palette: List of colors for paths
        margin: Figure margins as (left, right, top, bottom)
        generate_summary: Whether to generate path summary file
        node_color: Color for nodes
    """
    top_n_paths: int = 25
    show_purity: bool = True
    colored_paths: bool = True
    legend_position: str = 'left'
    last_layer_labels_position: str = 'right'
    width: int = 1600
    height: int = 800
    node_pad: int = 15
    node_thickness: int = 20
    link_opacity: float = 0.6
    color_palette: Optional[List[str]] = None
    margin: Tuple[int, int, int, int] = (250, 150, 50, 50)
    generate_summary: bool = True
    node_color: str = 'rgba(100, 100, 100, 0.8)'
    
    def __post_init__(self):
        """Initialize default color palette if not provided."""
        if self.color_palette is None:
            self.color_palette = [
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


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory visualization.
    
    Attributes:
        reduction_method: Dimensionality reduction method ('umap', 'tsne', 'pca', 'none')
        dimensions: Number of dimensions (2 or 3)
        backend: Plotting backend ('plotly' or 'matplotlib')
        color_by: How to color points ('cluster', 'class', 'path', 'layer', 'endpoint', 'pos', 'metric')
        color_palette: Optional custom color palette
        show_arrows: Whether to show trajectory arrows
        arrow_size: Size of trajectory arrows
        point_size: Size of data points
        line_width: Width of trajectory lines
        opacity: Opacity of markers and lines
        layer_separation: How to separate layers ('stepped', 'sequential', 'none')
        layer_height: Height offset for stepped visualization
        show_cluster_labels: Whether to show cluster labels
        show_layer_labels: Whether to show layer labels
        show_legend: Whether to show legend
        show_averages: Whether to show average trajectories
        show_error_bars: Whether to show error bars
        show_centroids: Whether to show cluster centroids
        max_samples: Maximum samples to visualize (None for all)
        cache_reduction: Whether to cache dimensionality reduction
        width: Figure width in pixels
        height: Figure height in pixels
        margin: Figure margins
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        perplexity: Perplexity for t-SNE
        random_state: Random seed for reproducibility
    """
    reduction_method: str = 'umap'
    dimensions: int = 3
    backend: str = 'plotly'
    color_by: str = 'cluster'
    color_palette: Optional[List[str]] = None
    show_arrows: bool = True
    arrow_size: float = 0.1
    point_size: float = 3.0
    line_width: float = 2.0
    opacity: float = 0.8
    layer_separation: str = 'stepped'
    layer_height: float = 0.1
    show_cluster_labels: bool = True
    show_layer_labels: bool = True
    show_legend: bool = True
    show_averages: bool = False
    show_error_bars: bool = False
    show_centroids: bool = False
    max_samples: Optional[int] = None
    cache_reduction: bool = True
    width: int = 1000
    height: int = 800
    margin: Dict[str, int] = field(default_factory=lambda: {'l': 50, 'r': 50, 't': 50, 'b': 50})
    n_neighbors: int = 15
    min_dist: float = 0.1
    perplexity: int = 30
    random_state: Optional[int] = 42
    
    def __post_init__(self):
        """Validate configuration."""
        if self.reduction_method not in ['umap', 'tsne', 'pca', 'none']:
            raise ValueError(f"Invalid reduction_method: {self.reduction_method}")
            
        if self.dimensions not in [2, 3]:
            raise ValueError(f"dimensions must be 2 or 3, got {self.dimensions}")
            
        if self.backend not in ['plotly', 'matplotlib']:
            raise ValueError(f"Invalid backend: {self.backend}")
            
        if self.color_by not in ['cluster', 'class', 'path', 'layer', 'endpoint', 'pos', 'metric']:
            raise ValueError(f"Invalid color_by: {self.color_by}")
            
        if self.layer_separation not in ['stepped', 'sequential', 'none']:
            raise ValueError(f"Invalid layer_separation: {self.layer_separation}")


@dataclass 
class SteppedLayerConfig:
    """Configuration for stepped layer visualization.
    
    Attributes:
        layers_per_plot: Number of layers to show per subplot
        width: Figure width in pixels
        height: Figure height in pixels
        colormap: Matplotlib colormap name
        show_transitions: Whether to show transition lines
        transition_alpha: Opacity of transition lines
        node_size: Size of cluster nodes
    """
    layers_per_plot: int = 4
    width: int = 1200
    height: int = 800
    colormap: str = 'tab10'
    show_transitions: bool = True
    transition_alpha: float = 0.3
    node_size: float = 50.0