"""Unified trajectory visualizer for concept movement analysis."""

from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict
import warnings

# Import visualization libraries with graceful fallback
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.proj3d import proj_transform
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Import dimensionality reduction methods
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    UMAP = None

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    TSNE = None
    PCA = None
    StandardScaler = None

from .base import BaseVisualizer
from .configs import TrajectoryConfig
from .exceptions import VisualizationError, InvalidDataError

logger = logging.getLogger(__name__)


class TrajectoryVisualizer(BaseVisualizer):
    """Unified trajectory visualizer for concept movement analysis.
    
    This class consolidates all trajectory visualization functionality,
    supporting both 2D and 3D visualizations with multiple backends.
    """
    
    def __init__(self, config: Optional[TrajectoryConfig] = None):
        """Initialize trajectory visualizer.
        
        Args:
            config: TrajectoryConfig object or None for defaults
            
        Raises:
            ImportError: If required backend is not available
        """
        super().__init__(config or TrajectoryConfig())
        
        # Check backend availability
        if self.config.backend == 'plotly' and not HAS_PLOTLY:
            raise ImportError("Plotly is required for plotly backend")
        elif self.config.backend == 'matplotlib' and not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for matplotlib backend")
            
        # Check dimensionality reduction availability
        if self.config.reduction_method == 'umap' and not HAS_UMAP:
            raise ImportError("UMAP is required for UMAP reduction")
        elif self.config.reduction_method in ['tsne', 'pca'] and not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for t-SNE/PCA reduction")
            
        # Cache for reduced data
        self._reduction_cache = {}
        self._reducer = None
        
    def create_figure(self, 
                     data: Dict[str, Any],
                     window: Optional[str] = None,
                     **kwargs) -> Union[go.Figure, plt.Figure]:
        """Create trajectory visualization.
        
        Args:
            data: Dictionary containing:
                - activations: Activation data (n_samples, n_layers, n_features)
                - labels: Optional label dictionaries
                - paths: Optional path assignments
                - metadata: Optional sample metadata
                - cluster_centers: Optional cluster centroids
                - windows: Optional window definitions
            window: Optional window to visualize ('early', 'middle', 'late')
            **kwargs: Additional options to override config
            
        Returns:
            Figure object (Plotly or Matplotlib)
            
        Raises:
            InvalidDataError: If input data is invalid
            VisualizationError: If creation fails
        """
        try:
            # Update config with any kwargs
            self.update_config(**kwargs)
            
            # Validate input data
            self._validate_trajectory_data(data)
            
            # Extract activations and apply windowing if needed
            activations, layer_indices = self._prepare_activations(data, window)
            
            # Apply dimensionality reduction
            reduced_data = self._reduce_dimensions(activations)
            
            # Create visualization based on backend
            if self.config.backend == 'plotly':
                fig = self._create_plotly_figure(
                    reduced_data, data, layer_indices
                )
            else:  # matplotlib
                fig = self._create_matplotlib_figure(
                    reduced_data, data, layer_indices
                )
                
            return fig
            
        except (InvalidDataError, VisualizationError):
            raise
        except Exception as e:
            logger.error(f"Failed to create trajectory visualization: {e}")
            raise VisualizationError(f"Trajectory creation failed: {e}")
            
    def _validate_trajectory_data(self, data: Dict[str, Any]) -> None:
        """Validate input data structure."""
        if not isinstance(data, dict):
            raise InvalidDataError("Data must be a dictionary")
            
        if 'activations' not in data:
            raise InvalidDataError("Missing 'activations' in data")
            
        activations = data['activations']
        if not isinstance(activations, np.ndarray):
            raise InvalidDataError("'activations' must be a numpy array")
            
        if activations.ndim != 3:
            raise InvalidDataError(
                f"'activations' must be 3D (samples, layers, features), "
                f"got shape {activations.shape}"
            )
            
    def _prepare_activations(self, 
                           data: Dict[str, Any], 
                           window: Optional[str]) -> Tuple[np.ndarray, List[int]]:
        """Prepare activations and determine layer indices.
        
        Args:
            data: Input data dictionary
            window: Optional window name
            
        Returns:
            Tuple of (prepared activations, layer indices)
        """
        activations = data['activations']
        n_samples, n_layers, n_features = activations.shape
        
        # Determine layer indices based on window
        if window and 'windows' in data:
            windows = data['windows']
            if window not in windows:
                raise InvalidDataError(f"Window '{window}' not found in data")
            layer_indices = windows[window]
            activations = activations[:, layer_indices, :]
        else:
            layer_indices = list(range(n_layers))
            
        # Apply sample limit if configured
        if self.config.max_samples and n_samples > self.config.max_samples:
            indices = np.random.choice(
                n_samples, self.config.max_samples, replace=False
            )
            activations = activations[indices]
            logger.info(f"Subsampled to {self.config.max_samples} samples")
            
        return activations, layer_indices
        
    def _reduce_dimensions(self, activations: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to activations.
        
        Args:
            activations: Activation array (n_samples, n_layers, n_features)
            
        Returns:
            Reduced array (n_samples, n_layers, n_components)
        """
        n_samples, n_layers, n_features = activations.shape
        
        # Check cache if enabled
        cache_key = (activations.shape, self.config.reduction_method, 
                     self.config.dimensions, self.config.random_state)
        if self.config.cache_reduction and cache_key in self._reduction_cache:
            logger.info("Using cached dimensionality reduction")
            return self._reduction_cache[cache_key]
            
        # Skip reduction if not needed
        if self.config.reduction_method == 'none':
            if n_features > 3:
                logger.warning(
                    f"No reduction applied but features ({n_features}) > 3. "
                    "Visualization may be limited."
                )
            return activations
            
        # Flatten for reduction
        flat_activations = activations.reshape(-1, n_features)
        
        # Standardize if using sklearn methods
        if self.config.reduction_method in ['tsne', 'pca']:
            scaler = StandardScaler()
            flat_activations = scaler.fit_transform(flat_activations)
            
        # Apply reduction
        if self.config.reduction_method == 'umap':
            reducer = UMAP(
                n_components=self.config.dimensions,
                n_neighbors=self.config.n_neighbors,
                min_dist=self.config.min_dist,
                random_state=self.config.random_state
            )
        elif self.config.reduction_method == 'tsne':
            reducer = TSNE(
                n_components=self.config.dimensions,
                perplexity=self.config.perplexity,
                random_state=self.config.random_state
            )
        elif self.config.reduction_method == 'pca':
            reducer = PCA(
                n_components=self.config.dimensions,
                random_state=self.config.random_state
            )
            
        logger.info(f"Applying {self.config.reduction_method} reduction...")
        reduced_flat = reducer.fit_transform(flat_activations)
        
        # Reshape back to (samples, layers, components)
        reduced_data = reduced_flat.reshape(
            n_samples, n_layers, self.config.dimensions
        )
        
        # Cache if enabled
        if self.config.cache_reduction:
            self._reduction_cache[cache_key] = reduced_data
            
        # Store reducer for potential reuse
        self._reducer = reducer
        
        return reduced_data
        
    def _create_plotly_figure(self,
                            reduced_data: np.ndarray,
                            data: Dict[str, Any],
                            layer_indices: List[int]) -> go.Figure:
        """Create Plotly figure for trajectory visualization.
        
        Args:
            reduced_data: Reduced activation data
            data: Original data dictionary
            layer_indices: Indices of layers being visualized
            
        Returns:
            Plotly Figure object
        """
        if self.config.dimensions == 3:
            return self._create_3d_plotly(reduced_data, data, layer_indices)
        else:
            return self._create_2d_plotly(reduced_data, data, layer_indices)
            
    def _create_3d_plotly(self,
                         reduced_data: np.ndarray,
                         data: Dict[str, Any],
                         layer_indices: List[int]) -> go.Figure:
        """Create 3D Plotly visualization."""
        n_samples, n_layers, _ = reduced_data.shape
        
        # Get color mapping
        colors, colorscale = self._get_colors(data, n_samples)
        
        # Create figure
        fig = go.Figure()
        
        # Apply layer separation if configured
        y_positions = self._get_layer_positions(n_layers)
        
        # Plot trajectories
        for i in range(n_samples):
            trajectory = reduced_data[i]
            
            # Add trajectory line
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=y_positions,
                z=trajectory[:, 2] if trajectory.shape[1] > 2 else np.zeros(n_layers),
                mode='lines+markers',
                line=dict(
                    color=colors[i],
                    width=self.config.line_width
                ),
                marker=dict(
                    size=self.config.point_size,
                    color=colors[i]
                ),
                opacity=self.config.opacity,
                name=f'Sample {i}',
                showlegend=False
            ))
            
            # Add arrows if configured
            if self.config.show_arrows and n_layers > 1:
                self._add_arrows_3d(fig, trajectory, y_positions, colors[i])
                
        # Add layer planes if stepped
        if self.config.layer_separation == 'stepped':
            self._add_layer_planes(fig, reduced_data, y_positions, layer_indices)
            
        # Add cluster centers if available
        if 'cluster_centers' in data and self.config.show_centroids:
            self._add_cluster_centers_3d(fig, data['cluster_centers'], y_positions)
            
        # Add cluster labels if configured
        if self.config.show_cluster_labels and 'labels' in data:
            self._add_cluster_labels_3d(fig, reduced_data, data['labels'], y_positions)
            
        # Update layout
        fig.update_layout(
            title=self._get_title(data),
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Layer',
                zaxis_title='Component 2',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=self.config.width,
            height=self.config.height,
            margin=self.config.margin
        )
        
        # Add legend if configured
        if self.config.show_legend:
            self._add_legend(fig, data)
            
        return fig
        
    def _create_2d_plotly(self,
                         reduced_data: np.ndarray,
                         data: Dict[str, Any],
                         layer_indices: List[int]) -> go.Figure:
        """Create 2D Plotly visualization."""
        # Implementation similar to 3D but without z-axis
        # This is a simplified version for brevity
        raise NotImplementedError("2D Plotly visualization not yet implemented")
        
    def _create_matplotlib_figure(self,
                                reduced_data: np.ndarray,
                                data: Dict[str, Any],
                                layer_indices: List[int]) -> plt.Figure:
        """Create Matplotlib figure for trajectory visualization."""
        if self.config.dimensions == 3:
            return self._create_3d_matplotlib(reduced_data, data, layer_indices)
        else:
            return self._create_2d_matplotlib(reduced_data, data, layer_indices)
            
    def _create_3d_matplotlib(self,
                            reduced_data: np.ndarray,
                            data: Dict[str, Any],
                            layer_indices: List[int]) -> plt.Figure:
        """Create 3D Matplotlib visualization."""
        n_samples, n_layers, _ = reduced_data.shape
        
        # Create figure
        fig = plt.figure(figsize=(self.config.width/100, self.config.height/100))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get color mapping
        colors, _ = self._get_colors(data, n_samples)
        
        # Apply layer separation
        y_positions = self._get_layer_positions(n_layers)
        
        # Plot trajectories
        for i in range(n_samples):
            trajectory = reduced_data[i]
            
            ax.plot(
                trajectory[:, 0],
                y_positions,
                trajectory[:, 2] if trajectory.shape[1] > 2 else np.zeros(n_layers),
                color=colors[i],
                linewidth=self.config.line_width,
                alpha=self.config.opacity
            )
            
            # Add markers
            ax.scatter(
                trajectory[:, 0],
                y_positions,
                trajectory[:, 2] if trajectory.shape[1] > 2 else np.zeros(n_layers),
                color=colors[i],
                s=self.config.point_size**2,
                alpha=self.config.opacity
            )
            
        # Set labels
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Layer')
        ax.set_zlabel('Component 2')
        ax.set_title(self._get_title(data))
        
        return fig
        
    def _create_2d_matplotlib(self,
                            reduced_data: np.ndarray,
                            data: Dict[str, Any],
                            layer_indices: List[int]) -> plt.Figure:
        """Create 2D Matplotlib visualization."""
        n_samples, n_layers, _ = reduced_data.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.config.width/100, self.config.height/100))
        
        # Get color mapping
        colors, _ = self._get_colors(data, n_samples)
        
        # Apply layer separation
        if self.config.layer_separation == 'sequential':
            x_positions = np.arange(n_layers)
        else:
            x_positions = self._get_layer_positions(n_layers)
            
        # Plot trajectories
        for i in range(n_samples):
            trajectory = reduced_data[i]
            
            if self.config.layer_separation == 'sequential':
                # Plot as time series
                ax.plot(
                    x_positions,
                    trajectory[:, 0],
                    color=colors[i],
                    linewidth=self.config.line_width,
                    alpha=self.config.opacity
                )
            else:
                # Plot in 2D space
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=colors[i],
                    linewidth=self.config.line_width,
                    alpha=self.config.opacity
                )
                
                # Add markers
                ax.scatter(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color=colors[i],
                    s=self.config.point_size**2,
                    alpha=self.config.opacity
                )
                
        # Set labels
        if self.config.layer_separation == 'sequential':
            ax.set_xlabel('Layer')
            ax.set_ylabel('Component 1')
        else:
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            
        ax.set_title(self._get_title(data))
        
        return fig
        
    def _get_colors(self, 
                   data: Dict[str, Any], 
                   n_samples: int) -> Tuple[List[str], Optional[Any]]:
        """Get color mapping for samples based on configuration.
        
        Args:
            data: Data dictionary with labels
            n_samples: Number of samples
            
        Returns:
            Tuple of (color list, colorscale)
        """
        # Use custom palette if provided
        if self.config.color_palette:
            palette = self.config.color_palette
        else:
            # Default palettes based on backend
            if self.config.backend == 'plotly':
                palette = [
                    'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
                    'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                    'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
                    'rgb(23, 190, 207)'
                ]
            else:  # matplotlib
                palette = list(plt.cm.tab10.colors) if plt else ['gray'] * 10
                
        # Get label array based on color_by
        if 'labels' in data and self.config.color_by in data['labels']:
            label_array = data['labels'][self.config.color_by]
            
            # Map labels to colors
            unique_labels = np.unique(label_array)
            if len(palette) > 0:
                label_to_color = {
                    label: palette[i % len(palette)] 
                    for i, label in enumerate(unique_labels)
                }
                colors = [label_to_color[label] for label in label_array]
            else:
                # Fallback if no palette
                colors = ['gray'] * n_samples
            
        else:
            # Default coloring by sample index
            if len(palette) > 0:
                colors = [palette[i % len(palette)] for i in range(n_samples)]
            else:
                colors = ['gray'] * n_samples
            
        return colors, None
        
    def _get_layer_positions(self, n_layers: int) -> np.ndarray:
        """Get y-positions for layers based on separation strategy.
        
        Args:
            n_layers: Number of layers
            
        Returns:
            Array of y-positions
        """
        if self.config.layer_separation == 'stepped':
            # Equal spacing with configured height
            return np.arange(n_layers) * self.config.layer_height
        elif self.config.layer_separation == 'sequential':
            # Linear spacing
            return np.linspace(0, 1, n_layers)
        else:  # none
            # All at same position
            return np.zeros(n_layers)
            
    def _add_arrows_3d(self, 
                      fig: go.Figure, 
                      trajectory: np.ndarray,
                      y_positions: np.ndarray,
                      color: str) -> None:
        """Add arrow annotations to 3D trajectory."""
        # Add cone markers for direction
        for i in range(len(trajectory) - 1):
            direction = trajectory[i+1] - trajectory[i]
            
            # Skip if no movement
            if np.allclose(direction, 0):
                continue
                
            # Normalize direction
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                
            # Add cone at midpoint
            midpoint = (trajectory[i] + trajectory[i+1]) / 2
            
            fig.add_trace(go.Cone(
                x=[midpoint[0]],
                y=[(y_positions[i] + y_positions[i+1]) / 2],
                z=[midpoint[2] if len(midpoint) > 2 else 0],
                u=[direction[0]],
                v=[0],  # No movement in layer dimension
                w=[direction[2] if len(direction) > 2 else 0],
                sizemode='absolute',
                sizeref=self.config.arrow_size,
                showscale=False,
                colorscale=[[0, color], [1, color]],
                showlegend=False
            ))
            
    def _add_layer_planes(self,
                         fig: go.Figure,
                         reduced_data: np.ndarray,
                         y_positions: np.ndarray,
                         layer_indices: List[int]) -> None:
        """Add semi-transparent planes at each layer."""
        x_range = [reduced_data[:, :, 0].min(), reduced_data[:, :, 0].max()]
        z_range = [reduced_data[:, :, 2].min(), reduced_data[:, :, 2].max()] if reduced_data.shape[2] > 2 else [0, 0]
        
        for i, (y_pos, layer_idx) in enumerate(zip(y_positions, layer_indices)):
            # Create grid for plane
            xx, zz = np.meshgrid(
                np.linspace(x_range[0], x_range[1], 2),
                np.linspace(z_range[0], z_range[1], 2)
            )
            yy = np.ones_like(xx) * y_pos
            
            fig.add_trace(go.Surface(
                x=xx,
                y=yy,
                z=zz,
                colorscale=[[0, 'rgba(200, 200, 200, 0.1)'], 
                           [1, 'rgba(200, 200, 200, 0.1)']],
                showscale=False,
                name=f'Layer {layer_idx}',
                showlegend=False
            ))
            
            # Add layer label if configured
            if self.config.show_layer_labels:
                fig.add_trace(go.Scatter3d(
                    x=[x_range[1] * 1.1],
                    y=[y_pos],
                    z=[z_range[1] * 1.1] if reduced_data.shape[2] > 2 else [0],
                    mode='text',
                    text=[f'Layer {layer_idx}'],
                    textposition='middle right',
                    showlegend=False
                ))
                
    def _get_title(self, data: Dict[str, Any]) -> str:
        """Generate appropriate title based on data and configuration."""
        parts = []
        
        # Add dataset name if available
        if 'metadata' in data and 'dataset' in data['metadata']:
            parts.append(data['metadata']['dataset'])
            
        # Add visualization type
        parts.append(f"{self.config.dimensions}D Trajectory Visualization")
        
        # Add coloring info
        parts.append(f"(colored by {self.config.color_by})")
        
        return " - ".join(parts)
        
    def save_figure(self,
                   fig: Union[go.Figure, plt.Figure],
                   output_path: Union[str, Path],
                   format: str = 'html',
                   **kwargs) -> None:
        """Save figure to file.
        
        Args:
            fig: Figure to save
            output_path: Output file path
            format: Output format
            **kwargs: Additional save parameters
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.backend == 'plotly':
            if format == 'html':
                fig.write_html(str(output_path), **kwargs)
            elif format in ['png', 'pdf', 'svg', 'jpeg']:
                fig.write_image(str(output_path), format=format, **kwargs)
            else:
                raise ValueError(f"Unsupported format for plotly: {format}")
        else:  # matplotlib
            if format == 'html':
                # Save as static image embedded in HTML
                import base64
                from io import BytesIO
                
                buffer = BytesIO()
                fig.savefig(buffer, format='png', **kwargs)
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode()
                
                html_content = f'''
                <html>
                <body>
                <img src="data:image/png;base64,{img_str}" />
                </body>
                </html>
                '''
                output_path.write_text(html_content)
            else:
                fig.savefig(str(output_path), format=format, **kwargs)
                
        logger.info(f"Saved trajectory visualization to {output_path}")
        
    def _add_cluster_centers_3d(self,
                               fig: go.Figure,
                               cluster_centers: np.ndarray,
                               y_positions: np.ndarray) -> None:
        """Add cluster center markers to 3D plot."""
        n_layers = len(y_positions)
        
        for layer_idx in range(n_layers):
            if layer_idx >= len(cluster_centers):
                continue
                
            centers = cluster_centers[layer_idx]
            n_clusters = len(centers)
            
            # Reduce cluster centers if needed
            if centers.shape[1] > 3:
                # Apply same reduction as data
                if self._reducer is not None:
                    centers_reduced = self._reducer.transform(centers)[:, :3]
                else:
                    # Fallback to first 3 dimensions
                    centers_reduced = centers[:, :3]
            else:
                centers_reduced = centers
                
            # Add cluster center markers
            fig.add_trace(go.Scatter3d(
                x=centers_reduced[:, 0],
                y=[y_positions[layer_idx]] * n_clusters,
                z=centers_reduced[:, 2] if centers_reduced.shape[1] > 2 else np.zeros(n_clusters),
                mode='markers',
                marker=dict(
                    size=self.config.point_size * 2,
                    color='black',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                name=f'Layer {layer_idx} Centers',
                showlegend=False
            ))
        
    def _add_cluster_labels_3d(self,
                              fig: go.Figure,
                              reduced_data: np.ndarray,
                              labels: Dict[str, Any],
                              y_positions: np.ndarray) -> None:
        """Add cluster label annotations to 3D plot."""
        if 'cluster_labels' not in labels:
            return
            
        cluster_labels = labels['cluster_labels']
        n_samples, n_layers = cluster_labels.shape
        
        # Get unique clusters per layer
        for layer_idx in range(n_layers):
            layer_clusters = np.unique(cluster_labels[:, layer_idx])
            
            for cluster_id in layer_clusters:
                # Find samples in this cluster
                mask = cluster_labels[:, layer_idx] == cluster_id
                cluster_points = reduced_data[mask, layer_idx, :]
                
                if len(cluster_points) > 0:
                    # Calculate centroid
                    centroid = cluster_points.mean(axis=0)
                    
                    # Add label at centroid
                    fig.add_trace(go.Scatter3d(
                        x=[centroid[0]],
                        y=[y_positions[layer_idx]],
                        z=[centroid[2] if len(centroid) > 2 else 0],
                        mode='text',
                        text=[f'C{cluster_id}'],
                        textposition='middle center',
                        textfont=dict(size=10, color='black'),
                        showlegend=False
                    ))
        
    def _add_legend(self,
                   fig: Union[go.Figure, plt.Figure],
                   data: Dict[str, Any]) -> None:
        """Add legend to figure based on color mapping."""
        if self.config.backend == 'plotly':
            # Plotly handles legends automatically with showlegend=True
            # Add custom legend entries if needed
            if 'paths' in data and self.config.color_by == 'path':
                # Add path legend similar to Sankey
                pass
        else:
            # Matplotlib legend
            if hasattr(fig, 'legend'):
                fig.legend()
                
    def add_archetypal_paths(self,
                            fig: Union[go.Figure, plt.Figure],
                            paths: List[Dict[str, Any]],
                            reduced_data: np.ndarray,
                            y_positions: np.ndarray,
                            emphasis_top_n: int = 7) -> None:
        """Add archetypal path overlays with emphasis.
        
        Args:
            fig: Figure to add paths to
            paths: List of path dictionaries with 'indices' and 'count' keys
            reduced_data: Reduced activation data
            y_positions: Y positions for layers
            emphasis_top_n: Number of top paths to emphasize
        """
        if self.config.backend == 'plotly':
            # Sort paths by frequency
            sorted_paths = sorted(paths, key=lambda p: p.get('count', 0), reverse=True)
            
            # Add top paths with emphasis
            for i, path_info in enumerate(sorted_paths[:emphasis_top_n]):
                indices = path_info['indices']
                count = path_info.get('count', 1)
                
                # Calculate path trajectory as average of samples
                path_trajectory = reduced_data[indices].mean(axis=0)
                
                # Line width based on frequency
                width = min(10, 2 + count / 10)
                
                fig.add_trace(go.Scatter3d(
                    x=path_trajectory[:, 0],
                    y=y_positions,
                    z=path_trajectory[:, 2] if path_trajectory.shape[1] > 2 else np.zeros(len(y_positions)),
                    mode='lines',
                    line=dict(
                        color=f'rgba(255, {50 + i * 30}, 0, 0.8)',
                        width=width
                    ),
                    name=f'Path {i+1} ({count} samples)',
                    showlegend=True
                ))
                
    def add_average_trajectories(self,
                               fig: Union[go.Figure, plt.Figure],
                               reduced_data: np.ndarray,
                               labels: Dict[str, Any],
                               y_positions: np.ndarray,
                               category_key: str = 'pos_labels') -> None:
        """Add average trajectories per category (e.g., POS tags).
        
        Args:
            fig: Figure to add trajectories to
            reduced_data: Reduced activation data
            labels: Label dictionary
            y_positions: Y positions for layers
            category_key: Key in labels dict for categories
        """
        if category_key not in labels:
            return
            
        categories = labels[category_key]
        unique_categories = np.unique(categories)
        
        # Color map for categories
        colors = {
            'NN': 'blue',      # Nouns
            'VB': 'red',       # Verbs
            'JJ': 'green',     # Adjectives
            'RB': 'orange',    # Adverbs
        }
        
        if self.config.backend == 'plotly':
            for category in unique_categories:
                # Get samples in this category
                mask = categories == category
                category_data = reduced_data[mask]
                
                if len(category_data) > 0:
                    # Calculate average trajectory
                    avg_trajectory = category_data.mean(axis=0)
                    
                    # Add average trajectory
                    fig.add_trace(go.Scatter3d(
                        x=avg_trajectory[:, 0],
                        y=y_positions,
                        z=avg_trajectory[:, 2] if avg_trajectory.shape[1] > 2 else np.zeros(len(y_positions)),
                        mode='lines+markers',
                        line=dict(
                            color=colors.get(category, 'gray'),
                            width=4
                        ),
                        marker=dict(
                            size=8,
                            color=colors.get(category, 'gray')
                        ),
                        name=f'Avg {category}',
                        showlegend=True
                    ))