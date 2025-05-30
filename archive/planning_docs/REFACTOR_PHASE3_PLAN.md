# Phase 3: Trajectory Visualization Consolidation Plan

## Objective
Create a unified `TrajectoryVisualizer` class that consolidates all trajectory visualization functionality while preserving unique features from each implementation.

## Current State Analysis

### Implementations to Consolidate
1. **visualization/traj_plot.py** - Main dashboard 3D trajectories
2. **concept_fragmentation/visualization/trajectories.py** - Core matplotlib implementation  
3. **experiments/gpt2/semantic_subtypes/generate_gpt2_trajectory_viz_*.py** - Multiple GPT-2 visualizers
4. **experiments/heart_disease/generate_heart_trajectory_*.py** - Patient pathway visualizers
5. **visualization/generate_stepped_layer_viz.py** - Stepped layer visualization

### Key Features to Preserve
- 3D interactive visualizations (Plotly)
- 2D static visualizations (Matplotlib)
- Multiple dimensionality reduction methods (UMAP, PCA, t-SNE)
- Flexible coloring schemes (class, cluster, path, POS, metric)
- Statistical overlays (averages, error bars, centroids)
- Windowed analysis (early/middle/late layers)
- Archetypal path highlighting

## Design for Unified TrajectoryVisualizer

```python
class TrajectoryVisualizer(BaseVisualizer):
    """Unified trajectory visualizer for concept movement analysis.
    
    Supports both 2D and 3D visualizations with multiple backends.
    """
    
    def __init__(self, config: TrajectoryConfig):
        """Initialize with configuration."""
        super().__init__(config)
        self.reducer = None
        self.reduced_data = None
        
    def create_figure(self, 
                     data: Dict[str, Any],
                     window: Optional[str] = None,
                     **kwargs) -> Union[go.Figure, plt.Figure]:
        """Create trajectory visualization.
        
        Args:
            data: Dictionary containing:
                - activations: Activation data
                - labels: Optional class/cluster labels
                - paths: Optional path assignments
                - metadata: Optional sample metadata
            window: Optional window ('early', 'middle', 'late')
            
        Returns:
            Figure object (Plotly or Matplotlib)
        """
        
    def _reduce_dimensions(self, 
                          activations: np.ndarray,
                          method: str) -> np.ndarray:
        """Apply dimensionality reduction."""
        
    def _create_3d_plotly(self, 
                         reduced_data: np.ndarray,
                         **kwargs) -> go.Figure:
        """Create 3D Plotly visualization."""
        
    def _create_2d_matplotlib(self,
                             reduced_data: np.ndarray,
                             **kwargs) -> plt.Figure:
        """Create 2D Matplotlib visualization."""
        
    def add_archetypal_paths(self,
                            fig: Any,
                            paths: List[Dict],
                            **kwargs) -> None:
        """Add archetypal path overlays."""
        
    def add_cluster_centers(self,
                           fig: Any,
                           centers: np.ndarray,
                           **kwargs) -> None:
        """Add cluster center markers."""
```

## Enhanced TrajectoryConfig

```python
@dataclass
class TrajectoryConfig:
    """Enhanced configuration for trajectory visualization."""
    
    # Dimensionality reduction
    reduction_method: str = 'umap'  # 'umap', 'tsne', 'pca', 'none'
    n_components: int = 3  # 2 or 3
    
    # Backend and display
    backend: str = 'plotly'  # 'plotly' or 'matplotlib'
    dimensions: int = 3  # 2 or 3
    
    # Visual appearance
    color_by: str = 'cluster'  # 'cluster', 'class', 'path', 'pos', 'metric'
    color_palette: Optional[List[str]] = None
    marker_size: float = 3.0
    line_width: float = 2.0
    opacity: float = 0.8
    
    # Layer handling
    layer_separation: str = 'stepped'  # 'stepped', 'sequential', 'none'
    layer_height: float = 0.1  # For stepped visualization
    
    # Annotations
    show_arrows: bool = True
    show_cluster_labels: bool = True
    show_layer_labels: bool = True
    show_legend: bool = True
    
    # Statistical overlays
    show_averages: bool = False
    show_error_bars: bool = False
    show_centroids: bool = False
    
    # Performance
    max_samples: Optional[int] = None  # Subsample for performance
    cache_reduction: bool = True
    
    # UMAP specific
    n_neighbors: int = 15
    min_dist: float = 0.1
    
    # t-SNE specific
    perplexity: int = 30
    
    # Output
    width: int = 1000
    height: int = 800
    margin: Dict[str, int] = field(default_factory=lambda: {
        'l': 50, 'r': 50, 't': 50, 'b': 50
    })
```

## Implementation Steps

### Step 1: Create TrajectoryVisualizer Class
1. Implement base structure inheriting from BaseVisualizer
2. Add dimensionality reduction methods
3. Implement both Plotly and Matplotlib backends
4. Add configuration validation

### Step 2: Port Core Features
1. **From traj_plot.py**: Layer separation, cluster centers, arrows
2. **From GPT-2**: Windowed analysis, POS coloring, average trajectories
3. **From Heart**: Archetypal paths, flow cones, path thickness
4. **From core**: Statistical overlays, multiple reduction methods

### Step 3: Add Enhanced Features
1. Automatic color palette selection based on data type
2. Intelligent label placement to avoid overlaps
3. Export to multiple formats (HTML, PNG, PDF, MP4)
4. Animation support for temporal data

### Step 4: Create Comprehensive Tests
1. Test different data shapes and types
2. Test all configuration options
3. Test both backends
4. Test edge cases (single layer, single sample)
5. Visual regression tests

### Step 5: Migration Support
1. Create wrapper functions for backward compatibility
2. Write migration script to update existing code
3. Create examples for common use cases
4. Document breaking changes

## Data Structure Requirements

### Input Data Format
```python
{
    'activations': np.ndarray,  # (n_samples, n_layers, n_features)
    'labels': Optional[Dict[str, np.ndarray]],  # Various label types
    'paths': Optional[List[Dict]],  # Archetypal path information
    'metadata': Optional[Dict[str, Any]],  # Sample metadata
    'cluster_centers': Optional[np.ndarray],  # Cluster centroids
    'windows': Optional[Dict[str, List[int]]]  # Layer windows
}
```

### Label Types
- `class_labels`: Integer array of class assignments
- `cluster_labels`: Integer array of cluster assignments per layer
- `path_labels`: Path membership for each sample
- `pos_labels`: Part-of-speech tags (for text data)
- `metric_values`: Continuous values for metric-based coloring

## Migration Strategy

1. **Create new implementation** without breaking existing code
2. **Add compatibility layer** for old function signatures
3. **Update documentation** with migration examples
4. **Gradual deprecation** of old implementations

## Quality Metrics

1. **Feature Coverage**: All existing features preserved
2. **Performance**: Equal or better than current implementations
3. **Code Reduction**: Target 70% reduction in trajectory code
4. **Test Coverage**: >90% coverage for new module
5. **Documentation**: Complete API docs and examples

## Risk Mitigation

1. **Risk**: Breaking existing visualizations
   - **Mitigation**: Extensive backward compatibility testing
   
2. **Risk**: Performance regression for large datasets
   - **Mitigation**: Profiling and optimization, caching support
   
3. **Risk**: Loss of unique visualization features
   - **Mitigation**: Comprehensive feature inventory and tests

## Success Criteria

- [ ] Single TrajectoryVisualizer handles all use cases
- [ ] Both Plotly and Matplotlib backends working
- [ ] All existing features preserved
- [ ] Performance equal or better
- [ ] Comprehensive test suite passing
- [ ] Migration guide and examples complete
- [ ] 70% reduction in trajectory visualization code