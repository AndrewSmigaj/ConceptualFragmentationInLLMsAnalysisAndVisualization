# Phase 2: Migrate and Consolidate Sankey Implementations

## Objective
Create a single, configurable `SankeyGenerator` class that consolidates features from all existing implementations and fixes known issues.

## Current Implementations Analysis

### 1. `generate_sankey_diagrams.py` (Base Implementation)
- **Strengths**: 
  - Parameterized for any k value
  - Clean structure with SankeyGenerator class
  - Includes path summary table generation
- **Features to keep**: 
  - Parameterized design
  - Path summary functionality
  - Basic Sankey structure

### 2. `generate_k10_sankeys.py` (K=10 Specific)
- **Strengths**: 
  - Windowed analysis integration
  - Path extraction logic
- **Features to keep**: 
  - Window-based path extraction
  - Integration with analysis pipeline

### 3. `generate_colored_sankeys_k10.py` (Colored Paths)
- **Strengths**: 
  - Colored archetypal paths
  - Path descriptions
  - Legend implementation
- **Features to keep**: 
  - Color coding system
  - Path description generation
  - Legend positioning

### 4. `generate_enhanced_sankeys_k10.py` (Custom Positioning)
- **Issues**: 
  - Creates "orange circle" with custom positioning
  - Overly complex node placement
- **Features to avoid**: 
  - Custom x/y positioning
  - Manual node placement

### 5. `generate_fixed_sankeys_k10.py` (Latest Fix)
- **Strengths**: 
  - Fixes label count issue
  - Only shows labels for visible clusters
  - Proper left/right label positioning
- **Features to keep**: 
  - Dynamic cluster detection
  - Separate left/right annotations

## Design for Unified SankeyGenerator

```python
class SankeyGenerator(BaseVisualizer):
    """Unified Sankey diagram generator for concept trajectory analysis.
    
    Consolidates all Sankey functionality into a single configurable class.
    """
    
    def __init__(self, config: SankeyConfig):
        """Initialize with configuration."""
        super().__init__(config)
        self.windowed_data = None
        self.labels = None
        self.purity_data = None
        
    def create_figure(self, 
                     data: Dict[str, Any],
                     window: str = 'early',
                     **kwargs) -> go.Figure:
        """Create Sankey diagram for specified window.
        
        Args:
            data: Dictionary containing:
                - windowed_analysis: Analysis results by window
                - labels: Semantic labels
                - purity_data: Optional purity scores
            window: Which window to visualize ('early', 'middle', 'late')
            
        Returns:
            Plotly Figure object
        """
        # Implementation combining best features
        
    def _find_visible_clusters(self, 
                              paths: List[Dict],
                              layers: List[int]) -> Dict[int, Set[int]]:
        """Find which clusters actually appear in paths."""
        # From generate_fixed_sankeys_k10.py
        
    def _generate_path_description(self, 
                                  semantic_labels: List[str]) -> str:
        """Generate concise path description."""
        # From generate_colored_sankeys_k10.py
        
    def _create_path_legend(self,
                           paths: List[Dict],
                           position: str = 'left') -> List[Dict]:
        """Create legend annotations for paths."""
        # Configurable positioning
        
    def _create_node_labels(self,
                           nodes: List[Dict],
                           last_layer_position: str = 'right') -> List[Dict]:
        """Create node label annotations."""
        # Separate handling for last layer
```

## Implementation Steps

### Step 1: Create SankeyGenerator Class
1. Inherit from BaseVisualizer
2. Implement core Sankey creation logic
3. Add configuration validation

### Step 2: Port Core Features
1. **From base**: Parameterized structure, path summaries
2. **From colored**: Path coloring, descriptions
3. **From fixed**: Dynamic cluster detection, proper labeling

### Step 3: Add New Features
1. Configurable color palettes
2. Multiple layout options
3. Export to different formats
4. Interactive features toggle

### Step 4: Fix Known Issues
1. Label overlap prevention
2. Dynamic cluster visibility
3. Proper margin calculation
4. Consistent color mapping

### Step 5: Create Tests
1. Test with different k values
2. Test all configuration options
3. Test edge cases (few paths, many clusters)
4. Visual regression tests

## Configuration Options

```python
@dataclass
class SankeyConfig:
    # Display options
    top_n_paths: int = 25
    show_purity: bool = True
    colored_paths: bool = True
    
    # Layout options
    legend_position: str = 'left'  # 'left', 'right', 'top', 'bottom', 'none'
    last_layer_labels_position: str = 'right'  # 'left', 'right', 'inline', 'none'
    
    # Style options
    color_palette: Optional[List[str]] = None
    node_color: str = 'rgba(100, 100, 100, 0.8)'
    link_opacity: float = 0.6
    
    # Size options
    width: int = 1600
    height: int = 800
    node_pad: int = 15
    node_thickness: int = 20
    
    # Advanced options
    show_all_clusters: bool = False  # Show even if not in top paths
    group_similar_paths: bool = False  # Group paths with same start/end
    animate_transitions: bool = False  # Add animation support
```

## Migration Strategy

1. **Create new implementation** in `concept_fragmentation/visualization/sankey.py`
2. **Test alongside old implementations** to ensure feature parity
3. **Create migration script** to update existing code
4. **Archive old implementations** after verification

## Quality Checks

1. **Feature Completeness**
   - All features from existing implementations preserved
   - New features properly integrated
   - Configuration options working

2. **Performance**
   - Fast rendering for large datasets
   - Efficient memory usage
   - Caching where appropriate

3. **Maintainability**
   - Clear code structure
   - Comprehensive documentation
   - Easy to extend

## Risk Mitigation

1. **Risk**: Breaking existing visualizations
   - **Mitigation**: Side-by-side testing, visual diffs

2. **Risk**: Missing edge cases
   - **Mitigation**: Comprehensive test suite

3. **Risk**: Performance regression
   - **Mitigation**: Benchmark against current implementations

## Success Criteria

- [ ] Single SankeyGenerator class handles all use cases
- [ ] All existing features preserved
- [ ] Known issues fixed (label overlap, cluster count)
- [ ] Comprehensive tests passing
- [ ] Performance equal or better
- [ ] Clear migration path documented