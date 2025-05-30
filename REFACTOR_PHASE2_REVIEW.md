# Phase 2 Plan Review

## Design Review

### ✅ Strengths

1. **Comprehensive Analysis**
   - Thoroughly analyzed all 5 implementations
   - Identified features to keep and avoid
   - Clear understanding of each implementation's purpose

2. **Smart Consolidation**
   - Keeps the best features from each implementation
   - Avoids problematic custom positioning
   - Maintains backward compatibility through configuration

3. **Flexible Configuration**
   - Extensive configuration options
   - Sensible defaults
   - Easy to extend for future needs

4. **Clear Migration Path**
   - Side-by-side testing strategy
   - Gradual migration approach
   - Archive plan for old code

### ⚠️ Improvements Needed

1. **Missing Helper Methods**
   - Need method for color assignment to paths
   - Need method for node positioning optimization
   - Need method for text overlap detection

2. **Data Structure Not Defined**
   - Need to specify exact structure of input data
   - Should define TypedDict for clarity
   - Need to handle missing data gracefully

3. **Export Functionality**
   - Plan mentions "export to different formats" but no implementation
   - Should include HTML, PNG, PDF, and data export

4. **Path Summary Integration**
   - Original has `create_path_summary_table` method
   - This should be part of the unified implementation

### 📋 Enhanced Design

#### Data Structures
```python
from typing import TypedDict, List, Dict, Any

class PathInfo(TypedDict):
    path: List[int]
    frequency: int
    representative_words: List[str]
    semantic_labels: List[str]
    percentage: float

class WindowedAnalysis(TypedDict):
    layers: List[int]
    total_paths: int
    unique_paths: int
    archetypal_paths: List[PathInfo]

class SankeyData(TypedDict):
    windowed_analysis: Dict[str, WindowedAnalysis]
    labels: Dict[str, Dict[str, Dict[str, Any]]]
    purity_data: Optional[Dict[str, Dict[str, Dict[str, float]]]]
```

#### Additional Methods
```python
class SankeyGenerator(BaseVisualizer):
    def _assign_path_colors(self, 
                           paths: List[PathInfo],
                           palette: List[str]) -> Dict[int, str]:
        """Assign colors to paths consistently."""
        
    def _detect_label_overlaps(self,
                              annotations: List[Dict]) -> List[Dict]:
        """Detect and resolve label overlaps."""
        
    def _optimize_node_positions(self,
                                clusters_by_layer: Dict[int, Set[int]],
                                layers: List[int]) -> Dict[Tuple[int, int], int]:
        """Optimize node positions to minimize crossings."""
        
    def create_path_summary(self,
                           data: SankeyData,
                           output_format: str = 'markdown') -> str:
        """Create summary table of paths."""
        
    def export_figure(self,
                     fig: go.Figure,
                     output_path: str,
                     formats: List[str] = ['html']) -> Dict[str, str]:
        """Export figure to multiple formats."""
```

#### Error Handling
```python
def create_figure(self, data: Dict[str, Any], window: str = 'early', **kwargs):
    try:
        # Validate input data
        self._validate_sankey_data(data)
        
        # Check window validity
        if window not in data['windowed_analysis']:
            raise InvalidDataError(f"Window '{window}' not found in data")
            
        # Create figure with error recovery
        fig = self._create_sankey_internal(data, window, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create Sankey: {e}")
        # Return error visualization or raise
        if self.config.fail_gracefully:
            return self._create_error_figure(str(e))
        raise
```

### 📊 Testing Strategy Enhancement

1. **Unit Tests**
   ```python
   def test_sankey_generator():
       # Test initialization
       # Test each configuration option
       # Test color assignment
       # Test path descriptions
       # Test error cases
   ```

2. **Integration Tests**
   ```python
   def test_sankey_with_real_data():
       # Load actual k=10 data
       # Generate all windows
       # Compare with reference outputs
   ```

3. **Visual Tests**
   ```python
   def test_visual_regression():
       # Generate figures
       # Compare with baseline images
       # Flag any visual differences
   ```

### 🔧 Implementation Order (Revised)

1. **Create data structures and type definitions**
2. **Implement core SankeyGenerator class**
3. **Add helper methods for layout optimization**
4. **Port visualization logic from best implementations**
5. **Add export functionality**
6. **Create comprehensive test suite**
7. **Benchmark performance**
8. **Create migration examples**

### 📝 Additional Considerations

1. **Accessibility**
   - Add ARIA labels for screen readers
   - Ensure color blind friendly palettes
   - Provide text alternatives

2. **Performance Optimization**
   - Cache computed layouts
   - Lazy load large datasets
   - Optimize for common cases

3. **Documentation**
   - Docstring examples for each method
   - Gallery of example outputs
   - Troubleshooting guide

This enhanced design provides a more robust foundation for the Sankey consolidation.