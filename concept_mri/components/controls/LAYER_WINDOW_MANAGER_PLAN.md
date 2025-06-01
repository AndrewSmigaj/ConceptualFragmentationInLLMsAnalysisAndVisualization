# Layer Window Manager Implementation Plan

## Overview
The Layer Window Manager is a critical component of Concept MRI that helps researchers split deep neural networks into analyzable windows for Archetypal Path Analysis (APA). It provides both manual configuration and experimental auto-detection of semantic boundaries using existing metrics.

## Design Principles
- **Leverage existing metrics** - No reimplementation of analysis code
- **Progressive disclosure** - Simple manual mode by default, experimental features clearly marked
- **Visual guidance** - Metrics plots help users identify interesting boundaries
- **Scalability** - Must handle networks from 10 to 100+ layers effectively

## Architecture

### File Structure
```
concept_mri/components/controls/
├── layer_window_manager.py       # Main UI component
├── window_detection_utils.py     # Metric integration utilities
└── window_callbacks.py          # Dash callbacks
```

### Component Dependencies
```python
# From concept_fragmentation (existing metrics)
- metrics.representation_stability
- metrics.cross_layer_metrics  
- metrics.token_path_metrics

# From concept_mri
- config.settings (theme colors)
- core.model_interface (model data)
```

## User Interface Design

### Layout
```
┌─────────────────────────────────────────────────────┐
│ 🔲 Layer Window Configuration  [32 layers] ℹ️        │
├─────────────────────────────────────────────────────┤
│                                                      │
│ Mode: ● Manual  ○ Auto-detect (Experimental) 🧪     │
│                                                      │
│ ┌─ Layer Metrics ─────────────────────────────────┐ │
│ │ [Interactive Plotly chart]                      │ │
│ │ - Representation Stability (blue)               │ │
│ │ - Path Density (green)                          │ │
│ │ - Suggested boundaries (red dashed)             │ │
│ │ 📍 Click to add boundary                        │ │
│ └──────────────────────────────────────────────────┘ │
│                                                      │
│ Current Windows:                                     │
│ • Early (L0-L3)    [🟦] [Edit] [×]                 │
│ • Middle (L4-L7)   [🟩] [Edit] [×]                 │
│ • Late (L8-L11)    [🟧] [Edit] [×]                 │
│                                                      │
│ [Presets ▼] [Add Window] [Clear All]               │
└──────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Day 1)
**Goal**: Create core infrastructure and metric integration

1. **window_detection_utils.py**
   ```python
   class WindowDetectionMetrics:
       def compute_boundary_metrics(activations_dict, clusters_dict=None)
       def normalize_metrics_for_display(metrics, n_layers)
       def aggregate_metrics_for_deep_networks(metrics, target_points=50)
   ```

2. **Basic component structure**
   - LayerWindowManager class
   - Component layout (Card, Header, Body)
   - State management stores

### Phase 2: Manual Mode (Day 2-3)
**Goal**: Implement fully functional manual window configuration

1. **Preset configurations**
   - GPT-2 style (proven configuration)
   - Quarters, Thirds, Halves
   - Dynamic adjustment for layer counts

2. **Custom window builder**
   - Interactive form for window creation
   - Validation (overlaps, bounds)
   - Color assignment from theme

3. **Window visualization**
   - List view with visual indicators
   - Edit/Delete functionality
   - Preview on metrics plot

### Phase 3: Metrics Visualization (Day 4)
**Goal**: Create interactive metrics plot for boundary identification

1. **Plotly implementation**
   - Multi-line plot (stability, density)
   - Click-to-add-boundary handler
   - Hover information
   - Responsive sizing

2. **Deep network handling**
   - Smart aggregation for >50 layers
   - Zoom/pan controls
   - Focus region selector

### Phase 4: Experimental Auto-detect (Day 5)
**Goal**: Add experimental boundary detection features

1. **Boundary detection algorithm**
   ```python
   def detect_boundaries(metrics, num_windows=3):
       # Find peaks in instability
       # Combine metrics for confidence
       # Return suggested boundaries with scores
   ```

2. **Experimental UI**
   - Clear "Experimental 🧪" labeling
   - Suggestion preview
   - Accept/Reject controls
   - Confidence indicators

### Phase 5: Integration & Polish (Day 6)
**Goal**: Connect to existing pipeline and handle edge cases

1. **Pipeline integration**
   - Connect to model store
   - Update clustering configuration
   - Sync with visualization components

2. **Edge cases**
   - Single layer networks
   - Very deep networks (>100 layers)
   - Missing metrics handling
   - Validation and error messages

## Technical Specifications

### Window Configuration Format
```python
{
    'mode': 'manual' | 'auto',
    'windows': {
        'window_name': {
            'start': int,
            'end': int,
            'color': str,  # hex color
            'confidence': float  # for auto-detected
        }
    },
    'metrics': {
        'stability': [...],
        'density': [...],
        'boundaries': [...]
    }
}
```

### Metric Integration
```python
# Use existing metrics without reimplementation
from concept_fragmentation.metrics import representation_stability

# Compute only what's needed for boundaries
stability_scores = representation_stability.compute_representation_stability(
    activations_dict=model_data['activations'],
    normalize=True
)
```

### Callback Structure
```python
# Main callbacks needed
- update_metrics_on_model_load
- handle_mode_switch
- handle_preset_selection
- handle_boundary_click
- handle_window_add/edit/delete
- handle_auto_detect
```

## Success Criteria

1. **Manual mode works seamlessly**
   - Users can create windows via presets or custom
   - Visual feedback is clear and immediate
   - Integration with pipeline is transparent

2. **Metrics guide decisions**
   - Plots clearly show interesting boundaries
   - Interactive features are intuitive
   - Deep networks are handled gracefully

3. **Experimental features are useful**
   - Auto-detection provides reasonable suggestions
   - Confidence scores help users decide
   - Clear that features are experimental

4. **No metric reimplementation**
   - All analysis uses existing code
   - Only UI and integration logic is new
   - Performance is maintained

## Future Enhancements

1. **Advanced auto-detection**
   - Architecture-aware boundaries
   - Semantic transition detection
   - Multi-metric consensus

2. **Window quality metrics**
   - Cohesion scores
   - Separation metrics
   - Coverage analysis

3. **Export/Import**
   - Save window configurations
   - Share between experiments
   - Batch processing

## Risk Mitigation

1. **Metric availability**
   - Gracefully handle missing metrics
   - Provide meaningful defaults
   - Clear user communication

2. **Performance**
   - Lazy computation of metrics
   - Efficient aggregation for deep networks
   - Responsive UI during computation

3. **User confusion**
   - Clear documentation
   - Intuitive defaults
   - Progressive disclosure of complexity