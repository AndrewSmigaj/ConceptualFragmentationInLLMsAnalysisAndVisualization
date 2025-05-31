# Concept MRI Implementation Plan

## Project Overview
Concept MRI is a clinical-themed web application for analyzing neural network concept organization using Concept Trajectory Analysis (CTA). It provides an intuitive interface for researchers to upload their models, configure analysis parameters, and explore how concepts flow through network layers.

## Design Philosophy
- **Fresh UI, Proven Backend**: Create a new, user-friendly dash application while leveraging all existing analysis infrastructure
- **Clinical Aesthetic**: Medical-inspired design with clean whites, teals, and high contrast
- **Progressive Disclosure**: Start simple, reveal complexity on demand
- **User Control**: Let researchers configure clustering, adjust parameters, and explore interactively

## Architecture Overview

### What We're Building New
1. **Clean Dash Application**
   - Purpose-built for end users (not internal research tool)
   - Better UI/UX control and design consistency
   - Clinical theme throughout

2. **User-Facing Components**
   - Model upload interface (drag & drop)
   - Dataset upload with feature mapping
   - API key configuration UI
   - Clustering configuration panel
   - Cluster information cards

3. **Visualization Wrappers**
   - Sankey diagram component (wraps existing SankeyGenerator)
   - Stepped UMAP component (wraps existing reducers)
   - 3D trajectory viewer (wraps existing TrajectoryVisualizer)

4. **Generic FF Network Support**
   - New LLM analysis adapter for any feedforward network
   - Feature-based descriptions (not demographic-specific)

### What We're Reusing (DRY Principle)
- `concept_fragmentation.activation.ActivationCollector` - Model analysis
- `concept_fragmentation.visualization.sankey.SankeyGenerator` - Sankey diagrams
- `concept_fragmentation.visualization.trajectory.TrajectoryVisualizer` - 3D paths
- `concept_fragmentation.analysis.transformer_dimensionality` - UMAP/dimensionality reduction
- All clustering algorithms (KMeans, DBSCAN)
- All metrics (gap statistic, silhouette, fragmentation, etc.)
- LLM infrastructure (factory, caching, batch processing)

## Directory Structure
```
concept_mri/
├── app.py                      # Main Dash application
├── config/
│   ├── settings.py            # UI-specific settings
│   └── llm_config.py         # API key management UI
├── core/
│   ├── model_interface.py    # Wrapper for ActivationCollector
│   ├── clustering_config.py  # Clustering configuration logic
│   └── llm_adapter.py        # Generic FF network LLM analysis
├── components/
│   ├── layouts/
│   │   └── main_layout.py   # Overall app layout
│   ├── visualizations/
│   │   ├── sankey_wrapper.py    # Wraps SankeyGenerator
│   │   ├── umap_stepped.py      # Wraps dimensionality reducers
│   │   └── trajectory_3d.py     # Wraps TrajectoryVisualizer
│   ├── controls/
│   │   ├── model_upload.py      # Model upload UI
│   │   ├── dataset_upload.py    # Dataset upload UI
│   │   ├── clustering_panel.py  # Clustering settings
│   │   └── api_keys_panel.py    # LLM API configuration
│   └── cards/
│       ├── cluster_card.py      # Cluster information display
│       └── diagnostic_card.py   # Diagnostic scanner results
├── tabs/
│   ├── ff_networks.py          # Main analysis tab
│   └── gpt_placeholder.py      # Future GPT support
├── assets/
│   └── styles/
│       ├── clinical.css        # Medical theme styling
│       └── components.css      # Component-specific styles
└── requirements.txt            # Minimal Dash dependencies
```

## Implementation Phases

### Phase 1: Core Infrastructure (Priority: High)
1. Create `app.py` with basic Dash structure
2. Implement `config/settings.py` for UI configuration
3. Create `components/layouts/main_layout.py` with tab structure
4. Set up `requirements.txt` with minimal dependencies

### Phase 2: Data Input (Priority: High)
1. Implement `components/controls/model_upload.py`
   - Support PyTorch (.pt, .pth), ONNX, TensorFlow
   - Show model architecture summary
   - Progress indicator for activation extraction
   
2. Implement `components/controls/dataset_upload.py`
   - CSV, NPZ, PKL support
   - Feature name extraction/mapping
   - Data preview functionality

3. Implement `components/controls/clustering_panel.py`
   - Algorithm selection (KMeans, DBSCAN)
   - Metric selection (Gap, Silhouette)
   - Manual k override option
   - Advanced settings (distance metrics, seeds)

### Phase 3: Visualization Components (Priority: High)
1. Create `components/visualizations/sankey_wrapper.py`
   - Integrate with existing SankeyGenerator
   - Add interactivity callbacks
   - Synchronize with other views

2. Create `components/visualizations/umap_stepped.py`
   - Use existing dimensionality reducers
   - Layer-by-layer visualization
   - Trajectory highlighting

3. Implement `components/cards/cluster_card.py`
   - LLM-generated descriptions
   - Feature importance display
   - Flow information
   - Interactive highlighting

### Phase 4: LLM Integration (Priority: High)
1. Implement `components/controls/api_keys_panel.py`
   - Secure API key input
   - Provider selection
   - Validation feedback

2. Create `core/llm_adapter.py`
   - Generic feature-based analysis
   - Cluster labeling for any model type
   - Path narrative generation

### Phase 5: Main Integration (Priority: High)
1. Implement `tabs/ff_networks.py`
   - Orchestrate all components
   - Handle state management
   - Coordinate visualizations
   - Export functionality

### Phase 6: Polish (Priority: Medium)
1. Create clinical CSS theme
2. Implement diagnostic scanner
3. Add 3D trajectory viewer
4. Create GPT placeholder tab

## Key Design Decisions

### State Management
- Use Dash's `dcc.Store` for session state
- Cache expensive computations (clustering, UMAP)
- Maintain synchronization between visualizations

### Performance Optimization
- Lazy loading for large datasets
- Progressive rendering for visualizations
- Debounced callbacks for smooth interaction

### User Experience
- Clear workflow: Upload → Configure → Analyze → Explore
- Helpful tooltips and documentation
- Export options for all visualizations
- Shareable URLs for specific views

## Success Criteria
1. Users can upload any PyTorch feedforward model
2. Clustering configuration is intuitive and flexible
3. Visualizations are interactive and synchronized
4. LLM-generated insights are meaningful
5. Clinical theme is professional and clean
6. Performance is smooth for typical models (< 10 layers, < 1000 samples)

## Future Extensions
- Support for more model formats
- Additional clustering algorithms
- Custom metric definitions
- Real-time training visualization
- Model comparison mode
- GPT/Transformer support

## Timeline Estimate
- Phase 1-2: 2-3 days (infrastructure + data input)
- Phase 3-4: 3-4 days (visualizations + LLM)
- Phase 5-6: 2-3 days (integration + polish)
- **Total: 7-10 days** for fully functional application

## Notes
- This is a fresh UI implementation that maximizes reuse of existing analysis code
- All heavy computation is delegated to existing, tested infrastructure
- The focus is on user experience and clinical design aesthetic
- Extensibility is built-in through the component architecture