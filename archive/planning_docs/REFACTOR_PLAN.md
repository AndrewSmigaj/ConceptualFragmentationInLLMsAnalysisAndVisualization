# Repository Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring plan to establish single sources of truth throughout the repository and eliminate duplicate implementations. The refactoring will create a clean, maintainable architecture with clear module boundaries and consistent interfaces.

## Current Issues

1. **Multiple Sankey Implementations**:
   - `generate_sankey_diagrams.py` (parameterized, base implementation)
   - `generate_k10_sankeys.py` (k=10 specific)
   - `generate_colored_sankeys_k10.py` (colored paths)
   - `generate_enhanced_sankeys_k10.py` (custom positioning - creates "orange circle")

2. **Multiple Labeling Implementations**:
   - At least 15 different labeling scripts in `experiments/gpt2/all_tokens/`
   - Various approaches: rule-based, LLM-based, consistency-focused
   - Final version: `create_primary_secondary_labels_k10.py`

3. **Duplicated Visualization Code**:
   - Sankey generation in multiple places
   - Trajectory visualization duplicated across experiments
   - Dashboard components scattered

4. **Configuration Management**:
   - Hard-coded values in multiple files
   - No central configuration for experiments
   - Inconsistent parameter passing

5. **Data Flow Issues**:
   - Multiple sources of truth for cluster labels
   - Inconsistent file naming conventions
   - No clear data pipeline

## Proposed Architecture

### 1. Core Library Structure

```
concept_fragmentation/
├── clustering/
│   ├── __init__.py
│   ├── base.py              # BaseClusterer interface
│   ├── kmeans.py            # KMeansClusterer implementation
│   └── paths.py             # Path extraction and analysis
│
├── labeling/
│   ├── __init__.py
│   ├── base.py              # BaseLabeler interface
│   ├── consistent.py        # ConsistentLabeler (primary/secondary approach)
│   ├── semantic_purity.py   # Semantic purity calculations
│   └── llm_analyzer.py      # Direct LLM analysis utilities
│
├── visualization/
│   ├── __init__.py
│   ├── sankey.py            # Single SankeyGenerator class
│   ├── trajectory.py        # TrajectoryVisualizer class
│   ├── stepped.py           # SteppedLayerVisualizer class
│   └── utils.py             # Shared visualization utilities
│
├── experiments/
│   ├── __init__.py
│   ├── base.py              # BaseExperiment class
│   └── config.py            # ExperimentConfig class
│
└── persistence/
    ├── __init__.py
    └── experiment_state.py   # Save/load experiment state
```

### 2. Experiment Structure

```
experiments/
├── configs/
│   ├── gpt2_k10.yaml
│   ├── gpt2_k5.yaml
│   └── heart_disease.yaml
│
├── gpt2/
│   ├── __init__.py
│   ├── run_experiment.py    # Main entry point
│   └── analysis/
│       ├── archetypal_paths.py
│       └── semantic_analysis.py
│
└── shared/
    ├── __init__.py
    └── utilities.py
```

### 3. Single Sources of Truth

#### SankeyGenerator
```python
# concept_fragmentation/visualization/sankey.py
class SankeyGenerator:
    """Single source of truth for Sankey diagram generation."""
    
    def __init__(self, config: SankeyConfig):
        self.config = config
    
    def generate(self, 
                 windowed_data: dict,
                 labels: dict,
                 options: SankeyOptions) -> go.Figure:
        """Generate Sankey with all options."""
        # Options include:
        # - colored_paths: bool
        # - top_n: int
        # - show_purity: bool
        # - legend_position: str
        # - label_last_layer: bool
```

#### ConsistentLabeler
```python
# concept_fragmentation/labeling/consistent.py
class ConsistentLabeler:
    """Single source of truth for consistent cluster labeling."""
    
    def label_clusters(self,
                      cluster_data: dict,
                      method: str = 'primary_secondary') -> dict:
        """Generate consistent labels across layers."""
        # Implements the primary/secondary approach
        # that was finalized in create_primary_secondary_labels_k10.py
```

## Migration Plan

### Phase 1: Create Core Structure (Day 1)
1. Create new directory structure under `concept_fragmentation/`
2. Define base interfaces (BaseClusterer, BaseLabeler, etc.)
3. Create configuration classes

### Phase 2: Migrate Sankey Generation (Day 2)
1. Consolidate all Sankey implementations into single class
2. Port features from each implementation:
   - Basic Sankey from `generate_sankey_diagrams.py`
   - Colored paths from `generate_colored_sankeys_k10.py`
   - Fix label positioning issues
3. Create comprehensive tests

### Phase 3: Migrate Labeling (Day 3)
1. Port `create_primary_secondary_labels_k10.py` to ConsistentLabeler
2. Archive all other labeling scripts
3. Update all references

### Phase 4: Experiment Refactoring (Day 4)
1. Create experiment configuration system
2. Refactor GPT-2 experiments to use new structure
3. Create single entry point for each experiment

### Phase 5: Archive Old Code (Day 5)
1. Move deprecated implementations to `archive/deprecated/`
2. Update all imports and references
3. Verify nothing breaks

### Phase 6: Documentation (Day 6)
1. Create comprehensive API documentation
2. Update README files
3. Create architecture diagram

## Official Implementations

The following will become the official implementations:

1. **Sankey Generation**: Merge best features from all implementations
   - Base: `generate_sankey_diagrams.py` (parameterized structure)
   - Colors: `generate_colored_sankeys_k10.py` (path coloring)
   - Labels: Fix from `generate_enhanced_sankeys_k10.py` (but without custom positioning)

2. **Cluster Labeling**: `create_primary_secondary_labels_k10.py` approach
   - Primary labels for consistency
   - Secondary labels for distinction
   - Alphabetical ordering for deterministic results

3. **Path Analysis**: `windowed_analysis_k10.json` format
   - Archetypal paths with frequencies
   - Semantic labels
   - Representative tokens

## Architecture Diagram

See `ARCHITECTURE.yaml` for machine-readable architecture specification.

## Success Criteria

1. Single import path for each functionality
2. No duplicate implementations
3. Clear configuration management
4. All tests passing
5. Documentation complete
6. Architecture diagram available

## Timeline

- **Week 1**: Core refactoring (Phases 1-4)
- **Week 2**: Testing and documentation (Phases 5-6)

## Risk Mitigation

1. Create full backup before starting
2. Implement changes incrementally
3. Maintain backward compatibility during transition
4. Keep detailed migration log

## Next Steps

1. Review and approve this plan
2. Create backup of current state
3. Begin Phase 1 implementation
4. Track progress in `REFACTOR_LOG.md`