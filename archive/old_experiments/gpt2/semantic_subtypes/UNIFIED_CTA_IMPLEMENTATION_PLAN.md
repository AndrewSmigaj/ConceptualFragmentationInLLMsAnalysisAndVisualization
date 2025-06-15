# Unified CTA Implementation Plan for GPT-2 Semantic Subtypes

## Overview
Implements a two-tier clustering strategy: macro clusters for trajectory analysis and micro clusters for explainability, analyzing 774 single-token words from 8 semantic subtypes across GPT-2's 13 layers.

## Core Design Principles
1. **Reuse existing infrastructure** - Leverage existing ETS functions, clustering code, and visualization tools
2. **Layer-aware parameters** - Different k ranges and thresholds for early/middle/late layers
3. **Fallback strategies** - Graceful degradation when quality thresholds can't be met
4. **Clear separation** - Macro clustering (structural) vs micro clustering (explainability)

## Implementation Phases

### Phase 1: Enhanced Preprocessing
**Goal**: Prepare activations with layer alignment

#### 1.1 Preprocessing Pipeline
- [x] Use existing StandardScaler from `sklearn`
- [x] Use existing PCA implementation
- [ ] Add Procrustes alignment between successive layers
  - Use `scipy.linalg.orthogonal_procrustes`
  - Align layer L+1 to layer L's coordinate system
- [ ] Save preprocessed data with clear versioning

```python
# Reuse from existing ets_revised_wrapper.py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
```

### Phase 2: Structural Clustering (Macro Level)
**Goal**: Create interpretable macro clusters for trajectories

#### 2.1 Layer-Specific K-Selection
- [ ] Implement gap statistic using existing K-means
  ```python
  # Layer-specific k ranges (not one-size-fits-all)
  K_RANGES = {
      'early': range(2, 6),    # Layers 0-3
      'middle': range(3, 8),   # Layers 4-8  
      'late': range(4, 10)     # Layers 9-12
  }
  ```
- [ ] Combine metrics with weights:
  - Silhouette score (40% weight)
  - Gap statistic (30% weight)
  - Semantic purity (30% weight)
- [ ] Fallback: If purity <65% at all k, use best available with warning

#### 2.2 Macro Clustering Execution
- [ ] Use existing `KMeans` from sklearn
- [ ] Store cluster assignments and centroids
- [ ] Calculate existing purity metric from `find_optimal_k_silhouette.py`

### Phase 3: Explainability Layer (Micro Clusters)
**Goal**: Create rule-based explanations within macro clusters

#### 3.1 Centroid-Based ETS (Per Macro Cluster)
- [ ] For each macro cluster:
  ```python
  # Start with 30th percentile (more reasonable than 20th)
  initial_percentile = 30
  
  # Use existing compute_dimension_thresholds but relative to centroid
  centroid = macro_cluster.mean(axis=0)
  distances = np.abs(points - centroid)
  thresholds = np.percentile(distances, initial_percentile, axis=0)
  ```

#### 3.2 Coverage-Purity Optimization
- [ ] Grid search α ∈ [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
- [ ] Targets (with flexibility):
  - Coverage: ≥80% (hard requirement)
  - Purity: ≥70% (soft target, best effort)
- [ ] Use existing ETS functions with scaled thresholds:
  ```python
  scaled_thresholds = thresholds * alpha
  labels = compute_ets_clustering(data, thresholds=scaled_thresholds)
  ```

#### 3.3 Micro-Cluster Constraints
- [ ] Minimum size: ≥3 tokens (more realistic than 5)
- [ ] Merge rule: centroids within 1.0σ (not 0.5σ)
- [ ] Anomaly bucket for residuals

### Phase 4: Path Analysis
**Goal**: Track semantic evolution through layers

#### 4.1 Trajectory Construction
- [ ] Use macro cluster IDs only (micro clusters for diagnostics)
- [ ] Build transition matrix between consecutive layers
- [ ] Identify splitting/merging events

#### 4.2 Path Metrics
```python
# Clear definitions:
fragmentation_score = n_splits / n_initial_paths
path_purity = n_consistent_subtype / n_total_in_path  
subtype_entropy = -sum(p * log(p)) for p in subtype_proportions
micro_turnover = n_micro_changes / n_total_transitions
```

#### 4.3 Path Selection
- [ ] Rank by composite score:
  - High fragmentation (interesting)
  - High purity (meaningful)
  - Multiple subtypes (diverse)
- [ ] Select top 20 paths

### Phase 5: LLM Interpretability (Simplified)
**Goal**: Generate human-readable explanations

#### 5.1 Cluster Naming
- [ ] Use existing LLM infrastructure from `concept_fragmentation/llm/`
- [ ] Single-pass generation (no consensus needed)
- [ ] Format: "{adjective} {noun}" (2-3 words max)

#### 5.2 Path Narration
- [ ] Template-based with LLM fill-in:
  ```
  "Words starting as [CLUSTER_0] in early layers,
   transition to [CLUSTER_5] in middle layers,
   representing a shift from [LLM_INTERPRETATION]"
  ```

### Phase 6: Diagnostics & Quality Control
**Goal**: Ensure results meet standards

#### 6.1 Quality Metrics
- [ ] Macro cluster purity ≥65% (or best available)
- [ ] Coverage ≥80% (hard requirement)
- [ ] Fragmentation ≤0.5 (warning if higher)
- [ ] PCA variance: PC1 >50% triggers axis-dominance warning

#### 6.2 Diagnostic Outputs
- [ ] Spectral clustering comparison (sanity check)
- [ ] Cluster size distribution plots
- [ ] Layer-wise stability metrics

### Phase 7: Visualization & Reporting
**Goal**: Communicate findings effectively

#### 7.1 Reuse Existing Visualizations
- [ ] Sankey diagrams from `generate_cluster_flow_diagrams.py`
- [ ] Silhouette plots from existing code
- [ ] Add new: τ-threshold heatmaps

#### 7.2 Structured Output
```
results/
├── unified_cta_config.json      # All parameters used
├── macro_clusters/              
│   └── layer_{i}_clusters.json  # Assignments, centroids, metrics
├── micro_clusters/              
│   └── layer_{i}_macro_{j}/     # ETS results per macro cluster
├── paths/
│   ├── trajectories.json        # All paths with metrics
│   └── top_20_paths.json        # Selected interesting paths
└── reports/
    ├── quality_metrics.json     # Did we meet thresholds?
    ├── llm_interpretations.json # Cluster names, narratives
    └── diagnostic_warnings.txt  # Any issues found
```

## Implementation Strategy

### Order of Development
1. **Preprocessing upgrades** (2 hours)
   - Add Procrustes alignment
   - Test on layers 0→1

2. **Gap statistic + k-selection** (3 hours)
   - Implement gap statistic
   - Layer-specific k ranges
   - Multi-metric optimization

3. **Centroid-based ETS** (4 hours)
   - Modify existing ETS to work per-cluster
   - Coverage-purity optimization
   - Merge rules

4. **Path construction** (2 hours)
   - Transition matrices
   - Metric calculations
   - Path ranking

5. **LLM integration** (2 hours)
   - Reuse existing LLM client
   - Template-based generation

6. **Diagnostics & visualization** (3 hours)
   - Quality checks
   - Reuse existing viz code
   - New heatmaps

### Key Design Decisions

1. **No reimplementation** - Use existing ETS, K-means, visualization code
2. **Layer-aware parameters** - Different k ranges and percentiles by layer depth
3. **Realistic thresholds** - 30th percentile, 3+ tokens, 1.0σ merge distance
4. **Graceful degradation** - Best-effort when thresholds can't be met
5. **Clear metric definitions** - Unambiguous formulas for all scores

### Success Criteria
- **Technical**: Pipeline runs end-to-end without errors
- **Quality**: Meets thresholds on ≥80% of layers
- **Interpretability**: LLM can name ≥90% of clusters meaningfully
- **Insights**: Discovers semantic evolution patterns in GPT-2

## Next Steps
1. Create `unified_cta/` directory structure
2. Start with preprocessing module
3. Test each component on layer 0 before full pipeline
4. Iterate based on diagnostic feedback