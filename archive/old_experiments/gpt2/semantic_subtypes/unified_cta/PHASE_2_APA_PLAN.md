# Phase 2: Archetypal Path Analysis with Unified CTA

## 🎯 Core Objectives

Continue with our unified CTA implementation (Gap statistic + centroid-based ETS) to perform Archetypal Path Analysis as defined in the paper:

1. **Execute full pipeline** - Process all 774 words through 12 GPT-2 layers using our fixed clustering
2. **Extract archetypal paths** - Identify common trajectories using the hybrid clustering approach
3. **Analyze semantic coherence** - Measure within-subtype and between-subtype patterns
4. **Generate APA metrics** - Cross-layer metrics (ρᶜ, J, F) from the paper
5. **LLM interpretation** - I narrate the archetypal paths

## 📋 Implementation Tasks

### 1. Full Pipeline Execution (Priority: HIGH)
**Goal**: Run our unified CTA pipeline on all 12 layers

```bash
python run_unified_pipeline.py --experiment full
```

**What this gives us**:
- **Macro clusters** via gap statistic (avoiding single-cluster problem)
- **Micro clusters** via centroid-based ETS (improved coverage)
- **Full trajectories** for each word: π_i = [c_i^0, c_i^1, ..., c_i^11]

### 2. Archetypal Path Extraction (Priority: HIGH)
**Goal**: Identify common paths using our existing APA implementation

Using `path_analysis.py`:
```python
# Extract archetypal paths
archetypal_paths = path_analyzer.identify_archetypal_paths(
    trajectories, 
    min_frequency=3  # Paths followed by at least 3 words
)

# Calculate APA metrics from paper
path_metrics = path_analyzer.calculate_trajectory_metrics(
    trajectories, 
    word_subtypes
)
```

### 3. Cross-Layer APA Metrics (Priority: HIGH)
**Goal**: Calculate the specific metrics from Section 4.1 of the paper

**Centroid Similarity (ρᶜ)**:
- Measure if clusters across layers represent similar concepts
- Already implemented in our quality checks

**Membership Overlap (J)**:
- Jaccard similarity of cluster membership across layers
- Implemented in path analysis

**Trajectory Fragmentation (F)**:
- Measure of path stability/switching
- Part of trajectory metrics

### 4. Semantic Subtype Analysis (Priority: HIGH)
**Goal**: Apply APA to understand semantic organization

For each of the 8 semantic subtypes:
1. Extract all trajectories for words in that subtype
2. Calculate within-subtype path coherence
3. Identify subtype-specific archetypal paths
4. Measure between-subtype differentiation

### 5. LLM-Powered APA Interpretation (Priority: MEDIUM)
**Goal**: Generate narratives for archetypal paths

Using `DirectInterpreter`:
```python
# For each archetypal path
for path_data in archetypal_paths:
    narrative = interpreter.narrate_path(
        path_data['path'],
        path_data['example_words'],
        cluster_names
    )
```

## 📊 APA Results to Generate

### Table: Layer-wise Clustering Results (Gap + ETS)
```
Layer | Gap-k | ETS Coverage | ETS Purity | Total Clusters
------|-------|--------------|------------|----------------
0     | 4     | 45%          | 87%        | 48 (4 macro × ~12 micro)
...   | ...   | ...          | ...        | ...
```

### Table: Top Archetypal Paths
```
Path ID | Trajectory Pattern      | Frequency | Semantic Type | Example Words
--------|------------------------|-----------|---------------|---------------
AP1     | [L0C0→L3C2→L6C1→L11C0]| 8.5%      | Action verbs  | run, jump, move
AP2     | [L0C1→L3C1→L6C3→L11C2]| 6.2%      | Abstract noun | freedom, justice
```

### Figure: APA Semantic Evolution
- Sankey diagram showing how semantic subtypes flow through clusters
- Critical layers where subtypes converge/diverge

### Analysis: Cross-Layer Metrics
- ρᶜ values showing conceptual persistence
- J values showing membership stability  
- F values showing trajectory fragmentation points

## 🔧 Why Our Hybrid Approach Enables Better APA

1. **Gap statistic macro-clustering**: Ensures we have multiple paths to analyze (not everything in one cluster)
2. **Centroid-based ETS micro-clustering**: Provides finer-grained trajectory resolution
3. **Two-tier structure**: Captures both coarse semantic categories and fine distinctions
4. **Layer-specific optimization**: Adapts to changing representations across depth

## 📝 Deliverables

1. **Complete APA dataset**: All 774 word trajectories with hybrid clustering
2. **Archetypal path catalog**: Common patterns with frequencies and examples
3. **Cross-layer metric analysis**: ρᶜ, J, F values across all layers
4. **Semantic coherence report**: Within/between subtype analysis
5. **LLM narratives**: Interpretable descriptions of key archetypal paths

## ✅ This Aligns With Paper Because:

- Uses APA methodology from Sections 3-4
- Calculates cross-layer metrics (ρᶜ, J, F) from Section 4.1
- Applies to GPT-2 semantic subtypes case study (Section 7.2)
- Generates LLM narratives as specified
- Our hybrid clustering fixes enable meaningful APA (vs single clusters)

---

**Next Step**: Run full pipeline with hybrid clustering to generate APA trajectories