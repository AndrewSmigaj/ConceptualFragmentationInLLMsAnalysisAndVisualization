# Phase 2: Concept Trajectory Analysis Implementation

## 🎯 Objectives

Now that we've fixed the clustering issue, Phase 2 focuses on the core CTA goals from the paper:
1. **Full-scale execution** - Run on all 12 GPT-2 layers with 774 words
2. **Concept trajectory tracking** - Identify paths words follow through clusters
3. **Archetypal path analysis** - Find common trajectory patterns
4. **Semantic coherence analysis** - Measure within-subtype and between-subtype patterns

## 📋 Task Breakdown

### 1. Full Pipeline Execution (Priority: HIGH)
**Goal**: Run the complete pipeline on all 12 GPT-2 layers

**Steps**:
```bash
# Run full analysis
python run_unified_pipeline.py --experiment full

# Monitor progress
tail -f results/unified_cta_config/[latest]/reports/unified_cta_final_report.txt
```

**Expected Outcomes**:
- 12 layers processed successfully
- Each layer shows k > 1 (no single clusters)
- Coverage improves from ~15% to >60%
- Purity maintained >75%

**Time Estimate**: 30-45 minutes (gap statistic is computationally intensive)

### 2. Path Analysis Validation (Priority: HIGH)
**Goal**: Verify words follow diverse trajectories through layers

**Implementation Plan**:
```python
# Create path diversity analysis script
experiments/gpt2/semantic_subtypes/unified_cta/analysis/
├── path_diversity_analyzer.py      # Calculate trajectory metrics
├── subtype_trajectory_analyzer.py  # Check if subtypes stay together
└── trajectory_visualizer.py        # Create trajectory plots
```

**Key Metrics**:
- **Path Entropy**: Measure of trajectory diversity (target: >2.0)
- **Subtype Coherence**: Do semantic subtypes follow similar paths?
- **Layer Transition Patterns**: How clusters merge/split across layers

**Validation Criteria**:
- At least 30% of words should have unique trajectories
- Semantic subtypes should show some clustering tendency
- No "convergence to single path" pattern

### 3. Cluster Interpretability (Priority: MEDIUM)
**Goal**: Generate meaningful descriptions for each cluster

**Approach**:
- I analyze cluster contents directly (no LLM API)
- Focus on semantic patterns within clusters
- Create cluster "fingerprints" based on word types

**Output Format**:
```json
{
  "layer_0": {
    "cluster_0": {
      "name": "Abstract Concepts",
      "dominant_subtypes": ["abstract", "cognitive"],
      "example_words": ["think", "believe", "understand"],
      "characteristics": "Words related to mental processes"
    }
  }
}
```

### 4. Comparative Analysis (Priority: MEDIUM)
**Goal**: Document improvements over baseline

**Create Comparison Report**:
```markdown
# Clustering Improvements

## Before (Single Cluster Problem)
- All layers: k=1
- Coverage: ~15%
- No meaningful trajectories

## After (Unified CTA)
- Layer 0: k=4, Layer 5: k=2, etc.
- Coverage: 60-80%
- Rich trajectory patterns
```

**Visualizations**:
1. **Cluster Count Comparison** - Bar chart showing k values per layer
2. **Coverage Improvement** - Line plot of coverage across layers
3. **Trajectory Diversity** - Sankey diagram of word flows

### 5. Performance Optimization (Priority: LOW)
**Goal**: Speed up analysis for future runs

**Areas to Optimize**:
- Cache gap statistic reference distributions
- Parallelize layer processing
- Add checkpointing for long runs

**Implementation**:
```python
# Add to config
cache_gap_references: bool = True
parallel_layers: bool = True
checkpoint_interval: int = 3  # Save every 3 layers
```

## 🔍 Quality Assurance Checklist

### Data Validation
- [ ] All 774 words processed
- [ ] All 12 layers analyzed
- [ ] No data loss during processing

### Clustering Quality
- [ ] No single-cluster layers
- [ ] Reasonable k values (2-10)
- [ ] Balanced cluster sizes (ratio < 10:1)

### Micro-clustering Performance
- [ ] Coverage >60% average
- [ ] Purity >75% average
- [ ] Multiple micro-clusters per macro

### Path Analysis
- [ ] Diverse trajectories confirmed
- [ ] Subtype patterns identified
- [ ] No degenerate paths

## 📊 Success Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Layers with k>1 | 100% | >90% |
| Average Coverage | >70% | >60% |
| Average Purity | >80% | >75% |
| Path Diversity | >2.0 | >1.5 |
| Unique Trajectories | >30% | >20% |

## 🚀 Implementation Strategy

### Phase 2.1: Full Pipeline Run (Day 1)
1. Run full 12-layer analysis
2. Monitor and debug any issues
3. Collect all metrics

### Phase 2.2: Path Analysis (Day 2)
1. Implement trajectory analyzers
2. Calculate diversity metrics
3. Identify semantic patterns

### Phase 2.3: Reporting (Day 3)
1. Generate cluster interpretations
2. Create comparison visualizations
3. Write final analysis report

## 📝 Deliverables

1. **Full Analysis Results**
   - `results/unified_cta_config/[timestamp]/`
   - All 12 layers processed
   - Complete metrics and trajectories

2. **Path Analysis Report**
   - Trajectory diversity metrics
   - Subtype coherence analysis
   - Visualization of paths

3. **Cluster Interpretations**
   - Semantic descriptions for all clusters
   - Pattern analysis across layers
   - Subtype organization insights

4. **Comparison Documentation**
   - Before/after metrics
   - Improvement visualizations
   - Technical report for paper

## 🔧 Troubleshooting Guide

### If pipeline fails on specific layer:
1. Check memory usage (gap statistic is memory intensive)
2. Verify data integrity for that layer
3. Try with reduced k_range

### If coverage remains low:
1. Adjust percentile thresholds per layer
2. Increase alpha search range
3. Consider different distance metrics

### If paths converge:
1. Check Procrustes alignment
2. Verify cluster stability
3. Analyze layer-specific patterns

---

**Ready to Execute**: Start with full pipeline run, then proceed with analysis