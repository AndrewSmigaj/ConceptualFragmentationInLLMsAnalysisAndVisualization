# Phase 2: Concept Trajectory Analysis - Paper-Aligned Implementation

## 🎯 Core CTA Objectives (From Paper)

Focus strictly on the Concept Trajectory Analysis methodology as defined in our paper:

1. **Track concept trajectories** through clustered activation spaces
2. **Identify archetypal paths** - common trajectories that many words follow  
3. **Measure within-subtype coherence** - do words of same type follow similar paths?
4. **Analyze between-subtype differentiation** - do different types follow different paths?
5. **Generate LLM narratives** - I interpret the paths (no external API)

## 📋 Implementation Tasks

### 1. Full Pipeline Execution (Priority: HIGH)
**Goal**: Process all 774 words through 12 GPT-2 layers

```bash
python run_unified_pipeline.py --experiment full
```

**Expected Output**:
- Cluster assignments for each word at each layer
- Full trajectory data: word -> [L0C2, L1C0, L2C3, ..., L11C1]
- Coverage and purity metrics per layer

### 2. Archetypal Path Analysis (Priority: HIGH)
**Goal**: Identify common trajectory patterns as specified in the paper

The existing `path_analysis.py` already implements:
- `identify_archetypal_paths()` - finds frequent trajectory patterns
- `calculate_trajectory_metrics()` - computes path diversity metrics

**Key Metrics from Paper**:
- **Path frequency**: How many words follow each archetypal path
- **Path diversity**: Entropy of trajectory distribution
- **Convergence patterns**: Do paths merge/diverge at specific layers?

### 3. Semantic Coherence Analysis (Priority: HIGH)
**Goal**: Quantify within-subtype trajectory coherence

**Implementation** (using existing functions):
```python
# For each semantic subtype (concrete_nouns, abstract_nouns, etc.)
# 1. Extract trajectories for all words in subtype
# 2. Calculate trajectory similarity within subtype
# 3. Compare to random baseline
```

**Metrics**:
- **Intra-subtype trajectory similarity**: Average pairwise similarity of paths within a subtype
- **Subtype coherence score**: How much more similar are paths within vs across subtypes

### 4. Between-Subtype Differentiation (Priority: HIGH)
**Goal**: Measure how different semantic types follow different paths

**Analysis**:
- Compare trajectory distributions between subtypes
- Identify layer-specific differentiation points
- Quantify separation using existing metrics

### 5. LLM-Powered Interpretation (Priority: MEDIUM)
**Goal**: Generate interpretable narratives for paths (I do this directly)

Using existing `DirectInterpreter` class:
- Interpret archetypal paths based on words that follow them
- Describe semantic evolution along trajectories
- Identify conceptual transitions at each layer

## 📊 Key Results to Generate (Matching Paper Sections)

### Table: Optimal Clustering Results
```
Layer | Optimal k | Silhouette | Coverage | Purity
------|-----------|------------|----------|--------
0     | 4         | X.XX       | XX%      | XX%
...   | ...       | ...        | ...      | ...
11    | 2         | X.XX       | XX%      | XX%
```

### Table: Archetypal Paths
```
Path ID | Trajectory              | Frequency | Example Words | Interpretation
--------|------------------------|-----------|---------------|----------------
1       | [L0C0→L1C2→...→L11C1] | 15.2%     | run, jump...  | Action progression
2       | [L0C1→L1C1→...→L11C0] | 12.8%     | red, blue...  | Property stability
```

### Figure: Within-Subtype Coherence
- Bar chart showing coherence scores for each semantic subtype
- Comparison to random baseline

### Figure: Layer-wise Semantic Evolution
- Heatmap showing when different subtypes differentiate
- Critical layers where semantic organization emerges

## ⚠️ What NOT to Add (Without Permission)

Following your guidance, I will NOT add:
- New metrics not discussed in the paper
- Additional visualizations beyond those specified
- External analysis tools or frameworks
- Complexity beyond the core CTA methodology

## 📝 Deliverables Aligned with Paper

1. **Full trajectory data** for all 774 words across 12 layers
2. **Archetypal path identification** with frequencies and examples
3. **Within-subtype coherence scores** for all 8 semantic categories
4. **Between-subtype differentiation analysis** 
5. **LLM-generated interpretations** of key paths (by me, not external API)
6. **Tables and figures** matching the placeholders in the paper

## 🔧 Using Existing Code

The implementation leverages:
- `path_analysis.py` - `identify_archetypal_paths()`, `calculate_trajectory_metrics()`
- `llm/interpretability.py` - `DirectInterpreter` for path narratives
- `results_manager.py` - Saving all results in structured format
- `visualization/unified_visualizer.py` - Only for paper-specified visualizations

No new functionality needed - just proper application of existing CTA pipeline.

---

**Next Step**: Run full pipeline and begin trajectory analysis using existing tools