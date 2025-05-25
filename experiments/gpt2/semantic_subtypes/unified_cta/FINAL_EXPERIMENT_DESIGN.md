# GPT-2 Semantic Subtypes: Final Unified CTA Experiment Design

## Overview
This experiment performs Archetypal Path Analysis (APA) on GPT-2 semantic representations, with innovations to handle fragmentation:

1. **Core APA Analysis** - Full trajectory analysis with cross-layer metrics
2. **Windowed Path Analysis** - Analyze paths in three windows to capture fragmentation patterns
3. **ETS Enhancement** - Add micro-clustering to reveal fine-grained patterns (optional comparison)

## Core Archetypal Path Analysis (PRIMARY GOAL)

### What We're Analyzing
- **566 words** from 8 semantic subtypes
- **12 layers** of GPT-2 (L0-L11)
- **Trajectories**: Each word's path through cluster assignments across layers

### Key Metrics (From Paper Section 4.1)

#### 1. Cross-Layer Centroid Similarity (ρᶜ)
- Cosine similarity between cluster centroids in adjacent layers
- Reveals if clusters maintain semantic consistency across layers

#### 2. Membership Overlap (J)
- Jaccard index between cluster memberships in adjacent layers
- Measures how cluster populations change between layers

#### 3. Trajectory Fragmentation (F)
- Entropy-based measure of path diversity
- High F = many different paths (fragmented)
- Low F = few dominant paths (converged)

#### 4. Semantic Coherence Metrics
- **Within-subtype coherence**: Do words from same semantic subtype follow similar paths?
- **Between-subtype differentiation**: Do different subtypes follow distinct paths?
- **Path purity**: Fraction of paths dominated by single semantic subtype

### Current Results Summary

#### Clustering Results (Gap Statistic K-means)
- **Early layers (0-3)**: 4, 2, 2, 2 clusters
- **Middle layers (4-7)**: 2 clusters each
- **Late layers (8-11)**: 2 clusters each
- **Pattern**: Initial diversity → rapid convergence to binary distinctions

#### Windowed Path Analysis
**Key Finding**: Massive convergence from early diversity to dominant patterns

| Window | Unique Paths | Dominant Path | Coverage |
|--------|--------------|---------------|----------|
| Early (L0-L3) | 19 | [L0_C1→L1_C1→L2_C1→L3_C1] | 27.2% |
| Middle (L4-L7) | 5 | [L4_C1→L5_C0→L6_C0→L7_C1] | 72.8% |
| Late (L8-L11) | 4 | [L8_C1→L9_C0→L10_C0→L11_C1] | 73.1% |

**Cross-Window Transitions**:
- Early→Middle: 19 paths converge to 5 (convergence ratio: 0.263)
- Middle→Late: 5 paths to 4 (convergence ratio: 0.800)

## Full Analysis Plan

### 1. Complete Cross-Layer Metrics (PRIORITY)
Calculate for all adjacent layer pairs:
- ρᶜ (centroid similarity)
- J (membership overlap)
- Path density
- Stability scores

### 2. Trajectory Metrics
For each word's full L0→L11 trajectory:
- Fragmentation score (F)
- Stability (consecutive same-cluster assignments)
- Convergence patterns

### 3. Semantic Subtype Analysis
For each of the 8 semantic subtypes:

**Animals** (45 words): cat, dog, bird, fish, horse, cow, bear, lion...
**Objects** (62 words): window, clock, computer, engine, table, door...
**Actions** (38 words): run, jump, walk, move, create, build...
**Properties** (41 words): small, large, good, wrong, happy, sad...
**Abstract** (29 words): freedom, justice, truth, beauty, idea...
**Social** (33 words): friend, family, group, team, society...
**Nature** (28 words): tree, water, sun, rain, mountain...
**Other** (290 words): remaining words

Calculate:
- Within-subtype path similarity
- Archetypal paths per subtype
- Between-subtype distances
- Subtype clustering quality

### 4. LLM Interpretation
Prepare data for LLM analysis:
- **All 28 windowed paths** with unique cluster labels
- **Cluster membership lists** (which words belong to each L{i}_C{j})
- **Transition patterns** between windows
- **Semantic subtype distributions** within paths

LLM tasks:
1. Label each unique cluster based on word membership
2. Interpret path meanings (what does [L0_C1→L1_C1→L2_C1→L3_C1] represent?)
3. Explain convergence patterns
4. Identify semantic organization principles

## ETS Micro-Clustering Enhancement (SECONDARY)

### Purpose
The dominant paths (especially the 72.8% middle path) likely contain substructure. ETS can reveal:
- Fine-grained semantic distinctions within macro clusters
- Whether convergence is semantic or artifactual

### Method
```python
# Within each macro cluster at each layer:
macro_cluster_activations = activations[labels == macro_id]
micro_labels = compute_ets_clustering(
    macro_cluster_activations,
    threshold_percentile=0.1
)
# Create hierarchical labels: L{layer}_C{macro}_M{micro}
```

### Expected Outcomes
- Dominant path [L4_C1→L5_C0→L6_C0→L7_C1] splits into semantic sub-paths
- Better separation of semantic subtypes
- More interpretable cluster assignments

## Deliverables

### Tables for Paper

**Table 1: Layer-wise Clustering and Metrics**
| Layer | k | Silhouette | ρᶜ to next | J to next | Unique paths through |
|-------|---|------------|------------|-----------|---------------------|
| 0 | 4 | 0.68 | 0.82 | 0.65 | 4 |
| 1 | 2 | 0.71 | 0.79 | 0.88 | 7 |
| ... | ... | ... | ... | ... | ... |

**Table 2: All Windowed Paths**
| Window | Path | Frequency | % | Example Words | Semantic Pattern |
|--------|------|-----------|---|---------------|------------------|
| Early | [L0_C1→L1_C1→L2_C1→L3_C1] | 154 | 27.2% | mouse, window, clock | Objects/Tools |
| Early | [L0_C0→L1_C1→L2_C1→L3_C1] | 131 | 23.1% | cat, dog, bird | Animals |
| ... | ... | ... | ... | ... | ... |

**Table 3: Semantic Subtype Coherence**
| Subtype | N | Most Common Path | Path Purity | Within-Coherence | 
|---------|---|------------------|-------------|------------------|
| Animals | 45 | [L4_C1→L5_C0→L6_C0→L7_C1] | 0.89 | 0.76 |
| Objects | 62 | [L4_C1→L5_C0→L6_C0→L7_C1] | 0.71 | 0.68 |
| ... | ... | ... | ... | ... |

### Visualizations
1. **Three Sankey diagrams** (early, middle, late windows)
2. **Convergence plot** showing path consolidation
3. **Semantic subtype trajectories** colored by type
4. **Cross-layer metric progression**

## Key Questions to Answer

1. **What drives the massive convergence from 19→5→4 paths?**
2. **Why does one path capture 72%+ of words in middle/late layers?**
3. **Do semantic subtypes maintain distinct paths or converge?**
4. **At which layers do semantic distinctions emerge/disappear?**
5. **What do the cross-layer metrics reveal about representation dynamics?**

## Implementation Status

✓ Clustering complete (Gap statistic k-means)
✓ Windowed path extraction with ALL paths
✓ Unique cluster labeling (L{layer}_C{cluster})
✓ Basic trajectory analysis

⏳ Complete cross-layer metrics (ρᶜ, J, F)
⏳ Semantic subtype analysis
⏳ LLM interpretation of paths and clusters
⏳ ETS micro-clustering comparison
⏳ Final visualizations