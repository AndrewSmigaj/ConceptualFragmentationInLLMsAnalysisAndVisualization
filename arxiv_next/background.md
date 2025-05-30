# Background: GPT-2 10k Token Probe Experiment

## Overview
This experiment analyzed how GPT-2 organizes its 10,000 most frequent tokens across its 12 layers using Concept Trajectory Analysis (CTA) with k=10 clustering.

## Key Experimental Details

### Data Selection
- **Model**: GPT-2 base (117M parameters, 12 layers)
- **Tokens**: Top 10,000 most frequent tokens from GPT-2's vocabulary
  - Selected by token frequency in training data (lower token IDs = more common)
  - Includes common words, punctuation, and important subwords
  - Full vocabulary is 50,257 tokens, we analyzed the most frequent 20%

### Token Composition
The 10k tokens consisted of:
- Complete words (the, and, of, etc.)
- Subword units (ing, ed, ly suffixes)
- Punctuation and special characters
- Tokens with leading spaces (indicating word boundaries)
- Common morphological patterns (plurals, verb forms, etc.)

### Clustering Approach
- **Method**: K-means clustering with k=10 per layer
- **Reasoning**: k=10 chosen after extensive experimentation
  - Too few clusters (k=3, k=5) missed nuanced categories
  - k=10 revealed meaningful semantic and grammatical distinctions
- **Layer-specific**: Each layer clustered independently
- **Labels**: LLM-based semantic labeling (NOT rule-based)

### Analysis Pipeline
1. **Activation Extraction**: Extracted activations for all 10k tokens from each of GPT-2's 12 layers
2. **Clustering**: Applied k-means (k=10) to each layer independently
3. **Trajectory Tracking**: Tracked how tokens move between clusters across layers
4. **Windowed Analysis**: Analyzed early (0-3), middle (4-7), and late (8-11) layers separately
5. **LLM Labeling**: Used Claude to generate semantic labels based on actual cluster contents
6. **Visualization**: Created Sankey diagrams and trajectory visualizations

## Key Findings from Session Context

### Layer Organization Evolution
- **Early layers (0-3)**: More specific linguistic categories
- **Middle layers (4-7)**: Transitional organization
- **Late layers (8-11)**: Abstract grammatical organization emerges

### Example Layer 11 Clusters (k=10)
- L11_C0: "Common Modifiers" (as, other, time, if, two)
- L11_C3: "Core Function Words" (the, of, and, for, with)
- L11_C6: "Spatial & Descriptive" (under, high, hand, head)
- L11_C9: "Core Grammar" (to, that, is, was)

### Critical Lessons Learned
1. **LLM labeling is essential**: Rule-based labeling systems completely miss the semantic nuance
2. **Semantic purity matters**: Need to calculate how well clusters match their semantic labels
3. **All clusters are meaningful**: Each of the 10 clusters captures distinct patterns

## Technical Implementation

### Key Scripts
1. `extract_top_10k_tokens.py` - Extracts the 10k most frequent tokens and their activations
2. `run_k10_analysis_pipeline.py` - Runs the complete analysis using unified CTA framework
3. `prepare_llm_analysis_top_10k.py` - Prepares data for LLM-based semantic labeling

### Data Flow
```
GPT-2 Vocabulary (50k tokens)
    ↓
Top 10k Most Frequent Tokens
    ↓
Extract Activations (12 layers × 10k tokens × 768 dims)
    ↓
K-means Clustering (k=10 per layer)
    ↓
Trajectory Analysis (track cluster transitions)
    ↓
LLM Semantic Labeling (Claude analyzes cluster contents)
    ↓
Visualization (Sankey diagrams, trajectories)
```

### Important Files and Locations
- **Cluster labels**: `experiments/gpt2/all_tokens/llm_labels_k10/cluster_labels_k10.json`
- **Clustering results**: `experiments/gpt2/all_tokens/clustering_results_k10/`
- **Windowed analysis**: `experiments/gpt2/all_tokens/k10_analysis_results/windowed_analysis_k10.json`
- **Visualizations**: `experiments/gpt2/all_tokens/k10_analysis_results/sankey_*_k10.html`

## Theoretical Implications

### Why k=10?
- Balances granularity with interpretability
- Captures both coarse grammatical categories and fine semantic distinctions
- Reveals hierarchical organization: broad categories subdivide into meaningful subcategories

### What This Reveals About GPT-2
1. **Hierarchical Processing**: Early layers capture surface features, late layers capture abstract grammar
2. **Semantic-to-Syntactic Transition**: Clear progression from meaning-based to function-based organization
3. **Stable Pathways**: Certain token groups (e.g., function words) maintain consistent trajectories

## Next Steps for Paper

### Research Questions
1. How does cluster granularity (k) affect our understanding of neural organization?
2. Can we identify universal organizational principles across different values of k?
3. What is the relationship between token frequency and trajectory stability?

### Proposed Analyses
1. Compare k=5, k=10, k=15 to understand organizational hierarchy
2. Analyze trajectory fragmentation as a function of k
3. Study how semantic purity changes with cluster granularity
4. Investigate whether rare tokens show different organizational patterns

### Key Messages
1. GPT-2 organizes language hierarchically, with granularity revealing different levels of structure
2. The 10k most frequent tokens show remarkably stable organizational patterns
3. LLM-assisted analysis is crucial for understanding semantic organization
4. The choice of k in clustering is not arbitrary but reveals different organizational scales

## Important Context for Claude

When analyzing this experiment:
1. The 10k tokens are the MOST FREQUENT, not just the first 10k by ID
2. k=10 was chosen after extensive experimentation, not arbitrarily
3. LLM labeling (using Claude) was essential - rule-based labeling failed completely
4. The semantic labels should reflect actual cluster contents, not preconceived categories
5. All 10 clusters (0-9) are meaningful and should be included in analysis
6. The windowed analysis (early/middle/late) reveals phase transitions in organization

This experiment demonstrates that GPT-2's organization of its most common tokens is both hierarchical and dynamic, with clear transitions from semantic to syntactic organization as information flows through the network.