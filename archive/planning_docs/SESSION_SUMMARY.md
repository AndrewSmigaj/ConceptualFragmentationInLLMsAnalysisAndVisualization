# CTA Repository Session Summary

## Quick Start for New Sessions

1. **Read the Architecture**: Start by reviewing `ARCHITECTURE.yaml` for the planned structure
2. **Check Current State**: Review `REFACTOR_PLAN.md` for refactoring status
3. **Understand the Problem**: We're analyzing how tokens move through transformer layers using clustering

## Current State (as of last session)

### What Works
- **K=10 clustering** on GPT-2's 10,000 most frequent tokens
- **Primary/Secondary labeling system** for consistent cluster labels
- **Windowed analysis** (early: layers 0-3, middle: 4-7, late: 8-11)
- **Semantic purity scores** showing cluster homogeneity

### Key Files
- **Latest Sankey Fix**: `generate_fixed_sankeys_k10.py` (fixes label count issue)
- **Official Labels**: `llm_labels_k10/cluster_labels_k10.json`
- **Analysis Results**: `k10_analysis_results/windowed_analysis_k10.json`

### Known Issues
1. **Multiple implementations** of the same functionality (see REFACTOR_PLAN.md)
2. **Sankey label overlap** in some visualizations
3. **No single source of truth** for core components

## The Labeling System

We use a **primary/secondary** labeling approach:
- **Primary labels** ensure consistency (same tokens → same primary label)
- **Secondary labels** provide distinction (e.g., "Content Words (Temporal)")
- Uses alphabetical ordering when clusters share >50% tokens

Example labels:
- "Content Words (Temporal)"
- "Function Words (Grammatical)" 
- "Punctuation (Sentence Boundaries)"

## Important Context

### Why This Matters
The user discovered that GPT-2 organizes tokens primarily by **grammatical function** rather than semantic meaning. This is a key finding for the paper.

### User Preferences
- Wants **single sources of truth** (no duplicate implementations)
- Prefers **direct LLM analysis** (where I analyze directly, not through API calls)
- Values **consistency** over variety in labeling
- Frustrated by reimplementations - wants clean architecture

### Recent Work
1. Created consistent labeling system after many iterations
2. Generated Sankey diagrams with colored archetypal paths
3. Fixed issue where 10 labels showed but only 7 clusters visible
4. Created comprehensive refactor plan and architecture diagram

## Next Steps

1. **Begin Refactoring**: Follow the phases in REFACTOR_PLAN.md
2. **Create Core Library**: Establish `concept_fragmentation/` with submodules
3. **Migrate Sankey**: Consolidate all Sankey implementations
4. **Archive Old Code**: Move deprecated files to archive
5. **Update Documentation**: Ensure all docs reflect new structure

## Command Reference

```bash
# Generate fixed Sankey diagrams
cd experiments/gpt2/all_tokens
python generate_fixed_sankeys_k10.py

# Run full k=10 analysis pipeline
python run_k10_analysis_pipeline.py

# Check clustering results
python analyze_results.py
```

## Architecture Overview

```
concept_fragmentation/
├── clustering/       # Single source for clustering algorithms
├── labeling/         # Single source for labeling systems  
├── visualization/    # Single source for all visualizations
├── experiments/      # Experiment management framework
└── persistence/      # State management

experiments/
├── configs/          # YAML configuration files
├── gpt2/            # GPT-2 specific experiments
└── shared/          # Shared utilities
```

## Remember
- Always check ARCHITECTURE.yaml at session start
- Use ConsistentLabeler for any labeling needs
- SankeyGenerator will be the single source for Sankey diagrams
- The refactor aims to eliminate ALL duplicate implementations