# Repository Reorganization Plan

## Overview
This document outlines recommended improvements to the repository structure based on best practices and future needs (including bigram experiments).

## Current Structure Analysis

### ✅ What's Working Well
1. **Clear separation**: Library code (`concept_fragmentation/`) vs experiments
2. **Modular design**: Well-organized submodules with clear purposes
3. **Base abstractions**: Good use of inheritance and interfaces
4. **Phased refactoring**: Systematic improvement approach

### ❌ Issues to Address
1. **Test file scatter**: Test files at root level instead of in test directories
2. **Experiment bloat**: `experiments/gpt2/all_tokens/` is 2.5GB with many duplicates
3. **Results management**: Outputs mixed with code
4. **Configuration sprawl**: Hard-coded values in multiple places
5. **Outdated documentation**: concept_fragmentation/README.md still references old framing

## Proposed Reorganization

### 1. Directory Structure
```
ConceptualFragmentationInLLMsAnalysisAndVisualization/
│
├── concept_fragmentation/          # Core CTA library [KEEP AS IS]
│   ├── activation/                # Activation extraction
│   ├── analysis/                  # Analysis algorithms
│   │   ├── cluster_paths.py       # Core CTA
│   │   ├── bigram_analysis.py    # NEW: For bigram support
│   │   └── ...
│   ├── clustering/                # Clustering implementations
│   ├── visualization/             # Unified visualizers
│   ├── llm/                      # LLM integration
│   └── README.md                 # UPDATE: CTA-focused docs
│
├── experiments/                   # Experiment implementations
│   ├── gpt2/
│   │   ├── semantic_subtypes/    # Main paper experiment [KEEP]
│   │   ├── bigrams/             # NEW: Bigram experiments
│   │   │   ├── config/
│   │   │   ├── data/
│   │   │   └── scripts/
│   │   └── all_tokens/          # CLEANUP: Archive most
│   ├── heart_disease/           # Medical AI [KEEP]
│   └── titanic/                 # Classic ML [KEEP]
│
├── data/                        # NEW: Centralized data directory
│   ├── raw/                     # Original datasets
│   ├── processed/               # Preprocessed data
│   └── activations/             # Extracted activations
│
├── results/                     # NEW: Structured results
│   ├── gpt2/
│   │   ├── semantic_subtypes/
│   │   └── bigrams/
│   ├── heart_disease/
│   └── titanic/
│
├── configs/                     # NEW: Centralized configs
│   ├── experiments/
│   ├── models/
│   └── visualization/
│
├── tests/                       # CONSOLIDATE: All tests here
│   ├── unit/
│   ├── integration/
│   └── experiments/
│
├── scripts/                     # Utility scripts [KEEP]
├── visualization/               # Dashboard [KEEP]
├── arxiv_submission/           # Paper materials [KEEP]
├── archive/                    # Archived code [KEEP]
├── docs/                       # Documentation [KEEP]
└── venv311/                    # Virtual environment [KEEP]
```

### 2. Immediate Actions

#### Phase 1: Clean Test Files
```bash
# Move all root-level test files
mkdir -p tests/legacy
mv test_*.py tests/legacy/
mv test_*.html tests/legacy/
```

#### Phase 2: Create Data Structure
```bash
# Create centralized data directory
mkdir -p data/{raw,processed,activations}
mkdir -p results/{gpt2/{semantic_subtypes,bigrams},heart_disease,titanic}
mkdir -p configs/{experiments,models,visualization}
```

#### Phase 3: Update Documentation
- Update concept_fragmentation/README.md for CTA
- Create configs/README.md for configuration guide
- Update ARCHITECTURE.md with new structure

### 3. Configuration Management

Create YAML configs to replace hard-coded values:

```yaml
# configs/experiments/gpt2_bigrams.yaml
experiment:
  name: gpt2_bigrams
  model: gpt2
  data:
    type: bigrams
    source: data/raw/bigrams/
  clustering:
    method: kmeans
    k_range: [3, 5, 10]
  output:
    dir: results/gpt2/bigrams/
```

### 4. Bigram Experiment Setup

For the upcoming bigram experiments:

```
experiments/gpt2/bigrams/
├── config.yaml                    # Experiment configuration
├── extract_bigram_activations.py  # Activation extraction
├── analyze_bigram_paths.py        # CTA analysis
├── generate_bigram_sankeys.py     # Visualization
└── README.md                      # Experiment documentation
```

### 5. Best Practices Implementation

1. **Use `__main__.py`** for package execution:
   ```python
   # concept_fragmentation/__main__.py
   if __name__ == "__main__":
       from .cli import main
       main()
   ```

2. **Centralized logging**:
   ```python
   # concept_fragmentation/utils/logging.py
   import logging
   logger = logging.getLogger("cta")
   ```

3. **Data registry**:
   ```python
   # concept_fragmentation/data/registry.py
   class DataRegistry:
       """Central registry for all datasets and results"""
   ```

4. **Experiment tracking**:
   ```python
   # concept_fragmentation/experiments/tracker.py
   class ExperimentTracker:
       """Track experiments, configs, and results"""
   ```

## Migration Strategy

1. **Week 1**: Move test files, create directory structure
2. **Week 2**: Implement configuration system
3. **Week 3**: Update documentation
4. **Week 4**: Set up bigram experiment framework

## Benefits

1. **Cleaner structure**: Easier to navigate and understand
2. **Better separation**: Code, data, and results clearly separated
3. **Scalability**: Ready for bigram and future experiments
4. **Maintainability**: Centralized configuration and data management
5. **Reproducibility**: Clear experiment tracking and result storage

## Risks and Mitigation

1. **Breaking changes**: Use symlinks during transition
2. **Lost files**: Keep comprehensive archive mapping
3. **Import errors**: Update all imports systematically
4. **Documentation drift**: Update docs immediately after changes

## Important Notes

1. **Activation chunks**: The 1.8GB activation chunks in `experiments/gpt2/all_tokens/activations/` should be preserved but potentially moved to `data/activations/gpt2/all_tokens/`
2. **Virtual environment**: Keep `venv311/` at root as per Python best practices
3. **Git management**: Update `.gitignore` to exclude large data files in new directories