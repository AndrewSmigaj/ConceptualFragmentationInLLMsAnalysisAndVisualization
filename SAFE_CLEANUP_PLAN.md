# Safe Cleanup Plan for Concept Fragmentation Repository

## Overview
This cleanup plan ensures we preserve all functionality needed for:
1. Reproducing the arxiv paper experiments (GPT-2 case study)
2. Running the interactive dashboard
3. Continuing development on the refactored architecture

## Essential Components to KEEP

### 1. **Core Infrastructure**
- `concept_fragmentation/` - The new refactored architecture (KEEP ALL)
- `arxiv_submission/` - Paper and figures (KEEP ALL)
- `visualization/dash_app.py` and related dashboard files (KEEP)
- `experiments/gpt2/semantic_subtypes/` - Main GPT-2 experiment (KEEP MOST)
- `experiments/gpt2/shared/` - Shared utilities (KEEP ALL)

### 2. **Essential Scripts for Paper Reproduction**
#### Main Entry Points:
- `experiments/gpt2/semantic_subtypes/gpt2_semantic_subtypes_experiment.py`
- `experiments/gpt2/semantic_subtypes/run_expanded_unified_cta.py`

#### Core Data:
- `experiments/gpt2/semantic_subtypes/data/` - All curated word lists
- `experiments/gpt2/semantic_subtypes/optimal_clusters_per_layer.json`
- `experiments/gpt2/semantic_subtypes/layer_clustering_config.json`

#### Essential Analysis:
- `experiments/gpt2/shared/gpt2_activation_extractor.py`
- `experiments/gpt2/shared/gpt2_apa_metrics.py`
- `experiments/gpt2/shared/gpt2_clustering_comparison.py`

### 3. **Dashboard Components**
- `visualization/dash_app.py` - Main dashboard
- `visualization/run_dashboard.py` - Dashboard runner
- `visualization/run_dashboard.bat` - Windows launcher
- All tab files: `llm_tab.py`, `gpt2_token_tab.py`, `path_metrics_tab.py`, etc.
- `visualization/data_interface.py` - Data loading

## Safe to Archive/Remove

### 1. **Duplicate Sankey Implementations** (Archive to `archive/old_sankey/`)
- `experiments/gpt2/all_tokens/generate_sankey_diagrams.py`
- `experiments/gpt2/all_tokens/generate_k10_sankeys.py`
- `experiments/gpt2/all_tokens/generate_k5_sankeys.py`
- `experiments/gpt2/all_tokens/generate_colored_sankeys_k10.py`
- `experiments/gpt2/all_tokens/generate_enhanced_sankeys_k10.py`
- `experiments/gpt2/all_tokens/generate_fixed_sankeys_k10.py`

### 2. **Duplicate Label Creation Scripts** (Archive to `archive/old_labels/`)
- `experiments/gpt2/all_tokens/create_*_labels_*.py` (14+ files)
- Keep only: `experiments/gpt2/all_tokens/llm_labels_k10/` (final results)

### 3. **One-off Test Scripts** (Can be removed)
- `test_5k_quick.py`
- `test_single_figure.py` 
- `quick_verify.py`
- `test_clustering_verification.py`

### 4. **Old Analysis Duplicates** (Archive to `archive/old_analysis/`)
In `experiments/gpt2/all_tokens/`:
- Multiple `analyze_*.py` scripts that do similar things
- Keep only the most comprehensive ones

### 5. **Temporary Outputs** (Can be removed after backing up results)
- `experiments/gpt2/all_tokens/activations/` (2.5GB) - IF we have the processed results
- Old HTML visualizations (keep only final paper figures)
- Log files older than 1 month

## Cautious Areas - Need Review

### 1. **experiments/gpt2/all_tokens/** 
This 2.5GB directory needs careful review:
- The activation chunks might be needed if we want to re-run analysis
- The clustering results are likely final and can be kept compressed
- Many scripts here are duplicates but some might be unique

### 2. **Multiple Dashboard Utilities**
- `basic_compile_dataset_info.py` vs `compile_dataset_info.py` vs `improved_compile_dataset_info.py`
- Need to identify which version the dashboard actually uses

### 3. **Visualization Scripts in Root visualization/**
- Many `gpt2_*.py` scripts that might be used by the dashboard
- Need to trace dashboard imports before removing

## Cleanup Steps

### Phase 1: Safe Archival (Low Risk)
1. Create archive directories:
   ```
   archive/old_sankey/
   archive/old_labels/
   archive/old_analysis/
   archive/old_test_outputs/
   ```

2. Move obvious duplicates to archive (keeping original paths in a mapping file)

3. Compress large data files that aren't actively used

### Phase 2: Test Everything (Before Deletion)
1. Run the GPT-2 experiment: `python experiments/gpt2/semantic_subtypes/gpt2_semantic_subtypes_experiment.py`
2. Launch the dashboard: `python visualization/run_dashboard.py`
3. Run the new unified visualizers to ensure they work
4. Check that paper figures can still be generated

### Phase 3: Final Cleanup (After Verification)
1. Remove archived files that have been untouched for 30+ days
2. Clean up temporary outputs and logs
3. Document what was removed in a cleanup log

## Critical Files - DO NOT REMOVE
- Any file referenced in `arxiv_submission/main.tex` or section files
- Any file imported by `visualization/dash_app.py`
- Any file in the new `concept_fragmentation/` architecture
- The curated word lists in `experiments/gpt2/semantic_subtypes/data/`
- Final clustering results referenced in the paper

## Estimated Space Savings
- Removing activation chunks (if backed up): ~2GB
- Archiving duplicate scripts: ~50MB
- Compressing old results: ~500MB
- **Total potential savings: ~2.5GB**

## Next Steps
1. Review this plan
2. Create archive directories
3. Start with Phase 1 (safe archival)
4. Test thoroughly before any deletion
5. Keep detailed log of what was moved/removed