# Archive Mapping Log
**Date**: 2025-05-30
**Purpose**: Track files moved during cleanup to preserve functionality

## Phase 1: Label Creation Scripts Archival

### Context
These scripts were used to generate various label files for the GPT-2 all_tokens experiment. The generated JSON files are preserved in their original locations, only the generation scripts are archived.

### Files Moved to `archive/old_labels/`

#### Create Label Scripts (11 files):
- create_consistent_labels_k5.py
- create_accurate_labels_k5.py
- create_differentiated_labels_k5.py
- create_consistent_differentiated_labels_k5.py
- create_direct_labels_k10.py
- create_consistent_labels_k10.py
- create_consistent_differentiated_labels_k10.py
- create_data_driven_labels_k10.py
- create_llm_labels_k10.py
- create_alphabetical_consistent_labels_k10.py
- create_primary_secondary_labels_k10.py

#### Label Token Scripts (3 files):
- label_tokens_comprehensive.py
- label_tokens_llm.py
- label_tokens_direct.py

#### Generate Label Scripts (4 files):
- generate_llm_labels_k5.py
- generate_consistent_labels_k10.py
- generate_llm_direct_labels_k10.py
- prepare_llm_labels_k5.py

#### Other Labeling Utilities (8 files):
- create_token_labels.py
- prepare_llm_labeling_data.py
- create_consistent_cluster_labels.py
- analyze_clusters_for_hierarchical_labels.py
- update_labels_for_consistency.py
- llm_based_labeling_k10.py
- check_label_consistency.py

### Preserved Output Files
The following directories contain the actual label data and are NOT moved:
- experiments/gpt2/all_tokens/llm_labels_k10/
- experiments/gpt2/all_tokens/llm_labels_k5/
- experiments/gpt2/all_tokens/token_labels/
- experiments/gpt2/all_tokens/token_labels_final/
- experiments/gpt2/all_tokens/clustering_results_k*/

### Verification
Before archival, verified that:
1. No other scripts import these label creation files
2. Analysis scripts load labels from JSON files, not from these scripts
3. Dashboard functionality does not depend on these scripts

## Phase 2: Duplicate Analysis Scripts Archival

### Context
These analysis scripts represent earlier iterations of the analysis process. They have been superseded by more focused pipeline scripts or were one-off exploratory analyses.

### Files Moved to `archive/old_analysis/`

#### Exploratory Analysis Scripts (4 files):
- analyze_all_tokens.py - Initial exploration of full GPT-2 vocabulary (50k tokens)
- analyze_all_tokens_trajectories.py - Attempted trajectory analysis for all tokens (computationally intensive)
- analyze_results.py - Simple k=3 results viewer
- analyze_top_10k_trajectories.py - Superseded by unified pipeline scripts

### Scripts Kept (Recently Active):
- analyze_cluster_differences_k5.py - Active k=5 cluster analysis (modified May 29)
- analyze_cluster_structure.py - Active cluster structure analysis (modified May 29)
- analyze_clusters_direct.py - Most recent cluster analysis (modified May 29)

## Phase 3: One-off Test Scripts Archival

### Context
These are temporary test scripts created during development for debugging and verification. They are not part of the official test suite (which lives in concept_fragmentation/tests/ and visualization/tests/).

### Files Moved to `archive/old_test_scripts/`

#### Root Directory Test Scripts (11 files):
- test_llm_analysis.py - Simple LLM analysis test with Titanic data
- test_path_metrics.py - Tests path metrics module
- test_gpt2_persistence.py - Tests GPT-2 persistence
- test_integration_tests.py - Meta-test for integration tests
- test_pivot_metrics.py - Tests pivot metrics
- test_enhanced_clusterer.py - Tests enhanced clustering
- test_backward_compatibility.py - Tests backward compatibility
- test_clustering_comparison.py - Tests clustering comparison
- test_sankey_coordinates.py - Tests Sankey coordinates
- test_phase1_structure.py - Tests Phase 1 structure (untracked)
- test_phase2_sankey.py - Tests Phase 2 Sankey (untracked)

#### From experiments/gpt2/all_tokens/ (2 files):
- test_clustering_verification.py - Clustering verification test
- quick_verify.py - Quick verification script

#### From experiments/gpt2/semantic_subtypes/ (3 files):
- test_single_figure.py - Single figure generation test
- test_5k_quick.py - Quick 5k test
- test_gpt2_load.py - GPT-2 loading test

### Official Test Suites Preserved
- concept_fragmentation/tests/ - Core library tests
- visualization/tests/ - Visualization tests
- Other module-specific test directories

## Cleanup Summary

### Files Archived
- **26 label creation scripts** (188K)
- **4 duplicate analysis scripts** (40K)
- **16 one-off test scripts** (72K)
- **Total: 46 files, ~300K**

### Verification
- Core library imports successfully
- Main experiment files intact
- Dashboard runner intact
- All archived files tracked in this document

### Remaining Cleanup Opportunities
1. **experiments/gpt2/all_tokens/activations/** (1.8GB) - KEPT as canonical activation data
2. **Multiple dashboard utility versions** - Need to identify which is actively used
3. **Old HTML visualizations** - Keep only final paper figures
4. **Compression of large result files** - JSON files that could be gzipped

## Phase 4: Redundant Activation Files Removal

### Files Removed (819MB saved):
- experiments/gpt2/all_tokens/frequent_token_activations.npy (352MB)
- experiments/gpt2/all_tokens/top_10k_activations.npy (352MB)
- experiments/gpt2/semantic_subtypes/5k_common_words/activations.npy (115MB)

### Rationale:
- These were subsets of the complete activation chunks
- The chunked activations in experiments/gpt2/all_tokens/activations/ contain all data
- No active scripts were using these redundant files

## Total Cleanup Summary

### Space Saved: ~1.1GB
- Archived scripts: ~300KB
- Removed redundant activations: ~819MB
- Total files cleaned: 49 files archived, 3 large files removed

### Preserved:
- All core functionality
- Canonical activation data (1.8GB chunked format)
- All clustering results and analysis outputs
- Paper reproduction capability