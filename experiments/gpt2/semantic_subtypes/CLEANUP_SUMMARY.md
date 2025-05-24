# Cleanup Summary

## Current Structure

### Main Directory (Core Files)
- `gpt2_semantic_subtypes_curator.py` - Data preparation
- `gpt2_semantic_subtypes_experiment.py` - Main experiment runner
- `gpt2_semantic_subtypes_statistics.py` - Statistics analysis
- `gpt2_semantic_subtypes_wordlists.py` - Word list definitions
- `activations_by_layer.pkl` - Preprocessed activations (774 x 768 x 13 layers)
- `word_metadata.pkl` - Word metadata and subtype mappings
- `layer_clustering_config.json` - Current clustering configuration

### Subdirectories
- `data/` - Original curated word lists and validation reports
- `analysis/` - Analysis scripts (cluster analysis, semantic insights)
- `visualization/` - Visualization scripts and outputs
- `utils/` - Utility scripts (ETS wrapper, threshold search, etc.)
- `tests/` - Test and diagnostic scripts
- `llm_analysis_data/` - Prepared data for LLM analysis
- `results/` - Empty, ready for new results
- `archive/` - Old experiment runs and intermediate files

### Archived
- 19 experiment run directories moved to `archive/experiment_runs/`
- Log files and intermediate JSONs moved to `archive/intermediate_files/`
- All PNG and HTML visualization outputs moved to `visualization/`

## Ready for Unified CTA Implementation
The workspace is now clean and organized for implementing the unified clustering and ETS strategy. All essential data files are preserved in the main directory, while experimental and intermediate files are safely archived.