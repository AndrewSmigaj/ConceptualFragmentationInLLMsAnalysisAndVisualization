# Semantic Subtypes Cleanup Plan

## 1. Archive Old Experiment Runs
Move to `archive/experiment_runs/`:
- semantic_subtypes_experiment_20250523_*
- semantic_subtypes_optimal_ets_20250523_*
- semantic_subtypes_optimal_experiment_20250523_*
- semantic_subtypes_revised_ets_20250523_*

## 2. Consolidate Test Scripts
Move to `tests/`:
- test_*.py files
- check_*.py files (diagnostic scripts)

## 3. Remove Intermediate Files
Delete or archive:
- *.pkl files (except essential ones like activations_by_layer.pkl)
- *.png files (visualization outputs)
- *.html files (interactive plots)
- *.log files
- *.json files (except essential configs)

## 4. Organize Core Scripts
Keep in main directory:
- gpt2_semantic_subtypes_curator.py (data preparation)
- gpt2_semantic_subtypes_experiment.py (main experiment)
- gpt2_semantic_subtypes_statistics.py (analysis)
- gpt2_semantic_subtypes_wordlists.py (word lists)

Move to subdirectories:
- Analysis scripts -> analysis/
- Visualization scripts -> visualization/
- Utility scripts -> utils/

## Essential Files to Keep
- data/ directory (curated word lists)
- activations_by_layer.pkl (reconstructed activations)
- word_metadata.pkl (word metadata)
- layer_clustering_config.json (current config)