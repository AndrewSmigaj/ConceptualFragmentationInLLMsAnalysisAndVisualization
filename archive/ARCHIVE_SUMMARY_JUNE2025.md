# Archive Summary - June 14, 2025

## What Was Archived

This archive contains old code and files that are no longer actively used in the project. The repository has been cleaned to focus on:
1. The arxiv papers (arxiv_apple/, arxiv_submission/, arxiv_next/)
2. The Concept MRI dashboard (concept_mri/)
3. Core library code (concept_fragmentation/)

### Archived Directories

#### old_visualization/
- The old dash-based visualization system
- Replaced by Concept MRI dashboard
- Includes all old visualization scripts and the visualization/ directory

#### old_experiments/
- experiments/gpt2/ - Large GPT-2 experiments (2.3GB - NOT MOVED due to size, needs manual archiving)

#### old_scripts/
- scripts/analysis/ - Old analysis scripts
- scripts/maintenance/ - Maintenance and cleanup scripts
- scripts/testing/ - Test scripts
- scripts/utilities/ - Utility scripts
- scripts/visualization/ - Visualization generation scripts
- tests/ - Old test directory
- Root level test_*.py files

#### old_results/
- All old results except apple_variety_test/
- Includes baselines/, cluster_paths/, figures/, gpt2/, heart_disease/, llm/, titanic/, etc.
- sankey_output/

#### old_docs/
- docs/ directory with old documentation

#### misc/
- __pycache__/
- cache/, logs/, tools/, configs/, hooks/, examples/, sample_data/
- Various root level files: debug outputs, references.bib, etc.

### What Remains Active

- **arxiv_apple/** - Current apple CTA paper
- **arxiv_submission/** - Submitted CTA paper
- **arxiv_next/** - Planning for next paper
- **concept_mri/** - The Concept MRI dashboard tool
- **concept_fragmentation/** - Core library
- **experiments/apple_variety/** - Current apple experiments
- **scripts/prepare_demo_models/** - Demo model training for Concept MRI
- **data/** - Core datasets
- **results/apple_variety_test/** - Current apple experiment results
- **venv311/** - Python virtual environment
- Core files: README.md, CLAUDE.md, CURRENTLY_WORKING_ON.md, ARCHITECTURE.yaml, local_config.py

### Notes

1. The experiments/gpt2/ directory is very large and may need special handling for archiving
2. All archived code is preserved for future reference
3. The repository is now much cleaner and focused on current work