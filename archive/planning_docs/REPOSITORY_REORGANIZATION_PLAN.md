# Repository Reorganization Plan

## Current Issues
- 40+ Python files cluttering the root directory
- Mixed experiment files, utilities, and test scripts
- GPT-2 specific files scattered throughout
- No clear separation between experiments, tools, and results
- Difficult to navigate and find relevant code

## Critical Dependencies to Preserve
1. **Dashboard**: `visualization/run_dashboard.py` is called by `run_dashboard.bat`
2. **ArXiv Paper**: References case study sections but no direct code dependencies
3. **Core Package**: `concept_fragmentation/` imports must remain intact
4. **Analysis Scripts**: Used by various workflows and experiments

## Proposed New Structure

```
ConceptualFragmentationInLLMsAnalysisAndVisualization/
├── experiments/              # All experiment-specific code
│   ├── gpt2/                # GPT-2 specific experiments
│   │   ├── pivot/           # Pivot experiment files
│   │   │   ├── gpt2_pivot_sentences.py
│   │   │   ├── gpt2_pivot_clusterer.py
│   │   │   ├── gpt2_pivot_llm_analysis_data.py
│   │   │   ├── data/       # Pivot data files
│   │   │   │   ├── gpt2_pivot_*.txt
│   │   │   │   ├── gpt2_pivot_*.json
│   │   │   │   └── gpt2_pivot_activations_metadata.json
│   │   │   └── results/
│   │   ├── pos/             # Part-of-speech experiment
│   │   │   ├── gpt2_pos_experiment.py
│   │   │   ├── data/       # POS data files
│   │   │   │   ├── gpt2_pos_*.txt
│   │   │   │   ├── gpt2_pos_*.json
│   │   │   │   └── gpt2_pos_activations_metadata.json
│   │   │   └── results/
│   │   ├── semantic_subtypes/  # Semantic subtypes experiment
│   │   │   ├── gpt2_semantic_subtypes_experiment.py
│   │   │   ├── gpt2_semantic_subtypes_curator.py
│   │   │   ├── gpt2_semantic_subtypes_statistics.py
│   │   │   ├── gpt2_semantic_subtypes_wordlists.py
│   │   │   ├── data/        # Curated word lists
│   │   │   │   ├── gpt2_semantic_subtypes_curated.json
│   │   │   │   ├── gpt2_semantic_subtypes_statistics.json
│   │   │   │   ├── gpt2_semantic_subtypes_*.txt
│   │   │   │   └── activations/
│   │   │   └── results/
│   │   └── shared/          # Shared GPT-2 utilities
│   │       ├── gpt2_activation_extractor.py
│   │       ├── gpt2_apa_metrics.py
│   │       ├── gpt2_clustering_comparison.py
│   │       └── gpt2_token_validator.py
│   ├── heart_disease/       # Heart disease experiment
│   │   ├── generate_heart_metrics_charts.py
│   │   ├── data/
│   │   │   └── analysis_results_heart_seed0.json
│   │   └── results/
│   └── titanic/             # Titanic experiment
│       ├── data/
│       │   └── analysis_results_titanic_seed0.json
│       └── results/
│
├── scripts/                 # Utility scripts and tools
│   ├── analysis/           # Analysis runners
│   │   ├── run_analysis.py
│   │   ├── run_cluster_paths.py
│   │   ├── llm_analysis_example.py
│   │   └── llm_path_analysis.py
│   ├── visualization/      # Visualization generators
│   │   ├── run_visualizations.py
│   │   ├── generate_paper_figures.py
│   │   ├── generate_labeled_paths_figure.py
│   │   └── integrate_figures.py
│   ├── utilities/          # General utilities
│   │   ├── check_activations.py
│   │   ├── debug_tokenization.py
│   │   ├── verify_tokenization.py
│   │   ├── create_layer4.py
│   │   ├── fix_paths.py
│   │   ├── enable_dimension_checks.py
│   │   ├── enable_logging.py
│   │   └── refresh_dashboard.py
│   ├── maintenance/        # Maintenance scripts
│   │   ├── clean-and-run-analysis.ps1
│   │   ├── housekeeping.ps1
│   │   ├── safe_cleanup.ps1
│   │   ├── run_heart_analysis.ps1
│   │   ├── run_full_pipeline.ps1
│   │   └── generate_critical_metrics.ps1
│   └── testing/            # Test scripts (stay in root for now)
│
├── sample_data/            # Example analysis results
│   ├── sample_analysis_results.json
│   └── mock_llm_analysis_results.json
│
├── concept_fragmentation/  # Core package (keep as is)
├── visualization/          # Visualization code (keep as is)
├── arxiv_submission/       # Paper materials (keep as is)
├── docs/                   # Documentation (keep as is)
├── tools/                  # Build tools (keep as is)
├── examples/               # Example scripts (keep as is)
├── tests/                  # Main test suite (keep as is)
├── logs/                   # Log files
├── cache/                  # Cache directory
├── data/                   # General data directory (keep existing)
├── results/                # General results directory (keep existing)
├── figures/                # Generated figures (keep existing)
├── venv311/               # Virtual environment
│
├── config.py              # Main configuration (stays in root)
├── requirements.txt       # Dependencies (stays in root)
├── README.md              # Main readme (stays in root)
├── README_LLM_TESTING.md  # LLM testing readme (stays in root)
├── paper.md               # Paper markdown (stays in root)
├── references.bib         # Bibliography (stays in root)
├── run_dashboard.bat      # Dashboard launcher (stays in root)
├── run_dashboard_with_paths.bat  # Alt launcher (stays in root)
├── test_*.py              # Test files (temporarily stay in root)
└── .gitignore            # Git ignore (stays in root)
```

## Migration Steps

### Phase 1: Create Directory Structure (Non-Breaking)
```bash
# Create new directories
mkdir -p experiments/gpt2/{pivot,pos,semantic_subtypes,shared}/{data,results}
mkdir -p experiments/{heart_disease,titanic}/{data,results}
mkdir -p scripts/{analysis,visualization,utilities,maintenance}
mkdir -p sample_data
```

### Phase 2: Copy Files First (Test Safety)
Instead of moving files immediately, copy them first to test:
```bash
# Copy GPT-2 files
cp gpt2_pivot_*.py experiments/gpt2/pivot/
cp gpt2_pos_*.py experiments/gpt2/pos/
cp gpt2_semantic_subtypes_*.py experiments/gpt2/semantic_subtypes/
cp gpt2_{activation_extractor,apa_metrics,clustering_comparison,token_validator}.py experiments/gpt2/shared/
```

### Phase 3: Update Import Paths
1. Create `__init__.py` files with backward compatibility imports
2. Add the experiments directory to Python path in key scripts
3. Update imports gradually with fallback imports

### Phase 4: Move Data Files
```bash
# Move data files after code is working
mv gpt2_pivot_*.{txt,json} experiments/gpt2/pivot/data/
mv gpt2_pos_*.{txt,json} experiments/gpt2/pos/data/
mv gpt2_semantic_subtypes_*.{json,txt} experiments/gpt2/semantic_subtypes/data/
mv analysis_results_heart_*.json experiments/heart_disease/data/
mv analysis_results_titanic_*.json experiments/titanic/data/
```

### Phase 5: Clean Up Root (After Verification)
Only after everything is working:
1. Remove duplicate files from root
2. Update documentation
3. Commit with clear message

## Safety Measures

### Import Compatibility Layer
Create `experiments/__init__.py`:
```python
# Backward compatibility imports
import sys
from pathlib import Path

# Add experiment directories to path
sys.path.insert(0, str(Path(__file__).parent / "gpt2" / "shared"))
sys.path.insert(0, str(Path(__file__).parent / "gpt2" / "pivot"))
sys.path.insert(0, str(Path(__file__).parent / "gpt2" / "pos"))
sys.path.insert(0, str(Path(__file__).parent / "gpt2" / "semantic_subtypes"))
```

### Testing Checklist
Before removing any files from root:
- [ ] Dashboard still launches correctly
- [ ] All imports in concept_fragmentation/ work
- [ ] GPT-2 semantic subtypes experiment runs
- [ ] Analysis scripts find data files
- [ ] ArXiv paper compiles
- [ ] Unit tests pass

## Benefits
- Clear separation of concerns
- Easy to find experiment-specific code
- Reusable components in shared directories
- Cleaner root directory (but not empty)
- Better organization for future experiments
- Maintains backward compatibility

## What Stays in Root
- Configuration files (config.py, requirements.txt)
- Documentation files (README.md, paper.md)
- Launch scripts (run_dashboard.bat)
- Test files (temporarily, can be moved later)
- Critical workflow scripts

## Priority Order
1. **Phase 1-2**: Create structure and copy files (no breaking changes)
2. **Phase 3**: Test everything works with copies
3. **Phase 4**: Move data files (low risk)
4. **Phase 5**: Clean up only after full verification

## Notes
- Use `git mv` to preserve history when moving files
- Keep copies until verified working
- Add clear README.md in each new directory
- Document any path changes needed
- Consider creating symlinks for backward compatibility if needed