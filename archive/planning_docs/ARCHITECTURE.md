# Concept Fragmentation Architecture

**Last Updated**: 2025-05-30  
**Purpose**: Single source of truth for project organization. Review this at the start of every session.

## 🔧 Environment Setup
**IMPORTANT**: This project uses a Python virtual environment located at `venv311/`
- Always activate the venv before running code: `source venv311/Scripts/activate` (Windows) or `source venv311/bin/activate` (Linux/Mac)
- Python version: 3.11
- All dependencies are installed in this venv

## 🎯 Core Principles
1. **One implementation per feature** - No duplicates
2. **Clear ownership** - Every file has a clear purpose
3. **Hierarchical organization** - Components build on each other
4. **Self-documenting structure** - Directory names explain purpose

## 📁 Directory Structure

```
ConceptualFragmentationInLLMsAnalysisAndVisualization/
│
├── 📦 concept_fragmentation/          # CORE LIBRARY (New Architecture)
│   ├── __init__.py
│   ├── activation/                    # Activation extraction & processing
│   │   ├── collector.py              # Collects activations from models
│   │   ├── processor.py              # Processes raw activations
│   │   └── storage.py                # Manages activation storage
│   │
│   ├── analysis/                      # Analysis algorithms
│   │   ├── cross_layer_metrics.py    # Cross-layer fragmentation metrics
│   │   ├── similarity_metrics.py      # Similarity calculations
│   │   ├── token_analysis.py         # Token-level analysis
│   │   └── transformer_metrics.py    # Transformer-specific metrics
│   │
│   ├── clustering/                    # Clustering algorithms (Phase 1 ✓)
│   │   ├── base.py                   # BaseClusterer abstract class
│   │   ├── paths.py                  # PathExtractor implementation
│   │   └── exceptions.py             # Clustering exceptions
│   │
│   ├── labeling/                      # Labeling strategies (Phase 1 ✓)
│   │   ├── base.py                   # BaseLabeler abstract class
│   │   └── exceptions.py             # Labeling exceptions
│   │
│   ├── visualization/                 # Visualization components (Phase 2-3 ✓)
│   │   ├── base.py                   # BaseVisualizer abstract class
│   │   ├── configs.py                # Configuration dataclasses
│   │   ├── sankey.py                 # Unified SankeyGenerator (Phase 2 ✓)
│   │   ├── trajectory.py             # Unified TrajectoryVisualizer (Phase 3 ✓)
│   │   └── exceptions.py             # Visualization exceptions
│   │
│   ├── experiments/                   # Experiment framework (Phase 1 ✓)
│   │   ├── base.py                   # BaseExperiment abstract class
│   │   └── config.py                 # ExperimentConfig management
│   │
│   ├── persistence/                   # State management (Phase 1 ✓)
│   │   ├── state.py                  # ExperimentState checkpointing
│   │   └── gpt2_persistence.py       # GPT-2 specific persistence
│   │
│   ├── llm/                          # LLM integration
│   │   ├── client.py                 # Base LLM client
│   │   ├── factory.py                # LLM provider factory
│   │   └── analysis.py               # LLM-powered analysis
│   │
│   └── utils/                        # Utilities (Phase 1 ✓)
│       ├── logging.py                # Logging configuration
│       ├── validation.py             # Data validation
│       └── path_utils.py             # Path manipulation
│
├── 🌐 visualization/                  # INTERACTIVE DASHBOARD
│   ├── dash_app.py                   # Main dashboard application
│   ├── run_dashboard.py              # Dashboard launcher
│   ├── data_interface.py             # Data loading interface
│   │
│   ├── 📊 Dashboard Tabs:
│   ├── llm_tab.py                    # LLM analysis integration
│   ├── gpt2_token_tab.py             # GPT-2 token analysis
│   ├── path_metrics_tab.py           # Path metrics visualization
│   ├── similarity_network_tab.py     # Network visualizations
│   ├── path_fragmentation_tab.py     # Fragmentation analysis
│   ├── cross_layer_viz.py            # Cross-layer visualizations
│   └── traj_plot.py                  # Trajectory plotting
│
├── 🧪 experiments/                    # EXPERIMENT IMPLEMENTATIONS
│   ├── gpt2/
│   │   ├── semantic_subtypes/        # Main GPT-2 case study
│   │   │   ├── 🎯 gpt2_semantic_subtypes_experiment.py  # MAIN ENTRY POINT
│   │   │   ├── data/                 # Curated word lists (1,228 words)
│   │   │   ├── *.json               # Configuration files
│   │   │   └── results/              # Experiment outputs
│   │   │
│   │   ├── shared/                   # Shared GPT-2 utilities
│   │   │   ├── gpt2_activation_extractor.py
│   │   │   ├── gpt2_apa_metrics.py
│   │   │   └── gpt2_clustering_comparison.py
│   │   │
│   │   └── all_tokens/              # ⚠️ CLEANUP CANDIDATE (2.5GB)
│   │
│   ├── heart_disease/                # Heart disease case study
│   └── titanic/                      # Titanic case study
│
├── 📄 arxiv_submission/               # PAPER & FIGURES
│   ├── main.tex                      # Paper source
│   ├── sections/                     # Paper sections
│   ├── figures/                      # Generated figures
│   └── references.bib                # Bibliography
│
├── 📚 docs/                          # DOCUMENTATION
│   ├── gpt2_analysis_guide.md        # GPT-2 analysis guide
│   ├── llm_integration_guide.md      # LLM integration guide
│   └── visualization_plan.md         # Visualization roadmap
│
├── 🧰 scripts/                       # UTILITY SCRIPTS
│   ├── analysis/                     # Analysis runners
│   ├── maintenance/                  # Cleanup & maintenance
│   └── utilities/                    # Helper scripts
│       └── migrate_sankey_usage.py   # Migration tool (Phase 2)
│
├── 🗃️ archive/                       # ARCHIVED CODE
│   └── (old implementations)         # Moved here during cleanup
│
├── 📂 data/                          # CENTRALIZED DATA (New)
│   ├── raw/                          # Original datasets
│   ├── processed/                    # Preprocessed data
│   └── activations/                  # Neural network activations
│
├── 📊 results/                       # EXPERIMENT RESULTS (New)
│   ├── gpt2/                         # GPT-2 experiments
│   ├── heart_disease/                # Medical AI results
│   └── titanic/                      # Classic ML results
│
├── ⚙️ configs/                       # CONFIGURATION FILES (New)
│   ├── experiments/                  # Experiment configs
│   ├── models/                       # Model configs
│   └── visualization/                # Viz settings
│
├── 🧪 tests/                         # CONSOLIDATED TESTS (New)
│   ├── unit/                         # Unit tests
│   ├── integration/                  # Integration tests
│   └── legacy/                       # Old test files
│
└── 📦 Key Entry Points:
    ├── 🚀 python visualization/run_dashboard.py          # Launch dashboard
    ├── 🧪 python experiments/gpt2/semantic_subtypes/gpt2_semantic_subtypes_experiment.py
    └── 📊 python -m concept_fragmentation.analysis.cross_layer_metrics
```

## 🔄 Refactoring Status

### ✅ Completed Phases
1. **Phase 1**: Core directory structure with base classes
2. **Phase 2**: Unified SankeyGenerator consolidation
3. **Phase 3**: Unified TrajectoryVisualizer (completed)
4. **Phase 4**: Repository reorganization (in progress)
   - Created centralized data/, results/, configs/, tests/ directories
   - Updated documentation for CTA focus
   - Structure designed to support future bigram experiments

### 🚧 Upcoming Phases
5. **Phase 5**: Consolidate GPT-2 analysis scripts
6. **Phase 6**: Unify dashboard components
7. **Phase 7**: Clean up experiments/gpt2/all_tokens/
8. **Phase 8**: Implement bigram experiment framework

## 📋 File Discipline Rules

### ✅ DO
- Check this architecture before creating ANY new file
- Use existing base classes and utilities
- Put new visualizations in concept_fragmentation/visualization/
- Create tests for new components
- Update this document when adding major components

### ❌ DON'T
- Create "test_quick.py" or "analyze_fast.py" files
- Duplicate existing functionality
- Put visualization code in experiments/
- Create multiple versions (basic_, improved_, enhanced_)
- Leave temporary files in the repo

## 🎯 When You Need To...

### Create a New Visualization
1. Check if concept_fragmentation/visualization/ has something similar
2. Extend BaseVisualizer
3. Add configuration to configs.py
4. Create tests in visualization/tests/

### Add New Analysis
1. Check concept_fragmentation/analysis/ first
2. Consider if it belongs in an existing module
3. Follow the pattern of existing analyzers

### Run an Experiment
1. Use concept_fragmentation/experiments/base.py
2. Store configs in experiments/[dataset]/config/
3. Save results in experiments/[dataset]/results/

### Add LLM Features
1. Use concept_fragmentation/llm/factory.py
2. Don't create new API clients
3. Follow existing patterns in llm/analysis.py

## 🚨 Cleanup Needed

### High Priority (Blocking Progress)
- [ ] experiments/gpt2/all_tokens/ - 2.5GB of duplicates
- [ ] Multiple sankey implementations → Use unified SankeyGenerator
- [ ] Duplicate label creation scripts

### Medium Priority (Confusing)
- [ ] Multiple dashboard utility versions
- [ ] Test scripts scattered everywhere
- [ ] Old visualization implementations

### Low Priority (Nice to Have)
- [ ] Compress old results
- [ ] Archive completed experiments
- [ ] Clean up logs

## 📝 Session Checklist

At the start of each session:
1. [ ] Review this ARCHITECTURE.md
2. [ ] Check REFACTOR_LOG.md for recent changes
3. [ ] Run key entry points to ensure they work
4. [ ] Update todo list based on priorities

## 🔍 Quick Reference

**Need to visualize trajectories?**
→ `from concept_fragmentation.visualization.trajectory import TrajectoryVisualizer`

**Need to create Sankey diagrams?**
→ `from concept_fragmentation.visualization.sankey import SankeyGenerator`

**Need to analyze cross-layer metrics?**
→ `from concept_fragmentation.analysis.cross_layer_metrics import ...`

**Need to run GPT-2 experiment?**
→ `python experiments/gpt2/semantic_subtypes/gpt2_semantic_subtypes_experiment.py`

**Need to launch dashboard?**
→ `python visualization/run_dashboard.py`

---

**Remember**: Every file should have ONE clear purpose and ONE clear location. When in doubt, ask: "Where would I look for this in 6 months?"