# Scripts Directory

This directory contains utility scripts for running analyses, generating visualizations, and maintaining the repository.

## Structure

- `analysis/` - Scripts for running various analyses
  - `run_analysis.py` - Main analysis runner
  - `run_cluster_paths.py` - Cluster path analysis
  - `llm_analysis_example.py` - LLM analysis example
  - `llm_path_analysis.py` - LLM-powered path analysis

- `visualization/` - Scripts for generating figures and visualizations
  - `run_visualizations.py` - Main visualization runner
  - `generate_paper_figures.py` - Generate figures for papers
  - `generate_labeled_paths_figure.py` - Generate labeled path diagrams
  - `integrate_figures.py` - Integrate figures into documents

- `utilities/` - General utility scripts
  - `check_activations.py` - Check activation data integrity
  - `debug_tokenization.py` - Debug tokenization issues
  - `verify_tokenization.py` - Verify tokenization correctness
  - `fix_paths.py` - Fix file paths
  - `enable_dimension_checks.py` - Enable dimension checking
  - `enable_logging.py` - Enable detailed logging
  - `refresh_dashboard.py` - Refresh dashboard data

- `maintenance/` - Repository maintenance scripts
  - `clean-and-run-analysis.ps1` - Clean and run analysis pipeline
  - `housekeeping.ps1` - General housekeeping tasks
  - `safe_cleanup.ps1` - Safe cleanup of temporary files
  - `run_full_pipeline.ps1` - Run complete analysis pipeline
  - `generate_critical_metrics.ps1` - Generate critical metrics

- `testing/` - Test scripts (temporarily in root, will be moved here)

## Usage

Most scripts can be run from the root directory:

```bash
python scripts/analysis/run_analysis.py
```

PowerShell scripts should be run from PowerShell:

```powershell
.\scripts\maintenance\run_full_pipeline.ps1
```