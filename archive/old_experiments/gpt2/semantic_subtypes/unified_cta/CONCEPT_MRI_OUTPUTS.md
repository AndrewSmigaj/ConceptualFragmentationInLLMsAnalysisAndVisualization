# GPT-2 Concept MRI Visualization Outputs

This directory contains the complete "Concept MRI" visualization suite for the GPT-2 semantic subtypes experiment.

## Main Dashboard
- **File**: `concept_mri_dashboard.html`
- **Description**: Unified dashboard with all visualizations and insights
- **Features**:
  - Overview with key metrics
  - Interactive Sankey diagrams for all windows
  - Complete path analysis tables
  - Key insights and interpretations
  - Word search interface (placeholder)

## Individual Components

### 1. Data Files
- `concept_mri_data.json` - Enhanced path data with grammatical annotations

### 2. Sankey Diagrams
- `results/unified_cta_config/unified_cta_20250524_073316/sankey_early_enhanced.html` - Early window (L0-L3)
- `results/unified_cta_config/unified_cta_20250524_073316/sankey_middle_enhanced.html` - Middle window (L4-L7)
- `results/unified_cta_config/unified_cta_20250524_073316/sankey_late_enhanced.html` - Late window (L8-L11)
- `results/unified_cta_config/unified_cta_20250524_073316/sankey_all_windows.html` - All windows combined

### 3. Path Analysis Tables
- `results/unified_cta_config/unified_cta_20250524_073316/path_analysis_tables.html` - Detailed path listings

### 4. Analysis Report
- `gpt2_semantic_analysis_report.md` - LLM interpretation of results

## Key Findings

1. **Grammatical Organization**: GPT-2 organizes by parts of speech, not semantic meaning
2. **Massive Convergence**: 72.8% of words converge to dominant noun pathway
3. **Path Reduction**: 19 paths → 5 paths → 4 paths across windows
4. **Adjective/Adverb Confusion**: Systematic mixing of these categories
5. **Verb Marginalization**: Verbs pushed to minor pathways (<1%)

## Usage

Open `concept_mri_dashboard.html` in a web browser for the full interactive experience.

## Generation Scripts

If you need to regenerate any components:
1. `prepare_mri_data.py` - Prepare enhanced data
2. `generate_mri_sankeys.py` - Generate Sankey diagrams
3. `generate_path_tables.py` - Generate path analysis tables