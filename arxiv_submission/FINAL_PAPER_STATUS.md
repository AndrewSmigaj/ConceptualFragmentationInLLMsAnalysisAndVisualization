# Final Paper Status Summary

## Completed Tasks

### 1. Fixed Stepped Trajectory Visualizations ✓
- **Issue**: Heart disease visualization was using synthetic random data instead of actual cluster paths
- **Solution**: Created `generate_heart_trajectory_from_paths.py` that loads actual data from `heart_seed_0_paths_with_centroids.json`
- **Result**: Heart disease visualization now matches the original dash app appearance

### 2. Switched to UMAP for All Visualizations ✓
- Changed from PCA to UMAP for dimensionality reduction
- Applied consistently across both GPT-2 and heart disease models
- UMAP parameters: n_neighbors=15, min_dist=0.1, n_components=3

### 3. Updated All Figures with Latest Data ✓
- GPT-2: Using full 1,228 words dataset
- Heart disease: Using actual patient data (303 patients)
- All figures regenerated with proper settings

### 4. Added Proper Color Coding ✓
- GPT-2: Color by grammatical type (noun/verb/adjective/adverb/other)
- Heart disease: Color by outcome (disease/no disease)
- Increased line thickness to 3 and opacity to 0.7 for visibility

### 5. Generated Sankey Diagrams with Labels ✓
- Structured path style with inline cluster labels
- Top 7 archetypal paths shown for both models
- Fixed cutoff issues (height=1300, margin bottom=300)
- Removed confusing background grids

### 6. Updated Paper Narrative ✓
- Changed from "purely grammatical" to "grammatical-dominant organization"
- Added nuance about semantic micro-organization within grammatical highways
- Key insight: "While grammatical function becomes the primary organizing principle, the existence of multiple paths (5 in late layers) and the fact that only 48.5% of words follow the dominant highway indicates that semantic distinctions persist within the grammatical framework."

## Figure Inventory

### GPT-2 Figures (6 total)
1. `gpt2_sankey_early.png` - Early window (L0-L3) Sankey diagram
2. `gpt2_sankey_middle.png` - Middle window (L4-L7) Sankey diagram  
3. `gpt2_sankey_late.png` - Late window (L8-L11) Sankey diagram
4. `gpt2_stepped_layer_early.png` - Early window 3D trajectories
5. `gpt2_stepped_layer_middle.png` - Middle window 3D trajectories
6. `gpt2_stepped_layer_late.png` - Late window 3D trajectories

### Heart Disease Figures (2 total)
1. `heart_sankey.png` - Structured path Sankey with 7 archetypes
2. `heart_stepped_layer_trajectories.png` - 3D trajectory visualization with actual data

### Mathematical Foundation Figures (4 total)
1. `cluster_entropy.png`
2. `intra_class_distance.png`
3. `subspace_angle.png`
4. `optimal_clusters.png`

## Pending Task
- LaTeX installation in progress for PDF generation
- Once complete, run: `cd arxiv_submission && pdflatex main.tex`

## Key Files Created/Modified
- `/experiments/heart_disease/generate_heart_trajectory_from_paths.py` - NEW
- `/arxiv_submission/sections/gpt2_semantic_subtypes_case_study.tex` - UPDATED
- All figure generation scripts updated to use UMAP and proper colors
- Multiple tracking documents created for reference

The paper is ready for PDF generation once LaTeX installation completes.