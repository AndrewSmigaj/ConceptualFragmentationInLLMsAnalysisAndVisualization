# Instructions to Generate Updated Figures

The scripts have been updated but need to be run in an environment with the required Python packages installed.

## Required Packages
- numpy
- plotly
- kaleido (for static image export)
- pandas
- scikit-learn
- umap-learn

## Scripts to Run

### 1. Heart Disease Figures

```bash
# Generate new heart sankey in GPT-2 style
cd experiments/heart_disease
python generate_heart_sankey_gpt2_style.py

# Generate heart trajectory with all patient lines (blue/red)
python generate_heart_trajectory_from_paths.py
```

### 2. GPT-2 Figures

```bash
# Generate GPT-2 trajectories colored by POS
cd experiments/gpt2/semantic_subtypes
python generate_gpt2_trajectory_viz_umap.py
```

## What Changed

1. **Heart Sankey**: Now uses inline cluster labels (GPT-2 style) instead of annotations
2. **Heart Trajectories**: Shows ALL 303 patient trajectories (blue for no disease, red for disease) plus 5 archetypal paths
3. **GPT-2 Trajectories**: Now colored by Part of Speech (green=nouns, red=verbs, blue=adjectives, yellow=adverbs) instead of by path

## Output Files
The scripts will generate new PNG files in `/arxiv_submission/figures/`:
- `heart_sankey.png` (updated)
- `heart_stepped_layer_trajectories.png` (updated)
- `gpt2_stepped_layer_early.png` (updated)
- `gpt2_stepped_layer_middle.png` (updated)
- `gpt2_stepped_layer_late.png` (updated)

After running these scripts, recompile the LaTeX to see the updated figures in the PDF.