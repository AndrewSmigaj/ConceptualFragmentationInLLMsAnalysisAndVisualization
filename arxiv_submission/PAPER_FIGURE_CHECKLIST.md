# Paper Figure Checklist

## Figures Referenced in Paper vs Generated

### Mathematical Foundation Section
No figures referenced in this section (metrics only)

### GPT-2 Case Study Section
✓ **Figure: gpt2_combined_sankey_caption** (included via `\input{figures/gpt2_combined_sankey_caption}`)
  - Generated: `gpt2_sankey_early.png`, `gpt2_sankey_middle.png`, `gpt2_sankey_late.png`
  
✓ **Figure: gpt2_trajectories** (subfigures)
  - (a) `figures/gpt2_stepped_layer_early.png` ✓
  - (b) `figures/gpt2_stepped_layer_middle.png` ✓
  - (c) `figures/gpt2_stepped_layer_late.png` ✓

### Heart Disease Case Study Section  
✓ **Figure: heart_sankey**
  - Referenced: `figures/heart_sankey.png` ✓
  
✓ **Figure: heart_trajectories**
  - Referenced: `figures/heart_stepped_layer_trajectories.png` ✓

### Other Sections
No additional figures referenced

## Summary
All figures referenced in the paper have been successfully generated:
- 6 GPT-2 figures (3 Sankey, 3 trajectory)
- 2 Heart disease figures (1 Sankey, 1 trajectory)

All figures use:
- Latest experimental data (1,228 words for GPT-2)
- UMAP for dimensionality reduction
- Proper color coding by class/grammatical type
- LLM-generated cluster labels
- Top 7 archetypal paths
- Thicker lines for visibility