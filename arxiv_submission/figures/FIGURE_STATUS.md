# Figure Generation Status

## GPT-2 Figures (All Generated)
- ✓ `gpt2_sankey_early.png` - Early window Sankey diagram with cluster labels
- ✓ `gpt2_sankey_middle.png` - Middle window Sankey diagram with cluster labels
- ✓ `gpt2_sankey_late.png` - Late window Sankey diagram with cluster labels
- ✓ `gpt2_stepped_layer_early.png` - Early window trajectory visualization (UMAP-based)
- ✓ `gpt2_stepped_layer_middle.png` - Middle window trajectory visualization (UMAP-based)
- ✓ `gpt2_stepped_layer_late.png` - Late window trajectory visualization (UMAP-based)

## Heart Disease Figures (All Generated)
- ✓ `heart_sankey.png` - Structured path Sankey with top 7 archetypal paths
- ✓ `heart_stepped_layer_trajectories.png` - Trajectory visualization using actual cluster path data (UMAP-based)

## Mathematical Foundation Figures (All Present)
- ✓ `cluster_entropy.png` - Cluster entropy over layers
- ✓ `intra_class_distance.png` - Intra-class distance metrics
- ✓ `subspace_angle.png` - Subspace angle between layers
- ✓ `optimal_clusters.png` - Optimal number of clusters per layer

## Key Updates Made:
1. **Heart Disease Visualization Fixed**: Switched from synthetic random data to actual cluster path data from the dash app
2. **UMAP Used Throughout**: All trajectory visualizations now use UMAP instead of PCA
3. **Proper Color Coding**: 
   - Heart disease: By outcome (disease/no disease)
   - GPT-2: By grammatical type (noun/verb/adjective)
4. **Thicker Lines**: Increased line thickness to 3 and opacity to 0.7 for better visibility
5. **Cluster Labels**: All Sankey diagrams show LLM-generated cluster labels
6. **Top 7 Paths**: Both models show top 7 archetypal paths with percentages

## Notes:
- All figures use the latest experimental data (1,228 words for GPT-2)
- Sankey diagrams have proper margins to prevent cutoff
- Background grids removed for clarity
- All figures saved in arxiv_submission/figures directory