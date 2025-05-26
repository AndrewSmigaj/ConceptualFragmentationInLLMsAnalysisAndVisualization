# New Figures Created in This Session

## Heart Disease Figures

1. **heart_sankey_labeled_final.png** - Fixed Sankey diagram with proper node separation
   - Increased padding to 40px
   - Unique y-positions for all nodes to prevent overlap
   - Shows 5 archetypal paths through the network

2. **heart_stepped_layer_trajectories.png** - UMAP-based 3D trajectory visualization
   - Shows patient trajectories through Input → Hidden1 → Hidden2 → Output
   - Archetypal paths as thick lines
   - Individual trajectories colored by class (green=no disease, red=disease)

## GPT-2 Figures

3. **gpt2_stepped_layer_early.png** - UMAP trajectories for layers 0-3
   - Shows early layer token movements
   - Entity superhighway and other grammatical paths

4. **gpt2_stepped_layer_middle.png** - UMAP trajectories for layers 4-7
   - Shows middle layer convergence patterns
   - Increasing dominance of entity superhighway

5. **gpt2_stepped_layer_late.png** - UMAP trajectories for layers 8-11
   - Shows late layer convergence
   - 72.8% convergence to entity superhighway

6. **gpt2_sankey_early_improved.png** - Improved early window Sankey
7. **gpt2_sankey_middle_improved.png** - Improved middle window Sankey
8. **gpt2_sankey_late_improved.png** - Improved late window Sankey
   - All with better layout, consistent colors, and "entity superhighway" terminology

## Key Updates
- All trajectory visualizations use UMAP (not PCA) as requested
- No endpoints on trajectory lines
- Archetypal paths shown as thick lines
- Individual trajectories shown as thin lines with appropriate opacity
- Heart sankey has no overlapping nodes
- GPT-2 terminology changed from "noun superhighway" to "entity superhighway"