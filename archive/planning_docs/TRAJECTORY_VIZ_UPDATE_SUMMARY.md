# Trajectory Visualization Update Summary

## Changes Made

### Heart Disease Visualization (`/experiments/heart_disease/generate_heart_trajectory_viz.py`)

1. **Color Coding by Disease Outcome**:
   - Updated colors to be darker and more saturated for better visibility
   - No Disease: Dark Green (`rgba(34, 139, 34, 1.0)`)
   - Disease: Crimson Red (`rgba(220, 20, 60, 1.0)`)
   - Colors are applied throughout ALL layers (not just output)

2. **Line Visibility Improvements**:
   - Increased line width from 2 to 4 for individual trajectories
   - Increased opacity from 0.3 to 0.8 for darker, more visible lines
   - Arrow opacity also increased to 0.8 to match

### GPT-2 Visualization (`/experiments/gpt2/semantic_subtypes/unified_cta/generate_gpt2_trajectory_viz.py`)

1. **Color Coding by Grammatical Type**:
   - Replaced archetype-based coloring with grammatical type coloring
   - Nouns: Dark Blue (`rgb(0, 0, 139)`)
   - Verbs: Crimson (`rgb(220, 20, 60)`)
   - Adjectives: Dark Green (`rgb(34, 139, 34)`)
   - Adverbs: Dark Orange (`rgb(255, 140, 0)`)
   - Other: Gray (`rgb(128, 128, 128)`)

2. **Word Type Detection**:
   - Added `load_word_type_data()` function to load word type mappings from:
     - Semantic subtypes wordlists
     - POS experiment data
   - Automatically categorizes tokens by their grammatical type

3. **Line Visibility Improvements**:
   - Increased individual trajectory line width from 1 to 3
   - Increased opacity from 0.2 to 0.7 for darker lines
   - Average path lines have width of 12 for clear visibility

4. **Updated Visualization**:
   - Shows average paths for each word type as thick lines
   - Individual word trajectories colored by their grammatical type
   - Updated title and subtitle to reflect grammatical type coloring

## How to Run

### Heart Disease:
```bash
python /experiments/heart_disease/generate_heart_trajectory_viz.py
```

### GPT-2:
```bash
python /experiments/gpt2/semantic_subtypes/unified_cta/generate_gpt2_trajectory_viz.py
```

## Output Files

Both scripts will generate:
- Interactive HTML files in their respective `results/` directories
- Static PNG images (if kaleido is installed)
- Copies in the `arxiv_submission/figures/` directory

## Notes

- The GPT-2 script will automatically detect and use available word type data
- If no actual activation data is found, it will create representative visualizations
- Verbs should now be properly visible with the crimson color throughout all layers