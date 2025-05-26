# Paper Cleanup Summary

## Major Changes Completed

### 1. Fixed Sankey Diagrams
- **Heart Disease**: Fixed overlapping clusters by adjusting vertical positions (0.2, 0.8) and increasing padding to 25px
- **GPT-2**: Created improved versions with better node positioning, consistent color scheme, and phase labels

### 2. Removed Wrong Case Study
- Commented out `gpt2_case_study.tex` (semantic pivot study) from main.tex
- Kept only the correct `gpt2_semantic_subtypes_case_study.tex`

### 3. Fixed Redundant Content
- Removed duplicate cross-layer metrics from `statistical_robustness.tex` (already in `mathematical_foundation.tex`)
- Kept statistical robustness focused on path reproducibility and trajectory coherence

### 4. Moved Content to Future Directions
- Moved detailed ETS explanation from mathematical foundation to future directions
- Moved unused metrics (ICPD, Path Interestingness, Feature Attribution) to future directions
- Restructured future directions with "Methodological Foundations" and "Advanced Applications" sections

### 5. Merged Short Sections
- Merged `background.tex` into `introduction.tex` as a subsection
- Merged `use_cases.tex` into `future_directions.tex` as "Practical Use Cases"
- Merged `societal_impact.tex` into `conclusion.tex` as "Limitations and Responsible Use"

### 6. Fixed Misplaced Figures
- Removed unrelated figures from:
  - `conclusion.tex` (cluster_entropy.png)
  - `reproducibility.tex` (subspace_angle.png)

### 7. Updated Terminology
- Changed "noun superhighway" to "entity superhighway" throughout (more accurate since it includes various entity types)
- "Cluster cards" terminology already only in future directions (no changes needed)

### 8. Citation Updates
- Updated future directions to clarify we didn't coin "Concept MRI" or "ETS"
- Added proper citation for ETS: \citep{kovalerchuk2024}

## Paper Structure After Cleanup

1. Abstract
2. Introduction (includes background)
3. Mathematical Foundation
4. Statistical Robustness
5. Experimental Design
6. LLM-Powered Analysis
7. Heart Disease Case Study
8. GPT-2 Semantic Subtypes Case Study
9. Reproducibility
10. Conclusion (includes limitations)
11. Future Directions (includes use cases, ETS details, unused metrics)

## Files Modified
- main.tex
- introduction.tex
- mathematical_foundation.tex
- statistical_robustness.tex
- future_directions.tex
- conclusion.tex
- abstract.tex
- gpt2_semantic_subtypes_case_study.tex
- gpt2_combined_sankey_caption.tex
- reproducibility.tex
- generate_labeled_heart_sankey.py
- generate_gpt2_paper_figures_improved.py (new file)

## Next Steps
- Compile LaTeX to ensure no errors
- Review figure references and ensure all are correctly placed
- Final proofreading pass