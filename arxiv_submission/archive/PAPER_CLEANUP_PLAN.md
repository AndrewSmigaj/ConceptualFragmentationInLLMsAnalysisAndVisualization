# Paper Cleanup Plan

## Critical Issues to Fix

### 1. WRONG GPT-2 CASE STUDY
- **Problem**: Section 9 (gpt2_case_study.tex) is about semantic pivots (202 sentences), not the main semantic subtypes experiment (566 words)
- **Action**: Remove this section from main.tex or rename to clarify it's a different experiment

### 2. REDUNDANT CONTENT
- **Cross-layer metrics defined twice**:
  - Section 4.6 in mathematical_foundation.tex
  - Section 5.1 in statistical_robustness.tex
- **Action**: Keep only in mathematical_foundation.tex, remove from statistical_robustness.tex

### 3. ETS CONTENT TO MOVE
- **Current locations**:
  - Abstract line 4: "unique cluster labeling"
  - Introduction line 22: "ETS-based clustering"
  - Mathematical Foundation section 4.4: Full ETS explanation
- **Action**: Move detailed ETS content to Future Directions

### 4. MISPLACED FIGURES
- **reproducibility.tex**: Has subspace_angle.png (should be in results)
- **societal_impact.tex**: Has cluster_trajectory.png (should be in results)
- **conclusion.tex**: Has cluster_entropy.png (should be in results)
- **future_directions.tex**: Has optimal_clusters.png (should be in results)
- **Action**: Remove these figure references or move to appropriate sections

### 5. SHORT SECTIONS TO MERGE
- **background.tex** → Merge into introduction.tex
- **use_cases.tex** → Merge into future_directions.tex
- **societal_impact.tex** → Expand or merge into conclusion.tex

### 6. OLD TERMINOLOGY
- **llm_powered_analysis.tex**: References to "cluster cards"
- **Action**: Update to current terminology

### 7. GENERATED CONTENT ISSUES
- **heart_report.tex** and **titanic_report.tex**: Poor LaTeX formatting
- **Unused files**: heart_narratives.tex, titanic_narratives.tex, etc.
- **Action**: Fix formatting or remove if not needed

## Section Status Summary

| Section | Status | Action |
|---------|---------|---------|
| abstract.tex | ✅ Good | Keep, minor ETS update |
| introduction.tex | ✅ Good | Keep, merge background |
| background.tex | ❌ Too short | Merge into introduction |
| mathematical_foundation.tex | ✅ Core | Keep, move ETS details |
| statistical_robustness.tex | ⚠️ Redundant | Remove redundant parts |
| experimental_design.tex | ✅ Good | Keep |
| llm_powered_analysis.tex | ⚠️ Needs cleanup | Update terminology |
| heart_disease_case_study.tex | ✅ Excellent | Keep |
| gpt2_case_study.tex | ❌ Wrong study | Remove or clarify |
| gpt2_semantic_subtypes_case_study.tex | ✅ Core finding | Keep |
| use_cases.tex | ❌ Too short | Merge to future |
| reproducibility.tex | ⚠️ Figure issue | Remove figure |
| societal_impact.tex | ❌ Too short | Expand or merge |
| conclusion.tex | ✅ Good | Remove figure |
| future_directions.tex | ✅ Good | Add ETS content |

## Execution Order

1. First: Remove wrong GPT-2 case study from main.tex
2. Second: Fix redundant metrics definitions
3. Third: Move ETS content to future directions
4. Fourth: Merge short sections
5. Fifth: Fix figure placements
6. Sixth: Clean up terminology and formatting