# Remaining Paper Work Plan

## 1. Review Changes Made So Far ✓ CHECK

### Completed Updates:
- **Abstract**: Emphasized heart disease, removed 8-day timeline, kept GPT-2 discovery prominent
- **Introduction**: Removed 8-day timeline, emphasized heart disease clinical applications
- **Future Directions**: Removed 8-day references (2 instances)
- **Conclusion**: Detailed heart disease findings with 4 archetypes, removed Titanic emphasis

### Quality Checks Needed:
- [ ] Verify all "8 days" references are removed
- [ ] Ensure heart disease is consistently emphasized
- [ ] Check that GPT-2 findings remain prominent
- [ ] Verify consistency of metrics across sections

## 2. Heart Disease Case Study Section 📊

### Current Status:
- Heart disease is mentioned in multiple sections
- Generated reports exist in `/sections/generated/heart_*.tex`
- Sankey diagram exists at `/results/figures/heart/membership_overlap_sankey.html`

### Plan:
1. Check if we need a dedicated section or if current coverage is sufficient
2. If needed, create `heart_disease_case_study.tex` that includes:
   - Overview of heart disease diagnosis challenge
   - CTA methodology application
   - Four archetype analysis
   - Sankey diagram visualization
   - Clinical implications

## 3. Mathematical Foundations Section 📐

### Review Needed:
- [ ] Ensure windowed analysis is formally defined
- [ ] Verify unique cluster labeling notation (L{layer}_C{cluster})
- [ ] Check all metrics match implementation:
  - Centroid similarity (ρ^c)
  - Membership overlap (J)
  - Trajectory fragmentation (F)
- [ ] Add Gap statistic methodology if not present

## 4. Experimental Design Section 🔬

### Updates Needed:
- [ ] Add unified CTA approach details
- [ ] Document three-phase methodology
- [ ] Include windowed analysis design
- [ ] Ensure heart disease dataset description is complete
- [ ] Update dataset statistics

## 5. Figures and Tables Review 📊

### Checklist:
- [ ] Verify all referenced figures exist
- [ ] Check figure captions are informative
- [ ] Ensure tables have proper formatting
- [ ] Add heart disease Sankey if not included
- [ ] Consider adding GPT-2 convergence visualization

## 6. Final Quality Checks ✅

### Consistency Checks:
- [ ] Metrics are consistent across all sections
- [ ] Terminology is uniform (CTA, Concept MRI, etc.)
- [ ] Citations are complete
- [ ] No placeholder text remains

### Proofreading:
- [ ] Grammar and spelling
- [ ] LaTeX compilation without errors
- [ ] References format
- [ ] Mathematical notation consistency

## Execution Order:

1. **First**: Review all changes for consistency (15 min)
2. **Second**: Check Mathematical Foundations (30 min)
3. **Third**: Update Experimental Design (30 min)
4. **Fourth**: Decide on heart disease case study (15 min)
5. **Fifth**: Review figures/tables (20 min)
6. **Last**: Final proofreading (30 min)

Total estimated time: 2-2.5 hours