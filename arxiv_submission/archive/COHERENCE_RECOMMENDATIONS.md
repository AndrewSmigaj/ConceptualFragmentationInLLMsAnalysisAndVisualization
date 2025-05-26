# Paper Coherence Recommendations

## 1. Major Issues to Address

### Remove Titanic Content
- Delete the Titanic report inclusion from LLM-Powered Analysis section
- Remove line: `\input{sections/generated/titanic_report}` from llm_powered_analysis.tex
- Focus exclusively on Heart Disease + GPT-2 case studies

### Add Missing GPT-2 Sankey Diagram
- GPT-2 section mentions "Concept MRI" visualization but has NO figure
- We created Sankey diagrams - need to add figure reference
- Suggested location: gpt2_semantic_subtypes_case_study.tex after the convergence discussion

### Fix Repetitive Content
The 72.8% finding is mentioned in:
- Abstract
- Introduction 
- GPT-2 case study
- Conclusion
- Future Directions

**Recommendation**: Keep brief mention in Abstract/Intro, full details only in case study

## 2. Structural Improvements

### Current Order (Problematic):
1. Introduction
2. Background
3. **Mathematical Foundation** (too early, too dense)
4. Statistical Robustness
5. Experimental Design
6. LLM-Powered Analysis
7. Heart Disease Case Study
8. GPT-2 Case Studies
9. Use Cases
10. Reproducibility
11. Societal Impact
12. Conclusion
13. Future Directions

### Recommended Order:
1. Introduction
2. Background
3. **Experimental Design** (what we did)
4. **Heart Disease Case Study** (simple example first)
5. **GPT-2 Case Studies** (complex example)
6. **Mathematical Foundation** (now readers have context)
7. Statistical Robustness
8. LLM-Powered Analysis (methodology only)
9. Reproducibility
10. Societal Impact
11. Conclusion
12. Future Directions

## 3. Terminology Standardization

### Cluster Labels
- **Use consistently**: L{layer}_C{cluster}
- Example: L4_C1, not L4C1 or L₄C₁

### Path Notation
- **Use consistently**: L0_C1 → L1_C0 → L2_C1
- Not: [L0_C1, L1_C0, L2_C1] or other formats

### Method Name
- First use: "Concept Trajectory Analysis (CTA)"
- Subsequent: "CTA" or "the CTA framework"

## 4. Specific Edits Needed

### In llm_powered_analysis.tex:
```latex
% DELETE THESE LINES:
\input{sections/generated/titanic_labels}
\input{sections/generated/titanic_narratives} 
\input{sections/generated/titanic_bias}
\input{sections/generated/titanic_report}
```

### In gpt2_semantic_subtypes_case_study.tex:
Add after convergence discussion:
```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.9\textwidth]{figures/gpt2_concept_flow_sankey.png}
    \caption{Concept MRI visualization showing word flow through GPT-2's layers. The dramatic convergence from 19 paths (early) to 5 paths (middle) to 4 paths (late) reveals how GPT-2 progressively organizes words by grammatical function rather than semantic meaning. Path thickness indicates the number of words following each trajectory.}
    \label{fig:gpt2_sankey}
\end{figure}
```

### In experimental_design.tex:
Remove duplicate content about metrics (already in Mathematical Foundation)

## 5. Content Consolidation

### "Concept MRI" Definition
Define ONCE in Experimental Design:
> "Concept MRI" is our visualization technique using Sankey diagrams to show concept flow through neural network layers, analogous to how medical MRI reveals internal structure.

Then just reference it elsewhere.

### Cross-layer Metrics
Keep full definition ONLY in Mathematical Foundation section.
Other sections should just reference: "using cross-layer metrics (Section X.X)"

## 6. Missing Elements

### Need to Create/Add:
1. GPT-2 Sankey diagram figure file
2. Transition paragraphs between major sections
3. Summary table comparing Heart Disease vs GPT-2 findings
4. Limitations discussion (add to Conclusion)

## 7. Quick Wins

### Remove these redundant files:
- titanic_labels.tex
- titanic_narratives.tex  
- titanic_bias.tex
- titanic_report.tex

### Standardize all cluster labels:
- Global find/replace: "L(\d)C(\d)" → "L$1_C$2"

### Fix path arrows:
- Global find/replace: "→" → "$\rightarrow$" (for proper LaTeX rendering)

## Implementation Priority

1. **HIGH**: Remove Titanic content
2. **HIGH**: Add GPT-2 Sankey figure
3. **HIGH**: Fix repetitive 72.8% mentions
4. **MEDIUM**: Standardize terminology
5. **MEDIUM**: Reorder sections for better flow
6. **LOW**: Add transitions and summary table

These changes will significantly improve the paper's coherence and readability while maintaining all key findings.