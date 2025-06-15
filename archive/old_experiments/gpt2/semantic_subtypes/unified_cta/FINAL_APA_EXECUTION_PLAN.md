# Final APA Execution Plan - Locked

## 🎯 Goal
Execute Archetypal Path Analysis using our unified CTA implementation to generate interpretable paths and clusters for GPT-2 semantic organization analysis.

## 📋 Task List (LOCKED)

### HIGH PRIORITY - Data Collection
1. **Run full 12-layer pipeline**: `python run_unified_pipeline.py --experiment full` (uses existing code)
2. **Extract within-layer metrics** from existing quality_checker results: silhouette scores, cluster size distributions for each layer
3. **Extract archetypal paths** using existing `identify_archetypal_paths()` with min_frequency=3, get path frequencies for top 10 paths
4. **For each top 10 archetypal paths**, extract representative words and their full L0->L11 cluster sequences using existing trajectory data
5. **Calculate cross-layer centroid similarity (ρᶜ)** using existing cross_layer_metrics.py functions
6. **Calculate membership overlap (J)** between adjacent layers using existing metrics
7. **Calculate trajectory fragmentation (F) scores** using existing trajectory_metrics functions
8. **For each semantic subtype**, extract trajectories and calculate similarity metrics using existing word_subtypes data
9. **Calculate between-subtype trajectory distribution distances** using existing similarity functions

### MEDIUM PRIORITY - LLM Analysis
10. **Compile ALL data for LLM analysis**: trajectories, archetypal paths, metrics, subtype analysis, comparison data
11. **LLM Stage 1**: Label all clusters across all layers
12. **LLM Stage 2**: Analyze archetypal paths, representative word paths, metrics, and generate comprehensive interpretable report
13. **Fill paper Table 1** with extracted layer-wise metrics (k, silhouette, coverage, purity)
14. **Fill paper Table 2** with top 10 archetypal path data and LLM interpretations

### LOW PRIORITY - Final Outputs
15. **Generate final visualizations** based on LLM analysis results

## 🔧 Implementation Notes

- **Uses existing unified CTA code only** - no new implementations
- **LLM provides interpretation** - we collect data, LLM analyzes for semantic meaning
- **Focus on interpretability** - clusters, paths, decision-making patterns
- **Two-stage LLM analysis** - cluster labeling, then comprehensive path analysis
- **Paper integration** - fills placeholders in arxiv submission

## 📊 Expected Deliverables

1. **Complete trajectory dataset** - 774 words × 12 layers with multi-cluster paths
2. **Top 10 archetypal paths** - with frequencies and representative words
3. **Cross-layer metrics** - ρᶜ, J, F values for all layers
4. **Semantic subtype analysis** - within/between coherence patterns
5. **LLM interpretations** - cluster labels and path narratives
6. **Paper tables and figures** - filled with actual results

## ✅ Success Criteria

- All 12 layers show k>1 clusters (no single-cluster problem)
- Rich archetypal path diversity for LLM analysis
- Semantic subtypes show interpretable trajectory patterns
- LLM generates meaningful cluster labels and path explanations
- Paper placeholders filled with comprehensive results

---

**Status**: Plan locked and ready for execution  
**Next Action**: Execute Task #1 (run full pipeline)  
**Estimated Time**: 2-3 hours for data collection, 1-2 hours for LLM analysis