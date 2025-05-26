# GPT-2 Semantic Subtypes Experiment: APA Execution Plan

## Core Methodology
**Archetypal Path Analysis (APA)**: Track paths that words take through cluster assignments across GPT-2's 13 layers, then use LLMs to interpret these paths.

## What We're Running

### 1. Extract Activations
- Process 774 single-token words through GPT-2
- Extract 768-dimensional vectors at each of 13 layers
- Output: `semantic_subtypes_activations.pkl`

### 2. Apply TWO Clustering Methods
As implemented in our code:

**K-means**:
- Silhouette-based optimal k selection (k=2-15)
- Applied independently at each layer

**ETS (Explainable Threshold Similarity)**:
- Dimension-wise threshold computation (default percentile=0.1)
- Finds natural clusters based on activation thresholds
- Falls back to k-means if dependencies are missing

### 3. Track Archetypal Paths
For each word and each clustering method:
- Record cluster assignment at each layer
- Create path: [L0C2 → L1C1 → L2C3 → ... → L12C1]
- Compare paths between k-means and ETS

### 4. Calculate APA Metrics
Our implemented metrics:
- **ρ^c**: Centroid similarity between adjacent layers
- **J**: Membership overlap between adjacent layers
- **F_i**: Fragmentation score per path
- **Path purity**: Same subtype → same path?

### 5. Semantic Organization Analysis
- **Within-subtype coherence**: Do words from same subtype follow similar paths?
- **Between-subtype differentiation**: Do different subtypes follow different paths?
- **Clustering method comparison**: K-means vs ETS performance

### 6. Data Preparation for User's Manual Analysis
Generate markdown file with:
- Cluster statistics for both methods
- Path analysis by semantic subtype  
- Cross-layer metrics (ρ^c, J)
- APA metrics (F_i fragmentation scores, path purity)
- Complete archetypal path listings
- Formatted for easy copy-paste into LLM for interpretability scoring

## Run Command
```bash
cd experiments/gpt2/semantic_subtypes
python gpt2_semantic_subtypes_experiment.py --output-dir results/$(date +%Y%m%d_%H%M%S)
```

## Expected Outputs

1. **Two sets of clustering results**:
   - `semantic_subtypes_kmeans_clustering.pkl`
   - `semantic_subtypes_hdbscan_clustering.pkl`

2. **Two sets of APA metrics**:
   - `semantic_subtypes_kmeans_apa_metrics.json`
   - `semantic_subtypes_hdbscan_apa_metrics.json`

3. **Comparison analysis**:
   - `clustering_methods_comparison.json`
   - Which method better captures semantic organization?

4. **Semantic analysis**:
   - `semantic_organization_analysis.json`
   - Within/between subtype patterns

## Key Questions

### Archetypal Path Analysis
1. What archetypal paths emerge for each semantic subtype?
2. Do words within the same subtype follow similar paths (within-subtype coherence)?
3. Do different subtypes follow distinct paths (between-subtype differentiation)?
4. At what layers do semantic distinctions become most pronounced?
5. Which layers show the highest fragmentation (F_i) scores?
6. What is the path purity for each semantic subtype?

### Clustering Method Comparison
7. Do K-means and ETS reveal similar archetypal paths?
8. Does ETS's threshold-based approach improve semantic clustering?
9. Which method better separates semantic subtypes?
10. Which method produces more interpretable cluster assignments?

### Semantic Organization Insights
11. How does GPT-2 organize semantic knowledge across layers?
12. Do grammatical categories (nouns, verbs, etc.) cluster together?
13. Within grammatical categories, how are semantic distinctions encoded?
14. What do the cross-layer metrics (ρ^c, J) reveal about representation stability?

## Timeline
- Activation extraction: ~30 minutes
- K-means clustering: ~5 minutes
- ETS clustering: ~5 minutes
- Metrics & analysis: ~5 minutes
- Total: ~45 minutes