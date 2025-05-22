# GPT-2 ARCHETYPAL PATH ANALYSIS – PROOF-OF-CONCEPT BRIEF

## PURPOSE
Demonstrate that a single semantic pivot ("but") in short sentences produces a measurable fork in Archetypal Path Analysis (APA) trajectories inside GPT-2. Success validates APA on transformer activations.

## MACHINE-READABLE SPEC

```json
{
  "experiment_name": "GPT2_APA_Pivot_Demo",
  "objective": "Show that a token-level semantic pivot provokes a measurable fork in APA trajectories within GPT-2.",
  "hypothesis": "A sentiment-contrasting pivot ('but' + sentiment flip) causes statistically significant divergence between pre-pivot and post-pivot paths compared to sentiment-consistent pivots ('but' + same sentiment) across ≥80% of contrast-class sentences.",
  "input_templates": {
    "contrast_class": {
      "pattern": "<positive clause> but <negative clause>",
      "examples": [
        "The movie was engaging but the ending ruined everything.",
        "The meal tasted awful but the service saved the experience."
      ]
    },
    "consistent_class": {
      "pattern": "<positive clause> but <positive clause>",
      "examples": [
        "The movie was engaging but the acting made it amazing.",
        "The meal tasted great but the service made it perfect."
      ]
    },
    "tokens_per_sentence": 10,
    "pivot_token_index": 4
  },
  "focus_tokens": {
    "pivot": "but",
    "post_pivot": "token_after_but"
  },
  "layer_alignment": {
    "method": "orthogonal_procrustes",
    "preprocessing": "per-layer PCA to 32 dimensions followed by whitening",
    "note": "NEW IMPLEMENTATION REQUIRED"
  },
  "clustering": {
    "algorithm": "existing_GPT2TokenClusterer",
    "k_selection": "silhouette_elbow",
    "expected_k_range": [4, 6],
    "label_format": "L<layer>C<cluster>"
  },
  "metrics": [
    "fragmentation_delta",
    "path_divergence_index", 
    "cluster_stability"
  ],
  "new_metrics_required": [
    "fragmentation_delta",
    "path_divergence_index"
  ],
  "pass_fail_criteria": {
    "pass": "fragmentation_delta > 1.0 && path_divergence_index > 0.4 within ±2 layers in ≥80% of sentences",
    "fail": "otherwise"
  },
  "controls": [
    "random_word_order",
    "monotone_sentiment", 
    "scrambled_weights"
  ],
  "visual_output": "sankey_single_sentence"
}
```

## IMPLEMENTATION STEPS (FOR AI AGENT)

### 1. Data Generation
- Produce N = 100 sentences that fit the template. Keep total tokens = 10.
- Store in plain text.

### 2. Model Setup
```python
# Use existing GPT2Adapter instead of raw model
from concept_fragmentation.models.transformer_adapter import GPT2Adapter
from transformers import GPT2Model, GPT2Tokenizer

model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
adapter = GPT2Adapter(model, tokenizer=tokenizer)
```

### 3. Activation Extraction
- Use existing `adapter.extract_activations()` method
- Capture activations from all 12 transformer blocks plus embedding layer (13 total)

### 4. Layer Alignment **[NEW IMPLEMENTATION REQUIRED]**
For each consecutive layer pair:
- Use existing `GPT2DimensionalityReducer` for PCA → 32 dims
- **NEW**: Implement whitening preprocessing  
- **NEW**: Implement orthogonal Procrustes alignment

### 5. Clustering
- Use existing `GPT2TokenClusterer` with k ∈ [4, 6]
- Choose k with highest silhouette (> 0.20)
- Label clusters L{layer}C{idx}

### 6. Path & Metric Computation **[PARTIALLY NEW]**
- **Path** = sequence of cluster IDs for each token across 13 layers
- **Fragmentation (F)** = unique IDs / 13 (existing logic)
- **NEW**: **Fragmentation Δ** = post-pivot F – pre-pivot F (average over sentences)
- **NEW**: **Path Divergence Index** = average Hamming distance between pre- and post-pivot paths
- **Cluster Stability** = bootstrap silhouette (existing via metrics framework)

### 7. Controls

| Control | Modification | Expected Result |
|---------|-------------|-----------------|
| Random word order | Shuffle tokens in each sentence | Flat fragmentation Δ, low divergence |
| Monotone sentiment | Replace "but" with "and"; keep sentiment consistent | No localized spike near pivot |
| Scrambled weights | Permute weight matrices within each GPT-2 block | Pivot effect disappears |

### 8. Pass / Fail Logic
- Pass if criteria in spec are met; else fail
- Store per-sentence metrics for audit

### 9. Visualization
Generate one Sankey diagram (pivot_sankey.png) for the first sentence:
- **Source** = token IDs at layer 0
- **Target** = cluster IDs at layer 12  
- Highlight pivot token ribbon

### 10. Output Package
Deliver folder:
```bash
metrics_summary.json
per_sentence_metrics.csv
pivot_sankey.png
run_config.json   # exact copy of machine-readable spec
README.txt        # brief description and pass/fail result
```

## IMPLEMENTATION NOTES

### Existing Components to Use:
- `GPT2Adapter` for model interface
- `GPT2TokenClusterer` for clustering
- `GPT2DimensionalityReducer` for PCA
- `TransformerMetricsCalculator` for stability metrics

### New Components Required:
1. **Orthogonal Procrustes alignment** for layer alignment
2. **Whitening preprocessing** for PCA output
3. **Fragmentation delta metric** calculation
4. **Path divergence index** calculation

### Design Validation:
✅ Uses existing architecture and components  
✅ Clear quantitative success criteria  
✅ Proper experimental controls  
✅ Machine-readable specification  
✅ Builds on established APA methodology  

## ANTICIPATED OBJECTIONS & REBUTTALS

| Objection | Mitigation in Design |
|-----------|---------------------|
| Layer embeddings incomparable | Alignment via PCA + Procrustes |
| Cluster count arbitrary | Silhouette-based selection, k-range disclosed |
| Effect could be random | Three independent controls |
| Hamming on labels is brittle | Optional centroid-distance analysis (appendix) |

## NEXT STEPS AFTER DEMO
1. Extend to longer texts with multiple pivots
2. Compare vanilla GPT-2 to sentiment-fine-tuned GPT-2  
3. Explore APA-based regularizers for interpretability-aware fine-tuning

---
**END OF BRIEF**