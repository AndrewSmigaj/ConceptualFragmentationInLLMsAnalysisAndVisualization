# GPT-2 ARCHETYPAL PATH ANALYSIS – PROOF-OF-CONCEPT EXPERIMENT (REVISED)

## PURPOSE
Demonstrate that semantic pivots ("but") produce measurable trajectory changes in GPT-2 using full APA methodology. Distinguish between syntactic pivot effects (all "but" sentences) and semantic contradiction effects (sentiment-contrasting sentences only).

## MACHINE-READABLE SPEC

```json
{
  "experiment_name": "GPT2_APA_Pivot_Full_Analysis",
  "objective": "Apply complete APA methodology to GPT-2 pivot sentences to generate narratives explaining syntactic vs semantic processing layers.",
  "hypothesis": "Syntactic pivot effects occur in early layers (1-3), semantic contradiction effects in middle layers (4-7), with distinct fragmentation patterns between contrast and consistent classes.",
  "input_templates": {
    "contrast_class": {
      "pattern": "<positive clause> but <negative clause>",
      "examples": [
        "The movie was engaging but the ending ruined everything.",
        "The meal tasted awful but the service saved the experience."
      ],
      "count": 50
    },
    "consistent_class": {
      "pattern": "<positive clause> but <positive clause>",
      "examples": [
        "The movie was engaging but the acting made it amazing.",
        "The meal tasted great but the service made it perfect."
      ],
      "count": 50
    },
    "tokens_per_sentence": 10,
    "pivot_token_index": 4
  },
  "apa_methodology": {
    "full_pipeline": true,
    "layer_alignment": "procrustes_after_pca",
    "clustering": "existing_GPT2TokenClusterer",
    "path_tracking": "full_13_layers",
    "narrative_generation": "llm_powered"
  },
  "metrics_to_compute": {
    "original_apa_metrics": [
      "silhouette_score",
      "ari", 
      "mi",
      "path_reproducibility",
      "path_purity",
      "centroid_similarity_rho_c",
      "jaccard_similarity_J",
      "fragmentation_score_F_i",
      "similarity_convergent_path_density_D",
      "interestingness_score_I",
      "k_star_optimal_clusters"
    ],
    "pivot_specific_metrics": [
      "simple_fragmentation_delta",
      "path_divergence_index"
    ]
  },
  "expected_layered_effects": {
    "layers_1_3": "syntactic_pivot_processing",
    "layers_4_7": "semantic_contradiction_processing",
    "layers_8_12": "consolidation_or_divergence"
  },
  "controls": [
    "random_word_order",
    "monotone_sentiment_and_replacement", 
    "scrambled_weights"
  ],
  "primary_goal": "generate_llm_narratives_explaining_processing_stages"
}
```

## REVISED IMPLEMENTATION PLAN

### Phase 1: Data Preparation
1. Generate 100 sentences (50 contrast, 50 consistent) fitting templates
2. Verify tokenization and pivot token index consistency

### Phase 2: Use Existing APA Infrastructure  
1. **GPT2Adapter**: Extract activations from all 13 layers
2. **Existing clustering pipeline**: Apply full APA clustering methodology
3. **Existing metrics calculation**: Compute ALL original APA metrics (ρ^c, J, F_i, D, I, k*, etc.)

### Phase 3: Add Pivot-Specific Analysis
1. **NEW**: Implement Procrustes alignment for layer comparison
2. **NEW**: Simple fragmentation delta (unique IDs / layers)
3. **NEW**: Path divergence index (Hamming distance)
4. **NEW**: Layer-specific pivot effect detection

### Phase 4: Full APA Analysis & Narrative Generation
1. **Use existing LLM narrative system** to generate explanations
2. Compare narrative quality between contrast/consistent classes
3. Generate layer-specific processing explanations

## COMPONENTS TO USE (NOT REIMPLEMENT)

### Existing Infrastructure:
- `GPT2Adapter` for model interface
- `GPT2TokenClusterer` for clustering
- `GPT2DimensionalityReducer` for PCA
- `cluster_paths.py` for path computation
- `similarity_metrics.py` for ρ^c, J calculations
- Existing LLM narrative generation system
- All original APA metrics (silhouette, ARI, MI, etc.)

### New Components Required:
1. **Procrustes alignment** implementation
2. **Simple fragmentation delta** calculation  
3. **Path divergence index** calculation
4. **Layer-specific pivot effect analysis**

## SUCCESS CRITERIA

### Quantitative:
- **Contrast class**: Higher fragmentation in layers 4-7 than consistent class
- **Both classes**: Similar syntactic fragmentation in layers 1-3
- **Statistical significance**: t-test between classes per layer

### Qualitative (Primary Goal):
- **LLM narratives** successfully distinguish syntactic vs semantic processing
- **Narrative quality**: Coherent explanations of layer-specific effects
- **Interpretability**: Clear identification of where different processing occurs

## OUTPUT DELIVERABLES

1. **Full APA Analysis Results**: All original metrics computed
2. **Layer-specific Analysis**: Fragmentation patterns by processing stage  
3. **LLM Narratives**: Explanations of syntactic vs semantic effects
4. **Comparative Analysis**: Contrast vs consistent class differences
5. **Processing Stage Identification**: Which layers handle what type of processing

## ANTICIPATED RESULTS

**Early Layers (1-3)**: Both classes show fragmentation due to "but" (syntactic)
**Middle Layers (4-7)**: Contrast class shows additional fragmentation (semantic)  
**Later Layers (8-12)**: Potential consolidation or continued processing

**LLM Narratives Should Explain**: "Tokens initially fragment due to syntactic pivot processing, then contrast-class tokens show additional semantic contradiction processing in middle layers..."

---
**This experiment validates APA on transformers while generating interpretable narratives about internal processing stages.**