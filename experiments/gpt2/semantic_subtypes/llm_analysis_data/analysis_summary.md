# GPT-2 Semantic Subtypes Clustering Analysis Summary

## Overview
This analysis examines how GPT-2 organizes 774 single-token words from 8 semantic subtypes across its 13 layers using optimal clustering configurations determined by elbow method.

## Data Prepared for Analysis

### 1. Cluster Contents (`cluster_contents_by_layer.json`)
- Detailed word lists for each cluster at key layers (0, 1, 2, 5, 7, 10, 11, 12)
- Subtype distribution within each cluster
- Purity scores and dominant subtypes

### 2. Semantic Profiles (`cluster_semantic_profiles.json`)
- How each semantic subtype distributes across clusters
- Layer-by-layer evolution of subtype clustering

### 3. Unexpected Groupings (`unexpected_groupings.json`)
- Word pairs from different subtypes that consistently cluster together
- Layer-specific surprising combinations

### 4. Outlier Analysis (`outlier_words_analysis.json`)
- Singleton clusters and their contents
- Consistently outlying words like 'depend'

### 5. Transition Patterns (`layer_transition_patterns.json`)
- How words move between clusters across layers
- Stability analysis by semantic subtype

## Key Questions for LLM Analysis

1. **Cluster Interpretation**: What semantic features unite words in each cluster that transcend our predefined categories?

2. **Emergent Organization**: What alternative semantic taxonomy has GPT-2 learned through distributional patterns?

3. **Layer Evolution**: How does semantic granularity and organization change across layers?

4. **Outlier Insights**: What makes certain words (especially 'depend') consistently outliers?

5. **Theoretical Implications**: What does this reveal about how transformers learn and organize meaning?

## Analysis Approach

1. Start with cluster interpretation at key layers
2. Identify emergent semantic patterns
3. Track evolution across layers
4. Analyze outliers and unexpected groupings
5. Synthesize findings into a theory of GPT-2's semantic organization

## Files Generated
- `llm_analysis_plan.json`: Structured analysis plan
- `cluster_contents_by_layer.json`: Main data file with cluster contents
- `cluster_semantic_profiles.json`: Subtype distribution analysis
- `unexpected_groupings.json`: Surprising word combinations
- `outlier_words_analysis.json`: Outlier and singleton analysis
- `layer_transition_patterns.json`: Cluster transition patterns
- `llm_analysis_prompts.json`: Specific prompts for analysis
