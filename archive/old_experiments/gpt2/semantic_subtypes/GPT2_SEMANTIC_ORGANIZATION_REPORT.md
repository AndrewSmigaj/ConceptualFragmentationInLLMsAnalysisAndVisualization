# GPT-2 Semantic Organization: A Comprehensive Analysis

## Executive Summary

This analysis examines how GPT-2 organizes 774 single-token words from 8 predefined semantic subtypes across its 13 layers. Using optimal clustering configurations (K-means with elbow method and matched ETS thresholds), we discovered that GPT-2 learns a fundamentally different semantic organization than traditional grammatical categories, instead grouping words by functional and experiential dimensions.

## Key Findings

### 1. Functional Over Formal Organization

GPT-2 consistently groups words based on their functional role in language rather than traditional grammatical categories:

- **"feel" + "nice"**: Despite being from different categories (stative verb + degree adverb), these words cluster together in 208 out of 208 layer comparisons
- **"clean" + "nice"**: Action verb + degree adverb pair together in 160 layer comparisons
- This suggests GPT-2 learns experiential/evaluative dimensions that transcend grammatical boundaries

### 2. Emergent Semantic Dimensions

The analysis reveals several key semantic dimensions that GPT-2 uses to organize words:

#### Evaluative/Experiential
- Words: good, bad, feel, nice, fine, poor
- Unites evaluative terms with experiential states
- Reflects how evaluation and experience are linguistically intertwined

#### Epistemic/Cognitive
- Words: know, think, believe, see, understand
- Groups mental processes regardless of sensory vs. cognitive distinction
- Suggests unified representation of knowledge acquisition

#### Modal/Auxiliary
- Words: can, could, may, might, will, would
- Maintains distinct category for possibility/necessity
- Critical for syntactic prediction tasks

#### Scalar/Degree
- Words: very, too, quite, rather, almost, nearly
- Groups intensifiers and degree modifiers
- Important for compositional semantics

### 3. Layer-wise Evolution

The semantic organization evolves systematically across layers:

#### Early Layers (0-2)
- **Purity scores**: 0.278-0.340
- Broad lexical categories mixing multiple subtypes
- Surface-level features dominate
- Example: Layer 0 Cluster 0 contains 71 degree adverbs, 62 stative verbs, and 22 manner adverbs

#### Middle Layers (5-7)
- **Purity scores**: 0.185-0.368
- Semantic role differentiation emerges
- Functional groupings become more prominent
- Cross-category clusters based on semantic similarity

#### Late Layers (10-12)
- **Purity scores**: 0.185-0.316
- Prediction-oriented reorganization
- Maintains syntactically relevant distinctions
- Degree and manner adverbs remain somewhat separated for modifier prediction

### 4. Outlier Analysis

Certain words consistently appear as outliers, revealing semantic complexity:

#### "depend"
- Outlier across multiple layers
- Modal-like conditional semantics don't fit action/stative dichotomy
- Encodes complex relational/conditional meaning

#### "sun"
- Frequently isolated despite being a concrete noun
- Unique semantic properties (celestial, singular, universal reference)
- Differs from typical concrete objects

#### Past Tense Forms
- "saw", "found" often separate from base forms
- Suggests aspectual/temporal dimensions in representation
- Past tense carries distinct semantic weight

### 5. Clustering Performance Comparison

#### K-means (Optimal k per layer)
- Most layers optimal at k=3 (except layer 0 with k=4)
- Average silhouette score: ~0.3-0.4
- Provides stable, interpretable clusters

#### ETS (Matched thresholds)
- Thresholds ranging from 0.99615 to 0.99740
- Successfully matched K-means cluster counts
- More sensitive to individual word properties

## Theoretical Implications

### 1. Usage-Based Semantics
GPT-2's organization strongly supports usage-based theories of meaning where:
- Words that appear in similar contexts develop similar representations
- Functional similarity trumps formal category membership
- Meaning emerges from distributional patterns

### 2. Prototype Theory
The clustering patterns suggest prototype-based categorization:
- Clusters have fuzzy boundaries with graded membership
- Central members (high purity regions) vs. peripheral members
- Cross-category clusters around functional prototypes

### 3. Compositional Semantics
The maintenance of certain distinctions (especially in late layers) reflects compositional needs:
- Degree modifiers remain somewhat distinct for scalar composition
- Modal auxiliaries maintain separate identity for syntactic prediction
- Balance between semantic similarity and syntactic function

## Methodological Insights

### 1. Optimal Clustering Configuration
- Elbow method effectively identifies natural cluster boundaries
- K=3 appears to be a sweet spot for most layers
- Layer-specific optimization captures evolution of representations

### 2. Cross-Method Validation
- K-means and ETS show convergent patterns
- Both methods identify similar unexpected groupings
- Validates robustness of findings

### 3. Multi-Scale Analysis
- Layer-by-layer analysis reveals developmental trajectory
- Cluster transition patterns show semantic stability
- Outlier analysis identifies semantically complex items

## Future Directions

1. **Contextual Analysis**: Examine how these clusters behave in actual text generation
2. **Cross-Model Comparison**: Compare with other language models (GPT-3, BERT)
3. **Multilingual Extension**: Test if similar patterns emerge across languages
4. **Fine-grained Categories**: Explore subcategories within emergent dimensions
5. **Behavioral Validation**: Test if clustering predicts substitutability in context

## Conclusions

This analysis reveals that GPT-2 develops a semantic organization fundamentally different from traditional linguistic categories. Rather than respecting grammatical boundaries, it groups words by functional and experiential dimensions that better reflect how words are actually used in context. This usage-based organization likely contributes to GPT-2's impressive ability to generate coherent text while maintaining semantic appropriateness.

The findings support theories of distributed semantics and prototype-based categorization, suggesting that neural language models may offer insights into how meaning emerges from language use. The systematic evolution across layers—from surface features to functional organization to prediction-oriented representations—provides a window into how transformers build increasingly abstract representations of language.

Most remarkably, the consistent pairing of words like "feel" and "nice" across hundreds of clustering comparisons suggests that GPT-2 has discovered deep semantic regularities that transcend our traditional grammatical categories, pointing toward a more psychologically real organization of the mental lexicon.