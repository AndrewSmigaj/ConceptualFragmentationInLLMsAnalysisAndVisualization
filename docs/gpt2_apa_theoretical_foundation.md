# GPT-2 Archetypal Path Analysis: Theoretical Foundation

This document establishes the theoretical foundation for applying Archetypal Path Analysis (APA) to GPT-2 transformer models, connecting the mathematical principles from the core APA framework to the specific challenges and opportunities of analyzing large language models.

## Overview

Archetypal Path Analysis provides a principled approach to understanding how neural networks process information by tracking datapoints (tokens, in the case of GPT-2) through clustered activation spaces across model layers. For transformer models like GPT-2, this approach reveals how linguistic concepts evolve, fragment, and reconverge as they flow through the attention mechanisms and feed-forward layers.

## Theoretical Foundations

### 1. APA Core Principles Applied to GPT-2

#### 1.1 Activation Space Geometry in Transformers

In GPT-2, each layer ℓ produces activations A^ℓ ∈ ℝ^{n×d_ℓ} where n is the sequence length and d_ℓ is the hidden dimension. Unlike feedforward networks where activations represent independent datapoints, transformer activations exhibit:

- **Positional Dependencies**: Each token's representation depends on its position and context
- **Attention-Mediated Interactions**: Token representations are influenced by attention patterns
- **Hierarchical Structure**: Information flows from surface features to abstract concepts across layers

#### 1.2 Token-Aware Clustering

Traditional APA clusters individual datapoints. For GPT-2, we cluster token representations while preserving:

- **Token Identity**: The specific token being represented
- **Positional Information**: Where the token appears in the sequence  
- **Contextual Relationships**: How the token relates to surrounding tokens

This leads to paths π_i = [c_i^1, c_i^2, ..., c_i^L] where each c_i^ℓ represents the cluster assignment for token i at layer ℓ.

#### 1.3 Layer-Specific Cluster Labeling

Following APA principles, we use layer-specific cluster labels (LℓCk) to avoid assuming relationships between clusters across layers. For GPT-2:

- **L0C0**: Cluster 0 in the embedding layer
- **L6C2**: Cluster 2 in transformer layer 6  
- **L11C1**: Cluster 1 in the final transformer layer

Relationships between clusters are validated through geometric similarity rather than assumed from label correspondence.

### 2. GPT-2-Specific Adaptations

#### 2.1 Attention-Weighted Path Analysis

GPT-2's attention mechanisms provide additional information about which tokens influence each other. We incorporate attention patterns into APA through:

**Attention-Weighted Cluster Similarity**:
$$\rho^{att}(C_i^ℓ, C_j^{ℓ'}) = \sum_{h} α_h \cdot \text{sim}(μ_i^ℓ, μ_j^{ℓ'}) \cdot A_{i,j}^{h}$$

where α_h weights different attention heads and A_{i,j}^h represents attention from token i to token j in head h.

**Attention Flow Correlation**:
We measure how attention patterns correlate with cluster path transitions:
$$corr(att, path) = \text{corr}(A_{i,j}^{ℓ}, δ(c_i^ℓ, c_j^{ℓ+1}))$$

where δ indicates whether tokens i and j transition to related clusters.

#### 2.2 Multi-Head Analysis

GPT-2's multi-head attention allows different heads to capture different types of relationships. We analyze:

- **Head Specialization**: Which attention heads correlate with specific types of cluster transitions
- **Head Agreement**: Consistency between different attention heads in supporting cluster paths
- **Head Evolution**: How attention patterns change across layers

#### 2.3 Sliding Window Analysis

Due to GPT-2's depth (12 layers for GPT-2 small), we use sliding window analysis:

- **Window Size**: Typically 3-4 layers to focus on local transformations
- **Overlap**: Windows overlap to capture cross-window transitions
- **Aggregation**: Results across windows are combined for global insights

### 3. Cross-Layer Metrics for GPT-2

#### 3.1 Token Path Fragmentation

Traditional fragmentation measures how dispersed a datapoint's path is. For GPT-2, we extend this to measure:

**Semantic Fragmentation**: How much a token's meaning changes across layers
$$F_{sem}(π_i) = \frac{1}{L-1} \sum_{ℓ=1}^{L-1} d_{sem}(c_i^ℓ, c_i^{ℓ+1})$$

where d_sem measures semantic distance between cluster centroids.

**Attention Fragmentation**: How attention patterns fragment across layers
$$F_{att}(π_i) = \frac{1}{L-1} \sum_{ℓ=1}^{L-1} H(A_i^ℓ)$$

where H is the entropy of token i's attention distribution at layer ℓ.

#### 3.2 Linguistic Concept Evolution

We track how linguistic concepts evolve through the transformer:

**Syntactic → Semantic Transition**: Early layers focus on syntax, later layers on semantics
**Local → Global Context**: Attention patterns evolve from local to global relationships  
**Concrete → Abstract**: Token representations become more abstract in deeper layers

#### 3.3 Cross-Layer Similarity Metrics

**Centroid Similarity with Attention Weighting**:
$$ρ^c_{att}(C_i^ℓ, C_j^{ℓ'}) = \text{sim}(μ_i^ℓ, μ_j^{ℓ'}) \cdot w_{att}(i,j,ℓ,ℓ')$$

where w_att incorporates attention-based weighting.

**Token Trajectory Consistency**:
Measures how consistently tokens follow similar paths:
$$TC(T) = \frac{1}{|T|^2} \sum_{i,j \in T} \text{sim}(π_i, π_j)$$

### 4. Bias Detection in GPT-2

#### 4.1 Demographic Path Analysis

For bias detection, we analyze how tokens associated with different demographic groups follow different paths:

**Group Path Divergence**:
$$D(G_1, G_2) = \frac{1}{L} \sum_{ℓ=1}^L \text{KL}(P(c^ℓ|G_1) || P(c^ℓ|G_2))$$

where P(c^ℓ|G) is the distribution over clusters at layer ℓ for group G.

**Attention Bias Metrics**:
$$B_{att}(G_1, G_2) = \sum_{h,ℓ} |A^{h,ℓ}_{G_1} - A^{h,ℓ}_{G_2}|$$

measuring differences in attention patterns between demographic groups.

#### 4.2 Stereotype Propagation

We track how stereotypical associations propagate through layers:

**Stereotype Amplification**:
$$SA(concept, group) = \frac{\text{sim}(concept, group)^{final}}{\text{sim}(concept, group)^{initial}}$$

measuring whether stereotype associations strengthen through the network.

### 5. LLM-Powered Narrative Generation

#### 5.1 Prompt Engineering for GPT-2 Analysis

We design specialized prompts that incorporate:

- **Cluster Statistics**: Centroid information, cluster sizes, purity measures
- **Path Information**: Token trajectories, transition patterns, fragmentation scores
- **Attention Patterns**: Head specialization, attention flow, correlation with paths
- **Linguistic Context**: Part-of-speech, syntactic roles, semantic categories

#### 5.2 Multi-Scale Narrative Generation

**Token-Level Narratives**: Explain individual token paths
```
"The word 'brilliant' starts in a neutral descriptive cluster (L0C3) but transitions 
through an evaluative cluster (L6C1) before settling in a positive sentiment cluster 
(L11C2), showing how the model processes evaluative meaning."
```

**Sequence-Level Narratives**: Explain overall sequence processing
```
"The sentence shows typical transformer processing: early layers focus on individual 
word meanings, middle layers build syntactic relationships, and final layers integrate 
semantic content for coherent understanding."
```

**Cross-Sequence Comparisons**: Compare processing across different inputs
```
"Sentences with male pronouns consistently route through different attention patterns 
than those with female pronouns, suggesting potential gender bias in the model's 
processing pathways."
```

### 6. Mathematical Validation Framework

#### 6.1 Stability Analysis

We validate GPT-2 path stability through:

**Cross-Initialization Consistency**: Analyze multiple random initializations
**Attention Perturbation Robustness**: Test sensitivity to attention modifications  
**Layer Removal Robustness**: Assess impact of removing intermediate layers

#### 6.2 Statistical Significance Testing

**Path Significance**: Test whether observed paths differ from random baselines
**Attention Correlation**: Validate attention-path correlations against null models
**Bias Detection**: Use permutation tests for demographic group differences

#### 6.3 Interpretability Validation

**Human Agreement Studies**: Validate LLM narratives against human interpretations
**Attention Visualization**: Use attention heatmaps to validate narrative claims
**Ablation Studies**: Remove components to test their contribution to interpretability

## Implementation Considerations

### 1. Computational Efficiency

**Selective Layer Analysis**: Focus on layers showing high fragmentation
**Attention Pruning**: Analyze only the most informative attention heads
**Batch Processing**: Process multiple sequences simultaneously

### 2. Memory Management

**Activation Caching**: Cache intermediate results for repeated analysis
**Gradient Checkpointing**: Use checkpointing for memory-efficient computation
**Streaming Analysis**: Process long sequences in chunks

### 3. Scalability

**Model Size Adaptation**: Adjust analysis complexity for different GPT-2 variants
**Distributed Computing**: Parallelize analysis across multiple GPUs/nodes
**Progressive Analysis**: Start with coarse analysis, refine as needed

## Research Applications

### 1. Model Understanding

- **Architecture Analysis**: Compare different transformer architectures
- **Training Dynamics**: Understand how paths evolve during training
- **Scaling Laws**: Analyze how path complexity scales with model size

### 2. Bias and Fairness

- **Stereotype Detection**: Identify biased processing pathways
- **Fairness Auditing**: Systematic analysis of group-based differences
- **Mitigation Strategies**: Design interventions based on path analysis

### 3. Model Improvement

- **Architecture Design**: Inform better transformer architectures
- **Training Objectives**: Design objectives that encourage fair paths
- **Interpretability Tools**: Create tools for model developers

## Conclusion

The theoretical foundation for GPT-2 Archetypal Path Analysis extends traditional APA principles to address the unique challenges of transformer models. By incorporating attention mechanisms, multi-head analysis, and specialized clustering approaches, we create a framework that can reveal how large language models process linguistic information while maintaining mathematical rigor and interpretability.

This foundation enables systematic analysis of model behavior, bias detection, and the generation of human-understandable explanations of complex transformer computations.