# Dual Embedding Experiment: Contextual Influence on Ambiguous Token Routing

## Executive Summary

This experiment investigates how contextual information influences the conceptual routing of ambiguous tokens through neural networks. Building on our observation that pronouns get channeled into either functional (determiner) or content (human social) pathways, we propose introducing a controlled "context embedding" alongside ambiguous tokens to study how context guides conceptual decision-making in neural networks.

## Background & Motivation

Our 10k token analysis revealed that pronouns follow two distinct "highways" through the network:
1. **Functional pathway**: Clustering with determiners and grammatical elements
2. **Content pathway**: Clustering with human/social concept words

This bifurcation suggests that networks must "decide" how to process ambiguous tokens. By introducing controlled contextual signals, we can study this decision-making process directly.

## Experimental Design

### Core Architecture

```
Input Layer:
├── Token Embedding (e.g., "they")
└── Context Embedding (e.g., "formal_discourse" vs "casual_narrative")
    ↓
Fusion Layer (concatenation or attention-like mechanism)
    ↓
Hidden Layers (track activations at each)
    ↓
Output Layer
```

### Phase 1: Pronoun Disambiguation

**Tokens**: he, she, they, it, we, you
**Context Types**:
- Syntactic: [+animate], [-animate], [+plural], [-plural]
- Semantic: [+formal], [+narrative], [+technical], [+social]
- Functional: [+subject], [+object], [+possessive]

### Phase 2: Polysemous Words

**Tokens**: bank, run, light, fair, present
**Context Types**:
- Domain indicators: [finance], [nature], [physics], [social]
- Part-of-speech signals: [verb], [noun], [adjective]

### Phase 3: Function Word Ambiguity

**Tokens**: that, which, as, since, while
**Context Types**:
- Grammatical role: [complementizer], [relative], [demonstrative]
- Clause type: [main], [subordinate], [interrogative]

## Methodology

### 1. Training Protocol
- Train multiple networks with dual inputs
- Vary fusion mechanisms:
  - Early fusion (concatenate at input)
  - Mid fusion (merge at middle layers)
  - Late fusion (combine near output)
  - Attention-style gating

### 2. Analysis Metrics

**Primary Metrics**:
- **Cluster Migration Rate**: Probability of context shifting cluster assignment
- **Activation Delta**: ||activations_with_context - activations_without||
- **Path Divergence**: KL divergence between paths with different contexts
- **Context Sensitivity Index**: Mutual information between context and final cluster

**Secondary Metrics**:
- Layer-wise context influence
- Stability of routing across training runs
- Generalization to unseen context-token pairs

### 3. Control Experiments
- Baseline: Random context embeddings (should show no systematic influence)
- Ablation: Remove context at different layers
- Shuffled: Mismatched context-token pairs
- Gradual: Interpolate between contexts

## Expected Outcomes

### Hypothesis 1: Context-Dependent Routing
Context embeddings will systematically shift ambiguous tokens between conceptual pathways, with stronger effects for more ambiguous tokens.

### Hypothesis 2: Layer-Specific Sensitivity
Early layers will show coarse routing decisions, while later layers will show fine-grained, context-sensitive adjustments.

### Hypothesis 3: Pathway Stability
Some tokens will have "preferred" pathways that are minimally affected by context, while others will be highly context-sensitive.

## Implications for LLM Understanding

### Decision Boundaries
- Map the "decision surfaces" where context tips routing one way or another
- Identify critical context features that maximally influence routing

### Interpretability Insights
- Visualize how context gradually shapes processing through layers
- Create "influence maps" showing which context dimensions affect which tokens

### Mechanistic Understanding
- Reveal whether conceptual highways are static infrastructure or dynamic paths
- Understand how networks implement context-sensitive processing without attention

## Technical Implementation

### Dataset Creation
```python
# Example structure
samples = [
    {
        "token": "they",
        "context_type": "animate_plural",
        "context_embedding": [1, 0, 1, 0, ...],
        "expected_pathway": "content"
    },
    {
        "token": "they",
        "context_type": "formal_subject",
        "context_embedding": [0, 1, 0, 1, ...],
        "expected_pathway": "functional"
    }
]
```

### Architecture Modifications
1. Dual input streams with configurable fusion
2. Pathway tracking through clustering at each layer
3. Attention-weight recording (for attention-style fusion)

### Analysis Pipeline
1. Train models with various fusion strategies
2. Extract activations for all token-context pairs
3. Cluster activations and identify pathways
4. Quantify context influence on routing
5. Visualize pathway shifts and decision boundaries

## Extensions & Future Work

### Multi-Context Integration
Study how multiple contextual signals combine to influence routing.

### Temporal Context
Investigate how sequential context (previous tokens) affects routing differently than static context.

### Cross-Model Analysis
Compare routing patterns across different architectures to identify universal vs. architecture-specific patterns.

### Natural Language Contexts
Move from controlled embeddings to natural language context sentences.

## Ethical Considerations

Understanding how context influences conceptual routing could reveal:
- How biases in training data create context-dependent behaviors
- Why models might process identical tokens differently based on surrounding context
- Potential manipulation vectors through carefully crafted contexts

## Success Criteria

1. **Quantifiable Influence**: Demonstrate >30% change in cluster assignments with context
2. **Systematic Patterns**: Show consistent routing changes across similar contexts
3. **Interpretable Mechanisms**: Identify specific layers/neurons responsible for context integration
4. **Generalization**: Effects hold across different architectures and datasets

## Timeline

- **Weeks 1-2**: Implement dual embedding architecture and training pipeline
- **Weeks 3-4**: Run Phase 1 (pronouns) experiments
- **Weeks 5-6**: Analyze results and refine methodology
- **Weeks 7-8**: Run Phases 2-3 and complete analysis
- **Week 9**: Synthesize findings and prepare visualizations

This experiment bridges our understanding of static concept organization with dynamic concept selection, providing crucial insights into how neural networks make conceptual decisions under ambiguity.