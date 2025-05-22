# Transformer-Specific Metrics for Neural Network Analysis

This document describes the transformer-specific metrics implemented for analyzing attention patterns, activation statistics, and other aspects of transformer model behavior.

## Overview

Transformer models like GPT-2 have architectural elements not found in traditional MLPs, including self-attention mechanisms, residual connections, and layer normalization. These elements create unique patterns of information flow that require specialized metrics for analysis.

Our transformer-specific metrics are designed to:

1. Measure information distribution in attention matrices
2. Quantify how attention patterns vary across layers
3. Analyze the importance of different attention heads
4. Measure the stability of representations across perturbations
5. Support the existing archetypal path analysis framework

## Metrics Implementation

### 1. Attention Entropy

The attention entropy metric measures how uniformly attention is distributed across tokens. Higher entropy values indicate more uniform attention, while lower values indicate more focused attention.

```python
from concept_fragmentation.metrics import calculate_attention_entropy

# Attention probabilities shape: [batch_size, n_heads, seq_len, seq_len]
entropy = calculate_attention_entropy(attention_probs)
```

### 2. Attention Sparsity

The attention sparsity metric quantifies how focused or sparse the attention distributions are. Higher sparsity values indicate that attention is concentrated on a smaller number of tokens.

```python
from concept_fragmentation.metrics import calculate_attention_sparsity

# With custom threshold for what's considered "zero"
sparsity = calculate_attention_sparsity(attention_probs, threshold=0.01)
```

### 3. Head Importance

This metric evaluates the importance of each attention head by measuring its influence on the model's outputs.

```python
from concept_fragmentation.metrics import calculate_head_importance

# Calculate importance scores
importance = calculate_head_importance(
    attention_probs,
    outputs,
    token_mask=token_mask
)
```

### 4. Cross-Attention Consistency

This metric measures the consistency of attention patterns across different layers, revealing how similar or different the focus is between layers.

```python
from concept_fragmentation.metrics import analyze_cross_attention_consistency

# Analyze consistency
consistency = analyze_cross_attention_consistency(
    attention_data,
    input_tokens=token_strings,
    focus_tokens=["important", "keywords"]
)
```

### 5. Activation Statistics

This metric computes various statistics about activations, including mean, standard deviation, sparsity, and L2 norm.

```python
from concept_fragmentation.metrics import calculate_activation_statistics

# Compute statistics
stats = calculate_activation_statistics(
    activations,
    token_mask=token_mask,
    per_token=True
)
```

### 6. Activation Sensitivity

This metric measures how sensitive activations are to input perturbations, revealing the stability of representations.

```python
from concept_fragmentation.metrics import calculate_activation_sensitivity

# Compute sensitivity
sensitivity = calculate_activation_sensitivity(
    base_activations,
    perturbed_activations,
    metric="cosine"
)
```

## Comprehensive Analysis with TransformerMetricsCalculator

The `TransformerMetricsCalculator` class provides a unified interface for computing all of these metrics, with built-in caching for efficiency.

```python
from concept_fragmentation.metrics import TransformerMetricsCalculator

# Create calculator
calculator = TransformerMetricsCalculator(use_cache=True)

# Compute multiple metrics
attention_metrics = calculator.compute_attention_metrics(attention_data)
activation_stats = calculator.compute_activation_statistics(activations)
cross_attention = calculator.compute_cross_attention_consistency(attention_data)
sensitivity = calculator.compute_activation_sensitivity(base_acts, perturbed_acts)
```

## Command-Line Analysis Tool

The module includes a command-line tool for analyzing transformer data files:

```bash
python -m concept_fragmentation.metrics.analyze_transformer \
    --input attention_data.pkl \
    --output analysis_results \
    --token-info tokens.json \
    --focus-tokens "important" "keywords" \
    --plot-figures
```

The tool automatically generates:
- JSON files with detailed metrics
- Markdown summary reports
- Visualizations of key metrics
- Cross-layer consistency analyses

## Integration with Archetypal Path Analysis

These transformer-specific metrics integrate with the existing archetypal path analysis framework, enabling:

1. Enhanced clustering of attention patterns
2. Path tracking through attention spaces
3. Comparison of different transformer architectures
4. Analysis of how information flows through the model

Example integration:

```python
from concept_fragmentation.analysis import TokenLevelAnalysis
from concept_fragmentation.metrics import TransformerMetricsCalculator

# Create analysis components
token_analyzer = TokenLevelAnalysis()
metrics_calculator = TransformerMetricsCalculator()

# Analyze token-level patterns
token_clusters = token_analyzer.cluster_token_activations(
    activations=layer_activations,
    layer_name="transformer_layer_5"
)

# Compute metrics on the same data
attention_metrics = metrics_calculator.compute_attention_metrics(
    attention_data=attention_data
)

# Combine analyses
combined_analysis = {
    "token_clusters": token_clusters,
    "attention_metrics": attention_metrics
}
```

## Interpretability Applications

These metrics provide insights into how transformer models process information:

1. **Attention Distribution:** Shows how the model distributes focus across input tokens
2. **Cross-Layer Consistency:** Reveals whether the model maintains consistent focus or shifts attention
3. **Head Specialization:** Identifies specialized attention heads for particular tasks
4. **Processing Stability:** Measures robustness of representations to input variations

These insights help explain model behavior and can guide architectural improvements.

## Future Work

Planned enhancements include:

1. **Multi-head projection analysis:** Studying how different heads project information
2. **Attention flow tracking:** Following attention patterns across layers
3. **Task-specific attention metrics:** Specialized metrics for different NLP tasks
4. **Integration with LLM explanations:** Using LLMs to narrate attention patterns
5. **Attention bias detection:** Identifying systematic biases in attention mechanisms