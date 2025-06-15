# Transformer Cross-Layer Analysis

This document provides detailed information about the transformer cross-layer analysis framework implemented in the `concept_fragmentation.analysis.transformer_cross_layer` module. This framework is designed to analyze relationships and information flow between layers in transformer models, with a focus on understanding concept fragmentation patterns.

## Overview

The transformer cross-layer analysis framework extends the concept fragmentation analysis pipeline to work with transformer-based models like GPT-2. It provides specialized functionality for analyzing attention patterns, token representations, and embedding spaces across different layers of transformer models.

Key features include:
- Layer relationship analysis via similarity metrics
- Attention flow direction analysis
- Token representation trajectory tracking
- Representation evolution analysis
- Embedding space comparisons

## Core Components

### `TransformerCrossLayerAnalyzer`

This is the main class for analyzing cross-layer relationships in transformer models. It provides methods for analyzing various aspects of cross-layer information flow and produces a comprehensive `CrossLayerTransformerAnalysisResult`.

```python
from concept_fragmentation.analysis.transformer_cross_layer import TransformerCrossLayerAnalyzer

analyzer = TransformerCrossLayerAnalyzer(
    cache_dir="./cache",
    random_state=42,
    use_cache=True
)

result = analyzer.analyze_transformer_cross_layer(
    layer_activations=activations_dict,
    attention_data=attention_dict,
    token_ids=token_ids,
    token_strings=token_strings,
    class_labels=class_labels,
    focus_tokens=["the", "a", "it"],
    layer_order=["layer0", "layer1", "layer2", "layer3"]
)
```

### `CrossLayerTransformerAnalysisResult`

This dataclass holds the results from cross-layer transformer analysis, including:
- Layer relationship metrics
- Attention flow metrics
- Token trajectory information
- Representation evolution metrics
- Embedding space comparisons
- Visualization data

```python
# Access the results
summary = result.get_summary()
print(f"Mean layer similarity: {summary['mean_layer_similarity']}")
print(f"Mean token stability: {summary['mean_token_stability']}")
```

## Analysis Methods

### Layer Relationship Analysis

This analysis measures the similarity between layer representations using various metrics like cosine similarity, centered kernel alignment (CKA), or correlation.

```python
relationships = analyzer.analyze_layer_relationships(
    layer_activations=activations_dict,
    similarity_metric="cosine",
    dimensionality_reduction=True,
    n_components=50
)
```

The returned metrics include:
- Similarity matrix between layer pairs
- A graph representation of layer relationships
- Statistics about the activation distributions in each layer

### Attention Flow Analysis

This analysis examines how attention patterns change across layers, measuring entropy differences and pattern consistency between consecutive layers.

```python
flow = analyzer.analyze_attention_flow(
    attention_data=attention_dict,
    layer_order=["layer0", "layer1", "layer2", "layer3"]
)
```

The returned metrics include:
- Attention entropy for each layer
- Entropy differences between consecutive layers
- Attention flow direction (whether attention becomes more focused or dispersed)
- Pattern consistency between layers
- A directed graph representation of attention flow

### Token Trajectory Analysis

This analysis tracks how token representations evolve across layers, measuring stability and semantic shift.

```python
trajectory = analyzer.analyze_token_trajectory(
    layer_activations=activations_dict,
    token_ids=token_ids,
    token_strings=token_strings,
    focus_tokens=["the", "a", "it"],
    layer_order=["layer0", "layer1", "layer2", "layer3"]
)
```

The returned metrics include:
- Token stability between consecutive layers
- Semantic shift between layers
- Average token stability across all tokens
- A graph representation of token trajectories

### Representation Evolution Analysis

This analysis examines how representations evolve across layers, including divergence between consecutive layers and class separation.

```python
evolution = analyzer.analyze_representation_evolution(
    layer_activations=activations_dict,
    class_labels=class_labels,
    layer_order=["layer0", "layer1", "layer2", "layer3"]
)
```

The returned metrics include:
- Representation divergence between consecutive layers
- Class separation in each layer
- Representation sparsity and compactness

### Embedding Comparisons

This analysis compares embedding spaces across layers after dimensionality reduction.

```python
embeddings = analyzer.analyze_embedding_comparisons(
    layer_activations=activations_dict,
    pca_components=3,
    layer_order=["layer0", "layer1", "layer2", "layer3"]
)
```

The returned metrics include:
- PCA-reduced embeddings for each layer
- Similarities between embedding spaces of different layers

## Command-Line Tool

The framework includes a command-line tool for batch analysis of transformer models:

```bash
python -m concept_fragmentation.analysis.analyze_transformer_cross_layer \
    --activation-path /path/to/activations.npy \
    --attention-path /path/to/attention.npy \
    --token-info-path /path/to/token_info.json \
    --class-labels-path /path/to/labels.npy \
    --output-dir ./results \
    --config ./config.json \
    --focus-tokens "the" "and" "of" \
    --verbose
```

The tool generates:
- JSON files with analysis results
- Visualization plots for each analysis type
- A Markdown report summarizing the findings

## Configuration Options

The analysis can be customized with various configuration options:

```json
{
    "similarity_metric": "cosine",  // Options: "cosine", "cka", "correlation"
    "dimensionality_reduction": true,
    "n_components": 50,
    "pca_components": 3,
    "random_state": 42,
    "use_cache": true,
    "verbose": true
}
```

## Integration with Transformer Models

### Data Extraction

To use this framework with transformer models, you need to extract activations and attention patterns from the model. Here's an example with GPT-2:

```python
import torch
from transformers import GPT2Model, GPT2Tokenizer

# Load model and tokenizer
model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize input text
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")

# Extract activations and attention
with torch.no_grad():
    outputs = model(**inputs, output_attentions=True, output_hidden_states=True)

# Get hidden states for each layer
hidden_states = outputs.hidden_states
layer_activations = {f"layer{i}": hidden_states[i+1].numpy() for i in range(len(hidden_states)-1)}

# Get attention patterns for each layer
attentions = outputs.attentions
attention_data = {f"layer{i}": attentions[i].numpy() for i in range(len(attentions))}

# Get token strings
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Run analysis
analyzer = TransformerCrossLayerAnalyzer()
result = analyzer.analyze_transformer_cross_layer(
    layer_activations=layer_activations,
    attention_data=attention_data,
    token_strings=tokens
)
```

### Using the Analysis Results

The analysis results can be used to gain insights into how transformer models process information:

```python
# Get overall summary
summary = result.get_summary()
print(f"Overall consistency in attention patterns: {summary['mean_attention_consistency']}")

# Identify layers with highest class separation
if "class_separation" in result.representation_evolution:
    best_layer = max(
        result.representation_evolution["class_separation"].items(),
        key=lambda x: x[1]
    )
    print(f"Layer with best class separation: {best_layer[0]} ({best_layer[1]:.4f})")

# Find tokens with highest stability
if "token_stability" in result.token_trajectory:
    token_stabilities = {}
    for token, values in result.token_trajectory["token_stability"].items():
        token_stabilities[token] = np.mean(list(values.values()))
    
    stable_tokens = sorted(token_stabilities.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Most stable tokens across layers:")
    for token, stability in stable_tokens:
        print(f"  {token}: {stability:.4f}")
```

## Visualization

The framework includes functionality for generating various visualizations:

1. **Layer Similarity Heatmap**: Shows the similarity between different layers
2. **Attention Flow Direction Network**: Visualizes how attention patterns change across layers
3. **Token Trajectory Heatmap**: Shows the stability of token representations across layers
4. **Class Separation Bar Chart**: Compares class separation across layers
5. **Representation Divergence Network**: Visualizes how representations evolve across layers

These visualizations can be generated using the command-line tool or programmatically:

```python
from concept_fragmentation.analysis.analyze_transformer_cross_layer import generate_plots

plot_paths = generate_plots(result, output_dir="./plots")
```

## Concept Fragmentation Insights

The cross-layer analysis provides several indicators of concept fragmentation in transformer models:

1. **Declining Token Stability**: If token stability decreases significantly in deeper layers, it may indicate fragmentation of token-level concepts.

2. **Attention Pattern Inconsistency**: Low attention pattern consistency between consecutive layers can indicate shifts in the model's focus.

3. **Representation Divergence Spikes**: Sudden increases in representation divergence between specific layers may indicate boundary points where concept representations significantly transform.

4. **Class Separation Non-monotonicity**: If class separation increases and then decreases across layers, it may indicate that concept information is being diffused or fragmented.

5. **Layer Similarity Patterns**: Clusters of similar layers separated by dissimilar transition layers can indicate distinct processing stages with potential fragmentation boundaries.

By analyzing these patterns, researchers can identify where and how concept fragmentation occurs in transformer models, leading to better understanding of model behavior and potential improvements in architecture.

## Performance Considerations

For large transformer models, the analysis can be computationally intensive. Consider the following tips:

- Use the caching functionality to avoid recomputing results
- Apply dimensionality reduction for high-dimensional spaces
- For very large models, analyze a subset of representative layers
- When analyzing long sequences, consider focusing on key tokens of interest
- Use batch processing for analyzing multiple examples

## References

1. Rogers, A., Kovaleva, O., & Rumshisky, A. (2020). A Primer in BERTology: What we know about how BERT works. Transactions of the Association for Computational Linguistics.

2. Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. In Proceedings of ACL.

3. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. In Proceedings of ICML.

4. Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. In Advances in Neural Information Processing Systems.