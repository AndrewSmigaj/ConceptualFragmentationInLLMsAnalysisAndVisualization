# GPT-2 Archetypal Path Analysis Guide

This guide explains how to use the GPT-2 model adapter for Archetypal Path Analysis (APA) to analyze concept fragmentation in GPT-2 models.

## Overview

Archetypal Path Analysis (APA) provides a principled geometry for cluster-based neural network interpretability. This implementation extends APA to GPT-2 transformer models, with a focus on analyzing the flow of concepts through model layers.

The key components in this implementation are:

1. **GPT2Adapter**: Specialized adapter for extracting activations from GPT-2 models with APA-specific methods
2. **Sliding Window Analysis**: Analysis of 3-layer windows to focus on high-fragmentation regions
3. **Token-Aware Clustering**: Clustering that respects token identities in sequence models
4. **Attention-Weighted Fragmentation**: Integration of attention patterns with path metrics

## Installation Requirements

To use the GPT-2 adapter, you need:

```
pip install torch transformers numpy scikit-learn matplotlib
```

## Basic Usage

Here's a minimal example of analyzing a GPT-2 model with APA:

```python
from concept_fragmentation.models.transformer_adapter import GPT2Adapter
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Initialize adapter for GPT-2 small model
adapter = GPT2Adapter(model, tokenizer=tokenizer)

# Extract activations for a text input using sliding windows
text = "The quick brown fox jumps over the lazy dog."
windows = adapter.extract_activations_for_windows(text, window_size=3)

# Analyze activations with APA (in a separate module)
from concept_fragmentation.analysis.cluster_paths import analyze_layer_paths
from concept_fragmentation.analysis.similarity_metrics import calculate_cross_layer_similarity

# Process each window
for window_name, window_data in windows.items():
    activations = window_data["activations"]
    
    # Format activations for APA analysis
    apa_activations = adapter.get_apa_formatted_activations(text, flatten_sequence=True)
    
    # Run APA analysis
    result = analyze_layer_paths(apa_activations, n_clusters=10)
    
    # Calculate cross-layer similarity
    similarity = calculate_cross_layer_similarity(result)
    
    print(f"Window {window_name} analyzed. Shape of similarity matrix: {similarity.shape}")
```

## Command Line Tool

We provide a command-line tool to run GPT-2 APA analysis:

```bash
python -m concept_fragmentation.analysis.gpt2_adapter_example \
    --model_type gpt2-medium \
    --input_text "The quick brown fox jumps over the lazy dog." \
    --window_size 3 \
    --stride 1 \
    --n_clusters 10 \
    --output_dir ./gpt2_apa_results
```

## Advanced Usage

### Customizing Analysis

You can customize the GPT-2 analysis by loading different model sizes and configuring the adapter:

```python
from concept_fragmentation.models.transformer_adapter import GPT2Adapter
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 medium model with custom settings
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained(
    "gpt2-medium",  # Use larger model
    output_hidden_states=True,
    output_attentions=True,
    cache_dir="./model_cache"  # Custom cache directory
)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", cache_dir="./model_cache")

# Move to device and set to eval mode
model.to(device)
model.eval()

# Initialize adapter
adapter = GPT2Adapter(model, tokenizer=tokenizer)
```

### Working with Attention Patterns

To analyze attention patterns integrated with APA:

```python
# Extract attention patterns with corresponding activations
text = "The quick brown fox jumps over the lazy dog."

# Get attention patterns for specific layers
attention_patterns = adapter.get_attention_patterns(text)

# Get token representations
token_representations = adapter.get_token_representations(text)

# Process each layer
for layer_name in attention_patterns:
    layer_attn = attention_patterns[layer_name]
    
    # Shape: [batch_size, n_heads, seq_len, seq_len]
    batch_size, n_heads, seq_len, _ = layer_attn.shape
    
    # Reshape if needed
    reshaped_attn = layer_attn.mean(axis=1)  # Average over heads
    
    print(f"Layer {layer_name} attention shape: {layer_attn.shape}")
```

### Analyzing Windows of Layers

The sliding window approach helps focus on high-fragmentation regions:

```python
# Extract activations for windows with specific stride
windows = adapter.extract_activations_for_windows(
    text,
    window_size=3,  # 3-layer windows
    stride=1,       # Move window by 1 layer at a time
    include_metadata=True  # Include token metadata
)

# Process specific window
window_name = "window_0_2"  # First three layers
window_data = windows[window_name]

# Get token metadata
metadata = window_data["metadata"]
tokens = metadata["tokens"]
token_ids = metadata["token_ids"]

print(f"Tokens in window {window_name}: {tokens}")
```

### Saving and Loading Activations

To save activations for later analysis:

```python
# Extract and save activations
windows = adapter.extract_activations_for_windows(text)
window_data = windows["window_0_2"]
activations = window_data["activations"]
metadata = window_data["metadata"]

# Save activations with metadata
metadata_file = adapter.save_activations_with_metadata(
    activations,
    metadata,
    output_dir="./activations",
    prefix="gpt2_analysis"
)

# The metadata file contains paths to all saved activations
print(f"Saved activations metadata to: {metadata_file}")

# Later, load the activations
loaded_activations, loaded_metadata = GPT2Adapter.load_activations_from_metadata(metadata_file)

print(f"Loaded {len(loaded_activations)} layers:")
for layer_name, layer_data in loaded_activations.items():
    print(f"  {layer_name}: shape {layer_data.shape}")
```

## Integration with Visualization Tools

The GPT-2 APA implementation integrates with the project's visualization tools:

```python
# Analyze with APA
from concept_fragmentation.analysis.cluster_paths import analyze_layer_paths
from concept_fragmentation.analysis.gpt2_path_extraction import extract_token_paths

# Get token-level analysis
windows = adapter.extract_activations_for_windows(text)
window_data = windows["window_0_2"]
activations = window_data["activations"]

# Run APA
apa_result = analyze_layer_paths(activations, n_clusters=10)

# Extract token paths
token_paths = extract_token_paths(apa_result, window_data["metadata"])

# Create visualization
from visualization.cross_layer_viz import plot_token_paths
from visualization.path_fragmentation_tab import create_path_fragmentation_viz

# Plot token paths
plot_token_paths(token_paths, "token_paths.png")

# Create interactive visualization
create_path_fragmentation_viz(apa_result, "path_fragmentation.html")
```

## Troubleshooting

### Dimension Mismatch Errors

If you encounter dimension mismatch errors, enable dimension logging:

```python
from concept_fragmentation.hooks.activation_hooks import set_dimension_logging

# Enable logging of dimensions
set_dimension_logging(True)

# Extract activations
adapter.get_layer_outputs(text)
```

### Memory Issues

For large models or long sequences, you may encounter memory issues. Try:

1. Reducing the context window size
2. Using a smaller model (e.g., GPT-2 small instead of GPT-2 XL)
3. Processing batches of texts separately

```python
# Use smaller model and CPU processing
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",  # Use smallest model
    output_hidden_states=True,
    output_attentions=True
)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Keep on CPU if memory is limited
model.to("cpu")
model.eval()

adapter = GPT2Adapter(model, tokenizer=tokenizer)
```

### Model Loading Issues

If you have trouble loading models, check that you have:

1. Installed the transformers library
2. Have internet access for downloading models
3. Sufficient disk space

You can specify a custom cache directory:

```python
model = GPT2LMHeadModel.from_pretrained(
    "gpt2",
    cache_dir="/path/to/model/cache",
    output_hidden_states=True,
    output_attentions=True
)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="/path/to/model/cache")
```

## Extending the Adapter

The adapter can be extended for custom analyses:

1. Subclass `GPT2Adapter` for custom extraction logic
2. Create new analysis methods that integrate with the existing architecture
3. Add new visualization techniques tailored to your specific analysis needs

## Attention Pattern Integration

### Analyzing Multi-Head Attention

GPT-2's multi-head attention provides rich information about token relationships:

```python
# Extract attention patterns with metadata
adapter = GPT2Adapter(model, tokenizer=tokenizer)
text = "The quick brown fox jumps over the lazy dog."

# Get detailed attention analysis
attention_analysis = adapter.get_attention_patterns(text)

for layer_idx, layer_attention in attention_analysis.items():
    print(f"Layer {layer_idx}:")
    
    # Attention shape: [batch_size, n_heads, seq_len, seq_len]
    batch_size, n_heads, seq_len, _ = layer_attention.shape
    
    # Analyze each head
    for head_idx in range(n_heads):
        head_attention = layer_attention[0, head_idx]  # Remove batch dimension
        
        # Find strongest attention connections
        max_attention = torch.max(head_attention, dim=1)
        print(f"  Head {head_idx}: Max attention values {max_attention.values}")
```

### Attention-Weighted Clustering

Integrate attention patterns with APA clustering:

```python
from concept_fragmentation.analysis.gpt2_attention_integration import attention_weighted_clustering

# Extract both activations and attention
windows = extractor.extract_activations_for_windows(text, include_attention=True)
window_data = windows["window_0_2"]

activations = window_data["activations"]
attention_patterns = window_data["attention"]

# Apply attention-weighted clustering
weighted_clusters = attention_weighted_clustering(
    activations=activations,
    attention_patterns=attention_patterns,
    n_clusters=10,
    attention_weight=0.5  # Balance between activation and attention similarity
)

print(f"Attention-weighted clusters: {weighted_clusters.keys()}")
```

## Cross-Layer Analysis

### Tracking Information Flow

Analyze how information flows between transformer layers:

```python
from concept_fragmentation.analysis.transformer_cross_layer import analyze_cross_layer_flow

# Extract full model activations
full_activations = extractor.get_all_layer_activations(text)

# Analyze cross-layer information flow
flow_analysis = analyze_cross_layer_flow(
    activations=full_activations,
    method="attention_weighted",
    window_size=3
)

# Visualize flow patterns
from visualization.cross_layer_viz import plot_information_flow
plot_information_flow(flow_analysis, save_path="cross_layer_flow.png")
```

### Layer-Specific Metrics

Compute metrics specific to transformer architectures:

```python
from concept_fragmentation.metrics.transformer_metrics import compute_transformer_metrics

# Get comprehensive transformer metrics
metrics = compute_transformer_metrics(
    activations=full_activations,
    attention_patterns=attention_analysis,
    include_attention_entropy=True,
    include_layer_similarity=True
)

print("Transformer-specific metrics:")
for metric_name, metric_value in metrics.items():
    print(f"  {metric_name}: {metric_value}")
```

## Bias Detection and Analysis

### Attention-Based Bias Detection

Use attention patterns to identify potential biases:

```python
from concept_fragmentation.analysis.gpt2_attention_integration import detect_attention_biases

# Analyze bias in attention patterns
bias_analysis = detect_attention_biases(
    text=text,
    attention_patterns=attention_patterns,
    token_metadata=window_data["metadata"],
    bias_types=["gender", "demographic", "sentiment"]
)

print("Detected attention biases:")
for bias_type, bias_score in bias_analysis.items():
    print(f"  {bias_type}: {bias_score}")
```

### Token-Level Bias Analysis

Examine bias at the individual token level:

```python
# Analyze token-specific attention patterns
token_bias_scores = {}
tokens = window_data["metadata"]["tokens"]

for i, token in enumerate(tokens):
    # Get attention focused on this token
    token_attention = attention_patterns[:, :, :, i]  # All attention TO token i
    
    # Compute bias metrics for this token
    token_bias = compute_token_bias_metrics(token_attention, token)
    token_bias_scores[token] = token_bias

print("Token-level bias analysis:")
for token, bias_score in sorted(token_bias_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  '{token}': {bias_score:.3f}")
```

## LLM Integration for Narrative Generation

### Automated Analysis Narratives

Generate natural language explanations of analysis results:

```python
from concept_fragmentation.llm.analysis import generate_gpt2_analysis_narrative

# Create narrative from analysis results
narrative = generate_gpt2_analysis_narrative(
    analysis_results=result,
    attention_patterns=attention_patterns,
    text=text,
    narrative_style="technical",  # Options: "technical", "accessible", "detailed"
    include_recommendations=True
)

print("Analysis Narrative:")
print(narrative)
```

### Custom Prompt Templates

Create specialized prompts for specific analysis types:

```python
from concept_fragmentation.llm.prompt_optimizer import GPT2AnalysisPromptTemplate

# Create custom prompt template
prompt_template = GPT2AnalysisPromptTemplate(
    analysis_type="attention_bias",
    include_technical_details=True,
    focus_areas=["gender_bias", "sentiment_analysis", "token_importance"]
)

# Generate analysis with custom prompt
custom_analysis = generate_gpt2_analysis_narrative(
    analysis_results=result,
    prompt_template=prompt_template
)
```

## Performance Optimization

### Efficient Batch Processing

Process multiple texts efficiently:

```python
from concept_fragmentation.llm.batch_processor import GPT2BatchProcessor

# Initialize batch processor
batch_processor = GPT2BatchProcessor(
    model_type="gpt2-medium",
    batch_size=8,
    max_workers=4
)

# Process batch of texts
texts = [
    "First analysis text...",
    "Second analysis text...",
    "Third analysis text..."
]

batch_results = batch_processor.process_batch(
    texts=texts,
    include_attention=True,
    save_intermediate=True,
    output_dir="./batch_results"
)

print(f"Processed {len(batch_results)} texts successfully")
```

### Memory-Efficient Analysis

Handle large models and long sequences:

```python
# Configure for memory efficiency - use large model with optimizations
model = GPT2LMHeadModel.from_pretrained(
    "gpt2-large",
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto"  # Automatic device placement
)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")

adapter = GPT2Adapter(model, tokenizer=tokenizer)

# Process with sliding windows for memory efficiency
windows = adapter.extract_activations_for_windows(
    long_text,
    window_size=3,
    stride=2  # Larger stride for memory efficiency
)
```

## API Reference

### Core Classes

```python
class GPT2Adapter(TransformerModelAdapter):
    """Main class for extracting activations from GPT-2 models with APA extensions."""
    
    def __init__(self, model: nn.Module, tokenizer: Optional[Any] = None)
    def extract_activations_for_windows(self, inputs: Any, window_size: int = 3, stride: int = 1) -> Dict
    def get_apa_formatted_activations(self, inputs: Any, flatten_sequence: bool = True) -> Dict[str, torch.Tensor]
    def get_attention_patterns(self, inputs: Any, layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]
    def save_activations_with_metadata(self, activations: Dict, metadata: Dict, output_dir: str) -> str
    
    @classmethod
    def load_activations_from_metadata(cls, metadata_filepath: str) -> Tuple[Dict, Dict]
```

### Utility Functions

```python
def attention_weighted_clustering(activations, attention_patterns, n_clusters, attention_weight=0.5)
def detect_attention_biases(text, attention_patterns, token_metadata, bias_types)
def compute_transformer_metrics(activations, attention_patterns, **kwargs)
def generate_gpt2_analysis_narrative(analysis_results, **kwargs)
```

## Related Documentation

- [Comprehensive Guide](gpt2_apa_comprehensive_guide.md) - Complete reference with theoretical background
- [Theoretical Foundation](gpt2_apa_theoretical_foundation.md) - Mathematical foundations
- [CLI Documentation](gpt2_analysis_cli.md) - Command-line interface
- [LLM Integration Guide](../docs/llm_integration_guide.md) - LLM-powered analysis features