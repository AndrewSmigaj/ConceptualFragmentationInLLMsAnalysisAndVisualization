# GPT-2 Archetypal Path Analysis: Comprehensive Guide

This guide provides a complete reference for using Archetypal Path Analysis (APA) with GPT-2 transformer models, combining theoretical foundations with practical implementation.

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Advanced Analysis](#advanced-analysis)
6. [Visualization Integration](#visualization-integration)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

## Overview

GPT-2 Archetypal Path Analysis extends the APA framework to transformer architectures, enabling interpretable analysis of concept flow through GPT-2 models. This implementation provides:

- **Token-aware clustering** that respects sequence structure
- **Attention-weighted path analysis** using multi-head attention patterns
- **Cross-layer metrics** optimized for transformer architectures
- **LLM-powered narrative generation** for analysis interpretation
- **Interactive visualizations** for attention and path exploration

### Key Features

- Support for all GPT-2 model sizes (117M to 1.5B parameters)
- Sliding window analysis for computational efficiency
- Attention pattern integration with activation analysis
- Bias detection through attention pattern analysis
- Automated report generation with visual summaries

## Theoretical Foundation

### Attention-Weighted Path Analysis

Traditional APA analyzes activation paths through network layers. For transformers, we extend this with attention-weighted clustering:

```
C_att(i,j) = Σ_h A_h(i,j) * ||a_i - a_j||²
```

Where:
- `A_h(i,j)` is the attention weight from token i to token j in head h
- `a_i, a_j` are activation vectors for tokens i and j
- The sum is over all attention heads

### Multi-Head Analysis

GPT-2's multi-head attention requires specialized handling:

```
Path_mh(t) = {C_0^h, C_1^h, ..., C_L^h | h ∈ [1, H]}
```

Where each token t has H parallel paths through L layers, one for each attention head.

### Cross-Layer Fragmentation Metrics

For transformer models, we compute layer-specific fragmentation:

```
F_attention(l) = Σ_h (1 - S_cosine(C_l^h, C_{l+1}^h))
F_residual(l) = 1 - S_cosine(R_l, R_{l+1})
```

Where `S_cosine` is cosine similarity between cluster centroids.

## Architecture

### Core Components

```
GPT2ActivationExtractor
├── Model Loading & Configuration
├── Activation Extraction Pipeline
├── Attention Pattern Capture
├── Token Metadata Management
└── Sliding Window Processing

GPT2PathAnalyzer
├── Attention-Weighted Clustering
├── Cross-Layer Metrics
├── Path Fragmentation Analysis
└── Bias Detection

Visualization Pipeline
├── Token Path Visualization
├── Attention Pattern Heatmaps
├── Cross-Layer Flow Diagrams
└── Interactive Dashboard
```

### Data Flow

1. **Input Processing**: Text → Tokens → GPT-2 Model
2. **Activation Extraction**: Model Layers → Activation Tensors + Attention Patterns
3. **APA Analysis**: Activations → Clusters → Paths → Metrics
4. **Visualization**: Analysis Results → Interactive Plots
5. **Narrative Generation**: Analysis + Context → LLM-Generated Explanations

## Quick Start

### Basic Analysis Workflow

```python
from concept_fragmentation.analysis.gpt2_model_adapter import GPT2ActivationExtractor
from concept_fragmentation.analysis.gpt2_path_analysis import GPT2PathAnalyzer
from visualization.gpt2_token_tab import create_gpt2_visualization

# 1. Initialize extractor
extractor = GPT2ActivationExtractor(model_type="gpt2")

# 2. Extract activations
text = "The quick brown fox jumps over the lazy dog."
analysis_data = extractor.extract_full_analysis(text)

# 3. Analyze paths
analyzer = GPT2PathAnalyzer()
results = analyzer.analyze_paths(analysis_data)

# 4. Visualize
fig = create_gpt2_visualization(results)
fig.show()
```

### Command Line Interface

```bash
# Run complete analysis
python -m concept_fragmentation.analysis.gpt2_adapter_example \
    --model_type gpt2-medium \
    --input_text "Your analysis text here" \
    --output_dir ./results \
    --include_attention \
    --generate_report

# Batch processing
python -m concept_fragmentation.analysis.gpt2_batch_processor \
    --input_file texts.jsonl \
    --output_dir ./batch_results \
    --model_type gpt2-large
```

## Advanced Analysis

### Attention Pattern Analysis

```python
# Extract attention patterns with activations
extractor = GPT2ActivationExtractor(
    model_type="gpt2-medium",
    include_attention=True,
    attention_head_analysis=True
)

# Analyze attention specialization
attention_data = extractor.extract_attention_patterns(text)
head_specialization = analyzer.analyze_attention_specialization(attention_data)

# Detect attention biases
bias_patterns = analyzer.detect_attention_biases(attention_data, text)
```

### Cross-Layer Interaction Analysis

```python
# Analyze information flow between layers
interaction_matrix = analyzer.compute_cross_layer_interactions(results)

# Identify critical layers for concept formation
critical_layers = analyzer.identify_critical_layers(interaction_matrix)

# Track concept emergence across layers
concept_emergence = analyzer.track_concept_emergence(results, target_concepts)
```

### Sliding Window Optimization

```python
# Use sliding windows for long sequences
config = GPT2ActivationConfig(
    window_size=3,
    stride=1,
    overlap_handling="average"
)

extractor = GPT2ActivationExtractor(config=config)
windowed_results = extractor.extract_windowed_analysis(long_text)

# Merge window results
merged_analysis = analyzer.merge_windowed_results(windowed_results)
```

## Visualization Integration

### Dashboard Integration

```python
from visualization.dash_app import create_gpt2_dashboard

# Create interactive dashboard
app = create_gpt2_dashboard(analysis_results)
app.run_server(debug=True, port=8050)
```

### Custom Visualizations

```python
from visualization.gpt2_token_sankey import create_token_sankey
from visualization.cross_layer_viz import plot_attention_flow

# Create Sankey diagram for token paths
sankey_fig = create_token_sankey(results, focus_token="fox")

# Plot attention flow between layers
flow_fig = plot_attention_flow(attention_data, layer_range=(6, 9))
```

### Export Options

```python
# Export high-resolution figures
analyzer.export_visualizations(
    results,
    output_dir="./figures",
    formats=["png", "svg", "pdf"],
    dpi=300
)

# Generate automated report
report = analyzer.generate_analysis_report(
    results,
    include_figures=True,
    narrative_style="technical"
)
```

## Performance Optimization

### Memory Management

```python
# For large models, use gradient accumulation
config = GPT2ActivationConfig(
    gradient_accumulation=True,
    activation_offload="disk",
    memory_limit="8GB"
)

# Process in batches
batch_processor = GPT2BatchProcessor(config)
results = batch_processor.process_batch(text_list, batch_size=4)
```

### GPU Optimization

```python
# Multi-GPU support
config = GPT2ActivationConfig(
    device_map="auto",
    torch_dtype="float16",
    low_cpu_mem_usage=True
)

# Mixed precision analysis
with torch.cuda.amp.autocast():
    results = extractor.extract_full_analysis(text)
```

### Caching Strategy

```python
# Enable intelligent caching
extractor.enable_caching(
    cache_dir="./cache",
    cache_activations=True,
    cache_attention=True,
    max_cache_size="5GB"
)

# Precompute common analyses
extractor.precompute_analyses(common_texts, cache_prefix="common_")
```

## Best Practices

### Text Preprocessing

```python
# Recommended preprocessing
def preprocess_text(text):
    # Ensure proper tokenization boundaries
    text = text.strip()
    
    # Handle special tokens
    text = extractor.tokenizer.clean_up_tokenization(text)
    
    return text

# Batch processing guidelines
def process_batch(texts, max_length=512):
    processed = []
    for text in texts:
        # Truncate to manageable length
        tokens = extractor.tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            text = extractor.tokenizer.decode(tokens)
        
        processed.append(preprocess_text(text))
    
    return processed
```

### Analysis Configuration

```python
# Recommended configurations by model size
CONFIGS = {
    "gpt2": GPT2ActivationConfig(
        context_window=512,
        n_clusters=8,
        window_size=3
    ),
    "gpt2-medium": GPT2ActivationConfig(
        context_window=768,
        n_clusters=12,
        window_size=4
    ),
    "gpt2-large": GPT2ActivationConfig(
        context_window=1024,
        n_clusters=16,
        window_size=5
    )
}
```

### Validation and Quality Control

```python
# Validate analysis results
validator = GPT2AnalysisValidator()

# Check data quality
quality_report = validator.validate_extraction(analysis_data)

# Verify attention patterns
attention_validity = validator.validate_attention_patterns(attention_data)

# Check clustering quality
cluster_metrics = validator.compute_cluster_quality(results)
```

## Troubleshooting

### Common Issues

#### Memory Errors
```python
# Reduce memory usage
config = GPT2ActivationConfig(
    model_type="gpt2",  # Use smaller model
    context_window=256,  # Reduce context
    device="cpu",       # Use CPU if needed
    attention_head_analysis=False  # Disable expensive analysis
)
```

#### Dimension Mismatches
```python
# Enable debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check tensor shapes
extractor.enable_shape_validation(True)
```

#### Attention Pattern Issues
```python
# Validate attention extraction
attention_validator = AttentionValidator()
is_valid = attention_validator.validate_patterns(attention_data)

if not is_valid:
    # Use fallback extraction
    attention_data = extractor.extract_attention_fallback(text)
```

### Performance Issues

#### Slow Analysis
```python
# Profile analysis steps
profiler = GPT2Profiler()
with profiler.profile("full_analysis"):
    results = analyzer.analyze_paths(data)

# Identify bottlenecks
bottlenecks = profiler.get_bottlenecks()
```

#### GPU Memory Issues
```python
# Use gradient checkpointing
config = GPT2ActivationConfig(
    use_gradient_checkpointing=True,
    attention_implementation="flash_attention"
)
```

## API Reference

### GPT2ActivationExtractor

```python
class GPT2ActivationExtractor:
    def __init__(self, model_type: str = "gpt2", config: GPT2ActivationConfig = None)
    
    def extract_full_analysis(self, text: str) -> Dict[str, Any]
    def extract_attention_patterns(self, text: str) -> Dict[str, torch.Tensor]
    def extract_windowed_analysis(self, text: str) -> Dict[str, Any]
    def save_activations(self, data: Dict, output_dir: str) -> str
    def load_activations(self, metadata_file: str) -> Dict[str, Any]
```

### GPT2PathAnalyzer

```python
class GPT2PathAnalyzer:
    def analyze_paths(self, data: Dict[str, Any]) -> GPT2AnalysisResults
    def analyze_attention_specialization(self, attention_data: Dict) -> Dict
    def detect_attention_biases(self, attention_data: Dict, text: str) -> Dict
    def compute_cross_layer_interactions(self, results: GPT2AnalysisResults) -> np.ndarray
    def generate_analysis_report(self, results: GPT2AnalysisResults) -> str
```

### Configuration Classes

```python
@dataclass
class GPT2ActivationConfig:
    model_type: GPT2ModelType = GPT2ModelType.SMALL
    context_window: int = 512
    device: str = "auto"
    include_attention: bool = True
    attention_head_analysis: bool = True
    window_size: int = 3
    stride: int = 1
    n_clusters: int = 10
    cache_dir: Optional[str] = None
```

For detailed implementation guides, see:
- [Theoretical Foundation](gpt2_apa_theoretical_foundation.md)
- [Basic Usage Guide](gpt2_analysis_guide.md)
- [CLI Documentation](gpt2_analysis_cli.md)
- [Integration Guide](../docs/llm_integration_guide.md)