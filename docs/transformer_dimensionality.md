# Enhanced Dimensionality Reduction for Transformer Models

This document describes the enhanced dimensionality reduction techniques implemented for transformer model activations in the Conceptual Fragmentation Analysis framework.

## Overview

Transformer models like GPT-2 have extremely high-dimensional activation spaces (often 768 to 4096 dimensions or more per token). Traditional dimensionality reduction techniques like PCA can struggle with these spaces due to:

1. The "curse of dimensionality"
2. Complex non-linear relationships in the data
3. Computational and memory constraints
4. Loss of important structure during reduction

Our enhanced approach addresses these challenges with specialized techniques for transformer models.

## Key Features

### 1. Hierarchical Progressive Reduction

For very high-dimensional spaces (>1000 dimensions), we implement a two-stage reduction process:

```
High-dimensional space (4096d) → Intermediate space (200d) → Low-dimensional space (50d)
           PCA/TruncatedSVD              UMAP/t-SNE/Random Projection
```

This preserves more of the structure than direct reduction to a low-dimensional space.

### 2. Adaptive Method Selection

The system automatically selects the best reduction method based on data characteristics:

- **Very high dimensions with few samples**: TruncatedSVD
- **Medium to large datasets with high dimensions**: PCA
- **Very low target dimensions (≤3)**: UMAP
- **Fallback method**: Random Projection

### 3. Efficient Caching System

The module implements a sophisticated caching system that:

- Caches reduction results to disk
- Uses hashing of data statistics to detect changes
- Efficiently handles the same activations with different parameters
- Preserves memory by storing only essential information

### 4. Specialized Pipeline Integration

The `DimensionalityReductionPipelineStage` seamlessly integrates with the existing pipeline architecture, allowing:

- Automatic processing of all layers in a model
- Filtering of layers by regular expressions
- Consistent metadata tracking
- Progress reporting and error handling

## Usage

### Basic Usage

```python
from concept_fragmentation.analysis.transformer_dimensionality import TransformerDimensionalityReducer

# Create reducer
reducer = TransformerDimensionalityReducer(cache_dir="cache", random_state=42)

# Simple reduction
result = reducer.reduce_dimensionality(
    activations=layer_activations,
    n_components=50,
    method="auto"
)

# Access reduced activations
reduced_data = result.reduced_activations

# Check metadata
if result.success:
    print(f"Reduced from {result.original_dim} to {result.reduced_dim} dimensions")
    print(f"Method used: {result.method}")
    if result.explained_variance is not None:
        print(f"Explained variance: {result.explained_variance:.4f}")
```

### Progressive Reduction for Very High Dimensions

```python
result = reducer.progressive_dimensionality_reduction(
    activations=transformer_activations,
    target_dim=50,
    initial_method="pca",
    secondary_method="umap"
)
```

### Pipeline Integration

```python
from concept_fragmentation.analysis.transformer_dimensionality import DimensionalityReductionPipelineStage
from concept_fragmentation.pipeline.pipeline import Pipeline

# Create pipeline with dimensionality reduction stage
pipeline = Pipeline([
    # Other stages...
    DimensionalityReductionPipelineStage(
        n_components=50,
        progressive=True,
        filter_layers="transformer_layer_.*"
    ),
    # More stages...
])

# Run pipeline
result = pipeline.run(input_data)

# Access reduced activations
reduced_activations = result["reduced_activations"]
```

### Command Line Usage

The module includes a command-line tool for batch processing:

```bash
python -m concept_fragmentation.analysis.reduce_dimensions \
    --input activations.pkl \
    --output reduced_activations \
    --components 50 \
    --method auto \
    --progressive \
    --format npz
```

## Implementation Details

### Methods

The module supports the following reduction methods:

1. **PCA**: Linear technique that maximizes variance along principal components
2. **TruncatedSVD**: Optimized for very high-dimensional, sparse data
3. **KernelPCA**: Non-linear extension of PCA using kernel methods
4. **UMAP**: Manifold learning technique preserving both local and global structure
5. **Random Projection**: Computationally efficient, preserves distances approximately
6. **Progressive Combinations**: Two-stage approaches that combine multiple methods

### Fallback Mechanisms

The system includes robust fallback mechanisms:

1. Method unavailability: Falls back to available methods
2. Failure handling: Returns original data with error information
3. Progressive fallback: Returns intermediate results if final reduction fails

### Performance Considerations

- **Memory usage**: Optimized for large transformer activations
- **Computation time**: Caching system minimizes redundant calculations
- **Scalability**: Works with very high-dimensional activations (tested up to 8192 dimensions)
- **Batch processing**: Efficiently handles batched data with sequence dimensions

## Integration with Token-Level Analysis

The module is fully integrated with the `TokenLevelAnalysis` class for analyzing transformer activations:

```python
from concept_fragmentation.analysis.token_analysis import TokenLevelAnalysis

# Create analyzer with enhanced dimensionality reduction
analyzer = TokenLevelAnalysis(max_k=10, use_cache=True)

# Cluster token activations with enhanced reduction
result = analyzer.cluster_token_activations(
    activations=transformer_activations,
    layer_name="transformer_layer_5",
    dimensionality_reduction=True,
    n_components=50
)
```

## Future Work

Planned enhancements include:

1. **Attention-weighted dimensionality reduction**: Using attention patterns to guide reduction
2. **Transformer-specific embedding methods**: Specialized techniques for language model spaces
3. **Layer-adaptive parameter selection**: Automatically selecting parameters based on layer position
4. **Cross-layer consistency methods**: Ensuring reductions are comparable across layers