# Concept Trajectory Analysis (CTA) Core Library

This is the core implementation of **Concept Trajectory Analysis (CTA)** - a method for tracking how neural networks organize concepts by following their paths through clustered activation spaces across layers.

## Overview

CTA provides a comprehensive framework for understanding how neural networks internally represent and transform concepts. The library includes:

- **Activation extraction** from any neural network layer
- **Clustering algorithms** including K-means and Explainable Threshold Similarity (ETS)
- **Trajectory tracking** through clustered activation spaces
- **Cross-layer metrics** for quantifying concept evolution
- **LLM integration** for generating interpretable explanations
- **Unified visualization** components for Sankey diagrams and trajectories

## Installation

```bash
# From the repository root
pip install -e .

# Or just install dependencies
pip install -r concept_fragmentation/requirements.txt
```

## Core Components

### 1. Activation Module (`activation/`)
- `collector.py` - Extracts activations from neural networks
- `processor.py` - Processes and normalizes activations
- `storage.py` - Efficient storage for large activation sets

### 2. Analysis Module (`analysis/`)
- `cluster_paths.py` - Core CTA algorithm for path analysis
- `cross_layer_metrics.py` - Metrics for layer relationships
- `similarity_metrics.py` - Various similarity calculations
- `transformer_metrics.py` - Specialized metrics for transformers

### 3. Clustering Module (`clustering/`)
- `base.py` - Abstract base class for clustering algorithms
- `paths.py` - Path extraction and analysis
- `exceptions.py` - Custom exceptions

### 4. Visualization Module (`visualization/`)
- `sankey.py` - Unified Sankey diagram generator
- `trajectory.py` - Unified trajectory visualizer
- `configs.py` - Visualization configurations

### 5. LLM Module (`llm/`)
- `client.py` - Base LLM client interface
- `factory.py` - Creates LLM clients for different providers
- `analysis.py` - LLM-powered cluster analysis

## Quick Start

### Basic CTA Analysis

```python
from concept_fragmentation.analysis.cluster_paths import analyze_layer_paths
from concept_fragmentation.activation.collector import ActivationCollector

# Collect activations from your model
collector = ActivationCollector(model)
activations = collector.collect(data_loader)

# Run CTA analysis
paths = analyze_layer_paths(
    activations,
    n_clusters=5,
    method='kmeans'
)

# Analyze trajectories
from concept_fragmentation.analysis.cross_layer_metrics import (
    calculate_trajectory_fragmentation
)
fragmentation = calculate_trajectory_fragmentation(paths)
```

### Visualization

```python
from concept_fragmentation.visualization.sankey import SankeyGenerator
from concept_fragmentation.visualization.configs import SankeyConfig

# Create Sankey diagram
config = SankeyConfig(
    color_scheme='gradient',
    show_percentages=True
)
generator = SankeyGenerator(config)
fig = generator.create_figure(paths, window='early')
fig.write_html('sankey_early.html')
```

### LLM Integration

```python
from concept_fragmentation.llm.factory import create_llm_client
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Create LLM client
client = create_llm_client('claude')
analyzer = ClusterAnalysis(client)

# Generate cluster labels
labels = analyzer.label_clusters(centroids)

# Create narratives
narratives = analyzer.generate_narratives(
    paths, labels, fragmentation_scores
)
```

## Architecture

The library follows a modular design with clear separation of concerns:

```
concept_fragmentation/
├── Base Components        # Abstract classes and interfaces
├── Core Algorithms       # CTA implementation
├── Model Adapters       # Support for different architectures
├── Metrics              # Quantitative measures
├── Visualization        # Unified visual components
└── Integration          # LLM and external tools
```

## Key Concepts

### Trajectory Fragmentation
Measures how coherent a concept's path is through the network:
```
F = 1 - (1/L) * Σ(persistence_score)
```

### Cross-Layer Similarity
Quantifies relationships between clusters in adjacent layers:
```
ρ(C_i, C_j) = cosine_similarity(centroid_i, centroid_j)
```

### Window Analysis
Groups layers for phase transition detection:
- Early layers: Surface features
- Middle layers: Semantic transitions
- Late layers: Task-specific organization

## Testing

Run the test suite:
```bash
python -m pytest concept_fragmentation/tests/
```

## Contributing

See the main repository README for contribution guidelines.

## License

MIT License - see LICENSE file for details.