# Concept Fragmentation in Neural Networks: Analysis and Visualization

This repository contains the code implementation and resources for the paper: **"Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models"**.

## Overview

We introduce a framework to quantify and interpret concept fragmentation in neural networks. Concept fragmentation occurs when datapoints of the same class are scattered across disjoint regions in the latent space, complicating interpretability and potentially revealing bias.

Our approach combines:
- **Quantitative metrics** (cluster entropy, subspace angles, intra-class pairwise distance)
- **Trajectory visualizations** to track activation patterns
- **LLM-based narrative synthesis** from computed archetype paths
- **Cross-layer metrics** to analyze relationships between representations across network layers

## Key Features

- Implementation of multiple fragmentation metrics
- Activation capturing and visualization tools
- Archetype path computation and analysis
- Cross-layer metrics for analyzing representation relationships
- Interactive dashboard for exploring neural network trajectories
- LLM integration for interpretive narratives
- Titanic passenger dataset case study
- Extensibility to large language models

## Project Structure

The project is organized into the following main components:

```
concept_fragmentation/
├── analysis/           # Analysis tools for evaluating model representations
├── data/               # Data loading and preprocessing utilities
├── experiments/        # Training and experiment scripts
├── hooks/              # Hooks for extracting activations from models
├── llm/                # Large Language Model integration for interpretability
├── metrics/            # Metrics for measuring conceptual fragmentation
├── models/             # Model definitions and utilities
├── notebooks/          # Example and demonstration notebooks
├── tests/              # Test suite
├── utils/              # Utility functions
└── visualization/      # Visualization tools for network trajectories
```

## Key Components

### Explainable Threshold Similarity (ETS)

ETS is a transparent clustering approach that defines clusters based on dimension-wise thresholds, making the cluster definitions directly interpretable in terms of the feature space.

### Cross-Layer Metrics

The framework implements several metrics for analyzing relationships between clusters across different network layers:

1. **Centroid Similarity (ρᶜ)**: Measures similarity between cluster centroids in different layers
2. **Membership Overlap (J)**: Quantifies how datapoints from one cluster map to clusters in another layer
3. **Trajectory Fragmentation (F)**: Measures how coherent or dispersed a datapoint's path is through the network
4. **Inter-Cluster Path Density**: Analyzes higher-order patterns in concept flow between layers

### LLM Integration for Interpretability

Our framework leverages Large Language Models (LLMs) to enhance interpretability:

1. **Automatic Cluster Labeling**: Generates human-readable labels for clusters based on their centroids
2. **Path Narratives**: Creates natural language explanations of how concepts evolve through network layers
3. **Multi-Provider Support**: Works with multiple LLM providers:
   - Grok (Meta AI)
   - Claude (Anthropic)
   - GPT (OpenAI)
   - Gemini (Google)
4. **Integration Features**:
   - Caching for efficiency and cost management
   - Batch processing for improved performance
   - Context-aware prompting incorporating centroids, demographics, and fragmentation metrics

For detailed documentation on the LLM integration, see:
- [LLM Integration Guide](docs/llm_integration_guide.md) - User guide for the LLM features
- [LLM Implementation Details](docs/llm_implementation_details.md) - Technical details for developers

### Visualization Dashboard

The interactive dashboard allows exploration of neural network trajectories and cross-layer relationships:

1. **3D Trajectory Visualization**: Shows how data flows through embedded representation spaces
2. **Cross-Layer Metrics Views**:
   - Centroid similarity heatmaps
   - Membership overlap Sankey diagrams
   - Trajectory fragmentation bar charts
   - Path density network graphs
3. **LLM Integration Tab**:
   - Cluster labeling for human-readable interpretations
   - Path narratives explaining data flow through the network
   - Provider selection and analysis configuration

## Usage

### Running the Dashboard

To launch the interactive visualization dashboard:

```bash
python visualization/main.py
```

This will start a Dash web application. Navigate to the displayed URL (typically http://127.0.0.1:8050/) in your web browser.

### Example Workflow

See `notebooks/demo_workflow.ipynb` for a complete demonstration of the analysis pipeline, from extracting activations to visualizing paths.

### LLM Integration

To generate interpretable explanations of clusters and paths:

```bash
# Set up your API keys
export OPENAI_API_KEY="your_key_here"  # For OpenAI
export ANTHROPIC_API_KEY="your_key_here"  # For Claude
export XAI_API_KEY="your_key_here"  # For Grok
export GEMINI_API_KEY="your_key_here"  # For Gemini

# Generate cluster paths data with centroids
python run_analysis.py cluster_paths --dataset titanic --seed 0 --compute_similarity

# Run LLM analysis directly
python concept_fragmentation/llm/demo.py --dataset titanic --seed 0 --provider grok

# Or use the interactive dashboard
python visualization/main.py
# Then navigate to the LLM Integration tab
```

See [LLM Integration Guide](docs/llm_integration_guide.md) for detailed instructions on using the LLM features.

### Programmatic LLM Analysis

You can also use the LLM integration directly in your code:

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Initialize analyzer with preferred provider
analyzer = ClusterAnalysis(provider="claude")

# Label clusters
labels = analyzer.label_clusters_sync(centroids)

# Generate narratives
narratives = analyzer.generate_path_narratives_sync(
    paths, labels, centroids, 
    fragmentation_scores=frag_scores
)

print(labels)
print(narratives[0])
```

## Getting Started

See the detailed documentation in the `concept_fragmentation` directory for installation instructions, usage examples, and API reference.

## Citation

If you use this code in your research, please cite our paper:

```
@article{
    title={Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models},
    author={Anonymous Submission},
    journal={ArXiv},
    year={2025}
}
```