# LLM Integration Guide

This document provides a comprehensive guide to the LLM integration in the Conceptual Fragmentation Analysis and Visualization framework. The framework uses Large Language Models (LLMs) to provide interpretable labels and narratives for neural network activations and cluster paths.

## Architecture Overview

The LLM integration consists of two main components:

1. **Core LLM Module** (`concept_fragmentation/llm/`): Handles all provider-specific implementations, API communication, caching, and high-level analysis functions.

2. **Dashboard Integration** (`visualization/llm_tab.py`): Provides a user interface for generating and displaying LLM-based analysis within the interactive dashboard.

## Core LLM Module

The core module is built with flexibility and extensibility in mind, supporting multiple LLM providers through a factory pattern.

### Supported Providers

- **OpenAI (GPT)** - Using the OpenAI API
- **Anthropic (Claude)** - Using the Anthropic API
- **Meta (Grok)** - Using the Meta AI API
- **Google (Gemini)** - Using the Google AI API

### Key Components

- **`factory.py`**: Creates LLM clients based on the specified provider
- **`client.py`**: Base client class with common functionality
- **`analysis.py`**: High-level API for cluster labeling and path narrative generation
- **`cache_manager.py`**: Handles caching of LLM responses for efficiency
- **`batch_processor.py`**: Manages concurrent API requests for better performance
- **`prompt_optimizer.py`**: Optimizes prompts for better results (token efficiency, clarity)

### Setting Up API Keys

API keys can be provided in two ways:

1. **Environment Variables**:
   ```
   # OpenAI
   export OPENAI_API_KEY="your_key_here"
   
   # Claude/Anthropic
   export ANTHROPIC_API_KEY="your_key_here"
   
   # Grok/Meta
   export XAI_API_KEY="your_key_here"
   
   # Gemini/Google
   export GEMINI_API_KEY="your_key_here"
   ```

2. **API Keys Module**:
   Create a file at `concept_fragmentation/llm/api_keys.py` with the following content:
   ```python
   # OpenAI
   OPENAI_KEY = "your_key_here"
   OPENAI_API_BASE = None  # Optional, for alternative endpoints
   
   # Grok/Meta
   XAI_API_KEY = "your_key_here"
   
   # Gemini/Google
   GEMINI_API_KEY = "your_key_here"
   ```

### Basic Usage

#### Creating an Analyzer

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Default (Grok)
analyzer = ClusterAnalysis()

# Specify a provider
analyzer = ClusterAnalysis(provider="claude")

# With additional options
analyzer = ClusterAnalysis(
    provider="openai",
    model="gpt-4o",  # Provider-specific model
    use_cache=True,  # Enable caching (default)
    optimize_prompts=True  # Enable prompt optimization
)
```

#### Labeling Clusters

```python
# Prepare centroids dictionary (cluster_id -> centroid)
centroids = {
    "layer1C0": np.array([0.1, 0.2, -0.3, ...]),
    "layer1C1": np.array([0.4, -0.1, 0.7, ...]),
    # ...
}

# Generate labels
labels = analyzer.label_clusters_sync(centroids)
print(labels)
# {'layer1C0': 'Age-related features', 'layer1C1': 'Gender characteristics', ...}
```

#### Generating Path Narratives

```python
# Define paths (path_id -> list of cluster IDs)
paths = {
    0: ["layer1C0", "layer2C1", "layer3C2"],
    1: ["layer1C1", "layer2C0", "layer3C1"],
    # ...
}

# Generate narratives
narratives = analyzer.generate_path_narratives_sync(
    paths,
    cluster_labels,  # From previous step
    centroids
)
print(narratives[0])
# "This path transitions from age-related features in the first layer to..."
```

### Advanced Features

#### Providing Additional Context

```python
# Add convergence points information
convergent_points = {
    0: [
        ("layer1C0", "layer3C2", 0.85),  # Early cluster, late cluster, similarity
        # ...
    ]
}

# Add fragmentation scores
fragmentation_scores = {
    0: 0.65,  # Path ID -> fragmentation score
    1: 0.23
}

# Add demographic information
demographic_info = {
    0: {
        "sex": {"male": 0.75, "female": 0.25},
        "survived_rate": 0.33
    }
}

# Generate enhanced narratives
narratives = analyzer.generate_path_narratives_sync(
    paths,
    cluster_labels,
    centroids,
    convergent_points=convergent_points,
    fragmentation_scores=fragmentation_scores,
    demographic_info=demographic_info
)
```

#### Cache Management

```python
# Get cache statistics
stats = analyzer.get_cache_stats()
print(stats)

# Clear cache
analyzer.clear_cache()

# Pre-warm cache with commonly used prompts
prompts = [
    ("Generate a label for cluster with features: ...", {"temperature": 0.2}),
    # ...
]
analyzer.prewarm_cache(prompts)
```

## Dashboard Integration

The dashboard provides an interactive interface for the LLM analysis functionality.

### Features

- **Provider Selection**: Choose between Grok, Claude, GPT, or Gemini
- **Analysis Type**: Generate cluster labels, path narratives, or both
- **Path Selection**: Analyze all paths or focus on specific subsets
- **Results Display**: View results in formatted tables and narrative cards
- **Auto-loading**: Automatically load existing results if available

### Using the Dashboard

1. **Setup**: Ensure API keys are configured as described above
2. **Launch**: Start the dashboard with `python -m visualization.main`
3. **Navigate**: Select the "LLM Analysis" tab
4. **Configure**:
   - Choose an LLM provider (based on available API keys)
   - Select analysis types (Cluster Labeling, Path Narratives)
   - Choose path selection criteria (if generating narratives)
5. **Generate**: Click "Generate LLM Analysis"
6. **Explore**: View the generated labels and narratives in their respective tabs

### Integration with Other Tabs

The LLM analysis results can enhance your understanding of the visualizations in other tabs:

- **Trajectory Visualization**: Use cluster labels to better understand what each cluster represents
- **Path Fragmentation**: Use path narratives to interpret the meaning of different paths
- **Similarity Network**: Understand the semantic relationships between clusters

## Best Practices

1. **Start with Grok**: It generally provides good results with lower latency and cost
2. **Enable Caching**: Keep caching enabled to improve performance and reduce API costs
3. **Cluster Labels First**: Generate cluster labels before path narratives for better context
4. **Path Selection**: For initial exploration, use "Top 10 Most Common" paths
5. **Demographic Context**: Include demographic information when available for richer narratives

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure API keys are correctly set
   - Check environment variables or api_keys.py file

2. **Provider Not Available**:
   - The dashboard will only show providers with valid API keys
   - Some providers may require additional dependencies

3. **Slow Performance**:
   - Use caching to improve performance
   - Consider lowering the number of paths analyzed
   - OpenAI and Claude typically have lower latency than other providers

4. **Quality of Results**:
   - Try different providers for comparative analysis
   - Ensure centroids contain meaningful feature information
   - Check that path information includes relevant convergent points

## Extending the LLM Integration

To add support for a new LLM provider:

1. Create a new client implementation in `concept_fragmentation/llm/`
2. Add the provider to the factory registration in `factory.py`
3. Update the provider selection UI in `visualization/llm_tab.py`

## References

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Anthropic API Documentation](https://docs.anthropic.com/claude/reference)
- [Meta AI API Documentation](https://ai.meta.com/llama/api/)
- [Google AI Documentation](https://ai.google.dev/gemini-api/docs)