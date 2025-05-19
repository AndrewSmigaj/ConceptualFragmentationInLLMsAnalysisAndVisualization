# LLM Integration for Concept Fragmentation Analysis

This module provides integration with Large Language Models (LLMs) to enhance the interpretability of neural network activation clusters and paths. It allows you to generate human-readable labels for clusters and narratives for paths through activation space.

## Features

- **Multi-Provider Support**: Works with multiple LLM providers:
  - Grok/xAI
  - Claude/Anthropic
  - GPT/OpenAI
  - Gemini/Google

- **Cluster Labeling**: Generate human-readable labels for activation clusters based on centroid features.

- **Path Narratives**: Generate explanations for paths through activation space, incorporating:
  - Fragmentation scores
  - Convergent points
  - Demographic information

- **Caching**: Response caching to reduce API costs and improve performance.

- **Dashboard Integration**: Interactive UI in the visualization dashboard.

## Getting Started

### Testing with Mock Data

For testing the LLM integration without API keys, we've provided mock results:

```bash
# The mock data is available at:
# - /data/llm/titanic_seed_0_mock.json
# - /results/llm/titanic_seed_0_mock.json

# Run the dashboard to see the LLM integration with mock data
python visualization/main.py
```

The mock data contains synthetic cluster labels and path narratives, allowing you to test the dashboard functionality without needing actual LLM API access.

### API Keys

To use the LLM integration, you'll need API keys for at least one provider. Set them up in one of these ways:

1. Create an `api_keys.py` file in the `concept_fragmentation/llm` directory:

```python
# OpenAI/Grok API credentials
OPENAI_KEY = "your-openai-key"
OPENAI_API_BASE = "https://api.grok.meta.com/v1"  # Optional for Grok compatibility

# xAI API credentials
XAI_API_KEY = "your-xai-key"

# Google Gemini API credentials
GEMINI_API_KEY = "your-gemini-key"
```

2. Set environment variables:
   - `OPENAI_API_KEY`
   - `XAI_API_KEY` or `GROK_API_KEY`
   - `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY`
   - `GEMINI_API_KEY` or `GOOGLE_API_KEY`

### Command-Line Usage

Run the demo script to generate cluster labels and path narratives:

```bash
cd concept_fragmentation/llm
python demo.py --dataset titanic --provider grok --output analysis_results.json
```

Options:
- `--dataset`: Dataset name (default: "titanic")
- `--seed`: Random seed (default: 0)
- `--provider`: LLM provider ("openai", "claude", "grok", "gemini")
- `--model`: Model name (default: provider-specific default)
- `--output`: Output file path
- `--no-cache`: Disable caching of LLM responses

### Dashboard Integration

The LLM integration is available in the Neural Network Trajectory Explorer dashboard. To use it:

1. Navigate to the "LLM Analysis" tab
2. Select an LLM provider
3. Choose analysis types (cluster labels, path narratives)
4. Click "Generate LLM Analysis"

## API Usage

### Direct Usage in Python

```python
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Create analyzer
analyzer = ClusterAnalysis(
    provider="grok",  # Use "openai", "claude", "grok", or "gemini"
    model="default",  # Uses provider's default, or specify a model
    use_cache=True    # Cache responses to disk
)

# Generate cluster labels
cluster_labels = analyzer.label_clusters_sync(
    cluster_centroids={
        "layer1C0": centroid1,
        "layer2C1": centroid2,
        # ...
    }
)

# Generate path narratives
path_narratives = analyzer.generate_path_narratives_sync(
    paths={
        0: ["layer1C0", "layer2C1", "layer3C2"],
        1: ["layer1C1", "layer2C2", "layer3C0"],
        # ...
    },
    cluster_labels=cluster_labels,
    cluster_centroids=centroids,
    fragmentation_scores={0: 0.8, 1: 0.2},
    # ...
)
```

## Implementation Details

### Architecture

- `analysis.py`: High-level API for LLM-based analysis
- `client.py`: Abstract base client class for LLM providers
- `responses.py`: Standardized response format
- `factory.py`: Factory for creating provider-specific clients
- Provider implementations:
  - `openai_client.py`: OpenAI/GPT client
  - `claude.py`: Anthropic/Claude client
  - `grok.py`: Grok/xAI client
  - `gemini.py`: Google Gemini client
- `demo.py`: Command-line demo script

### Adding New Providers

To add support for a new LLM provider:

1. Create a new client implementation that extends `BaseLLMClient`
2. Add a response parser in `ResponseParser` class
3. Register the client in `factory.py`

## Notes

- LLM prompts are designed to generate concise, meaningful descriptions.
- Response quality varies by provider and model.
- Caching helps reduce API costs for repeated analyses.

## Recent Updates

- **Centroids Integration**: Properly integrated cluster centroids into the cluster paths data structure
- **Multiple File Support**: LLM demo script now supports different naming conventions for cluster paths files
- **Verification Steps**: Added verification to ensure centroids are included in output files
- **Improved Error Handling**: Better error messages and fallbacks for missing data

## Future Enhancements

- **Enhanced Prompts**: Refined prompts for better quality interpretations
- **Comparative Analysis**: Tools for comparing different paths and understanding their relationships
- **Concept Drift Detection**: Specialized prompts to identify and explain concept drift
- **Report Generation**: Automatic generation of comprehensive analysis reports