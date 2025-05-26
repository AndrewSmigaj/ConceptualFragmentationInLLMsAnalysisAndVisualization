# Testing the LLM Integration

This document provides instructions for testing the LLM integration in the Conceptual Fragmentation project.

## Overview

The LLM integration enhances the interpretability of neural network analysis by:
1. Generating human-readable labels for activation clusters
2. Creating narratives for paths through activation space

## Quick Test with Mock Data

We've provided mock LLM analysis results so you can test the functionality without needing API keys:

1. **Start the dashboard**:
   ```bash
   python visualization/main.py
   ```

2. **Navigate to the LLM Tab**:
   - Select "Titanic" dataset and seed "0" at the top
   - Click on the "LLM Integration" tab
   - The dashboard should automatically load the mock results

3. **Explore the Results**:
   - View cluster labels in the "Cluster Labels" tab
   - View path narratives in the "Path Narratives" tab

## Setting Up Real LLM Integration

To use actual LLM APIs instead of mock data:

1. **Set up API keys**:
   Create `concept_fragmentation/llm/api_keys.py` with:
   ```python
   # Choose at least one of these providers
   OPENAI_KEY = "your-openai-key"  # For OpenAI GPT
   OPENAI_API_BASE = "https://api.grok.meta.com/v1"  # For Grok API
   XAI_API_KEY = "your-xai-key"  # For Grok/xAI
   GEMINI_API_KEY = "your-gemini-key"  # For Google Gemini
   ```

2. **Install dependencies**:
   ```bash
   pip install aiohttp requests numpy
   ```

3. **Generate proper cluster paths data**:
   ```bash
   python concept_fragmentation/analysis/cluster_paths.py --dataset titanic --seed 0 --compute_similarity
   ```

4. **Run the dashboard and generate analysis**:
   - Start the dashboard: `python visualization/main.py`
   - Go to the LLM Integration tab
   - Select a provider
   - Click "Generate LLM Analysis"

## Implementation Architecture

- `analysis.py`: High-level API for LLM analysis
- `client.py`: Abstract client base class
- `factory.py`: Factory for creating provider clients
- `responses.py`: Standardized response format
- Provider implementations:
  - `openai_client.py`: OpenAI/GPT + Grok support
  - `claude.py`: Anthropic Claude
  - `grok.py`: Grok native API
  - `gemini.py`: Google Gemini
- `demo.py`: Command-line demo script

## Demo Script Usage

You can also test with the demo script:

```bash
python concept_fragmentation/llm/demo.py --dataset titanic --seed 0_paths_with_centroids --provider grok --output llm_analysis_results.json
```

Options:
- `--dataset`: Dataset name (default: "titanic")
- `--seed`: Random seed (default: 0)
- `--provider`: LLM provider (grok, openai, claude, gemini)
- `--model`: Model name (default: provider's default)
- `--output`: Output file path
- `--no-cache`: Disable caching