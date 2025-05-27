# Concept Trajectory Analysis (CTA)

This repository introduces **Concept Trajectory Analysis (CTA)** — a method for tracking how neural networks internally organize and evolve concepts across layers. CTA clusters activation vectors per layer, traces datapoint trajectories, and uses large language models to narrate the internal semantic transitions.

## Overview

CTA (formerly Archetypal Path Analysis/APA) provides a principled framework for neural network interpretability that tracks datapoints through clustered activation spaces across layers. Our method reveals how models construct and shift internal boundaries between concepts, providing both mathematical rigor and intuitive understanding of how networks process information.

Our comprehensive approach combines:
- **Principled clustering geometry** with layer-specific labels and mathematical validation
- **Explainable Threshold Similarity (ETS)** for dimension-wise transparent clustering
- **Cross-layer metrics** (centroid similarity, membership overlap, trajectory fragmentation)
- **GPT-2 transformer integration** with attention-weighted path analysis
- **LLM-powered narrative generation** for human-readable explanations
- **Interactive visualizations** for exploring activation patterns and attention flows

## Key Contributions

- 📊 **Concept Fragmentation Metrics**  
  Measures like cluster entropy, silhouette, and subspace angles quantify how class-consistent datapoints split or drift in latent space.

- 🔁 **Latent Path Analysis**  
  By tracing datapoint cluster paths across layers, CTA reveals how models construct and shift internal boundaries between concepts.

- 🧠 **Narrated Cluster Transitions**  
  Uses GPT-style LLMs to convert path data into interpretable stories of how the model "sees" different datapoint subgroups.

- 🧪 **Case Studies**  
  - GPT-2 token behavior (1,228 words, analyzed by part-of-speech and semantics)  
  - Titanic dataset cluster narratives and survival transitions  
  - Heart disease prediction clusters with bias and subgroup drift analysis

## Key Features

### Core CTA Framework
- **Principled clustering** with layer-specific labels and geometric validation
- **Cross-layer metrics** for centroid similarity, membership overlap, and fragmentation
- **ETS clustering** for dimension-wise explainable cluster membership
- **Statistical robustness** testing with multiple seeds and validation measures

### GPT-2 Transformer Extension
- **Full GPT-2 model support** (117M to 1.5B parameters)
- **Attention-weighted path analysis** integrating multi-head attention patterns
- **Token-aware clustering** respecting sequence structure
- **Sliding window analysis** for computational efficiency on deep transformers

### Analysis and Visualization
- **Interactive dashboards** for exploring activation patterns and attention flows
- **Comprehensive visualizations** including Sankey diagrams, attention heatmaps, and trajectory plots
- **LLM-powered narratives** for human-readable explanations of analysis results
- **Automated report generation** combining quantitative metrics with qualitative insights

### Recent Results: GPT-2 5000-Word Experiment

Our latest experiment analyzed 3,262 common English words enriched with WordNet features:
- **Grammatical > Semantic Organization**: Words cluster by part-of-speech rather than semantic similarity
- **Hypernym Analysis**: Despite rich semantic hierarchies (avg 6.1 levels deep), clustering doesn't follow WordNet's IS-A relationships
- **Phase Transitions**: Semantic categories fragment in middle layers (3-8) before partial grammatical convergence
- **Polysemy Effects**: Words with multiple meanings (avg 7.48 senses) show similar trajectories to monosemous words

### Datasets and Applications
- **Titanic and Heart Disease** case studies demonstrating bias detection
- **GPT-2 text analysis** examples with attention pattern interpretation  
- **Enhanced WordNet integration** for semantic hierarchy analysis
- **Extensible framework** for custom datasets and model architectures

## Project Structure

The project is organized into the following main components:

```
concept_fragmentation/
├── analysis/           # Core CTA analysis tools and GPT-2 extensions
│   ├── cluster_paths.py           # Main CTA analysis functions
│   ├── similarity_metrics.py     # Cross-layer similarity calculations
│   ├── gpt2_attention_integration.py  # Attention-weighted analysis
│   └── gpt2_path_extraction.py   # Token path analysis for transformers
├── models/             # Model adapters and interfaces
│   ├── transformer_adapter.py    # GPT-2 adapter with APA methods
│   └── model_interfaces.py       # Unified model interface protocols
├── metrics/            # APA-specific metrics and transformer extensions
├── llm/                # LLM integration for narrative generation
├── hooks/              # Activation extraction hooks for various architectures
├── experiments/        # Training scripts and experimental configurations
├── tests/              # Comprehensive test suite
└── utils/              # Utility functions for path and data processing

visualization/          # Interactive visualization and dashboard components
├── dash_app.py         # Main dashboard application
├── gpt2_token_tab.py   # GPT-2 specific visualization components
└── path_metrics_tab.py # CTA metrics visualization

docs/                   # Comprehensive documentation
├── gpt2_analysis_guide.md         # GPT-2 usage guide
├── gpt2_apa_theoretical_foundation.md  # Mathematical foundations
└── integration_tests_guide.md     # Testing and validation guide
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

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for GPT-2 analysis)

### Core Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn plotly dash
```

### For GPT-2 Analysis

```bash
pip install transformers tokenizers
```

### For LLM Integration

```bash
pip install openai anthropic google-generativeai
```

### Development Installation

```bash
git clone https://github.com/yourusername/ConceptualFragmentationInLLMsAnalysisAndVisualization.git
cd ConceptualFragmentationInLLMsAnalysisAndVisualization
pip install -e .
```

## Usage

### Quick Start: Traditional CTA Analysis

For traditional neural network analysis on tabular data:

```python
from concept_fragmentation.analysis.cluster_paths import analyze_layer_paths
from concept_fragmentation.analysis.similarity_metrics import calculate_cross_layer_similarity

# Load your trained model and data
# Extract activations from each layer
# activations = {"layer_0": ..., "layer_1": ..., "layer_2": ...}

# Run CTA analysis
results = analyze_layer_paths(activations, n_clusters=10)

# Calculate cross-layer relationships
similarity_matrix = calculate_cross_layer_similarity(results)
```

### Quick Start: GPT-2 Analysis

For transformer analysis with attention integration:

```python
from concept_fragmentation.models.transformer_adapter import GPT2Adapter
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2", output_hidden_states=True, output_attentions=True)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
adapter = GPT2Adapter(model, tokenizer=tokenizer)

# Analyze text with sliding windows
text = "The quick brown fox jumps over the lazy dog."
windows = adapter.extract_activations_for_windows(text, window_size=3)

# Get attention patterns
attention_patterns = adapter.get_attention_patterns(text)

# Format for CTA analysis
cta_activations = adapter.get_apa_formatted_activations(text)  # Method name preserved for compatibility
```

### Running the Interactive Dashboard

To launch the visualization dashboard:

```bash
python visualization/dash_app.py
```

Navigate to http://127.0.0.1:8050/ to explore:
- **Path Fragmentation**: Traditional CTA analysis with trajectory visualization
- **GPT-2 Token Analysis**: Transformer-specific analysis with attention patterns
- **LLM Integration**: AI-powered explanations of analysis results

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

## Documentation

### Core CTA Framework
- [Theoretical Foundation](docs/gpt2_apa_theoretical_foundation.md) - Mathematical foundations and principles
- [Cross-Layer Metrics](docs/transformer_cross_layer.md) - Detailed metrics documentation
- [Integration Tests Guide](docs/integration_tests_guide.md) - Testing and validation

### GPT-2 Extensions
- [GPT-2 Analysis Guide](docs/gpt2_analysis_guide.md) - Comprehensive usage guide for transformers
- [GPT-2 Architecture Overview](docs/gpt2_apa_architecture.md) - Implementation architecture
- [Attention Integration Guide](docs/gpt2_attention_interpretation_guide.md) - Working with attention patterns

### LLM Integration
- [LLM Integration Guide](docs/llm_integration_guide.md) - User guide for AI-powered analysis
- [LLM Implementation Details](docs/llm_implementation_details.md) - Technical details for developers

### Examples and Tutorials
- [GPT-2 Analysis Tutorials](docs/gpt2_analysis_tutorials.md) - Step-by-step tutorials
- [Command Line Interface](docs/gpt2_analysis_cli.md) - CLI usage guide

## Project Philosophy

This work was developed through human-AI collaboration: a system of specialized AI agents—coders, critics, narrators—operating under direction of a single human researcher. It reflects a growing vision for interpretability driven by both creative insight and rigorous internal analysis.

CTA is part of the broader **Discordant Colony Optimization** initiative, which explores how divergent agents can surface hidden patterns, test assumptions, and guide paradigm shifts in machine learning understanding.

## Repository Contents

- `arxiv_submission/` — Paper source and figures ready for arXiv submission  
  - `main.tex` — Paper source  
  - `main.bbl` — Compiled bibliography  
  - `sections/` — All major paper sections, including case studies and generated LLM narratives  
  - `figures/` — Diagrams used in the paper (e.g., GPT-2 Sankey flows)
- `experiments/` — All experimental code and results
  - `gpt2/semantic_subtypes/` — 5000-word WordNet experiment
  - `heart_disease/` — Medical AI bias detection
  - `titanic/` — Classic ML interpretability

## Citation & Contact

Author: **Andrew Smigaj**  
Email: `smigaja@gmail.com`  
Status: Preprint-ready, seeking collaboration or institutional support for further research.

If you use or extend this method, attribution is appreciated:

```bibtex
@article{concept_trajectory_analysis_2025,
    title={How Neural Networks Organize Concepts: Introducing Concept Trajectory Analysis for Deep Learning Interpretability},
    author={Andrew Smigaj and Claude Anthropic and Grok xAI},
    journal={arXiv preprint},
    year={2025}
}
```

## Contributing

Contributions are welcome! Please see our contribution guidelines and submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.