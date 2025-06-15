# Concept Trajectory Analysis: How Neural Networks Organize Concepts

This repository contains the implementation of **Concept Trajectory Analysis (CTA)**, a novel interpretability method that tracks how neural networks organize concepts by following their paths through clustered activation spaces across layers.

## 📊 Key Findings

Our analysis of GPT-2 with 1,228 single-token words revealed groundbreaking insights:
- **48.5% of words converge to grammatical highways** where nouns—whether animals, objects, or abstracts—travel together
- **Grammatical > Semantic organization**: GPT-2 primarily organizes by part-of-speech rather than meaning (χ² = 95.90, p < 0.0001)
- **Phase transitions**: Semantic categories fragment in middle layers before converging to grammatical structures

## 🚀 What is CTA?

CTA combines geometric clustering with trajectory tracking to make neural organization visible and quantifiable. The method:
- Clusters activation vectors at each layer
- Tracks how data points move between clusters
- Quantifies concept fragmentation and convergence
- Uses LLMs to generate interpretable explanations

## 📚 Paper & Citation

**Title**: How Neural Networks Organize Concepts: Introducing Concept Trajectory Analysis for Deep Learning Interpretability

**Authors**: Andrew Smigaj, Claude Anthropic, Grok xAI

**Status**: Preprint ready for arXiv submission

📄 **[Read the full paper (PDF)](arxiv_submission/main.pdf)**

```bibtex
@article{concept_trajectory_analysis_2025,
    title={How Neural Networks Organize Concepts: Introducing Concept Trajectory Analysis for Deep Learning Interpretability},
    author={Andrew Smigaj and Claude Anthropic and Grok xAI},
    journal={arXiv preprint},
    year={2025}
}
```

## 🔬 Key Experiments

### 1. GPT-2 Language Organization (1,228 words)
- **Finding**: GPT-2 creates "grammatical highways" where words cluster by function, not meaning
- **Method**: Analyzed single-token words across 13 layers with windowed analysis
- **Result**: Discovered phase transition from semantic to grammatical organization

### 2. Medical AI Bias Detection (Heart Disease)
- **Finding**: Exposed demographic stratification in risk pathways
- **Method**: Tracked patient trajectories through prediction layers
- **Result**: Identified male overprediction bias (Path 4: 83% male composition)

### 3. Apple Quality Routing (In Progress)
- **Finding**: Premium varieties (Honeycrisp) often misrouted to juice processing
- **Method**: CTA analysis of quality routing neural network
- **Result**: Economic impact analysis of variety misclassification

## 🛠️ Installation

### Using Virtual Environment (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/ConceptualFragmentationInLLMsAnalysisAndVisualization.git
cd ConceptualFragmentationInLLMsAnalysisAndVisualization

# Activate the existing virtual environment
source venv311/Scripts/activate  # Windows
# or
source venv311/bin/activate      # Linux/Mac

# Install additional dependencies if needed
pip install -r requirements.txt
```

### Core Dependencies
- Python 3.11
- PyTorch, Transformers (for GPT-2 analysis)
- Scikit-learn, NumPy, Pandas
- Plotly, Dash (for visualization)
- OpenAI/Anthropic/Google APIs (for LLM integration)

## 💻 Quick Start

### Basic CTA Analysis
```python
from concept_fragmentation.analysis.cluster_paths import analyze_layer_paths

# Load your model and extract activations
activations = {"layer_0": ..., "layer_1": ..., "layer_2": ...}

# Run CTA analysis
results = analyze_layer_paths(activations, n_clusters=10)
```

### GPT-2 Token Analysis
```python
from experiments.gpt2.semantic_subtypes.gpt2_semantic_subtypes_experiment import GPT2SemanticSubtypesExperiment

# Run the main experiment
experiment = GPT2SemanticSubtypesExperiment()
results = experiment.run_analysis()
```

### Concept MRI Dashboard
```bash
# Launch the Concept MRI tool
cd concept_mri
python app.py
# Navigate to http://127.0.0.1:8050/
```

## 📁 Repository Structure

```
├── arxiv_submission/           # Submitted CTA paper
│   ├── main.tex               # LaTeX source
│   ├── figures/               # All paper figures
│   └── sections/              # Paper sections
│
├── arxiv_apple/               # Apple quality routing paper (in progress)
│   ├── apple_cta_paper_final.md
│   └── figures/              # Apple experiment figures
│
├── concept_mri/               # Web-based analysis tool
│   ├── app.py                # Main application
│   ├── tabs/                 # UI components
│   └── demos/               # Demo models
│
├── concept_fragmentation/      # Core CTA library
│   ├── analysis/              # Analysis algorithms
│   ├── clustering/            # Clustering methods
│   ├── visualization/         # Sankey & trajectory visualizers
│   └── llm/                   # LLM integration
│
├── experiments/               # Active experiments
│   └── apple_variety/        # Apple quality routing experiment
│
└── scripts/                  # Utility scripts
    └── prepare_demo_models/  # Demo model training
```

## 🔍 Key Features

### Core CTA Framework
- **Explainable Threshold Similarity (ETS)**: Transparent clustering with interpretable thresholds
- **Cross-layer metrics**: Centroid similarity, membership overlap, trajectory fragmentation
- **Statistical validation**: Multiple seeds, silhouette scores, purity metrics

### GPT-2 Extensions
- **Attention integration**: Combines activation clustering with attention patterns
- **Windowed analysis**: Identifies phase transitions in deep transformers
- **Token-aware clustering**: Respects linguistic structure

### LLM-Powered Interpretability
- **Automatic cluster labeling**: Human-readable descriptions
- **Path narratives**: Natural language explanations of concept evolution
- **Multi-provider support**: OpenAI, Anthropic, Google, xAI

### Interactive Visualizations
- **Sankey diagrams**: Show concept flow between layers
- **3D trajectories**: Visualize paths through activation space
- **Attention heatmaps**: Display transformer attention patterns

## 🧪 Reproducing Results

### Apple Quality Routing Analysis
```bash
cd experiments/apple_variety
python run_experiment.py --config config_test.yaml
```

### Launch Concept MRI Tool
```bash
cd concept_mri
python app.py
```

### Train Demo Models
```bash
cd scripts/prepare_demo_models
python train_demo_models.py --dataset heart_disease --variant optimal
```

## 📖 Documentation

- [Architecture Overview](ARCHITECTURE.yaml) - Repository structure and design
- [Current Work](CURRENTLY_WORKING_ON.md) - Active development status
- [Claude Instructions](CLAUDE.md) - Development guidelines
- [Concept MRI Guide](concept_mri/RUNNING_THE_APP.md) - Tool usage instructions

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Extending CTA to vision transformers
- Improving clustering algorithms
- Adding new visualization types
- Applying CTA to other domains

## 📧 Contact

**Author**: Andrew Smigaj  
**Email**: smigaja@gmail.com  
**Status**: Seeking collaboration and institutional support

## 🙏 Acknowledgments

This work emerged from human-AI collaboration, demonstrating how AI systems can assist in understanding AI systems. Special thanks to the open-source community for the tools and libraries that made this research possible.

## 📄 License

MIT License - see LICENSE file for details