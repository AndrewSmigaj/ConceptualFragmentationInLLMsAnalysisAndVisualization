# Concept Fragmentation in Neural Networks

This repository contains the code and resources for the paper: **"Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models"**.

## Overview

This project explores how neural networks internally represent concepts, focusing on the phenomenon of concept fragmentation - where datapoints of the same class are scattered across disjoint regions in the latent space. We introduce a comprehensive framework combining quantitative metrics, visualizations, and LLM-based narrative synthesis to analyze this phenomenon.

Key features:
- Implementation of fragmentation metrics: **Cluster Entropy**, **Subspace Angle**, and **Intra-Class Pairwise Distance**
- Tools for activation capture and visualization
- Archetype path computation and analysis
- LLM integration for interpretive narratives
- Extensive analysis on the Titanic dataset
- Future extensibility to large language models

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/concept-fragmentation.git
cd concept-fragmentation

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r concept_fragmentation/requirements.txt
```

## Usage

### Demo Workflow

The simplest way to get started is to run the demonstration script:

```bash
# Run the demo workflow on the Titanic dataset
python -m concept_fragmentation.experiments.train --dataset titanic

# Compute fragmentation metrics
python -m concept_fragmentation.experiments.evaluate --experiment_name titanic_baseline

# Generate visualizations
python -m concept_fragmentation.visualization.visualize --experiment_name titanic_baseline --method umap
```

### Metrics Computation

```bash
# Compute all fragmentation metrics
python -m concept_fragmentation.metrics.compute_all --model_path results/titanic_model.pth --dataset titanic

# Compute specific metric
python -m concept_fragmentation.metrics.cluster_entropy --model_path results/titanic_model.pth --dataset titanic
```

### Archetype Path Analysis

```bash
# Compute archetype paths
python -m concept_fragmentation.analysis.compute_paths --model_path results/titanic_model.pth --dataset titanic

# Generate LLM narratives for paths
python -m concept_fragmentation.analysis.generate_narratives --paths_file results/titanic_paths.json
```

### Generating Visualizations

```bash
# Create UMAP projections of layer activations
python -m concept_fragmentation.visualization.visualize --experiment_name titanic_baseline --method umap

# Generate trajectory visualizations
python -m concept_fragmentation.visualization.trajectories --experiment_name titanic_baseline

# Create metric plots
python -m concept_fragmentation.visualization.plot_metrics --metrics_file results/titanic_metrics.json
```

## Directory Structure

```
concept_fragmentation/
├── analysis/                  # Path analysis and LLM narrative tools
├── config.py                  # Configuration settings and hyperparameters
├── data/                      # Dataset loading and preprocessing
├── experiments/               # Training and evaluation scripts
├── hooks/                     # Activation hook utilities
├── metrics/                   # Fragmentation metrics implementations
├── models/                    # Neural network model implementations
├── tests/                     # Unit tests
├── utils/                     # Utility functions and helpers
├── visualization/             # Visualization tools
├── notebooks/                 # Jupyter notebooks for analysis
├── requirements.txt           # Dependencies list
└── README.md                  # This documentation
```

## Metrics

### Cluster Entropy
Measures the degree of fragmentation by computing normalized entropy of class assignments in clusters:
```
H(c) = -∑(k=1 to K) p_k(c) log_2 p_k(c) / log_2 K
```
where p_k(c) is the proportion of class c in cluster k.

### Subspace Angle
Quantifies the separation between class-specific activation subspaces using principal angles between subspaces. Decreasing angles indicate alignment of representations.

### Intra-Class Pairwise Distance (ICPD)
Calculates average Euclidean distance between same-class datapoints, quantifying spatial dispersion within a class.

### K-star (k*)
Determines optimal cluster count per layer using silhouette scores, revealing natural grouping tendencies.

## Archetype Path Analysis

Our framework computes archetype paths by tracking datapoint transitions across clusters layer by layer. These paths are then analyzed using LLMs to generate human-readable narratives describing:

- Cluster behaviors and characteristics
- Fairness implications and potential biases
- Model decision-making processes
- Conceptual roles and relationships

See the paper for detailed examples of archetype paths and their narratives.

## Titanic Case Study

The repository includes a comprehensive case study on the Titanic passenger dataset, demonstrating:

- Quantitative analysis of fragmentation metrics across network layers
- Visualization of activation trajectories
- Computation of dominant archetype paths
- LLM-generated narratives revealing model logic and fairness concerns

## Future Extensions

Our framework can be extended to analyze large language models by:

- Sampling top-k activations across diverse inputs
- Adapting metrics for token embeddings and attention mechanisms
- Enabling self-interpretation where LLMs narrate their own cluster paths
- Applying fragmentation metrics to detect biased representations

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

## License

This project is licensed under the MIT License - see the LICENSE file for details.
