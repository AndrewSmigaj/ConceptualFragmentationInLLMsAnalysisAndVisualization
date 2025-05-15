# Neural Network Concept Fragmentation Visualization

This module provides tools for visualizing concept fragmentation in neural networks by creating 3D trajectory visualizations that show how samples move through the network layers.

## Features

- **3D Embeddings**: Reduce dimensionality of layer activations using UMAP
- **Multi-panel Plots**: Visualize trajectories across multiple layers simultaneously
- **Comparison Views**: Compare baseline and regularized models side-by-side
- **Interactive Exploration**: Rotate, zoom, and filter visualizations
- **Static Exports**: Generate publication-quality PDF/SVG figures

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have access to the activation data at `D:/concept_fragmentation_results/` or update the path in `config.py`.

## Usage

### Command Line Interface

Generate visualizations for one or more datasets:

```bash
# Basic usage
python main.py titanic heart

# With options
python main.py titanic --seeds 0 1 2 --output-dir results/my_visualizations --max-samples 150
```

### Interactive Dashboard

Launch the interactive web interface:

```bash
python dash_app.py
```

Then open your browser to http://127.0.0.1:8050/

### Python API

Use the modules directly in your Python code:

```python
from visualization.data_interface import load_activations, get_best_config, get_baseline_config
from visualization.reducers import Embedder, embed_layer_activations
from visualization.traj_plot import plot_dataset_trajectories, save_figure

# Create an embedder
embedder = Embedder(n_components=3, random_state=42)

# Load and embed activations
dataset = "titanic"
seed = 0
baseline_config = get_baseline_config(dataset)
best_config = get_best_config(dataset)

baseline_embeddings = embed_layer_activations(dataset, baseline_config, seed, embedder=embedder)
best_embeddings = embed_layer_activations(dataset, best_config, seed, embedder=embedder)

# Create visualization
embeddings_dict = {
    "baseline": {seed: baseline_embeddings},
    "regularized": {seed: best_embeddings}
}

fig = plot_dataset_trajectories(dataset, embeddings_dict)
save_figure(fig, "titanic_trajectories.html")
```

## Key Components

- **data_interface.py**: Functions for loading and processing experiment data
- **reducers.py**: UMAP dimensionality reduction with caching
- **traj_plot.py**: 3D visualization using Plotly
- **dash_app.py**: Interactive web dashboard
- **main.py**: Command-line entry point
- **notebooks/**: Example Jupyter notebooks

## Customization

- **UMAP Parameters**: Adjust `n_neighbors` and `min_dist` to control the embedding
- **Visualization Options**: Control colors, sample counts, and highlighting
- **Output Formats**: Export to HTML (interactive) or PDF/PNG (static)

## Notebook Examples

See the Jupyter notebooks in the `notebooks/` directory for examples and sanity checks. 