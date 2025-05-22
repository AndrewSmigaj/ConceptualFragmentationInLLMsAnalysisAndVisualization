# Neural Network Concept Fragmentation Visualization

This module provides tools for visualizing concept fragmentation in neural networks by creating 3D trajectory visualizations that show how samples move through the network layers, along with cross-layer metrics visualizations.

## Features

- **3D Embeddings**: Reduce dimensionality of layer activations using UMAP
- **Multi-panel Plots**: Visualize trajectories across multiple layers simultaneously
- **Comparison Views**: Compare baseline and regularized models side-by-side
- **Interactive Exploration**: Rotate, zoom, and filter visualizations
- **Static Exports**: Generate publication-quality PDF/SVG figures
- **Cross-Layer Metrics**: Visualize relationships between clusters across layers

### Cross-Layer Metrics

The dashboard includes a dedicated tab for cross-layer metrics with the following visualizations:

1. **Centroid Similarity Heatmaps**: Shows similarity between cluster centroids across different layers
2. **Membership Overlap Flow**: Sankey diagram showing how samples flow between clusters across layers
3. **Trajectory Fragmentation**: Bar chart showing how much samples from the same class are split across clusters in each layer
4. **Inter-Cluster Path Density**: Network graph showing connections between clusters across adjacent layers

### New Paper Visualizations

The repository now includes scripts to generate all visualizations referenced in the Results section of the paper:

1. **Stepped-Layer Plots**: Visualize archetypal paths with layers offset along the y-axis
2. **Transition Matrix Heatmaps**: Display cluster-to-cluster transition probabilities
3. **Cross-Layer Similarity Heatmaps**: Show similarity between cluster centroids across layers
4. **Membership Overlap Sankey Diagrams**: Visualize flow of datapoints between clusters
5. **ETS Threshold Boundaries**: Display the threshold boundaries for explainable clustering

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

For cross-layer metrics visualizations, you need these additional dependencies:
```bash
pip install networkx scikit-learn scipy
```

Or use the safer runner which will install missing dependencies automatically:
```bash
python run_dash_safe.py
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

### Paper Visualizations

Generate all visualizations for the Results section of the paper:

```bash
# Generate all visualizations
python run_visualizations.py

# Generate visualizations for specific datasets and seeds
python run_visualizations.py --datasets titanic heart --seeds 0 1 2

# Generate only static visualizations (no interactive HTML)
python run_visualizations.py --static_only
```

Individual visualization scripts can also be run directly:

```bash
# Generate stepped-layer visualization
python visualization/generate_stepped_layer_viz.py --dataset titanic --seed 0

# Generate Sankey diagram
python visualization/generate_sankey_diagram.py --dataset titanic --seed 0

# Generate ETS threshold boundaries visualization
python visualization/generate_ets_visualization.py --dataset titanic --layer layer2
```

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
- **cross_layer_viz.py**: Cross-layer metrics visualizations
- **run_dash_safe.py**: Safe runner with dependency checks

### Paper Visualization Scripts

- **generate_stepped_layer_viz.py**: Generates stepped-layer visualizations for archetypal paths
- **generate_sankey_diagram.py**: Creates Sankey diagrams of membership overlap
- **generate_ets_visualization.py**: Visualizes ETS threshold boundaries
- **generate_all_visualizations.py**: Runs all visualization scripts
- **run_visualizations.py**: Master script to generate all paper visualizations

## Customization

- **UMAP Parameters**: Adjust `n_neighbors` and `min_dist` to control the embedding
- **Visualization Options**: Control colors, sample counts, and highlighting
- **Output Formats**: Export to HTML (interactive) or PDF/PNG (static)

## Notebook Examples

See the Jupyter notebooks in the `notebooks/` directory for examples and sanity checks.