# GPT-2 Archetypal Path Analysis Command-Line Tool

This document provides instructions for using the GPT-2 Archetypal Path Analysis (APA) command-line tool. This tool allows you to analyze GPT-2 transformer models using the APA framework, which helps identify and visualize concept trajectories through neural network layers.

## Overview

The GPT-2 APA tool provides the following functionality:

1. Loading GPT-2 models of different sizes (small, medium, large, XL)
2. Extracting and analyzing activations from text inputs
3. Performing cluster analysis to track concept flow
4. Visualizing token paths through 3-layer windows

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers (Hugging Face)
- NumPy, Matplotlib
- Plotly (for visualizations)

## Installation

The tool is part of the ConceptualFragmentationInLLMs package. No separate installation is required if you have the repository.

## Basic Usage

To run the GPT-2 APA analysis with default settings:

```bash
python run_gpt2_analysis.py --text "Your text to analyze here"
```

This will:
1. Load the small GPT-2 model
2. Extract activations for the provided text
3. Analyze the activations using a 3-layer sliding window
4. Save the results to the output directory

## Command-Line Options

### Input Options

| Option | Description |
|--------|-------------|
| `--input-file FILE` | Path to a text file for analysis |
| `--text TEXT` | Direct text input for analysis |
| `--sample` | Use a sample text for analysis |

### Model Options

| Option | Description |
|--------|-------------|
| `--model MODEL` | GPT-2 model size: gpt2, gpt2-medium, gpt2-large, or gpt2-xl |
| `--device DEVICE` | Computation device: cpu or cuda |
| `--context-window SIZE` | Maximum context window size (default: 512) |

### Analysis Options

| Option | Description |
|--------|-------------|
| `--window-size SIZE` | Size of sliding window (number of consecutive layers, default: 3) |
| `--stride STRIDE` | Stride for sliding window (default: 1) |
| `--n-clusters N` | Maximum number of clusters to use (default: 10) |
| `--seed SEED` | Random seed for reproducibility (default: 42) |

### Visualization Options

| Option | Description |
|--------|-------------|
| `--visualize` | Generate visualizations |
| `--highlight-tokens TOKENS` | Tokens to highlight in visualizations |
| `--min-path-count COUNT` | Minimum token count for paths in Sankey diagram (default: 1) |

### Output Options

| Option | Description |
|--------|-------------|
| `--output-dir DIR` | Output directory for results (default: results/gpt2_apa) |
| `--timestamp TIMESTAMP` | Timestamp to use for output directory (default: current time) |

## Examples

### Analyzing a Text File

```bash
python run_gpt2_analysis.py --input-file data/sample_text.txt --visualize
```

### Using a Larger Model with CUDA

```bash
python run_gpt2_analysis.py --text "Artificial intelligence research has made significant progress in recent years." --model gpt2-medium --device cuda --visualize
```

### Custom Clustering and Window Size

```bash
python run_gpt2_analysis.py --sample --window-size 4 --stride 2 --n-clusters 15 --visualize
```

### Highlighting Specific Tokens

```bash
python run_gpt2_analysis.py --text "Understanding the inner workings of neural networks is essential." --visualize --highlight-tokens "neural" "networks"
```

## Output Structure

The tool creates the following directory structure:

```
output_dir/
├── model_timestamp/
│   ├── activations/
│   │   └── (activation files)
│   ├── clusters/
│   │   └── (cluster analysis files)
│   ├── results/
│   │   └── (analysis results)
│   ├── visualizations/
│   │   └── (visualization files)
│   └── gpt2_apa_results.json (summary file)
```

## Visualization Types

1. **Token Path Sankey Diagram** - Shows how tokens flow through clusters across layers
2. **Token Path Comparison** - Compares paths for specific tokens
3. **Token Path Statistics** - Provides metrics on token path fragmentation

## Integration with Other Tools

The GPT-2 APA command-line tool integrates with the existing APA framework, using:

1. GPT-2 adapter from the transformer integration
2. Cluster analysis functions from the core APA implementation
3. Visualization tools for token path tracing

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors:
- Reduce the context window size
- Use a smaller model
- Use the CPU instead of GPU

### Visualization Dependencies

If visualizations are not available:
- Ensure Plotly is installed
- Check that the visualization dependencies are installed
- Use `pip install plotly` if needed

### Large Texts

For large texts:
- Consider reducing the context window
- Process texts in smaller chunks
- Increase the stride to sample fewer windows

## Advanced Usage

### Combining with Dashboard Visualization

You can load the results from the command-line tool in the dashboard:

1. Run the analysis with `--output-dir visualization/data/gpt2_apa`
2. Start the dashboard with `python visualization/dash_app.py`
3. Navigate to the GPT-2 Token Paths tab
4. Select your analysis results from the dropdown

## Additional Resources

- See the [GPT-2 Analysis Guide](gpt2_analysis_guide.md) for more information
- Refer to the [Archetypal Path Analysis](apa_overview.md) documentation for theoretical background