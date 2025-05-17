# Concept Fragmentation in Neural Networks: Analysis and Visualization

This repository contains the code implementation and resources for the paper: **"Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models"**.

## Overview

We introduce a framework to quantify and interpret concept fragmentation in neural networks. Concept fragmentation occurs when datapoints of the same class are scattered across disjoint regions in the latent space, complicating interpretability and potentially revealing bias.

Our approach combines:
- **Quantitative metrics** (cluster entropy, subspace angles, intra-class pairwise distance)
- **Trajectory visualizations** to track activation patterns
- **LLM-based narrative synthesis** from computed archetype paths

## Key Features

- Implementation of multiple fragmentation metrics
- Activation capturing and visualization tools
- Archetype path computation and analysis
- LLM integration for interpretive narratives
- Titanic passenger dataset case study
- Extensibility to large language models

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
