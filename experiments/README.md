# Experiments Directory

This directory contains all experimental code organized by experiment type.

## Structure

- `gpt2/` - GPT-2 related experiments
  - `pivot/` - Pivot word experiment
  - `pos/` - Part-of-speech experiment  
  - `semantic_subtypes/` - Semantic subtypes experiment (8 categories, 774 words)
  - `shared/` - Shared utilities for GPT-2 experiments
- `heart_disease/` - Heart disease classification experiment
- `titanic/` - Titanic survival prediction experiment

## Running Experiments

Each experiment directory contains its own scripts and data. See individual experiment directories for specific instructions.

### GPT-2 Semantic Subtypes Experiment

```bash
cd gpt2/semantic_subtypes
python gpt2_semantic_subtypes_experiment.py
```

## Import Notes

The experiments use relative imports to access shared utilities. The `__init__.py` file in this directory sets up the Python path for backward compatibility during the transition period.