# Configuration Directory

Centralized configuration management for all experiments and components.

## Structure

- `experiments/` - Experiment-specific configurations
- `models/` - Model architecture and hyperparameter configs
- `visualization/` - Visualization settings and styles

## Configuration Format

We use YAML for configuration files. Example:

```yaml
# configs/experiments/gpt2_semantic_subtypes.yaml
experiment:
  name: gpt2_semantic_subtypes
  description: Analyze 1,228 single-token words in GPT-2
  
model:
  name: gpt2
  variant: base  # base, medium, large, xl
  
data:
  word_lists:
    - action_verbs.txt
    - stative_verbs.txt
    - concrete_nouns.txt
    # ... etc
  
clustering:
  method: kmeans
  k_selection: elbow
  k_range: [2, 3, 4, 5, 10]
  
analysis:
  windows:
    early: [0, 1, 2, 3]
    middle: [4, 5, 6, 7, 8]
    late: [9, 10, 11, 12]
  
output:
  dir: results/gpt2/semantic_subtypes/
  save_activations: false
  save_trajectories: true
```

## Usage

```python
from concept_fragmentation.config import load_config

config = load_config('configs/experiments/gpt2_bigrams.yaml')
```

## Best Practices

1. Version control all configs
2. Use environment variables for sensitive data
3. Document all parameters
4. Provide sensible defaults
5. Validate configs before running experiments