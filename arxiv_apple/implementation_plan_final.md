# Apple Variety CTA Implementation Plan (Final)

## Architecture Review Summary

After thorough review of the codebase, this plan maximizes reuse of existing components while maintaining good software design principles.

## ✅ Components to Reuse (NO REIMPLEMENTATION NEEDED)

### 1. **Experiment Framework**
- **Use**: `BaseExperiment` from `concept_fragmentation/experiments/base.py`
- **Provides**: setup(), execute(), analyze(), visualize(), cleanup()
- **Benefits**: Logging, result saving, artifact management all handled

### 2. **Activation Collection**
- **Use**: `ActivationCollector` from `concept_fragmentation/activation/collector.py`
- **Features**: Streaming, batching, disk caching, memory efficient
- **Also use**: `ActivationHook` from `concept_fragmentation/hooks/activation_hooks.py`

### 3. **Path Extraction & Analysis**
- **Use**: `extract_paths()` from `concept_fragmentation/analysis/cross_layer_metrics.py`
- **Use**: `compute_trajectory_fragmentation()` for fragmentation scores
- **Use**: `PathExtractor` from `concept_fragmentation/clustering/paths.py` for archetypal paths

### 4. **Clustering Infrastructure**
- **Use**: Existing utilities in `concept_fragmentation/clustering/`
- **Use**: `determine_optimal_k()` from `concept_fragmentation/utils/cluster_paths.py`
- **Pattern**: Follow existing clustering patterns, don't create new approaches

### 5. **Visualization Components**
- **Use**: `SankeyGenerator` from `concept_fragmentation/visualization/sankey.py`
- **Use**: `TrajectoryVisualizer` from `concept_fragmentation/visualization/trajectory.py`
- **Config**: Use existing visualization configs

### 6. **Metrics Library**
- **Use**: All metrics from `concept_fragmentation/metrics/`
- **Includes**: `compute_centroid_similarity()`, `compute_membership_overlap()`
- **No need**: To implement any fragmentation calculations

## ❌ Anti-Patterns to Avoid

1. **DON'T** create custom activation extraction - ActivationCollector handles all cases
2. **DON'T** implement trajectory calculation - extract_paths() is comprehensive
3. **DON'T** write new fragmentation metrics - complete set already exists
4. **DON'T** create visualization from scratch - existing visualizers are configurable
5. **DON'T** add to core library - keep experiment-specific code in experiments/

## ✨ What We Need to Create (Minimal New Code)

### 1. **Apple Dataset Loader** (experiment-specific)
```python
# experiments/apple_variety/apple_dataset.py
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class AppleVarietyDataset:
    """Apple variety dataset loader following existing patterns."""
    
    def __init__(self, data_path: str, n_top_varieties: int = 10):
        self.data_path = data_path
        self.n_top_varieties = n_top_varieties
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split apple variety data."""
        # Load CSV
        # Select top N varieties
        # Feature engineering
        # Train/test split
        return train_df, test_df
```

### 2. **Experiment Configuration**
```yaml
# experiments/apple_variety/config.yaml
experiment:
  name: "apple_variety_classification"
  type: "classification"
  output_dir: "results/apple_variety"
  
dataset:
  name: "apple_variety"
  data_path: "arxiv_apple/apples_processed.csv"
  n_varieties: 10
  features: ["brix_numeric", "firmness_numeric", "red_pct_numeric", 
             "size_numeric", "season_numeric", "starch_numeric"]
  
model:
  type: "feedforward"
  architecture:
    layers: [8, 32, 64, 32, 10]
    activation: "relu"
    dropout: 0.3
    batch_norm: true
    
training:
  epochs: 200
  batch_size: 32
  learning_rate: 0.001
  optimizer: "adam"
  
clustering:
  method: "kmeans"
  k_selection: "gap_statistic"
  k_max: 10
  
visualization:
  sankey:
    height: 800
    width: 1600
    top_n_paths: 10
  trajectory:
    method: "umap"
    n_components: 3
```

### 3. **Minimal Experiment Runner**
```python
# experiments/apple_variety/run_experiment.py
from concept_fragmentation.experiments.base import BaseExperiment
from concept_fragmentation.activation.collector import ActivationCollector
from concept_fragmentation.analysis.cross_layer_metrics import extract_paths, compute_trajectory_fragmentation
from concept_fragmentation.visualization.sankey import SankeyGenerator
from concept_fragmentation.visualization.trajectory import TrajectoryVisualizer
from concept_fragmentation.config import ExperimentConfig
from apple_dataset import AppleVarietyDataset

class AppleVarietyExperiment(BaseExperiment):
    """Apple variety classification with CTA analysis."""
    
    def setup(self):
        """Load data and initialize components."""
        # Load config
        self.dataset = AppleVarietyDataset(
            self.config.dataset.data_path,
            self.config.dataset.n_varieties
        )
        self.train_df, self.test_df = self.dataset.load_data()
        
        # Initialize model (PyTorch)
        self.model = self._build_model()
        
        # Set up activation collector
        self.activation_collector = ActivationCollector(
            model=self.model,
            layer_patterns=["Linear"]  # Collect from all linear layers
        )
        
    def execute(self):
        """Train model and collect activations."""
        # Standard PyTorch training loop
        # Use activation_collector during forward passes
        # Return activations and predictions
        
    def analyze(self):
        """Analyze trajectories using existing tools."""
        # Use extract_paths() for trajectories
        # Use compute_trajectory_fragmentation() for metrics
        # Identify variety convergence patterns
        
    def visualize(self):
        """Create visualizations using existing components."""
        # Use SankeyGenerator for variety flow
        # Use TrajectoryVisualizer for 3D paths
        # Save to self.output_dir
```

### 4. **Figure Generation Script**
```python
# experiments/apple_variety/generate_figures.py
"""Generate publication-ready figures for apple variety paper."""
from pathlib import Path
import json

def generate_paper_figures(results_dir: str, output_dir: str):
    """Generate all figures for the paper."""
    # Load experiment results
    # Create Sankey diagrams
    # Create trajectory visualizations
    # Create confusion matrices
    # Save to arxiv_apple/figures/
```

## 📁 Final File Structure

```
experiments/apple_variety/
├── __init__.py
├── apple_dataset.py         # Dataset loader (experiment-specific)
├── config.yaml             # Experiment configuration
├── run_experiment.py       # Main runner inheriting BaseExperiment
└── generate_figures.py     # Figure generation for paper

arxiv_apple/
├── figures/                # Output figures for paper
├── apples_processed.csv    # Processed data
└── implementation_plan_final.md
```

## 🔧 Implementation Order

1. **Create apple_dataset.py** - Simple data loader following existing patterns
2. **Create config.yaml** - Standard experiment configuration
3. **Create run_experiment.py** - Minimal code, maximum reuse
4. **Run experiment** - Generate trajectories and analysis
5. **Create figures** - Publication-ready visualizations

## 💡 Design Principles Applied

1. **Maximum Reuse**: Using ALL existing infrastructure
2. **Separation of Concerns**: Experiment code stays in experiments/
3. **Configuration-Driven**: YAML config like all other experiments
4. **Single Source of Truth**: No duplication of functionality
5. **Open/Closed**: Extending existing classes, not modifying
6. **KISS**: Minimal new code, maximum leverage of existing tools

## 🎯 Success Criteria

- ✓ No reimplementation of existing functionality
- ✓ Follows established patterns in codebase
- ✓ Minimal new code (< 300 lines total)
- ✓ Reuses all visualization components
- ✓ Compatible with existing infrastructure
- ✓ Clear separation between library and experiment

This plan ensures robust implementation while maintaining codebase integrity and maximizing the value of existing components.