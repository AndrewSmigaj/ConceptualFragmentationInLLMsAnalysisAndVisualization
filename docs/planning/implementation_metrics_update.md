# New Fragmentation Metrics Implementation

This document summarizes the changes made to implement the three new metrics for the concept fragmentation project, as described in the updated paper.

## 1. New Metrics Implemented

### Intra-Class Pairwise Distance (ICPD)
- File: `concept_fragmentation/metrics/intra_class_distance.py`
- Formula: `ICPD_c(l) = 1/|S_c|² ∑_{i,j∈S_c} ‖f_l(x_i)−f_l(x_j)‖₂`
- Description: Measures the average distance between all pairs of points within a class.
- Normalization: Default behavior is to normalize by feature dimension.
- Key function: `compute_intra_class_distance()`

### Optimal Number of Clusters (k*)
- File: `concept_fragmentation/metrics/optimal_num_clusters.py`
- Formula: `k*_c(l) = arg max_k silhouette(k)`
- Description: Finds the optimal number of clusters for each class using silhouette score.
- Range: Default range is k ∈ [2,10].
- Key function: `compute_optimal_k()`

### Representation Stability (Δ-Norm)
- File: `concept_fragmentation/metrics/representation_stability.py`
- Formula: `Δ(l) = ‖f_{l}(X) - f_{l-1}(X)‖_F / ‖f_{l-1}(X)‖_F`
- Description: Measures the relative change in representations between consecutive layers.
- Key function: `compute_representation_stability()`

## 2. Scripts Updated

### Experiment Runner
- Added `--recompute_metrics` flag to `run_single_experiment.py`
- Updated `run_baseline_experiments.ps1` to support the new flag
- Created `run_cohesion_grid.ps1` for running cohesion grid experiments

### Metric Registration
- Updated `concept_fragmentation/metrics/__init__.py` to expose the new metrics

### Visualization
- Added the new metrics to `AVAILABLE_POINT_COLOR_METRICS` in `visualization/dash_app.py`
- Updated `create_fracture_graph()` function to support displaying all metrics together
- Added multi-y-axis support for better visualization

## 3. Paper Updates
- Abstract: Updated to mention the three new metrics
- Section 3.3: Rewritten to describe the new metrics in detail
- Metrics described as complementary approaches to quantify fragmentation

## 4. Running the Updated Code

### Recomputing Metrics on Existing Experiments

To recompute metrics on existing experiments, use the `--recompute_metrics` flag:

```
# For baselines
powershell -ExecutionPolicy Bypass -File .\run_baseline_experiments.ps1 -RecomputeMetrics

# For cohesion grid
powershell -ExecutionPolicy Bypass -File .\run_cohesion_grid.ps1 -RecomputeMetrics
```

### Running Dashboard with New Metrics

Launch the dashboard as usual:

```
python -m visualization.dash_app
```

The updated dashboard now includes:
- All three new metrics available in the dropdown
- A combined view showing all metrics when none is specifically selected
- Multi-y-axis display for better comparison

## 5. Expected Results

The updated metrics should show the following patterns for the Titanic dataset:

1. **ICPD**: Shows how intra-class dispersion changes across layers
2. **k***: Shows the "hump then drop" pattern mentioned in the paper - middle layers have more clusters
3. **Δ-Norm**: Shows peaks where major representation changes occur, correlating with k* changes

## 6. Next Steps

- Analyze results to validate the patterns described in the paper
- Generate new figures for the paper based on the updated metrics
- Consider adding these metrics to the standard experiment output 