# Implementation Plan: Concept Fragmentation in Neural Networks – Visualizing and Measuring Intra-Class Dispersion in Feedforward Models

## 1. Overview
This document describes the engineering roadmap for the paper *"Concept Fragmentation in Neural Networks: Visualizing and Measuring Intra-Class Dispersion in Feedforward Models."*  The goal is to build an **open-source, fully reproducible** toolkit that (a) trains baseline and regularized feed-forward models, (b) captures layer activations, (c) computes fragmentation metrics, and (d) produces publication-ready visualisations.

Key deliverables:
1. Metric implementations for **Cluster Entropy** and **Subspace Angle**
2. Activation tracing utilities that scale from tiny (3×3×3) to wide/deep networks
3. Cohesion regularisation (contrastive loss) plugin
4. Scripts for end-to-end experiments on four datasets (Titanic, Adult Income, Heart Disease, Fashion-MNIST subset)
5. Figures & tables for the final paper + interactive notebooks

---

## 2. Current Repository Status (2025-05-13)
✅ Implemented
* `concept_fragmentation/models/feedforward.py` – 3-layer MLP with named layers & activation cache
* `concept_fragmentation/models/regularizers.py` – Cohesion regulariser (InfoNCE-style)
* `concept_fragmentation/data/` – dataset loaders (`loaders.py`) and preprocessing utilities (`preprocessors.py`)

🚧 **Missing / Empty Stubs**
* Activation hooks (`concept_fragmentation/hooks/activation_hooks.py`)
* Fragmentation metrics (`metrics/cluster_entropy.py`, `metrics/subspace_angle.py`)
* Visualisation utilities (`visualization/activations.py`, `visualization/trajectories.py`)
* Experiment scripts (`experiments/train.py`, `experiments/evaluate.py`, `experiments/visualize.py`)
* Utility helpers (`utils/helpers.py`)
* Tests (`tests/`)
* `config.py`, `requirements.txt`, and repo-level `README.md`

Action items below focus on filling these gaps.

---

## 3. Repository Layout
```
concept_fragmentation/
├── config.py                  # All hyper-parameters & dataset paths
├── data/
│   ├── loaders.py             # Titanic, Adult, Heart, Fashion-MNIST
│   └── preprocessors.py       # One-hot, scaling, imputation
├── models/
│   ├── feedforward.py         # Baseline MLP (done)
│   └── regularizers.py        # Cohesion regulariser (done)
├── hooks/
│   └── activation_hooks.py    # Forward hooks for activation capture (TODO)
├── metrics/
│   ├── cluster_entropy.py     # Intra-class k-means entropy (TODO)
│   └── subspace_angle.py      # Pairwise principal angles (TODO)
├── visualization/
│   ├── activations.py         # PCA/UMAP scatter & layer overlays (TODO)
│   └── trajectories.py        # Per-sample activation paths (TODO)
├── experiments/
│   ├── train.py               # Training loop + logging (TODO)
│   ├── evaluate.py            # Metric computation (TODO)
│   └── visualize.py           # Figure generation (TODO)
├── utils/
│   └── helpers.py             # Seed fixing, logging, CLI wrappers (TODO)
├── tests/
│   ├── test_metrics.py        # Unit tests for metrics (TODO)
│   └── test_hooks.py          # Hook integrity tests (TODO)
└── notebooks/                 # Exploratory analysis & paper figures
```

---

## 4. Detailed Task Breakdown
### 4.1 Hooks Module – Activation Capture (Priority-High)
* Implement generic forward hook registration that stores activations in a memory-efficient dict keyed by layer name
* Add **top-k neuron selection** option for sparse visualisation (default *k* = 3)
* Provide context-manager wrapper so hooks are auto-removed after use

### 4.2 Metrics Module (Priority-High)
1. **Cluster Entropy**
   * Fit *k*-means **once** on the entire activation matrix of a layer.
   * **Automatic choice of K**: we evaluate the silhouette score for K = 2 … K_max (K_max = min(12, √N)) and pick the K that maximises the score.
   * For each class *c*, compute the proportion pₖ(c) of its samples that fall into each selected cluster k.
   * Compute entropy  H_c = −Σₖ pₖ(c) log₂ pₖ(c) and normalise by log₂ K.
   * Return per-class and aggregate (mean / max) scores
2. **Subspace Angle**
   * Take 90 % variance principal components per class
   * Bootstrap class activations (B=10), compute pairwise principal angles θ
   * Aggregate statistics: mean θ, std, 95 % CI

### 4.3 Visualisation Module
* `activations.py` – 2D/3D PCA & UMAP scatter, coloured by class & sub-cluster
* `trajectories.py` – line plots of per-sample activation over layers (tiny nets) and PCA-compressed paths (larger nets)
* All plots seeded for determinism, matplotlib + seaborn styling

### 4.4 Experiment Scripts
* `train.py`
  * CLI: dataset, model size, regularisation flag, seeds
  * Logs: train/val loss, accuracy, fragmentation metrics per epoch
* `evaluate.py`
  * Loads checkpoint, computes metrics on test split
* `visualize.py`
  * Generates & saves figures used in paper

### 4.5 Testing & CI
* PyTest suites covering metrics correctness, hook output shapes, reproducibility
* GitHub Actions workflow: lint → unit tests → notebook smoke test

### 4.6 Documentation & Config
* Populate `config.py` with default hyper-parameters and dataset paths
* Fill `requirements.txt` with pinned versions (see Section 7)
* Write a thorough `README.md` with quick-start instructions

---

## 5. Cohesion Regularisation Hyper-Parameter Grid
| Weight λ | Temperature τ | Similarity Threshold | Layers              |
|---------:|--------------:|----------------------|---------------------|
| 0 (baseline) | 0.07 | 0.0 | – |
| 0.1 | 0.07 | 0.0 | layer3 |
| 0.1 | 0.07 | 0.3 | layer3 |
| 0.5 | 0.07 | 0.3 | layer2+layer3 |

---

## 6. Timeline (≈10 Weeks)
1. **Week 1–2 Foundation** – Finish hooks, metrics, config, and basic tests
2. **Week 3–4 Experiments v0** – Implement training script, run tiny nets
3. **Week 5–6 Regularisation & Visuals** – Integrate cohesion loss; implement visualisation utilities
4. **Week 7–8 Full Experiments** – All datasets, hyper-parameter sweeps, statistical tests
5. **Week 9–10 Paper Prep** – Final figures, notebooks, documentation

---

## 7. Dependencies (`requirements.txt`)
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
umap-learn>=0.5.0
jupyter>=1.0.0
pytest>=6.0.0
```

---

## 8. Next Steps
1. **Metrics module** – implement and unit-test (`cluster_entropy.py`, `subspace_angle.py`)
2. **Activation hooks** – implement `activation_hooks.py`, integrate with `FeedforwardNetwork`
3. **Populate config & requirements** – essential for running anything
4. **Discuss**: Do we want to support non-tabular datasets beyond the current list in this release? 