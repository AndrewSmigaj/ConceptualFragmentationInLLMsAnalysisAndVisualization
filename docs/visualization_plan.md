# Concept Fragmentation – 3D Trajectory Visualization Plan

> Author: TODO – fill in when adopted
> Date: TODO – fill in when adopted

## 0. Purpose
This document breaks down **everything** needed to deliver interactive, publication-quality 3D trajectory visualizations that illustrate how cohesion regularization affects concept fragmentation across the *Titanic* and *Heart* datasets.  It is written so that either a human developer **or** an AI coding assistant can follow each step autonomously.

---

## 1. High-Level Goals
1. Per-layer 3-D embedding (UMAP) of activations for **baseline** vs. **cohesion-regularized** models.
2. Multi-panel figure showing trajectories of individual samples from *input* → *layer1* → *layer2* → *layer3*.
3. Interactive exploration (hover, rotate, filter, toggle configs) + static PDF/SVG export for the paper.
4. Highlight outlier trajectories (e.g. high-fare Titanic passengers) and quantify changes.

---

## 2. Data Inventory & Conventions
| Path template | Contents | Notes |
|---------------|----------|-------|
| `results/cohesion_stats.csv` | Aggregate metrics (entropy, angle, accuracy, etc.) | small; already tracked in repo |
| `D:/concept_fragmentation_results/analysis/cohesion_summary.csv` | Same info on production server | **Do not** commit raw file – read at runtime |
| `D:/concept_fragmentation_results/cohesion/{dataset}/{config}/seed_{seed}/layer_activations.pkl` | Dict: `{layer_name: np.ndarray(samples, hidden_dim)}` | ~100–300 MB per run |
| `training_history.json` | Epoch-level metrics | not needed for trajectories |

**Assumption**: the repo will mount or symlink the `D:/concept_fragmentation_results` directory at runtime (configurable in `config.py`).

---

## 3. Environment & Dependencies
Add/confirm in `requirements.txt` (versions pinned for reproducibility):
```
plotly>=5.18,<6
umap-learn>=0.5.4
pandas>=2.0
numpy>=1.25
scikit-learn>=1.3
scipy>=1.11
kaleido  # for static export
```
Optional for Dash dashboard:
```
dash>=2.14
```

**NOTE**: No environment variables – all configurable in `config.py`.

---

## 4. Directory Structure (repo-side)
```
visualization/
│
├─ data_interface.py        # I/O helpers to load activations & stats
├─ reducers.py              # wrap UMAP/other DR with caching
├─ traj_plot.py             # core Plotly figure builder
├─ dash_app.py              # optional interactive dashboard
├─ notebooks/
│   └─ sanity_checks.ipynb  # quick validation of embeddings
└─ cache/                   # *.npz or *.pkl cached embeddings (git-ignored)
```

---

## 5. Step-by-Step Implementation
### 5.1 Data Loading (`data_interface.py`)
1. `load_stats(csv_path) -> pd.DataFrame`
2. `get_best_config(dataset) -> Dict`  
   – uses rules from spec (w=0.1, τ=0.0, L3) but stays generic.
3. `load_activations(dataset, config, seed) -> Dict[layer, np.ndarray]`
4. `select_samples(df_stats, k_frag=20, k_norm=20) -> List[int]`  
   – returns indices of high & low fragmentation samples.

### 5.2 Dimensionality Reduction (`reducers.py`)
1. `Embedder` class encapsulating UMAP parameters, random_state.
2. `fit_transform(layer_act) -> np.ndarray(n_samples, 3)`.
3. Automatic disk caching (`cache/{hash}.npz`).
4. CLI utility `python reducers.py Titanic Heart` to pre-compute.

### 5.3 Trajectory Assembly
1. Load reduced coords for each (dataset, config, seed).
2. Stack into `coords[sample_i, layer_j, 3]` tensor.
3. Normalize/align coordinate systems **per layer** (no global alignment to avoid DR artefacts).

### 5.4 Plotting (`traj_plot.py`)
1. `build_multi_panel(coords_dict, meta_dict) -> plotly.graph_objs.Figure`.
2. Each subplot (`scene`) has:
   - 3-D scatter of points (opaque dots)
   - Arrow cones for end points (`Cone` trace) OR small `Mesh3d` arrowheads.
3. Trajectories: `go.Scatter3d(mode="lines", ...)` per sample.  
   – throttle to ≤200 lines by default; add legend toggle.
4. Color mapping:
   - Baseline vs. Regularized (palette from seaborn `colorblind` set).
   - Optional discrete class color per dataset.
5. Global layout:
   - Synchronized camera across scenes using shared `up`/`center/eye`.
   - Titles and axis labels (x/y/z → `UMAP-1/2/3`).
6. Export helpers: `fig.write_html(...)`, `fig.write_image(..., format="pdf")`.

### 5.5 Dashboard (`dash_app.py`, optional)
1. Dropdowns for Dataset / Config / Seed / Sample Subset.
2. Checkbox to toggle baseline vs. regularized traces.
3. Graph component hosting figure returned by `traj_plot`.
4. Run with `python dash_app.py`.

---

## 6. Performance & UX Tips
- Use WebGL (`Scatter3d`) for large point clouds.
- Provide a `--max-lines` CLI argument.
- Pre-compute and persist UMAP embeddings (**slow**).
- Arrowheads: keep `sizemode="absolute", sizeref=0.3` to avoid crowding.

---

## 7. Testing Checklist
- [ ] Unit test: embedding cache hits/misses
- [ ] Unit test: trajectory tensor shape consistency
- [ ] Visual sanity: run `notebooks/sanity_checks.ipynb` -> should show well-separated classes.
- [ ] Cross-browser check: Chrome, Firefox.
- [ ] Static export size < 10 MB per figure.

---

## 8. Milestones & Timeline
| Week | Deliverable |
|------|-------------|
| 1 | `data_interface.py`, `reducers.py` done + embeddings cached |
| 1 | Quick static prototype for Titanic baseline |
| 2 | `traj_plot.py` full figure (baseline vs. reg) |
| 2 | Static PDF/SVG exports for both datasets |
| 3 | Interactive Dash dashboard |
| 3 | Documentation + code comments |

---

## 9. Known Risks & Mitigations
1. **Large activation files** → stream from disk, load sample subset.
2. **UMAP instability** → fix `random_state`, log hyperparams.
3. **WebGL memory limits** → subsample or chunk rendering.

---

## 10. Next Actions (for AI agent)
1. Create `visualization/` directory scaffolding.
2. Implement `data_interface.py` with unit tests.
3. Prototype `Embedder` and cache logic.
4. Generate first demo figure for *Titanic* (seed 0) baseline.
5. Review output with human collaborator before scaling up.

---

> **End of Plan** 