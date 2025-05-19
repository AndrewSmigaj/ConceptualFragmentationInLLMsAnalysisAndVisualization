# PLAN – Integrating Per-Layer k-Means Cluster Centres & Fracture Scores  
## (to keep `paper.md` and the codebase consistent)

### 1  Motivation
`paper.md` Section 3.3 already specifies a silhouette-guided, per-layer search for the optimal number of clusters **k**; those clusters feed the fragmentation metrics.  The code should therefore

1. compute those layer-specific clusters once in the data-processing layer;
2. reuse exactly the same clusters for
   • metric computation (cluster entropy / fracture scores) and
   • visualisation (spheres marking cluster centres, optional point colouring);
3. expose simple toggles in Dash to view/hide the centres and (optionally) colour by cluster.

Keeping all clustering in **`data_interface.py`** (or a metrics module) preserves a single source of truth; `traj_plot.py` simply visualises the results.

---

### 2  Components & Responsibilities
| Component | Change | Responsibility |
|-----------|---------|----------------|
| **`data_interface.py`** | modify | • Load activations <br>• For each layer: silhouette search, pick best k <br>• Fit K-Means → store `centers`, `labels`, `k` <br>• Return dict `layer_clusters` <br>• Compute fracture scores from these labels |
| **`traj_plot.py`** | modify | • Accept `layer_clusters`, `show_cluster_centers`, `color_by` <br>• Plot spheres for centres when toggled <br>• Optionally colour points by cluster label |
| **`dash_app.py`** | modify | • On data load call clustering fn; store `layer_clusters` in `dcc.Store` <br>• UI: checkbox "Show cluster centres", dropdown "Colour by {Class, Metric, Cluster}", slider "Max k" <br>• Pass args to `build_single_scene_figure` |
| Docs (`paper.md`) | minor edit | Clarify that the Dash tool visualises the same clusters used for metrics |

---

### 3  Implementation Steps

#### 3.1  `data_interface.py`
1. `from sklearn.cluster import KMeans`; `from sklearn.metrics import silhouette_score`
2. Add `compute_layer_clusters(dataset, config, seed, max_k=10)` returning
```python
{layer_name: {"k": k_opt,
              "centers": ndarray(k,3),
              "labels": ndarray(n_samples)}}
```
3. Update `compute_fractured_off_scores` to take pre-computed clusters (or call the new function internally).
4. Cache results under `.../clusters.pkl`.

#### 3.2  `traj_plot.py`
Add parameters to `build_single_scene_figure`:
```python
layer_clusters=None,
show_cluster_centers=False,
color_by="class"  # {"class","metric","cluster"}
```
When plotting a layer:
* if `show_cluster_centers` → plot spheres at `centers` (Y-offset applied).  Use palette `QUALITATIVE_COLORS`.
* if `color_by=="cluster"` → colour points by `labels` for that layer.

#### 3.3  `dash_app.py`
* Extend loading callback to compute/store `layer_clusters` (respecting slider "max k").
* Add sidebar controls:
  * Checklist `cluster-opts` → "show_centres".
  * Dropdown `color-mode` → {Class, Metric, Cluster}.
  * Slider `max-k-slider` (4-15, default 10).
* Pass these to `build_single_scene_figure`.

---

### 4  Validation Checklist
* Dash launches; toggling "Show cluster centres" adds/removes spheres.
* Colour-mode "Cluster" recolours points consistently.
* Fracture scores use identical cluster labels.
* Clustering cached; reload is fast.

---

### 5  Estimated Effort
| Task | LoC Δ | Time |
|------|-------|------|
| `data_interface.py` enhancement | ~120 | 2 h |
| `traj_plot.py` mods | ~80 | 1.5 h |
| Dash UI & callbacks | ~60 | 1.5 h |
| Testing | — | 1 h |
| Doc tweak | — | 0.25 h |
| **Total** | | **≈6 h** |

---
*Prepared by: o3 assistant — 15 May 2025* 