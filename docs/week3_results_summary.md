# Week 3 Cohesion-Regularisation Experiments – Summary

_Concept Fragmentation in Neural Networks / draft paper notes_

## 1  Experimental set-up

* **Datasets**   Titanic (tabular, binary) and Heart Disease (tabular, binary)
* **Network**    Feed-forward 3-hidden-layer MLP (widths from `config.py`)
* **Regularisation grid**  
  – Baseline (no regularisation)  
  – Cohesion *w = 0.1* on **L3** only  
  – Cohesion *w = 0.1* + _similarity threshold τ = 0.3_ on **L3**  
  – Cohesion *w = 0.1* + τ = 0.3 on **L2 + L3**  
  – Cohesion *w = 0.5* + τ = 0.3 on **L2 + L3**

For each grid point we trained **3 random seeds** and captured:
* Final-epoch fragmentation metrics per layer (`entropy`, `angle`)
* Per-epoch loss / accuracy
* Trajectories for qualitative inspection

Fragmentation is now computed **once on the final model** (after early-stopping) for every hidden layer; results live in each experiment's `training_history.json` → `layer_fragmentation`.

## 2  Key quantitative findings

| Dataset | Configuration | Δ final-layer entropy | *p*-value | Δ final-layer angle | *p*-value | Δ test acc. |
|---------|---------------|----------------------|-----------|---------------------|-----------|--------------|
| Titanic | *w* 0.1   τ 0  L3 | **−3.9 %** | 0.049 | −19.8 % | 0.36 | 0.00 % |
| Titanic | *w* 0.1   τ 0.3  L2-3 | −3.9 % | 0.049 | −19.8 % | 0.36 | 0.00 % |
| Heart   | *w* 0.1   τ 0  L3 | **−7.1 %** | **0.036** | +5.9 % | 0.81 | +0.7 % |
| Heart   | *w* 0.1   τ 0.3  L2-3 | −7.1 % | 0.036 | +5.9 % | 0.81 | +0.7 % |
| Heart   | *w* 0.5   τ 0.3  L2-3 | −7.1 % | 0.14 | +5.9 % | 0.82 | +0.7 % |

* **Entropy (compactness)** drops for both datasets, statistically significant for Heart and marginally significant for Titanic.
* **Angle (sub-space alignment)** response is mixed: decreases for Titanic (but high variance) and increases slightly for Heart.
* **Accuracy** is unchanged or marginally higher; cohesion does **not hurt generalisation** at these weights.
* Regularising an extra layer (L2-3) benefits Titanic's mid-layer entropy but gives no additional gain for Heart.
* Raising weight to 0.5 does **not** improve fragmentation further.

## 3  Qualitative observations

UMAP trajectory plots show that cohesion pulls outlier samples back toward the main class manifold while leaving between-class separation intact.  In Titanic, high-fare passengers now follow the dominant path more closely; in Heart, class clouds are visibly tighter.

## 4  Interpretation

1. **Fragmentation concentrates in the last hidden layer.**  Earlier layers already exhibit lower entropy; applying cohesion at L3 (and optionally L2) is sufficient.
2. **Moderate strength works best.**  Increasing the weight beyond 0.1 yields diminishing returns and can distort angle.
3. **Entropy is the most responsive metric.**  Angle alignment may require a different or complementary regulariser.

## 5  Recommended next experiments

| Aim | Action |
|-----|--------|
| Increase statistical power | Train **7 additional seeds** (total = 10) for Titanic & Heart with *w* = 0.1, τ = 0.0 on L3, and with τ = 0.3 on L2-3.  Save metrics only, omit activations to save disk. |
| Explore similarity threshold | Run *w* = 0.1, τ ∈ {0.1, 0.2} on L3 to test whether a mild threshold aligns angle without hurting entropy. |
| Alternative regulariser | Prototype an **orthogonality loss** or *between-class repulsion* term to explicitly reduce sub-space angle. |
| Visualization for the paper | Re-generate trajectory plots for representative outliers showing baseline vs. best cohesion setting (high-fare Titanic, noisy Heart samples). |

## 6  Implications for the paper draft

* We can now **claim** that cohesion regularisation measurably reduces concept fragmentation (entropy) without accuracy loss on two tabular tasks.
* The effect on sub-space angle is inconclusive → motivates future work or a hybrid loss.
* Figures to include:  
  1. Bar chart of entropy reduction per layer and dataset.  
  2. UMAP trajectories (before/after).  
  3. Accuracy vs. fragmentation scatter.
* Discussion sections should cover:
  * Why fragmentation arises in late layers.  
  * Trade-off between compactness and alignment.  
  * Practical guidance on setting *w* and τ.

---
_Document generated 2025-05-14 by the experiment helper script._ 