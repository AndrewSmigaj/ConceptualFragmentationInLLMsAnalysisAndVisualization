# Remaining Implementation Plan  
*(updated after 2025-05-14 alignment meeting)*  

## 0  Key Decisions Incorporated  
| Ref. | Decision | Notes |
|------|----------|-------|
| 1.1  | **Subspace-Angle parameter** – adopt *variance-threshold* approach (retain ≥ 90 % variance) rather than a fixed `n_components`.  Update code, tests and config accordingly; manuscript already describes this behaviour so the code moves into full agreement with the paper. |
| 1.2  | Skip large CPU/GPU test-matrix & profiling.  Regular unit tests are sufficient. |
| 1.3  | Skip CI / GitHub-Actions setup for now. |
| 2    | Proceed with all baseline-experiment tasks **except dataset-caching** (datasets will be read directly via loader / preprocessing utilities). |
| 3    | Cohesion-regularisation grid search **does align** with the Methodology (§3.5) and planned ablations (§5.3) in the paper – keep as is. |
| 4-6 | Week-4/5/6 tasks accepted unchanged. |

---

## 1  Task Breakdown & Timeline (≈ 6 weeks)
### Week 1  – Metrics Refactor & Validation ✅
1. **Subspace-angle refactor** ✅
    • modify `subspace_angle.py` to accept `var_threshold` (default 0.90) and choose the minimal number of PCs meeting this threshold. ✅
    • deprecate `n_components` in public API (keep for backward compatibility). ✅
    • update `config.py` (`METRICS["subspace_angle"]` section) and unit tests. ✅
2. **Sanity run** – execute existing pytest suite locally after refactor. ✅

### Week 2  – Baseline Experiments (4 datasets × 3 seeds) 🔄
1. ✅ Finalise *preprocessing* pipelines (no persistent caching).  
2. 🔄 Train baseline (no regularisation) models:  
   `python experiments/train.py --dataset {titanic|adult|heart|fashion_mnist} --seeds 0 1 2`  
   - [x] Updated `baseline_run.py` script with proper logging and experiment directory structure
   - [x] Enhanced `train.py` to accept custom experiment directory paths and device selection
   - [x] Added proper utilities in `helpers.py` for seed fixing and result aggregation
   - [x] Created PowerShell script for running experiments (`run_baseline_experiments.ps1`)
   - [x] Fixed dataset loaders to handle categorical data properly in Adult and Heart datasets
   - [x] Fixed Fashion MNIST loader to properly flatten images for the feedforward model
   - [x] Created and verified test script to confirm all datasets now load correctly
   - [ ] Execute baseline experiments for all datasets and seeds
3. 🔄 For each run store:  
   - [x] checkpoint (`best_model.pt`) - saving mechanism implemented in train.py
   - [x] per-epoch activations (last hidden layer) - saving mechanism implemented in train.py
   - [x] fragmentation metrics (entropy & angle) - metrics calculation implemented in train.py
   - [x] accuracy / loss history (JSON) - history saving implemented in train.py
   - [ ] Execute runs to generate these artifacts

### Week 3  – Cohesion-Regularisation Grid
1. Implement `experiments/grid_run.py` that sweeps over the table in §5 of the implementation plan.  
2. Launch grid; write one JSONL line per config with final metrics and training stats.  
3. Aggregate interim results into `results/regularisation_summary.csv`.

### Week 4  – Analysis & Visualisation
1. Generate visuals using `visualization/` utilities:  
   • PCA/UMAP class scatter (best layer)  
   • Trajectory plots across layers  
   • Entropy & angle vs. epoch curves  
2. Begin `notebooks/results_analysis.ipynb` that loads JSON/CSV logs and auto-plots figures for the paper.

### Week 5  – Paper Figures & Writing
1. Replace **[Placeholder]** sections in `paper.md` with quantitative results and graphics.  
2. Add ablation tables comparing baseline vs. regularised fragmentation and accuracy deltas.  
3. Draft discussion on when fragmentation is harmful / benign (using empirical evidence).

### Week 6  – Polish & Release Candidate
1. Update `README.md` with quick-start guide, citation, dataset licences.  
2. Freeze `requirements.txt` with exact versions; add `environment.yml`.  
3. Tag `v0.1-rc`; verify a fresh clone reproduces main experiments end-to-end.

---

## 2  File-Level To-Dos
| File | Action | Status |
|------|--------|--------|
| `concept_fragmentation/metrics/subspace_angle.py` | implement variance-threshold logic; keep `n_components` alias. | ✅ |
| `concept_fragmentation/config.py` | set `METRICS["subspace_angle"]["var_threshold"] = 0.9`; mark `n_components` as legacy. | ✅ |
| `concept_fragmentation/tests/test_metrics.py` | adjust subspace-angle tests to call the new signature. | ✅ |
| `concept_fragmentation/experiments/baseline_run.py` | enhance script to support full baseline run with proper logging and directory structure | ✅ |
| `run_baseline_experiments.ps1` | PowerShell script to automate experiment execution | ✅ |
| `concept_fragmentation/data/loaders.py` | Fix handling of categorical data in Adult/Heart datasets and image flattening in Fashion MNIST | ✅ |
| `concept_fragmentation/data/preprocessors.py` | Fix handling of pandas category dtype and proper conversion to numeric values | ✅ |
| `experiments/grid_run.py` | **new** – grid-runner script (see Week 3). | ⏱️ Scheduled for Week 3 |
| `notebooks/results_analysis.ipynb` | **new** – auto-loads logs & produces figures. | ⏱️ Scheduled for Week 4 |

---

## 3  Risks & Mitigations
1. **Runtime for full grid** – may be heavy on CPU; use minibatch logging and consider early-stop criteria.  
2. **Variance-threshold edge cases** – very small classes might require fallback to a minimum component count; add safeguard in refactor.  
3. **Dataset availability** – ensure loaders auto-download (Fashion-MNIST) or gracefully prompt for CSV paths (Adult/Titanic/Heart).
4. **Week 2 Specific**: GPU availability may limit parallel experimentation; use CPU fallback option in training code (implemented).
5. **Data format issues** - ✅ Fixed categorical data handling in Adult and Heart datasets, and image flattening in Fashion MNIST.

---

## 4  Ownership & Communication
• Metric refactor & test updates – *Alice*  
• Baseline / grid experiments – *Bob*  
• Visualisation scripts & notebook – *Carol*  
• Paper write-up – *Dave*  
Progress sync meeting every Friday 16:00 UTC on shared Zoom.

---

*Document updated 2025-05-14* 