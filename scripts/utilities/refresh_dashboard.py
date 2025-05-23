#!/usr/bin/env python3
"""
Utility script to refresh all data the Dash dashboard relies on.

What it does
============
1. Deletes local embedding caches in ``visualization/cache`` (these hold UMAP
   ``.npz`` files).
2. Deletes cluster-cache pickles stored under
   ``<RESULTS_DIR>/cache/clusters`` where ``RESULTS_DIR`` comes from
   ``concept_fragmentation.config.RESULTS_DIR`` (or the hard-coded fallback
   used by ``visualization.data_interface.get_config_path``).
3. Optionally clears any other cache sub-folders under ``<RESULTS_DIR>/cache``.
4. (Re)computes embeddings for both the baseline and best configurations for
   the selected datasets/seeds (using ``visualization.reducers.embed_all_configs``).
5. (Re)computes optimal k-means clusters for every layer for those same runs
   (using ``visualization.data_interface.compute_layer_clusters``).

After running, the Dash dashboard will start from a clean cache and should
pick up all layers that exist in the latest ``layer_activations.pkl`` files.

Usage
-----
Run from the repository root:

    python refresh_dashboard.py --datasets heart titanic --seeds 0 1 2

Options:
    --datasets      Space-separated list of dataset names (default: heart titanic)
    --seeds         Space-separated list of integer seeds (default: 0)
    --clear-only    Only delete caches; do *not* regenerate embeddings/clusters.

Notes
-----
* This script does **not** retrain any models; it operates on activations that
  already exist under ``<RESULTS_DIR>``.
* If you have changed the training code and produced deeper networks, make sure
  you rerun training first so the new ``layer_activations.pkl`` files are
  present before executing this refresh script.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import os
from typing import List

# Local imports (relative to repo root)
from visualization.reducers import embed_all_configs
from visualization.data_interface import (
    get_baseline_config,
    get_best_config,
    compute_layer_clusters,
    get_config_path,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Cache-removal helpers
# -----------------------------------------------------------------------------

def _rm_tree(path: str):
    """Recursively delete *path* if it exists, printing what happens."""
    if os.path.isdir(path):
        shutil.rmtree(path)
        print(f"[refresh] Deleted directory: {path}")
    elif os.path.isfile(path):
        os.remove(path)
        print(f"[refresh] Deleted file: {path}")
    else:
        print(f"[refresh] Path not found (skipped): {path}")


def clear_caches() -> None:
    """Remove all known cache folders used by the dashboard."""

    # 1) Local embedding cache (UMAP results, etc.)
    local_embed_cache = os.path.join(REPO_ROOT, "visualization", "cache")
    _rm_tree(local_embed_cache)

    # 2) Global cache folders under RESULTS_DIR (clusters, any others)
    results_root = get_config_path()  # may be hard-coded or from config.py
    global_cache_root = os.path.join(results_root, "cache")
    _rm_tree(global_cache_root)

# -----------------------------------------------------------------------------
# Regeneration helpers
# -----------------------------------------------------------------------------

def regenerate(dataset_names: List[str], seeds: List[int]):
    """Re-embed activations and recompute clusters for *dataset_names*/*seeds*."""

    for dataset in dataset_names:
        print(f"\n[refresh] Processing dataset: {dataset}")

        # Step 1: compute embeddings for baseline + best config
        print("[refresh]  • Embedding activations (baseline & best)…")
        try:
            embed_all_configs(dataset, seeds=seeds)
        except Exception as e:
            print(f"[refresh]    ! Failed to embed {dataset}: {e}")
            continue

        # Step 2: recompute clusters per seed & configuration
        baseline_cfg = get_baseline_config(dataset)
        best_cfg = get_best_config(dataset)
        for seed in seeds:
            for cfg_name, cfg in (("baseline", baseline_cfg), ("best", best_cfg)):
                try:
                    print(
                        f"[refresh]  • Computing clusters ({cfg_name}, seed {seed})…"
                    )
                    compute_layer_clusters(dataset, cfg, seed)
                except Exception as e:
                    print(
                        f"[refresh]    ! Failed clustering {dataset} {cfg_name} seed {seed}: {e}"
                    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Clear dashboard caches and regenerate embeddings/cluster data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["heart", "titanic"],
        help="Datasets to process",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds to process",
    )
    p.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear caches; skip regeneration steps",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("[refresh] Clearing caches…")
    clear_caches()

    if args.clear_only:
        print("[refresh] Done (clear-only mode).")
        return

    print("[refresh] Regenerating embeddings and clusters…")
    regenerate(args.datasets, args.seeds)
    print("[refresh] All tasks completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted") 