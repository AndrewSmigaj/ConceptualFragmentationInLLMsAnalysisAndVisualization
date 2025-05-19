#!/usr/bin/env python3
"""
Main entry point for generating concept fragmentation visualizations.

This script combines functionality from the other modules to create
publication-quality 3D visualizations of neural network layer trajectories.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Ensure we can import from the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from visualization.data_interface import (
    load_stats, get_best_config, get_baseline_config, 
    select_samples, load_dataset_metadata
)
from visualization.reducers import Embedder, embed_layer_activations, embed_all_configs
from visualization.traj_plot import build_multi_panel, plot_dataset_trajectories, save_figure

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate 3D visualizations of concept fragmentation in neural networks."
    )
    
    # Required arguments
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Datasets to visualize (e.g., titanic, heart)"
    )
    
    # Options for data loading
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to use (default: [0, 1, 2])"
    )
    
    # Options for visualization
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/visualizations",
        help="Directory to save visualizations (default: results/visualizations)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to plot (default: 200)"
    )
    parser.add_argument(
        "--highlight-count",
        type=int,
        default=20,
        help="Number of high-fragmentation samples to highlight (default: 20)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["html", "pdf"],
        help="Output formats (default: [html, pdf])"
    )
    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="Disable trajectory arrows"
    )
    
    # UMAP parameters
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (default: 15)"
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (default: 0.1)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache UMAP embeddings (default: visualization/cache)"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set up output directory
    output_dir = os.path.join(parent_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up cache directory
    if args.cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    else:
        cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load statistics
    try:
        print("Loading statistics...")
        df_stats = load_stats()
        print(f"Loaded statistics with {len(df_stats)} rows.")
    except Exception as e:
        print(f"Error loading statistics: {e}")
        return
    
    # Process each dataset
    for dataset in args.datasets:
        try:
            print(f"\n=== Processing dataset: {dataset} ===")
            
            # Create embedder
            embedder = Embedder(
                n_components=3,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                random_state=42,
                cache_dir=cache_dir
            )
            
            # Get configurations
            best_config = get_best_config(dataset)
            baseline_config = get_baseline_config(dataset)
            
            print(f"Best config: {best_config}")
            print(f"Baseline config: {baseline_config}")
            
            # Embed activations
            print("\nEmbedding activations...")
            all_embeddings = {}
            
            # Baseline
            baseline_embeddings = {}
            for seed in args.seeds:
                print(f"  Embedding baseline config for seed {seed}...")
                baseline_embeddings[seed] = embed_layer_activations(
                    dataset, baseline_config, seed, embedder=embedder)
            all_embeddings["baseline"] = baseline_embeddings
            
            # Best config
            best_embeddings = {}
            for seed in args.seeds:
                print(f"  Embedding best config for seed {seed}...")
                best_embeddings[seed] = embed_layer_activations(
                    dataset, best_config, seed, embedder=embedder)
            all_embeddings["regularized"] = best_embeddings
            
            # Select samples to highlight (high fragmentation)
            print("\nSelecting samples to highlight...")
            # For now, just use random samples
            n_samples = list(baseline_embeddings.values())[0]["layer1"].shape[0]
            highlight_indices = np.random.choice(
                n_samples, size=min(args.highlight_count, n_samples), replace=False)
            print(f"  Selected {len(highlight_indices)} samples to highlight")
            
            # Create visualization
            print("\nCreating visualization...")
            fig = plot_dataset_trajectories(
                dataset,
                all_embeddings,
                n_samples=min(n_samples, args.max_samples),
                seed=42
            )
            
            # Save visualization in each format
            print("\nSaving visualization...")
            for fmt in args.formats:
                output_path = os.path.join(output_dir, f"{dataset}_trajectories.{fmt}")
                print(f"  Saving to {output_path}...")
                save_figure(fig, output_path, format=fmt)
            
            print(f"=== Completed {dataset} ===")
            
        except Exception as e:
            print(f"Error processing dataset {dataset}: {e}")
    
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main() 