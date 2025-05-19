#!/usr/bin/env python3
"""
Test script for concept fragmentation visualization.

This script provides a quick test of the visualization tools by:
1. Loading statistics (or using synthetic data if not available)
2. Creating a synthetic embedding for testing
3. Generating a simple trajectory visualization
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go

# Add visualization package to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Set up logging
log_file = os.path.join(script_dir, "test_log.txt")
def log(message):
    """Log a message to console and file."""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")

# Clear previous log
if os.path.exists(log_file):
    os.remove(log_file)

log(f"\n=== Visualization Test ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")

# Try to import our modules
try:
    from visualization.data_interface import (
        load_stats, get_best_config, get_baseline_config,
        load_activations, select_samples, load_dataset_metadata
    )
    from visualization.reducers import Embedder
    from visualization.traj_plot import (
        build_multi_panel, normalize_embeddings, 
        plot_dataset_trajectories, save_figure
    )
    log("[SUCCESS] Successfully imported visualization modules")
except Exception as e:
    log(f"[ERROR] Error importing visualization modules: {e}")
    sys.exit(1)

def load_real_data(dataset="titanic", seed=0):
    """Try to load real data or return None if not available."""
    try:
        log(f"Attempting to load real data for {dataset}, seed {seed}...")
        best_config = get_best_config(dataset)
        baseline_config = get_baseline_config(dataset)
        
        # Load activations
        baseline_activations = load_activations(dataset, baseline_config, seed)
        best_activations = load_activations(dataset, best_config, seed)
        
        log(f"[SUCCESS] Successfully loaded real activations")
        log(f"   Baseline layers: {list(baseline_activations.keys())}")
        log(f"   Sample shape: {baseline_activations[list(baseline_activations.keys())[0]].shape}")
        
        return baseline_activations, best_activations
    except Exception as e:
        log(f"[ERROR] Couldn't load real data: {e}")
        log("   Will use synthetic data instead")
        return None, None

def generate_synthetic_data(n_samples=100, n_layers=3, n_features=20):
    """Generate synthetic activation data for testing."""
    log(f"Generating synthetic data with {n_samples} samples, {n_layers} layers...")
    
    # Create synthetic high-dimensional data
    np.random.seed(42)
    layer_activations = {}
    
    # Class labels (for coloring)
    class_labels = np.random.choice([0, 1], size=n_samples)
    
    # Create activation patterns (with increasing class separation by layer)
    for i in range(n_layers):
        layer_name = f"layer{i+1}"
        
        # Base activations
        data = np.random.randn(n_samples, n_features)
        
        # Add class-specific signal that increases with layer depth
        separation = (i + 1) / n_layers * 2  # Stronger separation in deeper layers
        
        # Move class 0 and class 1 in different directions
        data[class_labels == 0] += np.random.randn(n_features) * separation
        data[class_labels == 1] -= np.random.randn(n_features) * separation
        
        layer_activations[layer_name] = data
    
    log(f"[SUCCESS] Generated synthetic activations for {n_layers} layers")
    return layer_activations, layer_activations  # Return same for baseline and best

def generate_3d_embedding(activations, use_umap=True):
    """Generate 3D embeddings of activation data."""
    embeddings = {}
    
    if use_umap:
        try:
            # Try to use real UMAP
            embedder = Embedder(n_components=3, random_state=42)
            for layer, data in activations.items():
                log(f"Applying UMAP to {layer} data...")
                embeddings[layer] = embedder.fit_transform(data)
        except Exception as e:
            log(f"[ERROR] Error using UMAP: {e}")
            log("   Using synthetic embeddings instead")
            use_umap = False
    
    if not use_umap:
        # Just create synthetic embeddings directly in 3D
        log("Creating synthetic 3D embeddings...")
        n_samples = list(activations.values())[0].shape[0]
        
        for layer_idx, (layer, data) in enumerate(activations.items()):
            # Create synthetic 3D points
            # We'll create two clusters that separate more in deeper layers
            embedding = np.random.randn(n_samples, 3) * 0.5
            
            # Get class info from the high-dim data (just for demonstration)
            # In real data, we'd use class labels if available
            class_center = np.mean(data, axis=0)
            synthetic_classes = (data[:, 0] > np.median(data[:, 0])).astype(int)
            
            # Apply transformations to simulate layer behavior
            # As we go deeper in the network, classes should separate more
            layer_factor = layer_idx / len(activations)
            
            # Move the classes apart
            for cls in [0, 1]:
                mask = synthetic_classes == cls
                direction = np.array([1.0 if cls == 0 else -1.0, 
                                      0.5 if cls == 0 else -0.5, 
                                      0.0])
                embedding[mask] += direction * layer_factor * 5
            
            embeddings[layer] = embedding
    
    log(f"[SUCCESS] Created 3D embeddings for {len(embeddings)} layers")
    return embeddings

def create_visualization(baseline_embeddings, best_embeddings, output_dir=None):
    """Create and save trajectory visualization."""
    # Prepare dictionary format
    embeddings_dict = {
        "baseline": {0: baseline_embeddings},
        "regularized": {0: best_embeddings}
    }
    
    # Create the multi-panel figure
    log("Building visualization...")
    fig = build_multi_panel(
        embeddings_dict,
        samples_to_plot=np.arange(min(50, list(baseline_embeddings.values())[0].shape[0])).tolist(),
        highlight_indices=np.arange(5).tolist(),  # Highlight first 5 points
        title="Neural Network Layer Trajectories",
        show_arrows=True,
        normalize=True
    )
    
    # Save figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_trajectories.html")
        save_figure(fig, output_path, format="html")
        log(f"[SUCCESS] Saved visualization to {output_path}")
    
    log("[SUCCESS] Visualization created successfully")
    return fig

def main():
    """Run visualization tests."""
    log("\n=== Testing Concept Fragmentation Visualization ===\n")
    
    # Try to load actual data
    baseline_activations, best_activations = load_real_data()
    
    # If real data not available, use synthetic
    if baseline_activations is None:
        baseline_activations, best_activations = generate_synthetic_data()
    
    # Generate 3D embeddings
    log("\n--- Creating Embeddings ---")
    baseline_embeddings = generate_3d_embedding(baseline_activations)
    best_embeddings = generate_3d_embedding(best_activations)
    
    # Create visualization
    log("\n--- Creating Visualization ---")
    output_dir = os.path.join(parent_dir, "results", "visualizations")
    fig = create_visualization(baseline_embeddings, best_embeddings, output_dir)
    
    log("\n=== Test Completed Successfully ===")
    log(f"Check {output_dir} for visualization output")
    log(f"Test log saved to: {log_file}")

if __name__ == "__main__":
    main() 