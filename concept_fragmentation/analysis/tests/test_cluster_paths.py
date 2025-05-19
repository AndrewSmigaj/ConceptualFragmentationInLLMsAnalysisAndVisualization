"""
Tests for the cluster path computation and path archetype functionality.
"""

import os
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Any

from concept_fragmentation.analysis.cluster_paths import (
    _natural_layer_sort,
    compute_cluster_paths,
    compute_path_archetypes,
    write_cluster_paths,
    compute_clusters_for_layer
)

def test_natural_layer_sort():
    """Test natural sorting of layer names."""
    layer_names = ["layer10", "layer2", "input", "layer1", "output"]
    expected_order = ["input", "layer1", "layer2", "layer10", "output"]
    
    sorted_layers = sorted(layer_names, key=_natural_layer_sort)
    assert sorted_layers == expected_order

def test_compute_clusters_for_layer():
    """Test computation of clusters for a layer."""
    # Create mock data
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [10, 10], [10, 11], [11, 10], [11, 11]
    ])
    
    # Compute clusters
    k, centers, labels = compute_clusters_for_layer(X, max_k=5, random_state=42)
    
    # Check results
    assert k == 2  # Should find 2 clusters
    assert centers.shape == (2, 2)  # 2 centers with 2 dimensions
    assert labels.shape == (8,)  # 8 labels for 8 data points
    assert len(np.unique(labels)) == 2  # 2 unique cluster labels

def test_compute_cluster_paths():
    """Test computation of cluster paths."""
    # Create mock layer clusters
    layer_clusters = {
        "input": {
            "k": 2,
            "centers": np.array([[0, 0], [1, 1]]),
            "labels": np.array([0, 0, 1, 1, 0])
        },
        "layer1": {
            "k": 3,
            "centers": np.array([[0, 0], [1, 1], [2, 2]]),
            "labels": np.array([0, 2, 1, 1, 2])
        },
        "layer2": {
            "k": 2,
            "centers": np.array([[0, 0], [1, 1]]),
            "labels": np.array([1, 0, 1, 0, 1])
        }
    }
    
    # Expected results
    expected_layer_names = ["input", "layer1", "layer2"]
    expected_paths = np.array([
        [0, 0, 1],
        [0, 2, 0],
        [1, 1, 1],
        [1, 1, 0],
        [0, 2, 1]
    ])
    
    # Compute paths
    paths, layer_names = compute_cluster_paths(layer_clusters)
    
    # Check results
    assert layer_names == expected_layer_names
    np.testing.assert_array_equal(paths, expected_paths)

def test_compute_path_archetypes():
    """Test computation of path archetypes."""
    # Create mock paths
    paths = np.array([
        [0, 0, 1],  # Path 1
        [0, 2, 0],  # Path 2
        [1, 1, 1],  # Path 3
        [1, 1, 0],  # Path 4
        [0, 0, 1],  # Path 1 (duplicate)
    ])
    layer_names = ["input", "layer1", "layer2"]
    
    # Create mock dataframe
    df = pd.DataFrame({
        "survived": [0, 1, 1, 0, 1],
        "age": [20, 30, 40, 50, 25],
        "sex": ["male", "female", "female", "male", "female"],
        "pclass": [3, 1, 2, 1, 3]
    })
    
    # Compute archetypes
    archetypes = compute_path_archetypes(
        paths,
        layer_names,
        df,
        target_column="survived",
        demographic_columns=["age", "sex", "pclass"],
        top_k=2,
        max_members=10
    )
    
    # Check results
    assert len(archetypes) == 2  # Top 2 archetypes
    
    # First archetype should be "0→0→1" with 2 members
    assert archetypes[0]["path"] == "0→0→1"
    assert archetypes[0]["count"] == 2
    assert archetypes[0]["survived_rate"] == 0.5  # 1 out of 2 survived
    assert "demo_stats" in archetypes[0]
    assert "age" in archetypes[0]["demo_stats"]
    assert "sex" in archetypes[0]["demo_stats"]
    assert "pclass" in archetypes[0]["demo_stats"]
    
    # Second archetype should have 1 member
    assert archetypes[1]["count"] == 1

def test_write_cluster_paths(tmp_path):
    """Test writing of cluster paths to JSON."""
    # Create mock layer clusters
    layer_clusters = {
        "input": {
            "k": 2,
            "centers": np.array([[0, 0], [1, 1]]),
            "labels": np.array([0, 0, 1, 1, 0])
        },
        "layer1": {
            "k": 3,
            "centers": np.array([[0, 0], [1, 1], [2, 2]]),
            "labels": np.array([0, 2, 1, 1, 2])
        },
        "layer2": {
            "k": 2,
            "centers": np.array([[0, 0], [1, 1]]),
            "labels": np.array([1, 0, 1, 0, 1])
        }
    }
    
    # Create mock dataframe
    df = pd.DataFrame({
        "survived": [0, 1, 1, 0, 1],
        "age": [20, 30, 40, 50, 25],
        "sex": ["male", "female", "female", "male", "female"],
        "pclass": [3, 1, 2, 1, 3]
    })
    
    # Create temporary output directory
    output_dir = os.path.join(tmp_path, "cluster_paths")
    
    # Write cluster paths
    output_path = write_cluster_paths(
        "titanic",
        0,
        layer_clusters,
        df,
        target_column="survived",
        demographic_columns=["age", "sex", "pclass"],
        output_dir=output_dir,
        top_k=3,
        max_members=10
    )
    
    # Check that the file exists
    assert os.path.exists(output_path)
    
    # Verify file content
    import json
    with open(output_path, 'r') as f:
        data = json.load(f)
    
    # Check basic structure
    assert "dataset" in data
    assert "seed" in data
    assert "layers" in data
    assert "paths" in data
    assert "survived" in data
    assert "path_archetypes" in data
    
    # Check contents
    assert data["dataset"] == "titanic"
    assert data["seed"] == 0
    assert data["layers"] == ["input", "layer1", "layer2"]
    assert len(data["paths"]) == 5  # Number of samples
    assert len(data["survived"]) == 5
    assert len(data["path_archetypes"]) == 3  # Top 3 archetypes 