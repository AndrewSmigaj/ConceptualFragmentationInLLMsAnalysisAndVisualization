"""
Test script for cross-layer metrics dashboard integration.

This script tests the integration between cross-layer metrics computation
and visualization components of the dashboard.
"""

import os
import sys
import numpy as np
import networkx as nx
from typing import Dict, Any

# Add parent directory to path to import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from concept_fragmentation.metrics.cross_layer_metrics import (
    compute_centroid_similarity, compute_membership_overlap,
    compute_trajectory_fragmentation, compute_path_density,
    analyze_cross_layer_metrics
)
from visualization.cross_layer_utils import (
    serialize_cross_layer_metrics, deserialize_cross_layer_metrics,
    networkx_to_dict, dict_to_networkx
)

# Test data generation functions
def generate_test_layer_clusters() -> Dict[str, Dict[str, Any]]:
    """Generate test layer clusters for testing."""
    np.random.seed(42)
    
    layer_clusters = {}
    n_samples = 100
    n_features = 10
    n_layers = 3
    
    for i in range(n_layers):
        layer_name = f"layer{i+1}"
        n_clusters = i + 2  # Different number of clusters per layer
        
        # Generate random cluster labels
        labels = np.random.randint(0, n_clusters, size=n_samples)
        
        # Generate random centroids
        centers = np.random.randn(n_clusters, n_features)
        
        # Store in dictionary
        layer_clusters[layer_name] = {
            "labels": labels,
            "centers": centers,
            "k": n_clusters
        }
    
    return layer_clusters

def test_cross_layer_metrics_computation():
    """Test cross-layer metrics computation."""
    print("Testing cross-layer metrics computation...")
    
    # Generate test data
    layer_clusters = generate_test_layer_clusters()
    class_labels = np.random.randint(0, 2, size=100)  # Binary classification
    
    # Compute metrics using the wrapper function
    metrics = analyze_cross_layer_metrics(
        layer_clusters=layer_clusters,
        class_labels=class_labels,
        config={"min_overlap": 0.1}
    )
    
    # Check that all expected metrics are computed
    assert "centroid_similarity" in metrics, "Centroid similarity not computed"
    assert "membership_overlap" in metrics, "Membership overlap not computed"
    assert "trajectory_fragmentation" in metrics, "Trajectory fragmentation not computed"
    assert "path_density" in metrics, "Path density not computed"
    assert "path_graph" in metrics, "Path graph not computed"
    
    print("✓ All expected metrics were computed successfully")
    
    # Check centroid similarity
    centroid_similarity = metrics["centroid_similarity"]
    assert isinstance(centroid_similarity, dict), "Centroid similarity should be a dictionary"
    assert len(centroid_similarity) > 0, "Centroid similarity should not be empty"
    
    # Check membership overlap
    membership_overlap = metrics["membership_overlap"]
    assert isinstance(membership_overlap, dict), "Membership overlap should be a dictionary"
    assert len(membership_overlap) > 0, "Membership overlap should not be empty"
    
    # Check trajectory fragmentation
    trajectory_fragmentation = metrics["trajectory_fragmentation"]
    assert isinstance(trajectory_fragmentation, dict), "Trajectory fragmentation should be a dictionary"
    assert len(trajectory_fragmentation) > 0, "Trajectory fragmentation should not be empty"
    
    # Check path density
    path_density = metrics["path_density"]
    assert isinstance(path_density, dict), "Path density should be a dictionary"
    assert len(path_density) > 0, "Path density should not be empty"
    
    # Check path graph
    path_graph = metrics["path_graph"]
    assert isinstance(path_graph, nx.Graph), "Path graph should be a NetworkX graph"
    assert path_graph.number_of_nodes() > 0, "Path graph should have nodes"
    
    print("✓ All metrics have the expected data types and structures")
    
    return metrics

def test_serialization():
    """Test serialization/deserialization of cross-layer metrics."""
    print("\nTesting serialization/deserialization...")
    
    # Compute metrics
    metrics = test_cross_layer_metrics_computation()
    
    # Serialize metrics
    serialized = serialize_cross_layer_metrics(metrics)
    
    # Check that all metrics are serialized
    assert "centroid_similarity" in serialized, "Centroid similarity not serialized"
    assert "membership_overlap" in serialized, "Membership overlap not serialized"
    assert "trajectory_fragmentation" in serialized, "Trajectory fragmentation not serialized"
    assert "path_density" in serialized, "Path density not serialized"
    assert "path_graph" in serialized, "Path graph not serialized"
    
    print("✓ All metrics were serialized successfully")
    
    # Check serialized path graph structure
    path_graph_dict = serialized["path_graph"]
    assert "nodes" in path_graph_dict, "Serialized path graph should have nodes"
    assert "edges" in path_graph_dict, "Serialized path graph should have edges"
    assert "directed" in path_graph_dict, "Serialized path graph should have directed flag"
    
    print("✓ Path graph was serialized with correct structure")
    
    # Deserialize metrics
    deserialized = deserialize_cross_layer_metrics(serialized)
    
    # Check that all metrics are deserialized
    assert "centroid_similarity" in deserialized, "Centroid similarity not deserialized"
    assert "membership_overlap" in deserialized, "Membership overlap not deserialized"
    assert "trajectory_fragmentation" in deserialized, "Trajectory fragmentation not deserialized"
    assert "path_density" in deserialized, "Path density not deserialized"
    assert "path_graph" in deserialized, "Path graph not deserialized"
    
    print("✓ All metrics were deserialized successfully")
    
    # Check deserialized path graph
    path_graph = deserialized["path_graph"]
    assert isinstance(path_graph, nx.Graph), "Deserialized path graph should be a NetworkX graph"
    assert path_graph.number_of_nodes() == metrics["path_graph"].number_of_nodes(), \
        "Deserialized path graph should have same number of nodes as original"
    assert path_graph.number_of_edges() == metrics["path_graph"].number_of_edges(), \
        "Deserialized path graph should have same number of edges as original"
    
    print("✓ Path graph was deserialized correctly")
    
    # Check that the node and edge attributes are preserved
    for node in path_graph.nodes():
        assert "layer" in path_graph.nodes[node], f"Node {node} should have layer attribute"
        assert "cluster_id" in path_graph.nodes[node], f"Node {node} should have cluster_id attribute"
    
    for edge in path_graph.edges():
        assert "weight" in path_graph.edges[edge], f"Edge {edge} should have weight attribute"
    
    print("✓ Node and edge attributes were preserved during serialization/deserialization")

if __name__ == "__main__":
    # Test cross-layer metrics computation
    test_cross_layer_metrics_computation()
    
    # Test serialization/deserialization
    test_serialization()
    
    print("\nAll tests passed!")