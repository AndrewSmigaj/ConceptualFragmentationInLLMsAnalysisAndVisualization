"""
Test clustering format for LLM analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from collections import Counter

# Import clustering functions directly
from concept_fragmentation.analysis.cluster_paths import (
    compute_cluster_paths,
    compute_clusters_for_layer
)

def test_clustering_to_llm_format():
    """Test converting clustering results to LLM format."""
    
    # Create synthetic activations
    np.random.seed(42)
    activations = {
        'layer_0': np.random.randn(100, 32),
        'layer_1': np.random.randn(100, 16), 
        'layer_2': np.random.randn(100, 8),
        'layer_3': np.random.randn(100, 4)
    }
    
    print("Step 1: Clustering each layer...")
    # Step 1: Compute clusters for each layer
    layer_clusters = {}
    for layer_name, layer_activations in activations.items():
        # For testing, use fixed k=3
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(layer_activations)
        centers = kmeans.cluster_centers_
        
        layer_clusters[layer_name] = {
            'k': 3,
            'centers': centers,
            'labels': labels,
            'activations': layer_activations
        }
        print(f"  {layer_name}: {len(np.unique(labels))} clusters")
    
    print("\nStep 2: Computing cluster paths...")
    # Step 2: Compute cluster paths
    unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths = compute_cluster_paths(layer_clusters)
    
    print(f"  Found {len(unique_paths)} samples")
    print(f"  Layers: {layer_names}")
    print(f"  Sample human readable paths:")
    for i in range(min(5, len(human_readable_paths))):
        print(f"    {human_readable_paths[i]}")
    
    print("\nStep 3: Finding archetypal paths...")
    # Step 3: Find most common paths
    path_counter = Counter(tuple(path) for path in original_paths)
    most_common_paths = path_counter.most_common(10)
    
    print(f"  Found {len(path_counter)} unique paths")
    print(f"  Top 3 most common:")
    for path_tuple, freq in most_common_paths[:3]:
        print(f"    {path_tuple}: {freq} samples")
    
    print("\nStep 4: Formatting for LLM...")
    # Step 4: Format for LLM analysis
    llm_paths = {}
    cluster_labels_dict = {}
    fragmentation_scores = {}
    
    for idx, (path_tuple, frequency) in enumerate(most_common_paths):
        # Find a sample that follows this path
        sample_idx = None
        for i, sample_path in enumerate(original_paths):
            if tuple(sample_path) == path_tuple:
                sample_idx = i
                break
        
        if sample_idx is not None:
            # Use the human readable path
            path_str = human_readable_paths[sample_idx]
            print(f"  Converting: {path_str}")
            
            # Convert "L0C1→L1C2→L2C0" to ["L0_C1", "L1_C2", "L2_C0"]
            # Split by arrow and format
            parts = path_str.split('→')
            llm_path = []
            for part in parts:
                # part is like "L0C1"
                # Convert to "L0_C1"
                if 'L' in part and 'C' in part:
                    layer_cluster = part.replace('C', '_C')
                    llm_path.append(layer_cluster)
            
            llm_paths[idx] = llm_path
            print(f"    -> {llm_path}")
            
            # Create cluster labels
            for cluster_id in llm_path:
                if cluster_id not in cluster_labels_dict:
                    # Parse layer and cluster number
                    parts = cluster_id.split('_')
                    layer_num = int(parts[0][1:])
                    cluster_num = int(parts[1][1:])
                    cluster_labels_dict[cluster_id] = f"Layer {layer_num} Cluster {cluster_num}"
            
            # Simple fragmentation score based on frequency
            total_samples = len(original_paths)
            path_frequency = frequency / total_samples
            fragmentation_scores[idx] = 1.0 - min(path_frequency * 5, 1.0)
    
    print(f"\nFinal LLM format:")
    print(f"  Paths: {len(llm_paths)}")
    print(f"  Cluster labels: {len(cluster_labels_dict)}")
    print(f"  Sample path: {llm_paths[0] if llm_paths else 'None'}")
    print(f"  Sample label: {list(cluster_labels_dict.items())[0] if cluster_labels_dict else 'None'}")
    
    # Verify format
    assert len(llm_paths) > 0, "No paths generated"
    assert len(cluster_labels_dict) > 0, "No cluster labels generated"
    
    for path_id, path in llm_paths.items():
        assert isinstance(path, list), f"Path {path_id} is not a list"
        for cluster_id in path:
            assert cluster_id.startswith('L'), f"Cluster ID {cluster_id} doesn't start with 'L'"
            assert '_C' in cluster_id, f"Cluster ID {cluster_id} doesn't contain '_C'"
    
    print("\n✅ All format checks passed!")
    
    return {
        'paths': llm_paths,
        'cluster_labels': cluster_labels_dict,
        'fragmentation_scores': fragmentation_scores
    }

if __name__ == "__main__":
    print("Testing clustering to LLM format conversion...")
    print("=" * 50)
    
    result = test_clustering_to_llm_format()
    
    print("\n" + "=" * 50)
    print("✅ Test completed successfully!")
    print("\nThe clustering panel will output data in this format:")
    print(f"- paths: Dict with {len(result['paths'])} archetypal paths")
    print(f"- cluster_labels: Dict with {len(result['cluster_labels'])} cluster labels") 
    print(f"- fragmentation_scores: Dict with {len(result['fragmentation_scores'])} scores")