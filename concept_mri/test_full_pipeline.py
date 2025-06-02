"""
Test the full Concept MRI pipeline with demo data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

def test_clustering_format():
    """Test that clustering outputs the correct format for LLM analysis."""
    
    # Create mock model data like what would come from model upload
    model_data = {
        'model_loaded': True,
        'activations': {
            'layer_0': np.random.randn(100, 32),
            'layer_1': np.random.randn(100, 16), 
            'layer_2': np.random.randn(100, 8),
            'layer_3': np.random.randn(100, 4)
        },
        'dataset': {
            'name': 'test_dataset',
            'num_samples': 100,
            'feature_names': ['feature_0', 'feature_1', 'feature_2']
        }
    }
    
    dataset_data = {
        'feature_names': ['feature_0', 'feature_1', 'feature_2']
    }
    
    # We'll simulate the clustering logic directly to avoid import issues
    from concept_fragmentation.analysis.cluster_paths import (
        compute_cluster_paths,
        compute_clusters_for_layer
    )
    
    # Simulate clustering button click
    clustering_results, _, _, status_msg, status_color, _ = run_clustering(
        n_clicks=1,
        algorithm='kmeans',
        k_method='manual',
        manual_k=3,
        distance='euclidean',
        seed=42,
        model_data=model_data,
        dataset_data=dataset_data,
        hierarchy_level=2,
        ets_mode='auto',
        ets_percentile=10,
        ets_min_threshold=1e-5,
        ets_batch_size=1000,
        ets_show_explanations=True,
        ets_compute_similarity=False
    )
    
    print("Clustering Status:", status_msg)
    print("Status Color:", status_color)
    
    if clustering_results:
        print("\nClustering Results Keys:", list(clustering_results.keys()))
        
        # Check LLM-ready format
        if 'paths' in clustering_results:
            print(f"\nNumber of archetypal paths: {len(clustering_results['paths'])}")
            print("Sample paths (first 3):")
            for i in range(min(3, len(clustering_results['paths']))):
                print(f"  Path {i}: {clustering_results['paths'][i]}")
        
        if 'cluster_labels' in clustering_results:
            print(f"\nNumber of cluster labels: {len(clustering_results['cluster_labels'])}")
            print("Sample labels:")
            for label_id, label in list(clustering_results['cluster_labels'].items())[:5]:
                print(f"  {label_id}: {label}")
        
        if 'fragmentation_scores' in clustering_results:
            print(f"\nFragmentation scores: {list(clustering_results['fragmentation_scores'].values())[:5]}")
        
        print(f"\nClustering completed: {clustering_results.get('completed', False)}")
        
        # Verify format matches LLM expectations
        assert 'paths' in clustering_results, "Missing 'paths' in results"
        assert 'cluster_labels' in clustering_results, "Missing 'cluster_labels' in results"
        assert 'completed' in clustering_results, "Missing 'completed' flag"
        
        # Check path format
        for path_id, path in clustering_results['paths'].items():
            assert isinstance(path, list), f"Path {path_id} is not a list"
            for cluster_id in path:
                assert cluster_id.startswith('L'), f"Cluster ID {cluster_id} doesn't start with 'L'"
                assert '_C' in cluster_id, f"Cluster ID {cluster_id} doesn't contain '_C'"
        
        print("\n✅ All format checks passed!")
        return True
    else:
        print("\n❌ Clustering failed!")
        return False

if __name__ == "__main__":
    print("Testing Concept MRI clustering pipeline...")
    print("=" * 50)
    
    success = test_clustering_format()
    
    if success:
        print("\n✅ Pipeline test successful!")
        print("\nNext steps:")
        print("1. Run the Concept MRI app: cd concept_mri && python app.py")
        print("2. Upload the demo model from concept_mri/demos/synthetic_demo/")
        print("3. Run clustering and test LLM analysis")
    else:
        print("\n❌ Pipeline test failed!")