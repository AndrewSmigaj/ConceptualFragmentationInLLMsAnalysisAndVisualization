"""
Test the full Concept MRI pipeline programmatically.
"""
import sys
import os
import json
import numpy as np
import torch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concept_fragmentation.analysis.cluster_paths import (
    compute_clusters_for_layer,
    compute_cluster_paths
)
from concept_fragmentation.llm.analysis import ClusterAnalysis
from local_config import OPENAI_KEY

def test_full_pipeline():
    """Test the complete pipeline from model loading to LLM analysis."""
    
    print("=" * 60)
    print("Testing Full Concept MRI Pipeline")
    print("=" * 60)
    
    # 1. Load demo model
    print("\n1. Loading demo model...")
    model_path = "concept_mri/demos/synthetic_demo/model_20250601_103926.pt"
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"[OK] Model loaded from {model_path}")
        print(f"  - Architecture: {checkpoint['architecture']}")
        print(f"  - Input size: {checkpoint['input_size']}")
        print(f"  - Output size: {checkpoint['output_size']}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # 2. Load dataset and activations
    print("\n2. Loading dataset and activations...")
    dataset_path = "concept_mri/demos/synthetic_demo/dataset.npz"
    activations_path = "concept_mri/demos/synthetic_demo/sample_activations.npz"
    
    try:
        dataset = np.load(dataset_path)
        activations_data = np.load(activations_path)
        print(f"[OK] Dataset loaded: {dataset['X'].shape}")
        print(f"[OK] Activations loaded:")
        for key in activations_data.files:
            print(f"  - {key}: {activations_data[key].shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return
    
    # 3. Run clustering
    print("\n3. Running clustering...")
    max_k = 5  # Maximum k to try
    layer_clusters = {}
    
    try:
        for layer_name in activations_data.files:
            layer_activations = activations_data[layer_name]
            # Use manual clustering for testing
            from sklearn.cluster import KMeans
            k = 3
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(layer_activations)
            centers = kmeans.cluster_centers_
            
            layer_clusters[layer_name] = {
                'k': k,
                'centers': centers,
                'labels': labels,
                'activations': layer_activations
            }
            
            unique_labels = np.unique(labels)
            print(f"[OK] Clustered {layer_name}: {len(unique_labels)} clusters")
    except Exception as e:
        print(f"[ERROR] Clustering failed: {e}")
        return
    
    # 4. Extract cluster paths
    print("\n4. Extracting cluster paths...")
    try:
        # compute_cluster_paths expects the full layer_clusters dict
        unique_paths, layer_names, id_to_layer_cluster, original_paths, human_readable_paths = compute_cluster_paths(layer_clusters)
        
        print(f"[OK] Extracted {len(human_readable_paths)} sample paths")
        print(f"  - Unique paths: {len(set(tuple(p) for p in original_paths))}")
        
        # Show first few paths
        from collections import Counter
        path_counter = Counter(tuple(path) for path in original_paths)
        most_common = path_counter.most_common(3)
        for i, (path_tuple, count) in enumerate(most_common):
            # Find a sample with this path
            for idx, p in enumerate(original_paths):
                if tuple(p) == path_tuple:
                    # Replace arrow for better terminal compatibility
                    readable_path = human_readable_paths[idx].replace('→', '->')
                    print(f"  - Path {i}: {readable_path} ({count} samples)")
                    break
    except Exception as e:
        print(f"[ERROR] Path extraction failed: {e}")
        return
    
    # 5. Format data for LLM
    print("\n5. Formatting data for LLM...")
    try:
        # Format paths for LLM - use most common paths
        llm_paths = {}
        cluster_labels = {}
        fragmentation_scores = {}
        
        # Get top 25 most common paths
        path_counter = Counter(tuple(path) for path in original_paths)
        most_common_paths = path_counter.most_common(25)
        
        for idx, (path_tuple, frequency) in enumerate(most_common_paths):
            # Find samples with this path
            sample_indices = [i for i, p in enumerate(original_paths) if tuple(p) == path_tuple]
            
            if sample_indices:
                # Use the human readable path from first sample
                path_str = human_readable_paths[sample_indices[0]]
                # Convert "L0C1→L1C2→L2C0" to ["L0_C1", "L1_C2", "L2_C0"]
                parts = path_str.split('→')
                llm_path = []
                for part in parts:
                    # part is like "L0C1", convert to "L0_C1"
                    if 'C' in part:
                        c_index = part.index('C')
                        layer_cluster = part[:c_index] + '_' + part[c_index:]
                        llm_path.append(layer_cluster)
                
                # Add path for this index (representing the archetype)
                llm_paths[idx] = llm_path
                
                # Add to cluster labels if not already there
                for cluster in llm_path:
                    if cluster not in cluster_labels:
                        layer_num = cluster.split('_')[0].replace('L', '')
                        cluster_num = cluster.split('_')[1].replace('C', '')
                        cluster_labels[cluster] = f"Layer {layer_num} Cluster {cluster_num}"
                
                # Simple fragmentation score based on path frequency
                fragmentation_scores[idx] = 1.0 - (frequency / len(original_paths))
        
        clustering_data = {
            'paths': llm_paths,
            'cluster_labels': cluster_labels,
            'fragmentation_scores': fragmentation_scores
        }
        
        print(f"[OK] Formatted data for LLM:")
        print(f"  - Archetypal paths: {len(llm_paths)}")
        print(f"  - Cluster labels: {len(cluster_labels)}")
        print(f"  - Fragmentation scores: {len(fragmentation_scores)}")
        
    except Exception as e:
        print(f"[ERROR] Data formatting failed: {e}")
        return
    
    # 6. Run LLM analysis (if API key available)
    print("\n6. Testing LLM analysis...")
    if not OPENAI_KEY:
        print("[SKIP] No OpenAI API key found")
    else:
        try:
            # Create analyzer
            analyzer = ClusterAnalysis(api_key=OPENAI_KEY)
            
            # Test with just interpretation category
            categories = ['interpretation']
            
            print(f"[OK] Running LLM analysis with categories: {categories}")
            print("     (This may take 10-30 seconds...)")
            
            # Run analysis using the correct method
            result = analyzer.generate_path_narratives_sync(
                paths=clustering_data['paths'],
                cluster_labels=clustering_data['cluster_labels'],
                fragmentation_scores=clustering_data['fragmentation_scores'],
                analysis_categories=categories  # Pass categories for comprehensive analysis
            )
            
            print(f"[OK] LLM analysis completed!")
            print(f"  - Response length: {len(result)} characters")
            
            # Show first part of response
            print("\nFirst 500 characters of response:")
            print("-" * 40)
            print(result[:500] + "...")
            
        except Exception as e:
            print(f"[ERROR] LLM analysis failed: {e}")
    
    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_full_pipeline()