"""
Simple test of the clustering format conversion logic.
"""
import numpy as np
from collections import Counter

def test_path_format_conversion():
    """Test the path format conversion logic."""
    
    # Simulate what compute_cluster_paths returns
    # human_readable_paths look like: "L0C1→L1C2→L2C0"
    sample_human_readable_paths = [
        "L0C1→L1C2→L2C0→L3C1",
        "L0C0→L1C1→L2C2→L3C0", 
        "L0C2→L1C0→L2C1→L3C2",
        "L0C1→L1C2→L2C0→L3C1",  # Duplicate
        "L0C0→L1C1→L2C2→L3C0",  # Duplicate
    ]
    
    # Simulate original_paths (numeric cluster IDs per layer)
    original_paths = [
        [1, 2, 0, 1],
        [0, 1, 2, 0],
        [2, 0, 1, 2],
        [1, 2, 0, 1],
        [0, 1, 2, 0],
    ]
    
    print("Step 1: Count path frequencies")
    path_counter = Counter(tuple(path) for path in original_paths)
    most_common_paths = path_counter.most_common(3)
    
    for path_tuple, freq in most_common_paths:
        print(f"  Path {path_tuple}: {freq} occurrences")
    
    print("\nStep 2: Convert to LLM format")
    llm_paths = {}
    cluster_labels_dict = {}
    
    for idx, (path_tuple, frequency) in enumerate(most_common_paths):
        # Find a sample with this path
        sample_idx = None
        for i, path in enumerate(original_paths):
            if tuple(path) == path_tuple:
                sample_idx = i
                break
        
        if sample_idx is not None:
            path_str = sample_human_readable_paths[sample_idx]
            print(f"\n  Original: {path_str.encode('ascii', 'replace').decode('ascii')}")
            
            # Convert "L0C1→L1C2→L2C0" to ["L0_C1", "L1_C2", "L2_C0"]
            parts = path_str.split('→')
            llm_path = []
            
            for part in parts:
                # part is like "L0C1"
                # Find where C is and insert underscore
                c_index = part.index('C')
                layer_cluster = part[:c_index] + '_' + part[c_index:]
                llm_path.append(layer_cluster)
            
            print(f"  Converted: {llm_path}")
            llm_paths[idx] = llm_path
            
            # Create labels
            for cluster_id in llm_path:
                if cluster_id not in cluster_labels_dict:
                    parts = cluster_id.split('_')
                    layer_num = int(parts[0][1:])
                    cluster_num = int(parts[1][1:])
                    cluster_labels_dict[cluster_id] = f"Layer {layer_num} Cluster {cluster_num}"
    
    print(f"\nResults:")
    print(f"  LLM paths: {llm_paths}")
    print(f"  Cluster labels: {list(cluster_labels_dict.items())[:5]}")
    
    # Verify format
    for path_id, path in llm_paths.items():
        for cluster_id in path:
            assert cluster_id.count('_') == 1, f"Invalid format: {cluster_id}"
            assert cluster_id.startswith('L'), f"Should start with L: {cluster_id}"
            assert 'C' in cluster_id, f"Should contain C: {cluster_id}"
    
    print("\n✅ Format conversion works correctly!")

if __name__ == "__main__":
    test_path_format_conversion()