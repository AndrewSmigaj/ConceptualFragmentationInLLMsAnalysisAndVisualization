#!/usr/bin/env python3
"""
Check if different layers have identical cluster assignments.
"""

import json
from pathlib import Path
import numpy as np
from collections import Counter

def check_layer_similarity():
    """Compare cluster assignments across layers."""
    base_dir = Path(__file__).parent
    
    # Load clustering results
    clustering_path = base_dir / "clustering_results_k10" / "clustering_results_k10.json"
    with open(clustering_path, 'r') as f:
        clustering_data = json.load(f)
    
    # Extract cluster assignments for each layer
    layer_assignments = {}
    for layer in range(12):
        layer_key = f"layer_{layer}"
        if layer_key in clustering_data['cluster_assignments']:
            layer_assignments[layer] = clustering_data['cluster_assignments'][layer_key]
    
    # Compare consecutive layers
    print("LAYER SIMILARITY ANALYSIS")
    print("=" * 60)
    print("\nComparing consecutive layers:")
    
    for layer in range(11):
        if layer in layer_assignments and (layer + 1) in layer_assignments:
            assignments1 = layer_assignments[layer]
            assignments2 = layer_assignments[layer + 1]
            
            # Count identical assignments
            identical = sum(1 for token_id in assignments1 
                          if token_id in assignments2 and 
                          assignments1[token_id] == assignments2[token_id])
            
            total = len(set(assignments1.keys()) & set(assignments2.keys()))
            if total > 0:
                similarity = (identical / total) * 100
                print(f"\nL{layer} vs L{layer+1}:")
                print(f"  Identical assignments: {identical}/{total} ({similarity:.1f}%)")
                
                # Show cluster distribution differences
                dist1 = Counter(assignments1.values())
                dist2 = Counter(assignments2.values())
                
                print("  Cluster size changes:")
                for cluster in range(10):
                    size1 = dist1.get(cluster, 0)
                    size2 = dist2.get(cluster, 0)
                    change = size2 - size1
                    if abs(change) > 50:  # Only show significant changes
                        print(f"    Cluster {cluster}: {size1} → {size2} ({change:+d})")
    
    # Check if L8 and L9 are identical
    print("\n" + "=" * 60)
    print("DETAILED L8 vs L9 COMPARISON:")
    
    if 8 in layer_assignments and 9 in layer_assignments:
        l8_assignments = layer_assignments[8]
        l9_assignments = layer_assignments[9]
        
        # Find tokens that changed clusters
        changed_tokens = []
        for token_id in l8_assignments:
            if token_id in l9_assignments:
                if l8_assignments[token_id] != l9_assignments[token_id]:
                    changed_tokens.append((token_id, l8_assignments[token_id], l9_assignments[token_id]))
        
        print(f"\nTokens that changed clusters: {len(changed_tokens)}")
        
        if len(changed_tokens) > 0:
            print("\nFirst 20 tokens that changed:")
            # Load token names
            with open(base_dir / "llm_labels_k10" / "llm_labeling_data.json", 'r') as f:
                labeling_data = json.load(f)
            
            # Create token lookup
            token_lookup = {}
            for cluster_key, cluster_info in labeling_data['clusters'].items():
                for i, token in enumerate(cluster_info['common_tokens']):
                    # This is approximate - we'd need the actual token mappings
                    pass
            
            for i, (token_id, old_cluster, new_cluster) in enumerate(changed_tokens[:20]):
                print(f"  Token {token_id}: L8_C{old_cluster} → L9_C{new_cluster}")
        
        # Show movement matrix
        print("\nCluster movement matrix (rows=L8, cols=L9):")
        movement_matrix = np.zeros((10, 10), dtype=int)
        for token_id in l8_assignments:
            if token_id in l9_assignments:
                old_cluster = l8_assignments[token_id]
                new_cluster = l9_assignments[token_id]
                movement_matrix[old_cluster, new_cluster] += 1
        
        # Print matrix with row/column labels
        print("     ", end="")
        for j in range(10):
            print(f"C{j:1d}", end="   ")
        print()
        
        for i in range(10):
            print(f"C{i}: ", end="")
            for j in range(10):
                count = movement_matrix[i, j]
                if count > 0:
                    if i == j:
                        print(f"\033[92m{count:4d}\033[0m", end=" ")  # Green for diagonal
                    else:
                        print(f"{count:4d}", end=" ")
                else:
                    print("   .", end=" ")
            print()
        
        # Calculate how many tokens stayed in same cluster
        diagonal_sum = np.trace(movement_matrix)
        total_tokens = np.sum(movement_matrix)
        stayed_percentage = (diagonal_sum / total_tokens) * 100
        
        print(f"\nTokens that stayed in same cluster: {diagonal_sum}/{total_tokens} ({stayed_percentage:.1f}%)")
        print(f"Tokens that moved to different cluster: {total_tokens - diagonal_sum} ({100 - stayed_percentage:.1f}%)")


if __name__ == "__main__":
    check_layer_similarity()