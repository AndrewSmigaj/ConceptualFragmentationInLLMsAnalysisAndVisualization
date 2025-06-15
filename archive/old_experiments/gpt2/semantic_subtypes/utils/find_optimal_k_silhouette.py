#!/usr/bin/env python3
"""
Find optimal k for each layer using silhouette score instead of elbow method.
Also calculate semantic purity to evaluate cluster meaningfulness.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
from collections import Counter
from typing import Dict, List, Tuple

def calculate_semantic_purity(labels: np.ndarray, word_subtypes: List[str]) -> float:
    """
    Calculate semantic purity - how well clusters align with semantic subtypes.
    
    Args:
        labels: Cluster labels for each word
        word_subtypes: Semantic subtype for each word
        
    Returns:
        Purity score (0-1)
    """
    if len(set(labels)) == 1:  # Only one cluster
        return 0.0
        
    total_correct = 0
    
    # For each cluster, count the most common subtype
    unique_labels = np.unique(labels)
    for label in unique_labels:
        # Get words in this cluster
        cluster_mask = labels == label
        cluster_subtypes = [word_subtypes[i] for i in range(len(labels)) if cluster_mask[i]]
        
        # Count most common subtype
        if cluster_subtypes:
            subtype_counts = Counter(cluster_subtypes)
            most_common_count = subtype_counts.most_common(1)[0][1]
            total_correct += most_common_count
    
    # Purity is the fraction of correctly "classified" words
    purity = total_correct / len(labels)
    return purity

def find_optimal_k_for_layer(activations: np.ndarray, 
                           word_subtypes: List[str],
                           k_range: range = range(2, 11),
                           random_state: int = 42) -> Dict:
    """
    Find optimal k using silhouette score and semantic purity.
    
    Args:
        activations: Activation matrix for this layer
        word_subtypes: Semantic subtype for each word
        k_range: Range of k values to try
        random_state: Random seed
        
    Returns:
        Dictionary with results for each k
    """
    results = {}
    
    for k in k_range:
        # Run K-means
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(activations)
        
        # Calculate metrics
        silhouette = silhouette_score(activations, labels)
        purity = calculate_semantic_purity(labels, word_subtypes)
        
        # Get cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        results[k] = {
            'silhouette': silhouette,
            'purity': purity,
            'cluster_sizes': counts.tolist(),
            'size_std': np.std(counts),
            'size_ratio': counts.max() / counts.min()
        }
        
        print(f"  k={k}: silhouette={silhouette:.3f}, purity={purity:.3f}, "
              f"sizes={counts.tolist()}")
    
    return results

def main():
    print("Finding optimal k using silhouette score and semantic purity")
    print("="*60)
    
    # Load activations
    activations_file = Path("activations_by_layer.pkl")
    if not activations_file.exists():
        print("Error: activations_by_layer.pkl not found!")
        return
        
    with open(activations_file, 'rb') as f:
        activations_data = pickle.load(f)
    
    # Load word metadata to get subtypes
    metadata_file = Path("word_metadata.pkl")
    if not metadata_file.exists():
        print("Error: word_metadata.pkl not found!")
        return
        
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    
    # Extract word subtypes
    word_subtypes = []
    semantic_subtypes = metadata.get('semantic_subtypes', {})
    
    # Create word to subtype mapping
    word_to_subtype = {}
    for subtype, words in semantic_subtypes.items():
        for word in words:
            word_to_subtype[word] = subtype
    
    # Get ordered list of subtypes for each word
    tokens = metadata.get('tokens', [])
    for token_list in tokens:
        # tokens are lists, get the first (and likely only) token
        if token_list:
            token = token_list[0]
            word_subtypes.append(word_to_subtype.get(token, 'unknown'))
        else:
            word_subtypes.append('unknown')
    
    print(f"Loaded {len(word_subtypes)} words with semantic subtypes")
    print(f"Subtypes: {set(word_subtypes)}")
    
    # Results for all layers
    all_results = {}
    optimal_config = {}
    
    # Process each layer
    for layer_idx in range(13):
        layer_key = f'layer_{layer_idx}'
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")
        
        activations = activations_data[layer_key]['activations']
        print(f"Activations shape: {activations.shape}")
        
        # Find optimal k
        results = find_optimal_k_for_layer(activations, word_subtypes)
        all_results[layer_key] = results
        
        # Choose best k based on silhouette score
        best_k = None
        best_silhouette = -1
        
        for k, metrics in results.items():
            # We could also weight by purity or size balance
            score = metrics['silhouette']
            
            # Optional: penalize very imbalanced clusters
            if metrics['size_ratio'] > 10:
                score *= 0.9
                
            if score > best_silhouette:
                best_silhouette = score
                best_k = k
        
        print(f"\nBest k={best_k} with silhouette={best_silhouette:.3f}")
        
        # Also show the k with best purity
        best_purity_k = max(results.items(), key=lambda x: x[1]['purity'])[0]
        print(f"Best purity k={best_purity_k} with purity={results[best_purity_k]['purity']:.3f}")
        
        optimal_config[layer_idx] = {
            'optimal_k': best_k,
            'silhouette': best_silhouette,
            'purity': results[best_k]['purity']
        }
    
    # Save results
    output_file = "optimal_k_silhouette_config.json"
    with open(output_file, 'w') as f:
        json.dump({
            'optimal_config': optimal_config,
            'detailed_results': all_results
        }, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Layer':<6} {'Best k':<8} {'Silhouette':<12} {'Purity':<10}")
    print("-"*60)
    
    for layer_idx in range(13):
        config = optimal_config[layer_idx]
        print(f"{layer_idx:<6} {config['optimal_k']:<8} "
              f"{config['silhouette']:<12.3f} {config['purity']:<10.3f}")

if __name__ == "__main__":
    main()