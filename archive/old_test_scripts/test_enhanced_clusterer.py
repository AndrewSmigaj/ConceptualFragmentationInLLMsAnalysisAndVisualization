#!/usr/bin/env python3
"""
Test script for enhanced GPT2PivotClusterer with semantic subtypes dataset.

This script tests both k-means and HDBSCAN clustering methods with the 774-word
semantic subtypes dataset to verify the enhanced clusterer functionality.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add the concept_fragmentation package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gpt2_pivot_clusterer import GPT2PivotClusterer
# Load curated words from existing file


def create_mock_activations(words: List[str], num_layers: int = 13, hidden_dim: int = 768) -> Dict[str, Any]:
    """
    Create mock activation data for testing the clusterer.
    
    Args:
        words: List of words to create activations for
        num_layers: Number of transformer layers
        hidden_dim: Hidden dimension size
        
    Returns:
        Mock activations data in the expected format
    """
    print(f"Creating mock activations for {len(words)} words...")
    
    # Create mock activations for each word
    activations = {}
    sentences = {}
    labels = {}
    
    for sent_idx, word in enumerate(words):
        # Create one sentence per word (single token)
        sentences[sent_idx] = word
        labels[sent_idx] = f"semantic_subtype_word_{sent_idx}"
        
        # Create mock activations for this word (token)
        activations[sent_idx] = {
            0: {}  # Only one token per sentence
        }
        
        for layer_idx in range(num_layers):
            # Create realistic-looking random activations
            # Use word hash to make activations somewhat consistent per word
            np.random.seed(hash(word) % (2**32 - 1))
            activation = np.random.randn(hidden_dim).tolist()
            activations[sent_idx][0][layer_idx] = activation
    
    return {
        'activations': activations,
        'sentences': sentences,
        'labels': labels,
        'metadata': {
            'model_name': 'gpt2',
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'dataset': 'semantic_subtypes_mock',
            'num_words': len(words)
        }
    }


def test_clustering_method(clusterer: GPT2PivotClusterer, activations_data: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """
    Test a specific clustering method.
    
    Args:
        clusterer: The clusterer instance
        activations_data: Mock activation data
        method_name: Name of the clustering method for logging
        
    Returns:
        Clustering results
    """
    print(f"\n{'='*60}")
    print(f"Testing {method_name} clustering...")
    print(f"{'='*60}")
    
    try:
        results = clusterer.cluster_all_layers(activations_data)
        
        # Print summary statistics
        print(f"\nClustering completed successfully!")
        print(f"Method: {results['metadata']['clustering_method']}")
        print(f"sklearn available: {results['metadata']['sklearn_available']}")
        print(f"HDBSCAN available: {results['metadata']['hdbscan_available']}")
        print(f"Number of layers processed: {len(results['layer_results'])}")
        print(f"Number of token paths: {sum(len(paths) for paths in results['token_paths'].values())}")
        
        # Print layer-wise statistics
        print(f"\nLayer-wise clustering statistics:")
        for layer_key, layer_data in results['layer_results'].items():
            print(f"  {layer_key}: k={layer_data['optimal_k']}, silhouette={layer_data['silhouette_score']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"Error during {method_name} clustering: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    print("Enhanced GPT2PivotClusterer Test")
    print("="*50)
    
    # Load the semantic subtypes words
    try:
        print("Loading semantic subtypes words...")
        with open("gpt2_semantic_subtypes_curated.json", "r") as f:
            curated_data = json.load(f)
        
        # Extract just the words (flatten the nested structure)
        all_words = []
        if "curated_words" in curated_data:
            for subtype_name, word_list in curated_data["curated_words"].items():
                all_words.extend(word_list)
        elif "curated_subtypes" in curated_data:
            for subtype_name, subtype_data in curated_data["curated_subtypes"].items():
                all_words.extend(subtype_data['words'])
        
        print(f"Loaded {len(all_words)} validated words from semantic subtypes dataset")
        
        # Limit to first 50 words for faster testing
        test_words = all_words[:50]
        print(f"Using first {len(test_words)} words for testing")
        
    except Exception as e:
        print(f"Could not load semantic subtypes words: {e}")
        print("Using fallback test words...")
        test_words = ["cat", "dog", "house", "run", "big", "quickly", "love", "think"]
    
    # Create mock activations
    activations_data = create_mock_activations(test_words)
    
    # Test 1: K-means clustering
    print("\n" + "="*60)
    print("TEST 1: K-means Clustering")
    print("="*60)
    
    kmeans_clusterer = GPT2PivotClusterer(
        k_range=(2, 8),
        random_state=42,
        clustering_method='kmeans'
    )
    
    kmeans_results = test_clustering_method(kmeans_clusterer, activations_data, "K-means")
    
    # Test 2: HDBSCAN clustering
    print("\n" + "="*60)
    print("TEST 2: HDBSCAN Clustering")
    print("="*60)
    
    hdbscan_clusterer = GPT2PivotClusterer(
        k_range=(2, 8),
        random_state=42,
        clustering_method='hdbscan'
    )
    
    hdbscan_results = test_clustering_method(hdbscan_clusterer, activations_data, "HDBSCAN")
    
    # Compare results if both succeeded
    if kmeans_results and hdbscan_results:
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        print(f"K-means avg silhouette: {np.mean([layer['silhouette_score'] for layer in kmeans_results['layer_results'].values()]):.3f}")
        print(f"HDBSCAN avg silhouette: {np.mean([layer['silhouette_score'] for layer in hdbscan_results['layer_results'].values()]):.3f}")
        
        # Compare number of clusters found
        kmeans_clusters = [layer['optimal_k'] for layer in kmeans_results['layer_results'].values()]
        hdbscan_clusters = [layer['optimal_k'] for layer in hdbscan_results['layer_results'].values()]
        
        print(f"K-means cluster counts: {kmeans_clusters}")
        print(f"HDBSCAN cluster counts: {hdbscan_clusters}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    
    if kmeans_results:
        print("[PASS] K-means clustering: PASSED")
    else:
        print("[FAIL] K-means clustering: FAILED")
    
    if hdbscan_results:
        print("[PASS] HDBSCAN clustering: PASSED")
    else:
        print("[FAIL] HDBSCAN clustering: FAILED")
    
    # Test invalid clustering method
    print("\n" + "="*60)
    print("TEST 3: Invalid Clustering Method")
    print("="*60)
    
    try:
        invalid_clusterer = GPT2PivotClusterer(clustering_method='invalid_method')
        invalid_results = invalid_clusterer.cluster_all_layers(activations_data)
        print("[FAIL] Invalid method test: FAILED (should have raised an error)")
    except ValueError as e:
        print(f"[PASS] Invalid method test: PASSED (correctly raised error: {e})")
    except Exception as e:
        print(f"[FAIL] Invalid method test: FAILED (unexpected error: {e})")


if __name__ == "__main__":
    main()