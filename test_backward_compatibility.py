#!/usr/bin/env python3
"""
Backward Compatibility Test for Enhanced GPT2PivotClusterer

This script verifies that the enhanced clusterer with new clustering_method
parameter maintains full backward compatibility with existing pivot and POS experiments.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add the concept_fragmentation package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gpt2_pivot_clusterer import GPT2PivotClusterer


def test_default_behavior():
    """Test that default initialization behavior is unchanged."""
    print("=== Testing Default Initialization ===")
    
    # Test 1: Default initialization (should work exactly as before)
    try:
        clusterer_old_style = GPT2PivotClusterer()
        print(f"[PASS] Default initialization successful")
        print(f"  - clustering_method: {clusterer_old_style.clustering_method}")
        print(f"  - k_range: {clusterer_old_style.k_range}")
        print(f"  - random_state: {clusterer_old_style.random_state}")
        
        if clusterer_old_style.clustering_method != 'kmeans':
            print(f"[FAIL] Default clustering_method should be 'kmeans', got '{clusterer_old_style.clustering_method}'")
            return False
            
    except Exception as e:
        print(f"[FAIL] Default initialization failed: {e}")
        return False
    
    # Test 2: Old-style initialization with explicit parameters (should work exactly as before)
    try:
        clusterer_explicit = GPT2PivotClusterer(k_range=(3, 7), random_state=123)
        print(f"[PASS] Explicit parameter initialization successful")
        print(f"  - clustering_method: {clusterer_explicit.clustering_method}")
        print(f"  - k_range: {clusterer_explicit.k_range}")
        print(f"  - random_state: {clusterer_explicit.random_state}")
        
        if clusterer_explicit.clustering_method != 'kmeans':
            print(f"[FAIL] Default clustering_method should be 'kmeans', got '{clusterer_explicit.clustering_method}'")
            return False
        if clusterer_explicit.k_range != (3, 7):
            print(f"[FAIL] k_range should be (3, 7), got {clusterer_explicit.k_range}")
            return False
        if clusterer_explicit.random_state != 123:
            print(f"[FAIL] random_state should be 123, got {clusterer_explicit.random_state}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Explicit parameter initialization failed: {e}")
        return False
    
    return True


def create_test_activations() -> Dict[str, Any]:
    """Create simple test activations that mimic the existing pivot experiment format."""
    print("Creating test activations for backward compatibility...")
    
    # Simple test data - 3 sentences, 2 tokens each
    activations = {
        0: {  # Sentence 0
            0: {0: [1.0, 2.0, 3.0], 1: [1.1, 2.1, 3.1], 2: [1.2, 2.2, 3.2]},  # Token 0
            1: {0: [4.0, 5.0, 6.0], 1: [4.1, 5.1, 6.1], 2: [4.2, 5.2, 6.2]}   # Token 1
        },
        1: {  # Sentence 1
            0: {0: [7.0, 8.0, 9.0], 1: [7.1, 8.1, 9.1], 2: [7.2, 8.2, 9.2]},   # Token 0
            1: {0: [10.0, 11.0, 12.0], 1: [10.1, 11.1, 12.1], 2: [10.2, 11.2, 12.2]}  # Token 1
        },
        2: {  # Sentence 2
            0: {0: [13.0, 14.0, 15.0], 1: [13.1, 14.1, 15.1], 2: [13.2, 14.2, 15.2]},  # Token 0
            1: {0: [16.0, 17.0, 18.0], 1: [16.1, 17.1, 18.1], 2: [16.2, 17.2, 18.2]}   # Token 1
        }
    }
    
    return {
        'activations': activations,
        'sentences': {0: "test sentence", 1: "another test", 2: "final test"},
        'labels': {0: "test", 1: "test", 2: "test"},
        'metadata': {
            'model_name': 'gpt2',
            'num_layers': 3,
            'hidden_dim': 3,
            'dataset': 'backward_compatibility_test',
            'experiment_type': 'pivot'
        }
    }


def test_clustering_functionality():
    """Test that clustering functionality produces expected results."""
    print("\n=== Testing Clustering Functionality ===")
    
    # Create test data
    test_data = create_test_activations()
    
    # Test with default (k-means) clustering
    try:
        clusterer = GPT2PivotClusterer(k_range=(2, 3), random_state=42)
        results = clusterer.cluster_all_layers(test_data)
        
        print("[PASS] Clustering completed successfully")
        
        # Verify result structure
        required_keys = ['layer_results', 'token_paths', 'sentences', 'labels', 'metadata']
        for key in required_keys:
            if key not in results:
                print(f"[FAIL] Missing required key in results: {key}")
                return False
        
        print(f"[PASS] All required result keys present: {required_keys}")
        
        # Verify metadata
        if results['metadata']['clustering_method'] != 'kmeans':
            print(f"[FAIL] Expected clustering_method 'kmeans', got '{results['metadata']['clustering_method']}'")
            return False
        
        print(f"[PASS] Metadata correctly indicates clustering_method: {results['metadata']['clustering_method']}")
        
        # Verify layer results structure
        expected_layers = 3
        if len(results['layer_results']) != expected_layers:
            print(f"[FAIL] Expected {expected_layers} layers, got {len(results['layer_results'])}")
            return False
        
        print(f"[PASS] Correct number of layers processed: {len(results['layer_results'])}")
        
        # Verify token paths structure
        expected_sentences = 3
        if len(results['token_paths']) != expected_sentences:
            print(f"[FAIL] Expected {expected_sentences} sentences in token paths, got {len(results['token_paths'])}")
            return False
        
        print(f"[PASS] Correct number of token paths: {len(results['token_paths'])}")
        
        # Verify each token has a path through all layers
        for sent_idx, tokens in results['token_paths'].items():
            for token_idx, path in tokens.items():
                if len(path) != expected_layers:
                    print(f"[FAIL] Token path should have {expected_layers} elements, got {len(path)} for sentence {sent_idx}, token {token_idx}")
                    return False
        
        print(f"[PASS] All token paths have correct length ({expected_layers} layers)")
        
        # Verify layer result structure
        for layer_key, layer_data in results['layer_results'].items():
            required_layer_keys = ['cluster_labels', 'cluster_centers', 'optimal_k', 'silhouette_score', 'layer_idx']
            for key in required_layer_keys:
                if key not in layer_data:
                    print(f"[FAIL] Missing required key in layer {layer_key}: {key}")
                    return False
        
        print(f"[PASS] All layer results have correct structure")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_format_compatibility():
    """Test that result format is exactly compatible with existing code."""
    print("\n=== Testing Result Format Compatibility ===")
    
    test_data = create_test_activations()
    
    try:
        # Test old-style clusterer (should behave exactly the same)
        old_style_clusterer = GPT2PivotClusterer(k_range=(2, 3), random_state=42)
        old_results = old_style_clusterer.cluster_all_layers(test_data)
        
        # Test new-style clusterer with explicit kmeans (should behave exactly the same)
        new_style_clusterer = GPT2PivotClusterer(k_range=(2, 3), random_state=42, clustering_method='kmeans')
        new_results = new_style_clusterer.cluster_all_layers(test_data)
        
        # Compare results - they should be identical except for clustering_method in metadata
        print("Comparing old-style vs new-style (explicit kmeans) results...")
        
        # Check structure equivalence
        if set(old_results.keys()) != set(new_results.keys()):
            print(f"[FAIL] Top-level keys differ: {set(old_results.keys())} vs {set(new_results.keys())}")
            return False
        
        # Check layer results equivalence
        if set(old_results['layer_results'].keys()) != set(new_results['layer_results'].keys()):
            print(f"[FAIL] Layer result keys differ")
            return False
        
        # Check token paths equivalence  
        if old_results['token_paths'] != new_results['token_paths']:
            print(f"[FAIL] Token paths differ between old and new style")
            return False
        
        print("[PASS] Old-style and new-style (explicit kmeans) produce identical results")
        
        # Check that metadata properly indicates clustering method
        if old_results['metadata']['clustering_method'] != 'kmeans':
            print(f"[FAIL] Old-style should use 'kmeans', got '{old_results['metadata']['clustering_method']}'")
            return False
            
        if new_results['metadata']['clustering_method'] != 'kmeans':
            print(f"[FAIL] New-style should use 'kmeans', got '{new_results['metadata']['clustering_method']}'")
            return False
        
        print("[PASS] Both methods correctly indicate 'kmeans' in metadata")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Result format compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all backward compatibility tests."""
    print("GPT2PivotClusterer Backward Compatibility Test")
    print("=" * 55)
    
    tests = [
        ("Default Behavior", test_default_behavior),
        ("Clustering Functionality", test_clustering_functionality),
        ("Result Format Compatibility", test_result_format_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                print(f"[PASS] {test_name}: PASSED")
                passed += 1
            else:
                print(f"[FAIL] {test_name}: FAILED")
        except Exception as e:
            print(f"[FAIL] {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 55)
    print("BACKWARD COMPATIBILITY TEST RESULTS")
    print("=" * 55)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[PASS] ALL BACKWARD COMPATIBILITY TESTS PASSED")
        print("+ Existing pivot/POS experiments will work unchanged")
        print("+ Default behavior is preserved")
        print("+ Result format is fully compatible")
        return True
    else:
        print("[FAIL] SOME BACKWARD COMPATIBILITY TESTS FAILED")
        print("! Existing experiments may be affected")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)