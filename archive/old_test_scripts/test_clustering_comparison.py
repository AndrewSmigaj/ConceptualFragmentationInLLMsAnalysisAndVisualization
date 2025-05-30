#!/usr/bin/env python3
"""
Test script for GPT-2 clustering comparison utility.

This script tests the ClusteringComparison class with mock data to verify
the comparison functionality works correctly.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# Add the concept_fragmentation package to the path
sys.path.insert(0, str(Path(__file__).parent))

from gpt2_clustering_comparison import ClusteringComparison


def create_mock_clustering_results(method_name: str, num_layers: int = 3, randomize: bool = True) -> Dict[str, Any]:
    """Create mock clustering results for testing."""
    np.random.seed(42 if not randomize else 123)  # Different seeds for different methods
    
    layer_results = {}
    token_paths = {}
    
    for layer_idx in range(num_layers):
        # Create mock layer results
        optimal_k = np.random.randint(2, 6)
        silhouette_score = np.random.uniform(0.1, 0.8)
        
        layer_results[f"layer_{layer_idx}"] = {
            'cluster_labels': {
                0: {0: f"L{layer_idx}C0", 1: f"L{layer_idx}C1"},
                1: {0: f"L{layer_idx}C1", 1: f"L{layer_idx}C0"}
            },
            'cluster_centers': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            'optimal_k': optimal_k,
            'silhouette_score': silhouette_score,
            'layer_idx': layer_idx
        }
    
    # Create mock token paths
    for sent_idx in range(2):  # 2 sentences
        token_paths[sent_idx] = {}
        for token_idx in range(2):  # 2 tokens per sentence
            path = []
            for layer_idx in range(num_layers):
                cluster_id = np.random.randint(0, 2)
                path.append(f"L{layer_idx}C{cluster_id}")
            token_paths[sent_idx][token_idx] = path
    
    return {
        'layer_results': layer_results,
        'token_paths': token_paths,
        'sentences': {0: "test sentence one", 1: "test sentence two"},
        'labels': {0: "test", 1: "test"},
        'metadata': {
            'clustering_method': method_name,
            'num_layers': num_layers,
            'sklearn_available': True,
            'hdbscan_available': method_name == 'hdbscan'
        }
    }


def test_comparison_functionality():
    """Test the clustering comparison functionality."""
    print("Testing ClusteringComparison functionality...")
    
    # Create mock results for both methods
    kmeans_results = create_mock_clustering_results('kmeans', randomize=False)
    hdbscan_results = create_mock_clustering_results('hdbscan', randomize=True)
    
    print(f"Created mock k-means results with {len(kmeans_results['layer_results'])} layers")
    print(f"Created mock HDBSCAN results with {len(hdbscan_results['layer_results'])} layers")
    
    # Test comparison
    comparator = ClusteringComparison()
    
    try:
        comparison = comparator.compare_methods(kmeans_results, hdbscan_results)
        print("[PASS] Comparison completed successfully")
        
        # Test that all expected keys are present
        expected_keys = ['methods', 'layer_comparison', 'overall_metrics', 'path_analysis']
        for key in expected_keys:
            if key not in comparison:
                print(f"[FAIL] Missing key in comparison: {key}")
                return False
        
        print(f"[PASS] All expected keys present: {expected_keys}")
        
        # Test method summaries
        kmeans_summary = comparison['methods']['kmeans']
        hdbscan_summary = comparison['methods']['hdbscan']
        
        if kmeans_summary['method_name'] != 'K-means':
            print(f"[FAIL] Expected K-means method name, got: {kmeans_summary['method_name']}")
            return False
        
        if hdbscan_summary['method_name'] != 'HDBSCAN':
            print(f"[FAIL] Expected HDBSCAN method name, got: {hdbscan_summary['method_name']}")
            return False
        
        print("[PASS] Method summaries have correct names")
        
        # Test layer comparison
        layer_comparison = comparison['layer_comparison']
        expected_layers = len(kmeans_results['layer_results'])
        
        if len(layer_comparison) != expected_layers:
            print(f"[FAIL] Expected {expected_layers} layers in comparison, got {len(layer_comparison)}")
            return False
        
        print(f"[PASS] Layer comparison has correct number of layers: {len(layer_comparison)}")
        
        # Test overall metrics
        overall_metrics = comparison['overall_metrics']
        required_overall_keys = ['silhouette_comparison', 'cluster_count_comparison', 'consistency_analysis']
        
        for key in required_overall_keys:
            if key not in overall_metrics:
                print(f"[FAIL] Missing key in overall_metrics: {key}")
                return False
        
        print(f"[PASS] Overall metrics have all required keys: {required_overall_keys}")
        
        # Test path analysis
        path_analysis = comparison['path_analysis']
        required_path_keys = ['total_paths_compared', 'average_similarity', 'identical_paths', 'different_paths']
        
        for key in required_path_keys:
            if key not in path_analysis:
                print(f"[FAIL] Missing key in path_analysis: {key}")
                return False
        
        print(f"[PASS] Path analysis has all required keys: {required_path_keys}")
        
        # Test that total paths is reasonable
        expected_total_paths = 4  # 2 sentences * 2 tokens each
        if path_analysis['total_paths_compared'] != expected_total_paths:
            print(f"[FAIL] Expected {expected_total_paths} total paths, got {path_analysis['total_paths_compared']}")
            return False
        
        print(f"[PASS] Total paths compared matches expected: {path_analysis['total_paths_compared']}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Comparison failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """Test report generation functionality."""
    print("\nTesting report generation...")
    
    # Create mock results
    kmeans_results = create_mock_clustering_results('kmeans', randomize=False)
    hdbscan_results = create_mock_clustering_results('hdbscan', randomize=True)
    
    comparator = ClusteringComparison()
    comparison = comparator.compare_methods(kmeans_results, hdbscan_results)
    
    try:
        # Test report generation
        report_text = comparator.generate_comparison_report(comparison)
        
        if not report_text:
            print("[FAIL] Report generation returned empty text")
            return False
        
        print("[PASS] Report generation completed successfully")
        
        # Test that report contains expected sections
        expected_sections = [
            "CLUSTERING METHODS COMPARISON REPORT",
            "METHOD SUMMARIES:",
            "K-means Clustering:",
            "HDBSCAN Clustering:",
            "OVERALL COMPARISON:",
            "CONSISTENCY ANALYSIS:",
            "TOKEN PATH ANALYSIS:"
        ]
        
        for section in expected_sections:
            if section not in report_text:
                print(f"[FAIL] Report missing expected section: {section}")
                return False
        
        print(f"[PASS] Report contains all expected sections")
        
        # Test report length (should be substantial)
        if len(report_text) < 1000:
            print(f"[FAIL] Report seems too short: {len(report_text)} characters")
            return False
        
        print(f"[PASS] Report has reasonable length: {len(report_text)} characters")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_saving():
    """Test data saving functionality."""
    print("\nTesting data saving functionality...")
    
    # Create mock results
    kmeans_results = create_mock_clustering_results('kmeans', randomize=False)
    hdbscan_results = create_mock_clustering_results('hdbscan', randomize=True)
    
    comparator = ClusteringComparison()
    comparison = comparator.compare_methods(kmeans_results, hdbscan_results)
    
    try:
        # Test saving comparison data
        test_output_file = "test_comparison_output.json"
        comparator.save_comparison_data(comparison, test_output_file)
        
        # Verify file was created and contains valid JSON
        if not Path(test_output_file).exists():
            print(f"[FAIL] Output file was not created: {test_output_file}")
            return False
        
        # Try to load and parse the JSON
        with open(test_output_file, 'r') as f:
            loaded_comparison = json.load(f)
        
        if not loaded_comparison:
            print("[FAIL] Loaded comparison data is empty")
            return False
        
        print("[PASS] Comparison data saved and loaded successfully")
        
        # Clean up test file
        Path(test_output_file).unlink()
        print("[PASS] Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Data saving test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all clustering comparison tests."""
    print("GPT-2 Clustering Comparison Utility Test")
    print("=" * 50)
    
    tests = [
        ("Comparison Functionality", test_comparison_functionality),
        ("Report Generation", test_report_generation),
        ("Data Saving", test_data_saving)
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
    
    print("\n" + "=" * 50)
    print("CLUSTERING COMPARISON TEST RESULTS")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[PASS] ALL CLUSTERING COMPARISON TESTS PASSED")
        print("+ Clustering comparison utility is working correctly")
        print("+ Report generation is functional")
        print("+ Data saving/loading works properly")
        return True
    else:
        print("[FAIL] SOME CLUSTERING COMPARISON TESTS FAILED")
        print("! Comparison utility may have issues")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)