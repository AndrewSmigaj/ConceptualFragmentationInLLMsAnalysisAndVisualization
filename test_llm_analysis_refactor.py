#!/usr/bin/env python3
"""
Test script to verify the refactored LLM analysis API.
"""

from concept_fragmentation.llm.analysis import ClusterAnalysis

def test_comprehensive_analysis():
    """Test the new comprehensive analysis API."""
    
    # Create analyzer instance
    analyzer = ClusterAnalysis(
        provider="openai",
        model="gpt-4",
        use_cache=True,
        debug=True
    )
    
    # Sample data
    paths = {
        0: ["L0_C1", "L1_C2", "L2_C1", "L3_C3"],
        1: ["L0_C2", "L1_C1", "L2_C3", "L3_C2"],
        2: ["L0_C1", "L1_C3", "L2_C2", "L3_C1"]
    }
    
    cluster_labels = {
        "L0_C1": "Basic Features",
        "L0_C2": "Simple Patterns",
        "L1_C1": "Mid-level Concepts",
        "L1_C2": "Abstract Features",
        "L1_C3": "Complex Patterns",
        "L2_C1": "High-level Semantic",
        "L2_C2": "Decision Boundary",
        "L2_C3": "Mixed Concepts",
        "L3_C1": "Final Decision A",
        "L3_C2": "Final Decision B",
        "L3_C3": "Final Decision C"
    }
    
    path_demographic_info = {
        0: {"gender": {"male": 0.7, "female": 0.3}, "age_group": {"young": 0.2, "middle": 0.5, "old": 0.3}},
        1: {"gender": {"male": 0.3, "female": 0.7}, "age_group": {"young": 0.6, "middle": 0.3, "old": 0.1}},
        2: {"gender": {"male": 0.5, "female": 0.5}, "age_group": {"young": 0.1, "middle": 0.4, "old": 0.5}}
    }
    
    fragmentation_scores = {
        0: 0.25,
        1: 0.68,
        2: 0.45
    }
    
    # Test synchronous wrapper
    print("Testing comprehensive analysis with bias detection...")
    try:
        result = analyzer.generate_path_narratives_sync(
            paths=paths,
            cluster_labels=cluster_labels,
            fragmentation_scores=fragmentation_scores,
            path_demographic_info=path_demographic_info,
            analysis_categories=['interpretation', 'bias']
        )
        
        print(f"\nResult type: {type(result)}")
        print(f"Result length: {len(result)} characters")
        print(f"\nFirst 500 characters of analysis:\n{result[:500]}...")
        
        # Verify it's a string, not a dict
        assert isinstance(result, str), "Expected string result, got dict"
        print("\n✓ Test passed: Returns comprehensive analysis as string")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comprehensive_analysis()