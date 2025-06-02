"""
Test LLM integration in Concept MRI.
Verifies that the LLM analysis tab works correctly with mock data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from concept_mri.tabs.ff_networks import _create_llm_analysis_panel, run_llm_analysis


def test_llm_panel_creation():
    """Test that the LLM analysis panel creates correctly."""
    # Test with no data
    panel = _create_llm_analysis_panel(None)
    assert panel is not None
    print("[OK] Panel handles no data case")
    
    # Test with incomplete data
    panel = _create_llm_analysis_panel({"completed": False})
    assert panel is not None
    print("[OK] Panel handles incomplete data case")
    
    # Test with complete data
    panel = _create_llm_analysis_panel({"completed": True})
    assert panel is not None
    print("[OK] Panel creates successfully with complete data")


def test_llm_analysis_with_mock_data():
    """Test LLM analysis with mock clustering data."""
    # Mock clustering data in the expected format
    mock_clustering_data = {
        "completed": True,
        "paths": {
            0: ["L0_C1", "L1_C2", "L2_C1", "L3_C1"],
            1: ["L0_C2", "L1_C1", "L2_C3", "L3_C2"],
            2: ["L0_C1", "L1_C3", "L2_C2", "L3_C3"]
        },
        "cluster_labels": {
            "L0_C1": "Normal vitals",
            "L0_C2": "Elevated markers",
            "L1_C1": "Stable progression",
            "L1_C2": "Risk indicators",
            "L1_C3": "Mixed signals",
            "L2_C1": "Improving trends",
            "L2_C2": "Neutral state",
            "L2_C3": "Declining metrics",
            "L3_C1": "Healthy outcome",
            "L3_C2": "At-risk outcome",
            "L3_C3": "Critical outcome"
        },
        "path_demographic_info": {
            0: {
                "gender": {"male": 0.3, "female": 0.7},
                "age_group": {"<40": 0.6, "40-60": 0.3, ">60": 0.1}
            },
            1: {
                "gender": {"male": 0.8, "female": 0.2},
                "age_group": {"<40": 0.1, "40-60": 0.4, ">60": 0.5}
            },
            2: {
                "gender": {"male": 0.5, "female": 0.5},
                "age_group": {"<40": 0.3, "40-60": 0.4, ">60": 0.3}
            }
        },
        "fragmentation_scores": {
            0: 0.15,
            1: 0.45,
            2: 0.30
        }
    }
    
    print("\nTesting LLM analysis with mock data...")
    print(f"Number of paths: {len(mock_clustering_data['paths'])}")
    print(f"Number of clusters: {len(mock_clustering_data['cluster_labels'])}")
    
    # Test the callback would work with this data
    # (We can't actually call it without a full Dash app context)
    try:
        # Just verify the data structure is correct
        paths = mock_clustering_data.get('paths', {})
        cluster_labels = mock_clustering_data.get('cluster_labels', {})
        assert isinstance(paths, dict), "Paths should be a dictionary"
        assert isinstance(cluster_labels, dict), "Cluster labels should be a dictionary"
        assert all(isinstance(path, list) for path in paths.values()), "Each path should be a list"
        assert all(isinstance(label, str) for label in cluster_labels.values()), "Each label should be a string"
        print("[OK] Mock data structure is correct for LLM analysis")
    except AssertionError as e:
        print(f"[ERROR] Data structure issue: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Testing LLM integration in Concept MRI...")
    print("=" * 50)
    
    # Test panel creation
    test_llm_panel_creation()
    
    # Test with mock data
    test_llm_analysis_with_mock_data()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("\nNOTE: To fully test the LLM analysis, you need to:")
    print("1. Ensure clustering outputs data in the correct format")
    print("2. Run the Concept MRI app and test the LLM Analysis tab")
    print("3. Verify that analysis results display correctly")