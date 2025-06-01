#!/usr/bin/env python3
"""
Test the refactored comprehensive LLM analysis with real API calls.
"""

import os
from concept_fragmentation.llm.analysis import ClusterAnalysis

# Import API keys from local config
try:
    from local_config import OPENAI_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_KEY
except ImportError:
    print("ERROR: Please create local_config.py with your API keys")
    print("See local_config.py.example for the format")
    exit(1)

def test_comprehensive_analysis():
    """Test the comprehensive analysis with bias detection."""
    
    print("Creating ClusterAnalysis instance...")
    analyzer = ClusterAnalysis(
        provider="openai",
        model="gpt-4",
        use_cache=True,
        debug=False  # Set to True to see prompts
    )
    
    # Create realistic test data for a neural network analyzing heart disease
    paths = {
        0: ["L0_C1", "L1_C2", "L2_C1", "L3_C1"],  # Path 0: Healthy pattern
        1: ["L0_C2", "L1_C1", "L2_C3", "L3_C2"],  # Path 1: At-risk pattern
        2: ["L0_C3", "L1_C3", "L2_C2", "L3_C2"],  # Path 2: Disease pattern
        3: ["L0_C1", "L1_C1", "L2_C2", "L3_C1"],  # Path 3: Mixed pattern
    }
    
    cluster_labels = {
        # Layer 0 - Basic features
        "L0_C1": "Normal vitals",
        "L0_C2": "Elevated markers", 
        "L0_C3": "Abnormal readings",
        # Layer 1 - Feature combinations
        "L1_C1": "Cardiovascular stress",
        "L1_C2": "Healthy metabolism",
        "L1_C3": "Multiple risk factors",
        # Layer 2 - Higher patterns
        "L2_C1": "Low risk profile",
        "L2_C2": "Moderate risk",
        "L2_C3": "High risk indicators",
        # Layer 3 - Final decisions
        "L3_C1": "No disease",
        "L3_C2": "Heart disease"
    }
    
    # Add demographic info showing potential bias
    path_demographic_info = {
        0: {
            "gender": {"male": 0.3, "female": 0.7},  # More females in healthy path
            "age_group": {"<40": 0.6, "40-60": 0.3, ">60": 0.1},
            "ethnicity": {"white": 0.5, "black": 0.2, "asian": 0.2, "other": 0.1}
        },
        1: {
            "gender": {"male": 0.6, "female": 0.4},  # More males in at-risk
            "age_group": {"<40": 0.2, "40-60": 0.5, ">60": 0.3},
            "ethnicity": {"white": 0.4, "black": 0.3, "asian": 0.2, "other": 0.1}
        },
        2: {
            "gender": {"male": 0.8, "female": 0.2},  # Heavy male bias in disease path
            "age_group": {"<40": 0.1, "40-60": 0.3, ">60": 0.6},
            "ethnicity": {"white": 0.3, "black": 0.4, "asian": 0.2, "other": 0.1}
        },
        3: {
            "gender": {"male": 0.5, "female": 0.5},
            "age_group": {"<40": 0.3, "40-60": 0.4, ">60": 0.3},
            "ethnicity": {"white": 0.4, "black": 0.3, "asian": 0.2, "other": 0.1}
        }
    }
    
    fragmentation_scores = {
        0: 0.15,  # Low fragmentation - consistent healthy path
        1: 0.45,  # Moderate fragmentation
        2: 0.25,  # Low fragmentation - consistent disease path
        3: 0.75   # High fragmentation - unclear patterns
    }
    
    print("\nTesting comprehensive analysis with bias detection...")
    print("="*60)
    
    try:
        # Call the refactored method
        result = analyzer.generate_path_narratives_sync(
            paths=paths,
            cluster_labels=cluster_labels,
            fragmentation_scores=fragmentation_scores,
            path_demographic_info=path_demographic_info,
            analysis_categories=['interpretation', 'bias']  # Request both analyses
        )
        
        # Verify the result
        print(f"\nResult type: {type(result)}")
        print(f"Result length: {len(result)} characters\n")
        
        # Display the full analysis
        print("COMPREHENSIVE ANALYSIS:")
        print("="*60)
        print(result)
        print("="*60)
        
        # Check that bias analysis was included
        if "BIAS ANALYSIS" in result:
            print("\n✓ Bias analysis section found")
        else:
            print("\n✗ Bias analysis section missing!")
            
        if "INTERPRETATION" in result:
            print("✓ Interpretation section found")
        else:
            print("✗ Interpretation section missing!")
            
        # Save the result for review
        with open("llm_analysis_output.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print("\n✓ Analysis saved to llm_analysis_output.txt")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        analyzer.close()
        print("\n✓ Analyzer closed properly")

if __name__ == "__main__":
    test_comprehensive_analysis()