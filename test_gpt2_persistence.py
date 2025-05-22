"""
Test script for GPT-2 persistence functionality.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic persistence functionality."""
    print("Testing GPT-2 persistence functionality...")
    
    try:
        from concept_fragmentation.persistence import GPT2AnalysisPersistence, save_gpt2_analysis, load_gpt2_analysis
        print("Successfully imported persistence modules")
        
        # Test creating persistence manager
        persistence_manager = GPT2AnalysisPersistence(
            base_dir="test_persistence",
            enable_cache=True,
            cache_ttl=3600
        )
        print("Successfully created persistence manager")
        
        # Test sample data
        sample_data = {
            "model_type": "gpt2-test",
            "layers": ["layer_0", "layer_1"],
            "token_metadata": {
                "tokens": ["Hello", " world"],
                "positions": [0, 1]
            }
        }
        
        # Test saving
        analysis_id = persistence_manager.save_analysis_results(
            analysis_data=sample_data,
            model_name="gpt2-test",
            input_text="Hello world"
        )
        print(f"Successfully saved analysis: {analysis_id}")
        
        # Test loading
        loaded_data = persistence_manager.load_analysis_results(analysis_id)
        if loaded_data:
            print("Successfully loaded analysis")
        else:
            print("Failed to load analysis")
            return False
        
        # Test listing
        analyses = persistence_manager.list_analyses()
        print(f"Found {len(analyses)} saved analyses")
        
        # Test export
        export_path = persistence_manager.export_analysis(
            analysis_id=analysis_id,
            export_format="json"
        )
        print(f"Successfully exported to: {export_path}")
        
        print("All tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)