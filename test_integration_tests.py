"""
Test script to verify GPT-2 integration tests work correctly.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_integration_test_setup():
    """Test that integration test components can be imported and run."""
    print("Testing GPT-2 integration test setup...")
    
    try:
        # Test data generator import
        from visualization.tests.fixtures.gpt2_test_data import GPT2TestDataGenerator, create_test_analysis_results
        print("Successfully imported test data generators")
        
        # Test validator imports
        from visualization.tests.utils.visualization_validators import (
            PlotlyFigureValidator, DashComponentValidator, DataStructureValidator
        )
        print("Successfully imported validators")
        
        # Test data generation
        generator = GPT2TestDataGenerator(seed=42)
        test_data = create_test_analysis_results(num_layers=2, seq_length=4, batch_size=1)
        print(f"Successfully generated test data with {len(test_data)} top-level keys")
        
        # Test validation
        validator = DataStructureValidator()
        result = validator.validate_token_paths(test_data['token_paths'], check_path_structure=True)
        
        if result:
            print("Test data validation passed")
        else:
            print(f"Test data validation failed: {validator.get_validation_errors()}")
            return False
        
        # Test figure validator
        fig_validator = PlotlyFigureValidator()
        print("Successfully created figure validator")
        
        # Test component validator
        comp_validator = DashComponentValidator()
        print("Successfully created component validator")
        
        # Test basic data structure
        required_keys = ['model_type', 'layers', 'activations', 'token_metadata', 'cluster_labels']
        for key in required_keys:
            if key not in test_data:
                print(f"Missing required key: {key}")
                return False
        
        print("Test data has all required keys")
        
        # Test token paths structure
        token_paths = test_data['token_paths']
        if not isinstance(token_paths, dict):
            print("Token paths is not a dictionary")
            return False
        
        if len(token_paths) == 0:
            print("Token paths is empty")
            return False
        
        # Check first token path structure
        first_token_id = list(token_paths.keys())[0]
        first_path = token_paths[first_token_id]
        
        required_path_keys = ['token_text', 'position', 'cluster_path']
        for key in required_path_keys:
            if key not in first_path:
                print(f"Token path missing key: {key}")
                return False
        
        print("Token paths have correct structure")
        
        # Test attention data structure
        attention_data = test_data['attention_data']
        if not isinstance(attention_data, dict):
            print("Attention data is not a dictionary")
            return False
        
        layers = test_data['layers']
        for layer in layers:
            if layer not in attention_data:
                print(f"Missing attention data for layer: {layer}")
                return False
            
            layer_data = attention_data[layer]
            required_attention_keys = ['entropy', 'head_agreement', 'num_heads']
            for key in required_attention_keys:
                if key not in layer_data:
                    print(f"Layer {layer} missing attention key: {key}")
                    return False
        
        print("Attention data has correct structure")
        
        print("All integration test setup checks passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_integration_imports():
    """Test that we can import the integration test module."""
    print("\\nTesting integration test module imports...")
    
    try:
        # Try to import the integration test module
        from visualization.tests.test_gpt2_integration import (
            TestGPT2DataPipeline,
            TestGPT2VisualizationGeneration,
            TestGPT2PersistenceIntegration,
            create_test_suite
        )
        print("Successfully imported integration test classes")
        
        # Test creating a test suite
        suite = create_test_suite()
        print(f"Successfully created test suite with {suite.countTestCases()} test cases")
        
        return True
        
    except Exception as e:
        print(f"Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minimal_integration_run():
    """Test running a minimal integration test."""
    print("\\nTesting minimal integration test run...")
    
    try:
        import unittest
        from visualization.tests.test_gpt2_integration import TestGPT2DataPipeline
        
        # Create test suite with just one test
        suite = unittest.TestSuite()
        suite.addTest(TestGPT2DataPipeline('test_data_loading_pipeline'))
        
        # Run with minimal output
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("Minimal integration test passed")
            return True
        else:
            print(f"Minimal integration test failed: {len(result.failures)} failures, {len(result.errors)} errors")
            
            # Print first failure/error for debugging
            if result.failures:
                print(f"First failure: {result.failures[0][1]}")
            if result.errors:
                print(f"First error: {result.errors[0][1]}")
            
            return False
        
    except Exception as e:
        print(f"Minimal test run failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration test setup checks."""
    print("GPT-2 Integration Tests Setup Verification")
    print("=" * 50)
    
    success = True
    
    # Test setup
    if not test_integration_test_setup():
        success = False
    
    # Test imports
    if not test_mock_integration_imports():
        success = False
    
    # Test minimal run
    if not test_minimal_integration_run():
        success = False
    
    print("\\n" + "=" * 50)
    if success:
        print("All integration test setup checks PASSED!")
        print("Integration tests are ready to use.")
    else:
        print("Some integration test setup checks FAILED!")
        print("Please review the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)