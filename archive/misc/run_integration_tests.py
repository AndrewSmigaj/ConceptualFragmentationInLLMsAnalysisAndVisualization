"""
Runner script for GPT-2 visualization integration tests.

This script runs the complete suite of integration tests for GPT-2 visualization
components, including data pipeline, visualization generation, persistence, and
performance tests.
"""

import sys
import os
import unittest
import time
from pathlib import Path
import argparse
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_integration_tests(
    test_classes: Optional[List[str]] = None,
    verbosity: int = 2,
    failfast: bool = False,
    buffer: bool = True
) -> bool:
    """
    Run GPT-2 integration tests.
    
    Args:
        test_classes: Specific test classes to run (None for all)
        verbosity: Verbosity level (0-2)
        failfast: Stop on first failure
        buffer: Buffer stdout/stderr during tests
        
    Returns:
        True if all tests passed, False otherwise
    """
    print("GPT-2 Visualization Integration Tests")
    print("=" * 50)
    
    try:
        # Import test modules
        from visualization.tests.test_gpt2_integration import (
            TestGPT2DataPipeline,
            TestGPT2VisualizationGeneration,
            TestGPT2PersistenceIntegration,
            TestGPT2PerformanceMetrics,
            create_test_suite
        )
        
        # Create test suite
        if test_classes:
            # Run specific test classes
            suite = unittest.TestSuite()
            
            class_map = {
                'pipeline': TestGPT2DataPipeline,
                'visualization': TestGPT2VisualizationGeneration,
                'persistence': TestGPT2PersistenceIntegration,
                'performance': TestGPT2PerformanceMetrics
            }
            
            for test_class_name in test_classes:
                if test_class_name in class_map:
                    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(class_map[test_class_name]))
                else:
                    print(f"Warning: Unknown test class '{test_class_name}'")
                    print(f"Available classes: {list(class_map.keys())}")
        else:
            # Run all tests
            suite = create_test_suite()
        
        # Configure test runner
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            failfast=failfast,
            buffer=buffer
        )
        
        # Record start time
        start_time = time.time()
        
        # Run tests
        print(f"Running {suite.countTestCases()} test cases...")
        print("-" * 50)
        
        result = runner.run(suite)
        
        # Calculate runtime
        runtime = time.time() - start_time
        
        # Print summary
        print("-" * 50)
        print(f"Tests completed in {runtime:.2f} seconds")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
        
        if result.wasSuccessful():
            print("\\nAll tests PASSED!")
            return True
        else:
            print("\\nSome tests FAILED!")
            
            # Print details of failures
            if result.failures:
                print("\\nFailures:")
                for i, (test, trace) in enumerate(result.failures, 1):
                    print(f"  {i}. {test}")
                    print(f"     {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
            
            if result.errors:
                print("\\nErrors:")
                for i, (test, trace) in enumerate(result.errors, 1):
                    print(f"  {i}. {test}")
                    print(f"     {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
            
            return False
            
    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure all required dependencies are installed.")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_specific_test(test_name: str, verbosity: int = 2) -> bool:
    """
    Run a specific test method.
    
    Args:
        test_name: Name of the test method (e.g., 'TestGPT2DataPipeline.test_data_loading_pipeline')
        verbosity: Verbosity level
        
    Returns:
        True if test passed, False otherwise
    """
    try:
        # Run specific test
        suite = unittest.TestLoader().loadTestsFromName(
            f"visualization.tests.test_gpt2_integration.{test_name}"
        )
        
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"Error running specific test: {e}")
        return False


def list_available_tests():
    """List all available tests."""
    try:
        from visualization.tests.test_gpt2_integration import create_test_suite
        
        suite = create_test_suite()
        
        print("Available Integration Tests:")
        print("=" * 30)
        
        test_classes = {
            'pipeline': [],
            'visualization': [],
            'persistence': [],
            'performance': []
        }
        
        # Group tests by class
        def collect_tests(test_suite):
            """Recursively collect all test cases from a test suite."""
            tests = []
            for test in test_suite:
                if hasattr(test, '_tests'):  # It's a test suite
                    tests.extend(collect_tests(test))
                else:  # It's a test case
                    tests.append(test)
            return tests
        
        all_tests = collect_tests(suite)
        
        for test in all_tests:
            test_id = test.id()
            class_name = test_id.split('.')[-2]  # Get class name
            method_name = test_id.split('.')[-1]  # Get method name
            
            if 'Pipeline' in class_name:
                test_classes['pipeline'].append(method_name)
            elif 'Visualization' in class_name:
                test_classes['visualization'].append(method_name)
            elif 'Persistence' in class_name:
                test_classes['persistence'].append(method_name)
            elif 'Performance' in class_name:
                test_classes['performance'].append(method_name)
        
        # Print grouped tests
        for category, tests in test_classes.items():
            if tests:
                print(f"\\n{category.capitalize()} Tests:")
                for test in tests:
                    print(f"  - {test}")
        
        print(f"\\nTotal: {suite.countTestCases()} test cases")
        
    except Exception as e:
        print(f"Error listing tests: {e}")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run GPT-2 visualization integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_integration_tests.py                    # Run all tests
  python run_integration_tests.py --classes pipeline # Run only pipeline tests
  python run_integration_tests.py --list            # List available tests
  python run_integration_tests.py --test TestGPT2DataPipeline.test_data_loading_pipeline
        """
    )
    
    parser.add_argument(
        '--classes',
        nargs='+',
        choices=['pipeline', 'visualization', 'persistence', 'performance'],
        help='Specific test classes to run'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Run a specific test method'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available tests'
    )
    
    parser.add_argument(
        '--verbosity', '-v',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Verbosity level (default: 2)'
    )
    
    parser.add_argument(
        '--failfast',
        action='store_true',
        help='Stop on first failure'
    )
    
    parser.add_argument(
        '--no-buffer',
        action='store_true',
        help='Don\'t buffer stdout/stderr during tests'
    )
    
    args = parser.parse_args()
    
    # Handle different command options
    if args.list:
        list_available_tests()
        return
    
    if args.test:
        success = run_specific_test(args.test, args.verbosity)
        sys.exit(0 if success else 1)
    
    # Run integration tests
    success = run_integration_tests(
        test_classes=args.classes,
        verbosity=args.verbosity,
        failfast=args.failfast,
        buffer=not args.no_buffer
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()