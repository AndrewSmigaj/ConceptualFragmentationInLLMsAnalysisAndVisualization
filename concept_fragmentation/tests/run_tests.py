#!/usr/bin/env python
"""
Script to run the Concept Fragmentation test suite.
"""

import os
import sys
import argparse
import unittest
import subprocess

def run_all_tests():
    """Run all tests using unittest."""
    print("Running all tests...")
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)))
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def run_specific_test(test_name):
    """Run a specific test."""
    print(f"Running {test_name} tests...")
    if 'metrics' in test_name:
        module_name = 'test_metrics'
    elif 'hooks' in test_name:
        module_name = 'test_hooks'
    elif 'e2e' in test_name or 'end' in test_name or 'workflow' in test_name:
        module_name = 'test_e2e'
    elif 'import' in test_name:
        module_name = 'test_imports'
    else:
        print(f"Unknown test name: {test_name}")
        return False
    
    test_loader = unittest.TestLoader()
    test_suite = test_loader.loadTestsFromName(f"concept_fragmentation.tests.{module_name}")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def run_with_pytest():
    """Run tests using pytest if available."""
    print("Running tests with pytest...")
    try:
        result = subprocess.run(['pytest', os.path.dirname(os.path.abspath(__file__))], 
                                capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors: {result.stderr}")
        return result.returncode == 0
    except FileNotFoundError:
        print("Pytest not found. Please install it with 'pip install pytest'")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the Concept Fragmentation test suite.")
    parser.add_argument('--test', choices=['all', 'metrics', 'hooks', 'e2e', 'imports'],
                        default='all', help='Specify which tests to run')
    parser.add_argument('--pytest', action='store_true', help='Use pytest instead of unittest')
    
    args = parser.parse_args()
    
    if args.pytest:
        success = run_with_pytest()
    elif args.test == 'all':
        success = run_all_tests()
    else:
        success = run_specific_test(args.test)
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 