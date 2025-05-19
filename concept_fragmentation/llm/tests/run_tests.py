"""
Test runner for LLM client implementation tests.
"""

import unittest
import os
import sys

# Add the parent directory to the path so we can import the LLM module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import all test modules
from test_base_client import TestBaseLLMClient
from test_factory import TestLLMClientFactory
from test_responses import TestLLMResponse, TestResponseParser
from test_rate_limit import TestRetryPolicy, TestRateLimiter
from test_config import TestConfig
from test_provider_clients import TestGrokClient, TestClaudeClient, TestOpenAIClient


def run_tests():
    """Run all LLM client tests."""
    # Create a test suite containing all the tests
    test_suite = unittest.TestSuite()
    
    # Add test cases from each test module
    test_suite.addTest(unittest.makeSuite(TestBaseLLMClient))
    test_suite.addTest(unittest.makeSuite(TestLLMClientFactory))
    test_suite.addTest(unittest.makeSuite(TestLLMResponse))
    test_suite.addTest(unittest.makeSuite(TestResponseParser))
    test_suite.addTest(unittest.makeSuite(TestRetryPolicy))
    test_suite.addTest(unittest.makeSuite(TestRateLimiter))
    test_suite.addTest(unittest.makeSuite(TestConfig))
    test_suite.addTest(unittest.makeSuite(TestGrokClient))
    test_suite.addTest(unittest.makeSuite(TestClaudeClient))
    test_suite.addTest(unittest.makeSuite(TestOpenAIClient))
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return success/failure for CI integration
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())