"""
Tests for the rate limiting and retry logic.
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock, call
import time
from typing import Dict, Any, Optional

from ..rate_limit import RateLimiter, RetryPolicy


class TestRetryPolicy(unittest.TestCase):
    """Test the RetryPolicy class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        policy = RetryPolicy()
        
        self.assertEqual(policy.max_retries, 3)
        self.assertEqual(policy.base_delay, 1.0)
        self.assertEqual(policy.max_delay, 60.0)
        self.assertEqual(policy.jitter_factor, 0.1)
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        policy = RetryPolicy(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            jitter_factor=0.2
        )
        
        self.assertEqual(policy.max_retries, 5)
        self.assertEqual(policy.base_delay, 2.0)
        self.assertEqual(policy.max_delay, 120.0)
        self.assertEqual(policy.jitter_factor, 0.2)
    
    def test_calculate_delay(self):
        """Test calculating delay with exponential backoff."""
        policy = RetryPolicy(
            base_delay=1.0,
            max_delay=60.0,
            jitter_factor=0.0  # Disable jitter for deterministic testing
        )
        
        # Test exponential backoff
        self.assertEqual(policy.calculate_delay(1), 1.0)   # 1.0 * 2^(1-1)
        self.assertEqual(policy.calculate_delay(2), 2.0)   # 1.0 * 2^(2-1)
        self.assertEqual(policy.calculate_delay(3), 4.0)   # 1.0 * 2^(3-1)
        self.assertEqual(policy.calculate_delay(4), 8.0)   # 1.0 * 2^(4-1)
        
        # Test max_delay cap
        policy = RetryPolicy(
            base_delay=10.0,
            max_delay=30.0,
            jitter_factor=0.0
        )
        
        self.assertEqual(policy.calculate_delay(1), 10.0)  # 10.0 * 2^(1-1) = 10.0
        self.assertEqual(policy.calculate_delay(2), 20.0)  # 10.0 * 2^(2-1) = 20.0
        self.assertEqual(policy.calculate_delay(3), 30.0)  # Would be 40.0, but capped at 30.0
    
    @patch('random.uniform')
    def test_calculate_delay_with_jitter(self, mock_uniform):
        """Test jitter factor in delay calculation."""
        mock_uniform.return_value = 0.05  # Fixed random value for testing
        
        policy = RetryPolicy(
            base_delay=1.0,
            max_delay=60.0,
            jitter_factor=0.1
        )
        
        # Expected: 1.0 * (1 + 0.05 * 0.1) = 1.005
        delay = policy.calculate_delay(1)
        self.assertAlmostEqual(delay, 1.005)
        
        # Expected: 2.0 * (1 + 0.05 * 0.1) = 2.01
        delay = policy.calculate_delay(2)
        self.assertAlmostEqual(delay, 2.01)


class TestRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Test the RateLimiter class."""
    
    def setUp(self):
        """Set up each test."""
        # Clear class level state
        RateLimiter._rate_limits = {}
    
    async def test_init_with_defaults(self):
        """Test initialization with default values."""
        limiter = RateLimiter("test-provider")
        
        self.assertEqual(limiter.provider, "test-provider")
        self.assertIsNone(limiter.requests_per_minute)
        self.assertIsInstance(limiter.retry_policy, RetryPolicy)
    
    async def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        retry_policy = RetryPolicy(max_retries=5)
        limiter = RateLimiter(
            provider="test-provider",
            requests_per_minute=60,
            retry_policy=retry_policy
        )
        
        self.assertEqual(limiter.provider, "test-provider")
        self.assertEqual(limiter.requests_per_minute, 60)
        self.assertIs(limiter.retry_policy, retry_policy)
    
    @patch('asyncio.sleep')
    async def test_execute_with_retry_success(self, mock_sleep):
        """Test successful execution with no retries needed."""
        limiter = RateLimiter("test-provider")
        
        # Create a mock function that succeeds
        mock_func = MagicMock()
        mock_func.return_value = "success"
        
        # Execute the function
        result = await limiter.execute_with_retry(mock_func, "arg1", kwarg1="value1")
        
        # Check results
        self.assertEqual(result, "success")
        mock_func.assert_called_once_with("arg1", kwarg1="value1")
        mock_sleep.assert_not_called()  # No sleep should be called on success
    
    @patch('asyncio.sleep')
    async def test_execute_with_retry_failure_then_success(self, mock_sleep):
        """Test retry logic when a function fails then succeeds."""
        limiter = RateLimiter("test-provider")
        
        # Create a mock function that fails once then succeeds
        mock_func = MagicMock()
        mock_func.side_effect = [
            Exception("Rate limit exceeded"),  # First call fails
            "success"                          # Second call succeeds
        ]
        
        # Execute the function
        result = await limiter.execute_with_retry(mock_func, "arg1", kwarg1="value1")
        
        # Check results
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)
        mock_sleep.assert_called_once()  # Sleep should be called once after the failure
    
    @patch('asyncio.sleep')
    async def test_execute_with_retry_all_failures(self, mock_sleep):
        """Test retry logic when all attempts fail."""
        limiter = RateLimiter("test-provider", retry_policy=RetryPolicy(max_retries=2))
        
        # Create a mock function that always fails
        mock_func = MagicMock()
        error = Exception("Rate limit exceeded")
        mock_func.side_effect = error
        
        # Execute the function and expect it to raise the exception
        with self.assertRaises(Exception) as context:
            await limiter.execute_with_retry(mock_func, "arg1", kwarg1="value1")
        
        # Check that the original exception was raised
        self.assertIs(context.exception, error)
        
        # Check call counts
        self.assertEqual(mock_func.call_count, 3)  # Initial + 2 retries
        self.assertEqual(mock_sleep.call_count, 2)  # Sleep called after each failure
    
    @patch('time.time')
    @patch('asyncio.sleep')
    async def test_rate_limiting(self, mock_sleep, mock_time):
        """Test rate limiting when requests_per_minute is specified."""
        # Fix the time to control the test
        mock_time.return_value = 1000.0
        
        # Create a rate limiter with 60 requests per minute (1 per second)
        limiter = RateLimiter("test-provider", requests_per_minute=60)
        
        # Mock function that succeeds
        mock_func = MagicMock()
        mock_func.return_value = "success"
        
        # First call should proceed immediately
        result1 = await limiter.execute_with_retry(mock_func)
        self.assertEqual(result1, "success")
        mock_sleep.assert_not_called()
        
        # Advance time by 0.5 seconds (less than the 1-second rate limit)
        mock_time.return_value = 1000.5
        
        # Second call should wait for the remaining time to meet the rate limit
        result2 = await limiter.execute_with_retry(mock_func)
        self.assertEqual(result2, "success")
        mock_sleep.assert_called_once_with(0.5)  # Should sleep for 0.5 seconds
    
    @patch('asyncio.sleep')
    async def test_provider_specific_rate_limits(self, mock_sleep):
        """Test provider-specific rate limit detection and handling."""
        limiter = RateLimiter("test-provider")
        
        # Create a mock function that simulates different provider errors
        openai_error = Exception("Rate limit exceeded")
        openai_error.headers = {'x-ratelimit-remaining': '0', 'x-ratelimit-reset-at': '2023-01-01T00:01:00Z'}
        
        claude_error = Exception("Too many requests")
        # Simulate Anthropic error format
        
        mock_func = MagicMock()
        mock_func.side_effect = [
            openai_error,  # First call fails with OpenAI-like error
            "success"      # Second call succeeds
        ]
        
        # Execute the function
        result = await limiter.execute_with_retry(mock_func)
        
        # Check results
        self.assertEqual(result, "success")
        self.assertEqual(mock_func.call_count, 2)
        mock_sleep.assert_called_once()  # Sleep should be called once after the failure


if __name__ == '__main__':
    unittest.main()