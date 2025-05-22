"""
Batch processing utilities for concurrent LLM operations.

This module provides helper functions for processing batches of LLM tasks concurrently,
with support for rate limiting and error handling.
"""

import asyncio
import time
import random
from typing import List, Dict, Any, Callable, Awaitable, TypeVar, Tuple, Optional, Union

T = TypeVar('T')  # Return type of the coroutine
K = TypeVar('K')  # Key type


class BatchProcessor:
    """
    Utility class for batch processing of concurrent LLM requests.
    
    Features:
    - Concurrency limiting to avoid rate limits
    - Automatic retries with exponential backoff
    - Progress reporting
    - Exception handling
    """
    
    def __init__(
        self,
        max_concurrency: int = 5,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        jitter: float = 0.1,
        report_progress: bool = False,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            max_concurrency: Maximum number of concurrent requests
            max_retries: Maximum number of retries for failed tasks
            retry_delay: Initial delay between retries in seconds
            retry_backoff: Factor to increase delay after each retry
            jitter: Random factor to add to delay to prevent thundering herd
            report_progress: Whether to print progress to stdout
            progress_callback: Optional callback function that receives 
                               (completed, total, elapsed_time) values
        """
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff
        self.jitter = jitter
        self.report_progress = report_progress
        self.progress_callback = progress_callback
        
        # Internal state
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.completed = 0
        self.failed = 0
        self.total = 0
        self.start_time = None
    
    async def process_batch(
        self,
        items: List[Tuple[K, Dict[str, Any]]],
        task_func: Callable[[Dict[str, Any]], Awaitable[T]]
    ) -> Dict[K, Union[T, Exception]]:
        """
        Process a batch of items concurrently.
        
        Args:
            items: List of (key, params) tuples
            task_func: Async function that takes params and returns a result
            
        Returns:
            Dictionary mapping keys to results or exceptions
        """
        self.start_time = time.time()
        self.total = len(items)
        self.completed = 0
        self.failed = 0
        
        if self.report_progress:
            print(f"Starting batch processing of {self.total} items...")
        
        # Create tasks for all items
        tasks = {
            key: self._process_item(key, params, task_func) 
            for key, params in items
        }
        
        # Wait for all tasks to complete
        results = {}
        for key, task in tasks.items():
            try:
                results[key] = await task
            except Exception as e:
                results[key] = e
                self.failed += 1
        
        elapsed = time.time() - self.start_time
        if self.report_progress:
            print(f"Completed batch processing in {elapsed:.2f}s ({self.completed} succeeded, {self.failed} failed)")
        
        return results
    
    async def _process_item(
        self,
        key: K,
        params: Dict[str, Any],
        task_func: Callable[[Dict[str, Any]], Awaitable[T]]
    ) -> T:
        """
        Process a single item with retries and concurrency limiting.
        
        Args:
            key: The key for this item
            params: Parameters for the task function
            task_func: Async function to execute
            
        Returns:
            The result of the task function
            
        Raises:
            Exception: If all retries fail
        """
        retries = 0
        delay = self.retry_delay
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                async with self.semaphore:
                    result = await task_func(params)
                
                # Success
                self.completed += 1
                elapsed = time.time() - self.start_time
                
                # Report progress if enabled
                if self.report_progress and self.completed % max(1, self.total // 10) == 0:
                    print(f"Progress: {self.completed}/{self.total} ({self.completed/self.total*100:.1f}%) in {elapsed:.2f}s")
                
                # Call progress callback if provided
                if self.progress_callback:
                    self.progress_callback(self.completed, self.total, elapsed)
                
                return result
                
            except Exception as e:
                last_exception = e
                retries += 1
                
                if retries > self.max_retries:
                    break
                
                # Add jitter to delay
                jitter_amount = self.jitter * delay * random.random()
                await asyncio.sleep(delay + jitter_amount)
                delay *= self.retry_backoff
        
        # All retries failed
        if self.report_progress:
            print(f"Failed to process item {key} after {retries} retries: {last_exception}")
        
        raise last_exception or Exception(f"Failed to process item {key} after {retries} retries")


async def process_items_concurrently(
    items: List[Tuple[K, Dict[str, Any]]],
    processor_func: Callable[[Dict[str, Any]], Awaitable[T]],
    max_concurrency: int = 5,
    max_retries: int = 3,
    report_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> Dict[K, Union[T, Exception]]:
    """
    Convenience function to process items concurrently.
    
    Args:
        items: List of (key, params) tuples
        processor_func: Async function to process each item
        max_concurrency: Maximum number of concurrent tasks
        max_retries: Maximum number of retries for failed tasks
        report_progress: Whether to print progress to stdout
        progress_callback: Optional callback function for tracking progress
        
    Returns:
        Dictionary mapping keys to results or exceptions
    """
    processor = BatchProcessor(
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        report_progress=report_progress,
        progress_callback=progress_callback
    )
    
    return await processor.process_batch(items, processor_func)


# Specific batch processing functions for common LLM tasks

async def batch_generate_labels(
    analyzer: 'ClusterAnalysis',
    items: Dict[str, Tuple[Any, Dict[str, Any]]],
    max_concurrency: int = 5,
    report_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> Dict[str, str]:
    """
    Generate labels for multiple items concurrently.
    
    Args:
        analyzer: ClusterAnalysis instance
        items: Dictionary mapping keys to (item, params) tuples
        max_concurrency: Maximum number of concurrent requests
        report_progress: Whether to print progress to stdout
        progress_callback: Optional callback function for tracking progress
        
    Returns:
        Dictionary mapping keys to labels
    """
    async def label_item(params):
        item, extra_params = params["item"], params["extra"]
        return await analyzer.label_cluster(item, **extra_params)
    
    # Convert items to format expected by processor
    processor_items = [
        (key, {"item": item, "extra": params})
        for key, (item, params) in items.items()
    ]
    
    results = await process_items_concurrently(
        processor_items,
        label_item,
        max_concurrency=max_concurrency,
        report_progress=report_progress,
        progress_callback=progress_callback
    )
    
    # Filter out exceptions
    labels = {}
    for key, result in results.items():
        if isinstance(result, Exception):
            print(f"Warning: Failed to generate label for {key}: {result}")
        else:
            labels[key] = result
    
    return labels


async def batch_generate_narratives(
    analyzer: 'ClusterAnalysis',
    paths: Dict[int, Dict[str, Any]],
    max_concurrency: int = 5,
    report_progress: bool = False,
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> Dict[int, str]:
    """
    Generate narratives for multiple paths concurrently.
    
    Args:
        analyzer: ClusterAnalysis instance
        paths: Dictionary mapping path IDs to parameter dictionaries for generate_path_narrative
        max_concurrency: Maximum number of concurrent requests
        report_progress: Whether to print progress to stdout
        progress_callback: Optional callback function for tracking progress
        
    Returns:
        Dictionary mapping path IDs to narratives
    """
    async def generate_narrative(params):
        return await analyzer.generate_path_narrative(**params)
    
    # Convert paths to format expected by processor
    processor_items = [
        (path_id, params) for path_id, params in paths.items()
    ]
    
    results = await process_items_concurrently(
        processor_items,
        generate_narrative,
        max_concurrency=max_concurrency,
        report_progress=report_progress,
        progress_callback=progress_callback
    )
    
    # Filter out exceptions
    narratives = {}
    for path_id, result in results.items():
        if isinstance(result, Exception):
            print(f"Warning: Failed to generate narrative for path {path_id}: {result}")
            narratives[path_id] = f"Error generating narrative: {str(result)}"
        else:
            narratives[path_id] = result
    
    return narratives