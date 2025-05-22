"""
Tests for the pipeline architecture.

This module contains tests for the flexible pipeline architecture, including
basic stage execution, streaming, and complex workflows.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from typing import Dict, List, Any

from concept_fragmentation.pipeline import (
    Pipeline, PipelineConfig, StreamingMode,
    PipelineStageBase, StreamingStage, FunctionStage,
    StreamingFunctionStage, ParallelStage, ConditionalStage,
    CachingStage, PipelineContext
)


class IdentityStage(PipelineStageBase):
    """Stage that returns its input unchanged."""
    
    def process(self, data: Any) -> Any:
        """Process the input data."""
        return data


class DoublingStage(PipelineStageBase):
    """Stage that doubles numeric inputs."""
    
    def process(self, data: Any) -> Any:
        """Process the input data."""
        if isinstance(data, (int, float, np.number)):
            return data * 2
        elif isinstance(data, list):
            return [self.process(item) for item in data]
        elif isinstance(data, dict):
            return {k: self.process(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data * 2
        elif isinstance(data, torch.Tensor):
            return data * 2
        else:
            return data


class StreamingDoublingStage(StreamingStage):
    """Stage that doubles numeric inputs in a streaming fashion."""
    
    def process_item(self, item: Any) -> Any:
        """Process a single item."""
        if isinstance(item, (int, float, np.number)):
            return item * 2
        elif isinstance(item, list):
            return [self.process_item(i) for i in item]
        elif isinstance(item, dict):
            return {k: self.process_item(v) for k, v in item.items()}
        elif isinstance(item, np.ndarray):
            return item * 2
        elif isinstance(item, torch.Tensor):
            return item * 2
        else:
            return item


class LoggingStage(PipelineStageBase):
    """Stage that logs its input and passes it through."""
    
    def __init__(self, name: str = "LoggingStage"):
        """Initialize the logging stage."""
        super().__init__(name=name)
        self.calls = []
    
    def process(self, data: Any) -> Any:
        """Process the input data."""
        self.calls.append(data)
        return data


class TestPipeline(unittest.TestCase):
    """Test cases for the Pipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)
    
    def test_basic_pipeline(self):
        """Test basic pipeline execution."""
        pipeline = Pipeline()
        
        # Add a simple stage
        pipeline.add_stage(DoublingStage())
        
        # Execute the pipeline
        result = pipeline.execute(5)
        
        # Check the result
        self.assertEqual(result, 10)
    
    def test_multiple_stages(self):
        """Test pipeline with multiple stages."""
        pipeline = Pipeline()
        
        # Add multiple doubling stages
        pipeline.add_stage(DoublingStage())
        pipeline.add_stage(DoublingStage())
        pipeline.add_stage(DoublingStage())
        
        # Execute the pipeline
        result = pipeline.execute(2)
        
        # Check the result (2 * 2 * 2 * 2 = 16)
        self.assertEqual(result, 16)
    
    def test_pipeline_context(self):
        """Test pipeline execution with context."""
        pipeline = Pipeline(PipelineConfig(use_context=True))
        
        # Add stages
        pipeline.add_stage(DoublingStage(), name="Double1")
        pipeline.add_stage(DoublingStage(), name="Double2")
        
        # Create context
        context = PipelineContext()
        
        # Execute the pipeline
        result = pipeline.execute(5, context=context)
        
        # Check the result
        self.assertEqual(result, 20)
        
        # Check context
        self.assertIn("Double1", context.stage_timings)
        self.assertIn("Double2", context.stage_timings)
        self.assertTrue(context.is_successful())
    
    def test_streaming_pipeline(self):
        """Test pipeline in streaming mode."""
        pipeline = Pipeline(PipelineConfig(streaming_mode=StreamingMode.FORCE))
        
        # Add streaming stages
        pipeline.add_stage(StreamingDoublingStage())
        pipeline.add_stage(StreamingDoublingStage())
        
        # Input as a list
        input_data = [1, 2, 3, 4, 5]
        
        # Execute the pipeline
        result = pipeline.execute(input_data)
        
        # Check the result (each item doubled twice)
        self.assertEqual(result, [4, 8, 12, 16, 20])
    
    def test_function_stage(self):
        """Test function-based pipeline stages."""
        def double(x):
            return x * 2
        
        pipeline = Pipeline()
        
        # Add function stage
        pipeline.add_stage(FunctionStage(double))
        
        # Execute the pipeline
        result = pipeline.execute(7)
        
        # Check the result
        self.assertEqual(result, 14)
    
    def test_streaming_function_stage(self):
        """Test streaming function-based pipeline stages."""
        def double(x):
            return x * 2
        
        pipeline = Pipeline(PipelineConfig(streaming_mode=StreamingMode.FORCE))
        
        # Add streaming function stage
        pipeline.add_stage(StreamingFunctionStage(double))
        
        # Input as a list
        input_data = [1, 2, 3, 4, 5]
        
        # Execute the pipeline
        result = pipeline.execute(input_data)
        
        # Check the result
        self.assertEqual(result, [2, 4, 6, 8, 10])
    
    def test_parallel_stage(self):
        """Test parallel execution of stages."""
        pipeline = Pipeline()
        
        # Create parallel stage with two doubling stages
        parallel_stage = ParallelStage([
            DoublingStage(),
            FunctionStage(lambda x: x + 10)
        ])
        
        pipeline.add_stage(parallel_stage)
        
        # Execute the pipeline
        result = pipeline.execute(5)
        
        # Check the result (should be a list with [10, 15])
        self.assertEqual(result, [10, 15])
    
    def test_conditional_stage(self):
        """Test conditional execution of stages."""
        pipeline = Pipeline()
        
        # Create conditional stage
        conditional_stage = ConditionalStage(
            predicates=[
                lambda x: x > 10,
                lambda x: x <= 10
            ],
            stages=[
                FunctionStage(lambda x: x * 2),
                FunctionStage(lambda x: x + 5)
            ]
        )
        
        pipeline.add_stage(conditional_stage)
        
        # Execute the pipeline with different inputs
        result1 = pipeline.execute(15)
        result2 = pipeline.execute(5)
        
        # Check the results
        self.assertEqual(result1, 30)  # 15 > 10, so 15 * 2 = 30
        self.assertEqual(result2, 10)  # 5 <= 10, so 5 + 5 = 10
    
    def test_caching_stage(self):
        """Test caching of stage results."""
        # Create a logging stage to track calls
        logging_stage = LoggingStage()
        
        # Create a caching stage around it
        caching_stage = CachingStage(
            stage=logging_stage,
            cache_key_fn=lambda x: str(x),
            in_memory=True
        )
        
        pipeline = Pipeline()
        pipeline.add_stage(caching_stage)
        
        # Execute the pipeline multiple times with the same input
        pipeline.execute(42)
        pipeline.execute(42)
        pipeline.execute(42)
        
        # Different input
        pipeline.execute(24)
        
        # Same input again
        pipeline.execute(42)
        
        # Check the logging stage was only called twice (once for each unique input)
        self.assertEqual(len(logging_stage.calls), 2)
        self.assertEqual(logging_stage.calls[0], 42)
        self.assertEqual(logging_stage.calls[1], 24)
    
    def test_pipeline_validation(self):
        """Test pipeline validation."""
        pipeline = Pipeline(PipelineConfig(streaming_mode=StreamingMode.FORCE))
        
        # Add a non-streaming stage (should cause validation error)
        pipeline.add_stage(DoublingStage())
        
        # Validate the pipeline
        errors = pipeline.validate()
        
        # Check that we got an error
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("does not support streaming" in error for error in errors))
    
    def test_dot_representation(self):
        """Test generation of DOT representation."""
        pipeline = Pipeline()
        
        # Add stages
        pipeline.add_stage(DoublingStage(), name="Double1")
        pipeline.add_stage(DoublingStage(), name="Double2")
        pipeline.add_stage(StreamingDoublingStage(), name="StreamingDouble")
        
        # Get DOT representation
        dot_repr = pipeline.get_dot_representation()
        
        # Check that it includes all the stages
        self.assertIn("Double1", dot_repr)
        self.assertIn("Double2", dot_repr)
        self.assertIn("StreamingDouble", dot_repr)
        
        # Check that it includes the connections
        self.assertIn("Double1 -> Double2", dot_repr)
        self.assertIn("Double2 -> StreamingDouble", dot_repr)


if __name__ == '__main__':
    unittest.main()