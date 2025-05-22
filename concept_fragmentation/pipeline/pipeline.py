"""
Flexible data processing pipeline architecture.

This module provides a pipeline system for processing neural network activations
and other data through a sequence of operations. It focuses on memory efficiency,
streaming capabilities, and flexible component composition.

Key components:
- PipelineStage: Protocol for stages in a processing pipeline
- Pipeline: Orchestrates execution of pipeline stages
- StreamingPipeline: Memory-efficient streaming execution of pipeline stages
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Protocol, TypeVar, Generator, Callable, Set
import logging
import time
import tempfile
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
U = TypeVar('U')

class PipelineStage(Protocol[T, U]):
    """Protocol defining a stage in the data processing pipeline."""
    
    def process(self, data: T) -> U:
        """Process input data and return output."""
        ...
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        ...
    
    def supports_type(self, input_type: type) -> bool:
        """Whether this stage supports the given input type."""
        ...
    
    def get_output_type(self) -> type:
        """Get the output type of this stage."""
        ...


class PipelineStageBase(ABC, PipelineStage[T, U]):
    """Base implementation of PipelineStage protocol."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the pipeline stage.
        
        Args:
            name: Optional name for this stage
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def process(self, data: T) -> U:
        """Process input data and return output."""
        pass
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        return False
    
    def supports_type(self, input_type: type) -> bool:
        """
        Check if this stage supports the given input type.
        
        Default implementation checks against the type annotation
        of the 'process' method's parameter.
        """
        import inspect
        sig = inspect.signature(self.process)
        param = list(sig.parameters.values())[0]  # Get the first parameter
        
        if param.annotation is inspect.Parameter.empty:
            # No type annotation, assume it supports all types
            return True
        
        # Check if input_type is a subclass of the annotated type
        try:
            return issubclass(input_type, param.annotation)
        except TypeError:
            # Handle complex types (Union, etc.)
            return True
    
    def get_output_type(self) -> type:
        """
        Get the output type of this stage.
        
        Default implementation uses the return type annotation
        of the 'process' method.
        """
        import inspect
        sig = inspect.signature(self.process)
        return sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else Any


class StreamingStage(PipelineStageBase[Generator[T, None, None], Generator[U, None, None]]):
    """Base class for stages that support streaming processing."""
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        return True
    
    @abstractmethod
    def process_item(self, item: T) -> U:
        """Process a single item from the stream."""
        pass
    
    def process(self, data_stream: Generator[T, None, None]) -> Generator[U, None, None]:
        """Process a stream of data and yield results."""
        for item in data_stream:
            yield self.process_item(item)


class FunctionStage(PipelineStageBase[T, U]):
    """Pipeline stage that wraps a function."""
    
    def __init__(
        self,
        func: Callable[[T], U],
        name: Optional[str] = None,
        can_stream_value: bool = False,
        input_type: Optional[type] = None,
        output_type: Optional[type] = None
    ):
        """
        Initialize a function-based pipeline stage.
        
        Args:
            func: Function to execute for this stage
            name: Optional name for this stage
            can_stream_value: Whether this function can process streaming data
            input_type: Input type this stage accepts
            output_type: Output type this stage produces
        """
        super().__init__(name=name or func.__name__)
        self.func = func
        self._can_stream = can_stream_value
        self._input_type = input_type
        self._output_type = output_type
    
    def process(self, data: T) -> U:
        """Execute the function on the input data."""
        return self.func(data)
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        return self._can_stream
    
    def supports_type(self, input_type: type) -> bool:
        """Whether this stage supports the given input type."""
        if self._input_type is None:
            return super().supports_type(input_type)
        
        try:
            return issubclass(input_type, self._input_type)
        except TypeError:
            return False
    
    def get_output_type(self) -> type:
        """Get the output type of this stage."""
        if self._output_type is None:
            return super().get_output_type()
        
        return self._output_type


class StreamingFunctionStage(StreamingStage[T, U]):
    """Pipeline stage that wraps a function for streaming processing."""
    
    def __init__(
        self,
        func: Callable[[T], U],
        name: Optional[str] = None,
        input_type: Optional[type] = None,
        output_type: Optional[type] = None
    ):
        """
        Initialize a streaming function-based pipeline stage.
        
        Args:
            func: Function to execute for each item
            name: Optional name for this stage
            input_type: Input type this stage accepts
            output_type: Output type this stage produces
        """
        super().__init__(name=name or func.__name__)
        self.func = func
        self._input_type = input_type
        self._output_type = output_type
    
    def process_item(self, item: T) -> U:
        """Process a single item using the function."""
        return self.func(item)
    
    def supports_type(self, input_type: type) -> bool:
        """Whether this stage supports the given input type."""
        if self._input_type is None:
            return super().supports_type(input_type)
        
        try:
            return issubclass(input_type, self._input_type)
        except TypeError:
            return False
    
    def get_output_type(self) -> type:
        """Get the output type of this stage."""
        if self._output_type is None:
            return super().get_output_type()
        
        return self._output_type


class ParallelStage(PipelineStageBase[T, List[Any]]):
    """Pipeline stage that executes multiple stages in parallel on the same input."""
    
    def __init__(
        self,
        stages: List[PipelineStage],
        name: Optional[str] = None,
        max_workers: Optional[int] = None
    ):
        """
        Initialize a parallel pipeline stage.
        
        Args:
            stages: List of stages to execute in parallel
            name: Optional name for this stage
            max_workers: Maximum number of worker threads
        """
        super().__init__(name=name or "ParallelStage")
        self.stages = stages
        self.max_workers = max_workers
    
    def process(self, data: T) -> List[Any]:
        """Execute all stages in parallel on the input data."""
        if not self.stages:
            return []
        
        if len(self.stages) == 1:
            return [self.stages[0].process(data)]
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(stage.process, data): i for i, stage in enumerate(self.stages)}
            
            for future in futures:
                try:
                    result = future.result()
                    results.append((futures[future], result))
                except Exception as e:
                    logger.error(f"Error in parallel stage {self.stages[futures[future]].name}: {e}")
                    results.append((futures[future], None))
        
        # Sort results by original stage order
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        # Can only stream if all stages can stream
        return all(stage.can_stream() for stage in self.stages)


class ConditionalStage(PipelineStageBase[T, Any]):
    """Pipeline stage that conditionally executes one of multiple stages based on a predicate."""
    
    def __init__(
        self,
        predicates: List[Callable[[T], bool]],
        stages: List[PipelineStage],
        default_stage: Optional[PipelineStage] = None,
        name: Optional[str] = None
    ):
        """
        Initialize a conditional pipeline stage.
        
        Args:
            predicates: List of functions that return True if the corresponding stage should be executed
            stages: List of stages to execute conditionally
            default_stage: Stage to execute if no predicate returns True
            name: Optional name for this stage
        """
        super().__init__(name=name or "ConditionalStage")
        
        if len(predicates) != len(stages):
            raise ValueError("Number of predicates must match number of stages")
        
        self.predicates = predicates
        self.stages = stages
        self.default_stage = default_stage
    
    def process(self, data: T) -> Any:
        """Execute the appropriate stage based on the predicate results."""
        for predicate, stage in zip(self.predicates, self.stages):
            if predicate(data):
                return stage.process(data)
        
        if self.default_stage:
            return self.default_stage.process(data)
        
        # If no stage was executed, return the input data
        return data
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        # Can stream if all potential stages can stream
        stages_to_check = self.stages
        if self.default_stage:
            stages_to_check = stages_to_check + [self.default_stage]
        
        return all(stage.can_stream() for stage in stages_to_check)


class CachingStage(PipelineStageBase[T, U]):
    """Pipeline stage that caches the results of another stage."""
    
    def __init__(
        self,
        stage: PipelineStage[T, U],
        cache_key_fn: Callable[[T], str],
        cache_dir: Optional[str] = None,
        in_memory: bool = True,
        name: Optional[str] = None
    ):
        """
        Initialize a caching pipeline stage.
        
        Args:
            stage: The stage whose results should be cached
            cache_key_fn: Function to generate a cache key from input data
            cache_dir: Directory to store cache files (if not in-memory)
            in_memory: Whether to cache in memory
            name: Optional name for this stage
        """
        super().__init__(name=name or f"Cached({stage.name})")
        self.stage = stage
        self.cache_key_fn = cache_key_fn
        self.in_memory = in_memory
        
        # Set up memory cache
        self.memory_cache = {} if in_memory else None
        
        # Set up disk cache if requested
        self.cache_dir = cache_dir
        if cache_dir and not in_memory:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> str:
        """Get the path to the cache file for a key."""
        import hashlib
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.pkl")
    
    def process(self, data: T) -> U:
        """Process input data, using cached result if available."""
        cache_key = self.cache_key_fn(data)
        
        # Try memory cache first
        if self.in_memory and cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Try disk cache if available
        if not self.in_memory and self.cache_dir:
            cache_path = self._get_cache_path(cache_key)
            if os.path.exists(cache_path):
                import pickle
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                return result
        
        # Cache miss, compute result
        result = self.stage.process(data)
        
        # Store in memory cache
        if self.in_memory:
            self.memory_cache[cache_key] = result
        
        # Store in disk cache
        if not self.in_memory and self.cache_dir:
            cache_path = self._get_cache_path(cache_key)
            import pickle
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        
        return result
    
    def can_stream(self) -> bool:
        """Whether this stage can process data in a streaming fashion."""
        # Caching stages cannot stream, as they need to cache the entire result
        return False


class PipelineContext:
    """Context object for pipeline execution."""
    
    def __init__(self):
        """Initialize the pipeline context."""
        self.data = {}
        self.start_time = time.time()
        self.stage_timings = {}
        self.errors = []
        self.memory_usage = {}
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the context."""
        return self.data.get(key, default)
    
    def add_timing(self, stage_name: str, elapsed: float) -> None:
        """Add timing information for a stage."""
        self.stage_timings[stage_name] = elapsed
    
    def add_error(self, stage_name: str, error: Exception) -> None:
        """Add an error that occurred during pipeline execution."""
        self.errors.append((stage_name, error))
    
    def record_memory_usage(self, stage_name: str) -> None:
        """Record memory usage after a stage completes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            self.memory_usage[stage_name] = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            # psutil not available
            pass
    
    def get_total_time(self) -> float:
        """Get the total time elapsed since the context was created."""
        return time.time() - self.start_time
    
    def is_successful(self) -> bool:
        """Whether the pipeline execution was successful."""
        return len(self.errors) == 0


class StreamingMode(Enum):
    """Enumeration of streaming modes for the pipeline."""
    AUTO = auto()
    FORCE = auto()
    DISABLE = auto()


@dataclass
class PipelineConfig:
    """Configuration for pipeline execution."""
    streaming_mode: StreamingMode = StreamingMode.AUTO
    memory_limit: Optional[int] = None  # In MB
    batch_size: Optional[int] = None
    use_context: bool = True
    use_disk_offload: bool = False
    temp_dir: Optional[str] = None
    max_workers: Optional[int] = None
    log_progress: bool = False
    progress_interval: float = 5.0  # In seconds


class Pipeline:
    """
    Executes pipeline stages with optimization for memory usage and performance.
    
    This class orchestrates the execution of pipeline stages, managing data flow
    between stages and optimizing for memory efficiency when needed.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration for pipeline execution
        """
        self.config = config or PipelineConfig()
        self.stages = []
        self._stage_names = set()
    
    def add_stage(self, stage: PipelineStage, name: Optional[str] = None) -> 'Pipeline':
        """
        Add a stage to the pipeline.
        
        Args:
            stage: Stage to add
            name: Optional name for the stage (defaults to stage's name)
            
        Returns:
            Self for method chaining
        """
        if name:
            stage_name = name
        elif hasattr(stage, 'name'):
            stage_name = stage.name
        else:
            stage_name = f"Stage_{len(self.stages)}"
        
        # Ensure unique names
        if stage_name in self._stage_names:
            base_name = stage_name
            counter = 1
            while stage_name in self._stage_names:
                stage_name = f"{base_name}_{counter}"
                counter += 1
        
        self._stage_names.add(stage_name)
        self.stages.append((stage, stage_name))
        
        return self
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove a stage from the pipeline by name.
        
        Args:
            stage_name: Name of the stage to remove
            
        Returns:
            True if the stage was removed, False if not found
        """
        for i, (_, name) in enumerate(self.stages):
            if name == stage_name:
                self.stages.pop(i)
                self._stage_names.remove(stage_name)
                return True
        
        return False
    
    def execute(
        self,
        input_data: Any,
        context: Optional[PipelineContext] = None
    ) -> Any:
        """
        Execute the pipeline on the input data.
        
        Args:
            input_data: Input data for the pipeline
            context: Optional context for the pipeline execution
            
        Returns:
            Result of the pipeline execution
        """
        if not self.stages:
            return input_data
        
        # Create context if needed
        if context is None and self.config.use_context:
            context = PipelineContext()
        
        # Determine if we should use streaming mode
        use_streaming = self._should_use_streaming()
        
        if use_streaming:
            return self._execute_streaming(input_data, context)
        else:
            return self._execute_normal(input_data, context)
    
    def _should_use_streaming(self) -> bool:
        """Determine if streaming mode should be used."""
        if self.config.streaming_mode == StreamingMode.FORCE:
            # Check if all stages support streaming
            for stage, _ in self.stages:
                if not stage.can_stream():
                    logger.warning(f"Forcing streaming mode but stage {getattr(stage, 'name', 'unnamed')} does not support streaming")
            return True
        
        if self.config.streaming_mode == StreamingMode.DISABLE:
            return False
        
        # Auto mode:
        # 1. Check if all stages support streaming
        all_can_stream = all(stage.can_stream() for stage, _ in self.stages)
        
        # 2. Check if memory limit is set and we have psutil
        memory_limit_set = self.config.memory_limit is not None
        
        # 3. Check if input/output type is compatible with streaming
        try:
            import inspect
            if len(self.stages) > 0:
                first_stage, _ = self.stages[0]
                first_param = list(inspect.signature(first_stage.process).parameters.values())[0]
                if first_param.annotation != inspect.Parameter.empty:
                    is_generator_input = hasattr(first_param.annotation, "__origin__") and first_param.annotation.__origin__ is Generator
                    if is_generator_input:
                        return True
        except Exception:
            # If we can't determine the type, assume it's not streaming
            pass
        
        # Use streaming if all stages support it and either memory limit is set or disk offload is enabled
        return all_can_stream and (memory_limit_set or self.config.use_disk_offload)
    
    def _execute_normal(
        self,
        input_data: Any,
        context: Optional[PipelineContext]
    ) -> Any:
        """Execute the pipeline in normal (non-streaming) mode."""
        result = input_data
        
        for i, (stage, name) in enumerate(self.stages):
            # Record progress
            if self.config.log_progress:
                logger.info(f"Executing stage {i+1}/{len(self.stages)}: {name}")
            
            # Measure execution time
            start_time = time.time()
            
            try:
                # Process the data
                result = stage.process(result)
                
                # Record timing
                elapsed = time.time() - start_time
                if context:
                    context.add_timing(name, elapsed)
                    context.record_memory_usage(name)
                
                if self.config.log_progress:
                    logger.info(f"Completed stage {name} in {elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"Error in stage {name}: {e}", exc_info=True)
                if context:
                    context.add_error(name, e)
                raise
        
        return result
    
    def _execute_streaming(
        self,
        input_data: Any,
        context: Optional[PipelineContext]
    ) -> Any:
        """Execute the pipeline in streaming mode."""
        # Check if input is already a generator
        if not hasattr(input_data, '__iter__') or isinstance(input_data, (dict, str, bytes)):
            # Convert to a generator yielding a single item
            def single_item_generator():
                yield input_data
            input_stream = single_item_generator()
        else:
            input_stream = input_data
        
        # Chain generators through each stage
        current_stream = input_stream
        
        for i, (stage, name) in enumerate(self.stages):
            # Record progress
            if self.config.log_progress:
                logger.info(f"Setting up streaming stage {i+1}/{len(self.stages)}: {name}")
            
            if not stage.can_stream():
                logger.warning(f"Stage {name} does not support streaming, accumulating data...")
                
                # Accumulate all data from the stream
                accumulated_data = []
                for item in current_stream:
                    accumulated_data.append(item)
                
                # Process accumulated data
                start_time = time.time()
                try:
                    result = stage.process(accumulated_data)
                    
                    # Record timing
                    elapsed = time.time() - start_time
                    if context:
                        context.add_timing(name, elapsed)
                        context.record_memory_usage(name)
                    
                    if self.config.log_progress:
                        logger.info(f"Completed accumulated stage {name} in {elapsed:.2f}s")
                
                except Exception as e:
                    logger.error(f"Error in accumulated stage {name}: {e}", exc_info=True)
                    if context:
                        context.add_error(name, e)
                    raise
                
                # Convert result back to a stream
                def result_generator():
                    yield result
                current_stream = result_generator()
            else:
                # Process the stream
                current_stream = self._create_streaming_processor(stage, name, current_stream, context)
        
        # If we're processing a stream and returning a generator, we need to accumulate the results
        # This is because the execution hasn't actually happened yet, just the pipeline setup
        if hasattr(current_stream, '__next__'):
            # Accumulate all results from the final stream
            result = []
            item_count = 0
            start_time = time.time()
            last_update = start_time
            
            try:
                for item in current_stream:
                    result.append(item)
                    item_count += 1
                    
                    # Log progress periodically
                    current_time = time.time()
                    if self.config.log_progress and current_time - last_update > self.config.progress_interval:
                        elapsed = current_time - start_time
                        rate = item_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Processed {item_count} items in {elapsed:.2f}s ({rate:.2f} items/s)")
                        last_update = current_time
            except Exception as e:
                logger.error(f"Error in streaming pipeline execution: {e}", exc_info=True)
                if context:
                    context.add_error("streaming_execution", e)
                raise
            
            # If we only have one result, return it directly
            if len(result) == 1:
                return result[0]
            
            # Otherwise, return the list of results
            return result
        
        # If we somehow got a non-generator (shouldn't happen in streaming mode), return it directly
        return current_stream
    
    def _create_streaming_processor(
        self,
        stage: PipelineStage,
        name: str,
        input_stream: Generator,
        context: Optional[PipelineContext]
    ) -> Generator:
        """Create a generator that processes items from input_stream through the stage."""
        # Record timing for each batch
        stage_start_time = time.time()
        item_count = 0
        last_update = stage_start_time
        
        for item in input_stream:
            # Process the item
            start_time = time.time()
            try:
                result = stage.process(item)
                
                # Update timing information
                elapsed = time.time() - start_time
                if context:
                    # Append timing to stage timing (will be averaged later)
                    if name in context.stage_timings:
                        # Average with existing timing
                        prev_timing = context.stage_timings[name]
                        context.stage_timings[name] = (prev_timing * item_count + elapsed) / (item_count + 1)
                    else:
                        context.stage_timings[name] = elapsed
                
                item_count += 1
                
                # Log progress periodically
                current_time = time.time()
                if self.config.log_progress and current_time - last_update > self.config.progress_interval:
                    stage_elapsed = current_time - stage_start_time
                    rate = item_count / stage_elapsed if stage_elapsed > 0 else 0
                    logger.info(f"Stage {name}: Processed {item_count} items in {stage_elapsed:.2f}s ({rate:.2f} items/s)")
                    last_update = current_time
                
                yield result
                
            except Exception as e:
                logger.error(f"Error in streaming stage {name}: {e}", exc_info=True)
                if context:
                    context.add_error(name, e)
                raise
    
    def validate(self) -> List[str]:
        """
        Validate the pipeline configuration.
        
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Check if all stages support streaming when in FORCE mode
        if self.config.streaming_mode == StreamingMode.FORCE:
            for stage, name in self.stages:
                if not stage.can_stream():
                    errors.append(f"Stage {name} does not support streaming, but streaming mode is FORCE")
        
        # Check type compatibility between stages
        if len(self.stages) > 1:
            for i in range(len(self.stages) - 1):
                current_stage, current_name = self.stages[i]
                next_stage, next_name = self.stages[i + 1]
                
                try:
                    current_output_type = current_stage.get_output_type()
                    if not next_stage.supports_type(current_output_type):
                        errors.append(f"Type mismatch between stages {current_name} and {next_name}")
                except Exception as e:
                    logger.warning(f"Could not validate type compatibility: {e}")
        
        return errors
    
    def get_dot_representation(self) -> str:
        """
        Get a DOT language representation of the pipeline for visualization.
        
        Returns:
            DOT language string representing the pipeline
        """
        dot_lines = ["digraph Pipeline {", "  rankdir=LR;"]
        
        for i, (stage, name) in enumerate(self.stages):
            # Node for this stage
            shape = "box" if stage.can_stream() else "ellipse"
            dot_lines.append(f'  stage_{i} [label="{name}", shape={shape}];')
            
            # Edge from previous stage
            if i > 0:
                dot_lines.append(f"  stage_{i-1} -> stage_{i};")
        
        dot_lines.append("}")
        
        return "\n".join(dot_lines)
    
    def plot(self, filename: Optional[str] = None) -> Any:
        """
        Plot the pipeline using graphviz.
        
        Args:
            filename: Optional filename to save the plot to
            
        Returns:
            Graphviz graph object
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError("Graphviz Python package not installed. Install with 'pip install graphviz'.")
        
        dot_repr = self.get_dot_representation()
        graph = graphviz.Source(dot_repr)
        
        if filename:
            graph.render(filename, view=False, cleanup=True)
        
        return graph


def create_function_pipeline(
    funcs: List[Callable],
    config: Optional[PipelineConfig] = None,
    streaming: bool = False,
    stage_names: Optional[List[str]] = None
) -> Pipeline:
    """
    Create a pipeline from a list of functions.
    
    Args:
        funcs: List of functions to use as pipeline stages
        config: Optional pipeline configuration
        streaming: Whether to use streaming function stages
        stage_names: Optional list of stage names
        
    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(config)
    
    for i, func in enumerate(funcs):
        name = stage_names[i] if stage_names and i < len(stage_names) else func.__name__
        
        if streaming:
            stage = StreamingFunctionStage(func, name=name)
        else:
            stage = FunctionStage(func, name=name)
        
        pipeline.add_stage(stage)
    
    return pipeline