"""
Flexible pipeline architecture for data processing.

This module provides a flexible pipeline system for processing neural network
activations and other data through a sequence of operations. It focuses on 
memory efficiency, streaming capabilities, and flexible component composition.

The pipeline architecture allows for:
- Memory-efficient processing of large activation datasets
- Streaming operations for reduced memory usage
- Composable stages that can be mixed and matched
- Conditional execution paths based on data properties
- Parallel execution of independent operations
- Caching of intermediate results
"""

from .pipeline import (
    Pipeline, PipelineStage, PipelineStageBase, StreamingStage,
    FunctionStage, StreamingFunctionStage, ParallelStage,
    ConditionalStage, CachingStage, PipelineContext,
    PipelineConfig, StreamingMode, create_function_pipeline
)

from .stages import (
    ActivationCollectionStage, ActivationProcessingStage,
    ClusteringStage, ClusterPathStage, PathArchetypeStage,
    PersistenceStage, LLMAnalysisStage
)

__all__ = [
    # Core pipeline classes
    'Pipeline',
    'PipelineStage',
    'PipelineStageBase',
    'StreamingStage',
    'FunctionStage',
    'StreamingFunctionStage',
    'ParallelStage',
    'ConditionalStage',
    'CachingStage',
    'PipelineContext',
    'PipelineConfig',
    'StreamingMode',
    'create_function_pipeline',
    
    # Specialized stages
    'ActivationCollectionStage',
    'ActivationProcessingStage',
    'ClusteringStage',
    'ClusterPathStage',
    'PathArchetypeStage',
    'PersistenceStage',
    'LLMAnalysisStage'
]