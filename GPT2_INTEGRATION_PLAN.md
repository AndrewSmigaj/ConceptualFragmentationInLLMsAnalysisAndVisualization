# GPT-2 Integration Plan for Archetypal Path Analysis

This document outlines the plan for extending the Archetypal Path Analysis (APA) framework to work with GPT-2 and other transformer-based models.

## Core Architectural Improvements

Before implementing GPT-2 specific features, we need to enhance the current architecture to better support complex models:

### 1. Model Architecture Abstraction

Create a unified interface for working with different model architectures:

```python
class ModelInterface(Protocol):
    """Unified interface for any model architecture."""
    
    def get_embeddings(self, inputs: Any) -> Tensor:
        """Get embedding layer outputs."""
        
    def get_layer_outputs(self, layer_indices: List[int]) -> Dict[int, Tensor]:
        """Get outputs from specific layers."""
    
    def get_attention_patterns(self, layer_indices: List[int]) -> Dict[int, Tensor]:
        """Get attention patterns from specific layers (for models with attention)."""
```

This abstraction will allow us to implement specialized adapters for different model types while maintaining a consistent interface throughout the codebase.

### 2. Enhanced Activation Collection

Redesign the activation collection system for better memory efficiency:

```python
class ActivationCollector:
    """Manages activation collection with memory efficiency."""
    
    def collect(
        self, 
        model: ModelInterface, 
        inputs: Any, 
        activation_points: List[str],
        streaming: bool = False,
        batch_size: Optional[int] = None
    ) -> Union[Dict[str, Tensor], Generator]:
        """Collect activations, optionally as a stream."""
```

This will allow processing large models like GPT-2 without memory issues, especially when handling many layers or large batch sizes.

### 3. Flexible Data Pipeline

Implement a proper data processing pipeline:

```python
class PipelineStage(Protocol):
    """A stage in the data processing pipeline."""
    
    def process(self, data: Any) -> Any:
        """Process input data and return output."""
    
    def can_stream(self) -> bool:
        """Whether this stage supports streaming processing."""

class Pipeline:
    """Executes pipeline stages with optimization."""
    
    def execute(
        self, 
        stages: List[PipelineStage], 
        input_data: Any, 
        memory_limit: Optional[int] = None
    ) -> Any:
        """Execute pipeline stages optimizing for memory usage."""
```

This architecture allows for more flexible composition of analysis steps and better memory management when working with large models.

## GPT-2 Specific Components

With the enhanced architecture in place, we can implement GPT-2 specific components:

### 1. Transformer Model Adapter

```python
class TransformerModelAdapter(ModelInterface):
    """Adapter for transformer models like GPT-2."""
    
    def __init__(self, model, model_type: str = "gpt2"):
        self.model = model
        self.model_type = model_type
        self._layer_mapping = self._create_layer_mapping()
    
    def _create_layer_mapping(self) -> Dict[str, nn.Module]:
        """Create mapping from logical names to actual model modules."""
        if self.model_type == "gpt2":
            return {
                f"transformer_block_{i}": self.model.transformer.h[i]
                for i in range(len(self.model.transformer.h))
            }
```

This adapter will provide a consistent interface to GPT-2's layers while hiding the architectural details.

### 2. Transformer-Specific Metrics

```python
class TransformerMetrics(MetricComputer):
    """Computes transformer-specific metrics."""
    
    def compute_attention_entropy(self, attention_weights: Tensor) -> float:
        """Compute entropy of attention distributions."""
    
    def compute_attention_sparsity(self, attention_weights: Tensor) -> float:
        """Compute sparsity of attention (% of near-zero weights)."""
    
    def compute_head_importance(self, attention_outputs: Dict[str, Tensor]) -> Dict[str, float]:
        """Compute importance scores for different attention heads."""
```

These metrics will provide insights specific to transformer models like GPT-2.

### 3. Token vs Sequence Level Analysis

```python
class TokenLevelAnalysis:
    """Analyzes activations at the token level."""
    
    def cluster_token_activations(self, activations: Tensor) -> Dict[str, Any]:
        """Cluster activations at the token level."""
    
    def track_token_paths(self, layer_activations: Dict[str, Tensor]) -> Dict[str, Any]:
        """Track how tokens move through activation space."""

class SequenceLevelAnalysis:
    """Analyzes activations at the sequence level."""
    
    def aggregate_token_activations(self, token_activations: Tensor, method: str = "mean") -> Tensor:
        """Aggregate token-level activations to sequence level."""
```

This will allow for both token-level and sequence-level analysis, which is important for language models.

## Implementation Plan

The implementation will proceed in phases:

### Phase 1: Core Architecture Enhancement (4 weeks)

1. **Week 1-2**: Implement model abstraction and adapter interfaces
   - Create ModelInterface protocol
   - Implement adapter for current MLP models
   - Write comprehensive tests
   
2. **Week 3-4**: Enhance activation collection and pipeline
   - Implement ActivationCollector with streaming
   - Create pipeline architecture
   - Implement persistence layer improvements

### Phase 2: GPT-2 Integration (3 weeks)

1. **Week 1**: GPT-2 model adapter implementation
   - Implement TransformerModelAdapter
   - Add support for extracting attention patterns
   - Create transformer-specific activation hooks
   
2. **Week 2**: Clustering and dimensionality reduction for transformer activations
   - Implement token vs sequence level analysis
   - Enhance dimensionality reduction for high-dim spaces
   - Create specialized clustering for transformer representations
   
3. **Week 3**: Metrics and analysis for transformers
   - Implement attention-specific metrics
   - Create cross-layer analysis for transformers
   - Develop path tracking for transformer models

### Phase 3: Visualization and LLM Integration (3 weeks)

1. **Week 1**: Enhance visualization for transformers
   - Implement hierarchical layer visualization
   - Create attention pattern visualizations
   - Add token highlighting in visualizations
   
2. **Week 2**: LLM prompt engineering for transformers
   - Develop transformer-specific narrative prompts
   - Implement attention pattern description
   - Create prompts for token relationships
   
3. **Week 3**: Dashboard integration and testing
   - Integrate all components in the dashboard
   - Comprehensive testing with different input types
   - Performance optimization

### Phase 4: Evaluation and Documentation (2 weeks)

1. **Week 1**: Evaluation on standard benchmarks
   - Compare with baseline methods
   - Analyze computational efficiency
   - User testing with interpretability researchers
   
2. **Week 2**: Comprehensive documentation
   - API documentation
   - Usage examples
   - Architecture documentation
   - Contribution guidelines

## Design Principles

Throughout implementation, these principles will guide development:

1. **Separation of Concerns**: Clearly separate model-specific code from general analysis
2. **Interface Stability**: Design stable interfaces that won't need to change with new models
3. **Memory Efficiency**: Prioritize approaches that can scale to large models
4. **Progressive Enhancement**: Ensure the system continues to work with simpler models
5. **Testability**: Design for comprehensive testing at all levels
6. **Documentation**: Document design decisions and architectural patterns

## Expected Outcomes

After implementation, the APA framework will:

1. Support analysis of GPT-2 from small to XL variants
2. Provide insights into how language models process text
3. Generate human-readable narratives explaining transformer behavior
4. Allow comparison between traditional MLPs and transformer models
5. Visualize attention patterns and their relationship to clusters
6. Identify key pathways through transformer layers
7. Track concept evolution across 12+ layers

## Future Extensions

This architecture will make it easier to:

1. Add support for other transformer models (BERT, OPT, Llama)
2. Implement more advanced metrics for language models
3. Scale to even larger models with distributed computing
4. Integrate with other interpretability tools and frameworks
5. Support multi-modal models in the future