# GPT-2 APA Architecture Overview

This document provides a comprehensive overview of the GPT-2 Archetypal Path Analysis (APA) system architecture, detailing how components interact to enable transformer-specific concept fragmentation analysis.

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     GPT-2 APA System                           │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Text Input  │  │ Config File │  │ Batch Input │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Processing Layer                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              GPT-2 Integration                              ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        ││
│  │  │  Tokenizer  │  │    Model    │  │ Attention   │        ││
│  │  │  Pipeline   │  │  Adapter    │  │ Extractor   │        ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                APA Analysis Engine                          ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        ││
│  │  │ Clustering  │  │ Path        │  │ Cross-Layer │        ││
│  │  │ Engine      │  │ Analysis    │  │ Metrics     │        ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘        ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  Output Layer                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Interactive │  │ Static      │  │ Analysis    │            │
│  │ Dashboard   │  │ Plots       │  │ Reports     │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

```
GPT2ActivationExtractor
├── Model Management
│   ├── GPT2ModelLoader
│   ├── TokenizerPipeline
│   └── ConfigurationManager
├── Activation Processing
│   ├── LayerActivationExtractor
│   ├── AttentionPatternExtractor
│   └── WindowingManager
└── Data Management
    ├── ActivationSerializer
    ├── CacheManager
    └── BatchProcessor

GPT2PathAnalyzer
├── Clustering Engine
│   ├── AttentionWeightedClustering
│   ├── TokenAwareClustering
│   └── ClusterValidator
├── Path Analysis
│   ├── CrossLayerPathTracker
│   ├── FragmentationMetrics
│   └── BiasDetector
└── Metrics Engine
    ├── TransformerMetrics
    ├── AttentionMetrics
    └── PathMetrics

VisualizationPipeline
├── Interactive Components
│   ├── DashboardGenerator
│   ├── TokenPathVisualizer
│   └── AttentionHeatmapGenerator
├── Static Plots
│   ├── SankeyDiagramGenerator
│   ├── CrossLayerFlowPlotter
│   └── MetricsPlotter
└── Report Generation
    ├── LLMNarrativeGenerator
    ├── FigureCompiler
    └── ReportFormatter
```

## Core Components

### 1. GPT2ActivationExtractor

**Purpose**: Primary interface for extracting activations and attention patterns from GPT-2 models.

**Key Responsibilities**:
- Model loading and configuration
- Activation extraction from specified layers
- Attention pattern capture across all heads
- Token metadata management
- Sliding window processing for long sequences

**Implementation Details**:
```python
class GPT2ActivationExtractor:
    def __init__(self, model_type: str, config: GPT2ActivationConfig):
        self.model_loader = GPT2ModelLoader(model_type)
        self.tokenizer = TokenizerPipeline(model_type)
        self.config = config
        self.cache_manager = CacheManager(config.cache_dir)
        
    def extract_activations_for_windows(self, text: str) -> Dict:
        # 1. Tokenize input text
        # 2. Create sliding windows
        # 3. Extract activations for each window
        # 4. Capture attention patterns
        # 5. Package with metadata
```

**Data Flow**:
```
Text Input → Tokenization → Model Forward Pass → Activation Hooks → Window Processing → Structured Output
```

### 2. GPT2PathAnalyzer

**Purpose**: Analyzes activation paths through GPT-2 layers using APA methodology.

**Key Responsibilities**:
- Attention-weighted clustering of activations
- Cross-layer path tracking
- Fragmentation metric computation
- Bias detection in attention patterns
- LLM-powered narrative generation

**Implementation Details**:
```python
class GPT2PathAnalyzer:
    def __init__(self):
        self.clustering_engine = AttentionWeightedClustering()
        self.path_tracker = CrossLayerPathTracker()
        self.bias_detector = BiasDetector()
        self.metrics_engine = TransformerMetrics()
        
    def analyze_paths(self, activation_data: Dict) -> GPT2AnalysisResults:
        # 1. Apply attention-weighted clustering
        # 2. Track paths across layers
        # 3. Compute fragmentation metrics
        # 4. Detect attention biases
        # 5. Generate analysis narrative
```

### 3. Visualization Pipeline

**Purpose**: Creates interactive and static visualizations of GPT-2 APA results.

**Key Responsibilities**:
- Interactive dashboard generation
- Token path visualization
- Attention pattern heatmaps
- Cross-layer flow diagrams
- Automated report generation

## Data Structures

### Core Data Types

```python
@dataclass
class GPT2ActivationData:
    """Container for GPT-2 activation data."""
    layer_activations: Dict[str, torch.Tensor]  # Layer name -> activations
    attention_patterns: Dict[str, torch.Tensor]  # Layer name -> attention weights
    token_metadata: TokenMetadata
    model_config: GPT2ModelConfig
    extraction_config: GPT2ActivationConfig

@dataclass
class TokenMetadata:
    """Metadata about tokens in the input sequence."""
    tokens: List[str]
    token_ids: List[int]
    positions: List[int]
    attention_mask: List[bool]
    special_tokens: Dict[int, str]

@dataclass
class GPT2AnalysisResults:
    """Results of GPT-2 APA analysis."""
    cluster_assignments: Dict[str, np.ndarray]
    path_metrics: Dict[str, float]
    attention_analysis: AttentionAnalysis
    bias_detection: BiasAnalysis
    narrative: str
    visualization_data: Dict[str, Any]
```

### Attention Data Structures

```python
@dataclass
class AttentionAnalysis:
    """Analysis of attention patterns."""
    head_specialization: Dict[str, Dict[int, float]]  # Layer -> Head -> Specialization score
    attention_entropy: Dict[str, np.ndarray]  # Layer -> Entropy values
    pattern_types: Dict[str, List[str]]  # Layer -> Pattern categories
    bias_scores: Dict[str, float]  # Bias type -> Score

@dataclass
class BiasAnalysis:
    """Results of bias detection analysis."""
    attention_biases: Dict[str, float]  # Bias type -> Score
    token_level_biases: Dict[str, Dict[str, float]]  # Token -> Bias type -> Score
    layer_specific_biases: Dict[str, Dict[str, float]]  # Layer -> Bias type -> Score
    recommendations: List[str]
```

## Processing Pipeline

### 1. Initialization Phase

```python
# Configuration and model loading
config = GPT2ActivationConfig(
    model_type="gpt2-medium",
    context_window=512,
    include_attention=True,
    window_size=3
)

extractor = GPT2ActivationExtractor(config=config)
analyzer = GPT2PathAnalyzer()
```

### 2. Extraction Phase

```python
# Text processing and activation extraction
text = "Input text for analysis"
activation_data = extractor.extract_full_analysis(text)

# Data structure:
# {
#   "layer_activations": {"layer_0": tensor, "layer_1": tensor, ...},
#   "attention_patterns": {"layer_0": tensor, "layer_1": tensor, ...},
#   "token_metadata": TokenMetadata(...),
#   "windows": {"window_0_2": {...}, "window_1_3": {...}, ...}
# }
```

### 3. Analysis Phase

```python
# APA analysis with transformer-specific enhancements
analysis_results = analyzer.analyze_paths(activation_data)

# Results include:
# - Cluster assignments for each layer
# - Cross-layer path metrics
# - Attention pattern analysis
# - Bias detection results
# - Generated narrative explanation
```

### 4. Visualization Phase

```python
# Generate interactive and static visualizations
from visualization.gpt2_token_tab import create_gpt2_visualization
from visualization.dash_app import create_dashboard

# Interactive dashboard
dashboard = create_dashboard(analysis_results)

# Static visualizations
figures = create_gpt2_visualization(analysis_results)
```

## Memory Management

### Efficient Processing Strategies

1. **Gradient Checkpointing**: Reduces memory usage during forward passes
2. **Activation Offloading**: Stores intermediate activations on disk
3. **Sliding Windows**: Processes long sequences in manageable chunks
4. **Sparse Attention**: Uses sparse attention computation when possible

### Memory Configuration

```python
memory_config = GPT2ActivationConfig(
    use_gradient_checkpointing=True,
    activation_offload="disk",
    max_sequence_length=1024,
    batch_size=4,
    memory_limit="8GB"
)
```

## Extensibility Points

### 1. Custom Model Adapters

```python
class CustomGPT2Adapter(GPT2ActivationExtractor):
    """Custom adapter for specialized GPT-2 analysis."""
    
    def extract_custom_features(self, text: str) -> Dict:
        # Custom feature extraction logic
        pass
        
    def apply_custom_clustering(self, activations: Dict) -> Dict:
        # Custom clustering algorithm
        pass
```

### 2. Analysis Extensions

```python
class CustomAnalyzer(GPT2PathAnalyzer):
    """Custom analyzer with domain-specific metrics."""
    
    def compute_domain_metrics(self, results: GPT2AnalysisResults) -> Dict:
        # Domain-specific analysis
        pass
        
    def detect_custom_patterns(self, attention_data: Dict) -> Dict:
        # Custom pattern detection
        pass
```

### 3. Visualization Extensions

```python
class CustomVisualizer:
    """Custom visualization components."""
    
    def create_custom_plot(self, data: Dict) -> plotly.graph_objects.Figure:
        # Custom plot generation
        pass
        
    def add_custom_dashboard_tab(self, app: dash.Dash, data: Dict):
        # Custom dashboard components
        pass
```

## Performance Considerations

### Computational Complexity

- **Activation Extraction**: O(L × H × S²) where L=layers, H=heads, S=sequence length
- **Clustering**: O(N × K × D) where N=samples, K=clusters, D=dimensions
- **Path Analysis**: O(L × N × K) for cross-layer metrics
- **Attention Analysis**: O(L × H × S²) for attention pattern processing

### Optimization Strategies

1. **Parallel Processing**: Multi-GPU support for large models
2. **Caching**: Intelligent caching of activations and results
3. **Batch Processing**: Efficient processing of multiple texts
4. **Memory Mapping**: Use memory-mapped files for large datasets

### Scalability Limits

- **Single GPU**: GPT-2 Large (774M parameters) with 512 token sequences
- **Multi-GPU**: GPT-2 XL (1.5B parameters) with 1024 token sequences
- **CPU-only**: GPT-2 Small (117M parameters) with 256 token sequences

## Integration Points

### External System Integration

1. **Hugging Face Transformers**: Direct integration with HF model hub
2. **PyTorch**: Native PyTorch tensor operations
3. **Plotly/Dash**: Interactive visualization framework
4. **LLM APIs**: Integration with external LLM services for narrative generation

### Data Persistence

```python
# Save analysis results
saver = AnalysisResultsSaver(
    output_dir="./results",
    format="json",  # Options: json, hdf5, pickle
    compression=True
)

metadata_path = saver.save_analysis(analysis_results)

# Load analysis results
loader = AnalysisResultsLoader()
loaded_results = loader.load_analysis(metadata_path)
```

## Testing Framework

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end pipeline testing
3. **Performance Tests**: Memory and speed benchmarks
4. **Regression Tests**: Ensure consistent results across versions

### Test Data Generation

```python
from visualization.tests.fixtures.gpt2_test_data import GPT2TestDataGenerator

# Generate test data
test_generator = GPT2TestDataGenerator()
mock_data = test_generator.create_mock_analysis_results(
    model_type="gpt2",
    sequence_length=128,
    n_layers=12
)
```

This architecture provides a robust, extensible foundation for GPT-2 Archetypal Path Analysis, supporting both research and production use cases while maintaining computational efficiency and interpretability.