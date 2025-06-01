# Concept MRI Architecture

## Overview
Concept MRI is a web-based tool for analyzing neural networks using Concept Trajectory Analysis (CTA). It provides interactive visualizations and LLM-powered interpretations of how concepts flow through neural network layers.

## Core Components

### 1. Control Components
- **ModelUploadPanel**: Handles model file uploads (.pt, .onnx, etc.)
- **DatasetUploadPanel**: Manages dataset uploads for analysis
- **ClusteringPanel**: Configuration for clustering algorithms (K-Means, DBSCAN, ETS)
- **LayerWindowManager**: Splits deep networks into analyzable windows
- **APIKeysPanel**: Manages LLM API credentials

### 2. Visualization Components
- **SankeyWrapper**: Window-aware Sankey diagrams for concept flow
- **SteppedTrajectoryVisualization**: Multiple modes for trajectory analysis
- **ClusterCards**: Rich cluster information display (standard, ETS, hierarchical)

### 3. Analysis Components

#### LLM Analysis Framework (New)
Extensible system for different types of LLM-powered analyses:

```
LLMAnalyzer (base class)
├── BiasAnalyzer: Detects unfair patterns and biases
├── InterpretabilityAnalyzer: Explains what the network learned
├── EfficiencyAnalyzer: Identifies optimization opportunities
├── RobustnessAnalyzer: Assesses vulnerabilities
└── CustomAnalyzer: User-defined analysis types
```

**Key Features:**
- Standardized data format for all analyzers
- Configuration-driven prompt templates
- Consistent output format
- Domain-aware analysis
- Result caching

**Data Flow:**
1. Network analysis data (paths, clusters, metrics)
2. Analyzer prepares and formats data
3. Generates domain-specific prompts
4. LLM provides interpretation
5. Results parsed and displayed

### 4. Data Flow

```
Model Upload → Activation Extraction → Clustering → Path Analysis
                                           ↓
                                    Window Management
                                           ↓
                              Visualizations + LLM Analysis
```

### 5. State Management
- **model-store**: Model data and activations
- **clustering-store**: Clustering results
- **window-config-store**: Layer window configuration
- **path-analysis-store**: Archetypal paths
- **cluster-labels-store**: LLM-generated labels
- **hierarchy-results-store**: Multi-level clustering

## Key Design Patterns

### 1. Component Pattern
Each UI component is self-contained with:
- Creation method
- Callback registration
- State management

### 2. Template Method Pattern (LLM Analyzers)
- Base class defines analysis workflow
- Subclasses implement specific prompts
- Shared data preparation

### 3. Registry Pattern (Analysis Types)
- Auto-discovery of available analyzers
- Plugin system for custom analyses
- Configuration-based enabling

## Integration Points

### With concept_fragmentation library:
- Uses existing clustering algorithms
- Leverages path extraction
- Imports metrics calculations
- Wraps visualization components

### With LLM providers:
- Supports multiple providers (OpenAI, Anthropic, etc.)
- Unified interface through llm_client
- Caching layer for responses

## Configuration

### Analysis Configuration
```yaml
# analysis_configs.yaml
bias:
  name: "Bias Analysis"
  required_metrics: ["path_distribution", "cluster_composition"]
  prompt_template: "bias_analysis_v1"
  
interpretability:
  name: "Interpretability Analysis"  
  required_metrics: ["cluster_semantics", "layer_progression"]
  prompt_template: "interpretability_v1"
```

### Prompt Templates
- Versioned for reproducibility
- Domain-aware placeholders
- Standardized output format

## Recent Enhancements

### Phase A Additions:
1. **ETS Clustering**: Explainable threshold-based clustering
2. **Hierarchy Control**: Macro/Meso/Micro analysis levels
3. **Window Management**: Analyze deep networks in segments
4. **Enhanced Visualizations**: Richer, interactive displays

### LLM Analysis Framework:
1. **Extensible Design**: Easy to add new analysis types
2. **Standardized Interface**: Consistent usage pattern
3. **Domain Awareness**: Interprets based on application context
4. **Configuration-Driven**: Prompts and parsers in config files