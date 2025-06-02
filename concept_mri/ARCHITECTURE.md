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

### 3. Storage & Session Management

#### Activation Storage System
The app uses a hybrid storage approach to handle numpy array activations that cannot be efficiently stored in Dash's JSON-based stores:

**Components:**
- **ActivationManager** (`core/activation_manager.py`): Central manager that handles both activation extraction and session-based storage
  - Extracts activations from models using the concept_fragmentation pipeline
  - Stores activations in memory with session-based access
  - Provides memory management with configurable limits (default 2GB)
  - Automatic cleanup of expired sessions (default 2 hour timeout)

**Design Rationale:**
- Dash stores serialize data to JSON, converting numpy arrays to lists
- This causes type errors when passing to analysis functions expecting numpy arrays
- Session storage keeps arrays in original format while providing multi-user support

**Usage Pattern:**
```python
# Storing activations (in activation extraction callback)
session_id = activation_manager.store_activations(
    session_id=session_id,  # From session-id-store
    activations=processed_activations,
    metadata={...}
)
model_data['activation_session_id'] = session_id

# Retrieving activations (in analysis callbacks)
activations = activation_manager.get_activations(session_id)
if activations is None:
    # Fall back to direct storage for backward compatibility
    activations = model_data.get('activations', {})
```

### 4. Analysis Components

#### LLM Analysis Framework (Updated)
Uses the refactored comprehensive analysis system from `concept_fragmentation.llm.analysis`:

**Key Features:**
- **Single API Call**: All archetypal paths analyzed together for better pattern detection
- **Analysis Categories**: Supports multiple analysis types in one call
  - `interpretation`: Conceptual understanding and decision patterns
  - `bias`: Cross-path demographic analysis and fairness detection
  - `efficiency`: Redundancy and optimization opportunities
  - `robustness`: Stability and vulnerability assessment
- **Comprehensive Context**: LLM sees all paths, demographics, and statistics at once

**Implementation:**
```python
# Using the refactored ClusterAnalysis
analyzer = ClusterAnalysis(provider="openai", model="gpt-4")
result = analyzer.generate_path_narratives_sync(
    paths=archetypal_paths,
    cluster_labels=labels,
    path_demographic_info=demographics,
    analysis_categories=['interpretation', 'bias']
)
# Returns comprehensive analysis string, not individual narratives
```

**Bias Detection Capabilities:**
- Identifies systematic demographic routing differences
- Detects unexpected segregation patterns
- Finds statistical anomalies across paths
- Provides actionable insights for fairness improvements

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
- Uses existing clustering algorithms (K-Means, DBSCAN, ETS)
- Leverages path extraction and archetypal path analysis
- Imports metrics calculations (fragmentation, purity, etc.)
- Wraps visualization components
- **Uses ClusterAnalysis for comprehensive LLM analysis**

### With LLM providers:
- Supports multiple providers (OpenAI, Anthropic, Gemini, Grok/xAI)
- Unified interface through `concept_fragmentation.llm.factory`
- Built-in caching layer for responses
- API keys configured via `local_config.py`

## Configuration

### API Configuration
```python
# local_config.py (gitignored)
OPENAI_KEY = "your-key"
XAI_API_KEY = "your-key"
GEMINI_API_KEY = "your-key"
MONGO_URI = "mongodb://..."
```

### Analysis Configuration
- Analysis categories selected at runtime
- No separate configuration files needed
- Prompts are built dynamically based on:
  - Selected analysis categories
  - Available data (demographics, statistics, etc.)
  - Network architecture (FF vs Transformer)

## Recent Enhancements

### Phase A Completed:
1. **Layer Window Manager**: 
   - Manual presets (GPT-2, thirds, quarters, halves)
   - Interactive window configuration
   - Experimental auto-detection using metrics
   - Visual metric plots for guidance

2. **ETS Clustering Integration**:
   - Threshold percentile controls
   - Batch size configuration
   - ETS-specific visualizations

3. **Macro/Meso/Micro Hierarchy**:
   - Adaptive K calculation based on hierarchy level
   - Integrated with all visualizations

4. **Enhanced Visualizations**:
   - Window-aware Sankey diagrams
   - Stepped trajectory visualization (3 modes)
   - Enhanced cluster cards with confidence intervals

### LLM Analysis Refactoring:
1. **Comprehensive Analysis**: Single API call with all paths
2. **Cross-Path Pattern Detection**: Essential for bias analysis
3. **Flexible Categories**: Choose analysis types at runtime
4. **Proven Bias Detection**: Successfully tested on heart disease data