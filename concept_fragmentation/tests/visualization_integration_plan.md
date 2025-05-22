# GPT-2 Visualization Integration Tests Plan

## Test Structure Overview

```
visualization/tests/
├── __init__.py
├── test_gpt2_integration.py           # Main integration test suite
├── test_gpt2_token_sankey.py          # Token Sankey diagram tests
├── test_gpt2_attention_flow.py        # Attention flow visualization tests
├── test_gpt2_metrics_dashboard.py     # Dashboard integration tests
├── test_gpt2_persistence_integration.py # Persistence integration tests
├── fixtures/
│   ├── __init__.py
│   ├── gpt2_test_data.py              # Mock data generators
│   └── dashboard_fixtures.py          # Dashboard testing utilities
└── utils/
    ├── __init__.py
    ├── visualization_validators.py     # Validation utilities
    └── mock_helpers.py                # Testing helper functions
```

## Test Categories

### 1. Data Pipeline Integration Tests
**File**: `test_gpt2_integration.py`

**Test Cases**:
- `test_gpt2_data_loading_pipeline()`: Complete data loading from mock results
- `test_token_metadata_processing()`: Token metadata extraction and processing
- `test_cluster_label_alignment()`: Alignment between tokens and cluster assignments
- `test_attention_data_integration()`: Attention data processing and validation
- `test_cross_layer_metrics_computation()`: Cross-layer metric calculations
- `test_error_handling_malformed_data()`: Error handling for malformed inputs

**Coverage**:
- Data loading functions in visualization modules
- Data transformation and preprocessing
- Integration between analysis results and visualization data structures

### 2. Token Sankey Visualization Tests  
**File**: `test_gpt2_token_sankey.py`

**Test Cases**:
- `test_extract_token_paths()`: Token path extraction from activations
- `test_sankey_data_preparation()`: Sankey diagram data structure creation
- `test_token_path_visualization()`: Complete visualization generation
- `test_token_highlighting()`: Selected token highlighting functionality
- `test_multi_layer_paths()`: Multi-layer token path tracking
- `test_sankey_interactivity()`: Interactive features and callbacks

**Coverage**:
- `extract_token_paths()` function
- `prepare_token_sankey_data()` function  
- `generate_token_sankey_diagram()` function
- Token selection and highlighting logic

### 3. Attention Flow Visualization Tests
**File**: `test_gpt2_attention_flow.py`

**Test Cases**:
- `test_attention_flow_extraction()`: Attention flow data extraction
- `test_attention_sankey_generation()`: Attention Sankey diagram creation
- `test_head_agreement_visualization()`: Head agreement network visualization
- `test_attention_entropy_heatmap()`: Attention entropy heatmap generation
- `test_attention_token_correlation()`: Token-attention correlation analysis
- `test_attention_layer_transitions()`: Cross-layer attention flow tracking

**Coverage**:
- `extract_attention_flow()` function
- `create_attention_sankey_diagram()` function
- `create_head_agreement_network()` function
- Attention-based metric calculations

### 4. Dashboard Integration Tests
**File**: `test_gpt2_metrics_dashboard.py`

**Test Cases**:
- `test_dashboard_tab_creation()`: GPT-2 metrics tab creation
- `test_summary_metrics_display()`: Summary metrics card functionality
- `test_drill_down_interactions()`: Drill-down panel interactions
- `test_visualization_tab_switching()`: Tab switching and state preservation
- `test_interactive_filtering()`: Interactive filtering across visualizations
- `test_linked_visualizations()`: Synchronized visualization updates
- `test_persistence_controls()`: Save/load functionality in dashboard

**Coverage**:
- `create_gpt2_metrics_tab()` function
- Dashboard callback functions
- Interactive component behavior
- State management across tabs

### 5. Persistence Integration Tests
**File**: `test_gpt2_persistence_integration.py`

**Test Cases**:
- `test_visualization_state_persistence()`: Saving/loading visualization states
- `test_analysis_export_with_visualizations()`: Export including visualization configs
- `test_session_visualization_restoration()`: Session-based visualization restoration
- `test_cached_visualization_data()`: Caching integration with visualizations
- `test_persistence_error_recovery()`: Error handling in persistence operations

**Coverage**:
- Integration between visualization components and persistence system
- Visualization state serialization/deserialization
- Cache integration with visualization data

## Test Data and Fixtures

### Mock Data Generator (`fixtures/gpt2_test_data.py`)

```python
class GPT2TestDataGenerator:
    """Generate realistic mock data for GPT-2 visualization tests."""
    
    def create_mock_analysis_results(self, num_layers=4, seq_length=8, batch_size=1):
        """Create comprehensive mock GPT-2 analysis results."""
        
    def create_mock_activations(self, layers, seq_length, batch_size, feature_dim=768):
        """Generate mock activation tensors."""
        
    def create_mock_attention_data(self, layers, num_heads=12, seq_length=8):
        """Generate mock attention weights and patterns."""
        
    def create_mock_cluster_labels(self, layers, seq_length, batch_size, num_clusters=3):
        """Generate mock cluster assignments."""
        
    def create_mock_token_metadata(self, seq_length, batch_size):
        """Generate mock token metadata."""
```

### Dashboard Testing Utilities (`fixtures/dashboard_fixtures.py`)

```python
class DashboardTestClient:
    """Utilities for testing Dash dashboard components."""
    
    def create_test_app(self):
        """Create test Dash app with GPT-2 components."""
        
    def simulate_callback(self, component_id, property, value):
        """Simulate dashboard callback interactions."""
        
    def validate_component_output(self, component, expected_structure):
        """Validate dashboard component structure and content."""
```

## Validation Utilities

### Visualization Validators (`utils/visualization_validators.py`)

```python
class PlotlyFigureValidator:
    """Validate Plotly figure structures and content."""
    
    def validate_sankey_diagram(self, figure, expected_nodes, expected_links):
        """Validate Sankey diagram structure."""
        
    def validate_heatmap(self, figure, expected_dimensions):
        """Validate heatmap structure."""
        
    def validate_network_graph(self, figure, expected_nodes, expected_edges):
        """Validate network graph structure."""

class DashComponentValidator:
    """Validate Dash component structures."""
    
    def validate_tab_structure(self, tab_component):
        """Validate tab component structure."""
        
    def validate_dropdown_options(self, dropdown, expected_options):
        """Validate dropdown options."""
        
    def validate_table_data(self, table, expected_columns, expected_rows):
        """Validate table data structure."""
```

## Test Execution Strategy

### 1. Test Environment Setup
- Create isolated test environment with temporary directories
- Mock external dependencies (transformers, file I/O)
- Set up test database/cache for persistence tests

### 2. Test Data Management
- Generate consistent test data across all test files
- Version test data to ensure reproducibility
- Clean up temporary test data after each test run

### 3. Test Execution Patterns
- Run tests in parallel where possible
- Use test fixtures for expensive setup operations
- Implement test timeouts for long-running visualization generation

### 4. Continuous Integration
- Run tests on multiple Python versions
- Test with different data sizes and configurations
- Include performance benchmarks for critical paths

## Success Criteria

### Coverage Targets
- **Unit Test Coverage**: >90% for individual functions
- **Integration Test Coverage**: >80% for end-to-end workflows
- **Dashboard Interaction Coverage**: 100% for user-facing callbacks

### Quality Metrics
- All tests pass on clean environments
- Test execution time < 2 minutes for full suite
- No memory leaks in visualization generation
- Consistent results across different platforms

### Functional Requirements
- All visualization components render without errors
- Interactive features work as expected
- Data integrity maintained throughout pipeline
- Error handling provides useful feedback
- Performance meets acceptable thresholds

## Implementation Priority

### Phase 1: Core Integration Tests
1. Data pipeline integration (`test_gpt2_integration.py`)
2. Basic visualization generation tests
3. Test data generators and fixtures

### Phase 2: Component-Specific Tests
1. Token Sankey visualization tests
2. Attention flow visualization tests
3. Dashboard component tests

### Phase 3: Advanced Integration
1. Cross-component interaction tests
2. Persistence integration tests
3. Performance and regression tests

This comprehensive testing plan ensures robust validation of all GPT-2 visualization functionality while maintaining consistency with the project's existing testing infrastructure.