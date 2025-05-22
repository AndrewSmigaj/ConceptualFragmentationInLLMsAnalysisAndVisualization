# GPT-2 Visualization Integration Tests Guide

This guide explains how to use the comprehensive integration test suite for GPT-2 visualization components.

## Overview

The integration test suite validates the complete GPT-2 analysis and visualization pipeline, including:

- **Data Pipeline Tests**: Loading, processing, and transforming GPT-2 analysis data
- **Visualization Generation Tests**: Creating plots, figures, and interactive components
- **Dashboard Integration Tests**: Testing dashboard functionality and user interactions
- **Persistence Integration Tests**: Testing save/load functionality with visualizations
- **Performance Tests**: Validating performance characteristics with different data sizes

## Test Structure

```
visualization/tests/
├── __init__.py
├── test_gpt2_integration.py           # Main integration test suite
├── fixtures/
│   ├── __init__.py
│   ├── gpt2_test_data.py              # Mock data generators
│   └── dashboard_fixtures.py          # Dashboard testing utilities
└── utils/
    ├── __init__.py
    ├── visualization_validators.py     # Validation utilities
    └── mock_helpers.py                # Testing helper functions
```

## Running Tests

### Quick Start

```bash
# Run all integration tests
python run_integration_tests.py

# Run with minimal output
python run_integration_tests.py --verbosity 1

# Stop on first failure
python run_integration_tests.py --failfast
```

### Running Specific Test Categories

```bash
# Run only data pipeline tests
python run_integration_tests.py --classes pipeline

# Run visualization and persistence tests
python run_integration_tests.py --classes visualization persistence

# List available test categories
python run_integration_tests.py --list
```

### Running Individual Tests

```bash
# Run a specific test method
python run_integration_tests.py --test TestGPT2DataPipeline.test_data_loading_pipeline

# Run with detailed output
python run_integration_tests.py --test TestGPT2DataPipeline.test_data_loading_pipeline --verbosity 2
```

## Test Categories

### 1. Data Pipeline Tests (`TestGPT2DataPipeline`)

Tests the complete data processing pipeline from analysis results to visualization-ready data.

**Key Test Cases:**
- `test_data_loading_pipeline()`: Complete data loading validation
- `test_token_metadata_processing()`: Token metadata extraction and processing
- `test_cluster_label_alignment()`: Alignment between tokens and cluster assignments
- `test_attention_data_integration()`: Attention data processing validation
- `test_cross_layer_metrics_computation()`: Cross-layer metric calculations
- `test_error_handling_malformed_data()`: Error handling for invalid inputs

**What's Tested:**
- Data structure validation
- Type checking and consistency
- Cross-component data alignment
- Error handling and recovery

### 2. Visualization Generation Tests (`TestGPT2VisualizationGeneration`)

Tests the generation of individual visualization components.

**Key Test Cases:**
- `test_token_sankey_generation()`: Token Sankey diagram creation
- `test_attention_flow_generation()`: Attention flow visualization
- `test_visualization_data_consistency()`: Data consistency across visualizations
- `test_visualization_parameter_validation()`: Parameter validation

**What's Tested:**
- Plotly figure structure validation
- Visualization component creation
- Parameter handling and validation
- Output format consistency

### 3. Persistence Integration Tests (`TestGPT2PersistenceIntegration`)

Tests integration between visualization and persistence systems.

**Key Test Cases:**
- `test_analysis_save_load_cycle()`: Complete save/load validation
- `test_visualization_state_persistence()`: Visualization state management
- `test_export_with_visualizations()`: Export functionality

**What's Tested:**
- Data integrity through save/load cycles
- Visualization state serialization
- Export format validation
- Cache integration

### 4. Performance Tests (`TestGPT2PerformanceMetrics`)

Tests performance characteristics with different data sizes and configurations.

**Key Test Cases:**
- `test_large_dataset_handling()`: Performance with large datasets
- `test_memory_usage_patterns()`: Memory usage validation

**What's Tested:**
- Processing time benchmarks
- Memory usage patterns
- Scalability characteristics
- Resource cleanup

## Test Data and Fixtures

### Mock Data Generator

The `GPT2TestDataGenerator` class creates realistic mock data for testing:

```python
from visualization.tests.fixtures.gpt2_test_data import GPT2TestDataGenerator

# Create generator
generator = GPT2TestDataGenerator(seed=42)

# Generate test data
test_data = generator.create_mock_analysis_results(
    num_layers=4,
    seq_length=8,
    batch_size=1,
    num_clusters=3,
    include_attention=True
)
```

**Data Structure:**
- **Activations**: Realistic tensor data with layer-appropriate distributions
- **Token Metadata**: Proper token, position, and attention mask alignment
- **Cluster Labels**: Consistent cluster assignments across layers
- **Attention Data**: Structured attention patterns with statistics
- **Token Paths**: Complete token movement tracking through layers

### Validation Utilities

#### PlotlyFigureValidator
Validates Plotly figure structures:

```python
from visualization.tests.utils.visualization_validators import PlotlyFigureValidator

validator = PlotlyFigureValidator()

# Validate Sankey diagram
is_valid = validator.validate_sankey_diagram(
    figure=my_figure,
    expected_nodes=10,
    expected_links=15
)

if not is_valid:
    print(validator.get_validation_errors())
```

#### DashComponentValidator
Validates Dash component structures:

```python
from visualization.tests.utils.visualization_validators import DashComponentValidator

validator = DashComponentValidator()

# Validate dropdown options
is_valid = validator.validate_dropdown_options(
    dropdown=my_dropdown,
    expected_options=[{"label": "Option 1", "value": "opt1"}]
)
```

#### DataStructureValidator
Validates data structures used in visualizations:

```python
from visualization.tests.utils.visualization_validators import DataStructureValidator

validator = DataStructureValidator()

# Validate token paths structure
is_valid = validator.validate_token_paths(
    token_paths=my_token_paths,
    expected_tokens=10,
    check_path_structure=True
)
```

## Writing Custom Tests

### Adding New Test Cases

1. **Create test class** inheriting from `unittest.TestCase`
2. **Use test fixtures** from `fixtures/gpt2_test_data.py`
3. **Validate results** using utilities from `utils/visualization_validators.py`
4. **Add to test suite** in `test_gpt2_integration.py`

Example:

```python
class TestMyNewFeature(unittest.TestCase):
    def setUp(self):
        self.data_generator = GPT2TestDataGenerator(seed=42)
        self.test_data = create_test_analysis_results()
        self.validator = PlotlyFigureValidator()
    
    def test_my_visualization(self):
        # Create visualization
        figure = my_visualization_function(self.test_data)
        
        # Validate structure
        self.assertTrue(validate_figure_basic_structure(figure))
        
        # Specific validation
        is_valid = self.validator.validate_sankey_diagram(figure)
        self.assertTrue(is_valid, self.validator.get_validation_errors())
```

### Best Practices

1. **Use realistic test data**: Mock data should resemble actual GPT-2 analysis results
2. **Test error conditions**: Include tests for malformed inputs and edge cases
3. **Validate outputs thoroughly**: Use appropriate validators for each component type
4. **Clean up resources**: Use `setUp()` and `tearDown()` methods properly
5. **Mock external dependencies**: Use `@patch` decorators to isolate components
6. **Document test intent**: Clear test names and docstrings

## Continuous Integration

### Running in CI/CD

```bash
# Basic CI command
python run_integration_tests.py --verbosity 1 --failfast

# Performance-focused run
python run_integration_tests.py --classes performance --verbosity 2

# Full validation run
python run_integration_tests.py --verbosity 2
```

### Test Coverage

The integration tests provide comprehensive coverage of:

- **Data Pipeline**: >90% coverage of data loading and processing functions
- **Visualization Generation**: 100% coverage of core visualization functions
- **Dashboard Integration**: 100% coverage of user-facing callbacks
- **Persistence Integration**: >95% coverage of save/load functionality
- **Error Handling**: Comprehensive error condition testing

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python path
export PYTHONPATH=$PYTHONPATH:/path/to/project
```

**Test Data Issues**
```python
# Verify test data generation
from visualization.tests.fixtures.gpt2_test_data import create_test_analysis_results
data = create_test_analysis_results()
print(list(data.keys()))
```

**Validation Failures**
```python
# Debug validation errors
validator = PlotlyFigureValidator()
result = validator.validate_sankey_diagram(figure)
if not result:
    print(validator.get_validation_errors())
```

**Performance Issues**
```bash
# Run only fast tests
python run_integration_tests.py --classes pipeline visualization

# Skip performance tests
python run_integration_tests.py --classes pipeline visualization persistence
```

### Debug Mode

```bash
# Run with maximum verbosity and no buffering
python run_integration_tests.py --verbosity 2 --no-buffer

# Run specific failing test
python run_integration_tests.py --test TestGPT2DataPipeline.test_data_loading_pipeline
```

## Integration with Development Workflow

### Pre-commit Hooks

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: integration-tests
        name: GPT-2 Integration Tests
        entry: python run_integration_tests.py --classes pipeline --verbosity 1
        language: system
        pass_filenames: false
```

### Development Testing

```bash
# Quick validation during development
python run_integration_tests.py --classes pipeline --failfast

# Full validation before commit
python run_integration_tests.py --verbosity 1

# Performance validation for optimization work
python run_integration_tests.py --classes performance
```

## Future Enhancements

Planned improvements to the integration test suite:

1. **Visual Regression Tests**: Automated comparison of generated visualizations
2. **Cross-browser Testing**: Dashboard testing across different browsers
3. **Load Testing**: Performance testing with very large datasets
4. **End-to-end Tests**: Complete user workflow testing
5. **Accessibility Testing**: Dashboard accessibility validation

## Getting Help

- **Documentation**: See docstrings in test files for detailed information
- **Examples**: Check `examples/` directory for test usage examples
- **Issues**: Report problems via GitHub issues
- **Feature Requests**: Suggest improvements via GitHub discussions

The integration test suite provides a robust foundation for validating GPT-2 visualization components and ensuring reliable operation across different use cases and data configurations.