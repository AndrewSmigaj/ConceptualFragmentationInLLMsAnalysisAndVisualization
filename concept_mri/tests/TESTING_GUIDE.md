# Concept MRI Testing Guide

## Overview

This guide covers testing procedures for the Concept MRI enhancements including ETS clustering, hierarchy control, window management, and enhanced visualizations.

## Test Structure

### 1. Test Data Generators (`test_data_generators.py`)
Generates realistic mock data for testing all components:
- Model data with activations
- Clustering results (K-Means, DBSCAN, ETS)
- Window configurations
- Path analysis results
- LLM-generated cluster labels
- Hierarchical clustering results

### 2. Component Tests (`test_components.py`)
Tests individual components in isolation:
- **ETS Clustering**: Threshold calculation, clustering, explanations
- **Hierarchy Control**: K calculation for macro/meso/micro levels
- **Window Manager**: Preset calculations, boundary detection
- **Sankey Filtering**: Path filtering by window
- **Cluster Cards**: Standard, ETS, and hierarchical card creation
- **Visualization Modes**: Different display modes for trajectories

### 3. Integration Tests (`test_integration.py`)
Tests component interactions:
- Window config → Sankey filtering
- Hierarchy level → Clustering parameters
- Clustering data → Visualizations
- ETS data flow through pipeline
- Data persistence across stores
- Callback chains
- Error handling

## Running Tests

### Quick Test
```bash
# Run all tests
cd concept_mri
python tests/run_all_tests.py
```

### Individual Test Suites
```bash
# Test data generators only
python -m unittest tests.test_data_generators

# Component tests only
python -m unittest tests.test_components

# Integration tests only
python -m unittest tests.test_integration
```

### Generate Test Data
```bash
# Create test data file for manual testing
python tests/test_data_generators.py
# This creates concept_mri_test_data.json
```

## Manual Testing Checklist

### 1. ETS Clustering
- [ ] Threshold percentile slider adjusts clustering granularity
- [ ] Min threshold prevents numerical issues
- [ ] Batch size setting works for large datasets
- [ ] Explanations toggle shows/hides cluster explanations
- [ ] Similarity matrix computation (when enabled) completes
- [ ] ETS visualization shows threshold distribution

### 2. Hierarchy Control
- [ ] Macro level creates few large clusters (2-10)
- [ ] Meso level creates balanced clusters (3-20)
- [ ] Micro level creates many small clusters (5-50)
- [ ] K slider range updates based on hierarchy level
- [ ] ETS threshold percentile adjusts with hierarchy

### 3. Layer Window Manager
- [ ] Presets apply correctly (GPT-2, thirds, quarters, halves)
- [ ] Manual window creation with custom names/colors
- [ ] Window deletion works
- [ ] Auto-detection mode shows experimental warning
- [ ] Metrics plot displays stability/density scores
- [ ] Window preview visualization updates

### 4. Window-Aware Visualizations
- [ ] Sankey diagram filters paths by selected window
- [ ] Window dropdown updates from Layer Window Manager
- [ ] Hierarchy selector switches between macro/meso/micro
- [ ] Stepped trajectory respects window boundaries
- [ ] Cluster cards show data for selected layer only

### 5. Enhanced Visualizations
- [ ] Stepped trajectory modes: individual/aggregated/heatmap
- [ ] Zoom controls work (zoom in/out/reset)
- [ ] Sample size control affects individual paths display
- [ ] Cluster cards show appropriate content for type
- [ ] Selection/comparison features in cluster cards
- [ ] Export functionality for all visualizations

## Test Scenarios

### Scenario 1: Complete ETS Analysis
1. Load model with 12 layers
2. Configure windows (GPT-2 preset)
3. Select ETS clustering with 10% threshold
4. Set hierarchy to meso level
5. Run analysis
6. View ETS cluster cards with explanations
7. Check threshold distribution in cards

### Scenario 2: Hierarchical Analysis
1. Load model and dataset
2. Run clustering at macro level
3. Switch to meso level and re-run
4. Switch to micro level and re-run
5. Compare cluster counts and distributions
6. View hierarchical cluster cards

### Scenario 3: Window-Based Analysis
1. Load deep network (>20 layers)
2. Use auto-detection for windows
3. Review suggested boundaries
4. Adjust windows manually
5. Run clustering for each window
6. Compare trajectories across windows

### Scenario 4: Edge Cases
1. Single layer network (should disable windowing)
2. Empty dataset (should show appropriate messages)
3. Very large dataset (test performance)
4. Invalid parameters (should handle gracefully)

## Performance Testing

### Metrics to Monitor
- Clustering time for different algorithms
- Visualization rendering time
- Memory usage with large datasets
- UI responsiveness during computation

### Benchmarks
- 100 samples, 10 layers: < 1 second
- 1000 samples, 20 layers: < 10 seconds
- 10000 samples, 50 layers: < 1 minute

## Debugging Tips

### Common Issues
1. **Import Errors**: Check all dependencies are installed
2. **Callback Errors**: Verify store IDs match between components
3. **Visualization Errors**: Check data format matches expected structure
4. **Memory Issues**: Reduce batch size for ETS clustering

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run app in debug mode
app.run_server(debug=True)
```

## Test Report

After running tests, check:
- `test_report.txt` - Human-readable report
- `test_report.json` - Machine-readable results

## Contributing Tests

When adding new features:
1. Add test data generation in `test_data_generators.py`
2. Add component tests in `test_components.py`
3. Add integration tests in `test_integration.py`
4. Update this guide with new test scenarios