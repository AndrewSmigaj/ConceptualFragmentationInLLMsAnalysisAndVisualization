# Test Suite for Concept Fragmentation Project

This directory contains test cases for the Concept Fragmentation project. The tests ensure that each component of the project works correctly and that they integrate properly.

## Test Organization

The test suite is organized into the following modules:

- `test_metrics.py`: Tests for the fragmentation metrics (cluster entropy and subspace angle)
- `test_hooks.py`: Tests for the activation hooks module
- `test_e2e.py`: End-to-end tests that validate the entire workflow

## Running the Tests

### Requirements

Make sure you have all the dependencies installed:

```bash
pip install -r ../requirements.txt
```

### Running All Tests

To run all tests:

```bash
# From the project root directory
python -m unittest discover concept_fragmentation/tests

# Or using pytest
pytest concept_fragmentation/tests
```

### Running Specific Test Files

To run tests for a specific module:

```bash
# Using unittest
python -m unittest concept_fragmentation/tests/test_metrics.py
python -m unittest concept_fragmentation/tests/test_hooks.py
python -m unittest concept_fragmentation/tests/test_e2e.py

# Using pytest
pytest concept_fragmentation/tests/test_metrics.py
```

### Running Specific Test Cases

To run a specific test case:

```bash
# Example: Run only the test_separated_clusters test in test_metrics.py
python -m unittest concept_fragmentation.tests.test_metrics.TestClusterEntropy.test_separated_clusters
```

## Test Coverage

The tests cover:

1. **Unit Tests**:
   - Correct calculation of cluster entropy and subspace angle metrics
   - Proper functioning of activation hooks
   - Edge cases handling

2. **Integration Tests**:
   - Capture of activations during model forward pass
   - Calculation of metrics on captured activations

3. **End-to-End Tests**:
   - Training a model on a synthetic dataset
   - Capturing activations and calculating metrics
   - Validating expected fragmentation patterns

## Adding New Tests

When adding new functionality to the project, please add corresponding tests:

1. For new metrics, add test cases to `test_metrics.py`
2. For hook modifications, add test cases to `test_hooks.py`
3. For changes that affect the full workflow, update `test_e2e.py`

Follow the existing test structure and naming conventions to maintain consistency.

## Continuous Integration

These tests are automatically run as part of the CI pipeline to ensure code quality and prevent regressions. 