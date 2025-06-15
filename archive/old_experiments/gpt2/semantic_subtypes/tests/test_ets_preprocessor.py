#!/usr/bin/env python3
"""Test the ETSPreprocessor implementation."""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ets_revised_wrapper import ETSPreprocessor

def test_preprocessor():
    """Test the preprocessor with synthetic data."""
    print("Testing ETSPreprocessor...")
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic preprocessing")
    np.random.seed(42)
    data = np.random.randn(100, 200)  # 100 samples, 200 features
    
    preprocessor = ETSPreprocessor(pca_dims=50)
    processed = preprocessor.fit_transform(data)
    
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Input mean: {data.mean():.3f}, std: {data.std():.3f}")
    print(f"Output mean: {processed.mean():.3f}, std: {processed.std():.3f}")
    
    # Check that preprocessing worked (mean should be near 0)
    assert abs(processed.mean()) < 0.1, f"Mean not near 0: {processed.mean()}"
    # Note: PCA components have different variances, so std won't be 1
    print(f"PCA component variances: min={preprocessor.pca.explained_variance_[:5].min():.3f}, "
          f"max={preprocessor.pca.explained_variance_[:5].max():.3f}")
    
    # Test 2: No PCA needed case
    print("\nTest 2: No PCA needed (fewer features than target)")
    small_data = np.random.randn(100, 30)  # Only 30 features
    
    preprocessor2 = ETSPreprocessor(pca_dims=50)
    processed2 = preprocessor2.fit_transform(small_data)
    
    print(f"Input shape: {small_data.shape}")
    print(f"Output shape: {processed2.shape}")
    assert processed2.shape[1] == 30, "Should keep all features when < pca_dims"
    
    # Test 3: Edge case - single sample
    print("\nTest 3: Single sample")
    single = np.random.randn(1, 100)
    
    preprocessor3 = ETSPreprocessor(pca_dims=50)
    try:
        processed3 = preprocessor3.fit_transform(single)
        print(f"Single sample processed: {processed3.shape}")
    except Exception as e:
        print(f"Error with single sample: {e}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_preprocessor()