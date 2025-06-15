import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from diagnostics.quality_checks import QualityDiagnostics
import numpy as np

# Test data
data = np.random.randn(100, 10)
labels = np.array([0]*50 + [1]*50)

checker = QualityDiagnostics()

# Test with string layer
try:
    result = checker.check_clustering_quality(data, labels, "layer_0")
    print("ERROR: Should have failed with string layer")
except Exception as e:
    print(f"Expected error with string: {e}")

# Test with int layer  
try:
    result = checker.check_clustering_quality(data, labels, 0)
    print(f"Success with int: n_clusters={result['n_clusters']}")
except Exception as e:
    print(f"Unexpected error with int: {e}")