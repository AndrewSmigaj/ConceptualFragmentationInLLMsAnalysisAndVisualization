#!/usr/bin/env python3
"""Check dependencies for GPT-2 semantic subtypes experiment."""

import sys

def check_dependencies():
    """Check if all required packages are available."""
    required = {
        'torch': 'PyTorch',
        'transformers': 'Hugging Face Transformers',
        'sklearn': 'scikit-learn',
        'numpy': 'NumPy',
        'pandas': 'pandas (for data manipulation)',
        'matplotlib': 'matplotlib (for visualization)',
        'seaborn': 'seaborn (for visualization)'
    }
    
    missing = []
    available = []
    
    for module, name in required.items():
        try:
            if module == 'sklearn':
                import sklearn
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
            else:
                __import__(module)
            available.append(f"✓ {name}")
        except ImportError:
            missing.append(f"✗ {name} (pip install {module})")
    
    # Check ETS availability
    try:
        sys.path.insert(0, '../../..')
        from concept_fragmentation.metrics.explainable_threshold_similarity import (
            compute_dimension_thresholds,
            compute_similarity_matrix,
            extract_clusters
        )
        available.append("✓ ETS clustering functions")
    except ImportError as e:
        missing.append(f"✗ ETS clustering functions: {e}")
    
    # Check GPT-2 components
    try:
        from experiments.gpt2.shared.gpt2_activation_extractor import SimpleGPT2ActivationExtractor
        available.append("✓ GPT-2 activation extractor")
    except ImportError as e:
        missing.append(f"✗ GPT-2 activation extractor: {e}")
    
    print("Dependency Check for GPT-2 Semantic Subtypes Experiment")
    print("=" * 50)
    print("\nAvailable:")
    for item in available:
        print(f"  {item}")
    
    if missing:
        print("\nMissing:")
        for item in missing:
            print(f"  {item}")
        print("\nPlease install missing dependencies before proceeding.")
        return False
    else:
        print("\n✓ All dependencies are available!")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)