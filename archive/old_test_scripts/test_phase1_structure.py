#!/usr/bin/env python3
"""Test Phase 1 structure implementation."""

import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    # Test main package
    try:
        import concept_fragmentation
        print(f"[OK] concept_fragmentation version {concept_fragmentation.__version__}")
    except ImportError as e:
        print(f"[FAIL] Failed to import concept_fragmentation: {e}")
        return False
        
    # Test clustering module
    try:
        from concept_fragmentation.clustering import BaseClusterer, ClustererError
        from concept_fragmentation.clustering.paths import PathExtractor
        print("[OK] Clustering module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import clustering module: {e}")
        return False
        
    # Test labeling module  
    try:
        from concept_fragmentation.labeling import BaseLabeler, LabelerError
        print("[OK] Labeling module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import labeling module: {e}")
        return False
        
    # Test visualization module
    try:
        from concept_fragmentation.visualization import BaseVisualizer, SankeyConfig
        print("[OK] Visualization module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import visualization module: {e}")
        return False
        
    # Test experiments module
    try:
        from concept_fragmentation.experiments import BaseExperiment, ExperimentConfig
        print("[OK] Experiments module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import experiments module: {e}")
        return False
        
    # Test persistence module
    try:
        from concept_fragmentation.persistence import ExperimentState
        print("[OK] Persistence module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import persistence module: {e}")
        return False
        
    # Test utils module
    try:
        from concept_fragmentation.utils import setup_logging, validate_path
        print("[OK] Utils module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import utils module: {e}")
        return False
        
    return True


def test_configurations():
    """Test configuration classes."""
    print("\nTesting configurations...")
    
    from concept_fragmentation.visualization.configs import SankeyConfig, TrajectoryConfig
    from concept_fragmentation.experiments.config import ExperimentConfig
    
    # Test SankeyConfig
    sankey_config = SankeyConfig(top_n_paths=30, show_purity=False)
    assert sankey_config.top_n_paths == 30
    assert sankey_config.show_purity == False
    assert len(sankey_config.color_palette) > 20
    print("[OK] SankeyConfig working correctly")
    
    # Test TrajectoryConfig
    traj_config = TrajectoryConfig(dimensions=2, reduction_method='tsne')
    assert traj_config.dimensions == 2
    assert traj_config.reduction_method == 'tsne'
    print("[OK] TrajectoryConfig working correctly")
    
    # Test ExperimentConfig
    exp_config = ExperimentConfig(
        name="test_experiment",
        k_values=[5, 10],
        layers=[0, 1, 2, 3]
    )
    assert exp_config.name == "test_experiment"
    assert exp_config.k_values == [5, 10]
    print("[OK] ExperimentConfig working correctly")
    
    return True


def test_path_extraction():
    """Test path extraction functionality."""
    print("\nTesting path extraction...")
    
    from concept_fragmentation.clustering.paths import PathExtractor
    import numpy as np
    
    # Create mock data
    cluster_labels = {
        '0': np.array([0, 1, 2, 0, 1]),
        '1': np.array([1, 2, 0, 1, 2]),
        '2': np.array([2, 0, 1, 2, 0])
    }
    
    tokens = [
        {'token_str': 'the', 'token_id': 0},
        {'token_str': 'and', 'token_id': 1},
        {'token_str': 'of', 'token_id': 2},
        {'token_str': 'to', 'token_id': 3},
        {'token_str': 'in', 'token_id': 4}
    ]
    
    # Test extraction
    extractor = PathExtractor()
    paths = extractor.extract_paths(cluster_labels, tokens, layers=[0, 1, 2])
    
    assert len(paths) == 5  # All tokens should have complete paths
    assert paths[0]['path'] == [0, 1, 2]
    assert paths[0]['token'] == 'the'
    print("[OK] Path extraction working correctly")
    
    # Test archetypal paths
    archetypal = extractor.find_archetypal_paths(top_n=3)
    assert len(archetypal) <= 3
    assert all('frequency' in p for p in archetypal)
    print("[OK] Archetypal path finding working correctly")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("PHASE 1 STRUCTURE TEST")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
        
    if not test_configurations():
        all_passed = False
        
    if not test_path_extraction():
        all_passed = False
        
    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)