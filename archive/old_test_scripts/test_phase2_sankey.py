#!/usr/bin/env python3
"""Test Phase 2 Sankey implementation."""

import sys
import json
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def test_sankey_import():
    """Test that SankeyGenerator can be imported."""
    print("Testing Sankey import...")
    
    try:
        from concept_fragmentation.visualization.sankey import SankeyGenerator
        from concept_fragmentation.visualization.configs import SankeyConfig
        print("[OK] SankeyGenerator imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import SankeyGenerator: {e}")
        return False


def test_sankey_with_real_data():
    """Test SankeyGenerator with actual k=10 data."""
    print("\nTesting SankeyGenerator with real data...")
    
    try:
        from concept_fragmentation.visualization.sankey import SankeyGenerator
        from concept_fragmentation.visualization.configs import SankeyConfig
        
        # Load real data
        data_dir = Path("experiments/gpt2/all_tokens")
        
        # Load windowed analysis
        windowed_path = data_dir / "k10_analysis_results" / "windowed_analysis_k10.json"
        if not windowed_path.exists():
            print(f"[SKIP] Test data not found at {windowed_path}")
            return True
            
        with open(windowed_path, 'r') as f:
            windowed_data = json.load(f)
            
        # Load labels
        labels_path = data_dir / "llm_labels_k10" / "cluster_labels_k10.json"
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                label_data = json.load(f)
                labels = label_data['labels']
        else:
            labels = {}
            
        # Load purity data
        purity_path = data_dir / "llm_labels_k10" / "semantic_purity_k10.json"
        if purity_path.exists():
            with open(purity_path, 'r') as f:
                purity_data = json.load(f)
        else:
            purity_data = {}
            
        # Create Sankey data structure
        sankey_data = {
            'windowed_analysis': windowed_data,
            'labels': labels,
            'purity_data': purity_data
        }
        
        # Test with different configurations
        configs = [
            SankeyConfig(),  # Default
            SankeyConfig(top_n_paths=15, show_purity=False),
            SankeyConfig(legend_position='right', colored_paths=False),
        ]
        
        for i, config in enumerate(configs):
            print(f"  Testing configuration {i+1}...")
            generator = SankeyGenerator(config)
            
            # Test each window
            for window in ['early', 'middle', 'late']:
                try:
                    fig = generator.create_figure(sankey_data, window=window)
                    assert fig is not None
                    print(f"    [OK] Created {window} window Sankey")
                except Exception as e:
                    print(f"    [FAIL] Failed to create {window} Sankey: {e}")
                    return False
                    
        # Test path summary
        summary = generator.create_path_summary(sankey_data)
        assert len(summary) > 0
        print("  [OK] Created path summary")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sankey_configurations():
    """Test SankeyConfig options."""
    print("\nTesting SankeyConfig...")
    
    try:
        from concept_fragmentation.visualization.configs import SankeyConfig
        
        # Test default config
        config = SankeyConfig()
        assert config.top_n_paths == 25
        assert config.show_purity == True
        assert len(config.color_palette) >= 20
        print("[OK] Default configuration correct")
        
        # Test custom config
        config = SankeyConfig(
            top_n_paths=30,
            legend_position='right',
            width=1800,
            height=900
        )
        assert config.top_n_paths == 30
        assert config.legend_position == 'right'
        assert config.width == 1800
        print("[OK] Custom configuration working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False


def test_sankey_methods():
    """Test individual SankeyGenerator methods."""
    print("\nTesting SankeyGenerator methods...")
    
    try:
        from concept_fragmentation.visualization.sankey import SankeyGenerator, PathInfo
        
        generator = SankeyGenerator()
        
        # Test path description generation
        labels = ["Function Words (Grammatical)", "Content Words (Temporal)", "Content Words (Temporal)"]
        desc = generator._generate_path_description(labels)
        assert "Function Words" in desc
        assert "Content Words" in desc
        print("[OK] Path description generation working")
        
        # Test visible cluster detection
        paths = [
            {'path': [0, 1, 2], 'frequency': 100},
            {'path': [1, 2, 3], 'frequency': 50},
            {'path': [0, 2, 3], 'frequency': 25}
        ]
        clusters = generator._find_visible_clusters(paths, [0, 1, 2])
        assert 0 in clusters[0]
        assert 1 in clusters[0]
        assert len(clusters[2]) == 2  # clusters 2 and 3
        print("[OK] Visible cluster detection working")
        
        # Test color assignment
        colors = generator._assign_path_colors(30)
        assert len(colors) == 30
        assert all(isinstance(c, str) for c in colors.values())
        print("[OK] Color assignment working")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Method tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_migration_example():
    """Create example showing how to migrate from old to new implementation."""
    print("\nCreating migration example...")
    
    example = '''# Migration Example: From Old to New SankeyGenerator

## Old Implementation (generate_colored_sankeys_k10.py):
```python
from generate_colored_sankeys_k10 import ColoredSankeyGenerator

generator = ColoredSankeyGenerator(base_dir, k=10)
generator.generate_all_colored_sankeys()
```

## New Implementation:
```python
from concept_fragmentation.visualization import SankeyGenerator, SankeyConfig

# Configure options
config = SankeyConfig(
    top_n_paths=25,
    colored_paths=True,
    legend_position='left',
    last_layer_labels_position='right'
)

# Create generator
generator = SankeyGenerator(config)

# Load your data
data = {
    'windowed_analysis': windowed_analysis,
    'labels': semantic_labels,
    'purity_data': purity_scores
}

# Generate for each window
for window in ['early', 'middle', 'late']:
    fig = generator.create_figure(data, window=window)
    generator.save_figure(fig, f"sankey_{window}_k10.html")
```

## Key Differences:
1. Single configurable class instead of multiple implementations
2. Explicit data structure instead of file paths
3. More control over individual options
4. Better error handling and validation
'''
    
    # Save example
    example_path = Path("MIGRATION_EXAMPLE_SANKEY.md")
    with open(example_path, 'w') as f:
        f.write(example)
    
    print(f"[OK] Migration example saved to {example_path}")
    return True


def main():
    """Run all Phase 2 Sankey tests."""
    print("="*60)
    print("PHASE 2 SANKEY TEST")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    if not test_sankey_import():
        all_passed = False
        
    if not test_sankey_configurations():
        all_passed = False
        
    if not test_sankey_methods():
        all_passed = False
        
    if not test_sankey_with_real_data():
        all_passed = False
        
    if not create_migration_example():
        all_passed = False
        
    print("\n" + "="*60)
    if all_passed:
        print("[OK] ALL SANKEY TESTS PASSED")
    else:
        print("[FAIL] SOME SANKEY TESTS FAILED")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)