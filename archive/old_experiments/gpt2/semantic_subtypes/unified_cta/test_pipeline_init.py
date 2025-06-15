#!/usr/bin/env python3
"""
Test script to verify pipeline initialization works after fixing parameters.
"""

import sys
from pathlib import Path

# Add unified_cta directory FIRST to ensure we get the right config.py
unified_cta_dir = Path(__file__).parent
sys.path.insert(0, str(unified_cta_dir))

# Add parent directories for other imports
parent_dir = unified_cta_dir.parent
grandparent_dir = parent_dir.parent
great_grandparent_dir = grandparent_dir.parent
project_root = great_grandparent_dir.parent

for directory in [parent_dir, grandparent_dir, great_grandparent_dir, project_root]:
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

print("Testing unified CTA pipeline initialization...")
print("=" * 60)

try:
    # Test imports
    print("1. Testing imports...")
    from run_unified_pipeline import UnifiedCTAPipeline
    from config import create_config_for_experiment
    print("   [OK] Imports successful")
    
    # Create test config
    print("\n2. Creating test configuration...")
    config = create_config_for_experiment('quick_test')
    print(f"   [OK] Config created: {config.layers_to_process} layers")
    
    # Try to initialize pipeline
    print("\n3. Initializing pipeline...")
    pipeline = UnifiedCTAPipeline(config)
    print("   [OK] Pipeline initialized successfully!")
    
    # Check components
    print("\n4. Checking pipeline components:")
    print(f"   - Preprocessor: {pipeline.preprocessor}")
    print(f"   - Clusterer: {pipeline.clusterer}")
    print(f"   - Quality checker: {pipeline.quality_checker}")
    if hasattr(pipeline, 'path_analyzer'):
        print(f"   - Path analyzer: {pipeline.path_analyzer}")
    if hasattr(pipeline, 'interpreter'):
        print(f"   - Interpreter: {pipeline.interpreter}")
    if hasattr(pipeline, 'visualizer'):
        print(f"   - Visualizer: {pipeline.visualizer}")
    
    print(f"\n   Output directory: {pipeline.results_manager.run_dir}")
    
    print("\nSUCCESS: All pipeline components initialized correctly!")
    
except Exception as e:
    print(f"\nERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nPipeline initialization test completed successfully!")