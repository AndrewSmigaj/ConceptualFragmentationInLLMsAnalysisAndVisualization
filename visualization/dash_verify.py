"""
Verification script for the dashboard app after cleanup.

This script attempts to import and initialize the dashboard application
without actually starting the server, just to verify that all core
components are working properly after cleanup operations.
"""

import os
import sys
import traceback

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    # Try to import the dashboard
    print("Testing dashboard application imports...")
    from visualization.dash_app import app
    
    # Try to import key components
    print("Testing key component imports...")
    from visualization.data_interface import load_stats, get_best_config
    from visualization.reducers import Embedder, embed_layer_activations
    from visualization.traj_plot import build_single_scene_figure
    from visualization.llm_tab import create_llm_tab
    
    # Perform minimal initialization tests
    print("Testing minimal app initialization...")
    # Just check if app has expected basic properties
    assert hasattr(app, 'layout'), "Dashboard app missing layout"
    assert hasattr(app, 'run_server'), "Dashboard app missing run_server method"
    
    # Print success message
    print("SUCCESS: Dashboard application verified successfully.")
    print("All core components are present and importable.")
    
    # Exit with success
    sys.exit(0)
except Exception as e:
    # Print error message
    print("ERROR: Failed to verify dashboard application.")
    print(f"Exception: {str(e)}")
    print("\nStacktrace:")
    traceback.print_exc()
    
    # Exit with error
    sys.exit(1)