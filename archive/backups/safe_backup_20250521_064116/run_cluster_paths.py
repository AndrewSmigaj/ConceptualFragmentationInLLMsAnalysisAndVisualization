"""
Helper script to run the cluster_paths.py with proper Python path setup.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from concept_fragmentation.analysis.cluster_paths_clean import main as cluster_paths_clean_main

if __name__ == "__main__":
    print("Running cluster path analysis using the clean wrapper...")
    try:
        cluster_paths_clean_main()
        print("Cluster paths script (clean wrapper) completed successfully!")
    except Exception as e:
        print(f"Error running cluster paths script (clean wrapper): {e}")
        sys.exit(1)