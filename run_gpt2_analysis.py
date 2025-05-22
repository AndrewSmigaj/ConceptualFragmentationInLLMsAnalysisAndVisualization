#!/usr/bin/env python
"""
Runner script for GPT-2 Archetypal Path Analysis.

This script provides a clean way to run the GPT-2 APA analysis without path hacks.
It automatically sets up the correct Python path and forwards arguments.
"""

import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the main function from the gpt2_cluster_paths module
from concept_fragmentation.analysis.gpt2_cluster_paths import main as gpt2_cluster_paths_main

if __name__ == "__main__":
    print("Running GPT-2 Archetypal Path Analysis...")
    try:
        gpt2_cluster_paths_main()
        print("GPT-2 analysis completed successfully!")
    except Exception as e:
        print(f"Error running GPT-2 analysis: {e}")
        sys.exit(1)