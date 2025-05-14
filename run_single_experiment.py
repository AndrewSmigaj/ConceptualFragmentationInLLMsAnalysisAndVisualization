"""
Single experiment runner for Concept Fragmentation project.
"""

import sys
import os
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import the baseline experiment runner
from concept_fragmentation.experiments.baseline_run import run_baseline_experiments

if __name__ == "__main__":
    # Run a single experiment as a test
    dataset = "titanic"
    seed = 0
    
    print(f"Running baseline experiment for {dataset} with seed {seed}")
    
    run_baseline_experiments(
        datasets=[dataset],
        seeds=[seed],
        force_rerun=True,
        device="cpu"
    )
    
    print("Experiment completed.") 