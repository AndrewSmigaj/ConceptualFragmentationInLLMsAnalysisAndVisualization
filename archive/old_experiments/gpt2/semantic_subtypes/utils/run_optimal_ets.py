#!/usr/bin/env python3
"""
Run the semantic subtypes experiment with optimal ETS threshold from elbow analysis.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from experiments.gpt2.semantic_subtypes.gpt2_semantic_subtypes_experiment import SemanticSubtypesExperiment

def main():
    """Run experiment with optimal ETS threshold."""
    
    # Optimal threshold from elbow analysis
    OPTIMAL_THRESHOLD = 0.992
    
    # Create new output directory for optimal results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"semantic_subtypes_optimal_ets_{timestamp}"
    
    print(f"Running semantic subtypes experiment with optimal ETS threshold: {OPTIMAL_THRESHOLD}")
    print(f"Output directory: {output_dir}")
    
    # Configuration for the experiment
    config = {
        'output_dir': output_dir,
        'curated_words_file': 'data/gpt2_semantic_subtypes_curated.json',
        'clustering_methods': ['kmeans', 'ets'],
        'ets_threshold_percentile': OPTIMAL_THRESHOLD,
        'k_range': (2, 15),
        'random_seed': 42
    }
    
    # Run the experiment
    experiment = SemanticSubtypesExperiment(config)
    results = experiment.run_complete_experiment()
    
    print(f"\nExperiment complete!")
    print(f"Results saved to: {output_dir}")
    
    # Save a summary of the configuration
    config_summary = {
        'experiment_type': 'semantic_subtypes_optimal_ets',
        'ets_threshold': OPTIMAL_THRESHOLD,
        'timestamp': timestamp,
        'config': config
    }
    
    with open(Path(output_dir) / 'experiment_config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    return output_dir

if __name__ == "__main__":
    output_dir = main()