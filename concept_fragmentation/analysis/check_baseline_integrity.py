"""
Check the integrity of baseline experiment results.

This script scans through the baseline results directory and verifies that:
1. All expected experiment directories exist
2. Each experiment directory contains the required output files
"""

import os
import sys
import logging
import argparse
import pandas as pd
import json
from typing import Dict, List, Tuple, Set

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_fragmentation.config import DATASETS, RESULTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_baseline_integrity(seeds: List[int] = [0, 1, 2]) -> Tuple[bool, List[str]]:
    """
    Check the integrity of baseline experiments.
    
    Args:
        seeds: List of seeds to check
        
    Returns:
        Tuple containing:
            - Boolean indicating if all experiments are complete
            - List of missing experiment strings
    """
    baseline_dir = os.path.join(RESULTS_DIR, "baselines")
    if not os.path.exists(baseline_dir):
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return False, [f"Baseline directory not found: {baseline_dir}"]
    
    # Check if baseline_summary.csv exists
    summary_csv_path = os.path.join(baseline_dir, "baseline_summary.csv")
    if not os.path.exists(summary_csv_path):
        logger.warning(f"Baseline summary CSV not found: {summary_csv_path}")
    else:
        logger.info(f"Baseline summary CSV found: {summary_csv_path}")
        # Read the summary to check if all experiments are there
        df = pd.read_csv(summary_csv_path)
        logger.info(f"Summary contains {len(df)} experiment entries")
    
    # Define required files for each experiment
    required_files = [
        "final_model.pt",
        "training_history.json",
        "layer_activations.pkl"
    ]
    
    # Track missing experiments
    missing_experiments = []
    missing_files = []
    
    # Expected number of experiments
    expected_count = len(DATASETS) * len(seeds)
    found_count = 0
    
    # Check each dataset directory
    for dataset_name in DATASETS.keys():
        dataset_dir = os.path.join(baseline_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            for seed in seeds:
                missing_experiments.append(f"{dataset_name} (seed {seed})")
            continue
        
        # Find all experiment directories for this dataset
        experiment_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        # Check if all seeds are present
        found_seeds = set()
        for exp_dir in experiment_dirs:
            for seed in seeds:
                if f"seed{seed}" in exp_dir:
                    found_seeds.add(seed)
                    
                    # Check if all required files exist
                    full_exp_dir = os.path.join(dataset_dir, exp_dir)
                    for req_file in required_files:
                        file_path = os.path.join(full_exp_dir, req_file)
                        if not os.path.exists(file_path):
                            missing_files.append(f"{dataset_name} (seed {seed}): {req_file}")
                            logger.warning(f"Missing file: {file_path}")
                    
                    found_count += 1
        
        # Report missing seeds
        for seed in seeds:
            if seed not in found_seeds:
                missing_experiments.append(f"{dataset_name} (seed {seed})")
                logger.warning(f"Missing experiment: {dataset_name} (seed {seed})")
    
    # Final report
    logger.info(f"Found {found_count}/{expected_count} expected experiments")
    
    if missing_experiments:
        logger.warning(f"Missing {len(missing_experiments)} experiments")
        for missing in missing_experiments:
            logger.warning(f"  - {missing}")
    else:
        logger.info("All expected experiments are present")
    
    if missing_files:
        logger.warning(f"Missing {len(missing_files)} required files")
        for missing in missing_files:
            logger.warning(f"  - {missing}")
    else:
        logger.info("All required files are present in existing experiments")
    
    # Validation of experiment outputs
    issues = []
    
    # Check training history files for completeness
    for dataset_name in DATASETS.keys():
        dataset_dir = os.path.join(baseline_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            continue
            
        experiment_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        for exp_dir in experiment_dirs:
            for seed in seeds:
                if f"seed{seed}" in exp_dir:
                    history_path = os.path.join(dataset_dir, exp_dir, "training_history.json")
                    if os.path.exists(history_path):
                        try:
                            with open(history_path, 'r') as f:
                                history = json.load(f)
                            
                            # Check that history has the expected keys
                            expected_keys = [
                                "train_loss", "train_accuracy", "test_loss", 
                                "test_accuracy", "entropy_fragmentation", "angle_fragmentation"
                            ]
                            
                            missing_keys = [key for key in expected_keys if key not in history]
                            if missing_keys:
                                issues.append(f"{dataset_name} (seed {seed}): Missing keys in training_history.json: {missing_keys}")
                            
                        except Exception as e:
                            issues.append(f"{dataset_name} (seed {seed}): Error reading training_history.json: {str(e)}")
    
    if issues:
        logger.warning(f"Found {len(issues)} issues with experiment outputs")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("All experiment outputs appear valid")
    
    # Write missing experiments to file
    if missing_experiments or missing_files or issues:
        output_path = os.path.join(RESULTS_DIR, "missing_baseline_runs.txt")
        with open(output_path, "w") as f:
            if missing_experiments:
                f.write("=== MISSING EXPERIMENTS ===\n")
                for missing in missing_experiments:
                    f.write(f"{missing}\n")
            
            if missing_files:
                f.write("\n=== MISSING FILES ===\n")
                for missing in missing_files:
                    f.write(f"{missing}\n")
            
            if issues:
                f.write("\n=== ISSUES ===\n")
                for issue in issues:
                    f.write(f"{issue}\n")
        
        logger.info(f"Wrote list of missing experiments and issues to {output_path}")
    
    return len(missing_experiments) == 0 and len(missing_files) == 0 and len(issues) == 0, missing_experiments + missing_files + issues

def main():
    """Main function to parse arguments and check baseline integrity."""
    parser = argparse.ArgumentParser(
        description="Check the integrity of baseline experiment results."
    )
    
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+", 
        default=[0, 1, 2],
        help="Random seeds to check"
    )
    
    args = parser.parse_args()
    
    # Check baseline integrity
    all_complete, missing = check_baseline_integrity(args.seeds)
    
    if all_complete:
        logger.info("All baseline experiments are complete and valid")
        sys.exit(0)
    else:
        logger.warning("Some baseline experiments are missing or have issues")
        sys.exit(1)

if __name__ == "__main__":
    main() 