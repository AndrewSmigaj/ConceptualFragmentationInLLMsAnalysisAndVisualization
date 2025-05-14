"""
Run cohesion grid experiments for the Concept Fragmentation project.

This script:
1. Runs experiments for all datasets with the cohesion regularization grid
2. Uses multiple seeds for reproducibility
3. Tracks experiment progress and handles failures gracefully
4. Outputs a summary of results
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_fragmentation.config import (
    COHESION_GRID, DATASETS, RANDOM_SEED, RESULTS_DIR
)
from concept_fragmentation.experiments.train import train_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "cohesion_grid.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Base directory for cohesion results
COHESION_DIR = os.path.join(RESULTS_DIR, "cohesion")
os.makedirs(COHESION_DIR, exist_ok=True)

def create_reg_params_id(reg_params: Dict[str, Any]) -> str:
    """
    Create a unique identifier for regularization parameters.
    
    Args:
        reg_params: Dictionary of regularization parameters
        
    Returns:
        String identifier for the parameters
    """
    if reg_params.get("weight", 0) == 0:
        return "baseline"
        
    # Format: w{weight}_t{temperature}_thr{threshold}_L{layers}
    weight = reg_params.get("weight", 0)
    temp = reg_params.get("temperature", 0.07)
    thr = reg_params.get("similarity_threshold", 0)
    
    layers_str = ""
    if "layers" in reg_params and reg_params["layers"]:
        layers_str = "_L" + "-".join([l.replace("layer", "") for l in reg_params["layers"]])
    
    return f"w{weight}_t{temp}_thr{thr}{layers_str}"

def create_experiment_dir(dataset_name: str, reg_params: Dict[str, Any], seed: int) -> str:
    """
    Create and return the experiment directory path.
    
    Args:
        dataset_name: Name of the dataset
        reg_params: Dictionary of regularization parameters
        seed: Random seed
        
    Returns:
        Path to the experiment directory
    """
    reg_id = create_reg_params_id(reg_params)
    dataset_dir = os.path.join(COHESION_DIR, dataset_name)
    experiment_dir = os.path.join(dataset_dir, reg_id, f"seed_{seed}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

def experiment_exists(dataset_name: str, reg_params: Dict[str, Any], seed: int) -> bool:
    """
    Check if an experiment with the given parameters has already been run.
    
    Args:
        dataset_name: Name of the dataset
        reg_params: Dictionary of regularization parameters
        seed: Random seed
        
    Returns:
        True if experiment exists and has completed successfully
    """
    experiment_dir = create_experiment_dir(dataset_name, reg_params, seed)
    
    # Check if experiment directory exists and contains required files
    required_files = ["final_model.pt", "training_history.json"]
    
    if not os.path.exists(experiment_dir):
        return False
    
    return all(os.path.exists(os.path.join(experiment_dir, file)) for file in required_files)

def run_cohesion_experiment(
    dataset_name: str,
    reg_params: Dict[str, Any],
    seed: int,
    force_rerun: bool = False
) -> bool:
    """
    Run a single cohesion experiment with the specified parameters.
    
    Args:
        dataset_name: Name of the dataset
        reg_params: Dictionary of regularization parameters
        seed: Random seed
        force_rerun: Whether to rerun even if the experiment already exists
        
    Returns:
        True if experiment completed successfully
    """
    # Check if experiment already exists
    if experiment_exists(dataset_name, reg_params, seed) and not force_rerun:
        logger.info(f"Experiment for {dataset_name} with seed {seed} and parameters {reg_params} already exists. Skipping.")
        return True
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(dataset_name, reg_params, seed)
    reg_id = create_reg_params_id(reg_params)
    
    # Log experiment start
    logger.info(f"Running experiment for {dataset_name} with seed {seed} and parameters {reg_params}")
    logger.info(f"Results will be saved to: {experiment_dir}")
    
    # Set up experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{reg_id}_seed{seed}_{timestamp}"
    
    try:
        # Determine if regularization should be used
        use_regularization = reg_params.get("weight", 0) > 0
        
        # Run the experiment
        train_model(
            dataset_name=dataset_name,
            use_regularization=use_regularization,
            save_activations=True,
            save_model=True,
            save_fragmentation_metrics=True,
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            random_seed=seed,
            # Pass regularization parameters if using regularization
            reg_params=reg_params if use_regularization else None
        )
        
        logger.info(f"Successfully completed experiment for {dataset_name} with seed {seed} and parameters {reg_params}")
        return True
    
    except Exception as e:
        logger.error(f"Error running experiment for {dataset_name} with seed {seed} and parameters {reg_params}: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create error log in experiment directory
        error_log_path = os.path.join(experiment_dir, "error.log")
        with open(error_log_path, "w") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        return False

def run_cohesion_grid(
    datasets: List[str] = None,
    seeds: List[int] = None,
    params_grid: List[Dict[str, Any]] = None,
    resume: bool = True,
    force_rerun: bool = False
) -> pd.DataFrame:
    """
    Run cohesion experiments for all combinations of datasets, seeds, and regularization parameters.
    
    Args:
        datasets: List of dataset names to run experiments for
        seeds: List of random seeds to use
        params_grid: List of regularization parameter dictionaries
        resume: Whether to resume from previous runs (skip completed experiments)
        force_rerun: Whether to force rerun of all experiments
        
    Returns:
        DataFrame with experiment progress summary
    """
    # Use defaults if not provided
    if datasets is None:
        datasets = list(DATASETS.keys())
    
    if seeds is None:
        seeds = [0, 1, 2]  # Default seeds
    
    if params_grid is None:
        params_grid = COHESION_GRID
    
    # Create a progress DataFrame to track experiment status
    progress_data = []
    
    # Total number of experiments
    total_experiments = len(datasets) * len(seeds) * len(params_grid)
    completed_experiments = 0
    successful_experiments = 0
    
    logger.info(f"Running cohesion grid experiments for {len(datasets)} datasets, "
                f"{len(seeds)} seeds, and {len(params_grid)} parameter sets")
    logger.info(f"Total experiments: {total_experiments}")
    
    # Keep track of start time
    start_time = time.time()
    
    # Run experiments for each dataset, seed, and parameter set
    for dataset_name in datasets:
        for seed in seeds:
            for reg_params in params_grid:
                # Skip baselines if already run in the baseline experiments
                if reg_params.get("weight", 0) == 0:
                    reg_id = "baseline"
                    baseline_dir = os.path.join(RESULTS_DIR, "baselines", dataset_name)
                    existing_baseline_dirs = [d for d in os.listdir(baseline_dir) if f"seed{seed}" in d] if os.path.exists(baseline_dir) else []
                    
                    if existing_baseline_dirs and not force_rerun:
                        logger.info(f"Baseline for {dataset_name} with seed {seed} already exists in baseline directory. Skipping.")
                        
                        # Add to progress data
                        progress_data.append({
                            "dataset": dataset_name,
                            "seed": seed,
                            "reg_id": reg_id,
                            "weight": reg_params.get("weight", 0),
                            "temperature": reg_params.get("temperature", 0.07),
                            "threshold": reg_params.get("similarity_threshold", 0),
                            "layers": reg_params.get("layers", []),
                            "status": "completed",
                            "experiment_dir": os.path.join(baseline_dir, existing_baseline_dirs[0]) if existing_baseline_dirs else None
                        })
                        
                        completed_experiments += 1
                        successful_experiments += 1
                        continue
                
                # Check if experiment already exists
                if resume and experiment_exists(dataset_name, reg_params, seed) and not force_rerun:
                    experiment_dir = create_experiment_dir(dataset_name, reg_params, seed)
                    reg_id = create_reg_params_id(reg_params)
                    
                    logger.info(f"Experiment for {dataset_name} with seed {seed} and parameters {reg_params} already exists. Skipping.")
                    
                    # Add to progress data
                    progress_data.append({
                        "dataset": dataset_name,
                        "seed": seed,
                        "reg_id": reg_id,
                        "weight": reg_params.get("weight", 0),
                        "temperature": reg_params.get("temperature", 0.07),
                        "threshold": reg_params.get("similarity_threshold", 0),
                        "layers": reg_params.get("layers", []),
                        "status": "completed",
                        "experiment_dir": experiment_dir
                    })
                    
                    completed_experiments += 1
                    successful_experiments += 1
                    continue
                
                # Run the experiment
                experiment_dir = create_experiment_dir(dataset_name, reg_params, seed)
                reg_id = create_reg_params_id(reg_params)
                
                success = run_cohesion_experiment(
                    dataset_name=dataset_name,
                    reg_params=reg_params,
                    seed=seed,
                    force_rerun=force_rerun
                )
                
                # Update progress
                completed_experiments += 1
                if success:
                    successful_experiments += 1
                
                # Calculate completion percentage and estimated time remaining
                completion_percentage = (completed_experiments / total_experiments) * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (completed_experiments / total_experiments) if completed_experiments > 0 else 0
                estimated_time_remaining = estimated_total_time - elapsed_time
                
                # Print progress
                logger.info(f"Progress: {completed_experiments}/{total_experiments} "
                           f"({completion_percentage:.2f}%) - "
                           f"ETA: {estimated_time_remaining/60:.2f} minutes")
                
                # Add to progress data
                progress_data.append({
                    "dataset": dataset_name,
                    "seed": seed,
                    "reg_id": reg_id,
                    "weight": reg_params.get("weight", 0),
                    "temperature": reg_params.get("temperature", 0.07),
                    "threshold": reg_params.get("similarity_threshold", 0),
                    "layers": reg_params.get("layers", []),
                    "status": "completed" if success else "failed",
                    "experiment_dir": experiment_dir
                })
                
                # Save progress to CSV
                progress_df = pd.DataFrame(progress_data)
                progress_csv_path = os.path.join(COHESION_DIR, "progress_cohesion.csv")
                progress_df.to_csv(progress_csv_path, index=False)
                logger.info(f"Progress saved to {progress_csv_path}")
    
    # Calculate final statistics
    total_time = time.time() - start_time
    success_rate = (successful_experiments / total_experiments) * 100
    
    logger.info(f"All cohesion grid experiments completed.")
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Successful experiments: {successful_experiments}")
    logger.info(f"Failed experiments: {total_experiments - successful_experiments}")
    logger.info(f"Success rate: {success_rate:.2f}%")
    logger.info(f"Total time: {total_time/60:.2f} minutes")
    
    # Create DataFrame with progress data
    progress_df = pd.DataFrame(progress_data)
    
    # Save final progress to CSV
    progress_csv_path = os.path.join(COHESION_DIR, "cohesion_experiments_summary.csv")
    progress_df.to_csv(progress_csv_path, index=False)
    logger.info(f"Final experiment summary saved to {progress_csv_path}")
    
    return progress_df

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(
        description="Run cohesion regularization grid experiments for the Concept Fragmentation project."
    )
    
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs="+", 
        choices=list(DATASETS.keys()),
        help="Datasets to run experiments for"
    )
    
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+", 
        default=[0, 1, 2],
        help="Random seeds for reproducibility"
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume from previous runs (skip completed experiments)"
    )
    
    parser.add_argument(
        "--force_rerun", 
        action="store_true",
        help="Force rerun of all experiments"
    )
    
    args = parser.parse_args()
    
    # Run cohesion grid experiments
    run_cohesion_grid(
        datasets=args.datasets,
        seeds=args.seeds,
        resume=args.resume,
        force_rerun=args.force_rerun
    )
    
    logger.info("Cohesion grid experiments completed.")

if __name__ == "__main__":
    main() 