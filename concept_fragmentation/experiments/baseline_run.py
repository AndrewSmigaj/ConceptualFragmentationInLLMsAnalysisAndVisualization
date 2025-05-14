"""
Baseline experiment runner for the Concept Fragmentation project.

This script:
1. Runs baseline models (no regularization) on all datasets
2. Uses 3 different random seeds (0, 1, 2) for reproducibility
3. Stores model checkpoints, activations, and metrics in the results directory
4. Generates a summary CSV with final metrics for all runs
"""

import os
import logging
import argparse
import json
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

from concept_fragmentation.config import RESULTS_DIR
from concept_fragmentation.experiments.train import train_model
from concept_fragmentation.utils.helpers import set_global_seed

# Create main results directory structure
BASELINE_RESULTS_DIR = os.path.join(RESULTS_DIR, "baselines")
os.makedirs(BASELINE_RESULTS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASELINE_RESULTS_DIR, "baseline_experiments.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_experiment_name(dataset, seed):
    """
    Create a standardized experiment name for consistent directory structure.
    
    Args:
        dataset: Dataset name
        seed: Random seed
        
    Returns:
        Experiment name string
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"{dataset}_baseline_seed{seed}_{timestamp}"


def generate_baseline_summary(datasets, seeds):
    """
    Generate a summary CSV with final metrics for all completed baseline runs.
    
    Args:
        datasets: List of datasets used in experiments
        seeds: List of seeds used in experiments
        
    Returns:
        Path to the generated summary CSV file
    """
    results = []
    
    for dataset in datasets:
        for seed in seeds:
            # Define experiment directory based on our naming convention
            dataset_dir = os.path.join(BASELINE_RESULTS_DIR, dataset)
            exp_dirs = [d for d in os.listdir(dataset_dir) if f"seed{seed}" in d]
            
            if not exp_dirs:
                logger.warning(f"No experiment directory found for {dataset} with seed {seed}")
                continue
                
            # Get the most recent experiment for this dataset/seed combination
            exp_dir = sorted(exp_dirs)[-1]
            full_exp_dir = os.path.join(dataset_dir, exp_dir)
            
            # Check for history file
            history_file = os.path.join(full_exp_dir, "training_history.json")
            if not os.path.exists(history_file):
                logger.warning(f"No training history found for {dataset} with seed {seed}")
                continue
            
            # Load training history
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Get final metrics
            final_metrics = {
                "dataset": dataset,
                "seed": seed,
                "test_accuracy": history["test_accuracy"][-1],
                "train_accuracy": history["train_accuracy"][-1],
                "test_loss": history["test_loss"][-1],
                "entropy_fragmentation": history["entropy_fragmentation"][-1],
                "angle_fragmentation": history["angle_fragmentation"][-1],
                "experiment_dir": full_exp_dir
            }
            
            results.append(final_metrics)
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Calculate means and standard deviations per dataset
        summary_stats = df.groupby("dataset").agg({
            "test_accuracy": ["mean", "std"],
            "train_accuracy": ["mean", "std"],
            "entropy_fragmentation": ["mean", "std"],
            "angle_fragmentation": ["mean", "std"]
        })
        
        # Save summary files
        summary_csv = os.path.join(BASELINE_RESULTS_DIR, "baseline_summary.csv")
        stats_csv = os.path.join(BASELINE_RESULTS_DIR, "baseline_stats.csv")
        
        df.to_csv(summary_csv, index=False)
        summary_stats.to_csv(stats_csv)
        
        logger.info(f"Baseline summary saved to {summary_csv}")
        logger.info(f"Baseline statistics saved to {stats_csv}")
        
        return summary_csv
    else:
        logger.warning("No results found to generate summary")
        return None


def run_baseline_experiments(
    datasets=None,
    seeds=None,
    force_rerun=False,
    device=None
):
    """
    Run baseline experiments for specified datasets and seeds.
    
    Args:
        datasets: List of datasets to use. If None, all datasets will be used.
        seeds: List of random seeds to use. If None, seeds 0, 1, 2 will be used.
        force_rerun: Whether to rerun experiments even if results already exist
        device: Device to run experiments on ("cuda" or "cpu")
    """
    if datasets is None:
        datasets = ["titanic", "adult", "heart", "fashion_mnist"]
    
    if seeds is None:
        seeds = [0, 1, 2]
    
    # Create dataset directories within baseline results
    for dataset in datasets:
        os.makedirs(os.path.join(BASELINE_RESULTS_DIR, dataset), exist_ok=True)
    
    logger.info(f"Running baseline experiments for datasets: {datasets}")
    logger.info(f"Using seeds: {seeds}")
    
    # Track successful runs
    successful_runs = []
    failed_runs = []
    
    # Run experiments for each dataset and seed
    for dataset in datasets:
        for seed in seeds:
            logger.info(f"Running experiment for dataset: {dataset} with seed: {seed}")
            
            # Create experiment name and directory
            experiment_name = create_experiment_name(dataset, seed)
            experiment_dir = os.path.join(BASELINE_RESULTS_DIR, dataset, experiment_name)
            
            # Check if this experiment has already been run
            if os.path.exists(experiment_dir) and not force_rerun:
                if os.path.exists(os.path.join(experiment_dir, "best_model.pt")):
                    logger.info(f"Experiment for {dataset} with seed {seed} already exists. Skipping.")
                    successful_runs.append((dataset, seed))
                    continue
            
            # Make sure experiment directory exists
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Set global seed for reproducibility
            set_global_seed(seed)
            
            try:
                # Set deterministic execution for reproducibility
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Run the experiment
                result = train_model(
                    dataset_name=dataset,
                    use_regularization=False,  # No regularization for baselines
                    save_activations=True,
                    save_model=True,
                    save_fragmentation_metrics=True,
                    experiment_name=experiment_name,
                    random_seed=seed,
                    experiment_dir=experiment_dir,
                    device=device
                )
                
                logger.info(f"Successfully completed experiment for dataset: {dataset} with seed: {seed}")
                successful_runs.append((dataset, seed))
                
            except Exception as e:
                logger.error(f"Failed to run experiment for dataset: {dataset} with seed: {seed}")
                logger.error(f"Error: {str(e)}")
                failed_runs.append((dataset, seed, str(e)))
    
    # Generate summary of all baseline runs
    if successful_runs:
        logger.info(f"Generating baseline summary...")
        summary_path = generate_baseline_summary(datasets, seeds)
        
        if summary_path:
            logger.info(f"Baseline summary available at: {summary_path}")
    
    # Print overall status
    logger.info(f"All baseline experiments completed.")
    logger.info(f"Successful runs: {len(successful_runs)}/{len(datasets) * len(seeds)}")
    
    if failed_runs:
        logger.warning(f"Failed runs: {len(failed_runs)}")
        for dataset, seed, error in failed_runs:
            logger.warning(f"  - {dataset}, seed {seed}: {error}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run baseline experiments for concept fragmentation analysis')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        choices=['titanic', 'adult', 'heart', 'fashion_mnist'],
                        help='Datasets to use (default: all)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds for reproducibility (default: 0, 1, 2)')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Force rerun of experiments even if results already exist')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to run experiments on (default: use available)')
    
    args = parser.parse_args()
    
    # Run baseline experiments with specified arguments
    run_baseline_experiments(
        datasets=args.datasets,
        seeds=args.seeds,
        force_rerun=args.force_rerun,
        device=args.device
    ) 