"""
Aggregate results from baseline and cohesion experiments.

This script:
1. Scans through experiment directories to find results
2. Extracts key metrics from training_history.json files
3. Computes statistics across seeds
4. Outputs summary CSV files for analysis
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concept_fragmentation.config import DATASETS, RESULTS_DIR, COHESION_GRID
from concept_fragmentation.experiments.train import convert_tensors_to_python

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_metrics_from_history(history_path: str) -> Dict[str, Any]:
    """
    Extract key metrics from a training history JSON file.
    
    Args:
        history_path: Path to training_history.json file
        
    Returns:
        Dictionary of extracted metrics
    """
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Extract key metrics
        metrics = {}
        
        # Final values
        if "test_accuracy" in history and len(history["test_accuracy"]) > 0:
            metrics["final_test_accuracy"] = history["test_accuracy"][-1]
        
        if "train_accuracy" in history and len(history["train_accuracy"]) > 0:
            metrics["final_train_accuracy"] = history["train_accuracy"][-1]
        
        if "entropy_fragmentation" in history and len(history["entropy_fragmentation"]) > 0:
            metrics["final_entropy"] = history["entropy_fragmentation"][-1]
        
        if "angle_fragmentation" in history and len(history["angle_fragmentation"]) > 0:
            metrics["final_angle"] = history["angle_fragmentation"][-1]
        
        if "layer_fragmentation" in history:
            layer_frag = history["layer_fragmentation"]
            for metric_type in ["entropy", "angle"]:
                if metric_type in layer_frag:
                    for layer_name, value in layer_frag[metric_type].items():
                        metrics[f"final_{metric_type}_{layer_name}"] = value
        
        # Best values
        if "test_accuracy" in history and len(history["test_accuracy"]) > 0:
            metrics["best_test_accuracy"] = max(history["test_accuracy"])
            metrics["best_epoch"] = history["test_accuracy"].index(metrics["best_test_accuracy"])
        
        # Experiment config
        if "config" in history:
            for key, value in history["config"].items():
                if isinstance(value, (int, float, str, bool)):
                    metrics[f"config_{key}"] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)):
                            metrics[f"config_{key}_{subkey}"] = subvalue
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error extracting metrics from {history_path}: {str(e)}")
        return {}

def aggregate_baseline_results() -> pd.DataFrame:
    """
    Aggregate results from baseline experiments.
    
    Returns:
        DataFrame of aggregated baseline results
    """
    baseline_dir = os.path.join(RESULTS_DIR, "baselines")
    
    if not os.path.exists(baseline_dir):
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return pd.DataFrame()
    
    # Collect results for each experiment
    results = []
    
    for dataset_name in DATASETS.keys():
        dataset_dir = os.path.join(baseline_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all experiment directories for this dataset
        experiment_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        for exp_dir in experiment_dirs:
            # Extract seed from directory name
            seed = None
            for i in range(10):
                if f"seed{i}" in exp_dir:
                    seed = i
                    break
            
            if seed is None:
                logger.warning(f"Could not determine seed for experiment: {exp_dir}")
                continue
            
            # Get history file path
            history_path = os.path.join(dataset_dir, exp_dir, "training_history.json")
            
            if not os.path.exists(history_path):
                logger.warning(f"Training history not found: {history_path}")
                continue
            
            # Extract metrics
            metrics = extract_metrics_from_history(history_path)
            
            if not metrics:
                logger.warning(f"Failed to extract metrics from: {history_path}")
                continue
            
            # Add experiment metadata
            metrics["dataset"] = dataset_name
            metrics["seed"] = seed
            metrics["experiment_type"] = "baseline"
            metrics["weight"] = 0.0
            metrics["reg_id"] = "baseline"
            metrics["experiment_dir"] = os.path.join(dataset_dir, exp_dir)
            
            results.append(metrics)
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        logger.info(f"Aggregated {len(df)} baseline experiment results")
        return df
    else:
        logger.warning("No baseline results found")
        return pd.DataFrame()

def aggregate_cohesion_results() -> pd.DataFrame:
    """
    Aggregate results from cohesion experiments.
    
    Returns:
        DataFrame of aggregated cohesion results
    """
    cohesion_dir = os.path.join(RESULTS_DIR, "cohesion")
    
    if not os.path.exists(cohesion_dir):
        logger.warning(f"Cohesion directory not found: {cohesion_dir}")
        return pd.DataFrame()
    
    # Collect results for each experiment
    results = []
    
    for dataset_name in DATASETS.keys():
        dataset_dir = os.path.join(cohesion_dir, dataset_name)
        
        if not os.path.exists(dataset_dir):
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all parameter directories
        param_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        
        for param_dir in param_dirs:
            param_path = os.path.join(dataset_dir, param_dir)
            
            # Find all seed directories
            seed_dirs = [d for d in os.listdir(param_path) if os.path.isdir(os.path.join(param_path, d))]
            
            for seed_dir in seed_dirs:
                # Extract seed from directory name
                seed = None
                if seed_dir.startswith("seed_"):
                    try:
                        seed = int(seed_dir.split("_")[1])
                    except:
                        pass
                
                if seed is None:
                    logger.warning(f"Could not determine seed for experiment: {seed_dir}")
                    continue
                
                # Get history file path
                history_path = os.path.join(param_path, seed_dir, "training_history.json")
                
                if not os.path.exists(history_path):
                    logger.warning(f"Training history not found: {history_path}")
                    continue
                
                # Extract metrics
                metrics = extract_metrics_from_history(history_path)
                
                if not metrics:
                    logger.warning(f"Failed to extract metrics from: {history_path}")
                    continue
                
                # Parse regularization parameters from directory name
                weight = 0.0
                temperature = 0.07
                threshold = 0.0
                layers = []
                
                # Add parameters from config if available
                if "config_regularization_weight" in metrics:
                    weight = metrics["config_regularization_weight"]
                elif param_dir.startswith("w"):
                    # Try to parse from directory name
                    parts = param_dir.split("_")
                    for part in parts:
                        if part.startswith("w"):
                            try:
                                weight = float(part[1:])
                            except:
                                pass
                        elif part.startswith("t"):
                            try:
                                temperature = float(part[1:])
                            except:
                                pass
                        elif part.startswith("thr"):
                            try:
                                threshold = float(part[3:])
                            except:
                                pass
                        elif part.startswith("L"):
                            layer_nums = part[1:].split("-")
                            layers = [f"layer{n}" for n in layer_nums]
                
                # Add experiment metadata
                metrics["dataset"] = dataset_name
                metrics["seed"] = seed
                metrics["experiment_type"] = "cohesion"
                metrics["weight"] = weight
                metrics["temperature"] = temperature
                metrics["threshold"] = threshold
                metrics["layers"] = str(layers)
                metrics["reg_id"] = param_dir
                metrics["experiment_dir"] = os.path.join(param_path, seed_dir)
                
                results.append(metrics)
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        logger.info(f"Aggregated {len(df)} cohesion experiment results")
        return df
    else:
        logger.warning("No cohesion results found")
        return pd.DataFrame()

def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics across seeds for each experiment configuration.
    
    Args:
        df: DataFrame of experiment results
        
    Returns:
        DataFrame of summary statistics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Define grouping columns (everything except seed, experiment_dir, and metrics)
    groupby_cols = [col for col in df.columns if col not in [
        "seed", "experiment_dir", "final_test_accuracy", "final_train_accuracy", 
        "final_entropy", "final_angle", "best_test_accuracy", "best_epoch"
    ]]
    
    # Define metrics to compute statistics for
    metric_cols = [
        "final_test_accuracy", "final_train_accuracy", 
        "final_entropy", "final_angle", "best_test_accuracy"
    ]
    
    # Compute mean and std for each metric
    stats = []
    
    for name, group in df.groupby(groupby_cols):
        if not isinstance(name, tuple):
            name = (name,)
        
        # Create a row with group values
        row = {col: val for col, val in zip(groupby_cols, name)}
        
        # Add count of experiments
        row["n_experiments"] = len(group)
        
        # Compute statistics for each metric
        for metric in metric_cols:
            if metric in group.columns:
                values = group[metric].dropna()
                if len(values) > 0:
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std()
                    row[f"{metric}_min"] = values.min()
                    row[f"{metric}_max"] = values.max()
        
        stats.append(row)
    
    # Create DataFrame
    stats_df = pd.DataFrame(stats)
    
    # Sort by dataset and weight
    if "dataset" in stats_df.columns and "weight" in stats_df.columns:
        stats_df = stats_df.sort_values(["dataset", "weight"])
    
    return stats_df

def compute_delta_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta metrics vs. baseline for each cohesion experiment.
    
    Args:
        df: DataFrame of all experiment results
        
    Returns:
        DataFrame with delta metrics added
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Get baseline results
    baseline_df = df[df["experiment_type"] == "baseline"].copy()
    
    # Metrics to compute deltas for
    delta_metrics = [
        "final_test_accuracy", "final_train_accuracy", 
        "final_entropy", "final_angle", "best_test_accuracy"
    ]
    
    # Initialize delta columns
    for metric in delta_metrics:
        result_df[f"delta_{metric}"] = np.nan
    
    # Compute deltas for each dataset and seed
    for dataset in df["dataset"].unique():
        for seed in df["seed"].unique():
            # Get baseline values for this dataset and seed
            baseline_row = baseline_df[(baseline_df["dataset"] == dataset) & (baseline_df["seed"] == seed)]
            
            if baseline_row.empty:
                logger.warning(f"No baseline found for {dataset} seed {seed}")
                continue
            
            # Get baseline values
            baseline_values = {}
            for metric in delta_metrics:
                if metric in baseline_row.columns:
                    baseline_values[metric] = baseline_row[metric].values[0]
            
            # Compute deltas for cohesion experiments
            for metric, baseline_value in baseline_values.items():
                # Find rows for this dataset and seed
                mask = (result_df["dataset"] == dataset) & (result_df["seed"] == seed)
                
                # Compute delta
                result_df.loc[mask, f"delta_{metric}"] = result_df.loc[mask, metric] - baseline_value
    
    return result_df

def run_statistical_tests(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Run statistical tests comparing cohesion results to baseline.
    
    Args:
        df: DataFrame of all experiment results
        
    Returns:
        Dictionary of test results
    """
    if df.empty:
        return {}
    
    test_results = {}
    
    # Metrics to run tests on
    test_metrics = ["final_test_accuracy", "final_entropy", "final_angle"]
    
    # For each dataset and regularization configuration, run tests
    for dataset in df["dataset"].unique():
        dataset_results = {}
        
        # Get baseline results for this dataset
        baseline_df = df[(df["dataset"] == dataset) & (df["experiment_type"] == "baseline")]
        
        if baseline_df.empty:
            logger.warning(f"No baseline results found for dataset: {dataset}")
            continue
        
        # Group cohesion results by weight, temperature, threshold, layers
        cohesion_df = df[(df["dataset"] == dataset) & (df["experiment_type"] == "cohesion")]
        
        if cohesion_df.empty:
            logger.warning(f"No cohesion results found for dataset: {dataset}")
            continue
        
        # Group by regularization parameters
        for name, group in cohesion_df.groupby(["weight", "temperature", "threshold", "layers"]):
            weight, temperature, threshold, layers = name
            
            # Create a key for this configuration
            config_key = f"w{weight}_t{temperature}_thr{threshold}_layers{layers}"
            config_results = {}
            
            # Run tests for each metric
            for metric in test_metrics:
                if metric not in baseline_df.columns or metric not in group.columns:
                    continue
                
                # Get values
                baseline_values = baseline_df[metric].dropna().values
                cohesion_values = group[metric].dropna().values
                
                if len(baseline_values) < 2 or len(cohesion_values) < 2:
                    logger.warning(f"Not enough data for statistical tests on {metric} for {dataset} {config_key}")
                    continue
                
                # Run t-test (paired if same number of samples)
                if len(baseline_values) == len(cohesion_values):
                    t_stat, p_value = stats.ttest_rel(baseline_values, cohesion_values)
                    test_type = "paired t-test"
                else:
                    t_stat, p_value = stats.ttest_ind(baseline_values, cohesion_values, equal_var=False)
                    test_type = "unpaired t-test"
                
                # Run Wilcoxon test if paired
                if len(baseline_values) == len(cohesion_values):
                    try:
                        w_stat, w_p_value = stats.wilcoxon(baseline_values, cohesion_values)
                        wilcoxon_result = {
                            "statistic": float(w_stat),
                            "p_value": float(w_p_value),
                            "significant_0.05": bool(w_p_value < 0.05)
                        }
                    except:
                        wilcoxon_result = None
                else:
                    wilcoxon_result = None
                
                # Store results
                config_results[metric] = {
                    "t_test": {
                        "type": test_type,
                        "statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant_0.05": bool(p_value < 0.05)
                    },
                    "wilcoxon_test": wilcoxon_result,
                    "baseline_mean": float(baseline_values.mean()),
                    "baseline_std": float(baseline_values.std()),
                    "cohesion_mean": float(cohesion_values.mean()),
                    "cohesion_std": float(cohesion_values.std()),
                    "mean_difference": float(cohesion_values.mean() - baseline_values.mean()),
                    "percent_change": float((cohesion_values.mean() - baseline_values.mean()) / baseline_values.mean() * 100)
                }
            
            dataset_results[config_key] = config_results
        
        test_results[dataset] = dataset_results
    
    return test_results

def main():
    """Main function to aggregate experiment results."""
    parser = argparse.ArgumentParser(
        description="Aggregate results from baseline and cohesion experiments."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(RESULTS_DIR, "analysis"),
        help="Directory to save aggregated results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Aggregate baseline results
    baseline_df = aggregate_baseline_results()
    
    # Aggregate cohesion results
    cohesion_df = aggregate_cohesion_results()
    
    # Combine results
    all_results_df = pd.concat([baseline_df, cohesion_df], ignore_index=True)
    
    if all_results_df.empty:
        logger.error("No experiment results found")
        sys.exit(1)
    
    logger.info(f"Aggregated {len(all_results_df)} total experiment results")
    
    # Compute deltas vs baseline
    all_results_df = compute_delta_vs_baseline(all_results_df)
    
    # Compute summary statistics
    stats_df = compute_summary_statistics(all_results_df)
    
    # Run statistical tests
    test_results = run_statistical_tests(all_results_df)
    
    # Save results to CSV
    all_results_path = os.path.join(args.output_dir, "cohesion_summary.csv")
    all_results_df.to_csv(all_results_path, index=False)
    logger.info(f"Saved aggregated results to {all_results_path}")
    
    stats_path = os.path.join(args.output_dir, "cohesion_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved summary statistics to {stats_path}")
    
    # Save test results to JSON
    if test_results:
        import json
        test_results_path = os.path.join(args.output_dir, "statistical_tests.json")
        with open(test_results_path, "w") as f:
            # Convert any tensors or numpy arrays to Python types
            serializable_results = convert_tensors_to_python(test_results)
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Saved statistical test results to {test_results_path}")
        
        # Also save a simplified version to TXT
        test_summary_path = os.path.join(args.output_dir, "statistical_tests_summary.txt")
        with open(test_summary_path, "w") as f:
            f.write("# Statistical Test Results Summary\n\n")
            
            for dataset, dataset_results in test_results.items():
                f.write(f"## Dataset: {dataset}\n\n")
                
                for config, config_results in dataset_results.items():
                    f.write(f"### Configuration: {config}\n\n")
                    
                    for metric, metric_results in config_results.items():
                        f.write(f"#### Metric: {metric}\n")
                        f.write(f"- Baseline: {metric_results['baseline_mean']:.4f} ± {metric_results['baseline_std']:.4f}\n")
                        f.write(f"- Cohesion: {metric_results['cohesion_mean']:.4f} ± {metric_results['cohesion_std']:.4f}\n")
                        f.write(f"- Difference: {metric_results['mean_difference']:.4f} ({metric_results['percent_change']:.2f}%)\n")
                        
                        t_test = metric_results['t_test']
                        f.write(f"- T-test ({t_test['type']}): p={t_test['p_value']:.4f} ")
                        f.write("(significant)\n" if t_test['significant_0.05'] else "(not significant)\n")
                        
                        if metric_results['wilcoxon_test']:
                            wilcoxon = metric_results['wilcoxon_test']
                            f.write(f"- Wilcoxon test: p={wilcoxon['p_value']:.4f} ")
                            f.write("(significant)\n" if wilcoxon['significant_0.05'] else "(not significant)\n")
                        
                        f.write("\n")
                    
                    f.write("\n")
            
        logger.info(f"Saved simplified statistical test results to {test_summary_path}")
    
    logger.info("Aggregation complete")

if __name__ == "__main__":
    main() 