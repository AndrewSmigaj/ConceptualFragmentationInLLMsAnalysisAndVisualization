#!/usr/bin/env python3
"""
Run the complete analysis pipeline for top 10k GPT-2 tokens.
"""

import subprocess
import sys
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd, description):
    """Run a command and check for errors."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Running: {description}")
    logging.info(f"Command: {cmd}")
    logging.info(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        logging.error(f"Error running {description}:")
        logging.error(result.stderr)
        return False
    
    logging.info(f"Completed in {elapsed/60:.1f} minutes")
    return True

def main():
    base_dir = Path(__file__).parent
    python_exe = str(Path(sys.executable).parent.parent.parent / "venv311" / "Scripts" / "python.exe")
    
    logging.info("Starting GPT-2 Top 10k Token Analysis Pipeline")
    
    # Step 1: Check if clustering is complete
    clustering_results = base_dir / "clustering_results_per_layer" / "optimal_labels_all_layers_top10k.json"
    if not clustering_results.exists():
        logging.error(f"Clustering results not found at {clustering_results}")
        logging.error("Please run cluster_top_10k_per_layer.py first")
        return
    
    # Step 2: Run trajectory analysis
    if not run_command(
        f"{python_exe} analyze_top_10k_trajectories.py",
        "Trajectory Analysis"
    ):
        return
    
    # Step 3: Generate Sankey visualizations
    if not run_command(
        f"{python_exe} generate_top_10k_sankey.py",
        "Sankey Visualizations"
    ):
        return
    
    # Step 4: Prepare LLM analysis data
    if not run_command(
        f"{python_exe} prepare_llm_analysis_top_10k.py",
        "LLM Analysis Data Preparation"
    ):
        return
    
    logging.info("\n" + "="*60)
    logging.info("Pipeline completed successfully!")
    logging.info("="*60)
    
    # Print summary of outputs
    logging.info("\nGenerated outputs:")
    logging.info("1. Trajectory analysis: trajectory_analysis/")
    logging.info("2. Sankey visualizations: sankey_visualizations/")
    logging.info("3. LLM analysis data: llm_analysis/")
    
    # Print next steps
    logging.info("\nNext steps:")
    logging.info("1. Review trajectory analysis report: trajectory_analysis/trajectory_analysis_report.md")
    logging.info("2. Open Sankey visualizations: sankey_visualizations/top_10k_sankey_combined.html")
    logging.info("3. Use LLM analysis data for interpretability analysis")

if __name__ == "__main__":
    main()
