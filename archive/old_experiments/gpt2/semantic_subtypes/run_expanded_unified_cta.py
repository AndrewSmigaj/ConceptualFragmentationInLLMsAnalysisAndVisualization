"""
Run the expanded GPT-2 experiment using the unified CTA pipeline.
This will extract activations and run the full analysis on the expanded word list.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent / "unified_cta"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Change to unified_cta directory
os.chdir(Path(__file__).parent / "unified_cta")

from run_unified_pipeline import main as run_pipeline
from prepare_complete_llm_data import prepare_complete_analysis_data

def update_paper_metrics(results_path):
    """Extract key metrics from results to update the paper."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\n=== KEY METRICS FOR PAPER UPDATE ===")
    
    # Total words analyzed
    total_words = results.get('total_words', 0)
    print(f"\nTotal words analyzed: {total_words}")
    
    # Grammatical distribution
    if 'word_distribution' in results:
        dist = results['word_distribution']
        print("\nGrammatical distribution:")
        for cat, stats in dist.items():
            print(f"  {cat}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    # Entity superhighway convergence
    if 'convergence_stats' in results:
        conv = results['convergence_stats']
        entity_convergence = conv.get('entity_superhighway_percentage', 0)
        print(f"\nEntity superhighway convergence: {entity_convergence:.1f}%")
        print(f"95% CI: {conv.get('ci_lower', 0):.1f}%-{conv.get('ci_upper', 0):.1f}%")
    
    # Path reduction
    if 'path_evolution' in results:
        paths = results['path_evolution']
        print(f"\nPath reduction:")
        print(f"  Early window: {paths.get('early', {}).get('unique_paths', 'N/A')} paths")
        print(f"  Middle window: {paths.get('middle', {}).get('unique_paths', 'N/A')} paths")
        print(f"  Late window: {paths.get('late', {}).get('unique_paths', 'N/A')} paths")
    
    # Stability metrics
    if 'stability_metrics' in results:
        stability = results['stability_metrics']
        print(f"\nStability analysis:")
        print(f"  Early: {stability.get('early', 'N/A')}")
        print(f"  Middle: {stability.get('middle', 'N/A')}")
        print(f"  Late: {stability.get('late', 'N/A')}")
    
    # Chi-square test
    if 'statistical_tests' in results:
        tests = results['statistical_tests']
        print(f"\nChi-square test for grammatical organization:")
        print(f"  χ² = {tests.get('chi_square', 'N/A')}")
        print(f"  p-value = {tests.get('p_value', 'N/A')}")
        print(f"  Cramér's V = {tests.get('cramers_v', 'N/A')}")
    
    # Fragmentation metrics
    if 'fragmentation_metrics' in results:
        frag = results['fragmentation_metrics']
        print(f"\nFragmentation metrics by window:")
        for window in ['early', 'middle', 'late']:
            if window in frag:
                w = frag[window]
                print(f"  {window.capitalize()}:")
                print(f"    FC: {w.get('fc', 'N/A')}")
                print(f"    CE: {w.get('ce', 'N/A')}")
                print(f"    SA: {w.get('sa', 'N/A')}°")
    
    print("\n" + "="*50)
    print("UPDATE THESE METRICS IN THE PAPER!")
    print("="*50)

def main():
    """Run the expanded experiment."""
    print("=== Running Expanded GPT-2 Experiment ===")
    print("Using unified CTA pipeline with expanded word list")
    
    # Check if expanded curated data exists
    expanded_data_path = Path("../data/gpt2_semantic_subtypes_curated_expanded.json")
    if not expanded_data_path.exists():
        print(f"ERROR: Expanded data not found at {expanded_data_path}")
        return
    
    # Copy to unified_cta data directory
    target_path = Path("data/gpt2_semantic_subtypes_curated_expanded.json")
    target_path.parent.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy2(expanded_data_path, target_path)
    print(f"Copied expanded data to {target_path}")
    
    # Update config to use expanded data
    config_updates = {
        "curated_data_path": "data/gpt2_semantic_subtypes_curated_expanded.json",
        "output_suffix": "_expanded",
        "results_dir": "results_expanded"
    }
    
    # Save config updates
    with open("config_expanded.json", 'w') as f:
        json.dump(config_updates, f, indent=2)
    
    try:
        # Run the pipeline
        print("\nStarting unified CTA pipeline...")
        start_time = time.time()
        
        # Run main pipeline
        run_pipeline()
        
        # Prepare LLM analysis data
        print("\nPreparing LLM analysis data...")
        prepare_complete_analysis_data()
        
        elapsed = time.time() - start_time
        print(f"\nPipeline completed in {elapsed:.1f} seconds")
        
        # Extract and display metrics for paper update
        results_path = Path("results_expanded/unified_analysis_results.json")
        if results_path.exists():
            update_paper_metrics(results_path)
        else:
            # Try alternate location
            results_path = Path("results/unified_analysis_results_expanded.json")
            if results_path.exists():
                update_paper_metrics(results_path)
            else:
                print("\nWARNING: Could not find results file to extract metrics")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()