"""
Prepare complete data for LLM analysis including ALL windowed paths.

This script compiles:
1. ALL paths from each window (not just archetypal)
2. Cross-layer metrics
3. Clustering information for each layer
4. Example words for each path
5. Transition patterns between windows
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
from collections import defaultdict, Counter

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def load_all_data(results_dir: str):
    """Load all necessary data from the experiment."""
    results_path = Path(results_dir)
    
    # Load windowed analysis results
    windowed_path = results_path / "windowed_analysis" / "all_windowed_paths_complete.pkl"
    with open(windowed_path, 'rb') as f:
        windowed_data = pickle.load(f)
    
    # Load clustering metrics for each layer
    layer_metrics = {}
    for layer_idx in range(12):
        metrics_path = results_path / "clustering" / f"layer_{layer_idx}" / "clustering_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                layer_metrics[f"layer_{layer_idx}"] = json.load(f)
    
    # Load trajectories
    trajectories_path = results_path / "paths" / "trajectories.pkl"
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    return windowed_data, layer_metrics, trajectories_data


def prepare_llm_data(results_dir: str):
    """Prepare comprehensive data package for LLM analysis."""
    
    print("Loading all experiment data...")
    windowed_data, layer_metrics, trajectories_data = load_all_data(results_dir)
    
    # Structure for LLM analysis
    llm_data = {
        "experiment_info": {
            "type": "GPT-2 Semantic Subtypes CTA Analysis",
            "n_words": len(trajectories_data['trajectories']),
            "n_layers": 12,
            "windows": {
                "early": "layers 0-3",
                "middle": "layers 4-7", 
                "late": "layers 8-11"
            }
        },
        "layer_clustering_info": {},
        "windowed_paths": {},
        "cross_window_transitions": windowed_data['cross_window_transitions'],
        "metrics_summary": {}
    }
    
    # Add layer clustering information
    print("\nPreparing layer clustering information...")
    for layer_name, metrics in layer_metrics.items():
        layer_idx = int(layer_name.split('_')[1])
        llm_data["layer_clustering_info"][f"layer_{layer_idx}"] = {
            "n_clusters": metrics.get("n_clusters", "unknown"),
            "silhouette_score": metrics.get("silhouette_score", "unknown"),
            "cluster_sizes": metrics.get("cluster_sizes", [])
        }
    
    # Add ALL paths for each window
    print("\nPreparing complete path data for each window...")
    for window_name in ['early', 'middle', 'late']:
        window_info = windowed_data[window_name]
        
        # Prepare path data for LLM
        paths_for_llm = []
        
        for path_info in window_info['all_paths']:
            # Include full information for each path
            path_data = {
                "cluster_sequence": path_info['path'],  # e.g., [1, 0, 0, 1]
                "frequency": path_info['frequency'],
                "percentage": path_info['percentage'],
                "n_words": path_info['n_words'],
                "example_words": path_info['example_words'][:10],  # Up to 10 examples
                "characteristics": {
                    "stability": path_info['stability'],
                    "n_unique_clusters": path_info['n_unique_clusters'],
                    "is_stable": path_info['is_stable'],
                    "is_wandering": path_info['is_wandering'],
                    "is_singleton": path_info['is_singleton']
                }
            }
            
            # For rare paths, include all words
            if path_info['frequency'] <= 5 and path_info['all_words']:
                path_data["all_words"] = path_info['all_words']
            
            paths_for_llm.append(path_data)
        
        llm_data["windowed_paths"][window_name] = {
            "window_info": window_info['window_info'],
            "summary": window_info['summary'],
            "metrics": window_info['metrics'],
            "all_paths": paths_for_llm
        }
    
    # Add transition analysis
    print("\nPreparing transition analysis...")
    transition_summary = {}
    
    for trans_type, trans_data in windowed_data['cross_window_transitions'].items():
        # Get top transitions with cluster sequences
        top_transitions = []
        for trans in trans_data['top_transitions'][:10]:  # Top 10 transitions
            top_transitions.append({
                "from_sequence": trans['from'],
                "to_sequence": trans['to'],
                "count": trans['count'],
                "percentage": trans['percentage']
            })
        
        transition_summary[trans_type] = {
            "convergence_ratio": trans_data['convergence_ratio'],
            "is_converging": trans_data['is_converging'],
            "from_paths_count": trans_data['from_paths'],
            "to_paths_count": trans_data['to_paths'],
            "top_transitions": top_transitions
        }
    
    llm_data["transition_analysis"] = transition_summary
    
    # Add overall metrics summary
    print("\nCalculating overall metrics...")
    
    # Path diversity progression
    path_diversity = []
    for window in ['early', 'middle', 'late']:
        total_paths = windowed_data[window]['summary']['total_unique_paths']
        dominant_percentage = windowed_data[window]['all_paths'][0]['percentage']
        diversity = 1 - (dominant_percentage / 100)
        path_diversity.append({
            "window": window,
            "unique_paths": total_paths,
            "diversity_score": round(diversity, 3),
            "dominant_path_percentage": dominant_percentage
        })
    
    # Fragmentation progression
    fragmentation_progression = []
    for window in ['early', 'middle', 'late']:
        frag = windowed_data[window]['metrics']['fragmentation']
        fragmentation_progression.append({
            "window": window,
            "fragmentation": frag
        })
    
    llm_data["metrics_summary"] = {
        "path_diversity_progression": path_diversity,
        "fragmentation_progression": fragmentation_progression,
        "convergence_pattern": "19 paths -> 5 paths -> 4 paths",
        "dominant_path_emergence": {
            "early": "No single dominant path (highest: 27.2%)",
            "middle": "Strong dominance emerges: [1,0,0,1] at 72.8%",
            "late": "Dominance continues: [1,0,0,1] at 73.1%"
        }
    }
    
    # Add instructions for LLM analysis
    llm_data["analysis_instructions"] = {
        "task_1": "Analyze and label each cluster based on the words it contains",
        "task_2": "Interpret the meaning of each path's cluster sequence",
        "task_3": "Explain the convergence pattern from early diversity to late dominance",
        "task_4": "Identify semantic patterns in path transitions",
        "focus_areas": [
            "What semantic properties do words in each path share?",
            "Why does [1,0,0,1] become so dominant in middle/late layers?",
            "What happens to the diverse early paths as they converge?",
            "Are there systematic patterns in which early paths map to which middle paths?"
        ]
    }
    
    # Save complete data package
    output_dir = Path(results_dir) / "llm_analysis_data"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON version
    with open(output_dir / "complete_paths_for_llm.json", 'w') as f:
        json.dump(llm_data, f, indent=2)
    
    # Save pickle version
    with open(output_dir / "complete_paths_for_llm.pkl", 'wb') as f:
        pickle.dump(llm_data, f)
    
    # Create a human-readable summary
    create_analysis_summary(llm_data, output_dir)
    
    print(f"\nComplete LLM data package saved to {output_dir}")
    
    return llm_data


def create_analysis_summary(llm_data: dict, output_dir: Path):
    """Create a summary document highlighting key patterns for analysis."""
    
    summary = []
    summary.append("GPT-2 SEMANTIC SUBTYPES - COMPLETE PATH ANALYSIS SUMMARY")
    summary.append("=" * 60)
    
    # Overview
    summary.append("\nOVERVIEW")
    summary.append("-" * 30)
    summary.append(f"Total words analyzed: {llm_data['experiment_info']['n_words']}")
    summary.append(f"Network layers: {llm_data['experiment_info']['n_layers']}")
    summary.append("Analysis windows: Early (L0-3), Middle (L4-7), Late (L8-11)")
    
    # Path counts by window
    summary.append("\n\nPATH DIVERSITY BY WINDOW")
    summary.append("-" * 30)
    for window in ['early', 'middle', 'late']:
        window_data = llm_data['windowed_paths'][window]
        summary.append(f"\n{window.upper()} WINDOW:")
        summary.append(f"  Total unique paths: {window_data['summary']['total_unique_paths']}")
        summary.append(f"  Singleton paths: {window_data['summary']['singleton_paths']}")
        summary.append(f"  Common paths (>=10 words): {window_data['summary']['common_paths_gte10']}")
        
        # Top 3 paths
        summary.append(f"\n  Top 3 paths:")
        for i, path in enumerate(window_data['all_paths'][:3]):
            summary.append(f"    {i+1}. {path['cluster_sequence']} - {path['frequency']} words ({path['percentage']}%)")
            summary.append(f"       Examples: {', '.join(path['example_words'][:5])}")
    
    # Key transitions
    summary.append("\n\nKEY TRANSITIONS")
    summary.append("-" * 30)
    for trans_type, trans_data in llm_data['transition_analysis'].items():
        summary.append(f"\n{trans_type.replace('_', ' ').title()}:")
        summary.append(f"  Convergence: {trans_data['from_paths_count']} -> {trans_data['to_paths_count']} paths")
        summary.append(f"  Top transition:")
        if trans_data['top_transitions']:
            top = trans_data['top_transitions'][0]
            summary.append(f"    {top['from_sequence']} -> {top['to_sequence']}")
            summary.append(f"    Frequency: {top['count']} ({top['percentage']}%)")
    
    # Key findings
    summary.append("\n\nKEY FINDINGS FOR LLM ANALYSIS")
    summary.append("-" * 30)
    summary.append("1. Massive convergence from 19 -> 5 -> 4 unique paths")
    summary.append("2. Dominant path [1,0,0,1] emerges in middle layers (72.8%)")
    summary.append("3. High stability in early layers (0.724) vs low in middle/late (~0.34)")
    summary.append("4. Most early diversity converges to the dominant middle path")
    
    # Save summary
    with open(output_dir / "analysis_summary.txt", 'w') as f:
        f.write('\n'.join(summary))
    
    print("\nAnalysis summary created: analysis_summary.txt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use the most complete results
        results_dir = "results/unified_cta_config/unified_cta_20250524_073316"
    
    # Prepare complete data package
    llm_data = prepare_llm_data(results_dir)