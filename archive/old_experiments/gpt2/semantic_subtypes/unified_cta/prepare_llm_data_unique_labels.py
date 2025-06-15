"""
Prepare LLM data with unique cluster labels to avoid confusion.

Converts integer cluster IDs to unique labels like L0_C0, L1_C2, etc.
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


def convert_to_unique_labels(trajectories: dict) -> tuple:
    """
    Convert integer cluster IDs to unique labels.
    
    Returns:
        (unique_trajectories, cluster_mapping)
    """
    # First pass: collect all unique clusters per layer
    clusters_per_layer = defaultdict(set)
    
    for word, traj in trajectories.items():
        for layer_idx, cluster_id in enumerate(traj):
            clusters_per_layer[layer_idx].add(cluster_id)
    
    # Create mapping from (layer, cluster_id) to unique label
    cluster_mapping = {}
    for layer_idx in sorted(clusters_per_layer.keys()):
        for cluster_id in sorted(clusters_per_layer[layer_idx]):
            unique_label = f"L{layer_idx}_C{cluster_id}"
            cluster_mapping[(layer_idx, cluster_id)] = unique_label
    
    # Convert trajectories
    unique_trajectories = {}
    for word, traj in trajectories.items():
        unique_traj = []
        for layer_idx, cluster_id in enumerate(traj):
            unique_label = cluster_mapping[(layer_idx, cluster_id)]
            unique_traj.append(unique_label)
        unique_trajectories[word] = unique_traj
    
    return unique_trajectories, cluster_mapping


def prepare_llm_data_with_unique_labels(results_dir: str):
    """Prepare complete LLM data with unique cluster labels."""
    
    results_path = Path(results_dir)
    
    print("Loading experiment data...")
    
    # Load windowed analysis
    windowed_path = results_path / "windowed_analysis" / "all_windowed_paths_complete.pkl"
    with open(windowed_path, 'rb') as f:
        windowed_data = pickle.load(f)
    
    # Load trajectories
    trajectories_path = results_path / "paths" / "trajectories.pkl"
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    # Load clustering metrics
    layer_metrics = {}
    for layer_idx in range(12):
        metrics_path = results_path / "clustering" / f"layer_{layer_idx}" / "clustering_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                layer_metrics[layer_idx] = json.load(f)
    
    print("Converting to unique cluster labels...")
    
    # Convert trajectories to unique labels
    unique_trajectories, cluster_mapping = convert_to_unique_labels(
        trajectories_data['trajectories']
    )
    
    # Prepare LLM data structure
    llm_data = {
        "experiment_info": {
            "type": "GPT-2 Semantic Subtypes Analysis - Windowed Paths",
            "n_words": len(unique_trajectories),
            "n_layers": 12,
            "windows": {
                "early": "layers 0-3",
                "middle": "layers 4-7",
                "late": "layers 8-11"
            },
            "cluster_labeling": "L{layer}_C{cluster} format for globally unique IDs"
        },
        "cluster_info": {},
        "windowed_paths_unique": {},
        "transition_analysis": {},
        "key_patterns": {}
    }
    
    # Add cluster information with unique labels
    print("Preparing cluster information...")
    for layer_idx, metrics in layer_metrics.items():
        # Get all unique cluster IDs for this layer from the mapping
        clusters_in_layer = sorted([
            label for (l_idx, c_id), label in cluster_mapping.items() 
            if l_idx == layer_idx
        ])
        
        llm_data["cluster_info"][f"layer_{layer_idx}"] = {
            "n_clusters": metrics.get("n_clusters", len(clusters_in_layer)),
            "unique_cluster_ids": clusters_in_layer,
            "silhouette_score": metrics.get("silhouette_score", "unknown")
        }
    
    # Convert windowed paths to unique labels
    print("Converting windowed paths to unique labels...")
    
    windows = {
        'early': (0, 3),
        'middle': (4, 7),
        'late': (8, 11)
    }
    
    for window_name in ['early', 'middle', 'late']:
        window_info = windowed_data[window_name]
        start_layer, end_layer = windows[window_name]
        
        # Convert all paths in this window
        unique_paths = []
        
        for path_info in window_info['all_paths']:
            # Convert path to unique labels
            unique_path = []
            for offset, cluster_id in enumerate(path_info['path']):
                layer_idx = start_layer + offset
                unique_label = cluster_mapping[(layer_idx, cluster_id)]
                unique_path.append(unique_label)
            
            unique_path_info = {
                "cluster_sequence": unique_path,
                "frequency": path_info['frequency'],
                "percentage": path_info['percentage'],
                "n_words": path_info['n_words'],
                "example_words": path_info['example_words'][:10],
                "characteristics": {
                    "stability": path_info.get('stability', 0),
                    "n_unique_clusters": path_info.get('n_unique_clusters', 0),
                    "is_stable": path_info.get('is_stable', False),
                    "is_wandering": path_info.get('is_wandering', False),
                    "is_singleton": path_info.get('is_singleton', False)
                }
            }
            
            # Include all words for rare paths
            if path_info['frequency'] <= 5 and path_info.get('all_words'):
                unique_path_info["all_words"] = path_info['all_words']
            
            unique_paths.append(unique_path_info)
        
        llm_data["windowed_paths_unique"][window_name] = {
            "window_info": window_info['window_info'],
            "summary": window_info['summary'],
            "metrics": window_info['metrics'],
            "all_paths": unique_paths
        }
    
    # Analyze transitions with unique labels
    print("Analyzing transitions with unique labels...")
    
    transition_patterns = {
        'early_to_middle': defaultdict(int),
        'middle_to_late': defaultdict(int)
    }
    
    for word, traj in unique_trajectories.items():
        # Extract windowed paths
        early_path = tuple(traj[0:4])
        middle_path = tuple(traj[4:8])
        late_path = tuple(traj[8:12])
        
        transition_patterns['early_to_middle'][(early_path, middle_path)] += 1
        transition_patterns['middle_to_late'][(middle_path, late_path)] += 1
    
    # Prepare transition analysis
    for trans_type, patterns in transition_patterns.items():
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Get unique from/to paths
        from_paths = set(p[0] for p, _ in sorted_patterns)
        to_paths = set(p[1] for p, _ in sorted_patterns)
        
        top_transitions = []
        for (from_path, to_path), count in sorted_patterns[:10]:
            top_transitions.append({
                "from_sequence": list(from_path),
                "to_sequence": list(to_path),
                "count": count,
                "percentage": round(count / len(unique_trajectories) * 100, 2)
            })
        
        llm_data["transition_analysis"][trans_type] = {
            "from_paths_count": len(from_paths),
            "to_paths_count": len(to_paths),
            "convergence_ratio": round(len(to_paths) / len(from_paths), 3) if len(from_paths) > 0 else 1.0,
            "top_transitions": top_transitions
        }
    
    # Extract key patterns
    print("Identifying key patterns...")
    
    # Find the dominant path in each window
    dominant_paths = {}
    for window in ['early', 'middle', 'late']:
        top_path = llm_data["windowed_paths_unique"][window]["all_paths"][0]
        dominant_paths[window] = {
            "sequence": top_path["cluster_sequence"],
            "percentage": top_path["percentage"],
            "examples": top_path["example_words"][:5]
        }
    
    llm_data["key_patterns"] = {
        "dominant_paths": dominant_paths,
        "convergence_pattern": {
            "early": f"{len(llm_data['windowed_paths_unique']['early']['all_paths'])} unique paths",
            "middle": f"{len(llm_data['windowed_paths_unique']['middle']['all_paths'])} unique paths",
            "late": f"{len(llm_data['windowed_paths_unique']['late']['all_paths'])} unique paths"
        },
        "key_insight": "Massive convergence from diverse early processing to dominant middle/late patterns"
    }
    
    # Add analysis instructions
    llm_data["analysis_tasks"] = [
        "1. Analyze what types of words belong to each unique cluster (e.g., L0_C1, L4_C0)",
        "2. Interpret the meaning of each path sequence (e.g., [L0_C0, L1_C1, L2_C1, L3_C1])",
        "3. Explain why certain paths become dominant in middle/late layers",
        "4. Identify semantic patterns in path transitions between windows",
        "5. Provide interpretable names for key clusters based on their word membership"
    ]
    
    # Save results
    output_dir = results_path / "llm_analysis_data"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON version
    with open(output_dir / "all_paths_unique_labels.json", 'w') as f:
        json.dump(llm_data, f, indent=2)
    
    # Create human-readable summary
    create_summary_report(llm_data, output_dir)
    
    print(f"\nLLM data with unique labels saved to {output_dir}")
    
    return llm_data


def create_summary_report(llm_data: dict, output_dir: Path):
    """Create a summary report with unique labels."""
    
    summary = []
    summary.append("GPT-2 SEMANTIC SUBTYPES - ALL PATHS WITH UNIQUE LABELS")
    summary.append("=" * 60)
    
    summary.append("\nCLUSTER LABELING SCHEME")
    summary.append("-" * 30)
    summary.append("Format: L{layer}_C{cluster}")
    summary.append("Example: L0_C1 = Layer 0, Cluster 1")
    summary.append("This ensures globally unique cluster identification")
    
    # Show cluster counts per layer
    summary.append("\n\nCLUSTERS PER LAYER")
    summary.append("-" * 30)
    for layer_idx in range(12):
        layer_key = f"layer_{layer_idx}"
        if layer_key in llm_data["cluster_info"]:
            info = llm_data["cluster_info"][layer_key]
            summary.append(f"Layer {layer_idx}: {info['n_clusters']} clusters - {', '.join(info['unique_cluster_ids'])}")
    
    # Show all paths by window
    summary.append("\n\nALL PATHS BY WINDOW")
    summary.append("-" * 30)
    
    for window in ['early', 'middle', 'late']:
        window_data = llm_data["windowed_paths_unique"][window]
        summary.append(f"\n{window.upper()} WINDOW ({window_data['window_info']['layer_range']}):")
        summary.append(f"Total paths: {len(window_data['all_paths'])}")
        
        # List ALL paths
        for i, path in enumerate(window_data['all_paths']):
            summary.append(f"\n  Path {i+1}: {' -> '.join(path['cluster_sequence'])}")
            summary.append(f"    Frequency: {path['frequency']} words ({path['percentage']}%)")
            summary.append(f"    Examples: {', '.join(path['example_words'][:5])}")
            if path.get('all_words') and len(path['all_words']) <= 5:
                summary.append(f"    All words: {', '.join(path['all_words'])}")
    
    # Transition patterns
    summary.append("\n\nKEY TRANSITIONS")
    summary.append("-" * 30)
    for trans_type, trans_data in llm_data["transition_analysis"].items():
        summary.append(f"\n{trans_type.replace('_', ' ').title()}:")
        summary.append(f"  Convergence: {trans_data['from_paths_count']} -> {trans_data['to_paths_count']} paths")
        
        # Show top 3 transitions
        for i, trans in enumerate(trans_data['top_transitions'][:3]):
            summary.append(f"\n  Transition {i+1}:")
            summary.append(f"    From: {' -> '.join(trans['from_sequence'])}")
            summary.append(f"    To:   {' -> '.join(trans['to_sequence'])}")
            summary.append(f"    Count: {trans['count']} ({trans['percentage']}%)")
    
    # Save summary
    with open(output_dir / "all_paths_unique_labels_summary.txt", 'w') as f:
        f.write('\n'.join(summary))
    
    print("\nSummary report created: all_paths_unique_labels_summary.txt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "results/unified_cta_config/unified_cta_20250524_073316"
    
    # Prepare data with unique labels
    llm_data = prepare_llm_data_with_unique_labels(results_dir)