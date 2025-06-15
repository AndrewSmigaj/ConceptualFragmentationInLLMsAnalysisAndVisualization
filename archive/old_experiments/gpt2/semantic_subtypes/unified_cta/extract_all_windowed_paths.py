"""
Extract ALL windowed paths (not just archetypal) for comprehensive analysis.

Since the windowed approach reveals relatively few unique paths,
we'll extract every path with its frequency and example words.
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
from collections import Counter, defaultdict

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from windowed_path_analysis import WindowedPathAnalyzer


def extract_all_paths(results_dir: str):
    """Extract ALL paths for each window, not just archetypal."""
    
    # Load trajectories
    trajectories_path = Path(results_dir) / "paths" / "trajectories.pkl"
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    trajectories = trajectories_data['trajectories']
    
    # Initialize analyzer
    analyzer = WindowedPathAnalyzer()
    
    # Results storage
    all_paths_data = {}
    
    for window_name in ['early', 'middle', 'late']:
        print(f"\n=== {window_name.upper()} WINDOW ===")
        
        # Extract windowed trajectories
        windowed_trajs = analyzer.extract_windowed_trajectories(trajectories, window_name)
        
        # Count all paths
        path_counts = Counter(tuple(traj) for traj in windowed_trajs.values())
        
        # Get words for each path
        path_to_words = defaultdict(list)
        for word, trajectory in windowed_trajs.items():
            path_key = tuple(trajectory)
            path_to_words[path_key].append(word)
        
        # Build complete path data
        window_paths = []
        
        for path, count in path_counts.most_common():  # Sorted by frequency
            # Calculate path characteristics
            words = path_to_words[path]
            
            # Path stability (how many clusters stay same)
            stability = sum(1 for i in range(len(path)-1) if path[i] == path[i+1]) / (len(path)-1)
            
            # Path range (how many different clusters visited)
            n_unique = len(set(path))
            
            # Is this a singleton path (only one word)?
            is_singleton = count == 1
            
            path_info = {
                'path': list(path),
                'frequency': count,
                'percentage': round(count / len(windowed_trajs) * 100, 2),
                'n_words': len(words),
                'example_words': words[:10],  # Up to 10 examples
                'all_words': words if count <= 5 else None,  # Include all words for rare paths
                'stability': round(stability, 3),
                'n_unique_clusters': n_unique,
                'is_stable': stability > 0.7,
                'is_wandering': n_unique > len(path) * 0.6,
                'is_singleton': is_singleton
            }
            
            window_paths.append(path_info)
        
        # Calculate window statistics
        total_paths = len(window_paths)
        singleton_paths = sum(1 for p in window_paths if p['is_singleton'])
        rare_paths = sum(1 for p in window_paths if p['frequency'] < 3)
        common_paths = sum(1 for p in window_paths if p['frequency'] >= 10)
        
        # Get trajectory metrics
        traj_metrics = analyzer.calculate_trajectory_metrics(windowed_trajs)
        
        # Store results
        all_paths_data[window_name] = {
            'window_info': {
                'name': window_name,
                'layers': analyzer.windows[window_name],
                'layer_range': f"L{analyzer.windows[window_name][0]}-L{analyzer.windows[window_name][1]}"
            },
            'summary': {
                'total_words': len(windowed_trajs),
                'total_unique_paths': total_paths,
                'singleton_paths': singleton_paths,
                'rare_paths_lt3': rare_paths,
                'common_paths_gte10': common_paths,
                'most_common_path_percentage': window_paths[0]['percentage'] if window_paths else 0
            },
            'metrics': {
                'fragmentation': round(traj_metrics['fragmentation'], 3),
                'stability': round(traj_metrics['stability'], 3),
                'convergence': round(traj_metrics['convergence'], 3),
                'avg_clusters_per_layer': round(np.mean(traj_metrics['unique_clusters_per_layer']), 2),
                'layer_entropies': [round(e, 3) for e in traj_metrics['layer_entropies']]
            },
            'all_paths': window_paths
        }
        
        # Print summary
        print(f"Layers: {analyzer.windows[window_name][0]}-{analyzer.windows[window_name][1]}")
        print(f"Total unique paths: {total_paths}")
        print(f"  - Singleton paths: {singleton_paths}")
        print(f"  - Rare paths (<3 occurrences): {rare_paths}")
        print(f"  - Common paths (>=10 occurrences): {common_paths}")
        print(f"Most common path: {window_paths[0]['path']} ({window_paths[0]['percentage']}%)")
        print(f"Path diversity: {1 - window_paths[0]['percentage']/100:.3f}")
        
        # Show path distribution
        print("\nPath frequency distribution:")
        for i, path_info in enumerate(window_paths[:5]):
            print(f"  {i+1}. {path_info['path']}: {path_info['frequency']} ({path_info['percentage']}%)")
            print(f"     Examples: {', '.join(path_info['example_words'][:5])}")
    
    # Add cross-window analysis with ALL paths
    print("\n=== CROSS-WINDOW PATH TRANSITIONS ===")
    
    # Track ALL path transitions
    path_transitions = {
        'early_to_middle': defaultdict(int),
        'middle_to_late': defaultdict(int)
    }
    
    for word, full_traj in trajectories.items():
        early_path = tuple(full_traj[0:4])
        middle_path = tuple(full_traj[4:8])
        late_path = tuple(full_traj[8:12])
        
        path_transitions['early_to_middle'][(early_path, middle_path)] += 1
        path_transitions['middle_to_late'][(middle_path, late_path)] += 1
    
    # Analyze transitions
    transition_analysis = {}
    
    for transition_type, transitions in path_transitions.items():
        # Sort by frequency
        sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate statistics
        total_transitions = sum(count for _, count in sorted_transitions)
        unique_transitions = len(sorted_transitions)
        
        # Find convergence/divergence patterns
        if transition_type == 'early_to_middle':
            from_paths = set(path[0] for path, _ in sorted_transitions)
            to_paths = set(path[1] for path, _ in sorted_transitions)
        else:
            from_paths = set(path[0] for path, _ in sorted_transitions)
            to_paths = set(path[1] for path, _ in sorted_transitions)
        
        convergence_ratio = len(to_paths) / len(from_paths) if len(from_paths) > 0 else 1.0
        
        transition_analysis[transition_type] = {
            'total_transitions': total_transitions,
            'unique_transitions': unique_transitions,
            'from_paths': len(from_paths),
            'to_paths': len(to_paths),
            'convergence_ratio': round(convergence_ratio, 3),
            'is_converging': convergence_ratio < 0.8,
            'top_transitions': [
                {
                    'from': list(trans[0]),
                    'to': list(trans[1]),
                    'count': count,
                    'percentage': round(count / total_transitions * 100, 2)
                }
                for trans, count in sorted_transitions[:10]
            ]
        }
        
        print(f"\n{transition_type.replace('_', ' ').title()}:")
        print(f"  From {len(from_paths)} paths -> To {len(to_paths)} paths")
        print(f"  Convergence ratio: {convergence_ratio:.3f}")
        print(f"  Top transition: {sorted_transitions[0][0][0]} -> {sorted_transitions[0][0][1]}")
        print(f"    Frequency: {sorted_transitions[0][1]} ({sorted_transitions[0][1]/total_transitions*100:.1f}%)")
    
    all_paths_data['cross_window_transitions'] = transition_analysis
    
    # Save comprehensive results
    output_dir = Path(results_dir) / "windowed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON version
    with open(output_dir / "all_windowed_paths.json", 'w') as f:
        json.dump(all_paths_data, f, indent=2)
    
    # Save pickle version with full data
    with open(output_dir / "all_windowed_paths_complete.pkl", 'wb') as f:
        pickle.dump(all_paths_data, f)
    
    # Create a summary report
    create_summary_report(all_paths_data, output_dir)
    
    print(f"\n\nComplete path data saved to {output_dir}")
    
    return all_paths_data


def create_summary_report(data: dict, output_dir: Path):
    """Create a human-readable summary report."""
    
    report = []
    report.append("WINDOWED PATH ANALYSIS - COMPLETE REPORT")
    report.append("=" * 50)
    report.append("")
    
    for window_name in ['early', 'middle', 'late']:
        window_data = data[window_name]
        report.append(f"\n{window_name.upper()} WINDOW ({window_data['window_info']['layer_range']})")
        report.append("-" * 30)
        
        summary = window_data['summary']
        report.append(f"Total words analyzed: {summary['total_words']}")
        report.append(f"Unique paths: {summary['total_unique_paths']}")
        report.append(f"  - Singleton paths: {summary['singleton_paths']} ({summary['singleton_paths']/summary['total_unique_paths']*100:.1f}%)")
        report.append(f"  - Rare paths (<3): {summary['rare_paths_lt3']} ({summary['rare_paths_lt3']/summary['total_unique_paths']*100:.1f}%)")
        report.append(f"  - Common paths (>=10): {summary['common_paths_gte10']}")
        
        metrics = window_data['metrics']
        report.append(f"\nMetrics:")
        report.append(f"  - Fragmentation: {metrics['fragmentation']}")
        report.append(f"  - Stability: {metrics['stability']}")
        report.append(f"  - Avg clusters/layer: {metrics['avg_clusters_per_layer']}")
        
        report.append(f"\nTop 5 paths:")
        for i, path in enumerate(window_data['all_paths'][:5]):
            report.append(f"  {i+1}. {path['path']} - {path['frequency']} words ({path['percentage']}%)")
            report.append(f"     Examples: {', '.join(path['example_words'][:5])}")
    
    # Cross-window transitions
    report.append("\n\nCROSS-WINDOW TRANSITIONS")
    report.append("-" * 30)
    
    for trans_type, trans_data in data['cross_window_transitions'].items():
        report.append(f"\n{trans_type.replace('_', ' ').title()}:")
        report.append(f"  Paths: {trans_data['from_paths']} -> {trans_data['to_paths']}")
        report.append(f"  Convergence ratio: {trans_data['convergence_ratio']}")
        report.append(f"  Top transitions:")
        for i, trans in enumerate(trans_data['top_transitions'][:3]):
            report.append(f"    {trans['from']} -> {trans['to']}: {trans['count']} ({trans['percentage']}%)")
    
    # Save report
    with open(output_dir / "windowed_paths_report.txt", 'w') as f:
        f.write('\n'.join(report))
    
    print("\nSummary report created: windowed_paths_report.txt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use the most complete results
        results_dir = "results/unified_cta_config/unified_cta_20250524_073316"
    
    # Extract all paths
    all_paths = extract_all_paths(results_dir)