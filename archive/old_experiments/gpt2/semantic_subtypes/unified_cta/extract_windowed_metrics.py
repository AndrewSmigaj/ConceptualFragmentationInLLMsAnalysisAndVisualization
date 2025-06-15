"""
Extract windowed metrics and archetypal paths from existing results.

This script focuses on extracting the key metrics needed for the paper
from the windowed analysis, handling serialization issues.
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


def extract_windowed_metrics(results_dir: str):
    """Extract key metrics for each window."""
    
    # Load trajectories
    trajectories_path = Path(results_dir) / "paths" / "trajectories.pkl"
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    trajectories = trajectories_data['trajectories']
    
    # Initialize analyzer
    analyzer = WindowedPathAnalyzer()
    
    # Results storage
    window_metrics = {}
    
    for window_name in ['early', 'middle', 'late']:
        print(f"\n=== {window_name.upper()} WINDOW ===")
        
        # Extract windowed trajectories
        windowed_trajs = analyzer.extract_windowed_trajectories(trajectories, window_name)
        
        # Get archetypal paths
        archetypal_paths = analyzer.identify_archetypal_paths(windowed_trajs, min_frequency=3)
        
        # Calculate basic metrics
        path_counts = Counter(tuple(traj) for traj in windowed_trajs.values())
        unique_paths = len(path_counts)
        total_trajs = len(windowed_trajs)
        
        # Get trajectory metrics
        traj_metrics = analyzer.calculate_trajectory_metrics(windowed_trajs)
        
        # Store results
        window_metrics[window_name] = {
            'layers': analyzer.windows[window_name],
            'total_trajectories': total_trajs,
            'unique_paths': unique_paths,
            'n_archetypal_paths': len(archetypal_paths),
            'fragmentation': traj_metrics['fragmentation'],
            'stability': traj_metrics['stability'],
            'convergence': traj_metrics['convergence'],
            'top_5_paths': []
        }
        
        # Add top 5 archetypal paths
        for i, path_info in enumerate(archetypal_paths[:5]):
            window_metrics[window_name]['top_5_paths'].append({
                'rank': i + 1,
                'path': path_info['path'],
                'frequency': path_info['frequency'],
                'percentage': round(path_info['percentage'], 2),
                'example_words': path_info['example_words'][:5],
                'stability': round(path_info['stability'], 3)
            })
        
        # Print summary
        print(f"Layers: {analyzer.windows[window_name][0]}-{analyzer.windows[window_name][1]}")
        print(f"Total trajectories: {total_trajs}")
        print(f"Unique paths: {unique_paths}")
        print(f"Archetypal paths (freq >= 3): {len(archetypal_paths)}")
        print(f"Fragmentation: {traj_metrics['fragmentation']:.3f}")
        print(f"Stability: {traj_metrics['stability']:.3f}")
        print(f"Convergence: {traj_metrics['convergence']:.3f}")
        
        if archetypal_paths:
            print(f"\nTop archetypal path:")
            top_path = archetypal_paths[0]
            print(f"  Path: {top_path['path']}")
            print(f"  Frequency: {top_path['frequency']} ({top_path['percentage']:.1f}%)")
            print(f"  Examples: {', '.join(top_path['example_words'][:5])}")
    
    # Calculate cross-window connections
    print("\n=== CROSS-WINDOW CONNECTIONS ===")
    
    # Track paths across windows
    path_connections = {
        'early_to_middle': defaultdict(int),
        'middle_to_late': defaultdict(int)
    }
    
    for word, full_traj in trajectories.items():
        early_path = tuple(full_traj[0:4])
        middle_path = tuple(full_traj[4:8])
        late_path = tuple(full_traj[8:12])
        
        path_connections['early_to_middle'][(early_path, middle_path)] += 1
        path_connections['middle_to_late'][(middle_path, late_path)] += 1
    
    # Find strong connections between archetypal paths
    for connection_type in ['early_to_middle', 'middle_to_late']:
        print(f"\n{connection_type.replace('_', ' ').title()}:")
        
        # Get archetypal paths for relevant windows
        if connection_type == 'early_to_middle':
            from_paths = {tuple(p['path']) for p in window_metrics['early']['top_5_paths']}
            to_paths = {tuple(p['path']) for p in window_metrics['middle']['top_5_paths']}
        else:
            from_paths = {tuple(p['path']) for p in window_metrics['middle']['top_5_paths']}
            to_paths = {tuple(p['path']) for p in window_metrics['late']['top_5_paths']}
        
        # Find connections
        strong_connections = []
        for (from_path, to_path), count in path_connections[connection_type].items():
            if from_path in from_paths and to_path in to_paths and count >= 3:
                strong_connections.append({
                    'from': list(from_path),
                    'to': list(to_path),
                    'count': count,
                    'percentage': count / len(trajectories) * 100
                })
        
        # Sort and display
        strong_connections.sort(key=lambda x: x['count'], reverse=True)
        for conn in strong_connections[:3]:
            print(f"  {conn['from']} -> {conn['to']}: {conn['count']} ({conn['percentage']:.1f}%)")
    
    # Save results
    output_dir = Path(results_dir) / "windowed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "windowed_metrics_summary.json", 'w') as f:
        json.dump(window_metrics, f, indent=2)
    
    print(f"\n\nResults saved to {output_dir}")
    
    return window_metrics


def prepare_sankey_data(results_dir: str, window_name: str):
    """Prepare data for a single window's Sankey diagram."""
    
    # Load trajectories
    trajectories_path = Path(results_dir) / "paths" / "trajectories.pkl"
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    trajectories = trajectories_data['trajectories']
    
    # Initialize analyzer
    analyzer = WindowedPathAnalyzer()
    
    # Extract windowed trajectories
    windowed_trajs = analyzer.extract_windowed_trajectories(trajectories, window_name)
    
    # Prepare Sankey data
    sankey_data = analyzer.prepare_sankey_data(windowed_trajs, window_name)
    
    # Save
    output_dir = Path(results_dir) / "windowed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / f"sankey_{window_name}.json", 'w') as f:
        json.dump(sankey_data, f, indent=2)
    
    print(f"Sankey data for {window_name} window saved")
    
    return sankey_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use the most complete results
        results_dir = "results/unified_cta_config/unified_cta_20250524_073316"
    
    # Extract metrics
    metrics = extract_windowed_metrics(results_dir)
    
    # Prepare Sankey data for each window
    for window in ['early', 'middle', 'late']:
        prepare_sankey_data(results_dir, window)