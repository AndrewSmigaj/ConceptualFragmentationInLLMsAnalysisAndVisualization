"""
Windowed path analysis for handling fragmentation in trajectories.

Extends existing PathAnalyzer to analyze paths in three windows:
- Early layers (0-3)
- Middle layers (4-7)  
- Late layers (8-11)
"""

import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import json
from collections import defaultdict

from paths.path_analysis import PathAnalyzer
from logging_config import setup_logging

logger = setup_logging(__name__)


class WindowedPathAnalyzer(PathAnalyzer):
    """
    Extends PathAnalyzer to handle windowed analysis of trajectories.
    
    This addresses fragmentation by analyzing paths in layer windows
    rather than requiring full L0->L11 paths.
    """
    
    def __init__(self):
        """Initialize windowed path analyzer."""
        super().__init__()
        self.windows = {
            'early': (0, 3),   # L0-L3
            'middle': (4, 7),  # L4-L7
            'late': (8, 11)    # L8-L11
        }
        logger.info("Initialized WindowedPathAnalyzer with windows: early(0-3), middle(4-7), late(8-11)")
    
    def extract_windowed_trajectories(self,
                                    trajectories: Dict[str, List[int]],
                                    window_name: str) -> Dict[str, List[int]]:
        """
        Extract trajectories for a specific window.
        
        Args:
            trajectories: Full word -> trajectory mapping
            window_name: 'early', 'middle', or 'late'
            
        Returns:
            Windowed trajectories
        """
        start_layer, end_layer = self.windows[window_name]
        windowed_trajectories = {}
        
        for word, full_traj in trajectories.items():
            # Extract window slice (inclusive of end_layer)
            windowed_traj = full_traj[start_layer:end_layer + 1]
            windowed_trajectories[word] = windowed_traj
        
        logger.info(f"Extracted {len(windowed_trajectories)} trajectories for {window_name} window "
                   f"(layers {start_layer}-{end_layer})")
        
        return windowed_trajectories
    
    def analyze_all_windows(self,
                          trajectories: Dict[str, List[int]],
                          word_subtypes: Optional[Dict[str, str]] = None,
                          min_frequency: int = 3) -> Dict:
        """
        Perform complete windowed analysis on trajectories.
        
        Args:
            trajectories: Full word -> trajectory mapping
            word_subtypes: Optional word -> subtype mapping
            min_frequency: Minimum frequency for archetypal paths
            
        Returns:
            Complete analysis results for all windows
        """
        results = {}
        
        for window_name in ['early', 'middle', 'late']:
            logger.info(f"\nAnalyzing {window_name} window...")
            
            # Extract windowed trajectories
            windowed_trajs = self.extract_windowed_trajectories(trajectories, window_name)
            
            # Run standard analyses on window
            window_results = {
                'window': window_name,
                'layers': self.windows[window_name],
                'path_patterns': self.extract_path_patterns(windowed_trajs),
                'trajectory_metrics': self.calculate_trajectory_metrics(windowed_trajs, word_subtypes),
                'archetypal_paths': self.identify_archetypal_paths(windowed_trajs, min_frequency),
                'layer_transitions': self.analyze_layer_transitions(windowed_trajs)
            }
            
            results[window_name] = window_results
        
        # Add cross-window analysis
        results['cross_window'] = self.analyze_cross_window_patterns(trajectories, results)
        
        return results
    
    def analyze_cross_window_patterns(self,
                                    trajectories: Dict[str, List[int]],
                                    window_results: Dict) -> Dict:
        """
        Analyze patterns across windows.
        
        Args:
            trajectories: Full trajectories
            window_results: Results from each window
            
        Returns:
            Cross-window analysis
        """
        # Track how archetypal paths connect across windows
        path_connections = defaultdict(lambda: defaultdict(int))
        
        for word, full_traj in trajectories.items():
            # Get path in each window
            early_path = tuple(full_traj[0:4])
            middle_path = tuple(full_traj[4:8])
            late_path = tuple(full_traj[8:12])
            
            # Track connections
            path_connections['early_to_middle'][(early_path, middle_path)] += 1
            path_connections['middle_to_late'][(middle_path, late_path)] += 1
        
        # Find common connections between archetypal paths
        archetypal_connections = {}
        
        for connection_type, connections in path_connections.items():
            # Get archetypal paths from relevant windows
            if connection_type == 'early_to_middle':
                from_paths = {tuple(p['path']) for p in window_results['early']['archetypal_paths']}
                to_paths = {tuple(p['path']) for p in window_results['middle']['archetypal_paths']}
            else:  # middle_to_late
                from_paths = {tuple(p['path']) for p in window_results['middle']['archetypal_paths']}
                to_paths = {tuple(p['path']) for p in window_results['late']['archetypal_paths']}
            
            # Find connections between archetypal paths
            archetypal_conns = []
            for (from_path, to_path), count in connections.items():
                if from_path in from_paths and to_path in to_paths and count >= 3:
                    archetypal_conns.append({
                        'from_path': list(from_path),
                        'to_path': list(to_path),
                        'count': count,
                        'percentage': count / len(trajectories) * 100
                    })
            
            archetypal_connections[connection_type] = sorted(
                archetypal_conns, 
                key=lambda x: x['count'], 
                reverse=True
            )[:10]  # Top 10 connections
        
        # Calculate stability across windows
        window_stability = []
        for window_name in ['early', 'middle', 'late']:
            stability = window_results[window_name]['trajectory_metrics']['stability']
            window_stability.append(stability)
        
        return {
            'archetypal_connections': archetypal_connections,
            'stability_progression': window_stability,
            'fragmentation_progression': [
                window_results[w]['trajectory_metrics']['fragmentation'] 
                for w in ['early', 'middle', 'late']
            ]
        }
    
    def prepare_sankey_data(self, 
                          windowed_trajectories: Dict[str, List[int]],
                          window_name: str) -> Dict:
        """
        Prepare data for Sankey diagram visualization of a specific window.
        
        Args:
            windowed_trajectories: Trajectories for this window
            window_name: Name of the window
            
        Returns:
            Data formatted for Sankey diagram
        """
        # Get layer indices for this window
        start_layer, end_layer = self.windows[window_name]
        
        # Build nodes and links
        nodes = []
        links = []
        node_map = {}
        
        # Create nodes for each cluster at each layer
        for layer_offset in range(end_layer - start_layer + 1):
            actual_layer = start_layer + layer_offset
            
            # Get unique clusters at this layer
            clusters_at_layer = set()
            for traj in windowed_trajectories.values():
                if layer_offset < len(traj):
                    clusters_at_layer.add(traj[layer_offset])
            
            for cluster in sorted(clusters_at_layer):
                node_id = f"L{actual_layer}_C{cluster}"
                node_map[node_id] = len(nodes)
                nodes.append({
                    'id': node_id,
                    'label': f"L{actual_layer}-C{cluster}",
                    'layer': actual_layer,
                    'cluster': cluster
                })
        
        # Create links between consecutive layers
        for layer_offset in range(end_layer - start_layer):
            transition_counts = defaultdict(int)
            
            for traj in windowed_trajectories.values():
                if layer_offset + 1 < len(traj):
                    from_cluster = traj[layer_offset]
                    to_cluster = traj[layer_offset + 1]
                    actual_from_layer = start_layer + layer_offset
                    actual_to_layer = actual_from_layer + 1
                    
                    from_id = f"L{actual_from_layer}_C{from_cluster}"
                    to_id = f"L{actual_to_layer}_C{to_cluster}"
                    transition_counts[(from_id, to_id)] += 1
            
            # Add links with counts
            for (from_id, to_id), count in transition_counts.items():
                if from_id in node_map and to_id in node_map:
                    links.append({
                        'source': node_map[from_id],
                        'target': node_map[to_id],
                        'value': count,
                        'from_id': from_id,
                        'to_id': to_id
                    })
        
        return {
            'nodes': nodes,
            'links': links,
            'window': window_name,
            'layers': list(range(start_layer, end_layer + 1)),
            'n_trajectories': len(windowed_trajectories)
        }


def run_windowed_analysis(results_dir: str) -> Dict:
    """
    Run windowed analysis on the most recent results.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        Complete windowed analysis results
    """
    # Load trajectories from most recent run
    trajectories_path = Path(results_dir) / "paths" / "trajectories.pkl"
    
    if not trajectories_path.exists():
        raise FileNotFoundError(f"Trajectories not found at {trajectories_path}")
    
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    trajectories = trajectories_data['trajectories']
    
    # Load word subtypes if available
    word_subtypes = None
    subtypes_path = Path(results_dir).parent.parent.parent / "data" / "word_subtypes.json"
    if subtypes_path.exists():
        with open(subtypes_path, 'r') as f:
            word_subtypes = json.load(f)
    
    # Run windowed analysis
    analyzer = WindowedPathAnalyzer()
    results = analyzer.analyze_all_windows(trajectories, word_subtypes)
    
    # Prepare Sankey data for each window
    sankey_data = {}
    for window_name in ['early', 'middle', 'late']:
        windowed_trajs = analyzer.extract_windowed_trajectories(trajectories, window_name)
        sankey_data[window_name] = analyzer.prepare_sankey_data(windowed_trajs, window_name)
    
    results['sankey_data'] = sankey_data
    
    # Save results
    output_dir = Path(results_dir) / "windowed_analysis"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "windowed_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(output_dir / "windowed_analysis_full.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Windowed analysis complete. Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    # Example usage - find most recent results
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find most recent results
        results_base = Path("results/unified_cta_config")
        if results_base.exists():
            result_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()], 
                               key=lambda x: x.name, reverse=True)
            if result_dirs:
                results_dir = result_dirs[0]
                logger.info(f"Using most recent results: {results_dir}")
            else:
                raise ValueError("No results found")
        else:
            raise ValueError("Results directory not found")
    
    results = run_windowed_analysis(str(results_dir))
    
    # Print summary
    print("\n=== Windowed Analysis Summary ===")
    for window_name in ['early', 'middle', 'late']:
        window_data = results[window_name]
        print(f"\n{window_name.upper()} Window (Layers {window_data['layers'][0]}-{window_data['layers'][1]}):")
        print(f"  - Unique paths: {window_data['path_patterns']['n_unique_paths']}")
        print(f"  - Archetypal paths: {len(window_data['archetypal_paths'])}")
        print(f"  - Fragmentation: {window_data['trajectory_metrics']['fragmentation']:.3f}")
        print(f"  - Stability: {window_data['trajectory_metrics']['stability']:.3f}")
        
        if window_data['archetypal_paths']:
            print(f"  - Top archetypal path: {window_data['archetypal_paths'][0]['path']} "
                  f"({window_data['archetypal_paths'][0]['frequency']} occurrences)")