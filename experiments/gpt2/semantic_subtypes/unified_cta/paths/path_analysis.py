"""
Path analysis for trajectory construction and metrics calculation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import sys
from pathlib import Path

# Add parent directories to path for imports
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import existing metrics - proper exploration of what's available
from concept_fragmentation.metrics import (
    compute_representation_stability,
    compute_layer_stability_profile,
    compute_entropy_fragmentation_score
)
from concept_fragmentation.utils.cluster_paths import (
    build_cluster_paths,
    find_representative_sample
)

from logging_config import setup_logging

logger = setup_logging(__name__)


class PathAnalyzer:
    """
    Constructs and analyzes token trajectories through layers.
    
    Uses existing path utilities and metrics without reimplementation.
    """
    
    def __init__(self):
        """Initialize path analyzer."""
        logger.info("Initialized PathAnalyzer")
    
    def construct_trajectories(self,
                             macro_labels: Dict[int, np.ndarray],
                             word_list: List[str]) -> Dict[str, List[int]]:
        """
        Construct trajectories for each word across layers.
        
        Args:
            macro_labels: Layer -> cluster labels mapping
            word_list: List of words in order
            
        Returns:
            Dictionary mapping word to its trajectory
        """
        # Convert to format expected by build_cluster_paths
        layer_cluster_labels = {}
        for layer in sorted(macro_labels.keys()):
            layer_name = f"layer_{layer}"
            layer_cluster_labels[layer_name] = macro_labels[layer].tolist()
        
        ordered_layers = [f"layer_{i}" for i in sorted(macro_labels.keys())]
        
        # Use existing function to build paths
        path_strings = build_cluster_paths(layer_cluster_labels, ordered_layers)
        
        # Convert path strings to trajectories
        trajectories = {}
        for idx, (word, path_str) in enumerate(zip(word_list, path_strings)):
            # Parse "0→1→2→3" format back to list
            trajectory = [int(x) for x in path_str.split("→")]
            trajectories[word] = trajectory
        
        logger.info(f"Constructed {len(trajectories)} trajectories across {len(macro_labels)} layers")
        return trajectories
    
    def extract_path_patterns(self, trajectories: Dict[str, List[int]]) -> Dict:
        """
        Extract unique paths and calculate diversity metrics.
        
        Args:
            trajectories: Word -> trajectory mapping
            
        Returns:
            Dictionary with path patterns and metrics
        """
        # Convert to matrix for analysis
        trajectory_matrix = np.array(list(trajectories.values()))
        n_words, n_layers = trajectory_matrix.shape
        
        # Extract unique paths
        unique_paths = []
        seen_paths = set()
        for trajectory in trajectories.values():
            path_tuple = tuple(trajectory)
            if path_tuple not in seen_paths:
                unique_paths.append(trajectory)
                seen_paths.add(path_tuple)
        
        # Calculate diversity metrics - transition diversity between layers
        diversity_scores = []
        for i in range(n_layers - 1):
            # Count unique transitions from layer i to i+1
            transitions = set(zip(trajectory_matrix[:, i], trajectory_matrix[:, i+1]))
            # Normalize by max possible transitions
            from_clusters = len(set(trajectory_matrix[:, i]))
            to_clusters = len(set(trajectory_matrix[:, i+1]))
            max_transitions = from_clusters * to_clusters
            diversity = len(transitions) / max_transitions if max_transitions > 0 else 0
            diversity_scores.append(diversity)
        
        # Group words by path
        path_to_words = defaultdict(list)
        for word, trajectory in trajectories.items():
            path_key = tuple(trajectory)
            path_to_words[path_key].append(word)
        
        # Calculate path frequencies
        path_counts = Counter(tuple(traj) for traj in trajectories.values())
        
        results = {
            'n_unique_paths': len(unique_paths),
            'total_trajectories': len(trajectories),
            'path_diversity_by_layer': diversity_scores,
            'mean_diversity': np.mean(diversity_scores) if diversity_scores else 0,
            'path_frequencies': dict(path_counts.most_common(20)),  # Top 20
            'path_to_words': {str(k): v[:5] for k, v in path_to_words.items()},  # Sample words
            'unique_paths': unique_paths
        }
        
        logger.info(f"Extracted {len(unique_paths)} unique paths from {len(trajectories)} trajectories")
        return results
    
    def calculate_trajectory_metrics(self,
                                   trajectories: Dict[str, List[int]],
                                   word_subtypes: Optional[Dict[str, str]] = None) -> Dict:
        """
        Calculate comprehensive trajectory metrics using existing implementations.
        
        Args:
            trajectories: Word -> trajectory mapping
            word_subtypes: Optional word -> subtype mapping
            
        Returns:
            Dictionary of trajectory metrics
        """
        # Convert to matrix format
        trajectory_matrix = np.array(list(trajectories.values()))
        word_list = list(trajectories.keys())
        n_words, n_layers = trajectory_matrix.shape
        
        # Calculate fragmentation - use entropy fragmentation score
        # This measures how spread out the clusters are
        layer_entropies = []
        for layer_idx in range(n_layers):
            layer_labels = trajectory_matrix[:, layer_idx]
            unique_labels, counts = np.unique(layer_labels, return_counts=True)
            # Normalize to get probabilities
            probs = counts / counts.sum()
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            layer_entropies.append(entropy)
        
        # Fragmentation increases with entropy
        fragmentation = np.mean(layer_entropies) / np.log(n_words)  # Normalize by max entropy
        
        # Calculate stability - how often tokens stay in same cluster
        stability_scores = []
        for i in range(n_layers - 1):
            same_cluster = np.sum(trajectory_matrix[:, i] == trajectory_matrix[:, i+1])
            stability_scores.append(same_cluster / n_words)
        stability = np.mean(stability_scores)
        
        # Calculate convergence - do paths converge to fewer clusters?
        unique_clusters_per_layer = []
        for layer_idx in range(n_layers):
            n_unique = len(np.unique(trajectory_matrix[:, layer_idx]))
            unique_clusters_per_layer.append(n_unique)
        
        # Convergence: reduction in clusters from early to late layers
        if len(unique_clusters_per_layer) > 1:
            early_clusters = np.mean(unique_clusters_per_layer[:n_layers//3])
            late_clusters = np.mean(unique_clusters_per_layer[-n_layers//3:])
            convergence = max(0, (early_clusters - late_clusters) / early_clusters)
        else:
            convergence = 0.0
        
        # Calculate semantic coherence if subtypes provided
        semantic_coherence = None
        if word_subtypes:
            coherence_scores = []
            
            # Group trajectories by subtype
            subtype_trajectories = defaultdict(list)
            for word, traj in trajectories.items():
                if word in word_subtypes:
                    subtype_trajectories[word_subtypes[word]].append(traj)
            
            # Calculate within-subtype path similarity
            for subtype, trajs in subtype_trajectories.items():
                if len(trajs) > 1:
                    # Count how many words of same subtype follow same path
                    path_counts = Counter(tuple(t) for t in trajs)
                    # Coherence = fraction following most common path
                    most_common_count = path_counts.most_common(1)[0][1]
                    coherence = most_common_count / len(trajs)
                    coherence_scores.append(coherence)
            
            semantic_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Layer transition analysis
        transition_counts = defaultdict(lambda: defaultdict(int))
        
        for trajectory in trajectory_matrix:
            for i in range(n_layers - 1):
                from_cluster = trajectory[i]
                to_cluster = trajectory[i + 1]
                transition_counts[i][(from_cluster, to_cluster)] += 1
        
        # Convert to regular dict for serialization
        transition_counts = {
            layer: dict(transitions) 
            for layer, transitions in transition_counts.items()
        }
        
        metrics = {
            'fragmentation': float(fragmentation),
            'stability': float(stability),
            'convergence': float(convergence),
            'semantic_coherence': semantic_coherence,
            'n_trajectories': len(trajectories),
            'n_layers': n_layers,
            'transition_counts': transition_counts,
            'avg_transitions_per_layer': {
                layer: len(transitions) 
                for layer, transitions in transition_counts.items()
            },
            'unique_clusters_per_layer': unique_clusters_per_layer,
            'layer_entropies': layer_entropies
        }
        
        logger.info(f"Calculated trajectory metrics: fragmentation={fragmentation:.3f}, "
                   f"stability={stability:.3f}, convergence={convergence:.3f}")
        
        return metrics
    
    def identify_archetypal_paths(self,
                                trajectories: Dict[str, List[int]],
                                min_frequency: int = 3) -> List[Dict]:
        """
        Identify archetypal (frequently occurring) paths.
        
        Args:
            trajectories: Word -> trajectory mapping
            min_frequency: Minimum occurrences to be considered archetypal
            
        Returns:
            List of archetypal paths with metadata
        """
        # Count path frequencies
        path_counts = Counter(tuple(traj) for traj in trajectories.values())
        
        # Get words for each path
        path_to_words = defaultdict(list)
        for word, trajectory in trajectories.items():
            path_key = tuple(trajectory)
            path_to_words[path_key].append(word)
        
        # Identify archetypal paths
        archetypal_paths = []
        
        for path, count in path_counts.items():
            if count >= min_frequency:
                # Calculate path characteristics
                words = path_to_words[path]
                
                # Path stability (how many clusters stay same)
                stability = sum(1 for i in range(len(path)-1) if path[i] == path[i+1]) / (len(path)-1)
                
                # Path range (how many different clusters visited)
                n_unique = len(set(path))
                
                archetypal_paths.append({
                    'path': list(path),
                    'frequency': count,
                    'percentage': count / len(trajectories) * 100,
                    'example_words': words[:10],  # First 10 examples
                    'stability': stability,
                    'n_unique_clusters': n_unique,
                    'is_stable': stability > 0.7,  # Mostly stays in same cluster
                    'is_wandering': n_unique > len(path) * 0.6  # Visits many clusters
                })
        
        # Sort by frequency
        archetypal_paths.sort(key=lambda x: x['frequency'], reverse=True)
        
        logger.info(f"Identified {len(archetypal_paths)} archetypal paths "
                   f"(≥{min_frequency} occurrences)")
        
        return archetypal_paths
    
    def analyze_layer_transitions(self,
                                trajectories: Dict[str, List[int]]) -> Dict:
        """
        Analyze cluster transitions between consecutive layers.
        
        Args:
            trajectories: Word -> trajectory mapping
            
        Returns:
            Layer-by-layer transition analysis
        """
        trajectory_matrix = np.array(list(trajectories.values()))
        n_layers = trajectory_matrix.shape[1]
        
        layer_analysis = {}
        
        for layer_idx in range(n_layers - 1):
            # Get transitions for this layer pair
            from_clusters = trajectory_matrix[:, layer_idx]
            to_clusters = trajectory_matrix[:, layer_idx + 1]
            
            # Build transition matrix
            unique_from = np.unique(from_clusters)
            unique_to = np.unique(to_clusters)
            
            transition_matrix = np.zeros((len(unique_from), len(unique_to)))
            
            for i, from_c in enumerate(unique_from):
                for j, to_c in enumerate(unique_to):
                    count = np.sum((from_clusters == from_c) & (to_clusters == to_c))
                    transition_matrix[i, j] = count
            
            # Calculate transition entropy (how spread out are transitions)
            transition_probs = transition_matrix / (transition_matrix.sum() + 1e-10)
            entropy = -np.sum(transition_probs * np.log2(transition_probs + 1e-10))
            
            # Find dominant transitions
            dominant_transitions = []
            for i, from_c in enumerate(unique_from):
                row = transition_matrix[i, :]
                if row.sum() > 0:
                    to_idx = np.argmax(row)
                    to_c = unique_to[to_idx]
                    prob = row[to_idx] / row.sum()
                    if prob > 0.5:  # Dominant if >50% go to same cluster
                        dominant_transitions.append({
                            'from': int(from_c),
                            'to': int(to_c),
                            'probability': float(prob),
                            'count': int(row[to_idx])
                        })
            
            layer_analysis[f'layer_{layer_idx}_to_{layer_idx+1}'] = {
                'n_from_clusters': len(unique_from),
                'n_to_clusters': len(unique_to),
                'transition_entropy': float(entropy),
                'n_dominant_transitions': len(dominant_transitions),
                'dominant_transitions': dominant_transitions[:5],  # Top 5
                'total_transitions': int(transition_matrix.sum())
            }
        
        return layer_analysis