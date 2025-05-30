"""Path extraction and analysis utilities."""

from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PathExtractor:
    """Extract and analyze token paths through neural network layers.
    
    This class provides utilities for extracting token trajectories through
    clustered activation spaces and finding archetypal (most common) paths.
    """
    
    def __init__(self):
        """Initialize the path extractor."""
        self.paths = []
        self.path_counts = Counter()
        
    def extract_paths(self, 
                     cluster_labels: Dict[str, np.ndarray],
                     tokens: List[Dict[str, Any]],
                     layers: List[int]) -> List[Dict[str, Any]]:
        """Extract token paths through specified layers.
        
        Args:
            cluster_labels: Dictionary mapping layer indices (as strings) to 
                          cluster assignments for each token
            tokens: List of token information dictionaries
            layers: List of layer indices to extract paths through
            
        Returns:
            List of path dictionaries containing:
                - token: The token string
                - path: List of cluster IDs
                - token_id: Original token ID
        """
        paths = []
        path_tuples = []
        
        # Extract path for each token
        for idx, token_info in enumerate(tokens):
            path = []
            
            # Get cluster assignment at each layer
            for layer in layers:
                layer_key = str(layer)
                if layer_key in cluster_labels and idx < len(cluster_labels[layer_key]):
                    cluster_id = int(cluster_labels[layer_key][idx])
                    path.append(cluster_id)
                else:
                    # Skip incomplete paths
                    break
                    
            # Only keep complete paths
            if len(path) == len(layers):
                path_tuple = tuple(path)
                path_tuples.append(path_tuple)
                
                paths.append({
                    'token': token_info.get('token_str', '').strip(),
                    'token_id': token_info.get('token_id', idx),
                    'path': path
                })
                
        # Count path frequencies
        self.path_counts = Counter(path_tuples)
        self.paths = paths
        
        logger.info(f"Extracted {len(paths)} complete paths through {len(layers)} layers")
        logger.info(f"Found {len(self.path_counts)} unique paths")
        
        return paths
    
    def find_archetypal_paths(self,
                             top_n: int = 25,
                             min_frequency: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find the most common (archetypal) paths.
        
        Args:
            top_n: Number of top paths to return
            min_frequency: Minimum frequency threshold (optional)
            
        Returns:
            List of archetypal path dictionaries containing:
                - path: The cluster sequence
                - frequency: Number of tokens following this path
                - representative_words: Example tokens following this path
                - percentage: Percentage of all tokens
        """
        if not self.path_counts:
            logger.warning("No paths extracted yet. Call extract_paths first.")
            return []
            
        # Get most common paths
        if min_frequency:
            common_paths = [
                (path, count) for path, count in self.path_counts.most_common()
                if count >= min_frequency
            ][:top_n]
        else:
            common_paths = self.path_counts.most_common(top_n)
            
        # Find representative tokens for each path
        archetypal_paths = []
        total_tokens = len(self.paths)
        
        for path_tuple, frequency in common_paths:
            # Find tokens that follow this path
            representative_tokens = []
            for path_info in self.paths:
                if tuple(path_info['path']) == path_tuple:
                    representative_tokens.append(path_info['token'])
                    if len(representative_tokens) >= 10:  # Limit examples
                        break
                        
            archetypal_paths.append({
                'path': list(path_tuple),
                'frequency': frequency,
                'representative_words': representative_tokens[:5],
                'example_words': representative_tokens[:5],  # Backward compatibility
                'percentage': (frequency / total_tokens) * 100 if total_tokens > 0 else 0
            })
            
        logger.info(f"Found {len(archetypal_paths)} archetypal paths")
        
        return archetypal_paths
    
    def get_path_transitions(self) -> Dict[Tuple[int, int], Counter]:
        """Analyze transitions between clusters across layers.
        
        Returns:
            Dictionary mapping (layer_i, layer_i+1) to Counter of 
            (cluster_from, cluster_to) transitions
        """
        transitions = defaultdict(Counter)
        
        for path_info in self.paths:
            path = path_info['path']
            for i in range(len(path) - 1):
                transition = (path[i], path[i + 1])
                layer_pair = (i, i + 1)
                transitions[layer_pair][transition] += 1
                
        return dict(transitions)
    
    def get_path_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about extracted paths.
        
        Returns:
            Dictionary containing:
                - total_paths: Total number of tokens with complete paths
                - unique_paths: Number of unique paths
                - most_common_path: The most frequent path
                - path_entropy: Shannon entropy of path distribution
        """
        if not self.path_counts:
            return {
                'total_paths': 0,
                'unique_paths': 0,
                'most_common_path': None,
                'path_entropy': 0.0
            }
            
        total = sum(self.path_counts.values())
        
        # Calculate entropy
        entropy = 0.0
        for count in self.path_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
                
        most_common = self.path_counts.most_common(1)[0] if self.path_counts else (None, 0)
        
        return {
            'total_paths': total,
            'unique_paths': len(self.path_counts),
            'most_common_path': list(most_common[0]) if most_common[0] else None,
            'most_common_frequency': most_common[1],
            'path_entropy': entropy
        }