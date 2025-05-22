"""
Pivot-specific metrics for GPT-2 APA analysis.

Contains metrics for analyzing semantic pivot effects in transformer models,
specifically fragmentation delta and path divergence index.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_simple_fragmentation(token_path: List[str], total_layers: int) -> float:
    """
    Calculate simple fragmentation score for a single token's path.
    
    Simple fragmentation = unique cluster IDs / total layers
    
    Args:
        token_path: List of cluster IDs for a token across layers (e.g., ['L0C1', 'L1C2', 'L2C1'])
        total_layers: Total number of layers analyzed
        
    Returns:
        Fragmentation score between 0 and 1
        
    Example:
        token_path = ['L0C1', 'L1C2', 'L2C1', 'L3C1'] 
        total_layers = 4
        unique_clusters = {'L0C1', 'L1C2', 'L2C1', 'L3C1'} = 4 unique
        fragmentation = 4 / 4 = 1.0
    """
    if not token_path or total_layers <= 0:
        return 0.0
    
    unique_clusters = set(token_path)
    fragmentation = len(unique_clusters) / total_layers
    
    return min(fragmentation, 1.0)  # Cap at 1.0


def calculate_fragmentation_delta(
    sentence_paths: Dict[int, List[str]], 
    pivot_token_index: int = 1,
    total_layers: int = 13
) -> float:
    """
    Calculate fragmentation delta for a sentence.
    
    Fragmentation delta = post-pivot fragmentation - pre-pivot fragmentation
    
    For 3-token sentences: "good but bad"
    - pre-pivot token: index 0 ("good")
    - pivot token: index 1 ("but") 
    - post-pivot token: index 2 ("bad")
    
    Args:
        sentence_paths: Dict mapping token index to list of cluster IDs across layers
                       e.g., {0: ['L0C1', 'L1C1', ...], 1: ['L0C2', 'L1C3', ...], 2: ['L0C1', 'L1C4', ...]}
        pivot_token_index: Index of the pivot token (default 1 for "but")
        total_layers: Total number of layers analyzed
        
    Returns:
        Fragmentation delta (post-pivot fragmentation - pre-pivot fragmentation)
        
    Example:
        pre-pivot fragmentation = 0.3 (token stays in similar clusters)
        post-pivot fragmentation = 0.8 (token jumps between many clusters)
        delta = 0.8 - 0.3 = 0.5
    """
    pre_pivot_index = pivot_token_index - 1
    post_pivot_index = pivot_token_index + 1
    
    # Check that we have the required token indices
    if pre_pivot_index not in sentence_paths or post_pivot_index not in sentence_paths:
        logger.warning(f"Missing token paths for pre-pivot {pre_pivot_index} or post-pivot {post_pivot_index}")
        return 0.0
    
    # Calculate fragmentation for each token
    pre_pivot_fragmentation = calculate_simple_fragmentation(
        sentence_paths[pre_pivot_index], total_layers
    )
    
    post_pivot_fragmentation = calculate_simple_fragmentation(
        sentence_paths[post_pivot_index], total_layers
    )
    
    delta = post_pivot_fragmentation - pre_pivot_fragmentation
    
    return delta


def calculate_path_divergence_index(
    sentence_paths: Dict[int, List[str]], 
    pivot_token_index: int = 1
) -> float:
    """
    Calculate path divergence index using Hamming distance.
    
    Measures how different the pre-pivot and post-pivot token paths are.
    Uses Hamming distance: number of positions where cluster IDs differ.
    
    Args:
        sentence_paths: Dict mapping token index to list of cluster IDs across layers
        pivot_token_index: Index of the pivot token (default 1 for "but")
        
    Returns:
        Path divergence index (normalized Hamming distance between 0 and 1)
        
    Example:
        pre-pivot path:  ['L0C1', 'L1C1', 'L2C2', 'L3C1']
        post-pivot path: ['L0C1', 'L1C3', 'L2C2', 'L3C4'] 
        differences at positions: [False, True, False, True] = 2 differences
        divergence = 2 / 4 = 0.5
    """
    pre_pivot_index = pivot_token_index - 1
    post_pivot_index = pivot_token_index + 1
    
    # Check that we have the required token indices
    if pre_pivot_index not in sentence_paths or post_pivot_index not in sentence_paths:
        logger.warning(f"Missing token paths for pre-pivot {pre_pivot_index} or post-pivot {post_pivot_index}")
        return 0.0
    
    pre_pivot_path = sentence_paths[pre_pivot_index]
    post_pivot_path = sentence_paths[post_pivot_index]
    
    # Ensure paths are same length
    min_length = min(len(pre_pivot_path), len(post_pivot_path))
    if min_length == 0:
        return 0.0
    
    # Calculate Hamming distance
    differences = sum(
        1 for i in range(min_length) 
        if pre_pivot_path[i] != post_pivot_path[i]
    )
    
    divergence = differences / min_length
    
    return divergence


def analyze_pivot_effects(
    all_sentence_paths: List[Dict[int, List[str]]],
    sentence_labels: List[str],
    pivot_token_index: int = 1,
    total_layers: int = 13
) -> Dict[str, Any]:
    """
    Analyze pivot effects across all sentences.
    
    Args:
        all_sentence_paths: List of sentence path dictionaries
        sentence_labels: List of labels ('contrast' or 'consistent') for each sentence
        pivot_token_index: Index of the pivot token
        total_layers: Total number of layers analyzed
        
    Returns:
        Dictionary containing analysis results
    """
    results = {
        'fragmentation_deltas': [],
        'path_divergence_indices': [],
        'sentence_labels': sentence_labels,
        'contrast_stats': {},
        'consistent_stats': {}
    }
    
    # Calculate metrics for each sentence
    for sentence_paths in all_sentence_paths:
        delta = calculate_fragmentation_delta(sentence_paths, pivot_token_index, total_layers)
        divergence = calculate_path_divergence_index(sentence_paths, pivot_token_index)
        
        results['fragmentation_deltas'].append(delta)
        results['path_divergence_indices'].append(divergence)
    
    # Calculate statistics by class
    contrast_deltas = [delta for delta, label in zip(results['fragmentation_deltas'], sentence_labels) if label == 'contrast']
    consistent_deltas = [delta for delta, label in zip(results['fragmentation_deltas'], sentence_labels) if label == 'consistent']
    
    contrast_divergences = [div for div, label in zip(results['path_divergence_indices'], sentence_labels) if label == 'contrast']
    consistent_divergences = [div for div, label in zip(results['path_divergence_indices'], sentence_labels) if label == 'consistent']
    
    results['contrast_stats'] = {
        'mean_fragmentation_delta': np.mean(contrast_deltas) if contrast_deltas else 0.0,
        'mean_path_divergence': np.mean(contrast_divergences) if contrast_divergences else 0.0,
        'count': len(contrast_deltas)
    }
    
    results['consistent_stats'] = {
        'mean_fragmentation_delta': np.mean(consistent_deltas) if consistent_deltas else 0.0,
        'mean_path_divergence': np.mean(consistent_divergences) if consistent_divergences else 0.0,
        'count': len(consistent_deltas)
    }
    
    return results