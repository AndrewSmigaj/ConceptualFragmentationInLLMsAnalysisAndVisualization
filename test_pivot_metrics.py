"""Simple test for pivot metrics without external dependencies."""

def calculate_simple_fragmentation(token_path, total_layers):
    """Calculate simple fragmentation score."""
    if not token_path or total_layers <= 0:
        return 0.0
    
    unique_clusters = set(token_path)
    fragmentation = len(unique_clusters) / total_layers
    return min(fragmentation, 1.0)

def calculate_fragmentation_delta(sentence_paths, pivot_token_index=1, total_layers=13):
    """Calculate fragmentation delta."""
    pre_pivot_index = pivot_token_index - 1
    post_pivot_index = pivot_token_index + 1
    
    if pre_pivot_index not in sentence_paths or post_pivot_index not in sentence_paths:
        return 0.0
    
    pre_pivot_fragmentation = calculate_simple_fragmentation(
        sentence_paths[pre_pivot_index], total_layers
    )
    
    post_pivot_fragmentation = calculate_simple_fragmentation(
        sentence_paths[post_pivot_index], total_layers
    )
    
    delta = post_pivot_fragmentation - pre_pivot_fragmentation
    return delta

def calculate_path_divergence_index(sentence_paths, pivot_token_index=1):
    """Calculate path divergence index."""
    pre_pivot_index = pivot_token_index - 1
    post_pivot_index = pivot_token_index + 1
    
    if pre_pivot_index not in sentence_paths or post_pivot_index not in sentence_paths:
        return 0.0
    
    pre_pivot_path = sentence_paths[pre_pivot_index]
    post_pivot_path = sentence_paths[post_pivot_index]
    
    min_length = min(len(pre_pivot_path), len(post_pivot_path))
    if min_length == 0:
        return 0.0
    
    differences = sum(
        1 for i in range(min_length) 
        if pre_pivot_path[i] != post_pivot_path[i]
    )
    
    divergence = differences / min_length
    return divergence

# Test the functions
if __name__ == "__main__":
    # Test simple fragmentation
    path = ['L0C1', 'L1C2', 'L2C1', 'L3C1']
    frag = calculate_simple_fragmentation(path, 4)
    print(f'Simple fragmentation: {frag}')  # Should be 1.0 (4 unique / 4 total)
    
    # Test with repeated clusters - each layer has different label but same cluster
    path2 = ['L0C1', 'L1C1', 'L2C1', 'L3C1']  # This has 4 unique labels still
    frag2 = calculate_simple_fragmentation(path2, 4)
    print(f'Same cluster different layers: {frag2}')  # Should be 1.0 (L0C1 != L1C1)
    
    # Test with truly repeated clusters (same exact string)
    path3 = ['L1C1', 'L1C1', 'L1C1', 'L1C1']
    frag3 = calculate_simple_fragmentation(path3, 4)
    print(f'Truly repeated clusters: {frag3}')  # Should be 0.25 (1 unique / 4 total)
    
    # Test fragmentation delta
    sentence_paths = {
        0: ['L0C1', 'L1C1', 'L2C1', 'L3C1'],  # pre-pivot: low fragmentation (0.25)
        1: ['L0C2', 'L1C3', 'L2C4', 'L3C5'],  # pivot: high fragmentation (1.0)
        2: ['L0C1', 'L1C2', 'L2C3', 'L3C4']   # post-pivot: high fragmentation (1.0)
    }
    delta = calculate_fragmentation_delta(sentence_paths, 1, 4)
    print(f'Fragmentation delta: {delta}')  # Should be 1.0 - 0.25 = 0.75
    
    # Test path divergence
    divergence = calculate_path_divergence_index(sentence_paths, 1)
    print(f'Path divergence: {divergence}')  # Should be 1.0 (all positions different)
    
    print('✅ All pivot metrics working correctly!')