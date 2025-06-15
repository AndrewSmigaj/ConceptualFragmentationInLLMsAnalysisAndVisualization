"""
Analyze how WordNet hypernym hierarchies relate to GPT-2 clustering

This analysis focuses on using the full hypernym chains to understand
semantic organization in GPT-2.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import wordnet as wn

def load_wordnet_features():
    """Load the saved WordNet features"""
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    with open(output_dir / "wordnet_features.json", 'r') as f:
        return json.load(f)

def analyze_hypernym_patterns(word_features):
    """Analyze hypernym patterns in the dataset"""
    
    print("\n=== Hypernym Analysis ===\n")
    
    # Collect all hypernym chains
    all_chains = []
    hypernym_stats = []
    
    for word in word_features:
        chains = word.get('hypernym_chains', [])
        # hypernym_chains is a list of chains
        for i, chain in enumerate(chains):
            if chain:  # Non-empty chain
                all_chains.append(chain)
                synset_name = word['synsets'][i] if i < len(word.get('synsets', [])) else f'synset_{i}'
                hypernym_stats.append({
                    'word': word['word'],
                    'synset': synset_name,
                    'chain': chain,
                    'depth': len(chain),
                    'primary_pos': word['primary_pos']
                })
    
    print(f"Total hypernym chains: {len(all_chains)}")
    
    # Analyze chain depths
    depths = [len(chain) for chain in all_chains]
    print(f"\nHypernym chain depths:")
    print(f"  Mean: {np.mean(depths):.2f}")
    print(f"  Max: {max(depths)}")
    print(f"  Min: {min(depths)}")
    
    # Find common hypernyms at different levels
    print("\nMost common hypernyms at each level:")
    for level in range(5):  # First 5 levels
        hypernyms_at_level = []
        for chain in all_chains:
            if len(chain) > level:
                hypernyms_at_level.append(chain[level])
        
        if hypernyms_at_level:
            common = Counter(hypernyms_at_level).most_common(5)
            print(f"\n  Level {level} (from word):")
            for hypernym, count in common:
                print(f"    {hypernym}: {count}")
    
    # Find convergence points
    print("\n\nTop convergence points (shared hypernyms):")
    all_hypernyms = []
    for chain in all_chains:
        all_hypernyms.extend(chain)
    
    convergence_points = Counter(all_hypernyms).most_common(10)
    for hypernym, count in convergence_points:
        print(f"  {hypernym}: {count} words")
    
    return hypernym_stats


def calculate_hypernym_distance(chain1, chain2):
    """Calculate semantic distance between two hypernym chains"""
    
    # Find common ancestor
    set1 = set(chain1)
    set2 = set(chain2)
    common = set1 & set2
    
    if not common:
        return float('inf')  # No common ancestor
    
    # Find lowest common ancestor (earliest in both chains)
    for i, h1 in enumerate(chain1):
        if h1 in set2:
            j = chain2.index(h1)
            return i + j  # Combined distance to common ancestor
    
    return float('inf')


def analyze_clustering_by_hypernyms(word_features, activations):
    """Analyze how words with similar hypernym structures cluster"""
    
    print("\n=== Clustering vs Hypernym Structure ===\n")
    
    # Build hypernym profiles for each word
    word_hypernyms = []
    valid_indices = []
    
    for i, word in enumerate(word_features):
        chains = word.get('hypernym_chains', [])
        if chains and len(chains) > 0:
            # Use first chain (primary synset)
            primary_chain = chains[0]
            if primary_chain:
                word_hypernyms.append({
                    'index': i,
                    'word': word['word'],
                    'chain': primary_chain,
                    'depth': len(primary_chain)
                })
                valid_indices.append(i)
    
    print(f"Analyzing {len(word_hypernyms)} words with hypernym chains")
    
    # Cluster final layer
    layer_idx = 11  # Final layer
    layer_acts = activations[:, layer_idx, :]
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = kmeans.fit_predict(layer_acts)
    
    # Analyze hypernym similarity within clusters
    print(f"\nAnalyzing hypernym coherence in final layer clusters:")
    
    for cluster_id in range(5):
        cluster_mask = labels == cluster_id
        cluster_words = [word_hypernyms[j] for j, i in enumerate(valid_indices) if cluster_mask[i]]
        
        if len(cluster_words) < 5:
            continue
        
        print(f"\n  Cluster {cluster_id} ({len(cluster_words)} words):")
        
        # Find most common hypernyms in this cluster
        cluster_hypernyms = []
        for word_data in cluster_words:
            cluster_hypernyms.extend(word_data['chain'])
        
        common_hypernyms = Counter(cluster_hypernyms).most_common(5)
        print("    Top shared hypernyms:")
        for hypernym, count in common_hypernyms:
            coverage = count / len(cluster_words) * 100
            print(f"      {hypernym}: {count} words ({coverage:.1f}% coverage)")
        
        # Calculate average pairwise hypernym distance
        if len(cluster_words) > 1:
            distances = []
            sample_size = min(50, len(cluster_words))
            sample_indices = np.random.choice(len(cluster_words), sample_size, replace=False)
            
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    idx_i = sample_indices[i]
                    idx_j = sample_indices[j]
                    dist = calculate_hypernym_distance(
                        cluster_words[idx_i]['chain'],
                        cluster_words[idx_j]['chain']
                    )
                    if dist != float('inf'):
                        distances.append(dist)
            
            if distances:
                print(f"    Average hypernym distance: {np.mean(distances):.2f}")
        
        # Show example words
        example_words = [w['word'] for w in cluster_words[:5]]
        print(f"    Example words: {', '.join(example_words)}")


def find_hypernym_based_groups(word_features):
    """Find natural semantic groups based on shared hypernyms"""
    
    print("\n=== Natural Semantic Groups from Hypernyms ===\n")
    
    # Group words by shared hypernyms at different levels
    hypernym_groups = defaultdict(list)
    
    for i, word in enumerate(word_features):
        chains = word.get('hypernym_chains', [])
        
        # Process each hypernym chain
        for chain in chains:
            if not chain:  # Skip empty chains
                continue
                
            # For nested lists, flatten if needed
            if isinstance(chain[0], list):
                chain = chain[0]
                
            # Group by hypernyms at different depths
            if len(chain) >= 1:
                hypernym_groups[f"direct_{chain[0]}"].append(i)
            if len(chain) >= 2:
                hypernym_groups[f"level2_{chain[1]}"].append(i)
            if len(chain) >= 3:
                hypernym_groups[f"level3_{chain[2]}"].append(i)
    
    # Find interesting groups
    interesting_groups = []
    
    for group_name, indices in hypernym_groups.items():
        if 10 <= len(indices) <= 100:  # Medium-sized groups
            words = [word_features[i]['word'] for i in indices[:10]]
            interesting_groups.append({
                'name': group_name,
                'size': len(indices),
                'indices': indices,
                'sample_words': words
            })
    
    # Sort by size
    interesting_groups.sort(key=lambda x: x['size'], reverse=True)
    
    print("Interesting semantic groups (10-100 words):")
    for group in interesting_groups[:20]:
        print(f"\n  {group['name']} ({group['size']} words)")
        print(f"    Examples: {', '.join(group['sample_words'])}")
    
    return interesting_groups


def analyze_semantic_coherence_across_layers(word_features, activations, semantic_groups):
    """Track how semantic groups stay together or split across layers"""
    
    print("\n=== Semantic Coherence Across Layers ===\n")
    
    # Analyze a few interesting semantic groups
    for group in semantic_groups[:5]:
        print(f"\nGroup: {group['name']} ({group['size']} words)")
        indices = group['indices']
        
        coherence_scores = []
        
        for layer_idx in range(12):
            # Cluster this layer
            layer_acts = activations[:, layer_idx, :]
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
            labels = kmeans.fit_predict(layer_acts)
            
            # Check how concentrated the group is
            group_labels = [labels[i] for i in indices]
            label_counts = Counter(group_labels)
            
            # Coherence = how much the group stays together
            # (1 - entropy of distribution across clusters)
            total = len(group_labels)
            entropy = -sum((count/total) * np.log(count/total) 
                          for count in label_counts.values() if count > 0)
            max_entropy = np.log(min(5, total))  # Max possible entropy
            coherence = 1 - (entropy / max_entropy) if max_entropy > 0 else 1
            
            coherence_scores.append(coherence)
            
            # Print distribution for key layers
            if layer_idx in [0, 6, 11]:
                dist_str = ', '.join([f"C{c}:{n}" for c, n in label_counts.most_common()])
                print(f"  Layer {layer_idx}: coherence={coherence:.3f}, distribution=[{dist_str}]")
        
        # Plot trajectory
        print(f"  Coherence trajectory: {' -> '.join([f'{c:.2f}' for c in coherence_scores])}")


def main():
    # Load data
    print("Loading data...")
    word_features = load_wordnet_features()
    
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    activations = np.load(output_dir / "activations.npy")
    
    # Analyze hypernym patterns
    hypernym_stats = analyze_hypernym_patterns(word_features)
    
    # Analyze clustering by hypernyms
    analyze_clustering_by_hypernyms(word_features, activations)
    
    # Find natural semantic groups
    semantic_groups = find_hypernym_based_groups(word_features)
    
    # Track semantic coherence
    analyze_semantic_coherence_across_layers(word_features, activations, semantic_groups)
    
    print("\n=== Key Questions We Can Now Answer ===")
    print("1. Do words sharing hypernyms cluster together?")
    print("2. At what layer do semantic categories merge/split?")
    print("3. Do deeper hypernym relationships predict tighter clustering?")
    print("4. Which semantic categories are most stable across layers?")
    print("5. How does hypernym distance relate to activation distance?")


if __name__ == "__main__":
    main()