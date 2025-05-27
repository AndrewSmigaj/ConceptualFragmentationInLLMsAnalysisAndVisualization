"""
Fast pattern discovery with higher k values
Focus on finding unexpected groupings
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def quick_clustering_analysis(word_features, activations):
    """Quick analysis with different k values"""
    
    n_words, n_layers, hidden_dim = activations.shape
    
    # Test specific k values
    k_values = [20, 50, 100, 150, 200]
    
    results = {}
    
    for k in k_values:
        print(f"\n=== Testing k={k} ===")
        
        # Focus on specific interesting layers
        for layer_idx in [0, 3, 6, 9, 11]:  # Early, early-mid, mid, late-mid, final
            print(f"\nLayer {layer_idx} with k={k}:")
            
            # Use MiniBatchKMeans for speed
            layer_acts = activations[:, layer_idx, :]
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
            labels = kmeans.fit_predict(layer_acts)
            
            # Analyze cluster sizes
            cluster_sizes = Counter(labels)
            
            # Find interesting patterns
            interesting_clusters = find_interesting_patterns(
                word_features, labels, k, layer_idx
            )
            
            results[f'k{k}_layer{layer_idx}'] = {
                'k': k,
                'layer': layer_idx,
                'cluster_sizes': dict(cluster_sizes),
                'interesting_patterns': interesting_clusters
            }
            
            # Print summary
            print(f"  Cluster size distribution: min={min(cluster_sizes.values())}, "
                  f"max={max(cluster_sizes.values())}, "
                  f"median={np.median(list(cluster_sizes.values()))}")
            print(f"  Found {len(interesting_clusters)} interesting patterns")
    
    return results


def find_interesting_patterns(word_features, labels, k, layer_idx):
    """Find clusters with interesting patterns"""
    
    interesting = []
    
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_words = [word_features[i] for i in range(len(word_features)) if cluster_mask[i]]
        
        if len(cluster_words) < 3:  # Skip tiny clusters
            continue
        
        pattern = analyze_cluster_pattern(cluster_words, cluster_id)
        
        if pattern['interest_score'] > 0:
            pattern['size'] = len(cluster_words)
            pattern['examples'] = [w['word'] for w in cluster_words[:15]]
            interesting.append(pattern)
    
    # Sort by interest score
    interesting.sort(key=lambda x: x['interest_score'], reverse=True)
    
    return interesting[:10]  # Top 10 most interesting


def analyze_cluster_pattern(cluster_words, cluster_id):
    """Analyze what patterns exist in a cluster"""
    
    pattern = {
        'cluster_id': cluster_id,
        'interest_score': 0,
        'patterns': []
    }
    
    # 1. Single POS domination (>90%)
    pos_counts = Counter(w['primary_pos'] for w in cluster_words if w['primary_pos'])
    if pos_counts:
        top_pos, top_count = pos_counts.most_common(1)[0]
        if top_count / len(cluster_words) > 0.9:
            pattern['patterns'].append(f"POS={top_pos} ({top_count}/{len(cluster_words)})")
            pattern['interest_score'] += 0.5
    
    # 2. Extreme polysemy
    polysemy = [w['polysemy_count'] for w in cluster_words]
    avg_poly = np.mean(polysemy)
    if avg_poly > 20 or avg_poly < 2:
        pattern['patterns'].append(f"Polysemy={avg_poly:.1f}")
        pattern['interest_score'] += 0.7
    
    # 3. Specific semantic domain (not entity.n.01)
    domains = []
    for w in cluster_words:
        domains.extend(w.get('semantic_domains', []))
    
    domain_counts = Counter(domains)
    for domain, count in domain_counts.most_common(3):
        if domain not in ['entity.n.01', 'abstraction.n.06'] and count > len(cluster_words) * 0.3:
            pattern['patterns'].append(f"Domain={domain} ({count}/{len(cluster_words)})")
            pattern['interest_score'] += 1.0
    
    # 4. Length clustering
    lengths = [len(w['word']) for w in cluster_words]
    if np.std(lengths) < 0.5 and len(cluster_words) > 5:
        pattern['patterns'].append(f"Length≈{np.mean(lengths):.1f}")
        pattern['interest_score'] += 0.6
    
    # 5. Frequency band
    ranks = [w['frequency_rank'] for w in cluster_words]
    if max(ranks) - min(ranks) < 100 and len(cluster_words) > 5:
        pattern['patterns'].append(f"Freq_rank={min(ranks)}-{max(ranks)}")
        pattern['interest_score'] += 0.8
    
    # 6. All concrete or all abstract
    concrete = sum(1 for w in cluster_words if w.get('is_concrete', False))
    abstract = sum(1 for w in cluster_words if w.get('is_abstract', False))
    
    if concrete > len(cluster_words) * 0.8:
        pattern['patterns'].append(f"Concrete ({concrete}/{len(cluster_words)})")
        pattern['interest_score'] += 0.6
    elif abstract > len(cluster_words) * 0.8:
        pattern['patterns'].append(f"Abstract ({abstract}/{len(cluster_words)})")
        pattern['interest_score'] += 0.6
    
    # 7. Animate clustering
    animate = sum(1 for w in cluster_words if w.get('is_animate', False))
    if animate > len(cluster_words) * 0.5:
        pattern['patterns'].append(f"Animate ({animate}/{len(cluster_words)})")
        pattern['interest_score'] += 0.9
    
    # 8. Check for semantic coherence within POS
    # E.g., all nouns that are animals, all verbs that are motion verbs
    if pos_counts and len(pos_counts) == 1:  # Single POS
        pos = list(pos_counts.keys())[0]
        
        # For nouns, check semantic subcategories
        if pos == 'n':
            # Check hypernym patterns
            hypernym_patterns = defaultdict(int)
            for w in cluster_words:
                for chain in w.get('hypernym_chains', []):
                    if len(chain) > 2:
                        hypernym_patterns[chain[2]] += 1  # Level 2 hypernym
            
            for hypernym, count in hypernym_patterns.items():
                if count > len(cluster_words) * 0.5:
                    pattern['patterns'].append(f"Noun_type={hypernym}")
                    pattern['interest_score'] += 1.2
    
    return pattern


def analyze_layer_transitions(word_features, activations):
    """Analyze how clusters change between adjacent layers with high k"""
    
    print("\n=== Analyzing layer transitions with k=100 ===")
    
    k = 100
    layer_labels = {}
    
    # Cluster each layer
    for layer_idx in range(12):
        layer_acts = activations[:, layer_idx, :]
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
        labels = kmeans.fit_predict(layer_acts)
        layer_labels[layer_idx] = labels
    
    # Track specific word groups
    word_groups = {
        'animals': [i for i, w in enumerate(word_features) 
                   if any('animal' in str(chain) for chain in w.get('hypernym_chains', []))],
        'artifacts': [i for i, w in enumerate(word_features) if w.get('is_artifact', False)],
        'emotions': [i for i, w in enumerate(word_features)
                    if any('feeling' in str(chain) or 'emotion' in str(chain) 
                          for chain in w.get('hypernym_chains', []))],
        'actions': [i for i, w in enumerate(word_features) 
                   if w['primary_pos'] == 'v' and w.get('is_physical', False)],
        'mental_verbs': [i for i, w in enumerate(word_features)
                        if w['primary_pos'] == 'v' and w.get('is_mental', False)]
    }
    
    # Track how these groups split/merge
    for group_name, indices in word_groups.items():
        if len(indices) < 10:
            continue
            
        print(f"\n{group_name} ({len(indices)} words):")
        
        for layer_idx in range(11):
            # How many clusters at this layer
            current_clusters = set(layer_labels[layer_idx][i] for i in indices)
            next_clusters = set(layer_labels[layer_idx + 1][i] for i in indices)
            
            print(f"  Layer {layer_idx}->{layer_idx+1}: "
                  f"{len(current_clusters)} clusters -> {len(next_clusters)} clusters")
            
            # Check for major splits or merges
            if len(next_clusters) > len(current_clusters) * 1.5:
                print(f"    SPLIT detected!")
            elif len(next_clusters) < len(current_clusters) * 0.7:
                print(f"    MERGE detected!")


def main():
    # Load data
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    
    print("Loading data...")
    with open(output_dir / "wordnet_features.json", 'r') as f:
        word_features = json.load(f)
    
    activations = np.load(output_dir / "activations.npy")
    print(f"Loaded {len(word_features)} words")
    
    # Quick clustering analysis
    results = quick_clustering_analysis(word_features, activations)
    
    # Print most interesting findings
    print("\n\n=== MOST INTERESTING PATTERNS FOUND ===")
    
    all_patterns = []
    for config, data in results.items():
        for pattern in data['interesting_patterns']:
            pattern['config'] = config
            all_patterns.append(pattern)
    
    # Sort by interest score
    all_patterns.sort(key=lambda x: x['interest_score'], reverse=True)
    
    for i, pattern in enumerate(all_patterns[:20]):
        print(f"\n{i+1}. {pattern['config']} - Cluster {pattern['cluster_id']} "
              f"(size={pattern['size']}, score={pattern['interest_score']:.2f})")
        print(f"   Patterns: {', '.join(pattern['patterns'])}")
        print(f"   Examples: {', '.join(pattern['examples'][:10])}")
    
    # Analyze transitions
    analyze_layer_transitions(word_features, activations)
    
    # Save results
    with open(output_dir / "high_k_patterns.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/high_k_patterns.json")


if __name__ == "__main__":
    main()