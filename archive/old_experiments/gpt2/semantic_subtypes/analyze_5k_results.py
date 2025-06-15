"""
Analyze the results from the 5k common words experiment
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import pickle

def load_results():
    """Load the saved results"""
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    
    # Load word features
    with open(output_dir / "wordnet_features.json", 'r') as f:
        word_features = json.load(f)
    
    # Load activations
    activations = np.load(output_dir / "activations.npy")
    
    print(f"Loaded {len(word_features)} words with WordNet features")
    print(f"Activations shape: {activations.shape}")
    
    return word_features, activations


def analyze_wordnet_distribution(word_features):
    """Analyze the distribution of WordNet features"""
    print("\n=== WordNet Feature Distribution ===\n")
    
    # POS distribution
    pos_counts = Counter()
    secondary_pos_counts = Counter()
    
    for word in word_features:
        if word['primary_pos']:
            pos_counts[word['primary_pos']] += 1
        if word['secondary_pos']:
            secondary_pos_counts[word['secondary_pos']] += 1
    
    print("Primary POS distribution:")
    total = sum(pos_counts.values())
    for pos, count in pos_counts.most_common():
        print(f"  {pos}: {count} ({count/total*100:.1f}%)")
    
    print("\nSecondary POS distribution:")
    for pos, count in secondary_pos_counts.most_common():
        print(f"  {pos}: {count}")
    
    # Polysemy statistics
    polysemy_counts = [word['polysemy_count'] for word in word_features]
    print(f"\nPolysemy statistics:")
    print(f"  Mean: {np.mean(polysemy_counts):.2f}")
    print(f"  Std: {np.std(polysemy_counts):.2f}")
    print(f"  Max: {max(polysemy_counts)}")
    print(f"  Monosemous: {sum(1 for p in polysemy_counts if p == 1)} ({sum(1 for p in polysemy_counts if p == 1)/len(polysemy_counts)*100:.1f}%)")
    
    # Semantic properties
    properties = ['is_concrete', 'is_abstract', 'is_animate', 'is_artifact', 'is_physical', 'is_mental']
    print("\nSemantic properties:")
    for prop in properties:
        count = sum(1 for word in word_features if word.get(prop, False))
        print(f"  {prop}: {count} ({count/len(word_features)*100:.1f}%)")
    
    # Combined properties
    concrete_nouns = sum(1 for w in word_features if w['primary_pos'] == 'n' and w.get('is_concrete', False))
    abstract_nouns = sum(1 for w in word_features if w['primary_pos'] == 'n' and w.get('is_abstract', False))
    print(f"\nCombined properties:")
    print(f"  Concrete nouns: {concrete_nouns}")
    print(f"  Abstract nouns: {abstract_nouns}")


def quick_cluster_analysis(word_features, activations):
    """Quick clustering analysis to test grammatical organization"""
    from sklearn.cluster import KMeans
    
    print("\n=== Quick Clustering Analysis (k=5) ===\n")
    
    # Cluster final layers
    for layer_idx in [9, 10, 11]:
        print(f"\nLayer {layer_idx}:")
        
        # Cluster
        layer_acts = activations[:, layer_idx, :]
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=3)
        labels = kmeans.fit_predict(layer_acts)
        
        # Analyze by POS
        pos_cluster_dist = defaultdict(lambda: Counter())
        
        for i, word in enumerate(word_features):
            pos = word['primary_pos']
            if pos:
                cluster = labels[i]
                pos_cluster_dist[pos][cluster] += 1
        
        # Print distribution
        for pos in ['n', 'v', 'a', 'r']:
            if pos in pos_cluster_dist:
                dist = pos_cluster_dist[pos]
                total = sum(dist.values())
                print(f"  {pos} ({total} words):", end="")
                for cluster in range(5):
                    count = dist[cluster]
                    print(f" C{cluster}:{count}({count/total*100:.0f}%)", end="")
                print()
        
        # Check concrete vs abstract nouns
        concrete_clusters = []
        abstract_clusters = []
        
        for i, word in enumerate(word_features):
            if word['primary_pos'] == 'n':
                if word.get('is_concrete', False):
                    concrete_clusters.append(labels[i])
                elif word.get('is_abstract', False):
                    abstract_clusters.append(labels[i])
        
        if concrete_clusters and abstract_clusters:
            concrete_dist = Counter(concrete_clusters)
            abstract_dist = Counter(abstract_clusters)
            
            print(f"\n  Concrete nouns ({len(concrete_clusters)}):", end="")
            for c in range(5):
                print(f" C{c}:{concrete_dist[c]}", end="")
            
            print(f"\n  Abstract nouns ({len(abstract_clusters)}):", end="")
            for c in range(5):
                print(f" C{c}:{abstract_dist[c]}", end="")
            print()


def analyze_paths_by_features(word_features, activations):
    """Analyze trajectory paths grouped by linguistic features"""
    from sklearn.cluster import KMeans
    
    print("\n=== Path Analysis by Linguistic Features ===\n")
    
    # First, cluster all layers with k=5
    paths = []
    for word_idx in range(len(word_features)):
        path = []
        for layer_idx in range(12):
            # Simple clustering for speed
            layer_acts = activations[:, layer_idx, :]
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=1)
            labels = kmeans.fit_predict(layer_acts)
            path.append(labels[word_idx])
        paths.append(path)
    
    # Group words by features
    feature_groups = defaultdict(list)
    
    for i, word in enumerate(word_features):
        # POS groups
        pos = word['primary_pos']
        if pos:
            feature_groups[f'pos_{pos}'].append(i)
        
        # Polysemy groups
        if word['polysemy_count'] == 1:
            feature_groups['monosemous'].append(i)
        elif word['polysemy_count'] > 3:
            feature_groups['highly_polysemous'].append(i)
        
        # Semantic groups
        if word.get('is_concrete', False):
            feature_groups['concrete'].append(i)
        if word.get('is_abstract', False):
            feature_groups['abstract'].append(i)
        
        # Combined groups
        if pos == 'n' and word.get('is_concrete', False):
            feature_groups['concrete_nouns'].append(i)
        if pos == 'n' and word.get('is_abstract', False):
            feature_groups['abstract_nouns'].append(i)
        if pos == 'v' and word.get('is_physical', False):
            feature_groups['physical_verbs'].append(i)
        if pos == 'v' and word.get('is_mental', False):
            feature_groups['mental_verbs'].append(i)
    
    # Analyze convergence for each group
    print("Late layer convergence (layers 9-11) by feature group:\n")
    
    for feature, indices in sorted(feature_groups.items()):
        if len(indices) < 10:  # Skip small groups
            continue
        
        # Calculate convergence in late layers
        convergence_scores = []
        for layer_idx in [9, 10, 11]:
            clusters_at_layer = [paths[i][layer_idx] for i in indices]
            unique_clusters = len(set(clusters_at_layer))
            convergence = 1.0 - (unique_clusters - 1) / max(1, len(indices) - 1)
            convergence_scores.append(convergence)
        
        avg_convergence = np.mean(convergence_scores)
        
        # Count unique full paths
        unique_paths = len(set(tuple(paths[i]) for i in indices))
        
        print(f"{feature:20} (n={len(indices):4}): "
              f"convergence={avg_convergence:.3f}, unique_paths={unique_paths}")


def main():
    # Load results
    word_features, activations = load_results()
    
    # Analyze WordNet distribution
    analyze_wordnet_distribution(word_features)
    
    # Quick clustering analysis
    quick_cluster_analysis(word_features, activations)
    
    # Path analysis by features
    analyze_paths_by_features(word_features, activations)
    
    print("\n=== Key Findings ===\n")
    print("1. The dataset successfully captured diverse linguistic features")
    print("2. Analysis shows how words with different WordNet properties cluster")
    print("3. We can now test if grammatical organization dominates over semantic")


if __name__ == "__main__":
    main()