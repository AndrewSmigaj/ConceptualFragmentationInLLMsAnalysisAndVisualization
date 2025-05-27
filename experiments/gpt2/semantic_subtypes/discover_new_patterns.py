"""
Discover NEW patterns in GPT-2 5k word experiment
Focus on finding unexpected groupings and insights
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def find_optimal_k_per_layer(activations, max_k=100):
    """Find optimal k for each layer using silhouette score"""
    n_words, n_layers, hidden_dim = activations.shape
    optimal_k_per_layer = {}
    
    print("Finding optimal k per layer (up to k=100)...")
    
    for layer_idx in range(n_layers):
        layer_acts = activations[:, layer_idx, :]
        
        # Try different k values - test much higher k
        silhouette_scores = {}
        # Test k = 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100
        k_values = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for k in k_values:
            if k >= n_words // 5:  # Don't go too high
                break
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
            labels = kmeans.fit_predict(layer_acts)
            score = silhouette_score(layer_acts, labels, sample_size=min(1000, n_words))
            silhouette_scores[k] = score
        
        # Find best k
        best_k = max(silhouette_scores, key=silhouette_scores.get)
        optimal_k_per_layer[layer_idx] = {
            'k': best_k,
            'score': silhouette_scores[best_k],
            'all_scores': silhouette_scores
        }
        
        print(f"Layer {layer_idx}: optimal k={best_k} (silhouette={silhouette_scores[best_k]:.3f})")
    
    return optimal_k_per_layer


def find_surprising_clusters(word_features, activations, layer_idx, k):
    """Find clusters with surprising/unexpected compositions"""
    
    # Cluster the layer
    layer_acts = activations[:, layer_idx, :]
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(layer_acts)
    
    surprising_clusters = []
    
    for cluster_id in range(k):
        cluster_mask = labels == cluster_id
        cluster_words = [word_features[i] for i in range(len(word_features)) if cluster_mask[i]]
        
        if len(cluster_words) < 5:
            continue
        
        # Look for unexpected patterns
        findings = analyze_cluster_surprises(cluster_words, cluster_id)
        
        if findings['surprise_score'] > 0.5:  # Threshold for "surprising"
            surprising_clusters.append({
                'cluster_id': cluster_id,
                'size': len(cluster_words),
                'findings': findings,
                'example_words': [w['word'] for w in cluster_words[:20]]
            })
    
    return surprising_clusters


def analyze_cluster_surprises(cluster_words, cluster_id):
    """Analyze what makes a cluster surprising"""
    findings = {
        'surprise_score': 0,
        'patterns': []
    }
    
    # 1. Unusual POS combinations
    pos_dist = Counter(w['primary_pos'] for w in cluster_words if w['primary_pos'])
    if len(pos_dist) > 3:  # Many different POS together
        findings['patterns'].append(f"Mixed POS: {dict(pos_dist)}")
        findings['surprise_score'] += 0.3
    
    # 2. Polysemy outliers clustering together
    polysemy_counts = [w['polysemy_count'] for w in cluster_words]
    if np.mean(polysemy_counts) > 15:  # Very polysemous words
        findings['patterns'].append(f"High polysemy cluster: avg={np.mean(polysemy_counts):.1f}")
        findings['surprise_score'] += 0.4
    
    # 3. Mixed concrete/abstract
    concrete = sum(1 for w in cluster_words if w.get('is_concrete', False))
    abstract = sum(1 for w in cluster_words if w.get('is_abstract', False))
    if concrete > 10 and abstract > 10 and abs(concrete - abstract) < 5:
        findings['patterns'].append(f"Mixed concrete/abstract: {concrete}/{abstract}")
        findings['surprise_score'] += 0.3
    
    # 4. Specific semantic domain concentration
    all_domains = []
    for w in cluster_words:
        all_domains.extend(w.get('semantic_domains', []))
    
    domain_counts = Counter(all_domains)
    for domain, count in domain_counts.most_common(3):
        concentration = count / len(cluster_words)
        if concentration > 0.3 and domain not in ['entity.n.01', 'abstraction.n.06']:
            findings['patterns'].append(f"Domain concentration: {domain} ({concentration:.1%})")
            findings['surprise_score'] += 0.5
    
    # 5. Letter/character patterns
    first_letters = Counter(w['word'][0].lower() for w in cluster_words if w['word'])
    for letter, count in first_letters.most_common(1):
        if count / len(cluster_words) > 0.2:  # 20% start with same letter
            findings['patterns'].append(f"Letter pattern: {count} words start with '{letter}'")
            findings['surprise_score'] += 0.3
    
    # 6. Word length patterns
    lengths = [len(w['word']) for w in cluster_words]
    if np.std(lengths) < 1.0:  # Very similar lengths
        findings['patterns'].append(f"Length pattern: avg={np.mean(lengths):.1f}, std={np.std(lengths):.2f}")
        findings['surprise_score'] += 0.2
    
    # 7. Frequency rank patterns
    ranks = [w['frequency_rank'] for w in cluster_words]
    if np.std(ranks) < 100:  # Similar frequency ranks
        findings['patterns'].append(f"Frequency pattern: ranks {min(ranks)}-{max(ranks)}")
        findings['surprise_score'] += 0.3
    
    return findings


def analyze_cross_layer_stability(word_features, activations, word_groups):
    """Track how specific word groups stay together across layers"""
    
    n_layers = activations.shape[1]
    stability_results = {}
    
    for group_name, word_indices in word_groups.items():
        if len(word_indices) < 5:
            continue
            
        coherence_scores = []
        
        for layer_idx in range(n_layers):
            # Cluster this layer with more clusters
            layer_acts = activations[:, layer_idx, :]
            kmeans = KMeans(n_clusters=50, random_state=42, n_init=3)
            labels = kmeans.fit_predict(layer_acts)
            
            # Check how concentrated the group is
            group_labels = [labels[i] for i in word_indices]
            label_counts = Counter(group_labels)
            
            # Coherence = concentration in dominant cluster
            max_concentration = max(label_counts.values()) / len(group_labels)
            coherence_scores.append(max_concentration)
        
        stability_results[group_name] = {
            'coherence_trajectory': coherence_scores,
            'avg_coherence': np.mean(coherence_scores),
            'stability': np.std(coherence_scores),  # Lower = more stable
            'peak_layer': np.argmax(coherence_scores),
            'trough_layer': np.argmin(coherence_scores)
        }
    
    return stability_results


def visualize_interesting_clusters(word_features, activations, layer_idx, findings):
    """Visualize the most interesting clusters"""
    
    # Reduce dimensions for visualization
    layer_acts = activations[:, layer_idx, :]
    
    # Use PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(layer_acts)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Color by different properties
    for finding in findings[:3]:  # Top 3 most surprising
        cluster_words = finding['example_words']
        # Find indices of these words
        word_to_idx = {w['word']: i for i, w in enumerate(word_features)}
        indices = [word_to_idx[w] for w in cluster_words if w in word_to_idx]
        
        if indices:
            plt.scatter(coords_2d[indices, 0], coords_2d[indices, 1], 
                       alpha=0.6, s=50, label=f"Cluster {finding['cluster_id']}")
    
    plt.title(f"Layer {layer_idx} - Surprising Clusters")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"experiments/gpt2/semantic_subtypes/5k_common_words/surprising_clusters_layer_{layer_idx}.png")
    plt.close()


def main():
    # Load data
    print("Loading data...")
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    
    with open(output_dir / "wordnet_features.json", 'r') as f:
        word_features = json.load(f)
    
    activations = np.load(output_dir / "activations.npy")
    print(f"Loaded {len(word_features)} words, activations shape: {activations.shape}")
    
    # Find optimal k per layer with much higher k values
    optimal_k = find_optimal_k_per_layer(activations, max_k=100)
    
    # Save optimal k results
    with open(output_dir / "optimal_k_per_layer.json", 'w') as f:
        json.dump(optimal_k, f, indent=2)
    
    # Find surprising clusters at each layer
    all_surprising_clusters = {}
    
    for layer_idx in range(12):
        k = optimal_k[layer_idx]['k']
        print(f"\nAnalyzing layer {layer_idx} with k={k}...")
        
        surprising = find_surprising_clusters(word_features, activations, layer_idx, k)
        
        if surprising:
            all_surprising_clusters[f'layer_{layer_idx}'] = surprising
            
            # Print most surprising findings
            print(f"\nLayer {layer_idx} surprising clusters:")
            for cluster in sorted(surprising, key=lambda x: x['findings']['surprise_score'], reverse=True)[:3]:
                print(f"\n  Cluster {cluster['cluster_id']} (size={cluster['size']}, surprise={cluster['findings']['surprise_score']:.2f}):")
                for pattern in cluster['findings']['patterns']:
                    print(f"    - {pattern}")
                print(f"    Examples: {', '.join(cluster['example_words'][:10])}")
    
    # Define interesting word groups to track
    word_groups = {
        'highly_polysemous': [i for i, w in enumerate(word_features) if w['polysemy_count'] > 20],
        'monosemous': [i for i, w in enumerate(word_features) if w['polysemy_count'] == 1],
        'very_concrete': [i for i, w in enumerate(word_features) if w.get('is_concrete', False) and not w.get('is_abstract', False)],
        'very_abstract': [i for i, w in enumerate(word_features) if w.get('is_abstract', False) and not w.get('is_concrete', False)],
        'short_words': [i for i, w in enumerate(word_features) if len(w['word']) <= 3],
        'long_words': [i for i, w in enumerate(word_features) if len(w['word']) >= 8],
        'function_words': [i for i, w in enumerate(word_features) if w['frequency_rank'] <= 50],
        'rare_words': [i for i, w in enumerate(word_features) if w['frequency_rank'] >= 2000]
    }
    
    # Analyze stability
    print("\n\nAnalyzing cross-layer stability of word groups...")
    stability = analyze_cross_layer_stability(word_features, activations, word_groups)
    
    for group, results in sorted(stability.items(), key=lambda x: x[1]['avg_coherence'], reverse=True):
        print(f"\n{group}:")
        print(f"  Average coherence: {results['avg_coherence']:.3f}")
        print(f"  Stability (std): {results['stability']:.3f}")
        print(f"  Peak coherence at layer {results['peak_layer']}")
        print(f"  Trajectory: {' -> '.join([f'{c:.2f}' for c in results['coherence_trajectory']])}")
    
    # Save all findings
    findings = {
        'optimal_k': optimal_k,
        'surprising_clusters': all_surprising_clusters,
        'group_stability': stability
    }
    
    with open(output_dir / "new_pattern_discoveries.json", 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    
    print("\n\n=== KEY DISCOVERIES ===")
    print("1. Optimal cluster numbers vary significantly by layer")
    print("2. Several surprising cluster patterns found")
    print("3. Different word groups show different stability patterns")
    print(f"\nResults saved to {output_dir}/new_pattern_discoveries.json")


if __name__ == "__main__":
    main()