"""
Analyze clusters with full WordNet-based metrics
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
import pickle

def load_data():
    """Load saved data"""
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    
    # Load WordNet features
    with open(output_dir / "wordnet_features.json", 'r') as f:
        word_features = json.load(f)
    
    # Load activations
    activations = np.load(output_dir / "activations.npy")
    
    return word_features, activations


def analyze_cluster_with_wordnet(cluster_words, cluster_id, layer_idx):
    """Analyze a single cluster with all WordNet metrics"""
    
    if not cluster_words:
        return None
    
    analysis = {
        'cluster_id': cluster_id,
        'layer': layer_idx,
        'size': len(cluster_words),
        'example_words': [w['word'] for w in cluster_words[:10]]
    }
    
    # POS distribution
    pos_counts = Counter(w['primary_pos'] for w in cluster_words if w['primary_pos'])
    total_pos = sum(pos_counts.values())
    analysis['pos_distribution'] = {pos: {'count': count, 'percentage': count/total_pos*100} 
                                   for pos, count in pos_counts.items()}
    
    # Secondary POS
    secondary_pos = Counter(w['secondary_pos'] for w in cluster_words if w['secondary_pos'])
    analysis['secondary_pos_count'] = dict(secondary_pos)
    
    # Semantic properties
    total = len(cluster_words)
    analysis['semantic_properties'] = {
        'concrete': sum(1 for w in cluster_words if w.get('is_concrete', False)) / total * 100,
        'abstract': sum(1 for w in cluster_words if w.get('is_abstract', False)) / total * 100,
        'animate': sum(1 for w in cluster_words if w.get('is_animate', False)) / total * 100,
        'artifact': sum(1 for w in cluster_words if w.get('is_artifact', False)) / total * 100,
        'physical': sum(1 for w in cluster_words if w.get('is_physical', False)) / total * 100,
        'mental': sum(1 for w in cluster_words if w.get('is_mental', False)) / total * 100
    }
    
    # Polysemy statistics
    polysemy_counts = [w['polysemy_count'] for w in cluster_words]
    analysis['polysemy_stats'] = {
        'mean': np.mean(polysemy_counts),
        'std': np.std(polysemy_counts),
        'max': max(polysemy_counts),
        'min': min(polysemy_counts),
        'monosemous_ratio': sum(1 for p in polysemy_counts if p == 1) / len(polysemy_counts) * 100
    }
    
    # Semantic domains (top 10)
    all_domains = []
    for w in cluster_words:
        all_domains.extend(w.get('semantic_domains', []))
    domain_counts = Counter(all_domains)
    analysis['top_semantic_domains'] = [
        {'domain': domain, 'count': count} 
        for domain, count in domain_counts.most_common(10)
    ]
    
    # Combined features analysis
    combined = {
        'concrete_nouns': sum(1 for w in cluster_words 
                             if w['primary_pos'] == 'n' and w.get('is_concrete', False)),
        'abstract_nouns': sum(1 for w in cluster_words 
                             if w['primary_pos'] == 'n' and w.get('is_abstract', False)),
        'physical_verbs': sum(1 for w in cluster_words 
                             if w['primary_pos'] == 'v' and w.get('is_physical', False)),
        'mental_verbs': sum(1 for w in cluster_words 
                            if w['primary_pos'] == 'v' and w.get('is_mental', False)),
        'animate_nouns': sum(1 for w in cluster_words 
                            if w['primary_pos'] == 'n' and w.get('is_animate', False))
    }
    analysis['combined_features'] = combined
    
    return analysis


def run_clustering_and_analysis(word_features, activations, k=5):
    """Run clustering and analyze with WordNet features"""
    
    print(f"\n=== Clustering with k={k} ===")
    
    n_words, n_layers, hidden_dim = activations.shape
    results = {
        'k': k,
        'n_words': n_words,
        'n_layers': n_layers,
        'layer_analyses': {}
    }
    
    # Cluster each layer
    for layer_idx in range(n_layers):
        print(f"\nAnalyzing Layer {layer_idx}...")
        
        # Cluster this layer
        layer_acts = activations[:, layer_idx, :]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(layer_acts)
        
        # Analyze each cluster
        layer_results = {
            'layer': layer_idx,
            'clusters': {}
        }
        
        for cluster_id in range(k):
            # Get words in this cluster
            cluster_mask = labels == cluster_id
            cluster_words = [word_features[i] for i in range(n_words) if cluster_mask[i]]
            
            # Analyze with WordNet features
            cluster_analysis = analyze_cluster_with_wordnet(cluster_words, cluster_id, layer_idx)
            if cluster_analysis:
                layer_results['clusters'][f'cluster_{cluster_id}'] = cluster_analysis
                
                # Print summary
                print(f"\n  Cluster {cluster_id} ({len(cluster_words)} words):")
                pos_dist = cluster_analysis['pos_distribution']
                if pos_dist:
                    top_pos = sorted(pos_dist.items(), key=lambda x: x[1]['count'], reverse=True)[:3]
                    print(f"    POS: {', '.join([f'{pos}:{data['count']}' for pos, data in top_pos])}")
                
                sem_props = cluster_analysis['semantic_properties']
                high_props = [prop for prop, pct in sem_props.items() if pct > 30]
                if high_props:
                    print(f"    High semantic properties: {', '.join(high_props)}")
                
                print(f"    Polysemy: mean={cluster_analysis['polysemy_stats']['mean']:.1f}, "
                      f"monosemous={cluster_analysis['polysemy_stats']['monosemous_ratio']:.1f}%")
        
        results['layer_analyses'][f'layer_{layer_idx}'] = layer_results
    
    # Path analysis
    print("\n=== Path Analysis ===")
    
    # Extract paths
    paths = []
    all_labels = {}
    
    for layer_idx in range(n_layers):
        layer_acts = activations[:, layer_idx, :]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(layer_acts)
        all_labels[layer_idx] = labels
    
    for word_idx in range(n_words):
        path = [f"L{layer}_C{all_labels[layer][word_idx]}" for layer in range(n_layers)]
        paths.append(path)
    
    # Analyze paths by WordNet features
    feature_groups = defaultdict(list)
    
    for i, word in enumerate(word_features):
        # Group by POS
        if word['primary_pos']:
            feature_groups[f"pos_{word['primary_pos']}"].append(i)
        
        # Group by semantic properties
        if word.get('is_concrete', False):
            feature_groups['concrete'].append(i)
        if word.get('is_abstract', False):
            feature_groups['abstract'].append(i)
        
        # Combined groups
        if word['primary_pos'] == 'n' and word.get('is_concrete', False):
            feature_groups['concrete_nouns'].append(i)
        if word['primary_pos'] == 'n' and word.get('is_abstract', False):
            feature_groups['abstract_nouns'].append(i)
    
    print("\nPath Convergence by Feature Group (Layers 9-11):")
    
    path_results = {}
    for feature, indices in sorted(feature_groups.items()):
        if len(indices) < 10:
            continue
        
        # Check convergence in late layers
        convergence_scores = []
        for layer in [9, 10, 11]:
            clusters_at_layer = [paths[i][layer] for i in indices]
            unique = len(set(clusters_at_layer))
            convergence = 1.0 - (unique - 1) / max(1, len(indices) - 1)
            convergence_scores.append(convergence)
        
        avg_convergence = np.mean(convergence_scores)
        unique_paths = len(set(tuple(paths[i]) for i in indices))
        
        path_results[feature] = {
            'n_words': len(indices),
            'avg_late_convergence': avg_convergence,
            'unique_paths': unique_paths
        }
        
        print(f"  {feature:20} (n={len(indices):4}): convergence={avg_convergence:.3f}, paths={unique_paths}")
    
    results['path_analysis'] = path_results
    
    return results


def main():
    # Load data
    print("Loading data...")
    word_features, activations = load_data()
    
    # Run analysis for different k values
    all_results = {}
    
    for k in [3, 5, 8]:
        results = run_clustering_and_analysis(word_features, activations, k)
        all_results[f'k_{k}'] = results
    
    # Save results
    output_dir = Path("experiments/gpt2/semantic_subtypes/5k_common_words")
    with open(output_dir / "wordnet_cluster_analysis.json", 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        json.dump(convert_numpy(all_results), f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/wordnet_cluster_analysis.json")
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print("1. Each cluster has been analyzed with full WordNet-based metrics")
    print("2. Clusters show distribution of POS, semantic properties, polysemy, and domains")
    print("3. Path analysis reveals convergence patterns for different linguistic groups")
    print("4. Results enable testing if grammatical organization dominates semantic grouping")


if __name__ == "__main__":
    main()