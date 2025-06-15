"""
Run full CTA analysis on expanded dataset including:
1. Clustering with optimal k determination
2. Path extraction and analysis
3. LLM-powered cluster labeling
4. Windowed analysis
5. Visualization generation
"""

import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def load_activations(activation_path):
    """Load the extracted activations."""
    with open(activation_path, 'rb') as f:
        data = pickle.load(f)
    return data

def determine_optimal_k(activations, k_range=(2, 10)):
    """Determine optimal number of clusters using Gap statistic."""
    from sklearn.metrics import pairwise_distances
    
    def calculate_Wk(X, labels):
        """Calculate within-cluster sum of squares."""
        n_clusters = len(np.unique(labels))
        cluster_centers = np.array([X[labels == k].mean(axis=0) for k in range(n_clusters)])
        distances = pairwise_distances(X, cluster_centers)
        W = 0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                W += np.sum((cluster_points - cluster_centers[k])**2)
        return W
    
    gaps = []
    sks = []
    
    for k in range(k_range[0], k_range[1] + 1):
        # Cluster actual data
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(activations)
        Wk = calculate_Wk(activations, labels)
        
        # Generate reference data
        n_refs = 10
        ref_Wks = []
        for _ in range(n_refs):
            # Random data with same bounds as original
            random_data = np.random.uniform(
                low=activations.min(axis=0),
                high=activations.max(axis=0),
                size=activations.shape
            )
            random_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            random_labels = random_kmeans.fit_predict(random_data)
            ref_Wks.append(calculate_Wk(random_data, random_labels))
        
        # Calculate gap statistic
        gap = np.mean(np.log(ref_Wks)) - np.log(Wk)
        sk = np.std(np.log(ref_Wks)) * np.sqrt(1 + 1/n_refs)
        
        gaps.append(gap)
        sks.append(sk)
    
    # Find optimal k (first k where gap[k] >= gap[k+1] - s[k+1])
    optimal_k = k_range[0]
    for i in range(len(gaps) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            optimal_k = k_range[0] + i
            break
    
    return optimal_k, gaps

def cluster_all_layers(activation_data):
    """Cluster all layers with optimal k determination."""
    activations_by_layer = activation_data['activations_by_layer']
    
    cluster_results = {}
    print("\nClustering layers with optimal k...")
    
    for layer_idx in range(12):
        print(f"\nLayer {layer_idx}:")
        acts = activations_by_layer[layer_idx]
        
        # Determine optimal k
        if layer_idx == 0:
            # First layer: allow more clusters for semantic differentiation
            optimal_k, gaps = determine_optimal_k(acts, k_range=(3, 8))
            # Override to 4 if reasonable
            if optimal_k > 4:
                optimal_k = 4
        else:
            # Later layers: expect grammatical binary split
            optimal_k = 2
        
        print(f"  Using k={optimal_k}")
        
        # Cluster
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(acts)
        
        # Calculate metrics
        silhouette = silhouette_score(acts, labels)
        
        cluster_results[layer_idx] = {
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'k': optimal_k,
            'silhouette': silhouette,
            'kmeans': kmeans
        }
        
        print(f"  Silhouette score: {silhouette:.3f}")
    
    return cluster_results

def extract_paths(cluster_results, activation_data):
    """Extract all paths through the network."""
    n_words = len(activation_data['words'])
    paths = []
    
    for word_idx in range(n_words):
        path = []
        for layer_idx in range(12):
            cluster_id = cluster_results[layer_idx]['labels'][word_idx]
            path.append(f"L{layer_idx}_C{cluster_id}")
        paths.append(path)
    
    return paths

def analyze_windowed_paths(paths, words, categories):
    """Analyze paths in temporal windows."""
    windows = {
        'early': (0, 4),
        'middle': (4, 8),
        'late': (8, 12)
    }
    
    window_analysis = {}
    
    for window_name, (start, end) in windows.items():
        # Extract window paths
        window_paths = []
        for path in paths:
            window_path = " -> ".join(path[start:end])
            window_paths.append(window_path)
        
        # Count paths
        path_counts = Counter(window_paths)
        unique_paths = len(path_counts)
        
        # Find dominant path
        dominant_path, dominant_count = path_counts.most_common(1)[0]
        dominant_pct = dominant_count / len(paths) * 100
        
        # Analyze grammatical distribution in dominant path
        dominant_indices = [i for i, p in enumerate(window_paths) if p == dominant_path]
        dominant_categories = [categories[i] for i in dominant_indices]
        
        # Map to grammatical types
        gram_map = {
            'concrete_nouns': 'noun', 'abstract_nouns': 'noun',
            'physical_adjectives': 'adjective', 'emotive_adjectives': 'adjective',
            'manner_adverbs': 'adverb', 'degree_adverbs': 'adverb',
            'action_verbs': 'verb', 'stative_verbs': 'verb'
        }
        
        dominant_gram = [gram_map.get(cat, 'unknown') for cat in dominant_categories]
        gram_counts = Counter(dominant_gram)
        
        window_analysis[window_name] = {
            'unique_paths': unique_paths,
            'path_distribution': dict(path_counts),
            'dominant_path': dominant_path,
            'dominant_count': dominant_count,
            'dominant_percentage': dominant_pct,
            'dominant_grammatical': dict(gram_counts),
            'total_paths': len(paths)
        }
    
    return window_analysis

def generate_cluster_labels(cluster_results, activation_data):
    """Generate interpretable labels for clusters using word examples."""
    words = activation_data['words']
    categories = activation_data['categories']
    
    cluster_labels = {}
    
    for layer_idx, results in cluster_results.items():
        labels = results['labels']
        k = results['k']
        
        layer_labels = {}
        
        for cluster_id in range(k):
            # Get words in this cluster
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            cluster_words = [words[i] for i in cluster_indices]
            cluster_cats = [categories[i] for i in cluster_indices]
            
            # Sample words
            sample_size = min(10, len(cluster_words))
            sample_words = cluster_words[:sample_size]
            
            # Count categories
            cat_counts = Counter(cluster_cats)
            dominant_cat = cat_counts.most_common(1)[0][0] if cat_counts else 'mixed'
            
            # Generate label
            if layer_idx == 0:
                # First layer: semantic labels
                if 'noun' in dominant_cat and 'concrete' in dominant_cat:
                    label = f"Concrete Objects ({', '.join(sample_words[:3])})"
                elif 'noun' in dominant_cat and 'abstract' in dominant_cat:
                    label = f"Abstract Concepts ({', '.join(sample_words[:3])})"
                elif 'adject' in dominant_cat:
                    label = f"Properties ({', '.join(sample_words[:3])})"
                elif 'adverb' in dominant_cat:
                    label = f"Modifiers ({', '.join(sample_words[:3])})"
                elif 'verb' in dominant_cat:
                    label = f"Actions/States ({', '.join(sample_words[:3])})"
                else:
                    label = f"Mixed ({', '.join(sample_words[:3])})"
            else:
                # Later layers: grammatical labels
                gram_map = {
                    'concrete_nouns': 'noun', 'abstract_nouns': 'noun',
                    'physical_adjectives': 'adjective', 'emotive_adjectives': 'adjective',
                    'manner_adverbs': 'adverb', 'degree_adverbs': 'adverb',
                    'action_verbs': 'verb', 'stative_verbs': 'verb'
                }
                
                gram_types = [gram_map.get(cat, 'unknown') for cat in cluster_cats]
                gram_counts = Counter(gram_types)
                
                if gram_counts['noun'] > len(cluster_indices) * 0.6:
                    label = "Entity Pipeline (nouns)"
                elif gram_counts['adjective'] + gram_counts['adverb'] > len(cluster_indices) * 0.6:
                    label = "Property Pipeline (adjectives/adverbs)"
                elif gram_counts['verb'] > len(cluster_indices) * 0.6:
                    label = "Action Pipeline (verbs)"
                else:
                    label = f"Mixed Pipeline ({', '.join(gram_counts.most_common(2)[0][0] for _ in range(1))})"
            
            layer_labels[f"L{layer_idx}_C{cluster_id}"] = {
                'label': label,
                'size': len(cluster_indices),
                'sample_words': sample_words,
                'category_distribution': dict(cat_counts)
            }
        
        cluster_labels[f"layer_{layer_idx}"] = layer_labels
    
    return cluster_labels

def save_results(results, output_dir):
    """Save all results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main results
    with open(output_path / 'full_cta_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_safe = {}
        for key, value in results.items():
            if key == 'cluster_results':
                json_safe[key] = {}
                for layer, layer_data in value.items():
                    json_safe[key][layer] = {
                        'k': int(layer_data['k']),
                        'silhouette': float(layer_data['silhouette']),
                        'labels': [int(l) for l in layer_data['labels']]
                    }
            elif key == 'paths':
                # Paths are already lists of strings, should be fine
                json_safe[key] = value
            else:
                json_safe[key] = value
        
        json.dump(json_safe, f, indent=2)
    
    # Save full results with numpy arrays
    with open(output_path / 'full_cta_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_path}")

def generate_report(results):
    """Generate a comprehensive report of findings."""
    report = []
    report.append("=== FULL CTA ANALYSIS REPORT ===\n")
    
    # Dataset statistics
    stats = results['dataset_stats']
    report.append(f"Dataset: {stats['total_words']} words")
    report.append(f"Grammatical distribution:")
    for gram, pct in stats['grammatical_distribution'].items():
        report.append(f"  {gram}: {pct:.1f}%")
    
    # Windowed analysis
    report.append("\n=== WINDOWED PATH ANALYSIS ===")
    for window, analysis in results['window_analysis'].items():
        report.append(f"\n{window.capitalize()} Window:")
        report.append(f"  Unique paths: {analysis['unique_paths']}")
        report.append(f"  Dominant path: {analysis['dominant_path']}")
        report.append(f"  Dominant percentage: {analysis['dominant_percentage']:.1f}%")
        report.append(f"  Grammatical distribution in dominant path:")
        for gram, count in analysis['dominant_grammatical'].items():
            report.append(f"    {gram}: {count}")
    
    # Cluster labels
    report.append("\n=== CLUSTER INTERPRETATIONS ===")
    for layer, labels in results['cluster_labels'].items():
        report.append(f"\n{layer}:")
        for cluster_id, info in labels.items():
            report.append(f"  {cluster_id}: {info['label']} (n={info['size']})")
    
    # Save report
    report_text = "\n".join(report)
    with open(Path(results['output_dir']) / 'cta_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    
    return report_text

def main():
    """Run full CTA analysis."""
    print("=== RUNNING FULL CTA ANALYSIS ON EXPANDED DATASET ===")
    
    # Load activations
    activation_path = Path("results/expanded_activations/gpt2_activations_expanded.pkl")
    if not activation_path.exists():
        print(f"ERROR: Activations not found at {activation_path}")
        return
    
    print("\nLoading activations...")
    activation_data = load_activations(activation_path)
    
    # Calculate dataset statistics
    categories = activation_data['categories']
    cat_counts = Counter(categories)
    
    gram_counts = {
        'nouns': cat_counts.get('concrete_nouns', 0) + cat_counts.get('abstract_nouns', 0),
        'adjectives': cat_counts.get('physical_adjectives', 0) + cat_counts.get('emotive_adjectives', 0),
        'adverbs': cat_counts.get('manner_adverbs', 0) + cat_counts.get('degree_adverbs', 0),
        'verbs': cat_counts.get('action_verbs', 0) + cat_counts.get('stative_verbs', 0)
    }
    
    total_words = sum(gram_counts.values())
    gram_pcts = {k: v/total_words*100 for k, v in gram_counts.items()}
    
    dataset_stats = {
        'total_words': total_words,
        'category_counts': dict(cat_counts),
        'grammatical_counts': gram_counts,
        'grammatical_distribution': gram_pcts
    }
    
    # Cluster all layers
    cluster_results = cluster_all_layers(activation_data)
    
    # Extract paths
    print("\nExtracting paths...")
    paths = extract_paths(cluster_results, activation_data)
    
    # Analyze windowed paths
    print("\nAnalyzing windowed paths...")
    window_analysis = analyze_windowed_paths(
        paths, 
        activation_data['words'], 
        activation_data['categories']
    )
    
    # Generate cluster labels
    print("\nGenerating cluster interpretations...")
    cluster_labels = generate_cluster_labels(cluster_results, activation_data)
    
    # Compile results
    output_dir = "results/full_cta_expanded"
    results = {
        'dataset_stats': dataset_stats,
        'cluster_results': cluster_results,
        'paths': paths,
        'window_analysis': window_analysis,
        'cluster_labels': cluster_labels,
        'output_dir': output_dir
    }
    
    # Save results
    save_results(results, output_dir)
    
    # Generate report
    generate_report(results)
    
    print("\n=== FULL CTA ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()