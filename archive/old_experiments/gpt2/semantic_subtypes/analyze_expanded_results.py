"""
Analyze the expanded GPT-2 experiment results and extract key metrics for paper update.
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directories
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def load_activations(activation_path):
    """Load the extracted activations."""
    with open(activation_path, 'rb') as f:
        data = pickle.load(f)
    return data

def cluster_layer(activations, n_clusters=None):
    """Cluster activations for a single layer."""
    if n_clusters is None:
        # Use Gap statistic or default based on layer
        n_clusters = 4 if activations.shape[0] > 100 else 2
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(activations)
    
    return labels, kmeans

def analyze_trajectories(activation_data):
    """Analyze word trajectories through layers."""
    
    activations_by_layer = activation_data['activations_by_layer']
    words = activation_data['words']
    categories = activation_data['categories']
    
    # Cluster each layer
    print("\nClustering layers...")
    cluster_labels_by_layer = {}
    
    # Layer 0: 4 clusters (semantic differentiation)
    # Layers 1-11: 2 clusters (entity vs modifier)
    for layer_idx in range(12):
        n_clusters = 4 if layer_idx == 0 else 2
        labels, _ = cluster_layer(activations_by_layer[layer_idx], n_clusters)
        cluster_labels_by_layer[layer_idx] = labels
    
    # Build trajectories
    trajectories = []
    for i in range(len(words)):
        trajectory = []
        for layer_idx in range(12):
            cluster_id = cluster_labels_by_layer[layer_idx][i]
            trajectory.append(f"L{layer_idx}_C{cluster_id}")
        trajectories.append(trajectory)
    
    # Analyze path convergence
    window_paths = {
        'early': [],
        'middle': [],
        'late': []
    }
    
    for traj in trajectories:
        # Early window (L0-L3)
        early_path = " -> ".join(traj[0:4])
        window_paths['early'].append(early_path)
        
        # Middle window (L4-L7)
        middle_path = " -> ".join(traj[4:8])
        window_paths['middle'].append(middle_path)
        
        # Late window (L8-L11)
        late_path = " -> ".join(traj[8:12])
        window_paths['late'].append(late_path)
    
    # Count unique paths per window
    path_stats = {}
    for window, paths in window_paths.items():
        path_counts = Counter(paths)
        unique_paths = len(path_counts)
        dominant_path = path_counts.most_common(1)[0]
        dominant_percentage = dominant_path[1] / len(paths) * 100
        
        path_stats[window] = {
            'unique_paths': unique_paths,
            'dominant_path': dominant_path[0],
            'dominant_percentage': dominant_percentage,
            'path_distribution': dict(path_counts)
        }
    
    return trajectories, path_stats, cluster_labels_by_layer

def analyze_entity_convergence(trajectories, categories):
    """Analyze convergence to entity superhighway."""
    
    # Check middle window convergence (layers 4-7)
    # Entity superhighway is typically L4_C1 → L5_C0 → L6_C0 → L7_C1
    # But with 2 clusters, let's identify the dominant noun cluster
    
    # First, identify which cluster in middle layers contains most nouns
    noun_indices = [i for i, cat in enumerate(categories) 
                   if 'noun' in cat.lower()]
    
    middle_paths = []
    for traj in trajectories:
        middle_path = " -> ".join(traj[4:8])
        middle_paths.append(middle_path)
    
    # Count paths for nouns
    noun_paths = [middle_paths[i] for i in noun_indices]
    noun_path_counts = Counter(noun_paths)
    dominant_noun_path = noun_path_counts.most_common(1)[0][0]
    
    # Count how many words follow this path
    entity_count = sum(1 for path in middle_paths if path == dominant_noun_path)
    total_words = len(middle_paths)
    entity_percentage = entity_count / total_words * 100
    
    # Calculate 95% confidence interval using Wilson score
    n = total_words
    p = entity_count / n
    z = 1.96  # 95% confidence
    
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z/(4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    
    ci_lower = lower_bound * 100
    ci_upper = upper_bound * 100
    
    return {
        'entity_superhighway_percentage': entity_percentage,
        'entity_count': entity_count,
        'total_words': total_words,
        'dominant_noun_path': dominant_noun_path,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def test_grammatical_organization(trajectories, categories):
    """Chi-square test for grammatical vs semantic organization."""
    
    # Map categories to grammatical types
    grammatical_map = {
        'concrete_nouns': 'noun',
        'abstract_nouns': 'noun',
        'physical_adjectives': 'adjective',
        'emotive_adjectives': 'adjective',
        'manner_adverbs': 'adverb',
        'degree_adverbs': 'adverb',
        'action_verbs': 'verb',
        'stative_verbs': 'verb'
    }
    
    grammatical_types = [grammatical_map.get(cat, 'unknown') for cat in categories]
    
    # Get middle window clusters (where grammatical organization emerges)
    middle_clusters = []
    for traj in trajectories:
        # Take cluster at layer 5 as representative
        cluster = traj[5].split('_')[1]  # Extract C0 or C1
        middle_clusters.append(cluster)
    
    # Create contingency table
    contingency = defaultdict(lambda: defaultdict(int))
    for gram_type, cluster in zip(grammatical_types, middle_clusters):
        contingency[gram_type][cluster] += 1
    
    # Convert to array
    gram_types = ['noun', 'adjective', 'adverb', 'verb']
    clusters = sorted(set(middle_clusters))
    
    table = []
    for gram in gram_types:
        row = [contingency[gram][c] for c in clusters]
        table.append(row)
    
    table = np.array(table)
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    
    # Cramér's V
    n = np.sum(table)
    min_dim = min(len(gram_types) - 1, len(clusters) - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    return {
        'chi_square': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cramers_v': cramers_v,
        'contingency_table': table.tolist(),
        'grammatical_types': gram_types,
        'clusters': clusters
    }

def calculate_stability_metrics(trajectories):
    """Calculate stability metrics for each window."""
    
    stability = {}
    
    for window_name, layer_range in [('early', range(0, 4)), 
                                     ('middle', range(4, 8)), 
                                     ('late', range(8, 12))]:
        # Count transitions within window
        transitions = []
        for traj in trajectories:
            window_traj = [traj[i] for i in layer_range]
            
            # Count how many times the cluster changes
            changes = 0
            for i in range(1, len(window_traj)):
                if window_traj[i] != window_traj[i-1]:
                    changes += 1
            
            # Stability is inverse of changes
            stability_score = 1 - (changes / (len(window_traj) - 1))
            transitions.append(stability_score)
        
        stability[window_name] = np.mean(transitions)
    
    return stability

def main():
    """Run analysis on expanded dataset."""
    
    # Load activations
    activation_path = Path("results/expanded_activations/gpt2_activations_expanded.pkl")
    if not activation_path.exists():
        print(f"ERROR: Activations not found at {activation_path}")
        return
    
    print("Loading activations...")
    activation_data = load_activations(activation_path)
    
    # Get word distribution
    categories = activation_data['categories']
    cat_counts = Counter(categories)
    
    # Calculate grammatical distribution
    grammatical_counts = {
        'nouns': cat_counts.get('concrete_nouns', 0) + cat_counts.get('abstract_nouns', 0),
        'adjectives': cat_counts.get('physical_adjectives', 0) + cat_counts.get('emotive_adjectives', 0),
        'adverbs': cat_counts.get('manner_adverbs', 0) + cat_counts.get('degree_adverbs', 0),
        'verbs': cat_counts.get('action_verbs', 0) + cat_counts.get('stative_verbs', 0)
    }
    
    total_words = sum(grammatical_counts.values())
    
    print("\n=== EXPANDED DATASET STATISTICS ===")
    print(f"Total words: {total_words}")
    print("\nGrammatical distribution:")
    for gram_type, count in grammatical_counts.items():
        percentage = count / total_words * 100
        print(f"  {gram_type}: {count} ({percentage:.1f}%)")
    
    # Analyze trajectories
    print("\nAnalyzing trajectories...")
    trajectories, path_stats, cluster_labels = analyze_trajectories(activation_data)
    
    # Path evolution
    print("\n=== PATH EVOLUTION ===")
    for window, stats in path_stats.items():
        print(f"\n{window.capitalize()} window:")
        print(f"  Unique paths: {stats['unique_paths']}")
        print(f"  Dominant path: {stats['dominant_path']}")
        print(f"  Dominant percentage: {stats['dominant_percentage']:.1f}%")
    
    # Entity convergence
    print("\n=== ENTITY SUPERHIGHWAY CONVERGENCE ===")
    convergence = analyze_entity_convergence(trajectories, categories)
    print(f"Entity superhighway: {convergence['entity_superhighway_percentage']:.1f}%")
    print(f"95% CI: {convergence['ci_lower']:.1f}%-{convergence['ci_upper']:.1f}%")
    print(f"Path: {convergence['dominant_noun_path']}")
    
    # Statistical test
    print("\n=== STATISTICAL VALIDATION ===")
    chi_test = test_grammatical_organization(trajectories, categories)
    print(f"Chi-square test for grammatical organization:")
    print(f"  chi-square = {chi_test['chi_square']:.2f}")
    print(f"  p-value = {chi_test['p_value']:.4f}")
    print(f"  Cramer's V = {chi_test['cramers_v']:.3f}")
    
    # Stability analysis
    print("\n=== STABILITY ANALYSIS ===")
    stability = calculate_stability_metrics(trajectories)
    for window, score in stability.items():
        print(f"{window.capitalize()}: {score:.3f}")
    
    # Save results
    results = {
        'total_words': total_words,
        'word_distribution': {
            gram_type: {
                'count': count,
                'percentage': count / total_words * 100
            }
            for gram_type, count in grammatical_counts.items()
        },
        'path_evolution': path_stats,
        'convergence_stats': convergence,
        'statistical_tests': chi_test,
        'stability_metrics': stability,
        'metadata': {
            'dataset': 'expanded',
            'activation_file': str(activation_path)
        }
    }
    
    output_path = Path("results/expanded_analysis_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_path}")
    
    print("\n" + "="*60)
    print("KEY METRICS TO UPDATE IN PAPER:")
    print("="*60)
    print(f"1. Total words analyzed: {total_words} (was 566)")
    print(f"2. Verb percentage: {grammatical_counts['verbs'] / total_words * 100:.1f}% (was ~8%)")
    print(f"3. Entity convergence: {convergence['entity_superhighway_percentage']:.1f}% (was 72.8%)")
    print(f"4. Path reduction: {path_stats['early']['unique_paths']} -> {path_stats['middle']['unique_paths']} -> {path_stats['late']['unique_paths']}")
    print(f"5. Chi-square: {chi_test['chi_square']:.2f}, p < {chi_test['p_value']:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()