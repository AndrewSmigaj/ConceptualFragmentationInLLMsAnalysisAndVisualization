#!/usr/bin/env python3
"""
Comprehensive window analysis for GPT-2 semantic subtypes.
Shows cluster counts, identifies interesting windows, and tracks representative words.
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import pandas as pd

def load_all_data():
    """Load all necessary data for analysis."""
    # Load K-means results
    with open("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_kmeans_clustering.pkl", 'rb') as f:
        kmeans_results = pickle.load(f)
    
    # Load ETS results (if available)
    ets_results = None
    ets_path = Path("semantic_subtypes_experiment_20250523_111112/semantic_subtypes_ets_clustering.pkl")
    if ets_path.exists():
        with open(ets_path, 'rb') as f:
            ets_results = pickle.load(f)
    
    # Load curated words
    with open("data/gpt2_semantic_subtypes_curated.json", 'r') as f:
        curated_data = json.load(f)
    
    return kmeans_results, ets_results, curated_data

def extract_cluster_counts(clustering_results: Dict[str, Any]) -> Dict[str, int]:
    """Extract number of clusters per layer."""
    cluster_counts = {}
    
    for layer_key, layer_data in clustering_results['layer_results'].items():
        if 'cluster_labels' in layer_data:
            # Count unique clusters
            all_clusters = set()
            for sent_clusters in layer_data['cluster_labels'].values():
                for cluster_id in sent_clusters.values():
                    all_clusters.add(cluster_id)
            cluster_counts[layer_key] = len(all_clusters)
        else:
            cluster_counts[layer_key] = layer_data.get('optimal_k', 0)
    
    return cluster_counts

def calculate_window_metrics(clustering_results: Dict[str, Any], window_size: int = 3) -> Dict[str, Dict]:
    """Calculate metrics for each sliding window."""
    layer_results = clustering_results['layer_results']
    num_layers = len(layer_results)
    num_windows = num_layers - window_size + 1
    
    window_metrics = {}
    
    for window_start in range(num_windows):
        window_end = window_start + window_size
        window_key = f"layers_{window_start}-{window_end-1}"
        
        # Count unique paths in this window
        window_paths = defaultdict(int)
        
        for sent_idx in range(len(clustering_results['sentences'])):
            path = []
            valid = True
            
            for offset in range(window_size):
                layer_idx = window_start + offset
                layer_key = f"layer_{layer_idx}"
                
                if (layer_key in layer_results and 
                    'cluster_labels' in layer_results[layer_key] and
                    sent_idx in layer_results[layer_key]['cluster_labels'] and
                    0 in layer_results[layer_key]['cluster_labels'][sent_idx]):
                    
                    cluster_id = layer_results[layer_key]['cluster_labels'][sent_idx][0]
                    path.append(f"L{layer_idx}C{cluster_id}")
                else:
                    valid = False
                    break
            
            if valid:
                path_str = " -> ".join(path)
                window_paths[path_str] += 1
        
        # Calculate metrics
        total_words = len(clustering_results['sentences'])
        unique_paths = len(window_paths)
        
        # Path entropy (diversity)
        if window_paths:
            path_probs = np.array(list(window_paths.values())) / total_words
            entropy = -np.sum(path_probs * np.log2(path_probs + 1e-10))
        else:
            entropy = 0
        
        # Fragmentation: average number of outgoing paths per cluster
        start_to_paths = defaultdict(set)
        for path in window_paths:
            parts = path.split(" -> ")
            if len(parts) >= 2:
                start_to_paths[parts[0]].add(parts[1])
        
        if start_to_paths:
            fragmentation = np.mean([len(paths) for paths in start_to_paths.values()])
        else:
            fragmentation = 1.0
        
        window_metrics[window_key] = {
            'start_layer': window_start,
            'end_layer': window_end - 1,
            'unique_paths': unique_paths,
            'entropy': entropy,
            'fragmentation': fragmentation,
            'coverage': sum(window_paths.values()) / total_words
        }
    
    return window_metrics

def select_representative_words(curated_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Select representative words from each semantic class."""
    representatives = {}
    
    curated_words = curated_data.get('curated_words', {})
    
    for subtype, words in curated_words.items():
        # Select diverse representatives
        if len(words) >= 5:
            # Take first, middle, and last words alphabetically
            sorted_words = sorted(words)
            indices = [0, len(sorted_words)//4, len(sorted_words)//2, 3*len(sorted_words)//4, -1]
            representatives[subtype] = [sorted_words[i] for i in indices]
        else:
            representatives[subtype] = words[:3]
    
    return representatives

def trace_word_paths(word: str, clustering_results: Dict[str, Any]) -> List[str]:
    """Trace the complete path of a word through all layers."""
    sentences = clustering_results['sentences']
    
    # Find word index
    try:
        word_idx = sentences.index(word)
    except ValueError:
        return []
    
    # Build path
    path = []
    for layer_idx in range(len(clustering_results['layer_results'])):
        layer_key = f"layer_{layer_idx}"
        
        if (layer_key in clustering_results['layer_results'] and
            'cluster_labels' in clustering_results['layer_results'][layer_key] and
            word_idx in clustering_results['layer_results'][layer_key]['cluster_labels'] and
            0 in clustering_results['layer_results'][layer_key]['cluster_labels'][word_idx]):
            
            cluster_id = clustering_results['layer_results'][layer_key]['cluster_labels'][word_idx][0]
            path.append(f"L{layer_idx}C{cluster_id}")
        else:
            path.append(f"L{layer_idx}C?")
    
    return path

def main():
    """Run comprehensive window analysis."""
    
    print("=== Comprehensive Window Analysis ===\n")
    
    # Load data
    print("Loading data...")
    kmeans_results, ets_results, curated_data = load_all_data()
    
    # Extract cluster counts
    print("\nExtracting cluster counts...")
    kmeans_counts = extract_cluster_counts(kmeans_results)
    ets_counts = extract_cluster_counts(ets_results) if ets_results else {}
    
    # Create cluster count table
    print("\n" + "="*60)
    print("CLUSTER COUNTS BY LAYER")
    print("="*60)
    
    layers = sorted([int(k.split('_')[1]) for k in kmeans_counts.keys()])
    
    print(f"{'Layer':<10} {'K-means':<15} {'ETS':<15}")
    print("-"*40)
    
    for layer in layers:
        layer_key = f"layer_{layer}"
        kmeans_count = kmeans_counts.get(layer_key, 0)
        ets_count = ets_counts.get(layer_key, 0) if ets_counts else "N/A"
        print(f"{layer:<10} {kmeans_count:<15} {ets_count:<15}")
    
    # Calculate window metrics for K-means
    print("\n\nCalculating window metrics...")
    window_metrics = calculate_window_metrics(kmeans_results)
    
    # Create window metrics table
    print("\n" + "="*60)
    print("3-LAYER WINDOW METRICS (K-means)")
    print("="*60)
    
    print(f"{'Window':<15} {'Paths':<10} {'Entropy':<10} {'Fragment':<10} {'Coverage':<10}")
    print("-"*55)
    
    interesting_windows = []
    
    for window_key, metrics in sorted(window_metrics.items()):
        interesting_score = metrics['entropy'] * metrics['fragmentation']
        interesting_windows.append((window_key, interesting_score, metrics))
        
        print(f"{window_key:<15} {metrics['unique_paths']:<10} "
              f"{metrics['entropy']:<10.2f} {metrics['fragmentation']:<10.2f} "
              f"{metrics['coverage']:<10.2f}")
    
    # Sort by interesting score
    interesting_windows.sort(key=lambda x: x[1], reverse=True)
    
    print("\n\nMost interesting windows (by entropy × fragmentation):")
    for i, (window, score, metrics) in enumerate(interesting_windows[:3]):
        print(f"{i+1}. {window}: score={score:.3f}")
    
    # Select and trace representative words
    print("\n\n" + "="*60)
    print("REPRESENTATIVE WORD PATHS")
    print("="*60)
    
    representatives = select_representative_words(curated_data)
    
    word_paths = {}
    
    for subtype, rep_words in representatives.items():
        print(f"\n{subtype.upper()}:")
        
        for word in rep_words:
            path = trace_word_paths(word, kmeans_results)
            if path:
                path_str = " -> ".join(path)
                word_paths[word] = {
                    'subtype': subtype,
                    'path': path,
                    'path_string': path_str
                }
                print(f"  {word}: {path_str}")
    
    # Analyze path patterns
    print("\n\n" + "="*60)
    print("PATH PATTERN ANALYSIS")
    print("="*60)
    
    # Group paths by subtype
    subtype_paths = defaultdict(list)
    for word, info in word_paths.items():
        subtype_paths[info['subtype']].append(info['path_string'])
    
    # Find common patterns
    for subtype, paths in subtype_paths.items():
        print(f"\n{subtype}:")
        
        # Find most common sub-paths
        for window_size in [3, 5]:
            print(f"  Common {window_size}-layer patterns:")
            
            subpaths = []
            for path in paths:
                parts = path.split(" -> ")
                for i in range(len(parts) - window_size + 1):
                    subpath = " -> ".join(parts[i:i+window_size])
                    subpaths.append((i, subpath))
            
            # Count frequencies
            subpath_counts = Counter(sp[1] for sp in subpaths)
            
            for subpath, count in subpath_counts.most_common(3):
                if count > 1:
                    start_layers = [sp[0] for sp in subpaths if sp[1] == subpath]
                    layer_range = f"L{min(start_layers)}-L{max(start_layers)+window_size-1}"
                    print(f"    {subpath} (appears {count}x in {layer_range})")
    
    # Save results
    results = {
        'cluster_counts': {
            'kmeans': kmeans_counts,
            'ets': ets_counts
        },
        'window_metrics': window_metrics,
        'interesting_windows': [(w, s) for w, s, _ in interesting_windows[:5]],
        'representative_paths': word_paths,
        'subtype_patterns': dict(subtype_paths)
    }
    
    with open("comprehensive_window_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n\nResults saved to: comprehensive_window_analysis.json")
    
    # Generate focused report for LLM
    print("\nGenerating LLM analysis report...")
    
    report_lines = [
        "# GPT-2 Semantic Subtypes - Window Analysis Report",
        "",
        "## Key Findings",
        "",
        "### Most Interesting Windows",
        ""
    ]
    
    for i, (window, score, metrics) in enumerate(interesting_windows[:3]):
        report_lines.extend([
            f"{i+1}. **{window}** (score: {score:.3f})",
            f"   - Unique paths: {metrics['unique_paths']}",
            f"   - Entropy: {metrics['entropy']:.2f}",
            f"   - Fragmentation: {metrics['fragmentation']:.2f}",
            ""
        ])
    
    report_lines.extend([
        "### Representative Path Analysis",
        "",
        "Words from each semantic subtype follow these paths:",
        ""
    ])
    
    for subtype, paths in subtype_paths.items():
        report_lines.append(f"**{subtype}**:")
        # Show first 3 paths
        for path in paths[:3]:
            word = [w for w, info in word_paths.items() 
                   if info['path_string'] == path and info['subtype'] == subtype][0]
            report_lines.append(f"- {word}: `{path}`")
        report_lines.append("")
    
    report_lines.extend([
        "## Questions for Analysis",
        "",
        "1. Why do layers 6-8 show the highest fragmentation?",
        "2. What semantic transformations occur in the most interesting windows?",
        "3. Do words from the same semantic subtype converge or diverge in their paths?",
        "4. How do the cluster counts relate to the semantic organization at each layer?"
    ])
    
    with open("window_analysis_llm_report.md", 'w') as f:
        f.write("\n".join(report_lines))
    
    print("LLM report saved to: window_analysis_llm_report.md")

if __name__ == "__main__":
    main()