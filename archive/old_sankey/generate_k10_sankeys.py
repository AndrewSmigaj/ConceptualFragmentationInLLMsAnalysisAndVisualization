#!/usr/bin/env python3
"""
Generate Sankey diagrams for k=10 clustering analysis.
Simpler direct approach without relying on complex pipeline.
"""

import json
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from experiments.gpt2.all_tokens.generate_sankey_diagrams import SankeyGenerator


def extract_paths_for_window(cluster_labels, tokens, window_layers):
    """Extract token paths for a specific window of layers."""
    paths = []
    path_counts = Counter()
    
    # For each token, extract its path through the window
    for idx, token_info in enumerate(tokens):
        path = []
        for layer in window_layers:
            layer_key = str(layer)
            if layer_key in cluster_labels and idx < len(cluster_labels[layer_key]):
                cluster_id = cluster_labels[layer_key][idx]
                path.append(cluster_id)
        
        if len(path) == len(window_layers):  # Complete path
            path_tuple = tuple(path)
            path_counts[path_tuple] += 1
            paths.append({
                'token': token_info['token_str'].strip(),
                'path': path
            })
    
    return paths, path_counts


def find_archetypal_paths(path_counts, paths, tokens, semantic_labels, top_n=15):
    """Find the most common paths and their characteristics."""
    # Sort paths by frequency
    sorted_paths = sorted(path_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    archetypal_paths = []
    for path_tuple, frequency in sorted_paths:
        path = list(path_tuple)
        
        # Find representative tokens for this path
        representative_tokens = []
        for p in paths:
            if tuple(p['path']) == path_tuple:
                representative_tokens.append(p['token'])
                if len(representative_tokens) >= 10:
                    break
        
        # Create semantic label path if labels available
        semantic_path = []
        if semantic_labels:
            # Determine which layers this path covers
            if all(l < 4 for l in range(len(path))):
                layers = [0, 1, 2, 3]
            elif all(4 <= l < 8 for l in range(len(path))):
                layers = [4, 5, 6, 7]
            else:
                layers = [8, 9, 10, 11]
            
            for i, (layer, cluster_id) in enumerate(zip(layers, path)):
                layer_key = f"layer_{layer}"
                cluster_key = f"L{layer}_C{cluster_id}"
                
                if (layer_key in semantic_labels and 
                    cluster_key in semantic_labels[layer_key]):
                    label = semantic_labels[layer_key][cluster_key]['label']
                    semantic_path.append(label)
                else:
                    semantic_path.append(f"C{cluster_id}")
        
        archetypal_paths.append({
            'path': path,
            'frequency': frequency,
            'representative_words': representative_tokens[:5],
            'example_words': representative_tokens[:5],
            'semantic_labels': semantic_path if semantic_path else None
        })
    
    return archetypal_paths


def main():
    """Generate k=10 Sankey diagrams."""
    base_dir = Path(__file__).parent
    k = 10
    
    print(f"\n{'='*60}")
    print(f"GENERATING K={k} SANKEY DIAGRAMS")
    print(f"{'='*60}")
    
    # Load clustering results
    with open(base_dir / f"clustering_results_k{k}" / f"all_labels_k{k}.json", 'r') as f:
        cluster_labels = json.load(f)
    
    # Load tokens
    with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    
    # Load semantic labels if available
    semantic_labels = {}
    labels_path = base_dir / f"llm_labels_k{k}" / f"cluster_labels_k{k}.json"
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            label_data = json.load(f)
            semantic_labels = label_data['labels']
    
    # Create results directory
    results_dir = base_dir / f"k{k}_analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Define windows
    windows = {
        'early': [0, 1, 2, 3],
        'middle': [4, 5, 6, 7],
        'late': [8, 9, 10, 11]
    }
    
    # Extract paths and create windowed analysis data
    windowed_results = {}
    
    for window_name, layers in windows.items():
        logging.info(f"Analyzing {window_name} window...")
        
        paths, path_counts = extract_paths_for_window(cluster_labels, tokens, layers)
        archetypal_paths = find_archetypal_paths(path_counts, paths, tokens, semantic_labels)
        
        windowed_results[window_name] = {
            'layers': layers,
            'total_paths': len(paths),
            'unique_paths': len(path_counts),
            'archetypal_paths': archetypal_paths
        }
        
        print(f"  {window_name}: {len(path_counts)} unique paths from {len(paths)} total")
    
    # Save windowed analysis
    output_path = results_dir / f"windowed_analysis_k{k}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(windowed_results, f, indent=2)
    
    logging.info(f"Saved windowed analysis to {output_path}")
    
    # Now generate Sankey diagrams using the parameterized generator
    generator = SankeyGenerator(base_dir, k=k)
    generator.generate_all_sankeys()
    generator.create_path_summary_table()
    
    print(f"\n{'='*60}")
    print(f"SANKEY GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nGenerated outputs in {results_dir}:")
    print(f"  - windowed_analysis_k{k}.json")
    print(f"  - sankey_early_k{k}.html")
    print(f"  - sankey_middle_k{k}.html")
    print(f"  - sankey_late_k{k}.html")
    print(f"  - top_paths_summary_k{k}.txt")


if __name__ == "__main__":
    main()