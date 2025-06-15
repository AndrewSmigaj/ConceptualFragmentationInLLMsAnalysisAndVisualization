#!/usr/bin/env python3
"""
Canonical script for clustering GPT-2 tokens per layer using proper KMeans.
Usage: python cluster_tokens.py --k 10
"""

import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TokenClusterAnalyzer:
    def __init__(self, base_dir: Path, k: int):
        self.base_dir = base_dir
        self.k = k
        self.results_dir = base_dir / "clustering_results" / f"k{k}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokens with activations
        logging.info("Loading frequent tokens with activations...")
        with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
            self.tokens = json.load(f)
        
        logging.info(f"Loaded {len(self.tokens)} tokens for k={k} clustering")
    
    def cluster_layer(self, layer: int) -> Tuple[np.ndarray, float]:
        """Cluster tokens for a specific layer using proper KMeans."""
        logging.info(f"Clustering layer {layer} with k={self.k}...")
        
        # Extract activations for this layer
        activations = []
        for token in self.tokens:
            if f'layer_{layer}' in token:
                activations.append(token[f'layer_{layer}'])
            else:
                # Handle missing data with zeros
                activations.append([0.0] * 768)  # GPT-2 hidden size
        
        activations = np.array(activations)
        logging.info(f"Layer {layer}: Clustering {activations.shape[0]} tokens with {activations.shape[1]} features")
        
        # Perform proper K-means clustering
        kmeans = KMeans(
            n_clusters=self.k, 
            random_state=42, 
            n_init=10,      # Multiple initializations for stability
            max_iter=300    # Allow full convergence
        )
        labels = kmeans.fit_predict(activations)
        
        # Calculate silhouette score
        if len(set(labels)) > 1:
            sil_score = silhouette_score(activations, labels)
        else:
            sil_score = -1.0
        
        logging.info(f"Layer {layer}: Silhouette score = {sil_score:.3f}")
        
        return labels, sil_score
    
    def analyze_all_layers(self):
        """Cluster all layers and save results."""
        logging.info(f"Starting k={self.k} clustering for all 12 layers...")
        
        all_labels = {}
        silhouette_scores = {}
        
        for layer in range(12):
            labels, sil_score = self.cluster_layer(layer)
            
            all_labels[f"layer_{layer}"] = labels.tolist()
            silhouette_scores[f"layer_{layer}"] = sil_score
        
        # Save clustering results
        results = {
            "metadata": {
                "k": self.k,
                "num_tokens": len(self.tokens),
                "num_layers": 12,
                "generated_at": datetime.now().isoformat(),
                "method": "kmeans_proper"
            },
            "silhouette_scores": silhouette_scores,
            **all_labels
        }
        
        output_path = self.results_dir / "cluster_labels.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Saved clustering results to {output_path}")
        
        # Create analysis summary
        self.create_analysis_summary(silhouette_scores)
        
        return results
    
    def create_analysis_summary(self, silhouette_scores: Dict[str, float]):
        """Create a summary of clustering quality."""
        summary_lines = [
            f"K={self.k} CLUSTERING ANALYSIS SUMMARY",
            "=" * 50,
            "",
            f"Number of clusters per layer: {self.k}",
            f"Total tokens analyzed: {len(self.tokens)}",
            f"Clustering method: KMeans (proper, n_init=10, max_iter=300)",
            "",
            "SILHOUETTE SCORES BY LAYER:",
            "-" * 30
        ]
        
        avg_score = 0
        valid_scores = 0
        
        for layer in range(12):
            layer_key = f"layer_{layer}"
            score = silhouette_scores[layer_key]
            summary_lines.append(f"Layer {layer:2d}: {score:6.3f}")
            
            if score > -1:
                avg_score += score
                valid_scores += 1
        
        if valid_scores > 0:
            avg_score /= valid_scores
            summary_lines.extend([
                "",
                f"Average silhouette score: {avg_score:.3f}",
                "",
                "Quality interpretation:",
                "  > 0.7: Strong clustering",
                "  > 0.5: Reasonable clustering", 
                "  > 0.3: Weak clustering",
                "  < 0.3: Poor clustering"
            ])
        
        # Save summary
        summary_path = self.results_dir / "clustering_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        logging.info(f"Saved clustering summary to {summary_path}")
        
        # Print to console
        print("\n" + "\n".join(summary_lines))


def main():
    """Run clustering analysis with specified k."""
    parser = argparse.ArgumentParser(description='Cluster GPT-2 tokens per layer using KMeans')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory containing token data')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    k = args.k
    
    print(f"\n{'='*60}")
    print(f"STARTING K={k} CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    
    # Check if required data exists
    if not (base_dir / "frequent_tokens_full.json").exists():
        print(f"\nERROR: frequent_tokens_full.json not found in {base_dir}!")
        print("Please run token extraction first.")
        return
    
    # Initialize analyzer
    analyzer = TokenClusterAnalyzer(base_dir, k=k)
    
    # Run clustering analysis
    results = analyzer.analyze_all_layers()
    
    print(f"\n{'='*60}")
    print(f"K={k} CLUSTERING COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {analyzer.results_dir}")
    print(f"Next step: Run analysis pipeline with --k {k}")


if __name__ == "__main__":
    main()