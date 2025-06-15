#!/usr/bin/env python3
"""
Cluster frequent GPT-2 tokens using proper KMeans.
Usage: python cluster_frequent_tokens.py --k 10
"""

import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from datetime import datetime
from typing import Dict, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FrequentTokenClusterer:
    def __init__(self, base_dir: Path, k: int):
        self.base_dir = base_dir
        self.k = k
        self.results_dir = base_dir / f"clustering_results_k{k}"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load token info
        logging.info("Loading frequent token information...")
        with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
            self.token_list = json.load(f)
        self.token_info = {t['token_id']: t for t in self.token_list}
        
        # Load activations
        logging.info("Loading frequent token activations...")
        self.activations = np.load(base_dir / "frequent_token_activations.npy")
        logging.info(f"Activations shape: {self.activations.shape}")
        
        self.num_tokens, self.num_layers, self.dim = self.activations.shape
    
    def cluster_layer(self, layer_activations: np.ndarray) -> Tuple[np.ndarray, float]:
        """Cluster a single layer with k clusters using proper KMeans."""
        kmeans = KMeans(
            n_clusters=self.k, 
            random_state=42, 
            n_init=20,      # Multiple initializations for stability
            max_iter=500    # Allow full convergence
        )
        labels = kmeans.fit_predict(layer_activations)
        
        # Calculate silhouette score
        sil_score = silhouette_score(
            layer_activations, 
            labels, 
            sample_size=min(5000, len(labels))
        )
        
        return labels, sil_score
    
    def cluster_all_layers(self):
        """Cluster all 12 layers and save results."""
        logging.info(f"Starting k={self.k} clustering for all layers...")
        
        all_labels = {}
        silhouette_scores = {}
        
        for layer in tqdm(range(self.num_layers), desc="Clustering layers"):
            logging.info(f"Clustering layer {layer}...")
            
            layer_activations = self.activations[:, layer, :]
            labels, sil_score = self.cluster_layer(layer_activations)
            
            # Store results
            all_labels[str(layer)] = labels.tolist()
            silhouette_scores[f"layer_{layer}"] = float(sil_score)
            
            logging.info(f"Layer {layer}: Silhouette score = {sil_score:.3f}")
        
        # Save results in the same format as k=5
        output_path = self.results_dir / f"all_labels_k{self.k}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_labels, f, indent=2)
        
        # Save metadata
        metadata = {
            "k": self.k,
            "num_tokens": self.num_tokens,
            "num_layers": self.num_layers,
            "generated_at": datetime.now().isoformat(),
            "method": "kmeans_proper",
            "silhouette_scores": silhouette_scores
        }
        
        metadata_path = self.results_dir / f"clustering_results_k{self.k}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Saved clustering results to {output_path}")
        logging.info(f"Saved metadata to {metadata_path}")
        
        # Print summary
        avg_silhouette = np.mean(list(silhouette_scores.values()))
        print(f"\nClustering complete!")
        print(f"K = {self.k}")
        print(f"Average silhouette score: {avg_silhouette:.3f}")
        print(f"Results saved to: {self.results_dir}")
        
        return all_labels, metadata


def main():
    """Run clustering with specified k value."""
    parser = argparse.ArgumentParser(description='Cluster frequent GPT-2 tokens')
    parser.add_argument('--k', type=int, required=True, help='Number of clusters')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    k = args.k
    
    # Check required files exist
    if not (base_dir / "frequent_token_activations.npy").exists():
        print("ERROR: frequent_token_activations.npy not found!")
        return
    
    if not (base_dir / "frequent_tokens_full.json").exists():
        print("ERROR: frequent_tokens_full.json not found!")
        return
    
    # Run clustering
    clusterer = FrequentTokenClusterer(base_dir, k)
    clusterer.cluster_all_layers()


if __name__ == "__main__":
    main()