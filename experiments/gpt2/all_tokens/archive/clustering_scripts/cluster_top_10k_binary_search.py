#!/usr/bin/env python3
"""
Cluster top 10k GPT-2 tokens using binary search to find optimal k.
Optimizes for silhouette score instead of gap statistic.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from datetime import datetime
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BinarySearchClusterer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "clustering_results_per_layer"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load token info
        logging.info("Loading top 10k token information...")
        with open(base_dir / "top_10k_tokens_full.json", 'r', encoding='utf-8') as f:
            self.token_list = json.load(f)
        self.token_info = {t['token_id']: t for t in self.token_list}
        
        # Load activations
        logging.info("Loading top 10k activations...")
        self.activations = np.load(base_dir / "top_10k_activations.npy")
        logging.info(f"Activations shape: {self.activations.shape}")
        
        self.num_tokens, self.num_layers, self.dim = self.activations.shape
    
    def evaluate_k(self, layer_activations: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
        """Evaluate clustering with k clusters, return silhouette score and labels."""
        logging.info(f"  Testing k={k}...")
        
        start_time = datetime.now()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(layer_activations)
        
        # Calculate silhouette score
        if k > 1:
            sil_score = silhouette_score(layer_activations, labels, sample_size=min(5000, len(labels)))
        else:
            sil_score = -1.0  # Invalid for k=1
        
        time_taken = (datetime.now() - start_time).total_seconds()
        logging.info(f"    k={k}: silhouette={sil_score:.4f} (took {time_taken:.1f}s)")
        
        return sil_score, labels
    
    def binary_search_optimal_k(self, layer_activations: np.ndarray, 
                               k_min: int = 2, k_max: int = 100) -> Tuple[int, float, np.ndarray]:
        """
        Use binary search to find k with best silhouette score.
        
        Strategy:
        1. Test anchor points to understand the curve
        2. Focus on promising regions
        3. Test ALL values in the best region to ensure we don't miss the optimum
        """
        # Cache results to avoid recomputation
        cache = {}
        
        def eval_with_cache(k):
            if k not in cache:
                score, labels = self.evaluate_k(layer_activations, k)
                cache[k] = (score, labels)
            return cache[k]
        
        # Step 1: Test anchor points to understand the curve
        anchor_ks = [k_min, 5, 10, 15, 20, 30, 40, 50, 75, k_max]
        anchor_ks = [k for k in anchor_ks if k_min <= k <= k_max]
        
        logging.info(f"  Testing anchor points: {anchor_ks}")
        for k in anchor_ks:
            eval_with_cache(k)
        
        # Find the k with best score so far
        best_k = max(cache.keys(), key=lambda k: cache[k][0])
        best_score = cache[best_k][0]
        
        logging.info(f"  Initial best: k={best_k} with score={best_score:.4f}")
        
        # Step 2: Identify promising regions
        # Find all local maxima in anchor points
        promising_regions = []
        
        # Always test around the global best
        search_radius = 10  # Test ±10 around best points
        left = max(k_min, best_k - search_radius)
        right = min(k_max, best_k + search_radius)
        promising_regions.append((left, right))
        
        # Also check if there are other local maxima worth exploring
        sorted_anchors = sorted(anchor_ks)
        for i in range(1, len(sorted_anchors) - 1):
            k = sorted_anchors[i]
            score = cache[k][0]
            prev_score = cache[sorted_anchors[i-1]][0]
            next_score = cache[sorted_anchors[i+1]][0]
            
            # Local maximum and reasonably good score
            if score > prev_score and score > next_score and score > 0.5 * best_score:
                left = max(k_min, k - search_radius // 2)
                right = min(k_max, k + search_radius // 2)
                if (left, right) not in promising_regions:
                    promising_regions.append((left, right))
        
        logging.info(f"  Exploring {len(promising_regions)} promising regions")
        
        # Step 3: Test ALL values in promising regions
        for region_idx, (left, right) in enumerate(promising_regions):
            logging.info(f"  Region {region_idx + 1}: testing k={left} to k={right}")
            for k in range(left, right + 1):
                if k not in cache:  # Don't retest
                    eval_with_cache(k)
        
        # Find final best
        best_k = max(cache.keys(), key=lambda k: cache[k][0])
        best_score, best_labels = cache[best_k]
        
        # Log all tested values for transparency
        tested_ks = sorted(cache.keys())
        logging.info(f"  Tested {len(tested_ks)} values: {tested_ks}")
        
        return best_k, best_score, best_labels
    
    def run_clustering(self):
        """Run binary search clustering for all layers."""
        all_labels = {}
        layer_results = {}
        
        for layer_idx in range(self.num_layers):
            logging.info(f"\n{'='*60}")
            logging.info(f"Layer {layer_idx}:")
            logging.info(f"{'='*60}")
            
            layer_activations = self.activations[:, layer_idx, :]
            
            # Use binary search to find optimal k
            best_k, best_score, labels = self.binary_search_optimal_k(
                layer_activations, k_min=2, k_max=100
            )
            
            logging.info(f"\nOptimal k for layer {layer_idx}: {best_k} (silhouette={best_score:.4f})")
            
            # Save results
            all_labels[layer_idx] = labels.tolist()
            layer_results[layer_idx] = {
                'optimal_k': best_k,
                'silhouette_score': float(best_score),
                'method': 'binary_search'
            }
            
            # Save labels
            np.save(self.results_dir / f"labels_layer{layer_idx}_k{best_k}.npy", labels)
        
        # Save all results
        with open(self.results_dir / "optimal_labels_all_layers_top10k.json", 'w') as f:
            json.dump(all_labels, f)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'binary_search_silhouette',
            'total_tokens': self.num_tokens,
            'num_layers': self.num_layers,
            'layer_results': layer_results
        }
        
        with open(self.results_dir / "per_layer_binary_search_results_top10k.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("Binary Search Clustering Results:")
        print("="*60)
        for layer_idx in range(self.num_layers):
            result = layer_results[layer_idx]
            print(f"Layer {layer_idx}: k={result['optimal_k']}, silhouette={result['silhouette_score']:.4f}")
        
        # Print average k and silhouette
        avg_k = np.mean([r['optimal_k'] for r in layer_results.values()])
        avg_sil = np.mean([r['silhouette_score'] for r in layer_results.values()])
        print(f"\nAverage k: {avg_k:.1f}")
        print(f"Average silhouette: {avg_sil:.4f}")


def main():
    base_dir = Path(__file__).parent
    clusterer = BinarySearchClusterer(base_dir)
    clusterer.run_clustering()


if __name__ == "__main__":
    main()
