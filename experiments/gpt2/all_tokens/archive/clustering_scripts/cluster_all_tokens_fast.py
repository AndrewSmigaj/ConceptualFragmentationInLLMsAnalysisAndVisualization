#!/usr/bin/env python3
"""
Fast clustering analysis for all GPT-2 tokens with k values up to 100.
Uses sampling for efficiency.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FastTokenClusterAnalyzer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "clustering_results_fast"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load token analysis
        logging.info("Loading token analysis...")
        with open(base_dir / "full_token_analysis.json", 'r', encoding='utf-8') as f:
            token_list = json.load(f)
        
        # Create token_id to info mapping
        self.token_info = {}
        for token_data in token_list:
            self.token_info[token_data['token_id']] = {
                'token': token_data['token_str'],
                'type': token_data['token_type'],
                'has_space': token_data.get('has_leading_space', False),
                'is_alphabetic': token_data.get('is_alphabetic', False),
                'is_punctuation': token_data.get('is_punctuation', False),
                'is_numeric': token_data.get('is_numeric', False),
                'is_subword': token_data.get('is_subword', False),
                'language': token_data.get('likely_language', 'unknown')
            }
            
        logging.info(f"Loaded info for {len(self.token_info)} tokens")
    
    def load_all_activations(self) -> np.ndarray:
        """Load all activation chunks and concatenate them."""
        logging.info("Loading all activation chunks...")
        
        all_activations = []
        chunk_files = sorted((self.base_dir / "activations").glob("activations_chunk_*.npy"))
        
        for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
            chunk = np.load(chunk_file)
            all_activations.append(chunk)
        
        # Concatenate all chunks
        activations = np.vstack(all_activations)
        logging.info(f"Loaded activations shape: {activations.shape}")
        
        return activations
    
    def cluster_layer_fast(self, activations: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
        """Cluster activations for a single layer using fast methods."""
        logging.info(f"Clustering with k={k}...")
        
        # Use MiniBatchKMeans with aggressive settings for speed
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=min(5000, len(activations) // 10),
            n_init=1,  # Single init for speed
            max_iter=50,  # Fewer iterations
            random_state=42,
            verbose=0
        )
        
        labels = kmeans.fit_predict(activations)
        
        # Calculate silhouette score on a smaller sample for speed
        sample_size = min(5000, len(activations))
        sample_idx = np.random.choice(len(activations), sample_size, replace=False)
        silhouette = silhouette_score(activations[sample_idx], labels[sample_idx])
        
        logging.info(f"k={k}: Silhouette score = {silhouette:.4f}")
        
        return labels, silhouette
    
    def analyze_clusters_fast(self, labels: np.ndarray, k: int) -> Dict:
        """Fast cluster composition analysis focusing on key statistics."""
        cluster_stats = defaultdict(lambda: {
            'total': 0,
            'types': Counter(),
            'has_space': 0,
            'is_punctuation': 0,
            'is_numeric': 0,
            'is_special': 0,
            'examples': []
        })
        
        for token_id, cluster_id in enumerate(labels):
            if token_id not in self.token_info:
                continue
                
            info = self.token_info[token_id]
            stats = cluster_stats[int(cluster_id)]
            
            stats['total'] += 1
            stats['types'][info['type']] += 1
            
            if info['has_space']:
                stats['has_space'] += 1
            if info['is_punctuation']:
                stats['is_punctuation'] += 1
            if info['is_numeric']:
                stats['is_numeric'] += 1
            if info['type'] == 'special':
                stats['is_special'] += 1
            
            # Keep first 5 examples only
            if len(stats['examples']) < 5:
                stats['examples'].append(info['token'])
        
        # Convert to summary statistics
        results = {}
        for cluster_id, stats in cluster_stats.items():
            total = stats['total']
            if total == 0:
                continue
                
            dominant_type = stats['types'].most_common(1)[0] if stats['types'] else ('unknown', 0)
            
            results[cluster_id] = {
                'total': total,
                'dominant_type': dominant_type[0],
                'dominant_type_pct': dominant_type[1] / total * 100,
                'has_space_pct': stats['has_space'] / total * 100,
                'is_punctuation_pct': stats['is_punctuation'] / total * 100,
                'is_numeric_pct': stats['is_numeric'] / total * 100,
                'is_special_pct': stats['is_special'] / total * 100,
                'examples': stats['examples']
            }
        
        return results
    
    def run_fast_analysis(self, k_values: List[int], layer_idx: int = 11):
        """Run fast clustering analysis on a single layer."""
        # Load all activations
        all_activations = self.load_all_activations()
        
        # Focus on final layer by default (most meaningful)
        logging.info(f"\nAnalyzing layer {layer_idx} (final layer)...")
        layer_activations = all_activations[:, layer_idx, :]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tokens': all_activations.shape[0],
            'layer_analyzed': layer_idx,
            'k_values': k_values,
            'silhouette_scores': {},
            'cluster_summaries': {},
            'interesting_findings': []
        }
        
        best_silhouette = -1
        best_k = None
        
        for k in k_values:
            # Cluster this k value
            labels, silhouette = self.cluster_layer_fast(layer_activations, k)
            results['silhouette_scores'][k] = float(silhouette)
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
            
            # Analyze clusters
            cluster_analysis = self.analyze_clusters_fast(labels, k)
            
            # Save summary statistics
            summary = {
                'num_clusters': len(cluster_analysis),
                'specialized_clusters': 0,
                'punctuation_clusters': 0,
                'numeric_clusters': 0,
                'special_clusters': 0
            }
            
            for cluster_id, stats in cluster_analysis.items():
                if stats['dominant_type_pct'] > 80:
                    summary['specialized_clusters'] += 1
                if stats['is_punctuation_pct'] > 50:
                    summary['punctuation_clusters'] += 1
                if stats['is_numeric_pct'] > 30:
                    summary['numeric_clusters'] += 1
                if stats['is_special_pct'] > 20:
                    summary['special_clusters'] += 1
            
            results['cluster_summaries'][k] = summary
            
            # Save labels for best k
            if k == best_k:
                np.save(self.results_dir / f"best_labels_k{k}.npy", labels)
                
                # Save detailed analysis for best k
                with open(self.results_dir / f"best_cluster_analysis_k{k}.json", 'w') as f:
                    json.dump(cluster_analysis, f, indent=2)
        
        results['best_k'] = best_k
        results['best_silhouette'] = float(best_silhouette)
        
        # Find interesting patterns
        for k, summary in results['cluster_summaries'].items():
            if summary['specialized_clusters'] > k * 0.5:  # More than 50% specialized
                results['interesting_findings'].append(
                    f"k={k}: {summary['specialized_clusters']}/{k} clusters are highly specialized (>80% single type)"
                )
            if summary['punctuation_clusters'] > 5:
                results['interesting_findings'].append(
                    f"k={k}: {summary['punctuation_clusters']} punctuation-dominated clusters"
                )
        
        # Save results
        with open(self.results_dir / "fast_clustering_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plot
        self.plot_results(results)
        
        # Generate summary report
        self.generate_report(results)
        
        return results
    
    def plot_results(self, results: Dict):
        """Generate visualization plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Silhouette scores
        k_values = sorted(results['silhouette_scores'].keys())
        scores = [results['silhouette_scores'][k] for k in k_values]
        
        ax1.plot(k_values, scores, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title(f'Clustering Quality - Layer {results["layer_analyzed"]}')
        ax1.grid(True, alpha=0.3)
        
        # Highlight best k
        best_k = results['best_k']
        best_score = results['best_silhouette']
        ax1.plot(best_k, best_score, 'r*', markersize=15, label=f'Best k={best_k}')
        ax1.legend()
        
        # Plot 2: Cluster specialization
        specialized_counts = [results['cluster_summaries'][k]['specialized_clusters'] for k in k_values]
        ax2.bar(k_values, specialized_counts, alpha=0.7, color='green')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Number of Specialized Clusters')
        ax2.set_title('Highly Specialized Clusters (>80% single type)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'clustering_analysis.png', dpi=150)
        plt.close()
    
    def generate_report(self, results: Dict):
        """Generate human-readable report."""
        report_lines = [
            "# GPT-2 All Token Fast Clustering Analysis",
            f"\nAnalyzed {results['total_tokens']:,} tokens",
            f"Layer analyzed: {results['layer_analyzed']} (final layer)",
            f"K values tested: {results['k_values']}",
            f"\n## Best Configuration",
            f"- Best k: {results['best_k']}",
            f"- Best silhouette score: {results['best_silhouette']:.4f}",
            "\n## Clustering Results Summary\n"
        ]
        
        # Add table of results
        report_lines.append("| k | Silhouette | Specialized | Punctuation | Numeric | Special |")
        report_lines.append("|---|------------|-------------|-------------|---------|---------|")
        
        for k in sorted(results['k_values']):
            summary = results['cluster_summaries'][k]
            silhouette = results['silhouette_scores'][k]
            report_lines.append(
                f"| {k} | {silhouette:.4f} | "
                f"{summary['specialized_clusters']} | "
                f"{summary['punctuation_clusters']} | "
                f"{summary['numeric_clusters']} | "
                f"{summary['special_clusters']} |"
            )
        
        # Add interesting findings
        if results['interesting_findings']:
            report_lines.append("\n## Interesting Findings\n")
            for finding in results['interesting_findings']:
                report_lines.append(f"- {finding}")
        
        # Add interpretation
        report_lines.extend([
            "\n## Interpretation",
            "",
            "The clustering analysis reveals how GPT-2 organizes its full vocabulary:",
            "",
            "1. **Token Type Organization**: The presence of specialized clusters suggests GPT-2",
            "   groups tokens by their linguistic function (punctuation, numbers, words).",
            "",
            "2. **Subword Patterns**: With higher k values, we should see whether subwords",
            "   cluster by morphological patterns (prefixes, suffixes, stems).",
            "",
            "3. **Optimal Clustering**: The best k value balances between too few clusters",
            "   (mixing different token types) and too many (overfitting to individual tokens)."
        ])
        
        with open(self.results_dir / "clustering_report.md", 'w') as f:
            f.write('\n'.join(report_lines))


def main():
    base_dir = Path(__file__).parent
    analyzer = FastTokenClusterAnalyzer(base_dir)
    
    # K values up to 100 as requested
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Run fast analysis on final layer
    results = analyzer.run_fast_analysis(k_values, layer_idx=11)
    
    logging.info(f"\nAnalysis complete! Best k={results['best_k']} with silhouette={results['best_silhouette']:.4f}")
    logging.info(f"Results saved to {analyzer.results_dir}")


if __name__ == "__main__":
    main()