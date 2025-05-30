#!/usr/bin/env python3
"""
Cluster all GPT-2 tokens using their activations with k values up to 100.
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

class AllTokenClusterAnalyzer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "clustering_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load token analysis
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
    
    def cluster_layer(self, activations: np.ndarray, k: int) -> np.ndarray:
        """Cluster activations for a single layer."""
        logging.info(f"Clustering with k={k}...")
        
        # Use MiniBatchKMeans for efficiency
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=1000,
            n_init=3,
            random_state=42,
            verbose=0
        )
        
        labels = kmeans.fit_predict(activations)
        
        # Calculate silhouette score on a sample for efficiency
        if len(activations) > 10000:
            sample_idx = np.random.choice(len(activations), 10000, replace=False)
            silhouette = silhouette_score(activations[sample_idx], labels[sample_idx])
        else:
            silhouette = silhouette_score(activations, labels)
        
        logging.info(f"k={k}: Silhouette score = {silhouette:.4f}")
        
        return labels, silhouette
    
    def analyze_clusters(self, labels: np.ndarray, k: int) -> Dict:
        """Analyze cluster composition by token types."""
        cluster_analysis = defaultdict(lambda: {
            'total': 0,
            'types': Counter(),
            'has_space': 0,
            'is_alphabetic': 0,
            'is_punctuation': 0,
            'is_numeric': 0,
            'is_subword': 0,
            'languages': Counter(),
            'example_tokens': []
        })
        
        for token_id, cluster_id in enumerate(labels):
            if token_id not in self.token_info:
                continue
                
            info = self.token_info[token_id]
            analysis = cluster_analysis[int(cluster_id)]
            
            analysis['total'] += 1
            analysis['types'][info['type']] += 1
            
            if info['has_space']:
                analysis['has_space'] += 1
            if info['is_alphabetic']:
                analysis['is_alphabetic'] += 1
            if info['is_punctuation']:
                analysis['is_punctuation'] += 1
            if info['is_numeric']:
                analysis['is_numeric'] += 1
            if info['is_subword']:
                analysis['is_subword'] += 1
            
            analysis['languages'][info['language']] += 1
            
            # Keep first 10 examples
            if len(analysis['example_tokens']) < 10:
                analysis['example_tokens'].append(info['token'])
        
        # Convert to percentages and find dominant patterns
        results = {}
        for cluster_id, analysis in cluster_analysis.items():
            total = analysis['total']
            if total == 0:
                continue
                
            results[cluster_id] = {
                'total': total,
                'dominant_type': analysis['types'].most_common(1)[0] if analysis['types'] else ('unknown', 0),
                'type_distribution': dict(analysis['types']),
                'has_space_pct': analysis['has_space'] / total * 100,
                'is_alphabetic_pct': analysis['is_alphabetic'] / total * 100,
                'is_punctuation_pct': analysis['is_punctuation'] / total * 100,
                'is_numeric_pct': analysis['is_numeric'] / total * 100,
                'is_subword_pct': analysis['is_subword'] / total * 100,
                'dominant_language': analysis['languages'].most_common(1)[0] if analysis['languages'] else ('unknown', 0),
                'language_distribution': dict(analysis['languages']),
                'example_tokens': analysis['example_tokens']
            }
        
        return results
    
    def find_interesting_patterns(self, cluster_analysis: Dict, k: int) -> List[Dict]:
        """Find interesting patterns in clustering results."""
        patterns = []
        
        for cluster_id, analysis in cluster_analysis.items():
            # Pattern 1: Highly specialized clusters (>80% of one type)
            dominant_type, count = analysis['dominant_type']
            if count / analysis['total'] > 0.8:
                patterns.append({
                    'type': 'specialized_cluster',
                    'cluster_id': cluster_id,
                    'dominant_type': dominant_type,
                    'purity': count / analysis['total'],
                    'size': analysis['total'],
                    'examples': analysis['example_tokens'][:5]
                })
            
            # Pattern 2: Punctuation-heavy clusters
            if analysis['is_punctuation_pct'] > 50:
                patterns.append({
                    'type': 'punctuation_cluster',
                    'cluster_id': cluster_id,
                    'punctuation_pct': analysis['is_punctuation_pct'],
                    'size': analysis['total'],
                    'examples': analysis['example_tokens'][:5]
                })
            
            # Pattern 3: Numeric clusters
            if analysis['is_numeric_pct'] > 30:
                patterns.append({
                    'type': 'numeric_cluster',
                    'cluster_id': cluster_id,
                    'numeric_pct': analysis['is_numeric_pct'],
                    'size': analysis['total'],
                    'examples': analysis['example_tokens'][:5]
                })
            
            # Pattern 4: Non-English clusters
            dominant_lang, lang_count = analysis['dominant_language']
            if dominant_lang != 'english' and lang_count / analysis['total'] > 0.5:
                patterns.append({
                    'type': 'non_english_cluster',
                    'cluster_id': cluster_id,
                    'language': dominant_lang,
                    'language_purity': lang_count / analysis['total'],
                    'size': analysis['total'],
                    'examples': analysis['example_tokens'][:5]
                })
            
            # Pattern 5: Subword-heavy clusters
            if analysis['is_subword_pct'] > 70:
                patterns.append({
                    'type': 'subword_cluster',
                    'cluster_id': cluster_id,
                    'subword_pct': analysis['is_subword_pct'],
                    'size': analysis['total'],
                    'examples': analysis['example_tokens'][:5]
                })
        
        return patterns
    
    def run_clustering_analysis(self, k_values: List[int], layers_to_analyze: List[int] = None):
        """Run clustering analysis for specified k values and layers."""
        if layers_to_analyze is None:
            layers_to_analyze = [0, 5, 11]  # Early, middle, late layers
        
        # Load all activations
        all_activations = self.load_all_activations()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tokens': all_activations.shape[0],
            'layers_analyzed': layers_to_analyze,
            'k_values': k_values,
            'results_by_layer': {}
        }
        
        for layer_idx in layers_to_analyze:
            logging.info(f"\nAnalyzing layer {layer_idx}...")
            layer_activations = all_activations[:, layer_idx, :]
            
            layer_results = {
                'silhouette_scores': {},
                'cluster_analyses': {},
                'patterns': {}
            }
            
            for k in k_values:
                # Cluster this layer
                labels, silhouette = self.cluster_layer(layer_activations, k)
                layer_results['silhouette_scores'][k] = float(silhouette)
                
                # Analyze clusters
                cluster_analysis = self.analyze_clusters(labels, k)
                layer_results['cluster_analyses'][k] = cluster_analysis
                
                # Find patterns
                patterns = self.find_interesting_patterns(cluster_analysis, k)
                layer_results['patterns'][k] = patterns
                
                # Save labels for this k
                np.save(self.results_dir / f"labels_layer{layer_idx}_k{k}.npy", labels)
            
            results['results_by_layer'][layer_idx] = layer_results
        
        # Save full results
        with open(self.results_dir / "clustering_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def plot_silhouette_scores(self, results: Dict):
        """Plot silhouette scores across k values and layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for layer_idx, layer_results in results['results_by_layer'].items():
            k_values = []
            scores = []
            
            for k, score in sorted(layer_results['silhouette_scores'].items(), key=lambda x: int(x[0])):
                k_values.append(int(k))
                scores.append(score)
            
            ax.plot(k_values, scores, marker='o', label=f'Layer {layer_idx}')
        
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'silhouette_scores.png', dpi=150)
        plt.close()
    
    def generate_summary_report(self, results: Dict):
        """Generate a human-readable summary report."""
        report_lines = [
            "# GPT-2 All Token Clustering Analysis",
            f"\nAnalyzed {results['total_tokens']} tokens",
            f"Layers analyzed: {results['layers_analyzed']}",
            f"K values tested: {results['k_values']}",
            "\n## Key Findings\n"
        ]
        
        for layer_idx, layer_results in results['results_by_layer'].items():
            report_lines.append(f"\n### Layer {layer_idx}")
            
            # Best k by silhouette score
            best_k = max(layer_results['silhouette_scores'].items(), key=lambda x: x[1])
            report_lines.append(f"Best k by silhouette score: k={best_k[0]} (score={best_k[1]:.4f})")
            
            # Interesting patterns for each k
            for k in sorted(results['k_values']):
                patterns = layer_results['patterns'].get(str(k), [])
                if patterns:
                    report_lines.append(f"\n**k={k} patterns:**")
                    
                    pattern_types = defaultdict(list)
                    for pattern in patterns:
                        pattern_types[pattern['type']].append(pattern)
                    
                    for ptype, plist in pattern_types.items():
                        report_lines.append(f"- {len(plist)} {ptype.replace('_', ' ')} clusters found")
                        if ptype == 'specialized_cluster':
                            types_found = set(p['dominant_type'] for p in plist)
                            report_lines.append(f"  Types: {', '.join(sorted(types_found))}")
        
        with open(self.results_dir / "clustering_summary.md", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))


def main():
    base_dir = Path(__file__).parent
    
    analyzer = AllTokenClusterAnalyzer(base_dir)
    
    # K values up to 100 as requested
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Run analysis
    results = analyzer.run_clustering_analysis(k_values)
    
    # Generate visualizations and report
    analyzer.plot_silhouette_scores(results)
    analyzer.generate_summary_report(results)
    
    logging.info("Analysis complete! Results saved to clustering_results/")


if __name__ == "__main__":
    main()