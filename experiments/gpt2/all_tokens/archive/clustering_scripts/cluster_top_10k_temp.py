#!/usr/bin/env python3
"""
Force k=5 clustering for all layers of top 10k GPT-2 tokens.
This allows us to explore finer-grained patterns beyond the binary k=2 organization.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
from datetime import datetime
from typing import Dict, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FixedKClusterer:
    def __init__(self, base_dir: Path, k: int = 5):
        self.base_dir = base_dir
        self.k = k
        self.results_dir = base_dir / f"clustering_results_k{k}"
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
    
    def cluster_layer(self, layer_activations: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """Cluster a single layer with k clusters."""
        logging.info(f"  Clustering with k={self.k}...")
        
        start_time = datetime.now()
        kmeans = KMeans(n_clusters=self.k, random_state=42, n_init=20, max_iter=500)
        labels = kmeans.fit_predict(layer_activations)
        
        # Calculate silhouette score
        sil_score = silhouette_score(layer_activations, labels, sample_size=min(5000, len(labels)))
        
        # Calculate cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
        
        # Calculate inertia
        inertia = kmeans.inertia_
        
        time_taken = (datetime.now() - start_time).total_seconds()
        logging.info(f"    Completed: silhouette={sil_score:.4f}, inertia={inertia:.2f} (took {time_taken:.1f}s)")
        
        return labels, sil_score, {
            'cluster_sizes': cluster_sizes,
            'inertia': float(inertia),
            'silhouette_score': float(sil_score),
            'time_seconds': time_taken
        }
    
    def analyze_cluster_content(self, labels: np.ndarray, layer_idx: int) -> Dict:
        """Analyze what types of tokens are in each cluster."""
        cluster_analysis = {}
        
        for cluster_id in range(self.k):
            cluster_mask = labels == cluster_id
            cluster_token_ids = np.where(cluster_mask)[0]
            
            # Analyze token types
            token_types = {}
            morphological_patterns = {}
            sample_tokens = []
            
            for idx in cluster_token_ids[:100]:  # Sample first 100 tokens
                token_id = self.token_list[idx]['token_id']
                token_info = self.token_info[token_id]
                
                # Track token type
                token_type = token_info['token_type']
                token_types[token_type] = token_types.get(token_type, 0) + 1
                
                # Track morphological patterns
                token_str = token_info['token_str'].strip()
                if token_str.endswith('ing'):
                    pattern = 'suffix_ing'
                elif token_str.endswith('ed'):
                    pattern = 'suffix_ed'
                elif token_str.endswith('ly'):
                    pattern = 'suffix_ly'
                elif token_str.endswith('er'):
                    pattern = 'suffix_er'
                elif token_str.endswith('s') and len(token_str) > 2:
                    pattern = 'suffix_s'
                elif token_info['is_punctuation']:
                    pattern = 'punctuation'
                elif token_info['is_numeric']:
                    pattern = 'numeric'
                else:
                    pattern = 'other'
                
                morphological_patterns[pattern] = morphological_patterns.get(pattern, 0) + 1
                
                # Collect sample tokens
                if len(sample_tokens) < 20:
                    sample_tokens.append(token_str)
            
            cluster_analysis[f"L{layer_idx}_C{cluster_id}"] = {
                'size': int(len(cluster_token_ids)),
                'percentage': float(len(cluster_token_ids) / len(labels) * 100),
                'token_types': token_types,
                'morphological_patterns': morphological_patterns,
                'sample_tokens': sample_tokens
            }
        
        return cluster_analysis
    
    def run_clustering(self):
        """Run k=5 clustering for all layers."""
        all_labels = {}
        layer_results = {}
        cluster_contents = {}
        
        logging.info(f"\nRunning k={self.k} clustering for all {self.num_layers} layers...\n")
        
        for layer_idx in tqdm(range(self.num_layers), desc="Clustering layers"):
            logging.info(f"\n{'='*60}")
            logging.info(f"Layer {layer_idx}:")
            logging.info(f"{'='*60}")
            
            layer_activations = self.activations[:, layer_idx, :]
            
            # Cluster the layer
            labels, sil_score, metrics = self.cluster_layer(layer_activations)
            
            # Analyze cluster contents
            content_analysis = self.analyze_cluster_content(labels, layer_idx)
            
            # Save results
            all_labels[layer_idx] = labels.tolist()
            layer_results[layer_idx] = {
                'k': self.k,
                'silhouette_score': float(sil_score),
                **metrics
            }
            cluster_contents[layer_idx] = content_analysis
            
            # Save labels as numpy array
            np.save(self.results_dir / f"labels_layer{layer_idx}_k{self.k}.npy", labels)
            
            # Print cluster summary
            logging.info(f"\nCluster sizes for layer {layer_idx}:")
            for cluster_id, info in content_analysis.items():
                logging.info(f"  {cluster_id}: {info['size']} tokens ({info['percentage']:.1f}%)")
        
        # Save all results
        logging.info("\nSaving results...")
        
        # Save all labels
        with open(self.results_dir / f"all_labels_k{self.k}.json", 'w') as f:
            json.dump(all_labels, f)
        
        # Save clustering metrics
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': f'fixed_k_{self.k}',
            'k': self.k,
            'total_tokens': self.num_tokens,
            'num_layers': self.num_layers,
            'layer_results': layer_results
        }
        
        with open(self.results_dir / f"clustering_results_k{self.k}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save cluster content analysis
        with open(self.results_dir / f"cluster_contents_k{self.k}.json", 'w') as f:
            json.dump(cluster_contents, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print(f"k={self.k} Clustering Results Summary:")
        print("="*60)
        
        # Average silhouette score
        avg_sil = np.mean([r['silhouette_score'] for r in layer_results.values()])
        print(f"\nAverage silhouette score: {avg_sil:.4f}")
        
        # Show how cluster sizes vary across layers
        print("\nCluster size stability across layers:")
        for cluster_id in range(self.k):
            sizes = []
            for layer_idx in range(self.num_layers):
                for cluster_name, info in cluster_contents[layer_idx].items():
                    if cluster_name.endswith(f"_C{cluster_id}"):
                        sizes.append(info['percentage'])
                        break
            if sizes:
                print(f"  Cluster {cluster_id}: {np.mean(sizes):.1f}% ± {np.std(sizes):.1f}%")
        
        print(f"\nResults saved to: {self.results_dir}")
        
        # Create a formatted clustering report
        self.create_clustering_report(cluster_contents)
    
    def create_clustering_report(self, cluster_contents: Dict):
        """Create a human-readable clustering report."""
        report_path = self.results_dir / f"clustering_report_k{self.k}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Top 10k GPT-2 Tokens - k={self.k} Clustering Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for layer_idx in range(self.num_layers):
                f.write(f"\n## Layer {layer_idx}\n\n")
                
                layer_content = cluster_contents[layer_idx]
                for cluster_name, info in sorted(layer_content.items()):
                    f.write(f"### {cluster_name} ({info['size']} tokens, {info['percentage']:.1f}%)\n\n")
                    
                    # Token types
                    f.write("**Token Types:**\n")
                    for token_type, count in sorted(info['token_types'].items(), 
                                                  key=lambda x: x[1], reverse=True)[:5]:
                        f.write(f"- {token_type}: {count}\n")
                    
                    # Morphological patterns
                    f.write("\n**Morphological Patterns:**\n")
                    for pattern, count in sorted(info['morphological_patterns'].items(), 
                                               key=lambda x: x[1], reverse=True)[:5]:
                        f.write(f"- {pattern}: {count}\n")
                    
                    # Sample tokens
                    f.write("\n**Sample Tokens:** ")
                    f.write(", ".join(f'"{t}"' for t in info['sample_tokens'][:10]))
                    f.write("\n\n")
        
        logging.info(f"Clustering report saved to: {report_path}")


def main():
    base_dir = Path(__file__).parent
    
    # First check if activations exist
    if not (base_dir / "top_10k_activations.npy").exists():
        logging.error("top_10k_activations.npy not found. Please run extract_top_10k_tokens.py first.")
        return
    
    clusterer = FixedKClusterer(base_dir, k=5)
    clusterer.run_clustering()


if __name__ == "__main__":
    main()