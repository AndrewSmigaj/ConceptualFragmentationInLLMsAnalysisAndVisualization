#!/usr/bin/env python3
"""
Quick analysis to find optimal k for key layers (early, middle, late).
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuickOptimalKAnalyzer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "optimal_k_quick"
        self.results_dir.mkdir(exist_ok=True)
        
        logging.info("Loading token analysis...")
        with open(base_dir / "full_token_analysis.json", 'r', encoding='utf-8') as f:
            token_list = json.load(f)
        
        # Create token_id to info mapping
        self.token_info = {}
        for token_data in token_list:
            self.token_info[token_data['token_id']] = {
                'token': token_data['token_str'],
                'type': token_data['token_type']
            }
    
    def load_all_activations(self) -> np.ndarray:
        """Load all activation chunks."""
        logging.info("Loading activation chunks...")
        all_activations = []
        chunk_files = sorted((self.base_dir / "activations").glob("activations_chunk_*.npy"))
        
        for chunk_file in tqdm(chunk_files, desc="Loading"):
            chunk = np.load(chunk_file)
            all_activations.append(chunk)
        
        activations = np.vstack(all_activations)
        logging.info(f"Loaded shape: {activations.shape}")
        return activations
    
    def test_k_range(self, layer_activations: np.ndarray, k_values: List[int]) -> Dict:
        """Test a range of k values and return scores."""
        scores = {}
        
        for k in k_values:
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                batch_size=5000,
                n_init=1,
                max_iter=30,  # Even fewer iterations
                random_state=42,
                verbose=0
            )
            
            labels = kmeans.fit_predict(layer_activations)
            
            # Sample for silhouette
            sample_size = min(3000, len(layer_activations))
            sample_idx = np.random.choice(len(layer_activations), sample_size, replace=False)
            score = silhouette_score(layer_activations[sample_idx], labels[sample_idx])
            
            scores[k] = float(score)
            logging.info(f"    k={k}: {score:.4f}")
        
        return scores
    
    def run_quick_analysis(self):
        """Analyze key layers with focused k ranges."""
        all_activations = self.load_all_activations()
        
        # Test specific layers: early (0,1), middle (5,6), late (10,11)
        test_layers = [0, 1, 5, 6, 10, 11]
        # Smaller k range for speed
        k_values = [10, 20, 30, 50, 70, 100]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_tokens': all_activations.shape[0],
            'layers_tested': test_layers,
            'k_values': k_values,
            'layer_results': {}
        }
        
        for layer_idx in test_layers:
            logging.info(f"\nLayer {layer_idx}:")
            layer_activations = all_activations[:, layer_idx, :]
            
            scores = self.test_k_range(layer_activations, k_values)
            
            # Find optimal k
            optimal_k = max(scores.items(), key=lambda x: x[1])[0]
            best_score = scores[optimal_k]
            
            results['layer_results'][layer_idx] = {
                'scores': scores,
                'optimal_k': optimal_k,
                'best_score': best_score
            }
            
            logging.info(f"  Optimal: k={optimal_k} (score={best_score:.4f})")
        
        # Save results
        with open(self.results_dir / "quick_optimal_k.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualization
        self.plot_results(results)
        
        # Generate summary
        self.generate_summary(results)
        
        return results
    
    def plot_results(self, results: Dict):
        """Create visualization of optimal k per layer."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Optimal k per layer
        layers = sorted(results['layer_results'].keys(), key=int)
        optimal_ks = [results['layer_results'][l]['optimal_k'] for l in layers]
        best_scores = [results['layer_results'][l]['best_score'] for l in layers]
        
        ax1.bar([str(l) for l in layers], optimal_ks, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Optimal k')
        ax1.set_title('Optimal Number of Clusters by Layer')
        
        # Add value labels
        for i, (layer, k) in enumerate(zip(layers, optimal_ks)):
            ax1.text(i, k + 1, str(k), ha='center', va='bottom')
        
        # Plot 2: Silhouette scores
        k_values = results['k_values']
        
        for layer in layers:
            scores = results['layer_results'][layer]['scores']
            score_values = [scores[k] for k in k_values]
            ax2.plot(k_values, score_values, 'o-', label=f'Layer {layer}', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Clustering Quality Across k Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'optimal_k_analysis.png', dpi=150)
        plt.close()
    
    def generate_summary(self, results: Dict):
        """Generate summary report."""
        with open(self.results_dir / "optimal_k_summary.md", 'w') as f:
            f.write("# Quick Optimal k Analysis for GPT-2 Layers\n\n")
            f.write("## Results Summary\n\n")
            f.write("| Layer | Type | Optimal k | Best Score |\n")
            f.write("|-------|------|-----------|------------|\n")
            
            for layer in sorted(results['layer_results'].keys(), key=int):
                layer_type = "Early" if layer <= 1 else "Middle" if layer <= 6 else "Late"
                opt_k = results['layer_results'][layer]['optimal_k']
                score = results['layer_results'][layer]['best_score']
                f.write(f"| {layer} | {layer_type} | {opt_k} | {score:.4f} |\n")
            
            f.write("\n## Key Findings\n\n")
            
            # Calculate averages
            early_layers = [l for l in results['layer_results'].keys() if l <= 1]
            middle_layers = [l for l in results['layer_results'].keys() if 1 < l <= 6]
            late_layers = [l for l in results['layer_results'].keys() if l > 6]
            
            early_k = np.mean([results['layer_results'][l]['optimal_k'] for l in early_layers])
            middle_k = np.mean([results['layer_results'][l]['optimal_k'] for l in middle_layers])
            late_k = np.mean([results['layer_results'][l]['optimal_k'] for l in late_layers])
            
            f.write(f"- **Early layers (0-1)**: Average optimal k = {early_k:.0f}\n")
            f.write(f"- **Middle layers (5-6)**: Average optimal k = {middle_k:.0f}\n")
            f.write(f"- **Late layers (10-11)**: Average optimal k = {late_k:.0f}\n")
            
            f.write("\n## Interpretation\n\n")
            f.write("The optimal k values show how GPT-2's token representations evolve:\n")
            f.write("- Early layers use simpler clustering (basic token types)\n")
            f.write("- Middle and late layers may require more clusters for complex patterns\n")
            f.write("- Positive silhouette scores in later layers indicate better-defined clusters\n")


def main():
    base_dir = Path(__file__).parent
    analyzer = QuickOptimalKAnalyzer(base_dir)
    
    results = analyzer.run_quick_analysis()
    
    logging.info(f"\nAnalysis complete! Results saved to {analyzer.results_dir}")


if __name__ == "__main__":
    main()