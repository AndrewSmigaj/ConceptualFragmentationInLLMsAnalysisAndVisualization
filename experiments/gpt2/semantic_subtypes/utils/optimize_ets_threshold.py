#!/usr/bin/env python3
"""
Find optimal ETS threshold using multi-criteria optimization.
Implements a two-stage search: coarse then fine-grained.
"""

import sys
import pickle
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import Counter

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

class ETSThresholdOptimizer:
    """Optimizer for finding the best ETS threshold for semantic clustering."""
    
    def __init__(self, activations_data: Dict, word_to_subtype: Dict):
        """
        Initialize optimizer with activation data.
        
        Args:
            activations_data: Dictionary with activation vectors
            word_to_subtype: Mapping from words to their semantic subtypes
        """
        self.activations_data = activations_data
        self.word_to_subtype = word_to_subtype
        self.results = []
        
    def extract_layer_activations(self, layer_idx: int, sample_indices: Optional[List[int]] = None):
        """Extract activations for a specific layer."""
        activations = []
        words = []
        indices = []
        
        # Get all sentence indices
        all_indices = sorted(self.activations_data['activations'].keys())
        
        # Use sample or all indices
        indices_to_use = sample_indices if sample_indices is not None else all_indices
        
        for sent_idx in indices_to_use:
            if sent_idx in self.activations_data['activations']:
                # For single-word inputs, token_idx is 0
                token_idx = 0
                if token_idx in self.activations_data['activations'][sent_idx]:
                    if layer_idx in self.activations_data['activations'][sent_idx][token_idx]:
                        activations.append(
                            self.activations_data['activations'][sent_idx][token_idx][layer_idx]
                        )
                        if sent_idx < len(self.activations_data['sentences']):
                            words.append(self.activations_data['sentences'][sent_idx])
                        indices.append(sent_idx)
        
        return activations, words, indices
    
    def calculate_semantic_coherence(self, cluster_labels: List[int], words: List[str]) -> float:
        """
        Calculate how well clusters align with semantic subtypes.
        
        Returns:
            coherence_score: 0 to 1, where 1 means perfect alignment
        """
        if not words or len(set(cluster_labels)) <= 1:
            return 0.0
        
        # Group words by cluster
        cluster_to_words = {}
        for label, word in zip(cluster_labels, words):
            if label not in cluster_to_words:
                cluster_to_words[label] = []
            cluster_to_words[label].append(word)
        
        # Calculate purity for each cluster
        total_purity = 0.0
        total_words = 0
        
        for cluster_words in cluster_to_words.values():
            # Get subtypes for words in this cluster
            subtypes = [self.word_to_subtype.get(w, 'unknown') for w in cluster_words]
            
            # Find most common subtype
            if subtypes:
                subtype_counts = Counter(subtypes)
                most_common_count = subtype_counts.most_common(1)[0][1]
                purity = most_common_count / len(subtypes)
                total_purity += purity * len(subtypes)
                total_words += len(subtypes)
        
        return total_purity / total_words if total_words > 0 else 0.0
    
    def calculate_interpretability_score(self, n_clusters: int, n_samples: int) -> float:
        """
        Calculate interpretability based on number of clusters.
        
        Prefers 10-50 clusters for 774 words (1.3% - 6.5%).
        """
        if n_samples == 0:
            return 0.0
            
        cluster_ratio = n_clusters / n_samples
        
        # Ideal range: 1.3% to 6.5% of samples as clusters (10-50 for 774 words)
        if 0.013 <= cluster_ratio <= 0.065:
            return 1.0
        elif cluster_ratio < 0.013:
            # Too few clusters - linear penalty
            return max(0, cluster_ratio / 0.013)
        else:
            # Too many clusters - exponential decay
            excess = cluster_ratio - 0.065
            return max(0, np.exp(-10 * excess))
    
    def search_thresholds(self, percentiles: np.ndarray, 
                         layer_idx: int = 11,
                         sample_size: Optional[int] = None) -> List[Dict]:
        """
        Search across threshold percentiles and evaluate clustering quality.
        
        Args:
            percentiles: Array of percentiles to test
            layer_idx: Which layer to test on (default: 11 - highest semantic content)
            sample_size: Number of samples to use (None = all)
            
        Returns:
            List of result dictionaries
        """
        from gpt2_pivot_clusterer import GPT2PivotClusterer
        
        # Get sample indices if needed
        all_indices = sorted(self.activations_data['activations'].keys())
        if sample_size and sample_size < len(all_indices):
            # Stratified sampling by subtype
            sample_indices = self._get_stratified_sample(sample_size)
        else:
            sample_indices = None
        
        # Extract activations for the layer
        activations, words, indices = self.extract_layer_activations(layer_idx, sample_indices)
        
        print(f"\nSearching thresholds on layer {layer_idx} with {len(activations)} samples")
        print("Percentile | Clusters | Silhouette | Coherence | Interp. | Quality")
        print("-" * 70)
        
        results = []
        
        for percentile in percentiles:
            clusterer = GPT2PivotClusterer(
                clustering_method='ets',
                threshold_percentile=float(percentile),
                random_state=42
            )
            
            if not clusterer._setup_sklearn() or not clusterer.ets_available:
                print(f"  {percentile:.2f}    |  ETS not available")
                continue
                
            try:
                labels, centers, n_clusters, silhouette = clusterer._cluster_with_ets(activations)
                
                # Skip if clustering produced too few or all singleton clusters
                if n_clusters < 2 or n_clusters >= len(activations) * 0.95:
                    print(f"  {percentile:.2f}    |   {n_clusters:4d}   |   Skip - extreme clustering")
                    continue
                
                # Calculate metrics
                coherence = self.calculate_semantic_coherence(labels, words)
                interpretability = self.calculate_interpretability_score(n_clusters, len(activations))
                
                # Composite quality score with adjusted weights
                # Higher weight on interpretability to avoid too many clusters
                quality = (
                    0.3 * max(0, silhouette) +  # Silhouette can be negative
                    0.4 * coherence + 
                    0.3 * interpretability
                )
                
                result = {
                    'percentile': percentile,
                    'n_clusters': n_clusters,
                    'silhouette': silhouette,
                    'coherence': coherence,
                    'interpretability': interpretability,
                    'quality': quality,
                    'cluster_ratio': n_clusters / len(activations)
                }
                
                results.append(result)
                
                print(f"  {percentile:.2f}    |   {n_clusters:4d}   |   {silhouette:6.3f} |"
                      f"   {coherence:.3f}   |  {interpretability:.3f}  | {quality:.3f}")
                
            except Exception as e:
                print(f"  {percentile:.2f}    |  Error: {str(e)[:40]}...")
        
        return results
    
    def _get_stratified_sample(self, sample_size: int) -> List[int]:
        """Get stratified sample indices ensuring representation from each subtype."""
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Group indices by subtype
        subtype_indices = {}
        for idx in sorted(self.activations_data['activations'].keys()):
            if idx < len(self.activations_data['sentences']):
                word = self.activations_data['sentences'][idx]
                subtype = self.word_to_subtype.get(word, 'unknown')
                if subtype not in subtype_indices:
                    subtype_indices[subtype] = []
                subtype_indices[subtype].append(idx)
        
        # Sample proportionally from each subtype
        samples_per_subtype = max(1, sample_size // len(subtype_indices))
        sample_indices = []
        
        for subtype, indices in subtype_indices.items():
            n_samples = min(samples_per_subtype, len(indices))
            if n_samples > 0:
                sampled = np.random.choice(indices, n_samples, replace=False)
                sample_indices.extend(sampled)
        
        # Shuffle and return requested size
        np.random.shuffle(sample_indices)
        return sample_indices[:sample_size]
    
    def find_optimal_threshold(self) -> Tuple[float, Dict]:
        """
        Run two-stage search to find optimal threshold.
        
        Returns:
            optimal_threshold: Best threshold percentile
            all_results: Dictionary with all search results
        """
        # Stage 1: Coarse search
        print("\n" + "="*70)
        print("STAGE 1: Coarse Search")
        print("="*70)
        
        coarse_percentiles = np.arange(0.1, 1.0, 0.1)
        coarse_results = self.search_thresholds(
            coarse_percentiles, 
            layer_idx=11,
            sample_size=200  # Use sample for efficiency
        )
        
        if not coarse_results:
            print("No valid results from coarse search!")
            return None, {}
        
        # Find best coarse result
        best_coarse = max(coarse_results, key=lambda x: x['quality'])
        print(f"\nBest coarse threshold: {best_coarse['percentile']:.1f} "
              f"(quality: {best_coarse['quality']:.3f}, "
              f"clusters: {best_coarse['n_clusters']})")
        
        # Stage 2: Fine search around best region
        print("\n" + "="*70)
        print("STAGE 2: Fine Search")
        print("="*70)
        
        # Wider search range to find better options
        fine_percentiles = np.arange(
            max(0.05, best_coarse['percentile'] - 0.2),
            min(0.98, best_coarse['percentile'] + 0.2),
            0.02
        )
        
        fine_results = self.search_thresholds(
            fine_percentiles,
            layer_idx=11,
            sample_size=None  # Use all samples
        )
        
        # Combine results
        all_results = {
            'coarse_results': coarse_results,
            'fine_results': fine_results,
            'optimal': None
        }
        
        # Find optimal from all results
        valid_results = [r for r in (coarse_results + fine_results) 
                        if 10 <= r['n_clusters'] <= 100]  # Reasonable range
        
        if valid_results:
            optimal_result = max(valid_results, key=lambda x: x['quality'])
            all_results['optimal'] = optimal_result
            
            print("\n" + "="*70)
            print("OPTIMAL THRESHOLD FOUND")
            print("="*70)
            print(f"Threshold percentile: {optimal_result['percentile']:.2f}")
            print(f"Number of clusters: {optimal_result['n_clusters']}")
            print(f"Silhouette score: {optimal_result['silhouette']:.3f}")
            print(f"Semantic coherence: {optimal_result['coherence']:.3f}")
            print(f"Interpretability: {optimal_result['interpretability']:.3f}")
            print(f"Overall quality: {optimal_result['quality']:.3f}")
            
            return optimal_result['percentile'], all_results
        else:
            print("\nNo results in reasonable range (10-100 clusters)")
            # Return best available even if outside range
            if fine_results:
                optimal_result = max(fine_results, key=lambda x: x['quality'])
            else:
                optimal_result = best_coarse
            all_results['optimal'] = optimal_result
            
            return optimal_result['percentile'], all_results
    
    def visualize_results(self, results_dict: Dict, output_dir: Path):
        """Create visualization of the optimization results."""
        all_results = results_dict.get('coarse_results', []) + results_dict.get('fine_results', [])
        
        if not all_results:
            print("No results to visualize")
            return
        
        # Sort by percentile
        all_results.sort(key=lambda x: x['percentile'])
        
        percentiles = [r['percentile'] for r in all_results]
        n_clusters = [r['n_clusters'] for r in all_results]
        silhouettes = [r['silhouette'] for r in all_results]
        coherences = [r['coherence'] for r in all_results]
        qualities = [r['quality'] for r in all_results]
        
        optimal = results_dict.get('optimal')
        optimal_p = optimal['percentile'] if optimal else None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Number of clusters
        axes[0, 0].plot(percentiles, n_clusters, 'b-o', markersize=4)
        if optimal_p:
            axes[0, 0].axvline(optimal_p, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].axhline(10, color='green', linestyle=':', alpha=0.5, label='Min preferred')
        axes[0, 0].axhline(50, color='green', linestyle=':', alpha=0.5, label='Max preferred')
        axes[0, 0].set_xlabel('Threshold Percentile')
        axes[0, 0].set_ylabel('Number of Clusters')
        axes[0, 0].set_title('Clusters vs Threshold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Silhouette score
        axes[0, 1].plot(percentiles, silhouettes, 'g-o', markersize=4)
        if optimal_p:
            axes[0, 1].axvline(optimal_p, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_xlabel('Threshold Percentile')
        axes[0, 1].set_ylabel('Silhouette Score')
        axes[0, 1].set_title('Clustering Quality vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Semantic coherence
        axes[1, 0].plot(percentiles, coherences, 'm-o', markersize=4)
        if optimal_p:
            axes[1, 0].axvline(optimal_p, color='red', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Threshold Percentile')
        axes[1, 0].set_ylabel('Semantic Coherence')
        axes[1, 0].set_title('Semantic Alignment vs Threshold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Overall quality
        axes[1, 1].plot(percentiles, qualities, 'k-o', linewidth=2, markersize=6)
        if optimal_p:
            axes[1, 1].axvline(optimal_p, color='red', linestyle='--', alpha=0.7, 
                               label=f'Optimal: {optimal_p:.2f}')
        axes[1, 1].set_xlabel('Threshold Percentile')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_title('Overall Quality vs Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('ETS Threshold Optimization Results', fontsize=14)
        plt.tight_layout()
        
        output_file = output_dir / 'ets_threshold_optimization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
        plt.close()


def main():
    """Run the optimization process."""
    # Load data
    results_dir = Path("semantic_subtypes_experiment_20250523_111112")
    
    print("Loading activation data...")
    with open(results_dir / "semantic_subtypes_activations.pkl", 'rb') as f:
        activations_data = pickle.load(f)
    
    # Get word to subtype mapping
    word_to_subtype = activations_data.get('word_to_subtype', {})
    
    if not word_to_subtype:
        print("Warning: No word_to_subtype mapping found. Semantic coherence will be 0.")
    
    # Create optimizer
    optimizer = ETSThresholdOptimizer(activations_data, word_to_subtype)
    
    # Find optimal threshold
    optimal_threshold, results = optimizer.find_optimal_threshold()
    
    if optimal_threshold:
        # Visualize results
        optimizer.visualize_results(results, results_dir)
        
        # Save results
        with open(results_dir / 'ets_optimization_results.pkl', 'wb') as f:
            pickle.dump({
                'optimal_threshold': optimal_threshold,
                'results': results
            }, f)
        
        print(f"\nOptimization complete! Optimal threshold: {optimal_threshold:.2f}")
        print(f"Results saved to: {results_dir}/ets_optimization_results.pkl")
        
        return optimal_threshold
    else:
        print("\nOptimization failed - no valid results found")
        return None


if __name__ == "__main__":
    optimal = main()