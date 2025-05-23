#!/usr/bin/env python3
"""
Prepare data for LLM analysis of K-means vs ETS clustering results.

This script formats clustering results, paths, and metrics into a format
that's easy to copy-paste into an LLM for interpretability analysis.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict


class LLMAnalysisDataPreparer:
    """Prepare clustering results for LLM analysis."""
    
    def __init__(self, output_format: str = 'markdown'):
        """
        Initialize the data preparer.
        
        Args:
            output_format: 'markdown' or 'json'
        """
        self.output_format = output_format
    
    def prepare_clustering_comparison(self, 
                                    kmeans_results: Dict[str, Any],
                                    ets_results: Dict[str, Any],
                                    word_subtypes: Dict[str, List[str]]) -> str:
        """
        Prepare a comparison of K-means vs ETS clustering for LLM analysis.
        
        Args:
            kmeans_results: Results from K-means clustering
            ets_results: Results from ETS clustering
            word_subtypes: Mapping of subtype to word list
            
        Returns:
            Formatted string ready for LLM analysis
        """
        if self.output_format == 'markdown':
            return self._format_markdown_comparison(kmeans_results, ets_results, word_subtypes)
        else:
            return self._format_json_comparison(kmeans_results, ets_results, word_subtypes)
    
    def _format_markdown_comparison(self, kmeans_results, ets_results, word_subtypes):
        """Format comparison in markdown for LLM."""
        output = []
        
        # Header
        output.append("# K-means vs ETS Clustering Comparison for GPT-2 Semantic Subtypes")
        output.append("\n## Task for LLM")
        output.append("Please analyze these clustering results and determine which method produces more interpretable archetypal paths for understanding GPT-2's semantic organization.")
        
        # Dataset info
        output.append("\n## Dataset")
        output.append(f"- Total words: {sum(len(words) for words in word_subtypes.values())}")
        output.append("- Semantic subtypes:")
        for subtype, words in word_subtypes.items():
            output.append(f"  - {subtype}: {len(words)} words")
        
        # K-means results
        output.append("\n## K-means Clustering Results")
        output.extend(self._format_method_results(kmeans_results, "K-means"))
        
        # ETS results  
        output.append("\n## ETS Clustering Results")
        output.extend(self._format_method_results(ets_results, "ETS"))
        
        # Path comparison
        output.append("\n## Path Analysis Comparison")
        output.extend(self._format_path_comparison(kmeans_results, ets_results, word_subtypes))
        
        # Questions for LLM
        output.append("\n## Questions for Analysis")
        output.append("1. Which clustering method produces more coherent archetypal paths?")
        output.append("2. Which method better separates semantic subtypes?")
        output.append("3. Are the cluster assignments semantically meaningful?")
        output.append("4. Which method would be easier to interpret for understanding GPT-2's semantic processing?")
        output.append("5. What do the paths tell us about how GPT-2 organizes semantic knowledge?")
        
        return "\n".join(output)
    
    def _format_method_results(self, results: Dict[str, Any], method_name: str) -> List[str]:
        """Format results for a single clustering method."""
        output = []
        
        # Layer-wise clustering info
        if 'layer_results' in results:
            output.append(f"\n### {method_name} Layer-wise Clustering")
            for layer_idx in range(13):
                layer_key = f'layer_{layer_idx}'
                if layer_key in results['layer_results']:
                    layer_data = results['layer_results'][layer_key]
                    output.append(f"- Layer {layer_idx}: {layer_data.get('optimal_k', 'N/A')} clusters, "
                                f"silhouette score: {layer_data.get('silhouette_score', 0):.3f}")
        
        # Path statistics
        if 'token_paths' in results:
            all_paths = []
            for sent_paths in results['token_paths'].values():
                all_paths.extend(list(sent_paths.values()))
            
            path_tuples = [tuple(path) for path in all_paths]
            path_counts = Counter(path_tuples)
            
            output.append(f"\n### {method_name} Path Statistics")
            output.append(f"- Total paths: {len(all_paths)}")
            output.append(f"- Unique paths: {len(path_counts)}")
            output.append(f"- Top 5 most common paths:")
            
            for path, count in path_counts.most_common(5):
                path_str = " → ".join(path)
                output.append(f"  - {path_str} (occurs {count} times)")
        
        # Metrics
        if 'cross_layer_metrics' in results:
            metrics = results['cross_layer_metrics']
            output.append(f"\n### {method_name} Cross-layer Metrics")
            
            if 'centroid_similarity_rho_c' in metrics:
                avg_rho_c = sum(metrics['centroid_similarity_rho_c'].values()) / len(metrics['centroid_similarity_rho_c'])
                output.append(f"- Average ρ^c (centroid similarity): {avg_rho_c:.3f}")
            
            if 'membership_overlap_J' in metrics:
                avg_j = sum(metrics['membership_overlap_J'].values()) / len(metrics['membership_overlap_J'])
                output.append(f"- Average J (membership overlap): {avg_j:.3f}")
        
        return output
    
    def _format_path_comparison(self, kmeans_results, ets_results, word_subtypes):
        """Format path comparison between methods."""
        output = []
        
        # Extract paths by subtype for both methods
        kmeans_paths_by_subtype = self._get_paths_by_subtype(kmeans_results, word_subtypes)
        ets_paths_by_subtype = self._get_paths_by_subtype(ets_results, word_subtypes)
        
        output.append("\n### Path Coherence by Semantic Subtype")
        
        for subtype in word_subtypes:
            output.append(f"\n**{subtype}:**")
            
            # K-means paths for this subtype
            if subtype in kmeans_paths_by_subtype:
                km_paths = kmeans_paths_by_subtype[subtype]
                km_unique = len(set(tuple(p) for p in km_paths))
                output.append(f"- K-means: {len(km_paths)} words follow {km_unique} unique paths")
            
            # ETS paths for this subtype
            if subtype in ets_paths_by_subtype:
                ets_paths = ets_paths_by_subtype[subtype]
                ets_unique = len(set(tuple(p) for p in ets_paths))
                output.append(f"- ETS: {len(ets_paths)} words follow {ets_unique} unique paths")
        
        return output
    
    def _get_paths_by_subtype(self, results, word_subtypes):
        """Group paths by semantic subtype."""
        paths_by_subtype = defaultdict(list)
        
        # This is a simplified version - actual implementation would need
        # to map word indices to subtypes properly
        if 'sentences' in results and 'token_paths' in results:
            for sent_idx, word in results['sentences'].items():
                # Find which subtype this word belongs to
                for subtype, words in word_subtypes.items():
                    if word in words:
                        if sent_idx in results['token_paths']:
                            for token_paths in results['token_paths'][sent_idx].values():
                                paths_by_subtype[subtype].append(token_paths)
                        break
        
        return paths_by_subtype
    
    def prepare_ets_interpretability_data(self, ets_results: Dict[str, Any]) -> str:
        """
        Prepare ETS-specific interpretability data (thresholds).
        
        Args:
            ets_results: Results from ETS clustering
            
        Returns:
            Formatted string with ETS threshold information
        """
        output = []
        output.append("\n## ETS Dimension Thresholds for Interpretability")
        output.append("\nETS clusters words based on dimension-wise thresholds.")
        output.append("Words are in the same cluster if ALL dimensions differ by less than their thresholds.")
        
        # This would include actual threshold data when available
        output.append("\n### Threshold Analysis")
        output.append("- Which dimensions have the tightest thresholds (most discriminative)?")
        output.append("- Do certain dimensions distinguish between semantic subtypes?")
        output.append("- Can we identify 'semantic dimensions' in GPT-2's representation?")
        
        return "\n".join(output)
    
    def save_for_llm_analysis(self, 
                             kmeans_results: Dict[str, Any],
                             ets_results: Dict[str, Any],
                             word_subtypes: Dict[str, List[str]],
                             output_file: str):
        """
        Save formatted data for LLM analysis.
        
        Args:
            kmeans_results: K-means clustering results
            ets_results: ETS clustering results
            word_subtypes: Word subtype mapping
            output_file: Output file path
        """
        # Prepare main comparison
        comparison_text = self.prepare_clustering_comparison(
            kmeans_results, ets_results, word_subtypes
        )
        
        # Add ETS-specific interpretability data
        ets_text = self.prepare_ets_interpretability_data(ets_results)
        
        # Combine all text
        full_text = comparison_text + "\n" + ets_text
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(full_text)
        
        print(f"Saved LLM analysis data to: {output_path}")
        print(f"File size: {len(full_text)} characters")
        print("\nYou can now copy-paste this file's contents into an LLM for interpretability analysis.")


def main():
    """Example usage."""
    preparer = LLMAnalysisDataPreparer()
    
    # Example data (would be loaded from actual results)
    example_kmeans = {
        'layer_results': {
            'layer_0': {'optimal_k': 4, 'silhouette_score': 0.35},
            'layer_12': {'optimal_k': 6, 'silhouette_score': 0.45}
        },
        'token_paths': {
            0: {0: ['L0C1', 'L1C2', 'L2C2', 'L12C3']},
            1: {0: ['L0C1', 'L1C2', 'L2C3', 'L12C4']}
        }
    }
    
    example_ets = {
        'layer_results': {
            'layer_0': {'optimal_k': 3, 'silhouette_score': 0.38},
            'layer_12': {'optimal_k': 8, 'silhouette_score': 0.42}
        },
        'token_paths': {
            0: {0: ['L0C0', 'L1C1', 'L2C1', 'L12C2']},
            1: {0: ['L0C0', 'L1C1', 'L2C2', 'L12C3']}
        }
    }
    
    example_subtypes = {
        'concrete_nouns': ['cat', 'dog', 'table'],
        'abstract_nouns': ['love', 'justice', 'freedom']
    }
    
    # Generate comparison
    comparison = preparer.prepare_clustering_comparison(
        example_kmeans, example_ets, example_subtypes
    )
    print(comparison[:500] + "...")


if __name__ == "__main__":
    main()