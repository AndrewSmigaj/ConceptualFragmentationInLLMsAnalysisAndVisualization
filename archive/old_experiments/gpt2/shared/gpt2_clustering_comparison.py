#!/usr/bin/env python3
"""
GPT-2 Clustering Comparison Utility

This module provides utility functions to compare different clustering methods
(k-means vs HDBSCAN) on the same dataset and analyze their performance differences.

Designed to work with GPT2PivotClusterer results and provide comprehensive
comparison metrics and visualizations.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


class ClusteringComparison:
    """
    Utility class for comparing clustering results from different methods.
    
    This class provides methods to:
    1. Compare clustering results between k-means and HDBSCAN
    2. Calculate comparative metrics (silhouette scores, cluster counts, etc.)
    3. Generate comparison visualizations
    4. Export comparison reports
    """
    
    def __init__(self):
        """Initialize the clustering comparison utility."""
        self.results_cache = {}
        
    def compare_methods(self, 
                       kmeans_results: Dict[str, Any], 
                       hdbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare k-means and HDBSCAN clustering results.
        
        Args:
            kmeans_results: Results from k-means clustering
            hdbscan_results: Results from HDBSCAN clustering
            
        Returns:
            Dictionary containing comparison metrics and analysis
        """
        print("Comparing k-means vs HDBSCAN clustering results...")
        
        comparison = {
            'methods': {
                'kmeans': self._extract_method_summary(kmeans_results, 'K-means'),
                'hdbscan': self._extract_method_summary(hdbscan_results, 'HDBSCAN')
            },
            'layer_comparison': self._compare_layers(kmeans_results, hdbscan_results),
            'overall_metrics': self._compute_overall_metrics(kmeans_results, hdbscan_results),
            'path_analysis': self._compare_token_paths(kmeans_results, hdbscan_results)
        }
        
        return comparison
    
    def _extract_method_summary(self, results: Dict[str, Any], method_name: str) -> Dict[str, Any]:
        """Extract summary statistics for a clustering method."""
        layer_results = results['layer_results']
        
        # Extract key metrics
        silhouette_scores = [layer['silhouette_score'] for layer in layer_results.values()]
        cluster_counts = [layer['optimal_k'] for layer in layer_results.values()]
        
        summary = {
            'method_name': method_name,
            'num_layers': len(layer_results),
            'num_sentences': len(results['sentences']),
            'num_token_paths': sum(len(paths) for paths in results['token_paths'].values()),
            'silhouette_scores': {
                'mean': float(np.mean(silhouette_scores)),
                'std': float(np.std(silhouette_scores)),
                'min': float(np.min(silhouette_scores)),
                'max': float(np.max(silhouette_scores))
            },
            'cluster_counts': {
                'mean': float(np.mean(cluster_counts)),
                'std': float(np.std(cluster_counts)),
                'min': int(np.min(cluster_counts)),
                'max': int(np.max(cluster_counts)),
                'total_unique': int(np.sum(cluster_counts))
            },
            'metadata': results['metadata']
        }
        
        return summary
    
    def _compare_layers(self, kmeans_results: Dict[str, Any], hdbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare layer-by-layer clustering results."""
        kmeans_layers = kmeans_results['layer_results']
        hdbscan_layers = hdbscan_results['layer_results']
        
        layer_comparison = {}
        
        for layer_key in kmeans_layers.keys():
            if layer_key in hdbscan_layers:
                kmeans_layer = kmeans_layers[layer_key]
                hdbscan_layer = hdbscan_layers[layer_key]
                
                layer_comparison[layer_key] = {
                    'layer_idx': kmeans_layer['layer_idx'],
                    'kmeans': {
                        'optimal_k': kmeans_layer['optimal_k'],
                        'silhouette_score': kmeans_layer['silhouette_score'],
                        'num_centers': len(kmeans_layer['cluster_centers'])
                    },
                    'hdbscan': {
                        'optimal_k': hdbscan_layer['optimal_k'],
                        'silhouette_score': hdbscan_layer['silhouette_score'],
                        'num_centers': len(hdbscan_layer['cluster_centers'])
                    },
                    'differences': {
                        'k_diff': hdbscan_layer['optimal_k'] - kmeans_layer['optimal_k'],
                        'silhouette_diff': hdbscan_layer['silhouette_score'] - kmeans_layer['silhouette_score']
                    }
                }
        
        return layer_comparison
    
    def _compute_overall_metrics(self, kmeans_results: Dict[str, Any], hdbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall comparison metrics."""
        kmeans_summary = self._extract_method_summary(kmeans_results, 'K-means')
        hdbscan_summary = self._extract_method_summary(hdbscan_results, 'HDBSCAN')
        
        overall_metrics = {
            'silhouette_comparison': {
                'kmeans_mean': kmeans_summary['silhouette_scores']['mean'],
                'hdbscan_mean': hdbscan_summary['silhouette_scores']['mean'],
                'difference': hdbscan_summary['silhouette_scores']['mean'] - kmeans_summary['silhouette_scores']['mean'],
                'improvement_pct': ((hdbscan_summary['silhouette_scores']['mean'] - kmeans_summary['silhouette_scores']['mean']) / 
                                   abs(kmeans_summary['silhouette_scores']['mean']) * 100) if kmeans_summary['silhouette_scores']['mean'] != 0 else 0
            },
            'cluster_count_comparison': {
                'kmeans_total': kmeans_summary['cluster_counts']['total_unique'],
                'hdbscan_total': hdbscan_summary['cluster_counts']['total_unique'],
                'difference': hdbscan_summary['cluster_counts']['total_unique'] - kmeans_summary['cluster_counts']['total_unique']
            },
            'consistency_analysis': self._analyze_clustering_consistency(kmeans_results, hdbscan_results)
        }
        
        return overall_metrics
    
    def _compare_token_paths(self, kmeans_results: Dict[str, Any], hdbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare token path patterns between methods."""
        kmeans_paths = kmeans_results['token_paths']
        hdbscan_paths = hdbscan_results['token_paths']
        
        path_similarities = []
        path_differences = []
        
        for sent_idx in kmeans_paths.keys():
            if sent_idx in hdbscan_paths:
                for token_idx in kmeans_paths[sent_idx].keys():
                    if token_idx in hdbscan_paths[sent_idx]:
                        kmeans_path = kmeans_paths[sent_idx][token_idx]
                        hdbscan_path = hdbscan_paths[sent_idx][token_idx]
                        
                        # Calculate path similarity (number of matching cluster assignments)
                        matches = sum(1 for k_cluster, h_cluster in zip(kmeans_path, hdbscan_path) 
                                    if k_cluster == h_cluster)
                        similarity = matches / len(kmeans_path) if len(kmeans_path) > 0 else 0
                        path_similarities.append(similarity)
                        
                        if similarity < 1.0:  # Paths differ
                            path_differences.append({
                                'sentence': sent_idx,
                                'token': token_idx,
                                'kmeans_path': kmeans_path,
                                'hdbscan_path': hdbscan_path,
                                'similarity': similarity
                            })
        
        path_analysis = {
            'total_paths_compared': len(path_similarities),
            'average_similarity': float(np.mean(path_similarities)) if path_similarities else 0,
            'identical_paths': sum(1 for sim in path_similarities if sim == 1.0),
            'different_paths': len(path_differences),
            'similarity_distribution': {
                'min': float(np.min(path_similarities)) if path_similarities else 0,
                'max': float(np.max(path_similarities)) if path_similarities else 0,
                'std': float(np.std(path_similarities)) if path_similarities else 0
            },
            'example_differences': path_differences[:5]  # Show first 5 examples
        }
        
        return path_analysis
    
    def _analyze_clustering_consistency(self, kmeans_results: Dict[str, Any], hdbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency between clustering methods."""
        # This could be expanded to include more sophisticated consistency measures
        layer_comparison = self._compare_layers(kmeans_results, hdbscan_results)
        
        silhouette_improvements = 0
        silhouette_degradations = 0
        k_increases = 0
        k_decreases = 0
        
        for layer_data in layer_comparison.values():
            silhouette_diff = layer_data['differences']['silhouette_diff']
            k_diff = layer_data['differences']['k_diff']
            
            if silhouette_diff > 0:
                silhouette_improvements += 1
            elif silhouette_diff < 0:
                silhouette_degradations += 1
                
            if k_diff > 0:
                k_increases += 1
            elif k_diff < 0:
                k_decreases += 1
        
        consistency = {
            'silhouette_improvements': silhouette_improvements,
            'silhouette_degradations': silhouette_degradations,
            'cluster_count_increases': k_increases,
            'cluster_count_decreases': k_decreases,
            'layers_analyzed': len(layer_comparison)
        }
        
        return consistency
    
    def generate_comparison_report(self, comparison: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate a human-readable comparison report.
        
        Args:
            comparison: Comparison results from compare_methods()
            output_path: Optional path to save the report
            
        Returns:
            Report text
        """
        report = []
        report.append("=" * 80)
        report.append("GPT-2 CLUSTERING METHODS COMPARISON REPORT")
        report.append("=" * 80)
        
        # Method summaries
        kmeans_summary = comparison['methods']['kmeans']
        hdbscan_summary = comparison['methods']['hdbscan']
        
        report.append("\nMETHOD SUMMARIES:")
        report.append("-" * 40)
        
        report.append(f"\nK-means Clustering:")
        report.append(f"  - Layers processed: {kmeans_summary['num_layers']}")
        report.append(f"  - Average silhouette score: {kmeans_summary['silhouette_scores']['mean']:.3f}")
        report.append(f"  - Total clusters found: {kmeans_summary['cluster_counts']['total_unique']}")
        report.append(f"  - Average clusters per layer: {kmeans_summary['cluster_counts']['mean']:.1f}")
        
        report.append(f"\nHDBSCAN Clustering:")
        report.append(f"  - Layers processed: {hdbscan_summary['num_layers']}")
        report.append(f"  - Average silhouette score: {hdbscan_summary['silhouette_scores']['mean']:.3f}")
        report.append(f"  - Total clusters found: {hdbscan_summary['cluster_counts']['total_unique']}")
        report.append(f"  - Average clusters per layer: {hdbscan_summary['cluster_counts']['mean']:.1f}")
        
        # Overall comparison
        overall = comparison['overall_metrics']
        
        report.append(f"\nOVERALL COMPARISON:")
        report.append("-" * 40)
        report.append(f"Silhouette Score Difference: {overall['silhouette_comparison']['difference']:+.3f}")
        report.append(f"Silhouette Improvement: {overall['silhouette_comparison']['improvement_pct']:+.1f}%")
        report.append(f"Total Cluster Count Difference: {overall['cluster_count_comparison']['difference']:+d}")
        
        # Consistency analysis
        consistency = overall['consistency_analysis']
        report.append(f"\nCONSISTENCY ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Layers with silhouette improvements: {consistency['silhouette_improvements']}/{consistency['layers_analyzed']}")
        report.append(f"Layers with silhouette degradations: {consistency['silhouette_degradations']}/{consistency['layers_analyzed']}")
        report.append(f"Layers with more clusters (HDBSCAN): {consistency['cluster_count_increases']}/{consistency['layers_analyzed']}")
        report.append(f"Layers with fewer clusters (HDBSCAN): {consistency['cluster_count_decreases']}/{consistency['layers_analyzed']}")
        
        # Path analysis
        path_analysis = comparison['path_analysis']
        report.append(f"\nTOKEN PATH ANALYSIS:")
        report.append("-" * 40)
        report.append(f"Total paths compared: {path_analysis['total_paths_compared']}")
        report.append(f"Average path similarity: {path_analysis['average_similarity']:.3f}")
        report.append(f"Identical paths: {path_analysis['identical_paths']}")
        report.append(f"Different paths: {path_analysis['different_paths']}")
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"Comparison report saved to: {output_path}")
        
        return report_text
    
    def save_comparison_data(self, comparison: Dict[str, Any], output_path: str):
        """Save comparison data as JSON."""
        with open(output_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Comparison data saved to: {output_path}")


def compare_clustering_files(kmeans_file: str, hdbscan_file: str, output_dir: str = ".") -> Dict[str, Any]:
    """
    Convenience function to compare clustering results from saved files.
    
    Args:
        kmeans_file: Path to k-means clustering results (.pkl file)
        hdbscan_file: Path to HDBSCAN clustering results (.pkl file)  
        output_dir: Directory to save comparison outputs
        
    Returns:
        Comparison results dictionary
    """
    import pickle
    
    print(f"Loading k-means results from: {kmeans_file}")
    with open(kmeans_file, 'rb') as f:
        kmeans_results = pickle.load(f)
    
    print(f"Loading HDBSCAN results from: {hdbscan_file}")
    with open(hdbscan_file, 'rb') as f:
        hdbscan_results = pickle.load(f)
    
    # Perform comparison
    comparator = ClusteringComparison()
    comparison = comparator.compare_methods(kmeans_results, hdbscan_results)
    
    # Generate outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save comparison data
    comparison_file = output_dir / "clustering_comparison.json"
    comparator.save_comparison_data(comparison, str(comparison_file))
    
    # Generate report
    report_file = output_dir / "clustering_comparison_report.txt"
    report_text = comparator.generate_comparison_report(comparison, str(report_file))
    
    print("\nComparison completed!")
    print(f"Results saved to: {output_dir}")
    
    return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare GPT-2 clustering methods")
    parser.add_argument("--kmeans", required=True, help="Path to k-means results (.pkl)")
    parser.add_argument("--hdbscan", required=True, help="Path to HDBSCAN results (.pkl)")
    parser.add_argument("--output", default=".", help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    comparison = compare_clustering_files(args.kmeans, args.hdbscan, args.output)
    
    # Print summary to console
    print("\n" + "="*60)
    print("QUICK SUMMARY:")
    print("="*60)
    overall = comparison['overall_metrics']
    print(f"Silhouette improvement: {overall['silhouette_comparison']['improvement_pct']:+.1f}%")
    print(f"Cluster count difference: {overall['cluster_count_comparison']['difference']:+d}")
    print(f"Path similarity: {comparison['path_analysis']['average_similarity']:.3f}")