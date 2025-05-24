"""
Quality diagnostics for unified CTA pipeline.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import sys
from pathlib import Path

# Add parent directories to path for imports
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sklearn.metrics import silhouette_score

from logging_config import setup_logging

logger = setup_logging(__name__)


class QualityDiagnostics:
    """
    Comprehensive quality checks for the CTA pipeline.
    """
    
    def __init__(self,
                 coverage_threshold: float = 0.8,
                 purity_threshold: float = 0.6,
                 silhouette_threshold: float = 0.3):
        """
        Initialize diagnostics with quality thresholds.
        
        Args:
            coverage_threshold: Minimum acceptable coverage
            purity_threshold: Minimum acceptable purity
            silhouette_threshold: Minimum acceptable silhouette score
        """
        self.coverage_threshold = coverage_threshold
        self.purity_threshold = purity_threshold
        self.silhouette_threshold = silhouette_threshold
        
        logger.info(f"Initialized QualityDiagnostics with thresholds: "
                   f"coverage={coverage_threshold}, purity={purity_threshold}, "
                   f"silhouette={silhouette_threshold}")
    
    def check_preprocessing_quality(self,
                                  original_data: Dict[int, np.ndarray],
                                  processed_data: Dict[int, np.ndarray]) -> Dict:
        """
        Check quality of preprocessing (alignment, scaling, etc).
        
        Args:
            original_data: Layer -> original activations
            processed_data: Layer -> processed activations
            
        Returns:
            Dictionary of preprocessing quality metrics
        """
        results = {
            'layer_checks': {},
            'overall_quality': True,
            'issues': []
        }
        
        layers = sorted(original_data.keys())
        
        for layer in layers:
            orig = original_data[layer]
            proc = processed_data[layer]
            
            # Extract layer index if layer is a string
            if isinstance(layer, str) and layer.startswith('layer_'):
                layer_idx = int(layer.split('_')[1])
            else:
                layer_idx = layer
            
            # Check dimensions preserved
            if orig.shape[0] != proc.shape[0]:
                results['issues'].append(f"Layer {layer}: Sample count mismatch")
                results['overall_quality'] = False
            
            # Check for NaN/Inf
            if np.any(np.isnan(proc)) or np.any(np.isinf(proc)):
                results['issues'].append(f"Layer {layer}: Contains NaN or Inf values")
                results['overall_quality'] = False
            
            # Check variance preservation (shouldn't collapse to zero)
            orig_var = np.var(orig, axis=0)
            proc_var = np.var(proc, axis=0)
            
            if np.any(proc_var < 1e-10):
                n_collapsed = np.sum(proc_var < 1e-10)
                results['issues'].append(f"Layer {layer}: {n_collapsed} dimensions collapsed")
            
            # Check alignment quality (for layers > 0)
            if layer_idx > 0:
                # Find previous layer in processed data
                prev_layer_key = None
                if isinstance(layer, str):
                    prev_layer_key = f'layer_{layer_idx - 1}'
                else:
                    prev_layer_key = layer_idx - 1
                
                if prev_layer_key in processed_data:
                    prev_proc = processed_data[prev_layer_key]
                else:
                    # Skip alignment check if previous layer not found
                    avg_corr = 1.0
                    alignment_quality = "n/a"
                    results['layer_checks'][layer] = {
                        'shape_preserved': bool(orig.shape[0] == proc.shape[0]),
                        'no_nan_inf': bool(not (np.any(np.isnan(proc)) or np.any(np.isinf(proc)))),
                        'variance_preserved': bool(np.all(proc_var >= 1e-10)),
                        'alignment_correlation': float(avg_corr),
                        'alignment_quality': alignment_quality
                    }
                    continue
                    
                prev_proc = processed_data[prev_layer_key]
                # Simple correlation check between consecutive layers
                correlations = []
                for i in range(min(10, proc.shape[1])):  # Check first 10 dims
                    if i < prev_proc.shape[1]:
                        corr = np.corrcoef(proc[:, i], prev_proc[:, i])[0, 1]
                        correlations.append(corr)
                
                avg_corr = np.mean(correlations) if correlations else 0
                alignment_quality = "good" if avg_corr > 0.5 else "poor"
            else:
                avg_corr = 1.0
                alignment_quality = "n/a"
            
            results['layer_checks'][layer] = {
                'shape_preserved': bool(orig.shape[0] == proc.shape[0]),
                'no_nan_inf': bool(not (np.any(np.isnan(proc)) or np.any(np.isinf(proc)))),
                'variance_preserved': bool(np.all(proc_var >= 1e-10)),
                'alignment_correlation': float(avg_corr),
                'alignment_quality': alignment_quality
            }
        
        # Overall preprocessing score
        n_passed = sum(1 for check in results['layer_checks'].values() 
                      if check['shape_preserved'] and check['no_nan_inf'])
        results['preprocessing_score'] = n_passed / len(layers)
        
        if results['preprocessing_score'] < 0.9:
            results['overall_quality'] = False
            
        return results
    
    def check_clustering_quality(self,
                               data: np.ndarray,
                               labels: np.ndarray,
                               layer: int) -> Dict:
        """
        Check quality of clustering results.
        
        Args:
            data: Activation data
            labels: Cluster labels
            layer: Layer number
            
        Returns:
            Dictionary of clustering quality metrics
        """
        n_clusters = len(np.unique(labels))
        n_samples = len(labels)
        
        results = {
            'layer': layer,
            'n_clusters': n_clusters,
            'n_samples': n_samples,
            'quality_checks': {},
            'issues': []
        }
        
        # Check for degenerate clustering
        if n_clusters == 1:
            results['issues'].append("Only one cluster found")
            results['silhouette_score'] = 0.0
        elif n_clusters == n_samples:
            results['issues'].append("Each point is its own cluster")
            results['silhouette_score'] = -1.0
        else:
            # Calculate silhouette score
            try:
                sil_score = silhouette_score(data, labels)
                results['silhouette_score'] = float(sil_score)
                
                if sil_score < self.silhouette_threshold:
                    results['issues'].append(f"Low silhouette score: {sil_score:.3f}")
            except:
                results['silhouette_score'] = None
                results['issues'].append("Could not calculate silhouette score")
        
        # Check cluster balance
        cluster_sizes = Counter(labels)
        max_size = max(cluster_sizes.values())
        min_size = min(cluster_sizes.values())
        
        results['cluster_sizes'] = {int(k): int(v) for k, v in cluster_sizes.items()}
        results['size_ratio'] = max_size / min_size if min_size > 0 else float('inf')
        
        if results['size_ratio'] > 10:
            results['issues'].append(f"Highly imbalanced clusters (ratio: {results['size_ratio']:.1f})")
        
        # Quality summary
        results['quality_checks']['reasonable_k'] = 2 <= n_clusters <= n_samples / 5
        results['quality_checks']['balanced'] = results['size_ratio'] < 10
        results['quality_checks']['good_silhouette'] = results.get('silhouette_score', 0) > self.silhouette_threshold
        
        results['passed_checks'] = sum(results['quality_checks'].values())
        results['total_checks'] = len(results['quality_checks'])
        
        return results
    
    def check_micro_clustering_quality(self,
                                     micro_results: List[Dict],
                                     layer: int) -> Dict:
        """
        Check quality of micro-clustering within macro clusters.
        
        Args:
            micro_results: List of micro-clustering results per macro cluster
            layer: Layer number
            
        Returns:
            Quality assessment of micro-clustering
        """
        results = {
            'layer': layer,
            'n_macro_clusters': len(micro_results),
            'macro_cluster_checks': [],
            'overall_metrics': {},
            'issues': []
        }
        
        total_coverage = []
        total_purity = []
        total_anomalies = 0
        total_points = 0
        
        for i, micro in enumerate(micro_results):
            n_micro = micro['n_micro_clusters']
            n_anomalies = micro['n_anomalies']
            coverage = micro['coverage']
            purity = micro['purity']
            cluster_size = n_micro + n_anomalies
            
            # Track totals
            total_coverage.append(coverage)
            total_purity.append(purity)
            total_anomalies += n_anomalies
            total_points += cluster_size
            
            # Check this macro cluster
            cluster_check = {
                'macro_cluster_id': i,
                'n_micro_clusters': n_micro,
                'n_anomalies': n_anomalies,
                'coverage': coverage,
                'purity': purity,
                'passes_coverage': coverage >= self.coverage_threshold,
                'passes_purity': purity >= self.purity_threshold,
                'anomaly_ratio': n_anomalies / cluster_size if cluster_size > 0 else 0
            }
            
            # Flag issues
            if not cluster_check['passes_coverage']:
                results['issues'].append(f"Macro cluster {i}: Low coverage ({coverage:.3f})")
            
            if not cluster_check['passes_purity']:
                results['issues'].append(f"Macro cluster {i}: Low purity ({purity:.3f})")
                
            if cluster_check['anomaly_ratio'] > 0.5:
                results['issues'].append(f"Macro cluster {i}: High anomaly ratio ({cluster_check['anomaly_ratio']:.3f})")
            
            results['macro_cluster_checks'].append(cluster_check)
        
        # Overall metrics
        results['overall_metrics'] = {
            'avg_coverage': np.mean(total_coverage) if total_coverage else 0,
            'avg_purity': np.mean(total_purity) if total_purity else 0,
            'total_anomaly_ratio': total_anomalies / total_points if total_points > 0 else 0,
            'clusters_passing_coverage': sum(1 for c in results['macro_cluster_checks'] if c['passes_coverage']),
            'clusters_passing_purity': sum(1 for c in results['macro_cluster_checks'] if c['passes_purity'])
        }
        
        # Overall quality assessment
        coverage_pass_rate = results['overall_metrics']['clusters_passing_coverage'] / len(micro_results)
        purity_pass_rate = results['overall_metrics']['clusters_passing_purity'] / len(micro_results)
        
        results['quality_score'] = (coverage_pass_rate + purity_pass_rate) / 2
        results['meets_standards'] = results['quality_score'] >= 0.8
        
        return results
    
    def check_path_quality(self,
                         trajectories: Dict[str, List[int]],
                         path_metrics: Dict) -> Dict:
        """
        Check quality of trajectory paths.
        
        Args:
            trajectories: Word -> trajectory mapping
            path_metrics: Calculated path metrics
            
        Returns:
            Path quality assessment
        """
        results = {
            'n_trajectories': len(trajectories),
            'metrics': path_metrics,
            'quality_checks': {},
            'issues': []
        }
        
        # Check for degenerate paths
        all_same = all(traj == list(trajectories.values())[0] 
                      for traj in trajectories.values())
        
        if all_same:
            results['issues'].append("All trajectories are identical")
            results['quality_checks']['diverse_paths'] = False
        else:
            results['quality_checks']['diverse_paths'] = True
        
        # Check fragmentation
        if path_metrics['fragmentation'] > 0.8:
            results['issues'].append(f"High fragmentation: {path_metrics['fragmentation']:.3f}")
            results['quality_checks']['reasonable_fragmentation'] = False
        else:
            results['quality_checks']['reasonable_fragmentation'] = True
        
        # Check stability
        if path_metrics['stability'] < 0.2:
            results['issues'].append(f"Low stability: {path_metrics['stability']:.3f}")
            results['quality_checks']['adequate_stability'] = False
        else:
            results['quality_checks']['adequate_stability'] = True
        
        # Check convergence
        if path_metrics['convergence'] < 0.1:
            results['issues'].append(f"Poor convergence: {path_metrics['convergence']:.3f}")
            results['quality_checks']['shows_convergence'] = False
        else:
            results['quality_checks']['shows_convergence'] = True
        
        # Path length consistency
        path_lengths = [len(traj) for traj in trajectories.values()]
        if len(set(path_lengths)) > 1:
            results['issues'].append("Inconsistent path lengths detected")
            results['quality_checks']['consistent_lengths'] = False
        else:
            results['quality_checks']['consistent_lengths'] = True
        
        # Overall path quality
        results['passed_checks'] = sum(results['quality_checks'].values())
        results['total_checks'] = len(results['quality_checks'])
        results['quality_score'] = results['passed_checks'] / results['total_checks']
        
        return results
    
    def generate_quality_report(self,
                              preprocessing_results: Dict,
                              clustering_results: List[Dict],
                              micro_clustering_results: List[Dict],
                              path_results: Dict) -> Dict:
        """
        Generate comprehensive quality report for entire pipeline.
        
        Args:
            preprocessing_results: Preprocessing quality checks
            clustering_results: Per-layer clustering quality
            micro_clustering_results: Per-layer micro-clustering quality
            path_results: Path quality assessment
            
        Returns:
            Comprehensive quality report
        """
        report = {
            'overall_assessment': {},
            'component_scores': {},
            'critical_issues': [],
            'recommendations': [],
            'detailed_results': {
                'preprocessing': preprocessing_results,
                'clustering': clustering_results,
                'micro_clustering': micro_clustering_results,
                'paths': path_results
            }
        }
        
        # Component scores
        report['component_scores']['preprocessing'] = preprocessing_results.get('preprocessing_score', 0)
        
        # Clustering score (average across layers)
        clustering_scores = []
        for result in clustering_results:
            score = result['passed_checks'] / result['total_checks']
            clustering_scores.append(score)
        report['component_scores']['clustering'] = np.mean(clustering_scores) if clustering_scores else 0
        
        # Micro-clustering score (average across layers)
        micro_scores = [r['quality_score'] for r in micro_clustering_results]
        report['component_scores']['micro_clustering'] = np.mean(micro_scores) if micro_scores else 0
        
        # Path score
        report['component_scores']['paths'] = path_results.get('quality_score', 0)
        
        # Overall assessment
        avg_score = np.mean(list(report['component_scores'].values()))
        report['overall_assessment']['quality_score'] = float(avg_score)
        report['overall_assessment']['status'] = 'good' if avg_score >= 0.8 else 'needs_improvement'
        
        # Collect critical issues
        if preprocessing_results.get('issues'):
            report['critical_issues'].extend(preprocessing_results['issues'])
        
        for i, result in enumerate(clustering_results):
            if result.get('issues'):
                for issue in result['issues']:
                    report['critical_issues'].append(f"Layer {i}: {issue}")
        
        if path_results.get('issues'):
            report['critical_issues'].extend(path_results['issues'])
        
        # Generate recommendations
        if report['component_scores']['preprocessing'] < 0.8:
            report['recommendations'].append("Review preprocessing pipeline, especially alignment")
        
        if report['component_scores']['clustering'] < 0.7:
            report['recommendations'].append("Consider adjusting k-selection strategy or using different features")
        
        if report['component_scores']['micro_clustering'] < 0.7:
            report['recommendations'].append("Adjust ETS percentile thresholds for better coverage")
        
        if report['component_scores']['paths'] < 0.7:
            report['recommendations'].append("High fragmentation suggests clustering instability - review parameters")
        
        # Summary
        report['summary'] = (
            f"Pipeline quality: {report['overall_assessment']['status']} "
            f"(score: {avg_score:.3f}). "
            f"{len(report['critical_issues'])} critical issues found. "
            f"{len(report['recommendations'])} recommendations provided."
        )
        
        logger.info(report['summary'])
        
        return report