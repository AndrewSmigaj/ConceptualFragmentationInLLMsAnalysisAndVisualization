"""
Results management for unified CTA pipeline.
Follows existing patterns from semantic_subtypes experiment.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil
import sys

# Add parent directories to path for imports
unified_cta_dir = Path(__file__).parent
if str(unified_cta_dir) not in sys.path:
    sys.path.insert(0, str(unified_cta_dir))

from config import UnifiedCTAConfig
from logging_config import setup_logging

logger = setup_logging(__name__)


class UnifiedCTAResultsManager:
    """
    Manages results directory structure and exports for unified CTA pipeline.
    Follows existing patterns from semantic_subtypes experiment.
    """
    
    def __init__(self, config: UnifiedCTAConfig, run_id: Optional[str] = None):
        """
        Initialize results manager.
        
        Args:
            config: Pipeline configuration
            run_id: Optional run identifier (generates timestamp if not provided)
        """
        self.config = config
        
        # Generate run ID following existing pattern
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"unified_cta_{timestamp}"
        
        self.run_id = run_id
        
        # Setup directory structure following existing patterns
        base_output_dir = Path(config.output_dir)
        self.run_dir = base_output_dir / run_id
        
        # Create subdirectories following semantic_subtypes pattern
        self.subdirs = {
            'config': self.run_dir / 'config',
            'preprocessing': self.run_dir / 'preprocessing', 
            'clustering': self.run_dir / 'clustering',
            'micro_clustering': self.run_dir / 'micro_clustering',
            'paths': self.run_dir / 'paths',
            'interpretability': self.run_dir / 'interpretability',
            'quality': self.run_dir / 'quality',
            'visualizations': self.run_dir / 'visualizations',
            'reports': self.run_dir / 'reports',
            'exports': self.run_dir / 'exports'
        }
        
        # Create all directories
        for subdir in self.subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized results manager for run: {run_id}")
        logger.info(f"Output directory: {self.run_dir}")
    
    def save_config(self):
        """Save pipeline configuration."""
        config_file = self.subdirs['config'] / 'pipeline_config.json'
        self.config.save(config_file)
        
        # Also save a human-readable summary
        summary_file = self.subdirs['config'] / 'config_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Unified CTA Pipeline Configuration\n")
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")
            
            f.write(f"Data Sources:\n")
            f.write(f"  Activations: {self.config.activations_path}\n")
            f.write(f"  Word list: {self.config.word_list_path}\n\n")
            
            f.write(f"Pipeline Stages:\n")
            f.write(f"  Layers to process: {self.config.layers_to_process}\n")
            f.write(f"  Micro clustering: {self.config.enable_micro_clustering}\n")
            f.write(f"  Path analysis: {self.config.enable_path_analysis}\n")
            f.write(f"  Interpretability: {self.config.enable_interpretability}\n")
            f.write(f"  Visualization: {self.config.enable_visualization}\n\n")
            
            f.write(f"Quality Thresholds:\n")
            f.write(f"  Min silhouette: {self.config.quality.min_silhouette_score}\n")
            f.write(f"  Min coverage: {self.config.quality.min_coverage}\n")
            f.write(f"  Min purity: {self.config.quality.min_purity}\n")
    
    def save_preprocessing_results(self, 
                                 original_data: Dict[int, Any],
                                 processed_data: Dict[int, Any],
                                 preprocessing_metrics: Dict[str, Any]):
        """Save preprocessing stage results."""
        # Save processed data (following existing pickle pattern)
        processed_file = self.subdirs['preprocessing'] / 'processed_activations.pkl'
        with open(processed_file, 'wb') as f:
            pickle.dump({
                'processed_data': processed_data,
                'preprocessing_metrics': preprocessing_metrics,
                'timestamp': datetime.now(),
                'config': self.config.preprocessing.to_dict()
            }, f)
        
        # Save metrics as JSON for easy reading
        metrics_file = self.subdirs['preprocessing'] / 'preprocessing_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(preprocessing_metrics, f, indent=2)
        
        logger.info(f"Saved preprocessing results to {self.subdirs['preprocessing']}")
    
    def save_clustering_results(self, 
                              layer: int,
                              macro_labels: Any,
                              clustering_metrics: Dict[str, Any]):
        """Save clustering results for a specific layer."""
        # Handle both string and int layer identifiers
        if isinstance(layer, str):
            layer_dir = self.subdirs['clustering'] / layer
        else:
            layer_dir = self.subdirs['clustering'] / f'layer_{layer:02d}'
        layer_dir.mkdir(exist_ok=True)
        
        # Save cluster labels (following existing pattern)
        labels_file = layer_dir / 'cluster_labels.pkl'
        with open(labels_file, 'wb') as f:
            pickle.dump({
                'labels': macro_labels,
                'metrics': clustering_metrics,
                'layer': layer,
                'timestamp': datetime.now()
            }, f)
        
        # Save metrics as JSON
        metrics_file = layer_dir / 'clustering_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(clustering_metrics, f, indent=2)
        
        logger.info(f"Saved clustering results for layer {layer}")
    
    def save_micro_clustering_results(self,
                                    layer: int, 
                                    micro_results: List[Dict]):
        """Save micro-clustering results for a specific layer."""
        # Handle both string and int layer identifiers
        if isinstance(layer, str):
            layer_dir = self.subdirs['micro_clustering'] / layer
        else:
            layer_dir = self.subdirs['micro_clustering'] / f'layer_{layer:02d}'
        layer_dir.mkdir(exist_ok=True)
        
        # Save micro-clustering results
        micro_file = layer_dir / 'micro_clustering_results.pkl'
        with open(micro_file, 'wb') as f:
            pickle.dump({
                'micro_results': micro_results,
                'layer': layer,
                'timestamp': datetime.now()
            }, f)
        
        # Save summary as JSON
        summary = {
            'layer': layer,
            'n_macro_clusters': len(micro_results),
            'micro_cluster_summary': []
        }
        
        for i, result in enumerate(micro_results):
            summary['micro_cluster_summary'].append({
                'macro_cluster_id': i,
                'n_micro_clusters': result.get('n_micro_clusters', 0),
                'n_anomalies': result.get('n_anomalies', 0),
                'coverage': result.get('coverage', 0),
                'purity': result.get('purity', 0)
            })
        
        summary_file = layer_dir / 'micro_clustering_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved micro-clustering results for layer {layer}")
    
    def save_path_analysis_results(self,
                                 trajectories: Dict[str, List[int]],
                                 path_metrics: Dict[str, Any],
                                 archetypal_paths: List[Dict],
                                 layer_transitions: Dict[str, Any]):
        """Save path analysis results."""
        # Save trajectories
        trajectories_file = self.subdirs['paths'] / 'trajectories.pkl'
        with open(trajectories_file, 'wb') as f:
            pickle.dump({
                'trajectories': trajectories,
                'path_metrics': path_metrics,
                'archetypal_paths': archetypal_paths,
                'layer_transitions': layer_transitions,
                'timestamp': datetime.now()
            }, f)
        
        # Save path metrics as JSON
        metrics_file = self.subdirs['paths'] / 'path_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(path_metrics, f, indent=2)
        
        # Save archetypal paths as JSON
        paths_file = self.subdirs['paths'] / 'archetypal_paths.json'
        with open(paths_file, 'w') as f:
            json.dump(archetypal_paths, f, indent=2)
        
        logger.info(f"Saved path analysis results")
    
    def save_interpretability_results(self,
                                     cluster_names: Dict,
                                     path_narratives: List[Dict],
                                     summary_insights: Dict):
        """Save interpretability analysis results."""
        # Save cluster names
        names_file = self.subdirs['interpretability'] / 'cluster_names.json'
        # Convert tuple keys to strings for JSON serialization
        serializable_names = {f"{k[0]}_{k[1]}": v for k, v in cluster_names.items()}
        with open(names_file, 'w') as f:
            json.dump(serializable_names, f, indent=2)
        
        # Save path narratives
        narratives_file = self.subdirs['interpretability'] / 'path_narratives.json'
        with open(narratives_file, 'w') as f:
            json.dump(path_narratives, f, indent=2)
        
        # Save summary insights
        insights_file = self.subdirs['interpretability'] / 'summary_insights.json'
        with open(insights_file, 'w') as f:
            json.dump(summary_insights, f, indent=2)
        
        # Create human-readable report
        report_file = self.subdirs['interpretability'] / 'interpretability_report.txt'
        with open(report_file, 'w') as f:
            f.write("UNIFIED CTA INTERPRETABILITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Cluster Names ({len(cluster_names)} total):\n")
            for (layer, cluster_id), name in list(cluster_names.items())[:20]:  # First 20
                f.write(f"  Layer {layer}, Cluster {cluster_id}: {name}\n")
            if len(cluster_names) > 20:
                f.write(f"  ... and {len(cluster_names) - 20} more\n")
            f.write("\n")
            
            f.write(f"Path Narratives ({len(path_narratives)} total):\n")
            for i, narrative in enumerate(path_narratives[:10]):  # First 10
                f.write(f"  Path {i+1}: {narrative.get('narrative', 'N/A')[:100]}...\n")
            f.write("\n")
            
            f.write("Key Insights:\n")
            for insight in summary_insights.get('key_findings', []):
                f.write(f"  • {insight}\n")
        
        logger.info(f"Saved interpretability results")
    
    def save_quality_assessment(self, quality_report: Dict[str, Any]):
        """Save quality assessment results."""
        # Save full quality report
        report_file = self.subdirs['quality'] / 'quality_report.json'
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        # Create human-readable summary
        summary_file = self.subdirs['quality'] / 'quality_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("UNIFIED CTA QUALITY ASSESSMENT\n")
            f.write("=" * 40 + "\n\n")
            
            overall = quality_report.get('overall_assessment', {})
            f.write(f"Overall Status: {overall.get('status', 'unknown').upper()}\n")
            f.write(f"Quality Score: {overall.get('quality_score', 0):.3f}\n\n")
            
            f.write("Component Scores:\n")
            for component, score in quality_report.get('component_scores', {}).items():
                status = "✓" if score >= 0.7 else "⚠" if score >= 0.5 else "✗"
                f.write(f"  {status} {component}: {score:.3f}\n")
            f.write("\n")
            
            issues = quality_report.get('critical_issues', [])
            if issues:
                f.write(f"Critical Issues ({len(issues)}):\n")
                for issue in issues:
                    f.write(f"  • {issue}\n")
                f.write("\n")
            
            recommendations = quality_report.get('recommendations', [])
            if recommendations:
                f.write(f"Recommendations ({len(recommendations)}):\n")
                for rec in recommendations:
                    f.write(f"  • {rec}\n")
        
        logger.info(f"Saved quality assessment")
    
    def save_visualization_results(self, visualization_results: Dict[str, Any]):
        """Save visualization results and metadata."""
        # Copy visualization files to results directory
        for viz_name, viz_data in visualization_results.items():
            if 'output_path' in viz_data:
                source_path = Path(viz_data['output_path'])
                if source_path.exists():
                    dest_path = self.subdirs['visualizations'] / source_path.name
                    shutil.copy2(source_path, dest_path)
                    logger.info(f"Copied {viz_name} to results directory")
        
        # Save visualization metadata
        metadata_file = self.subdirs['visualizations'] / 'visualization_metadata.json'
        serializable_results = {}
        for name, data in visualization_results.items():
            # Remove non-serializable figure objects
            serializable_data = {k: v for k, v in data.items() if k != 'figure'}
            serializable_results[name] = serializable_data
        
        with open(metadata_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved visualization results")
    
    def generate_final_report(self,
                            results_summary: Dict[str, Any]):
        """Generate comprehensive final report."""
        report_file = self.subdirs['reports'] / 'unified_cta_final_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("UNIFIED CONCEPT TRAJECTORY ANALYSIS (CTA) REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Run ID: {self.run_id}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Configuration: {self.config.to_dict()}\n\n")
            
            f.write("PIPELINE EXECUTION SUMMARY\n")
            f.write("-" * 30 + "\n")
            for stage, status in results_summary.get('stage_status', {}).items():
                symbol = "[OK]" if status.get('success', False) else "[FAIL]"
                f.write(f"{symbol} {stage}: {status.get('message', 'Unknown')}\n")
            f.write("\n")
            
            f.write("KEY METRICS\n")
            f.write("-" * 15 + "\n")
            metrics = results_summary.get('key_metrics', {})
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")
            
            f.write("FILES GENERATED\n")
            f.write("-" * 18 + "\n")
            for subdir_name, subdir_path in self.subdirs.items():
                files = list(subdir_path.glob('*'))
                f.write(f"{subdir_name}: {len(files)} files\n")
                for file_path in files[:3]:  # Show first 3 files
                    f.write(f"  - {file_path.name}\n")
                if len(files) > 3:
                    f.write(f"  ... and {len(files) - 3} more\n")
        
        logger.info(f"Generated final report: {report_file}")
        return report_file
    
    def export_for_sharing(self) -> Path:
        """Create a sharable export package."""
        export_file = self.subdirs['exports'] / f'{self.run_id}_export.zip'
        
        # Create zip file with key results
        import zipfile
        with zipfile.ZipFile(export_file, 'w') as zipf:
            # Add key files
            key_files = [
                self.subdirs['config'] / 'pipeline_config.json',
                self.subdirs['reports'] / 'unified_cta_final_report.txt',
                self.subdirs['quality'] / 'quality_summary.txt',
                self.subdirs['interpretability'] / 'interpretability_report.txt'
            ]
            
            for file_path in key_files:
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
            
            # Add visualizations
            viz_dir = self.subdirs['visualizations']
            for viz_file in viz_dir.glob('*'):
                if viz_file.is_file():
                    zipf.write(viz_file, f'visualizations/{viz_file.name}')
        
        logger.info(f"Created export package: {export_file}")
        return export_file