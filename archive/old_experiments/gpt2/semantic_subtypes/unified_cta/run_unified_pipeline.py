"""
Main pipeline orchestrator for unified CTA analysis.
Ties together all components without reimplementing existing functionality.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import traceback
import sys
from datetime import datetime

# Add parent directories to path for imports
unified_cta_dir = Path(__file__).parent
if str(unified_cta_dir) not in sys.path:
    sys.path.insert(0, str(unified_cta_dir))

# Import our modules (no reimplementation)
from config import UnifiedCTAConfig, create_default_config, create_config_for_experiment
from results_manager import UnifiedCTAResultsManager
from preprocessing.pipeline import PreprocessingPipeline
from clustering.structural import StructuralClusterer
from explainability.ets_micro import CentroidBasedETS
from paths.path_analysis import PathAnalyzer
from llm.interpretability import DirectInterpreter
from diagnostics.quality_checks import QualityDiagnostics
from visualization.unified_visualizer import UnifiedVisualizer
from logging_config import setup_logging

logger = setup_logging(__name__)


class UnifiedCTAPipeline:
    """
    Main orchestrator for unified Concept Trajectory Analysis.
    
    Processes data layer-by-layer, then analyzes cross-layer trajectories.
    Uses existing components without reimplementation.
    """
    
    def __init__(self, config: UnifiedCTAConfig):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.results_manager = UnifiedCTAResultsManager(config)
        
        # Initialize components
        self.preprocessor = PreprocessingPipeline(
            n_components=config.preprocessing.pca_components,
            align_layers=config.preprocessing.apply_procrustes
        )
        
        self.clusterer = StructuralClusterer()
        
        self.quality_checker = QualityDiagnostics(
            coverage_threshold=config.quality.min_coverage,
            purity_threshold=config.quality.min_purity,
            silhouette_threshold=config.quality.min_silhouette_score
        )
        
        if config.enable_path_analysis:
            self.path_analyzer = PathAnalyzer()
        
        if config.enable_interpretability:
            self.interpreter = DirectInterpreter()
        
        if config.enable_visualization:
            viz_output_dir = self.results_manager.subdirs['visualizations']
            self.visualizer = UnifiedVisualizer(viz_output_dir)
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        # Track results
        self.stage_results = {}
        self.stage_status = {}
        
        logger.info(f"Initialized UnifiedCTAPipeline")
        logger.info(f"Layers to process: {config.layers_to_process}")
        logger.info(f"Output directory: {self.results_manager.run_dir}")
    
    def load_data(self) -> Tuple[Dict[int, np.ndarray], List[str], Dict[str, str]]:
        """
        Load activation data and word information.
        
        Returns:
            Tuple of (activations_by_layer, word_list, word_subtypes)
        """
        logger.info("Loading input data...")
        
        # Load activations (following existing pattern)
        activations_path = Path(self.config.activations_path)
        if not activations_path.exists():
            raise FileNotFoundError(f"Activations file not found: {activations_path}")
        
        with open(activations_path, 'rb') as f:
            activations_data = pickle.load(f)
        
        # Extract activations by layer
        activations_by_layer = {}
        word_list = []
        
        # Handle different data formats
        if isinstance(activations_data, dict):
            # Check if it's the nested format with layer_X keys
            if any(k.startswith('layer_') for k in activations_data.keys()):
                # Extract activations from each layer dict
                for layer_key, layer_data in activations_data.items():
                    if layer_key.startswith('layer_'):
                        if isinstance(layer_data, dict) and 'activations' in layer_data:
                            activations_by_layer[layer_key] = layer_data['activations']
                        else:
                            activations_by_layer[layer_key] = layer_data
                
                # Extract word list if available
                if 'word_list' in activations_data:
                    word_list = activations_data['word_list']
                elif 'layer_0' in activations_data and 'word_list' in activations_data['layer_0']:
                    word_list = activations_data['layer_0']['word_list']
            elif 'activations' in activations_data:
                activations_by_layer = activations_data['activations']
                word_list = activations_data.get('word_list', [])
        else:
            # Assume direct format
            activations_by_layer = activations_data
        
        # Load word subtypes
        word_subtypes = {}
        word_list_path = Path(self.config.word_list_path)
        if word_list_path.exists():
            with open(word_list_path, 'r') as f:
                curated_data = json.load(f)
            
            if "curated_words" in curated_data:
                subtypes_words = curated_data["curated_words"]
                # Create word -> subtype mapping
                for subtype, words in subtypes_words.items():
                    for word in words:
                        word_subtypes[word] = subtype
                
                # Update word list if empty
                if not word_list:
                    word_list = []
                    for words in subtypes_words.values():
                        word_list.extend(words)
        
        logger.info(f"Loaded {len(activations_by_layer)} layers")
        logger.info(f"Word list: {len(word_list)} words")
        logger.info(f"Word subtypes: {len(word_subtypes)} mapped words")
        
        # Filter layers to process
        filtered_activations = {}
        for layer_idx in self.config.layers_to_process:
            layer_key = f'layer_{layer_idx}'
            if layer_key in activations_by_layer:
                filtered_activations[layer_key] = activations_by_layer[layer_key]
            elif layer_idx in activations_by_layer:
                filtered_activations[layer_idx] = activations_by_layer[layer_idx]
        
        return filtered_activations, word_list, word_subtypes
    
    def run_preprocessing_stage(self, 
                              activations_by_layer: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Run preprocessing stage on all layers.
        
        Args:
            activations_by_layer: Raw activations
            
        Returns:
            Processed activations
        """
        logger.info("=" * 50)
        logger.info("STAGE 1: PREPROCESSING")
        logger.info("=" * 50)
        
        try:
            # Process all layers
            processed_data = self.preprocessor.fit_transform(activations_by_layer)
            
            # Quality check
            preprocessing_quality = self.quality_checker.check_preprocessing_quality(
                activations_by_layer, processed_data
            )
            
            # Save results
            self.results_manager.save_preprocessing_results(
                activations_by_layer, processed_data, preprocessing_quality
            )
            
            self.stage_results['preprocessing'] = {
                'processed_data': processed_data,
                'quality_metrics': preprocessing_quality
            }
            
            self.stage_status['preprocessing'] = {
                'success': preprocessing_quality['overall_quality'],
                'message': f"Processed {len(processed_data)} layers"
            }
            
            if not preprocessing_quality['overall_quality']:
                logger.warning("Preprocessing quality issues detected!")
                for issue in preprocessing_quality['issues']:
                    logger.warning(f"  - {issue}")
            
            logger.info("Preprocessing stage completed")
            return processed_data
            
        except Exception as e:
            logger.error(f"Preprocessing stage failed: {e}")
            self.stage_status['preprocessing'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_clustering_stage(self, 
                           processed_data: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Run macro clustering stage on each layer independently.
        
        Args:
            processed_data: Preprocessed activations
            
        Returns:
            Macro cluster labels by layer
        """
        logger.info("=" * 50)
        logger.info("STAGE 2: MACRO CLUSTERING")
        logger.info("=" * 50)
        
        macro_labels = {}
        clustering_quality_results = []
        
        try:
            for layer in sorted(processed_data.keys()):
                logger.info(f"Clustering layer {layer}...")
                
                # Get layer-specific config
                layer_config = self.config.get_layer_config(layer)
                
                # Find optimal k using gap statistic
                data = processed_data[layer]
                k_range = layer_config.k_range
                
                optimal_k, gap_results = self.clusterer.find_optimal_k_gap(
                    data, k_range=k_range
                )
                
                logger.info(f"  Layer {layer}: optimal k = {optimal_k}")
                
                # Perform clustering
                labels = self.clusterer.cluster_with_k(data, optimal_k)
                macro_labels[layer] = labels
                
                # Quality check
                # Extract layer index from string if needed
                if isinstance(layer, str) and layer.startswith('layer_'):
                    layer_idx = int(layer.split('_')[1])
                else:
                    layer_idx = layer
                clustering_quality = self.quality_checker.check_clustering_quality(
                    data, labels, layer_idx
                )
                clustering_quality_results.append(clustering_quality)
                
                # Prepare metrics for saving
                clustering_metrics = {
                    'optimal_k': optimal_k,
                    'gap_results': gap_results,
                    'quality_check': clustering_quality,
                    'layer_config': layer_config.to_dict()
                }
                
                # Save layer results
                self.results_manager.save_clustering_results(
                    layer, labels, clustering_metrics
                )
                
                # Log quality
                if clustering_quality['issues']:
                    logger.warning(f"  Quality issues for layer {layer}:")
                    for issue in clustering_quality['issues']:
                        logger.warning(f"    - {issue}")
            
            self.stage_results['clustering'] = {
                'macro_labels': macro_labels,
                'quality_results': clustering_quality_results
            }
            
            # Overall clustering status
            successful_layers = sum(1 for result in clustering_quality_results 
                                  if result['passed_checks'] >= result['total_checks'] * 0.7)
            
            self.stage_status['clustering'] = {
                'success': successful_layers >= len(processed_data) * 0.8,
                'message': f"Successfully clustered {successful_layers}/{len(processed_data)} layers"
            }
            
            logger.info("Macro clustering stage completed")
            return macro_labels
            
        except Exception as e:
            logger.error(f"Clustering stage failed: {e}")
            self.stage_status['clustering'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_micro_clustering_stage(self,
                                 processed_data: Dict[int, np.ndarray],
                                 macro_labels: Dict[int, np.ndarray],
                                 word_list: List[str],
                                 word_subtypes: Dict[str, str]) -> Dict[int, List[Dict]]:
        """
        Run micro clustering stage within each macro cluster.
        
        Args:
            processed_data: Preprocessed activations
            macro_labels: Macro cluster labels
            word_list: List of words
            word_subtypes: Word -> subtype mapping
            
        Returns:
            Micro clustering results by layer
        """
        if not self.config.enable_micro_clustering:
            logger.info("Micro clustering disabled, skipping")
            return {}
        
        logger.info("=" * 50)
        logger.info("STAGE 3: MICRO CLUSTERING")
        logger.info("=" * 50)
        
        micro_results_by_layer = {}
        micro_quality_results = []
        
        try:
            for layer in sorted(processed_data.keys()):
                logger.info(f"Micro-clustering layer {layer}...")
                
                # Get layer-specific config
                layer_config = self.config.get_layer_config(layer)
                
                # Initialize ETS for this layer
                ets = CentroidBasedETS(
                    initial_percentile=layer_config.ets_percentile,
                    min_cluster_size=3,
                    merge_threshold=1.0,
                    coverage_target=layer_config.coverage_target,
                    purity_target=layer_config.purity_target
                )
                
                data = processed_data[layer]
                labels = macro_labels[layer]
                
                # Process each macro cluster
                micro_results = []
                unique_macro_labels = np.unique(labels)
                
                for macro_id in unique_macro_labels:
                    # Get points in this macro cluster
                    macro_mask = labels == macro_id
                    macro_points = data[macro_mask]
                    macro_indices = np.where(macro_mask)[0]
                    
                    if len(macro_points) < 3:  # Skip tiny clusters
                        continue
                    
                    # Get word subtypes for this cluster
                    cluster_subtypes = None
                    if word_list and word_subtypes:
                        cluster_subtypes = []
                        for idx in macro_indices:
                            if idx < len(word_list):
                                word = word_list[idx]
                                subtype = word_subtypes.get(word, 'unknown')
                                cluster_subtypes.append(subtype)
                    
                    # Calculate centroid
                    centroid = macro_points.mean(axis=0)
                    
                    # Run micro-clustering
                    micro_result = ets.create_micro_clusters(
                        macro_points, centroid, macro_indices, cluster_subtypes
                    )
                    
                    micro_results.append(micro_result)
                
                micro_results_by_layer[layer] = micro_results
                
                # Quality check
                micro_quality = self.quality_checker.check_micro_clustering_quality(
                    micro_results, layer
                )
                micro_quality_results.append(micro_quality)
                
                # Save results
                self.results_manager.save_micro_clustering_results(layer, micro_results)
                
                # Log summary
                total_micro = sum(r['n_micro_clusters'] for r in micro_results)
                total_anomalies = sum(r['n_anomalies'] for r in micro_results)
                avg_coverage = np.mean([r['coverage'] for r in micro_results])
                avg_purity = np.mean([r['purity'] for r in micro_results])
                
                logger.info(f"  Layer {layer}: {total_micro} micro-clusters, "
                           f"{total_anomalies} anomalies, "
                           f"coverage={avg_coverage:.3f}, purity={avg_purity:.3f}")
            
            self.stage_results['micro_clustering'] = {
                'micro_results': micro_results_by_layer,
                'quality_results': micro_quality_results
            }
            
            # Overall micro clustering status
            successful_layers = sum(1 for result in micro_quality_results 
                                  if result['meets_standards'])
            
            self.stage_status['micro_clustering'] = {
                'success': successful_layers >= len(processed_data) * 0.7,
                'message': f"Successfully micro-clustered {successful_layers}/{len(processed_data)} layers"
            }
            
            logger.info("Micro clustering stage completed")
            return micro_results_by_layer
            
        except Exception as e:
            logger.error(f"Micro clustering stage failed: {e}")
            self.stage_status['micro_clustering'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_path_analysis_stage(self,
                              macro_labels: Dict[int, np.ndarray],
                              word_list: List[str],
                              word_subtypes: Dict[str, str]) -> Tuple[Dict, Dict, List, Dict]:
        """
        Run path analysis stage to extract trajectories and metrics.
        
        Args:
            macro_labels: Macro cluster labels by layer
            word_list: List of words
            word_subtypes: Word -> subtype mapping
            
        Returns:
            Tuple of (trajectories, path_metrics, archetypal_paths, layer_transitions)
        """
        if not self.config.enable_path_analysis:
            logger.info("Path analysis disabled, skipping")
            return {}, {}, [], {}
        
        logger.info("=" * 50)
        logger.info("STAGE 4: PATH ANALYSIS")
        logger.info("=" * 50)
        
        try:
            # Construct trajectories
            trajectories = self.path_analyzer.construct_trajectories(macro_labels, word_list)
            
            # Extract path patterns
            path_patterns = self.path_analyzer.extract_path_patterns(trajectories)
            
            # Calculate trajectory metrics
            path_metrics = self.path_analyzer.calculate_trajectory_metrics(
                trajectories, word_subtypes
            )
            
            # Identify archetypal paths
            archetypal_paths = self.path_analyzer.identify_archetypal_paths(
                trajectories, min_frequency=3
            )
            
            # Analyze layer transitions
            layer_transitions = self.path_analyzer.analyze_layer_transitions(trajectories)
            
            # Quality check
            path_quality = self.quality_checker.check_path_quality(trajectories, path_metrics)
            
            # Combine results
            combined_metrics = {**path_metrics, **path_patterns}
            
            # Save results
            self.results_manager.save_path_analysis_results(
                trajectories, combined_metrics, archetypal_paths, layer_transitions
            )
            
            self.stage_results['path_analysis'] = {
                'trajectories': trajectories,
                'path_metrics': combined_metrics,
                'archetypal_paths': archetypal_paths,
                'layer_transitions': layer_transitions,
                'quality_check': path_quality
            }
            
            self.stage_status['path_analysis'] = {
                'success': path_quality['quality_score'] >= 0.6,
                'message': f"Analyzed {len(trajectories)} trajectories, "
                          f"found {len(archetypal_paths)} archetypal paths"
            }
            
            logger.info(f"Path analysis completed: {len(trajectories)} trajectories, "
                       f"{len(archetypal_paths)} archetypal paths")
            
            return trajectories, combined_metrics, archetypal_paths, layer_transitions
            
        except Exception as e:
            logger.error(f"Path analysis stage failed: {e}")
            self.stage_status['path_analysis'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_interpretability_stage(self,
                                 macro_labels: Dict[int, np.ndarray],
                                 micro_results: Dict[int, List[Dict]],
                                 trajectories: Dict[str, List[int]],
                                 archetypal_paths: List[Dict],
                                 word_list: List[str],
                                 word_subtypes: Dict[str, str]) -> Tuple[Dict, List, Dict]:
        """
        Run interpretability stage - I do cluster naming and path narration directly.
        
        Returns:
            Tuple of (cluster_names, path_narratives, summary_insights)
        """
        if not self.config.enable_interpretability:
            logger.info("Interpretability disabled, skipping")
            return {}, [], {}
        
        logger.info("=" * 50)
        logger.info("STAGE 5: INTERPRETABILITY ANALYSIS")
        logger.info("=" * 50)
        
        try:
            cluster_names = {}
            path_narratives = []
            
            # Name clusters for each layer
            for layer in sorted(macro_labels.keys()):
                logger.info(f"Analyzing clusters for layer {layer}...")
                
                labels = macro_labels[layer]
                unique_labels = np.unique(labels)
                
                for cluster_id in unique_labels:
                    # Get words in this cluster
                    cluster_mask = labels == cluster_id
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    cluster_words = []
                    for idx in cluster_indices:
                        if idx < len(word_list):
                            cluster_words.append(word_list[idx])
                    
                    if not cluster_words:
                        continue
                    
                    # Name this cluster (I do this directly)
                    cluster_analysis = self.interpreter.name_cluster(
                        cluster_words, layer, cluster_id, word_subtypes
                    )
                    
                    cluster_names[(layer, cluster_id)] = cluster_analysis['cluster_name']
                    
                    logger.info(f"  Layer {layer}, Cluster {cluster_id}: "
                               f"{cluster_analysis['cluster_name']} "
                               f"({len(cluster_words)} words)")
            
            # Narrate archetypal paths (I do this directly)
            logger.info("Creating path narratives...")
            for i, path_data in enumerate(archetypal_paths[:20]):  # Top 20 paths
                path = path_data['path']
                words = path_data['example_words']
                
                # Create narrative
                narrative = self.interpreter.narrate_path(path, words, cluster_names)
                path_narratives.append(narrative)
                
                logger.info(f"  Path {i+1}: {narrative['characteristics']['path_type']} "
                           f"({len(words)} words)")
            
            # Analyze micro-cluster patterns
            micro_analyses = []
            for layer, micro_layer_results in micro_results.items():
                for i, micro_result in enumerate(micro_layer_results):
                    # Get words for this macro cluster
                    macro_mask = macro_labels[layer] == i
                    macro_indices = np.where(macro_mask)[0]
                    macro_words = [word_list[idx] for idx in macro_indices if idx < len(word_list)]
                    
                    if macro_words:
                        micro_analysis = self.interpreter.analyze_micro_clusters(
                            micro_result, macro_words, layer
                        )
                        micro_analyses.append(micro_analysis)
            
            # Generate summary insights (I do this directly)
            summary_insights = self.interpreter.generate_summary_insights(
                {(l, c): {'cluster_name': name, 'n_words': 10, 'layer': l} 
                 for (l, c), name in cluster_names.items()},
                path_narratives,
                {}  # layer_transitions would go here
            )
            
            # Save results
            self.results_manager.save_interpretability_results(
                cluster_names, path_narratives, summary_insights
            )
            
            self.stage_results['interpretability'] = {
                'cluster_names': cluster_names,
                'path_narratives': path_narratives,
                'micro_analyses': micro_analyses,
                'summary_insights': summary_insights
            }
            
            self.stage_status['interpretability'] = {
                'success': True,
                'message': f"Named {len(cluster_names)} clusters, "
                          f"narrated {len(path_narratives)} paths"
            }
            
            logger.info("Interpretability analysis completed")
            return cluster_names, path_narratives, summary_insights
            
        except Exception as e:
            logger.error(f"Interpretability stage failed: {e}")
            self.stage_status['interpretability'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_quality_assessment_stage(self) -> Dict[str, Any]:
        """
        Run final quality assessment across all stages.
        
        Returns:
            Comprehensive quality report
        """
        logger.info("=" * 50)
        logger.info("STAGE 6: QUALITY ASSESSMENT")
        logger.info("=" * 50)
        
        try:
            # Gather results from all stages
            preprocessing_results = self.stage_results.get('preprocessing', {}).get('quality_metrics', {})
            clustering_results = self.stage_results.get('clustering', {}).get('quality_results', [])
            micro_clustering_results = self.stage_results.get('micro_clustering', {}).get('quality_results', [])
            path_results = self.stage_results.get('path_analysis', {}).get('quality_check', {})
            
            # Generate comprehensive quality report
            quality_report = self.quality_checker.generate_quality_report(
                preprocessing_results, clustering_results, micro_clustering_results, path_results
            )
            
            # Save quality assessment
            self.results_manager.save_quality_assessment(quality_report)
            
            self.stage_results['quality_assessment'] = quality_report
            
            self.stage_status['quality_assessment'] = {
                'success': quality_report['overall_assessment']['status'] == 'good',
                'message': f"Overall quality: {quality_report['overall_assessment']['status']} "
                          f"(score: {quality_report['overall_assessment']['quality_score']:.3f})"
            }
            
            logger.info("Quality assessment completed")
            return quality_report
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            self.stage_status['quality_assessment'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_visualization_stage(self,
                              trajectories: Dict[str, List[int]],
                              macro_labels: Dict[int, np.ndarray],
                              path_metrics: Dict[str, Any],
                              archetypal_paths: List[Dict],
                              cluster_names: Dict) -> Dict[str, Any]:
        """
        Run visualization stage to create all visualizations.
        
        Returns:
            Visualization results
        """
        if not self.config.enable_visualization:
            logger.info("Visualization disabled, skipping")
            return {}
        
        logger.info("=" * 50)
        logger.info("STAGE 7: VISUALIZATION")
        logger.info("=" * 50)
        
        try:
            visualization_results = {}
            
            # 1. Trajectory Sankey diagram
            logger.info("Creating trajectory Sankey diagram...")
            sankey_result = self.visualizer.create_trajectory_sankey(
                trajectories, 
                max_words=self.config.max_words_for_viz
            )
            visualization_results['trajectory_sankey'] = sankey_result
            
            # 2. Cluster evolution plot
            logger.info("Creating cluster evolution plot...")
            evolution_result = self.visualizer.create_cluster_evolution_plot(macro_labels)
            visualization_results['cluster_evolution'] = evolution_result
            
            # 3. Path diversity heatmap
            logger.info("Creating path diversity heatmap...")
            diversity_result = self.visualizer.create_path_diversity_heatmap(trajectories)
            visualization_results['path_diversity'] = diversity_result
            
            # 4. Archetypal paths visualization
            logger.info("Creating archetypal paths visualization...")
            paths_result = self.visualizer.create_archetypal_paths_visualization(archetypal_paths)
            visualization_results['archetypal_paths'] = paths_result
            
            # 5. Summary dashboard
            logger.info("Creating summary dashboard...")
            dashboard_result = self.visualizer.create_summary_dashboard(
                trajectories, path_metrics, cluster_names, archetypal_paths
            )
            visualization_results['summary_dashboard'] = dashboard_result
            
            # Save visualization results
            self.results_manager.save_visualization_results(visualization_results)
            
            self.stage_results['visualization'] = visualization_results
            
            self.stage_status['visualization'] = {
                'success': True,
                'message': f"Created {len(visualization_results)} visualizations"
            }
            
            logger.info("Visualization stage completed")
            return visualization_results
            
        except Exception as e:
            logger.error(f"Visualization stage failed: {e}")
            self.stage_status['visualization'] = {
                'success': False,
                'message': f"Failed: {str(e)}"
            }
            raise
    
    def run_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete unified CTA pipeline.
        
        Returns:
            Pipeline execution summary
        """
        start_time = datetime.now()
        
        logger.info("STARTING UNIFIED CTA PIPELINE")
        logger.info(f"Run ID: {self.results_manager.run_id}")
        logger.info(f"Configuration: {len(self.config.layers_to_process)} layers")
        
        # Save configuration
        self.results_manager.save_config()
        
        try:
            # Load data
            activations_by_layer, word_list, word_subtypes = self.load_data()
            
            # Stage 1: Preprocessing
            processed_data = self.run_preprocessing_stage(activations_by_layer)
            
            # Stage 2: Macro clustering (per layer)
            macro_labels = self.run_clustering_stage(processed_data)
            
            # Stage 3: Micro clustering (optional)
            micro_results = self.run_micro_clustering_stage(
                processed_data, macro_labels, word_list, word_subtypes
            )
            
            # Stage 4: Path analysis (cross-layer)
            trajectories, path_metrics, archetypal_paths, layer_transitions = self.run_path_analysis_stage(
                macro_labels, word_list, word_subtypes
            )
            
            # Stage 5: Interpretability (I do the analysis)
            cluster_names, path_narratives, summary_insights = self.run_interpretability_stage(
                macro_labels, micro_results, trajectories, archetypal_paths, word_list, word_subtypes
            )
            
            # Stage 6: Quality assessment
            quality_report = self.run_quality_assessment_stage()
            
            # Stage 7: Visualization
            visualization_results = self.run_visualization_stage(
                trajectories, macro_labels, path_metrics, archetypal_paths, cluster_names
            )
            
            # Generate final report
            end_time = datetime.now()
            duration = end_time - start_time
            
            execution_summary = {
                'run_id': self.results_manager.run_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration.total_seconds(),
                'stage_status': self.stage_status,
                'key_metrics': {
                    'total_words': len(word_list),
                    'layers_processed': len(macro_labels),
                    'total_clusters': sum(len(np.unique(labels)) for labels in macro_labels.values()),
                    'unique_trajectories': len(set(tuple(t) for t in trajectories.values())) if trajectories else 0,
                    'archetypal_paths': len(archetypal_paths),
                    'overall_quality_score': quality_report.get('overall_assessment', {}).get('quality_score', 0)
                },
                'output_directory': str(self.results_manager.run_dir)
            }
            
            # Generate final report
            report_file = self.results_manager.generate_final_report(execution_summary)
            
            # Create export package
            export_file = self.results_manager.export_for_sharing()
            
            logger.info("UNIFIED CTA PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Duration: {duration}")
            logger.info(f"Output: {self.results_manager.run_dir}")
            logger.info(f"Report: {report_file}")
            logger.info(f"Export: {export_file}")
            
            return execution_summary
            
        except Exception as e:
            logger.error(f"PIPELINE FAILED: {e}")
            logger.error(traceback.format_exc())
            
            # Generate failure report
            failure_summary = {
                'run_id': self.results_manager.run_id,
                'start_time': start_time,
                'end_time': datetime.now(),
                'stage_status': self.stage_status,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
            self.results_manager.generate_final_report(failure_summary)
            
            raise


def main():
    """Main entry point for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Unified CTA Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--experiment', type=str, default='full', 
                       choices=['full', 'quick_test', 'high_quality'],
                       help='Experiment type')
    parser.add_argument('--layers', type=str, help='Comma-separated list of layers to process')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config:
        config = UnifiedCTAConfig.load(Path(args.config))
    else:
        config = create_config_for_experiment(args.experiment)
    
    # Override layers if specified
    if args.layers:
        config.layers_to_process = [int(x.strip()) for x in args.layers.split(',')]
    
    # Run pipeline
    pipeline = UnifiedCTAPipeline(config)
    summary = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("UNIFIED CTA PIPELINE SUMMARY")
    print("="*60)
    print(f"Run ID: {summary['run_id']}")
    print(f"Duration: {summary['duration_seconds']:.1f} seconds")
    print(f"Output: {summary['output_directory']}")
    
    for stage, status in summary['stage_status'].items():
        symbol = "✓" if status['success'] else "✗"
        print(f"{symbol} {stage}: {status['message']}")


if __name__ == "__main__":
    main()