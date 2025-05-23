#!/usr/bin/env python3
"""
GPT-2 Semantic Subtypes Experiment Runner

This script orchestrates the complete semantic subtypes experiment using the enhanced
clustering infrastructure. It analyzes how GPT-2 organizes semantic knowledge across
8 semantic subtypes using 774 validated single-token words.

The experiment tests both k-means and HDBSCAN clustering methods and provides
comprehensive analysis of GPT-2's internal semantic organization.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the concept_fragmentation package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # Root directory
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))  # Shared GPT-2 utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))  # For pivot clusterer

# Import existing components (no reimplementation)
from gpt2_activation_extractor import SimpleGPT2ActivationExtractor
from gpt2_pivot_clusterer import GPT2PivotClusterer
from gpt2_apa_metrics import GPT2APAMetricsCalculator
from gpt2_clustering_comparison import ClusteringComparison


class SemanticSubtypesExperiment:
    """
    Main experiment runner for GPT-2 semantic subtypes analysis.
    
    This class orchestrates the complete experiment pipeline:
    1. Load validated semantic subtypes words
    2. Extract GPT-2 activations
    3. Apply both k-means and HDBSCAN clustering
    4. Calculate APA metrics
    5. Generate comparison analysis
    6. Analyze semantic organization patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment with configuration.
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.results = {}
        
        # Create output directory BEFORE setting up logging
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Now setup logging (which uses self.output_dir)
        self.logger = self._setup_logging()
        
        self.logger.info(f"Initialized semantic subtypes experiment")
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def _setup_logging(self) -> logging.Logger:
        """Set up experiment logging."""
        logger = logging.getLogger("semantic_subtypes_experiment")
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def load_semantic_subtypes_words(self) -> Dict[str, List[str]]:
        """
        Load the 774 validated semantic subtypes words.
        
        Returns:
            Dictionary mapping subtype names to word lists
        """
        self.logger.info("Loading semantic subtypes words...")
        
        curated_file = Path(self.config['curated_words_file'])
        if not curated_file.exists():
            raise FileNotFoundError(f"Curated words file not found: {curated_file}")
        
        with open(curated_file, 'r') as f:
            curated_data = json.load(f)
        
        # Extract words by subtype
        if "curated_words" in curated_data:
            subtypes_words = curated_data["curated_words"]
        else:
            raise ValueError("Invalid curated words file format")
        
        # Validate word counts
        total_words = sum(len(words) for words in subtypes_words.values())
        self.logger.info(f"Loaded {len(subtypes_words)} semantic subtypes")
        self.logger.info(f"Total words: {total_words}")
        
        for subtype, words in subtypes_words.items():
            self.logger.info(f"  {subtype}: {len(words)} words")
        
        return subtypes_words
    
    def extract_activations(self, subtypes_words: Dict[str, List[str]]) -> str:
        """
        Extract GPT-2 activations for all semantic subtypes words.
        
        Args:
            subtypes_words: Dictionary mapping subtype names to word lists
            
        Returns:
            Path to saved activations file
        """
        self.logger.info("Extracting GPT-2 activations...")
        
        # Flatten all words for activation extraction
        all_words = []
        word_to_subtype = {}
        
        for subtype, words in subtypes_words.items():
            for word in words:
                all_words.append(word)
                word_to_subtype[word] = subtype
        
        self.logger.info(f"Extracting activations for {len(all_words)} words")
        
        # Use activation extractor (it doesn't take parameters in constructor)
        extractor = SimpleGPT2ActivationExtractor()
        
        # Setup the model
        if not extractor.setup_model():
            raise RuntimeError("Failed to setup GPT-2 model")
        
        # Extract activations
        activations_data = extractor.extract_activations(all_words)
        
        # Add semantic subtype information to metadata
        activations_data['semantic_subtypes'] = subtypes_words
        activations_data['word_to_subtype'] = word_to_subtype
        activations_data['experiment_config'] = self.config
        
        # Save activations
        activations_file = self.output_dir / "semantic_subtypes_activations.pkl"
        extractor.save_activations(activations_data, str(activations_file))
        
        self.logger.info(f"Activations saved to: {activations_file}")
        return str(activations_file)
    
    def run_clustering_analysis(self, activations_file: str, method: str) -> Dict[str, Any]:
        """
        Run clustering analysis with specified method.
        
        Args:
            activations_file: Path to activations file
            method: Clustering method ('kmeans' or 'hdbscan')
            
        Returns:
            Clustering results
        """
        self.logger.info(f"Running {method} clustering analysis...")
        
        # Load activations
        import pickle
        with open(activations_file, 'rb') as f:
            activations_data = pickle.load(f)
        
        # Create clusterer with specified method
        clusterer = GPT2PivotClusterer(
            k_range=self.config.get('k_range', (2, 15)),
            random_state=self.config.get('random_state', 42),
            clustering_method=method
        )
        
        # Run clustering
        clustering_results = clusterer.cluster_all_layers(activations_data)
        
        # Save results
        results_file = self.output_dir / f"semantic_subtypes_{method}_clustering.pkl"
        clusterer.save_results(clustering_results, str(results_file))
        
        self.logger.info(f"{method} clustering completed")
        self.logger.info(f"Results saved to: {results_file}")
        
        return clustering_results
    
    def calculate_apa_metrics(self, clustering_results: Dict[str, Any], method: str) -> Dict[str, Any]:
        """
        Calculate APA metrics for clustering results.
        
        Args:
            clustering_results: Results from clustering analysis
            method: Clustering method name for logging
            
        Returns:
            APA metrics
        """
        self.logger.info(f"Calculating APA metrics for {method}...")
        
        # Use existing APA metrics calculator
        metrics_calculator = GPT2APAMetricsCalculator()
        
        # Calculate metrics
        apa_metrics = metrics_calculator.compute_all_metrics(clustering_results)
        
        # Save metrics
        metrics_file = self.output_dir / f"semantic_subtypes_{method}_apa_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(apa_metrics, f, indent=2)
        
        self.logger.info(f"APA metrics calculated for {method}")
        self.logger.info(f"Metrics saved to: {metrics_file}")
        
        return apa_metrics
    
    def generate_comparison_analysis(self, kmeans_results: Dict[str, Any], 
                                   hdbscan_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison analysis between clustering methods.
        
        Args:
            kmeans_results: K-means clustering results
            hdbscan_results: HDBSCAN clustering results
            
        Returns:
            Comparison analysis
        """
        self.logger.info("Generating clustering method comparison...")
        
        # Use clustering comparison utility
        comparator = ClusteringComparison()
        comparison = comparator.compare_methods(kmeans_results, hdbscan_results)
        
        # Save comparison data
        comparison_file = self.output_dir / "clustering_methods_comparison.json"
        comparator.save_comparison_data(comparison, str(comparison_file))
        
        # Generate comparison report
        report_file = self.output_dir / "clustering_methods_comparison_report.txt"
        comparator.generate_comparison_report(comparison, str(report_file))
        
        self.logger.info("Clustering method comparison completed")
        self.logger.info(f"Comparison data: {comparison_file}")
        self.logger.info(f"Comparison report: {report_file}")
        
        return comparison
    
    def analyze_semantic_organization(self, clustering_results: Dict[str, Any], 
                                    activations_file: str) -> Dict[str, Any]:
        """
        Analyze semantic organization patterns across subtypes.
        
        Args:
            clustering_results: Clustering results to analyze
            activations_file: Path to activations file with subtype information
            
        Returns:
            Semantic organization analysis
        """
        self.logger.info("Analyzing semantic organization patterns...")
        
        # Load activations with subtype information
        import pickle
        with open(activations_file, 'rb') as f:
            activations_data = pickle.load(f)
        
        word_to_subtype = activations_data.get('word_to_subtype', {})
        semantic_subtypes = activations_data.get('semantic_subtypes', {})
        
        # Analyze clustering patterns by semantic subtype
        subtype_analysis = {}
        
        for subtype, words in semantic_subtypes.items():
            self.logger.info(f"Analyzing subtype: {subtype}")
            
            # Find sentences/tokens corresponding to this subtype's words
            subtype_tokens = []
            for sent_idx, sentence in clustering_results['sentences'].items():
                if sentence in words:
                    subtype_tokens.append((sent_idx, sentence))
            
            # Analyze clustering patterns for this subtype
            subtype_paths = []
            for sent_idx, word in subtype_tokens:
                if sent_idx in clustering_results['token_paths']:
                    for token_idx, path in clustering_results['token_paths'][sent_idx].items():
                        subtype_paths.append(path)
            
            # Calculate subtype-specific metrics
            if subtype_paths:
                # Calculate fragmentation patterns
                path_lengths = [len(path) for path in subtype_paths]
                unique_clusters_per_path = [len(set(path)) for path in subtype_paths]
                fragmentation_scores = [unique / length for unique, length in 
                                      zip(unique_clusters_per_path, path_lengths) if length > 0]
                
                subtype_analysis[subtype] = {
                    'word_count': len(words),
                    'analyzed_tokens': len(subtype_tokens),
                    'paths_analyzed': len(subtype_paths),
                    'avg_fragmentation': sum(fragmentation_scores) / len(fragmentation_scores) if fragmentation_scores else 0,
                    'fragmentation_std': np.std(fragmentation_scores) if fragmentation_scores else 0,
                    'unique_clusters_per_layer': {}
                }
                
                # Calculate layer-wise cluster distribution
                if subtype_paths:
                    num_layers = len(subtype_paths[0]) if subtype_paths else 0
                    for layer_idx in range(num_layers):
                        layer_clusters = [path[layer_idx] for path in subtype_paths if len(path) > layer_idx]
                        unique_clusters = len(set(layer_clusters))
                        subtype_analysis[subtype]['unique_clusters_per_layer'][f'layer_{layer_idx}'] = unique_clusters
            
            else:
                subtype_analysis[subtype] = {
                    'word_count': len(words),
                    'analyzed_tokens': 0,
                    'paths_analyzed': 0,
                    'avg_fragmentation': 0,
                    'fragmentation_std': 0,
                    'unique_clusters_per_layer': {}
                }
        
        # Save semantic organization analysis
        analysis_file = self.output_dir / "semantic_organization_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(subtype_analysis, f, indent=2)
        
        self.logger.info("Semantic organization analysis completed")
        self.logger.info(f"Analysis saved to: {analysis_file}")
        
        return subtype_analysis
    
    def prepare_llm_analysis_data(self, kmeans_results: Dict[str, Any], 
                                 ets_results: Dict[str, Any],
                                 subtypes_words: Dict[str, List[str]]) -> None:
        """
        Prepare data for LLM interpretability analysis.
        
        Args:
            kmeans_results: K-means clustering results
            ets_results: ETS clustering results  
            subtypes_words: Word lists by semantic subtype
        """
        self.logger.info("Preparing data for LLM analysis...")
        
        # Import the preparer
        sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
        from prepare_llm_analysis_data import LLMAnalysisDataPreparer
        
        preparer = LLMAnalysisDataPreparer(output_format='markdown')
        
        # Save the formatted comparison for LLM analysis
        output_file = self.output_dir / "llm_analysis_data.md"
        preparer.save_for_llm_analysis(
            kmeans_results, 
            ets_results,
            subtypes_words,
            str(output_file)
        )
        
        self.logger.info(f"LLM analysis data saved to: {output_file}")
        self.logger.info("Ready for copy-paste into LLM for interpretability scoring")
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive experiment summary.
        
        Returns:
            Experiment summary
        """
        self.logger.info("Generating experiment summary...")
        
        summary = {
            'experiment_config': self.config,
            'timestamp': datetime.now().isoformat(),
            'output_directory': str(self.output_dir),
            'results_files': {
                'activations': 'semantic_subtypes_activations.pkl',
                'kmeans_clustering': 'semantic_subtypes_kmeans_clustering.pkl',
                'ets_clustering': 'semantic_subtypes_ets_clustering.pkl',
                'kmeans_metrics': 'semantic_subtypes_kmeans_apa_metrics.json',
                'ets_metrics': 'semantic_subtypes_ets_apa_metrics.json',
                'semantic_analysis': 'semantic_organization_analysis.json',
                'llm_analysis_data': 'llm_analysis_data.md'
            },
            'experiment_status': 'completed',
            'infrastructure_used': {
                'activation_extractor': 'SimpleGPT2ActivationExtractor',
                'clusterer': 'GPT2PivotClusterer (with K-means and ETS)',
                'metrics_calculator': 'GPT2APAMetricsCalculator',
                'llm_data_preparer': 'LLMAnalysisDataPreparer'
            }
        }
        
        # Save summary
        summary_file = self.output_dir / "experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info("Experiment summary generated")
        self.logger.info(f"Summary saved to: {summary_file}")
        
        return summary
    
    def run_complete_experiment(self) -> Dict[str, Any]:
        """
        Run the complete semantic subtypes experiment.
        
        Returns:
            Complete experiment results
        """
        self.logger.info("Starting complete semantic subtypes experiment")
        
        try:
            # Step 1: Load semantic subtypes words
            subtypes_words = self.load_semantic_subtypes_words()
            
            # Step 2: Extract activations
            activations_file = self.extract_activations(subtypes_words)
            
            # Step 3: Run k-means clustering
            kmeans_results = self.run_clustering_analysis(activations_file, 'kmeans')
            
            # Step 4: Run ETS clustering  
            ets_results = self.run_clustering_analysis(activations_file, 'ets')
            
            # Step 5: Calculate APA metrics for both methods
            kmeans_metrics = self.calculate_apa_metrics(kmeans_results, 'kmeans')
            ets_metrics = self.calculate_apa_metrics(ets_results, 'ets')
            
            # Step 6: Prepare data for LLM analysis
            self.prepare_llm_analysis_data(kmeans_results, ets_results, subtypes_words)
            
            # Step 7: Analyze semantic organization (using k-means results)
            semantic_analysis = self.analyze_semantic_organization(kmeans_results, activations_file)
            
            # Step 8: Generate experiment summary
            summary = self.generate_experiment_summary()
            
            self.logger.info("Complete semantic subtypes experiment finished successfully")
            
            return {
                'subtypes_words': subtypes_words,
                'activations_file': activations_file,
                'kmeans_results': kmeans_results,
                'ets_results': ets_results,
                'kmeans_metrics': kmeans_metrics,
                'ets_metrics': ets_metrics,
                'semantic_analysis': semantic_analysis,
                'summary': summary,
                'llm_analysis_file': str(self.output_dir / "llm_analysis_data.md")
            }
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def create_default_config() -> Dict[str, Any]:
    """Create default experiment configuration."""
    return {
        'model_name': 'gpt2',
        'curated_words_file': 'gpt2_semantic_subtypes_curated.json',
        'output_dir': f'semantic_subtypes_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'k_range': (2, 15),
        'random_state': 42,
        'max_memory_mb': 8000,
        'cache_dir': None
    }


def main():
    """Main entry point for semantic subtypes experiment."""
    parser = argparse.ArgumentParser(
        description="Run GPT-2 semantic subtypes experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--model", default="gpt2", 
                       choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
                       help="GPT-2 model to use")
    parser.add_argument("--words-file", default="gpt2_semantic_subtypes_curated.json",
                       help="Path to curated words file")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--k-min", type=int, default=2,
                       help="Minimum k for clustering")
    parser.add_argument("--k-max", type=int, default=15,
                       help="Maximum k for clustering")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random state for reproducibility")
    parser.add_argument("--max-memory", type=int, default=8000,
                       help="Maximum memory usage in MB")
    parser.add_argument("--cache-dir", default=None,
                       help="Cache directory for model/tokenizer")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    config.update({
        'model_name': args.model,
        'curated_words_file': args.words_file,
        'k_range': (args.k_min, args.k_max),
        'random_state': args.random_state,
        'max_memory_mb': args.max_memory,
        'cache_dir': args.cache_dir
    })
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    
    # Run experiment
    experiment = SemanticSubtypesExperiment(config)
    results = experiment.run_complete_experiment()
    
    print(f"\n{'='*60}")
    print("SEMANTIC SUBTYPES EXPERIMENT COMPLETED")
    print(f"{'='*60}")
    print(f"Output directory: {experiment.output_dir}")
    print(f"Total semantic subtypes: {len(results['subtypes_words'])}")
    print(f"Total words analyzed: {sum(len(words) for words in results['subtypes_words'].values())}")
    print(f"Clustering methods: k-means, HDBSCAN")
    print(f"Results available in: {experiment.output_dir}")


if __name__ == "__main__":
    # Add numpy import for semantic analysis
    import numpy as np
    main()