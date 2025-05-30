#!/usr/bin/env python3
"""
Run complete k=10 analysis pipeline by reusing existing unified CTA code.
DRY approach - no reimplementation, just adaptation.
"""

import json
import sys
from pathlib import Path
import numpy as np
import logging
from datetime import datetime

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import existing components - no reimplementation!
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_subtypes" / "unified_cta"))
from windowed_path_analysis import WindowedPathAnalyzer
from visualization.unified_visualizer import UnifiedVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class K10AnalysisPipeline:
    """Adapter to run existing CTA pipeline with k=10 data."""
    
    def __init__(self, base_dir: Path, k: int = 10):
        self.base_dir = base_dir
        self.k = k
        self.results_dir = base_dir / f"k{k}_analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load k=10 clustering results
        with open(base_dir / f"clustering_results_k{k}" / f"all_labels_k{k}.json", 'r') as f:
            self.cluster_labels = json.load(f)
        
        # Load tokens
        with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
            self.tokens = json.load(f)
            self.word_list = [t['token_str'].strip() for t in self.tokens]
        
        # Load semantic labels if they exist
        semantic_path = base_dir / f"llm_labels_k{k}" / f"cluster_labels_k{k}.json"
        if semantic_path.exists():
            with open(semantic_path, 'r') as f:
                label_data = json.load(f)
                self.semantic_labels = label_data['labels']
        else:
            logging.warning(f"Semantic labels not found at {semantic_path}")
            self.semantic_labels = {}
        
        # Initialize existing analyzers
        self.path_analyzer = WindowedPathAnalyzer()
        self.visualizer = UnifiedVisualizer(output_dir=str(self.results_dir))
    
    def prepare_unified_format(self):
        """Convert k=10 data to unified CTA format."""
        logging.info(f"Converting k={self.k} clustering results to unified format...")
        
        unified_data = {}
        
        # For each layer, convert cluster assignments
        for layer in range(12):
            layer_key = str(layer)  # Use numeric key
            layer_labels = self.cluster_labels[layer_key]
            
            # Convert to the format expected by windowed analysis
            token_assignments = {}
            for token_idx, token_info in enumerate(self.tokens):
                token_str = token_info['token_str'].strip()
                cluster_id = layer_labels[token_idx]
                token_assignments[token_str] = cluster_id
            
            unified_data[f"L{layer}"] = token_assignments
        
        return unified_data
    
    def run_windowed_analysis(self):
        """Run windowed path analysis using existing code."""
        logging.info("Running windowed trajectory analysis...")
        
        unified_data = self.prepare_unified_format()
        
        # Run path analysis for each window
        windows = {
            'early': list(range(4)),    # Layers 0-3
            'middle': list(range(4, 8)), # Layers 4-7
            'late': list(range(8, 12))   # Layers 8-11
        }
        
        all_results = {}
        
        for window_name, layer_range in windows.items():
            logging.info(f"Analyzing {window_name} window (layers {layer_range[0]}-{layer_range[-1]})...")
            
            # Extract paths for this window
            window_paths = self.path_analyzer.extract_windowed_paths(
                unified_data, 
                layer_range,
                min_frequency=1
            )
            
            # Find archetypal paths
            archetypal_paths = self.path_analyzer.find_archetypal_paths(
                window_paths,
                top_n=15
            )
            
            # Add representative words and semantic labels
            for path_info in archetypal_paths:
                path = path_info['path']
                representative_words = []
                
                # Get tokens that follow this exact path
                for token_str in self.word_list:
                    token_path = []
                    for layer_idx in layer_range:
                        layer_key = f"L{layer_idx}"
                        if token_str in unified_data[layer_key]:
                            token_path.append(unified_data[layer_key][token_str])
                    
                    if token_path == path:
                        representative_words.append(token_str)
                        if len(representative_words) >= 5:
                            break
                
                path_info['representative_words'] = representative_words
                path_info['example_words'] = representative_words[:5]
                
                # Add semantic labels if available
                if self.semantic_labels:
                    semantic_path = []
                    for i, (layer_idx, cluster_id) in enumerate(zip(layer_range, path)):
                        layer_key = f"layer_{layer_idx}"
                        cluster_key = f"L{layer_idx}_C{cluster_id}"
                        
                        if (layer_key in self.semantic_labels and 
                            cluster_key in self.semantic_labels[layer_key]):
                            label = self.semantic_labels[layer_key][cluster_key]['label']
                            semantic_path.append(label)
                        else:
                            semantic_path.append(f"C{cluster_id}")
                    
                    path_info['semantic_labels'] = semantic_path
            
            all_results[window_name] = {
                'layers': layer_range,
                'total_paths': len(window_paths),
                'unique_paths': len(set(tuple(p) for p in window_paths)),
                'archetypal_paths': archetypal_paths
            }
        
        # Save results
        output_path = self.results_dir / f"windowed_analysis_k{self.k}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)
        
        logging.info(f"Windowed analysis complete! Results saved to {output_path}")
        return all_results
    
    def create_trajectory_data(self):
        """Create trajectory data for visualization."""
        logging.info("Creating trajectory visualization data...")
        
        unified_data = self.prepare_unified_format()
        
        # Build trajectories for each token
        trajectories = []
        
        for token_info in self.tokens:
            token_str = token_info['token_str'].strip()
            trajectory = []
            
            for layer in range(12):
                layer_key = f"L{layer}"
                if token_str in unified_data[layer_key]:
                    cluster_id = unified_data[layer_key][token_str]
                    trajectory.append(cluster_id)
                else:
                    trajectory.append(-1)  # Missing data
            
            trajectories.append({
                'token': token_str,
                'trajectory': trajectory,
                'frequency': token_info.get('frequency', 1)
            })
        
        # Save trajectory data
        traj_path = self.results_dir / f"trajectories_k{self.k}.json"
        with open(traj_path, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, indent=2)
        
        logging.info(f"Trajectory data saved to {traj_path}")
        return trajectories


def main():
    """Run the complete k=10 analysis pipeline."""
    base_dir = Path(__file__).parent
    k = 10
    
    print(f"\n{'='*60}")
    print(f"STARTING K={k} ANALYSIS PIPELINE")
    print(f"{'='*60}")
    
    # Check if clustering results exist
    clustering_dir = base_dir / f"clustering_results_k{k}"
    if not clustering_dir.exists():
        print(f"\nERROR: Clustering results not found at {clustering_dir}")
        print(f"Please run clustering analysis with k={k} first.")
        return
    
    # Initialize pipeline
    pipeline = K10AnalysisPipeline(base_dir, k=k)
    
    # Run windowed analysis
    windowed_results = pipeline.run_windowed_analysis()
    
    # Create trajectory data
    trajectories = pipeline.create_trajectory_data()
    
    print(f"\n{'='*60}")
    print(f"K={k} ANALYSIS PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {pipeline.results_dir}")
    print(f"- windowed_analysis_k{k}.json")
    print(f"- trajectories_k{k}.json")
    
    # Print summary
    print(f"\nWindow Analysis Summary:")
    for window, data in windowed_results.items():
        print(f"  {window.capitalize()}: {data['unique_paths']} unique paths from {data['total_paths']} total")


if __name__ == "__main__":
    main()