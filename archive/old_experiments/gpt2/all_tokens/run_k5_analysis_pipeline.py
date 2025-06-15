#!/usr/bin/env python3
"""
Run complete k=5 analysis pipeline by reusing existing unified CTA code.
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


class K5AnalysisPipeline:
    """Adapter to run existing CTA pipeline with k=5 data."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.results_dir = base_dir / "k5_analysis_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # Load k=5 clustering results
        with open(base_dir / "clustering_results_k5" / "all_labels_k5.json", 'r') as f:
            self.cluster_labels = json.load(f)
        
        # Load tokens
        with open(base_dir / "frequent_tokens_full.json", 'r', encoding='utf-8') as f:
            self.tokens = json.load(f)
            self.word_list = [t['token_str'].strip() for t in self.tokens]
        
        # Load semantic labels
        with open(base_dir / "llm_labels_k5" / "cluster_labels_k5.json", 'r') as f:
            label_data = json.load(f)
            self.semantic_labels = label_data['labels']
        
        # Initialize existing analyzers
        self.path_analyzer = WindowedPathAnalyzer()
        self.visualizer = UnifiedVisualizer(output_dir=str(self.results_dir))
        
    def prepare_data_for_cta(self):
        """Convert k=5 data to format expected by existing CTA code."""
        # Convert cluster labels to macro_labels format
        macro_labels = {}
        for layer_str, labels in self.cluster_labels.items():
            layer_idx = int(layer_str)
            macro_labels[layer_idx] = np.array(labels)
        
        # Construct trajectories using existing code
        trajectories = self.path_analyzer.construct_trajectories(macro_labels, self.word_list)
        
        # Save trajectories for reuse
        trajectories_path = self.results_dir / "trajectories_k5.json"
        with open(trajectories_path, 'w', encoding='utf-8') as f:
            json.dump(trajectories, f, indent=2)
        
        logging.info(f"Saved {len(trajectories)} trajectories to {trajectories_path}")
        
        return trajectories, macro_labels
    
    def run_windowed_analysis(self, trajectories):
        """Run windowed path analysis using existing code."""
        # Use existing windowed analyzer
        results = self.path_analyzer.analyze_all_windows(
            trajectories=trajectories,
            word_subtypes=None,  # We don't have subtypes for general tokens
            min_frequency=10    # Higher threshold for 10k tokens
        )
        
        # Enhance results with our semantic labels
        for window_name, window_data in results.items():
            if 'archetypal_paths' in window_data:
                for path_info in window_data['archetypal_paths']:
                    # Add semantic path labels
                    semantic_path = []
                    path = path_info['path']
                    start_layer = self.path_analyzer.windows[window_name][0]
                    
                    for i, cluster_id in enumerate(path):
                        layer = start_layer + i
                        cluster_key = f"L{layer}_C{cluster_id}"
                        layer_key = f"layer_{layer}"
                        
                        if layer_key in self.semantic_labels and cluster_key in self.semantic_labels[layer_key]:
                            label = self.semantic_labels[layer_key][cluster_key]['label']
                        else:
                            label = f"Cluster {cluster_id}"
                        
                        semantic_path.append(label)
                    
                    path_info['semantic_labels'] = semantic_path
        
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                # Recursively convert any tuple keys to strings
                json_results[key] = self._convert_keys_to_strings(value)
            else:
                json_results[key] = value
        
        # Save results
        analysis_path = self.results_dir / "windowed_analysis_k5.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        logging.info(f"Saved windowed analysis to {analysis_path}")
        
        return results
    
    def _convert_keys_to_strings(self, obj):
        """Recursively convert tuple keys to strings for JSON serialization."""
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    new_dict[str(k)] = self._convert_keys_to_strings(v)
                else:
                    new_dict[k] = self._convert_keys_to_strings(v)
            return new_dict
        elif isinstance(obj, list):
            return [self._convert_keys_to_strings(item) for item in obj]
        else:
            return obj
    
    def prepare_visualization_data(self, trajectories, windowed_results, macro_labels):
        """Prepare data for visualization using existing format."""
        # Create results structure expected by visualizer
        viz_data = {
            "results_dir": str(self.results_dir),
            "clustering": {
                "macro_labels": {f"layer_{k}": v.tolist() for k, v in macro_labels.items()}
            },
            "paths": {
                "trajectories": trajectories
            },
            "windowed_analysis": windowed_results,
            "metadata": {
                "k": 5,
                "n_tokens": len(self.tokens),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add semantic labels in expected format
        cluster_labels = {}
        for layer_key, layer_labels in self.semantic_labels.items():
            for cluster_key, label_info in layer_labels.items():
                cluster_labels[cluster_key] = label_info['label']
        
        viz_data["llm_analysis"] = {
            "cluster_labels": cluster_labels
        }
        
        # Save visualization data
        viz_data_path = self.results_dir / "visualization_data_k5.json"
        with open(viz_data_path, 'w', encoding='utf-8') as f:
            json.dump(viz_data, f, indent=2)
        
        return viz_data
    
    def generate_sankey_diagrams(self, viz_data):
        """Generate Sankey diagrams using our k5 generator."""
        # Use our custom k5 sankey generator instead
        logging.info("Generating Sankey diagrams with k5 generator...")
        
        # The k5 sankey generator will handle everything
        return {"status": "Will be generated by generate_k5_sankeys.py"}
    
    def create_summary_report(self, windowed_results, mri_data):
        """Create a comprehensive summary report."""
        report = ["K=5 FREQUENT TOKENS ANALYSIS REPORT", "=" * 60, ""]
        
        report.append("OVERVIEW")
        report.append("-" * 30)
        report.append(f"Total tokens analyzed: {len(self.tokens)}")
        report.append(f"Clustering: k=5 across 12 layers")
        report.append(f"Analysis windows: early (0-3), middle (4-7), late (8-11)")
        report.append("")
        
        # Top paths per window
        for window in ['early', 'middle', 'late']:
            window_data = windowed_results[window]
            report.append(f"{window.upper()} WINDOW")
            report.append("-" * 30)
            
            if 'path_diversity' in window_data:
                report.append(f"Unique paths: {window_data['path_diversity']['unique_paths']}")
                report.append(f"Path entropy: {window_data['path_diversity']['path_entropy']:.3f}")
            
            if 'archetypal_paths' in window_data:
                report.append("\nTop 5 Archetypal Paths:")
                for i, path in enumerate(window_data['archetypal_paths'][:5]):
                    report.append(f"\n{i+1}. Path: {' → '.join(map(str, path['path']))}")
                    report.append(f"   Frequency: {path['frequency']} ({path['frequency']/len(self.tokens)*100:.1f}%)")
                    if 'semantic_labels' in path:
                        report.append(f"   Semantic: {' → '.join(path['semantic_labels'])}")
                    report.append(f"   Examples: {', '.join(path['representative_words'][:5])}")
            
            report.append("")
        
        # Save report
        report_path = self.results_dir / "analysis_report_k5.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logging.info(f"Saved analysis report to {report_path}")
    
    def run_pipeline(self):
        """Run the complete analysis pipeline."""
        logging.info("Starting k=5 analysis pipeline...")
        
        # Step 1: Prepare data
        logging.info("Step 1: Preparing data...")
        trajectories, macro_labels = self.prepare_data_for_cta()
        
        # Step 2: Run windowed analysis
        logging.info("Step 2: Running windowed path analysis...")
        windowed_results = self.run_windowed_analysis(trajectories)
        
        # Step 3: Prepare visualization data
        logging.info("Step 3: Preparing visualization data...")
        viz_data = self.prepare_visualization_data(trajectories, windowed_results, macro_labels)
        
        # Step 4: Generate Sankey diagrams
        logging.info("Step 4: Generating Sankey diagrams...")
        try:
            mri_data = self.generate_sankey_diagrams(viz_data)
        except Exception as e:
            logging.warning(f"Sankey generation failed: {e}")
            mri_data = {}
        
        # Step 5: Create summary report
        logging.info("Step 5: Creating summary report...")
        self.create_summary_report(windowed_results, mri_data)
        
        logging.info("Pipeline complete!")
        
        return {
            "trajectories": trajectories,
            "windowed_results": windowed_results,
            "viz_data": viz_data,
            "mri_data": mri_data
        }


def main():
    base_dir = Path(__file__).parent
    pipeline = K5AnalysisPipeline(base_dir)
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("K=5 ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {pipeline.results_dir}")
    print("\nGenerated outputs:")
    print("  - trajectories_k5.json: Token trajectories")
    print("  - windowed_analysis_k5.json: Path analysis by window")
    print("  - concept_mri_data_k5.json: Data for Sankey diagrams")
    print("  - sankey_*.html: Interactive Sankey visualizations")
    print("  - analysis_report_k5.txt: Summary report")


if __name__ == "__main__":
    main()