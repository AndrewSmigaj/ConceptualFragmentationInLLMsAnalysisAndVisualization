#!/usr/bin/env python3
"""
Run the semantic subtypes experiment with proper ETS clustering.
Uses the already extracted activations to save time.
"""

import os
import sys
import pickle
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pivot"))

def run_ets_clustering():
    """Run ETS clustering on the existing activations."""
    from gpt2_pivot_clusterer import GPT2PivotClusterer
    
    # Use the existing activations
    results_dir = Path("semantic_subtypes_experiment_20250523_111112")
    
    # Load activations
    print("Loading existing activations...")
    with open(results_dir / "semantic_subtypes_activations.pkl", 'rb') as f:
        activations_data = pickle.load(f)
    
    print(f"Loaded activations for {len(activations_data['sentences'])} words")
    
    # Try different threshold percentiles for ETS
    for percentile in [0.5, 0.7, 0.9]:
        print(f"\n{'='*60}")
        print(f"Running ETS clustering with threshold_percentile={percentile}")
        print(f"{'='*60}")
        
        # Create clusterer
        clusterer = GPT2PivotClusterer(
            clustering_method='ets',
            threshold_percentile=percentile,
            random_state=42
        )
        
        # Setup and run clustering
        if clusterer._setup_sklearn() and clusterer.ets_available:
            try:
                # Run clustering
                ets_results = clusterer.cluster_all_layers(activations_data)
                
                # Save results
                output_file = results_dir / f"semantic_subtypes_ets_p{int(percentile*100)}_clustering.pkl"
                with open(output_file, 'wb') as f:
                    pickle.dump(ets_results, f)
                    
                print(f"ETS clustering completed and saved to {output_file}")
                
                # Print summary
                print("\nLayer-wise clustering summary:")
                for layer_key, layer_data in sorted(ets_results['layer_results'].items()):
                    print(f"  {layer_key}: {layer_data.get('optimal_k', 'N/A')} clusters, "
                          f"silhouette={layer_data.get('silhouette_score', 0):.3f}")
                          
                # Count unique paths
                unique_paths = set()
                for sent_paths in ets_results['token_paths'].values():
                    for token_paths in sent_paths.values():
                        if token_paths:
                            path_str = ' -> '.join(token_paths)
                            unique_paths.add(path_str)
                            
                print(f"\nTotal unique paths: {len(unique_paths)}")
                
            except Exception as e:
                print(f"ETS clustering failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ETS not available")

if __name__ == "__main__":
    run_ets_clustering()