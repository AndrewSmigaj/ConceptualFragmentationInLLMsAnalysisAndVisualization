"""
Extract cross-layer metrics from the unified CTA experiment results.

This includes:
- Cross-layer centroid similarity (ρᶜ)
- Membership overlap (J) between adjacent layers
- Trajectory fragmentation (F) scores
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
from collections import defaultdict

# Add project root to path
root_dir = Path(__file__).parent.parent.parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from concept_fragmentation.metrics.cross_layer_metrics import (
    compute_centroid_similarity,
    compute_membership_overlap,
    compute_trajectory_fragmentation,
    compute_path_density
)


def load_clustering_results(results_dir: str):
    """Load clustering results for all layers."""
    results_path = Path(results_dir)
    
    # Load cluster labels and metrics for each layer
    layer_clusters = {}
    
    for layer_idx in range(12):  # 0-11
        layer_dir = results_path / "clustering" / f"layer_{layer_idx}"
        
        if not layer_dir.exists():
            print(f"Warning: No clustering results for layer {layer_idx}")
            continue
            
        # Load cluster labels
        labels_path = layer_dir / "cluster_labels.pkl"
        if labels_path.exists():
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
        else:
            print(f"Warning: No cluster labels for layer {layer_idx}")
            continue
            
        # Load clustering metrics (contains centroids)
        metrics_path = layer_dir / "clustering_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        # For cross-layer metrics, we need centers. Since we don't have them saved,
        # we'll compute them from the activations if available
        layer_clusters[f"layer_{layer_idx}"] = {
            "labels": labels,
            "metrics": metrics,
            "n_clusters": metrics.get("n_clusters", len(np.unique(labels)))
        }
    
    return layer_clusters


def compute_all_cross_layer_metrics(results_dir: str):
    """Compute all cross-layer metrics."""
    
    print("Loading clustering results...")
    layer_clusters = load_clustering_results(results_dir)
    
    # Load trajectories for additional analysis
    trajectories_path = Path(results_dir) / "paths" / "trajectories.pkl"
    with open(trajectories_path, 'rb') as f:
        trajectories_data = pickle.load(f)
    
    trajectories = trajectories_data['trajectories']
    word_list = trajectories_data['word_list']
    
    # Since we don't have cluster centers saved, we'll compute similarity based on membership overlap
    print("\nComputing membership overlap between adjacent layers...")
    
    # Membership overlap (Jaccard index)
    overlap_results = compute_membership_overlap(layer_clusters, normalize=True)
    
    # Extract adjacent layer overlaps
    adjacent_overlaps = {}
    jaccard_scores = {}
    
    for layer_idx in range(11):  # 0-10
        layer1 = f"layer_{layer_idx}"
        layer2 = f"layer_{layer_idx + 1}"
        
        if (layer1, layer2) in overlap_results:
            overlap_matrix = overlap_results[(layer1, layer2)]
            # Average Jaccard index across all cluster pairs
            avg_jaccard = np.mean(overlap_matrix)
            max_jaccard = np.max(overlap_matrix)
            
            adjacent_overlaps[f"{layer_idx}->{layer_idx+1}"] = {
                "average_jaccard": float(avg_jaccard),
                "max_jaccard": float(max_jaccard),
                "overlap_matrix_shape": overlap_matrix.shape
            }
            jaccard_scores[f"L{layer_idx}->L{layer_idx+1}"] = float(avg_jaccard)
    
    print("\nComputing path density between adjacent layers...")
    
    # Path density
    density_scores, path_graph = compute_path_density(layer_clusters, min_overlap=0.1)
    
    # Convert to simple dict
    path_densities = {}
    for (layer1, layer2), density in density_scores.items():
        path_densities[f"{layer1}->{layer2}"] = float(density)
    
    print("\nComputing trajectory fragmentation scores...")
    
    # For trajectory fragmentation, we'll use the actual trajectory data
    # Calculate how fragmented paths are at each layer
    fragmentation_by_layer = {}
    
    for layer_idx in range(12):
        layer_key = f"layer_{layer_idx}"
        if layer_key not in layer_clusters:
            continue
            
        labels = layer_clusters[layer_key]["labels"]
        n_clusters = layer_clusters[layer_key]["n_clusters"]
        
        # Count unique paths up to this layer
        partial_paths = set()
        for word, traj in trajectories.items():
            partial_path = tuple(traj[:layer_idx+1])
            partial_paths.add(partial_path)
        
        # Fragmentation: ratio of unique partial paths to total words
        fragmentation = len(partial_paths) / len(trajectories)
        
        # Also calculate entropy-based fragmentation
        cluster_counts = np.bincount(labels)
        cluster_probs = cluster_counts / cluster_counts.sum()
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
        max_entropy = np.log(n_clusters) if n_clusters > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        fragmentation_by_layer[layer_key] = {
            "path_fragmentation": float(fragmentation),
            "entropy_fragmentation": float(normalized_entropy),
            "n_unique_partial_paths": len(partial_paths),
            "n_clusters": n_clusters
        }
    
    print("\nCalculating stability metrics...")
    
    # Calculate stability: how often tokens stay in same cluster
    stability_scores = []
    for layer_idx in range(11):  # Compare adjacent layers
        layer1_key = f"layer_{layer_idx}"
        layer2_key = f"layer_{layer_idx + 1}"
        
        if layer1_key in layer_clusters and layer2_key in layer_clusters:
            labels1 = layer_clusters[layer1_key]["labels"]
            labels2 = layer_clusters[layer2_key]["labels"]
            
            # Count how many stay in same cluster
            same_cluster = np.sum(labels1 == labels2)
            stability = same_cluster / len(labels1)
            stability_scores.append(stability)
    
    avg_stability = np.mean(stability_scores) if stability_scores else 0
    
    # Compile all results
    results = {
        "adjacent_layer_overlaps": adjacent_overlaps,
        "jaccard_indices": jaccard_scores,
        "path_densities": path_densities,
        "fragmentation_by_layer": fragmentation_by_layer,
        "stability_scores": {
            f"L{i}->L{i+1}": float(score) 
            for i, score in enumerate(stability_scores)
        },
        "average_stability": float(avg_stability),
        "summary": {
            "avg_jaccard": float(np.mean(list(jaccard_scores.values()))),
            "avg_path_density": float(np.mean(list(path_densities.values()))),
            "avg_fragmentation": float(np.mean([
                v["entropy_fragmentation"] 
                for v in fragmentation_by_layer.values()
            ])),
            "total_layers_analyzed": len(layer_clusters)
        }
    }
    
    # Save results
    output_dir = Path(results_dir) / "cross_layer_analysis"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "cross_layer_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    create_summary_report(results, output_dir)
    
    print(f"\nCross-layer metrics saved to {output_dir}")
    
    return results


def create_summary_report(results: dict, output_dir: Path):
    """Create a human-readable summary report."""
    
    report = []
    report.append("CROSS-LAYER METRICS ANALYSIS")
    report.append("=" * 50)
    
    # Summary statistics
    report.append("\nSUMMARY STATISTICS")
    report.append("-" * 30)
    summary = results["summary"]
    report.append(f"Average Jaccard Index: {summary['avg_jaccard']:.3f}")
    report.append(f"Average Path Density: {summary['avg_path_density']:.3f}")
    report.append(f"Average Fragmentation: {summary['avg_fragmentation']:.3f}")
    report.append(f"Average Stability: {results['average_stability']:.3f}")
    
    # Layer-by-layer Jaccard indices
    report.append("\n\nADJACENT LAYER JACCARD INDICES")
    report.append("-" * 30)
    for layer_pair, jaccard in results["jaccard_indices"].items():
        report.append(f"{layer_pair}: {jaccard:.3f}")
    
    # Path densities
    report.append("\n\nPATH DENSITIES")
    report.append("-" * 30)
    for layer_pair, density in results["path_densities"].items():
        report.append(f"{layer_pair}: {density:.3f}")
    
    # Fragmentation by layer
    report.append("\n\nFRAGMENTATION BY LAYER")
    report.append("-" * 30)
    for layer, frag_data in results["fragmentation_by_layer"].items():
        report.append(f"{layer}:")
        report.append(f"  Path fragmentation: {frag_data['path_fragmentation']:.3f}")
        report.append(f"  Entropy fragmentation: {frag_data['entropy_fragmentation']:.3f}")
        report.append(f"  Unique partial paths: {frag_data['n_unique_partial_paths']}")
        report.append(f"  Number of clusters: {frag_data['n_clusters']}")
    
    # Stability scores
    report.append("\n\nSTABILITY SCORES")
    report.append("-" * 30)
    for layer_pair, stability in results["stability_scores"].items():
        report.append(f"{layer_pair}: {stability:.3f}")
    
    # Save report
    with open(output_dir / "cross_layer_metrics_report.txt", 'w') as f:
        f.write('\n'.join(report))
    
    print("\nSummary report created: cross_layer_metrics_report.txt")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Use the most complete results
        results_dir = "results/unified_cta_config/unified_cta_20250524_073316"
    
    # Compute all metrics
    metrics = compute_all_cross_layer_metrics(results_dir)