"""
Demo script for LLM-based analysis of concept clusters and paths.

This script demonstrates how to use the LLM integration to:
1. Generate human-readable labels for clusters
2. Generate narratives for paths through activation space
3. Perform bias audits on cluster paths
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union

# Make sure we can import from parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from concept_fragmentation.llm.analysis import ClusterAnalysis
from concept_fragmentation.llm.bias_audit import compute_bias_scores, generate_bias_report, analyze_bias_with_llm


def load_cluster_paths_data(dataset: str, seed: int) -> Dict[str, Any]:
    """
    Load cluster paths data from the appropriate file.
    
    Args:
        dataset: The name of the dataset
        seed: The random seed used for clustering
        
    Returns:
        The cluster paths data as a dictionary
    """
    # Get data directory
    data_dir = os.path.join(os.getcwd(), "data")
    
    # Construct the file path
    # Try both naming conventions
    file_paths = [
        os.path.join(data_dir, "cluster_paths", f"{dataset}_seed_{seed}.json"),
        os.path.join(data_dir, "cluster_paths", f"{dataset}_seed_{seed}_paths.json"),
        os.path.join(data_dir, "cluster_paths", f"{dataset}_seed_{seed}_paths_with_centroids.json")
    ]
    
    # Find the first file that exists
    file_path = None
    for path in file_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    # Check if we found a file
    if file_path is None:
        print(f"Error: No cluster paths file found. Looked for:")
        for path in file_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    # Load the data
    with open(file_path, "r") as f:
        data = json.load(f)
    
    return data


def extract_centroids_from_data(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract centroids from cluster paths data.
    
    Args:
        data: The cluster paths data
        
    Returns:
        Dictionary mapping cluster IDs to centroids
    """
    centroids = {}
    
    # Check if id_mapping exists
    if "id_mapping" not in data:
        print("Error: No ID mapping found in cluster paths data")
        return centroids
    
    # Get the centroid data and ID mapping
    if "unique_centroids" in data:
        unique_centroids = data["unique_centroids"]
        id_mapping = data["id_mapping"]
        
        # Map unique IDs to cluster IDs
        for unique_id, centroid in unique_centroids.items():
            if unique_id in id_mapping:
                mapping = id_mapping[unique_id]
                layer_name = mapping["layer_name"]
                original_id = mapping["original_id"]
                cluster_id = f"{layer_name}C{original_id}"
                
                # Convert centroid to numpy array
                centroids[cluster_id] = np.array(centroid)
    
    return centroids


def extract_paths_from_data(data: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    Extract paths from cluster paths data.
    
    Args:
        data: The cluster paths data
        
    Returns:
        Dictionary mapping path IDs to lists of cluster IDs
    """
    paths = {}
    
    # Extract path archetypes
    if "path_archetypes" in data:
        archetypes = data["path_archetypes"]
        
        for i, archetype in enumerate(archetypes):
            if "path" in archetype:
                path_str = archetype["path"]
                
                # Split the path string into cluster IDs
                path_parts = path_str.split("→")
                
                # Get layer information
                layers = data.get("layers", [])
                
                # Create the path with full cluster IDs
                path = []
                for j, part in enumerate(path_parts):
                    if j < len(layers):
                        layer = layers[j]
                        cluster_id = f"{layer}C{part}"
                        path.append(cluster_id)
                    else:
                        # Fallback if layer information is missing
                        path.append(f"L{j}C{part}")
                
                paths[i] = path
    
    return paths


def extract_convergent_points(data: Dict[str, Any]) -> Dict[int, List[Tuple[str, str, float]]]:
    """
    Extract convergent points from cluster paths data.
    
    Args:
        data: The cluster paths data
        
    Returns:
        Dictionary mapping path IDs to lists of convergent points
    """
    convergent_points = {}
    
    # Check if similarity data exists
    if "similarity" in data and "convergent_paths" in data["similarity"]:
        convergent_paths = data["similarity"]["convergent_paths"]
        
        # Get layer information
        layers = data.get("layers", [])
        
        for path_id_str, convergences in convergent_paths.items():
            try:
                path_id = int(path_id_str)
                path_convergences = []
                
                for convergence in convergences:
                    early_layer = convergence.get("early_layer", 0)
                    late_layer = convergence.get("late_layer", 0)
                    early_cluster = convergence.get("early_cluster", "?")
                    late_cluster = convergence.get("late_cluster", "?")
                    similarity = convergence.get("similarity", 0)
                    
                    # Create cluster IDs
                    early_layer_name = layers[early_layer] if early_layer < len(layers) else f"L{early_layer}"
                    late_layer_name = layers[late_layer] if late_layer < len(layers) else f"L{late_layer}"
                    
                    early_id = f"{early_layer_name}C{early_cluster}"
                    late_id = f"{late_layer_name}C{late_cluster}"
                    
                    path_convergences.append((early_id, late_id, similarity))
                
                if path_convergences:
                    convergent_points[path_id] = path_convergences
                    
            except ValueError:
                # Skip if path_id is not an integer
                continue
    
    return convergent_points


def extract_fragmentation_scores(data: Dict[str, Any]) -> Dict[int, float]:
    """
    Extract fragmentation scores from cluster paths data.
    
    Args:
        data: The cluster paths data
        
    Returns:
        Dictionary mapping path IDs to fragmentation scores
    """
    fragmentation_scores = {}
    
    # Check if similarity data exists
    if "similarity" in data and "fragmentation_scores" in data["similarity"]:
        frag_data = data["similarity"]["fragmentation_scores"]
        
        if "scores" in frag_data:
            scores = frag_data["scores"]
            
            # Create a dictionary mapping path IDs to scores
            for i, score in enumerate(scores):
                fragmentation_scores[i] = score
    
    return fragmentation_scores


def extract_demographic_info(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Extract demographic information from cluster paths data.
    
    Args:
        data: The cluster paths data
        
    Returns:
        Dictionary mapping path IDs to demographic information
    """
    demographic_info = {}
    
    # Extract path archetypes
    if "path_archetypes" in data:
        archetypes = data["path_archetypes"]
        
        for i, archetype in enumerate(archetypes):
            if "demo_stats" in archetype:
                demographic_info[i] = archetype["demo_stats"]
            
            # Add any other interesting statistics
            if "survived_rate" in archetype:
                if i not in demographic_info:
                    demographic_info[i] = {}
                demographic_info[i]["survival_rate"] = archetype["survived_rate"]
    
    return demographic_info


def save_results(results: Dict[str, Any], output_file: str):
    """
    Save results to a JSON file.
    
    Args:
        results: The results to save
        output_file: The path to the output file
    """
    # Create the directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save the results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    """Main function for the demo script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Demo script for LLM-based analysis")
    parser.add_argument("--dataset", type=str, default="titanic", help="Dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--provider", type=str, default="grok", help="LLM provider (openai, claude, grok, gemini)")
    parser.add_argument("--model", type=str, default="default", help="LLM model")
    parser.add_argument("--output", type=str, default="analysis_results.json", help="Output file path")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching of LLM responses")
    parser.add_argument("--skip-bias-audit", action="store_true", help="Skip bias audit analysis")
    args = parser.parse_args()
    
    # Load cluster paths data
    print(f"Loading cluster paths data for dataset {args.dataset} with seed {args.seed}...")
    data = load_cluster_paths_data(args.dataset, args.seed)
    
    # Extract centroids, paths, and other information
    centroids = extract_centroids_from_data(data)
    paths = extract_paths_from_data(data)
    convergent_points = extract_convergent_points(data)
    fragmentation_scores = extract_fragmentation_scores(data)
    demographic_info = extract_demographic_info(data)
    
    print(f"Found {len(centroids)} clusters and {len(paths)} paths")
    
    # Create the cluster analysis object
    analyzer = ClusterAnalysis(
        provider=args.provider,
        model=args.model,
        use_cache=not args.no_cache
    )
    
    # Generate labels for clusters
    print(f"Generating labels for {len(centroids)} clusters using {args.provider}...")
    cluster_labels = analyzer.label_clusters_sync(centroids)
    
    # Print some example labels
    print("\nExample cluster labels:")
    for cluster_id, label in list(cluster_labels.items())[:5]:
        print(f"{cluster_id}: {label}")
    
    # Generate narratives for paths
    print(f"\nGenerating narratives for {len(paths)} paths...")
    path_narratives = analyzer.generate_path_narratives_sync(
        paths,
        cluster_labels,
        centroids,
        convergent_points,
        fragmentation_scores,
        demographic_info
    )
    
    # Print some example narratives
    print("\nExample path narratives:")
    for path_id, narrative in list(path_narratives.items())[:3]:
        path_str = " → ".join(paths[path_id])
        print(f"\nPath {path_id} ({path_str}):")
        print(narrative)
    
    # Perform bias audit if requested and demographic info is available
    bias_report = {}
    bias_analysis = ""
    
    if not args.skip_bias_audit:
        print("\nPerforming bias audit...")
        if demographic_info:
            # Determine demographic columns based on dataset
            demographic_columns = []
            if args.dataset == "titanic":
                demographic_columns = ["sex", "age", "pclass"]
            elif args.dataset == "adult":
                demographic_columns = ["sex", "race", "age", "education"]
            elif args.dataset == "heart":
                demographic_columns = ["sex", "age"]
            
            # Filter to include only available columns
            available_demo_cols = []
            for path_id, demos in demographic_info.items():
                for col in demographic_columns:
                    if col in demos and col not in available_demo_cols:
                        available_demo_cols.append(col)
            
            if available_demo_cols:
                print(f"Analyzing demographic factors: {', '.join(available_demo_cols)}")
                
                # Generate bias report
                bias_report = generate_bias_report(
                    paths=paths,
                    demographic_info=demographic_info,
                    demographic_columns=available_demo_cols,
                    cluster_labels=cluster_labels
                )
                
                # Analyze bias with LLM
                print("Generating bias analysis with LLM...")
                bias_analysis = analyze_bias_with_llm(analyzer, bias_report)
                
                print("\nBias Analysis Excerpt:")
                # Print first 3 lines of the analysis
                analysis_lines = bias_analysis.split('\n')
                for line in analysis_lines[:3]:
                    print(line)
                print("...")
            else:
                print("No demographic information available for bias analysis")
                bias_analysis = "No demographic information available for bias analysis"
        else:
            print("No demographic information available for bias analysis")
            bias_analysis = "No demographic information available for bias analysis"
    else:
        print("Bias audit skipped (--skip-bias-audit flag provided)")
        bias_analysis = "Bias audit skipped"

    # Save the results
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "provider": args.provider,
        "model": analyzer.model,
        "cluster_labels": cluster_labels,
        "path_narratives": path_narratives,
        "bias_report": bias_report,
        "bias_analysis": bias_analysis
    }
    
    save_results(results, args.output)


if __name__ == "__main__":
    main()