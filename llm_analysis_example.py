#!/usr/bin/env python
"""
Example script for using LLM-powered analysis to generate narratives for neural network activations.

This script demonstrates how to:
1. Load pre-computed cluster paths data
2. Use an LLM to generate human-readable labels for clusters
3. Generate narratives explaining paths through activation space
4. Perform bias auditing on the clustering results
"""

import os
import json
import sys
import numpy as np
from typing import Dict, List, Any

# Ensure we can import from the project root
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from concept_fragmentation.llm.analysis import ClusterAnalysis
from concept_fragmentation.llm.bias_audit import generate_bias_report, analyze_bias_with_llm


def run_llm_analysis(dataset="titanic", seed=0, provider="grok", model="default"):
    """
    Run LLM analysis on cluster paths data.
    
    Args:
        dataset: Name of the dataset (e.g., "titanic")
        seed: Random seed used for the experiment
        provider: LLM provider ("grok", "claude", "openai", "gemini")
        model: Model name (or "default" to use provider's default model)
    
    Returns:
        Dictionary containing analysis results
    """
    # Load cluster paths data
    print(f"Loading cluster paths data for {dataset}, seed {seed}...")
    
    # Try different possible file locations
    file_paths = [
        f"data/cluster_paths/{dataset}_seed_{seed}_paths.json",
        f"data/cluster_paths/{dataset}_seed_{seed}_paths_with_centroids.json",
        f"visualization/data/cluster_paths/{dataset}_seed_{seed}_paths.json"
    ]
    
    data = None
    for path in file_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            print(f"Loaded data from {path}")
            break
    
    if data is None:
        raise FileNotFoundError(f"Could not find cluster paths data for {dataset}, seed {seed}")
    
    # Extract cluster centroids
    centroids = {}
    if "unique_centroids" in data and "id_mapping" in data:
        print("Extracting cluster centroids...")
        for unique_id, centroid_values in data["unique_centroids"].items():
            if unique_id in data["id_mapping"]:
                mapping = data["id_mapping"][unique_id]
                layer_name = mapping["layer_name"]
                original_id = mapping["original_id"]
                
                # Construct a human-readable cluster ID
                cluster_id = f"L{mapping['layer_idx']}C{original_id}"
                centroids[cluster_id] = np.array(centroid_values)
        
        print(f"Extracted {len(centroids)} cluster centroids")
    else:
        print("Warning: No centroids found in data")
    
    # Extract paths
    paths = {}
    if "path_archetypes" in data:
        print("Extracting archetypal paths...")
        for i, archetype in enumerate(data["path_archetypes"]):
            if "path" in archetype:
                path_str = archetype["path"]
                # Handle both Unicode arrow and ASCII representation
                if "→" in path_str:
                    path_parts = path_str.split("→")
                else:
                    path_parts = path_str.split("->")  # Fallback to ASCII representation
                paths[i] = path_parts
        
        print(f"Extracted {len(paths)} archetypal paths")
    
    # Extract convergent points if available
    convergent_points = {}
    if "similarity" in data and "convergent_paths" in data["similarity"]:
        print("Extracting convergent points...")
        for path_id_str, convergences in data["similarity"]["convergent_paths"].items():
            try:
                path_id = int(path_id_str)
                path_convergences = []
                
                for convergence in convergences:
                    early_cluster = convergence.get("early_cluster", 0)
                    late_cluster = convergence.get("late_cluster", 0)
                    similarity = convergence.get("similarity", 0.0)
                    
                    path_convergences.append((str(early_cluster), str(late_cluster), similarity))
                
                if path_convergences:
                    convergent_points[path_id] = path_convergences
            except ValueError:
                continue
    
    # Extract fragmentation scores if available
    fragmentation_scores = {}
    if "similarity" in data and "fragmentation_scores" in data["similarity"]:
        frag_data = data["similarity"]["fragmentation_scores"]
        if "scores" in frag_data:
            scores = frag_data["scores"]
            for i, score in enumerate(scores):
                fragmentation_scores[i] = score
    
    # Extract demographic information for bias analysis
    demographic_info = {}
    if "path_archetypes" in data:
        for i, archetype in enumerate(data["path_archetypes"]):
            if "demo_stats" in archetype:
                demographic_info[i] = archetype["demo_stats"]
    
    # Initialize the LLM client
    print(f"Initializing {provider} LLM client...")
    analyzer = ClusterAnalysis(
        provider=provider,
        model=model,
        use_cache=False,  # Disable cache to see updated results
        optimize_prompts=True  # Optimize prompts for better results
    )
    
    # Generate human-readable labels for clusters based on the dataset
    print(f"Generating labels for {len(centroids)} clusters...")
    
    # Override the method to use dataset-specific prompting
    # Save original method to restore later
    original_label_method = analyzer.label_cluster
    
    # Create dataset-appropriate features dictionary
    dataset_info = {
        "titanic": {
            "description": "survival prediction for Titanic passengers",
            "features": ["survival probability", "age", "gender", "passenger class", "fare", 
                        "siblings/spouse count", "parents/children count", "embarkation port"]
        },
        "heart": {
            "description": "heart disease diagnosis",
            "features": ["disease probability", "age", "gender", "chest pain type", "blood pressure", 
                        "cholesterol level", "blood sugar", "ECG results", "maximum heart rate"]
        }
    }
    
    # Get dataset-specific info or use defaults
    info = dataset_info.get(dataset, {
        "description": "demographic data analysis", 
        "features": ["probability", "age", "gender", "health indicators", "risk factors"]
    })
    
    # Create new version of the label_cluster method with modified prompt
    async def dataset_aware_label_cluster(self, cluster_centroid, **kwargs):
        # Create prompt with dataset context
        prompt = f"""You are an AI expert analyzing neural network activations from a {info['description']} model. 
        
        The model processes data with features like: {', '.join(info['features'])}.
        
        Based on the activation patterns in this cluster centroid, provide a concise, meaningful label 
        that describes what real-world pattern or concept this cluster might represent in the context of {dataset} data.
        
        Your label should relate to concepts like {', '.join(info['features'][:3])} rather than generic visual features.
        
        Cluster label:"""
        
        # Generate with dataset-aware prompt
        response = await self._generate_with_cache(
            prompt=prompt,
            temperature=0.2,
            max_tokens=20,
            system_prompt=f"You are a specialist in {info['description']} analysis."
        )
        
        # Clean up the response
        label = response.text.strip().strip('"\'')
        if len(label) > 50:
            label = label[:47] + "..."
        return label
    
    # Temporarily replace the method
    import types
    analyzer.label_cluster = types.MethodType(dataset_aware_label_cluster, analyzer)
    
    # Generate labels with the modified method
    cluster_labels = analyzer.label_clusters_sync(centroids)
    
    # Restore original method
    analyzer.label_cluster = original_label_method
    
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
    for path_id, narrative in list(path_narratives.items())[:2]:
        # Use ASCII arrow instead of Unicode to avoid encoding issues
        path_str = "->".join(paths[path_id])
        print(f"\nPath {path_id} ({path_str}):")
        print(narrative)
    
    # Perform bias audit if demographic info is available
    bias_analysis = None
    if demographic_info:
        print("\nPerforming bias audit...")
        
        # Determine relevant demographic columns
        demo_columns = []
        for _, demos in demographic_info.items():
            for col in demos:
                if col not in demo_columns:
                    demo_columns.append(col)
        
        if demo_columns:
            print(f"Analyzing demographic factors: {', '.join(demo_columns)}")
            try:
                bias_report = generate_bias_report(
                    paths=paths,
                    demographic_info=demographic_info,
                    demographic_columns=demo_columns,
                    cluster_labels=cluster_labels
                )
                
                # Generate LLM analysis of bias
                print("Generating bias analysis with LLM...")
                bias_analysis = analyze_bias_with_llm(analyzer, bias_report)
                
                print("\nBias Analysis Summary:")
                print("\n".join(bias_analysis.split("\n")[:3]) + "...")
            except Exception as e:
                print(f"Error during bias analysis: {e}")
                bias_analysis = f"Error during bias analysis: {e}"
    
    # Compile results
    results = {
        "dataset": dataset,
        "seed": seed,
        "llm": {
            "provider": provider,
            "model": analyzer.model
        },
        "cluster_labels": cluster_labels,
        "path_narratives": path_narratives,
        "bias_analysis": bias_analysis
    }
    
    # Save results to file
    output_file = f"analysis_results_{dataset}_seed{seed}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM analysis on cluster paths")
    parser.add_argument("--dataset", type=str, default="titanic", help="Dataset name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--provider", type=str, default="grok", 
                        choices=["grok", "claude", "openai", "gemini"],
                        help="LLM provider")
    parser.add_argument("--model", type=str, default="default", help="Model name")
    
    args = parser.parse_args()
    
    if args.dataset == "both":
        print("\n=== Analyzing Titanic Dataset ===")
        run_llm_analysis(
            dataset="titanic",
            seed=args.seed,
            provider=args.provider,
            model=args.model
        )
        
        print("\n=== Analyzing Heart Disease Dataset ===")
        run_llm_analysis(
            dataset="heart",
            seed=args.seed,
            provider=args.provider,
            model=args.model
        )
    else:
        run_llm_analysis(
            dataset=args.dataset,
            seed=args.seed,
            provider=args.provider,
            model=args.model
        )