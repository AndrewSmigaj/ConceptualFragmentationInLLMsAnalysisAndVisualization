"""
Simple test script to verify the cluster path export functionality 
by running it on Titanic dataset with seed 0.
"""
import os
import sys
from concept_fragmentation.data.loaders import get_dataset_loader
from concept_fragmentation.analysis.cluster_paths import (
    load_clusters_from_cache, 
    compute_clusters_for_layer, 
    load_experiment_activations, 
    write_cluster_paths
)

def main():
    # Setup
    dataset_name = "titanic"
    seed = 0
    max_k = 10
    output_dir = "data/cluster_paths"
    
    print(f"Testing cluster path export for {dataset_name} (seed {seed})...")
    
    # Load dataset
    try:
        dataset_loader = get_dataset_loader(dataset_name)
        train_df, test_df = dataset_loader.load_data()
        df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)  # Use full dataset
        print(f"Successfully loaded dataset with {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
        
    # Set target/demographic columns
    target_column = "survived"
    demographic_columns = ["age", "sex", "pclass", "fare"]
    
    # Try to load clusters from cache first
    layer_clusters = load_clusters_from_cache(dataset_name, "baseline", seed, max_k)
    
    # If no cached clusters found, try to load activations and compute clusters
    if layer_clusters is None:
        try:
            # Load activations and compute clusters
            results_dir = os.path.join("results", "baselines", dataset_name, f"baseline_seed{seed}")
            activations = load_experiment_activations(results_dir)
            print(f"Loaded activations for {len(activations)} layers")
            
            # Compute clusters for each layer
            layer_clusters = {}
            for layer_name, layer_activations in activations.items():
                print(f"Computing clusters for layer {layer_name}...")
                k, centers, labels = compute_clusters_for_layer(layer_activations, max_k=max_k, random_state=seed)
                print(f"  Found {k} clusters")
                layer_clusters[layer_name] = {
                    "k": k,
                    "centers": centers,
                    "labels": labels
                }
        except Exception as e:
            print(f"Error loading activations or computing clusters: {e}")
            print("Make sure the experiment has been run or visualization cache exists.")
            return 1
    else:
        print(f"Using clusters from cache with layers: {list(layer_clusters.keys())}")
    
    # If we still don't have clusters, return error
    if not layer_clusters:
        print("No cluster information available - can't continue.")
        return 1
    
    # Write cluster paths
    try:
        output_path = write_cluster_paths(
            dataset_name,
            seed,
            layer_clusters,
            df,
            target_column=target_column,
            demographic_columns=demographic_columns,
            output_dir=output_dir,
            top_k=3,
            max_members=50
        )
        print(f"Successfully wrote cluster paths to {output_path}")
    except Exception as e:
        print(f"Error writing cluster paths: {e}")
        return 1
    
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    # Import pandas here to avoid import error if not in function
    import pandas as pd
    sys.exit(main()) 