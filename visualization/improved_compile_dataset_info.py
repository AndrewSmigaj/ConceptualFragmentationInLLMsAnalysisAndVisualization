"""
Improved script to compile information about dataset files for quantitative results.
Uses only standard library modules with improved file classification.
"""

import os
import json
from typing import Dict, List, Any, Optional


def find_dataset_files(repo_root: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Find all files related to each dataset with improved classification.
    
    Args:
        repo_root: Root directory of the repository
        
    Returns:
        Dictionary mapping dataset names to dictionaries of file types to file paths
    """
    datasets = ["titanic", "heart", "adult"]
    file_types = [
        ("paths_with_centroids", ["paths_with_centroids"]),
        ("cluster_stats", ["cluster_stats"]),
        ("paths", ["paths.json"]),  # Look for exact path pattern to avoid misclassification
        ("embedding", [".pkl"]),
        ("llm", ["llm"])
    ]
    
    # Special handling for detecting paths files - backward compatibility
    def find_paths_files(search_path, dataset):
        """Helper to find paths files in either naming format"""
        paths = []
        import glob
        
        # Look for both file naming conventions
        paths_with = glob.glob(os.path.join(search_path, f"**/*{dataset}_seed_*_paths_with_centroids.json"), recursive=True)
        paths_plain = glob.glob(os.path.join(search_path, f"**/*{dataset}_seed_*_paths.json"), recursive=True)
        
        # Prefer paths_with_centroids if present, otherwise use regular paths
        return paths_with or paths_plain
    
    dataset_files = {dataset: {file_type[0]: [] for file_type in file_types} for dataset in datasets}
    
    # Search for files in data and results directories
    search_dirs = ["data", "results", "visualization/data", "visualization/cache"]
    
    for search_dir in search_dirs:
        search_path = os.path.join(repo_root, search_dir)
        if not os.path.exists(search_path):
            continue
        
        # For each dataset, find paths files using special backward-compatible handling
        for dataset in datasets:
            # Find paths files in either format
            paths_files = find_paths_files(search_path, dataset)
            for file_path in paths_files:
                # Determine if it's a paths_with_centroids or paths file
                if "paths_with_centroids" in file_path.lower():
                    dataset_files[dataset]["paths_with_centroids"].append(file_path)
                else:
                    dataset_files[dataset]["paths"].append(file_path)
            
        # Recursively search for other file types
        for root, _, files in os.walk(search_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip paths files as they're handled separately above
                if "_paths.json" in file_path or "_paths_with_centroids.json" in file_path:
                    continue
                
                # Check if file belongs to any dataset
                for dataset in datasets:
                    if dataset in file_path.lower():
                        # Determine file type using specific markers
                        classified = False
                        for file_type, markers in file_types:
                            # Skip paths and paths_with_centroids as they're handled separately
                            if file_type in ["paths", "paths_with_centroids"]:
                                continue
                                
                            if any(marker in file_path.lower() for marker in markers):
                                dataset_files[dataset][file_type].append(file_path)
                                classified = True
                                break
                        
                        # If no specific type found, categorize as "other"
                        if not classified:
                            if "other" not in dataset_files[dataset]:
                                dataset_files[dataset]["other"] = []
                            dataset_files[dataset]["other"].append(file_path)
    
    return dataset_files


def get_available_seeds(dataset_files: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[int]]:
    """
    Determine available seeds for each dataset.
    
    Args:
        dataset_files: Dictionary mapping dataset names to file types and paths
        
    Returns:
        Dictionary mapping dataset names to lists of available seeds
    """
    available_seeds = {}
    
    for dataset, file_types in dataset_files.items():
        seeds = set()
        
        for file_type, files in file_types.items():
            for file_path in files:
                # Extract seed from filename
                if "seed" in file_path.lower():
                    # Find the seed number after "seed_" or "seed"
                    parts = os.path.basename(file_path).split("seed")
                    if len(parts) > 1:
                        seed_part = parts[1].lstrip("_")
                        # Extract the number
                        seed = ""
                        for char in seed_part:
                            if char.isdigit():
                                seed += char
                            else:
                                break
                                
                        if seed:
                            seeds.add(int(seed))
        
        available_seeds[dataset] = sorted(list(seeds))
    
    return available_seeds


def validate_dataset_files(dataset_files: Dict[str, Dict[str, List[str]]],
                         available_seeds: Dict[str, List[int]]) -> Dict[str, List[str]]:
    """
    Validate that essential files exist for each dataset and seed.
    
    Args:
        dataset_files: Dictionary mapping dataset names to file types and paths
        available_seeds: Dictionary mapping dataset names to lists of available seeds
        
    Returns:
        Dictionary mapping dataset names to lists of validation warnings
    """
    validation_warnings = {dataset: [] for dataset in dataset_files}
    
    essential_file_types = ["paths", "paths_with_centroids", "cluster_stats"]
    
    for dataset, seeds in available_seeds.items():
        if not seeds:
            validation_warnings[dataset].append(f"No seeds found for {dataset} dataset")
            continue
            
        for seed in seeds:
            for file_type in essential_file_types:
                found = False
                for file_path in dataset_files[dataset].get(file_type, []):
                    if f"seed_{seed}" in file_path or f"seed{seed}" in file_path:
                        found = True
                        break
                
                if not found:
                    validation_warnings[dataset].append(
                        f"Missing {file_type} file for {dataset} dataset, seed {seed}")
    
    return validation_warnings


def count_available_metrics(dataset_files: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict[str, bool]]:
    """
    Count which metrics can be computed for each dataset based on available files.
    
    Args:
        dataset_files: Dictionary mapping dataset names to file types and paths
        
    Returns:
        Dictionary mapping dataset names to available metrics
    """
    metrics = {}
    
    for dataset in dataset_files:
        metrics[dataset] = {
            "silhouette_scores": False,
            "ari_values": False,
            "path_reproducibility": False,
            "fragmentation_scores": False,
            "similarity_convergent_paths": False,
            "ets_vs_kmeans": False
        }
        
        # Check for metrics that require cluster_stats
        if dataset_files[dataset].get("cluster_stats", []):
            metrics[dataset]["silhouette_scores"] = True
        
        # Check for metrics that require paths
        if dataset_files[dataset].get("paths", []):
            metrics[dataset]["path_reproducibility"] = True
            metrics[dataset]["fragmentation_scores"] = True
            
        # Check for metrics that require paths_with_centroids
        if dataset_files[dataset].get("paths_with_centroids", []):
            metrics[dataset]["similarity_convergent_paths"] = True
            
        # Check for metrics that require multiple seeds
        if len(get_available_seeds({dataset: dataset_files[dataset]})[dataset]) > 1:
            metrics[dataset]["ari_values"] = True
            
        # Check for ETS vs k-means comparison
        if any("ets" in file.lower() for file_type in dataset_files[dataset].values() 
             for file in file_type):
            metrics[dataset]["ets_vs_kmeans"] = True
            
    return metrics


def summarize_dataset_info(dataset_files: Dict[str, Dict[str, List[str]]],
                         available_seeds: Dict[str, List[int]],
                         validation_warnings: Dict[str, List[str]],
                         available_metrics: Dict[str, Dict[str, bool]]) -> Dict[str, Any]:
    """
    Create a summary of dataset information.
    
    Args:
        dataset_files: Dictionary mapping dataset names to file types and paths
        available_seeds: Dictionary mapping dataset names to lists of available seeds
        validation_warnings: Dictionary mapping dataset names to lists of validation warnings
        available_metrics: Dictionary mapping dataset names to available metrics
        
    Returns:
        Dictionary with dataset summary information
    """
    summary = {}
    
    for dataset in dataset_files:
        summary[dataset] = {
            "seeds": available_seeds[dataset],
            "file_counts": {
                file_type: len(files) 
                for file_type, files in dataset_files[dataset].items()
            },
            "warnings": validation_warnings[dataset],
            "available_metrics": available_metrics[dataset]
        }
        
        # Check if we have enough data for key metrics
        can_compute_key_metrics = (
            available_metrics[dataset]["silhouette_scores"] or
            available_metrics[dataset]["path_reproducibility"] or
            available_metrics[dataset]["similarity_convergent_paths"]
        )
        
        summary[dataset]["can_compute_key_metrics"] = can_compute_key_metrics
    
    return summary


def save_dataset_info(dataset_files: Dict[str, Dict[str, List[str]]],
                     summary: Dict[str, Any],
                     output_path: str):
    """
    Save dataset information to a JSON file.
    
    Args:
        dataset_files: Dictionary mapping dataset names to file types and paths
        summary: Dictionary with dataset summary information
        output_path: Path to save output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "dataset_files": dataset_files,
        "summary": summary
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Dataset information saved to {output_path}")


def print_dataset_summary(summary: Dict[str, Any]):
    """
    Print a summary of dataset information to the console.
    
    Args:
        summary: Dictionary with dataset summary information
    """
    print("\n=== Dataset Summary ===")
    
    for dataset, info in summary.items():
        print(f"\n{dataset.upper()} Dataset:")
        print(f"  Seeds: {info['seeds']}")
        print(f"  Files:")
        for file_type, count in info["file_counts"].items():
            print(f"    {file_type}: {count}")
        
        print(f"  Available Metrics:")
        for metric, available in info["available_metrics"].items():
            print(f"    {metric.replace('_', ' ').title()}: {'Yes' if available else 'No'}")
        
        if info["warnings"]:
            print(f"  Warnings:")
            for warning in info["warnings"]:
                print(f"    - {warning}")
        
        print(f"  Can Compute Key Metrics: {'Yes' if info['can_compute_key_metrics'] else 'No'}")


def main():
    # Detect project root dynamically (directory one level above visualization)
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    
    # Find dataset files
    dataset_files = find_dataset_files(repo_root)
    
    # Get available seeds
    available_seeds = get_available_seeds(dataset_files)
    
    # Validate dataset files
    validation_warnings = validate_dataset_files(dataset_files, available_seeds)
    
    # Count available metrics
    available_metrics = count_available_metrics(dataset_files)
    
    # Summarize dataset information
    summary = summarize_dataset_info(
        dataset_files, available_seeds, validation_warnings, available_metrics)
    
    # Print summary
    print_dataset_summary(summary)
    
    # Save dataset information
    output_path = os.path.join(repo_root, "data", "dataset_info.json")
    save_dataset_info(dataset_files, summary, output_path)


if __name__ == "__main__":
    main()