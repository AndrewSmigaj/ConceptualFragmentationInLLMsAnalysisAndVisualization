"""
Script to compile information about dataset files for quantitative results.

This script scans the repository for data files related to the Titanic, Heart, and Adult
datasets, organizing them by dataset and content type for easier analysis in subsequent tasks.
"""

import os
import json
import glob
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


def find_dataset_files(repo_root: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Find all files related to each dataset.
    
    Args:
        repo_root: Root directory of the repository
        
    Returns:
        Dictionary mapping dataset names to dictionaries of file types to file paths
    """
    datasets = ["titanic", "heart", "adult"]
    file_types = ["paths", "paths_with_centroids", "cluster_stats", "embedding", "llm"]
    
    dataset_files = {dataset: {file_type: [] for file_type in file_types} for dataset in datasets}
    
    # Search for files in data and results directories
    search_dirs = ["data", "results", "visualization/data", "visualization/cache"]
    
    for search_dir in search_dirs:
        search_path = os.path.join(repo_root, search_dir)
        if not os.path.exists(search_path):
            continue
            
        # Recursively search for files
        for root, _, files in os.walk(search_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file belongs to any dataset
                for dataset in datasets:
                    if dataset in file_path.lower():
                        # Determine file type
                        for file_type in file_types:
                            if file_type in file_path.lower() or (
                                file_type == "embedding" and file_path.endswith(".pkl")):
                                dataset_files[dataset][file_type].append(file_path)
                                break
                        
                        # If no specific type found, categorize as "other"
                        if all(file_type not in file_path.lower() for file_type in file_types):
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


def summarize_dataset_info(dataset_files: Dict[str, Dict[str, List[str]]],
                         available_seeds: Dict[str, List[int]],
                         validation_warnings: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Create a summary of dataset information.
    
    Args:
        dataset_files: Dictionary mapping dataset names to file types and paths
        available_seeds: Dictionary mapping dataset names to lists of available seeds
        validation_warnings: Dictionary mapping dataset names to lists of validation warnings
        
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
            "warnings": validation_warnings[dataset]
        }
        
        # Check if we have enough data for full metrics
        if summary[dataset]["seeds"] and all(
            file_type in dataset_files[dataset] and dataset_files[dataset][file_type]
            for file_type in ["paths", "paths_with_centroids"]
        ):
            summary[dataset]["complete_data_available"] = True
        else:
            summary[dataset]["complete_data_available"] = False
    
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
        
        if info["warnings"]:
            print(f"  Warnings:")
            for warning in info["warnings"]:
                print(f"    - {warning}")
        
        print(f"  Complete Data Available: {info['complete_data_available']}")


def main():
    # Repository root
    repo_root = "/mnt/c/Repos/ConceptualFragmentationInLLMsAnalysisAndVisualization"
    
    # Find dataset files
    dataset_files = find_dataset_files(repo_root)
    
    # Get available seeds
    available_seeds = get_available_seeds(dataset_files)
    
    # Validate dataset files
    validation_warnings = validate_dataset_files(dataset_files, available_seeds)
    
    # Summarize dataset information
    summary = summarize_dataset_info(dataset_files, available_seeds, validation_warnings)
    
    # Print summary
    print_dataset_summary(summary)
    
    # Save dataset information
    output_path = os.path.join(repo_root, "data", "dataset_info.json")
    save_dataset_info(dataset_files, summary, output_path)


if __name__ == "__main__":
    main()